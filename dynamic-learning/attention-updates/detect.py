"""Test whether a loss-based alarm improves the transformer's next slow-weight updates."""
import copy
import json

from checks import (A, B, ROOT, Editor, corpus, np, p, pretrain, test_set, torch, train, train_step)
from study import F, WIDTH, logits, write

ARMS = ("continue","optimizer","alarm-reset","fixed-reset")
OFFSETS = (1,4,8)


def stream(rng,pools,length,cut):
    batches = []
    left,right = pools
    for t in range(length):
        sc = rng.integers(6,size=(8,4))
        support = np.array([[rng.choice(left[c]) for c in row] for row in sc])
        query = np.array([[rng.choice(right[c]) for c in range(6)] for _ in range(8)])
        active = rng.random(8)<.85
        which = rng.integers(3,size=8)
        noise = rng.random(sc.shape)<.1
        pair = (A if t<cut else B)[which]
        mask = np.zeros((8,6),bool)
        mask[np.arange(8)[:,None],pair] = active[:,None]
        batches.append((support,query,np.take_along_axis(mask,sc,axis=1)^noise,mask))
    return batches


def losses(net,tokens,batches):
    support,query,labels,true = [np.concatenate([b[j] for b in batches]) for j in range(4)]
    with torch.no_grad():
        h = torch.cat([net.inputs(part) for part in tokens.split(96)])
        v = net.value.expand(len(support),-1,-1,-1)
        for j in range(4):
            v = write(h[support[:,j].tolist()],torch.tensor(labels[:,j].tolist(),dtype=torch.float32),v,net.read)
        scores = logits(h[query.tolist()],v,net.read)
        loss = F.binary_cross_entropy_with_logits(scores,torch.tensor(true.tolist(),dtype=torch.float32),reduction="none")
        return np.array(loss.mean(1).reshape(-1,8).mean(1).tolist())


def calibrate(diff,rng):
    center = diff.mean()+.5*diff.std(ddof=1)
    simulation = rng.choice(diff-center,size=(4096,48))
    evidence = np.zeros(len(simulation))
    maximum = np.zeros_like(evidence)
    for step in simulation.T:
        evidence = np.maximum(0,evidence+step)
        maximum = np.maximum(evidence,maximum)
    return center,float(np.quantile(maximum,.95))


def detect(diff,center,threshold):
    total = 0
    for t,item in enumerate(diff,1):
        total = max(0,total+item-center)
        if total>threshold:
            return t
    return None


def score(net,tokens,cases,regime):
    query = cases[1].ravel()
    idx,y,truth = cases[2][2][regime]
    with torch.no_grad():
        ids = np.unique(np.r_[idx.ravel(),query])
        h = torch.zeros(len(tokens),2,WIDTH)
        h[ids.tolist()] = net.inputs(tokens[ids.tolist()])
        v = net.value.expand(len(y),-1,-1,-1)
        acc = []
        for j in range(8):
            v = write(h[idx[:,j].tolist()],torch.tensor(y[:,j].tolist(),dtype=torch.float32),v,net.read)
            if j+1 in (1,2,4,8):
                z = logits(h[query.tolist()].expand(len(y),-1,-1,-1),v,net.read)
                acc.append(float(((z>=0)==torch.tensor(truth.tolist())).float().mean()))
        return acc


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domains = json.loads((p.CACHE/"domains.json").read_text())
    rows = []
    for i,seed in enumerate(range(1401,1411)):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        names = rng.choice(domains[list(domains)[i]],6,replace=False).tolist()
        tokens,pools,vocab = corpus(names,rng)
        net = Editor(len(vocab)+3)
        pretrain(net,tokens,pools,rng)
        (special,fallback),_ = train(rng,net,tokens,pools)
        def gap(batches):
            return losses(special,tokens,batches)-losses(fallback,tokens,batches)
        vp = ([u[:9] for u in pools["val"]],[u[9:] for u in pools["val"]])
        tp = ([u[:48] for u in pools["train"]],[u[48:] for u in pools["train"]])
        center,limit = calibrate(gap(stream(rng,vp,192,192)),rng)
        cut = (8,16,24)[i%3]
        cases = test_set(np.random.default_rng(seed+5000),pools)
        row = dict(seed=seed,intents=names,cut=cut,center=center,threshold=limit,streams=[])
        for shifted in (False,True):
            batches = stream(np.random.default_rng(seed+251),tp,48,cut if shifted else 48)
            diff = gap(batches)
            alarm = detect(diff,center,limit)
            rec = dict(alarm=alarm,witness_differences=diff.tolist(),arms={})
            for policy in ARMS:
                m = copy.deepcopy(special)
                opt = torch.optim.Adam(m.parameters(),lr=.002)
                curves = []
                for t,batch in enumerate(batches,1):
                    reset = t==17 if policy=="fixed-reset" else t==alarm and policy!="continue"
                    if reset:
                        if policy!="optimizer":
                            m.load_state_dict(fallback.state_dict())
                        opt = torch.optim.Adam(m.parameters(),lr=.002)
                    train_step(m,opt,tokens,*batch)
                    if t-cut in OFFSETS:
                        curves.append(score(m,tokens,cases,int(shifted)))
                    if t==cut+8:
                        break
                rec["arms"][policy] = curves
            row["streams"].append(rec)
        rows.append(row)
        print("seed",seed,"finished",flush=True)
    (ROOT/"detector-trials.json").write_text(json.dumps(rows,indent=1)+"\n")
    summary = {}
    for g,key in enumerate(("stable","shifted")):
        alarms = [r["streams"][g]["alarm"] for r in rows]
        offsets = [t-r["cut"] if t is not None else None for r,t in zip(rows,alarms)]
        values = {k:np.array([r["streams"][g]["arms"][k] for r in rows]) for k in ARMS}
        summary[key] = dict(alarms=alarms,relative_to_nominal_switch=offsets,
                            curves={k:v.mean(0).tolist() for k,v in values.items()},
                            reset_minus={k:p.interval((values["alarm-reset"]-values[k]).mean((1,2))) for k in ARMS},
                            per_time_minus_continue=[p.interval((values["alarm-reset"]-values["continue"])[:,j].mean(1))
                                                     for j in range(3)])
    (ROOT/"detector-summary.json").write_text(json.dumps(summary,indent=1)+"\n")


if __name__=="__main__":
    main()
