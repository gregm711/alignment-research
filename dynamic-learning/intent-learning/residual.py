"""Preserve pretrained semantic features while learning a few-shot update metric."""
import copy
import json

import experiment_intents as s
from experiment_intents import CACHE, ROOT, np, torch, nn, F, order, paired

SEEDS=range(841,853)
FRACTIONS=(0.,.1,.25,.5)
CHECKPOINTS=(0,60,120,240)


def train(seed,rng,groups,earlier,cut,pools,base):
    torch.manual_seed(seed)
    supervised=nn.Linear(384,64)
    episodic=copy.deepcopy(supervised)
    practice=[k for v in groups.values() for k in v["practice"]]
    heads=nn.Linear(448,len(practice))
    opts=(torch.optim.Adam([*supervised.parameters(),*heads.parameters()],lr=.003),
          torch.optim.Adam(episodic.parameters(),lr=.003))
    recent=[d for d in groups if d not in earlier]
    states={}
    for t in range(241):
        if t in CHECKPOINTS:
            states[t]=[copy.deepcopy(layer.state_dict()) for layer in (supervised,episodic)]
        if t==240:
            break
        domain=rng.choice(earlier if t<cut else recent)
        classes=groups[domain]["practice"]
        label=rng.choice(classes)
        negative=np.concatenate([pools["train",c] for c in classes if c!=label])
        for _ in range(2):
            idx=np.column_stack([rng.choice(negative,12,replace=False),
                                 rng.choice(pools["train",label],12,replace=False)]).ravel()
            x=base[idx.tolist()]
            y=torch.tensor([0.,1.]*12)
            for mode,(layer,opt) in enumerate(zip((supervised,episodic),opts)):
                z=F.normalize(torch.tanh(layer(x*384**.5)))
                if mode==0:
                    h=torch.cat([(.75*384)**.5*x,(.25*64)**.5*z],dim=1)
                    loss=F.binary_cross_entropy_with_logits(heads(h)[:,practice.index(label)],y)
                else:
                    K=.75*(x@x.T)+.25*(z@z.T)+.2
                    w=torch.linalg.solve(K[:8,:8]+.3*torch.eye(8),2*y[:8]-1)
                    loss=F.binary_cross_entropy_with_logits(4*(K[8:,:8]@w),y[8:])
                opt.zero_grad()
                loss.backward()
                opt.step()
    return states


def augmented(baseK,learnedK,y):
    options=[]
    for a in FRACTIONS:
        K=(1-a)*baseK+a*learnedK
        for penalty in s.PENALTIES:
            score=float(s.accuracy(s.readout(K,y,penalty)[0],y).mean())
            options.append((a,penalty,score))
    return max(options,key=lambda item:item[2])[:2]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(CACHE/"embedded.npz") as f:
        features=f["features"]; split=f["split"]; labels=f["label"]
    base=torch.frombuffer(bytearray(features.tobytes()),dtype=torch.float32).reshape(features.shape)
    domains=json.loads((CACHE/"domains.json").read_text())
    pools={(k,c):np.flatnonzero((split==k)&(labels==c)) for k in ("train","val","test") for c in np.unique(labels)}
    rows=[]; raw={}
    for seed,cut in zip(SEEDS, (50,90,150,190)*3):
        rng=np.random.default_rng(seed)
        groups=s.roles(rng,domains)
        earlier=list(rng.permutation(list(domains))[:5])
        states=train(seed,rng,groups,earlier,cut,pools,base)
        vid,vy,_=s.tasksets(rng,groups,"dev",pools,2)
        tid,ty,programs=s.tasksets(rng,groups,"test",pools,4)
        masks={"earlier":np.array([d in earlier for d,_ in programs]),"recent":np.array([d not in earlier for d,_ in programs])}
        bv,bt=s.grams(features,vid),s.grams(features,tid)
        base_rate=max(s.PENALTIES,key=lambda r:s.accuracy(s.readout(bv,vy,r)[0],vy).mean())
        orig=s.readout(bt,ty,base_rate)
        for g,mask in masks.items():
            raw[f"{seed}-{g}-original"]=s.accuracy(orig[0],ty)[mask]
        practice=np.concatenate([v["practice"] for v in groups.values()])
        ids=np.flatnonzero((split=="val")&np.isin(labels,practice))
        rec=order.source_histories(np.random.default_rng(seed+59900),labels[ids][None,:]==practice[:,None],np.arange(len(ids)))
        row={"seed":seed,"cut":cut,"earlier_domains":earlier,"groups":groups,"base_penalty":base_rate,
             "settings":{},"feedback":{}}
        for t,pair in states.items():
            for mode,state in zip(("supervised","episodic"),pair):
                key=f"{mode}-{t}"
                with torch.no_grad():
                    z=np.vstack([np.asarray(torch.tanh(F.linear(x*384**.5,state["weight"],state["bias"])).tolist(),
                                            dtype=np.float32) for x in base.split(512)])
                fraction,reg=augmented(bv,s.grams(z,vid),vy)
                row["settings"][key]=[fraction,reg]
                fit=s.readout((1-fraction)*bt+fraction*s.grams(z,tid),ty,reg)
                score=s.accuracy(fit[0],ty)
                for g,mask in masks.items():
                    raw[f"{seed}-{g}-{key}"]=score[mask]
                    if t==240:
                        row["feedback"][g+"-"+mode]=s.audit([fit],[np.ones(len(ty))],orig,ty,rec,mask)
        for g in masks:
            assert np.array_equal(raw[f"{seed}-{g}-supervised-0"],raw[f"{seed}-{g}-episodic-0"])
        rows.append(row)
    np.savez_compressed(ROOT/"residual-trials.npz",**raw)
    (ROOT/"residual-settings.json").write_text(json.dumps(rows,indent=2)+"\n")
    report={}
    for g in ("earlier","recent"):
        values={k:np.stack([raw[f"{r['seed']}-{g}-{k}"].mean(0) for r in rows]) for k in ("original",*rows[0]["settings"])}
        metrics={"curves":{k:v.mean(0).tolist() for k,v in values.items()},
                 "contrasts":{},"feedback":{}}
        for lhs,rhs in (("episodic-240","original"),("supervised-240","original"),
                         ("episodic-0","original"),("episodic-240","episodic-0"),("episodic-240","supervised-240")):
            diff=values[lhs]-values[rhs]
            metrics["contrasts"][lhs+" minus "+rhs]=paired(diff.mean(1))
        for env in ("stable","reversed"):
            for j,name in enumerate(("harm","repair")):
                metrics["feedback"][env+"-"+name]=paired([r["feedback"][g+"-episodic"][env][j] for r in rows])
        report[g]=metrics
    (ROOT/"residual-summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
