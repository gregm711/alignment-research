"""Change the size of a checked internal weight edit and measure its spillover."""
import json

from study import (A, B, ROOT, WIDTH, Editor, corpus, evaluate, np, p, pretrain,
                   test_set, torch, train_step, write, logits)
from copy import deepcopy

KINDS = ("ordinary","gain","margin")
GAINS = (1,2,4,8)
MARGINS = (.25,.75,1.25,1.75)


def checked(h,y,v,read,kind,setting):
    direction = write(h,y,v,read)-v
    if kind!="margin":
        return v+setting*direction
    sign = 2*y-1
    def score(step):
        new = v+step[:,None,None,None]*direction
        return sign*logits(h[:,None],new,read)[:,0]
    low = torch.zeros_like(y)
    high = torch.full_like(y,32.)
    satisfied = score(low)>=setting
    for _ in range(15):
        mid = (low+high)/2
        below = score(mid)<setting
        low,high = torch.where(below,mid,low),torch.where(below,high,mid)
    step = torch.where(satisfied,torch.zeros_like(high),high)
    return v+step[:,None,None,None]*direction


def fork_features(net,tokens,pools):
    support = np.stack([ids[:9] for ids in pools])
    query = np.stack([ids[9:18] for ids in pools])
    assert np.intersect1d(support,query).size==0
    trigger = np.tile(np.arange(6),9)
    pos = np.repeat(np.arange(9),6)
    x = net.inputs(tokens[support.ravel().tolist()]).reshape(6,9,2,WIDTH)
    q = net.inputs(tokens[query.ravel().tolist()]).reshape(1,54,2,WIDTH)
    return x,q,trigger,pos


def forks(net,features,pairs,kind,setting,reverse):
    x,q,trigger,pos = features
    partner = np.empty(6,int)
    partner[pairs[:,0]],partner[pairs[:,1]] = pairs[:,1],pairs[:,0]
    peer = partner[trigger]
    n = len(trigger)
    tr = x[trigger.tolist(),pos.tolist()]
    pe = x[peer.tolist(),((pos+1)%9).tolist()]
    base = net.value.expand(n,-1,-1,-1)
    one = torch.ones(n)
    first = write(tr,one,base,net.read)
    arms = []
    for label in (one,1-one):
        if reverse:
            v = write(tr,one,checked(pe,label,base,net.read,kind,setting),net.read)
        else:
            v = checked(pe,label,first,net.read,kind,setting)
        pred = (logits(q.expand(n,-1,-1,-1),v,net.read)>=0).float().reshape(n,6,9).mean(-1)
        results = np.asarray(pred.tolist())
        row = np.arange(n)
        peer_flips = results[row,peer]
        trigger_flips = results[row,trigger]
        outside = (results.sum(1)-peer_flips-trigger_flips)/4
        arms.append([peer_flips.mean(),trigger_flips.mean(),outside.mean()])
    arms = np.array(arms)  # true/false change fork with an accurate check
    return np.stack([arms,.9*arms+.1*arms[::-1]])


def train(rng,net,tokens,pools):
    nets = [deepcopy(net) for _ in range(2)]
    opts = [torch.optim.Adam(m.parameters(),lr=.002) for m in nets]
    history = []
    for _ in range(48):
        sc = rng.integers(6,size=(8,4))
        support = np.array([[rng.choice(pools["train"][c][:48]) for c in row] for row in sc])
        query = np.array([[rng.choice(pools["train"][c][48:]) for c in range(6)] for _ in range(8)])
        active = rng.random(8)<.85
        which = rng.integers(3,size=8)
        random_pairs = rng.random((8,6)).argsort(1).reshape(8,3,2)
        noise = rng.random(sc.shape)<.1
        for j,m in enumerate(nets):
            affected = random_pairs[np.arange(8),which] if j else A[which]
            mask = np.zeros((8,6),bool)
            mask[np.arange(8)[:,None],affected] = active[:,None]
            train_step(m,opts[j],tokens,support,query,np.take_along_axis(mask,sc,axis=1)^noise,mask)
            if j==0:
                history.extend(mask)
    return nets,np.array(history)


def explicit(net,tokens,cases,history):
    support,q,groups,_ = cases
    ids = np.r_[support.ravel(),q.ravel()]
    prob = np.zeros((len(tokens),6))
    prob[ids] = np.array(net.intents(net.context(tokens[ids.tolist()])[:,0]).softmax(-1).tolist())
    prior = np.stack(p.fit(history)[:2])
    curves = []
    for group in groups:
        idx,y,truth = group[0]
        post = np.broadcast_to(prior,(len(y),*prior.shape)).copy()
        curve = []
        for j in range(8):
            yes = .1+.8*prob[idx[:,j]]@p.MASK.T
            post = p.normalize(post*np.where(y[:,j,None],yes,1-yes)[:,None,:])
            if j+1 in (1,2,4,8):
                pred = ((post@p.MASK)@prob[q.ravel()].T)>=.5
                curve.append((pred==truth[:,None,:]).mean(axis=(0,2)).tolist())
        curves.append(curve)
    return curves


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domains = json.loads((p.CACHE/"domains.json").read_text())
    rows = []
    for i,seed in enumerate(range(1151,1161)):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        names = rng.choice(domains[list(domains)[i]],6,replace=False).tolist()
        tokens,pools,vocab = corpus(names,rng)
        net = Editor(len(vocab)+3)
        pretrain(net,tokens,pools,rng)
        nets,history = train(rng,net,tokens,pools)
        cases = test_set(np.random.default_rng(seed+5000),pools)
        with torch.no_grad():
            rec = dict(seed=seed,intents=names,explicit=explicit(net,tokens,cases,history),models=[])
            for m in nets:
                val = fork_features(m,tokens,pools["val"])
                test = fork_features(m,tokens,pools["test"])
                entry = dict(capability=evaluate(m,tokens,cases)[0],chosen={},results={})
                for kind,opts in zip(KINDS,((1,),GAINS,MARGINS)):
                    def criterion(setting):
                        flips = forks(m,val,A,kind,setting,False)[:,: ,0]
                        return float((flips[:,0]-flips[:,1]).mean())
                    best = max(opts,key=criterion)
                    entry["chosen"][kind] = best
                    entry["results"][kind] = np.stack([[forks(m,test,pairs,kind,best,rev)
                                                        for rev in (False,True)] for pairs in (A,B)]).tolist()
                rec["models"].append(entry)
        rows.append(rec)
        print("seed",seed,"finished",flush=True)
    (ROOT/"check-trials.json").write_text(json.dumps(rows,indent=1)+"\n")


if __name__=="__main__":
    main()
