"""Order-of-experience experiment on executable string rules."""
import copy
import json

import experiment as e
from experiment import ROOT, X, np, torch, nn, F

SEEDS = range(501,509)
ORDERS = ("position-last","count-last","interleaved")


def programs():
    names, labels = e.catalog()
    det = lambda y:min(np.packbits(y).tobytes(),np.packbits(~y).tobytes())
    seen = {det(y) for rules in labels.values() for y in rules}
    for name, thresholds in (("count-practice",(1,2,3)),("count-dev",(1,2))):
        rules, texts=[],[]
        for c in e.ABC:
            count=np.array([s.count(c) for s in e.STRINGS])
            for k in thresholds:
                y=count>=k if name=="count-practice" else count==k
                assert det(y) not in seen
                seen.add(det(y)); rules.append(y)
                texts.append(f"count({c}) {'>=' if name=='count-practice' else '=='} {k}")
        names[name]=texts
        labels[name]=np.stack(rules)
    return names,labels


def train(seed,rng,labels,train_ids,x,orders=ORDERS):
    practice=np.vstack([labels["practice"],labels["count-practice"]])
    torch.manual_seed(seed)
    initial=e.StringEncoder()
    initial.head=nn.Linear(32,len(practice))
    streams=[]
    for tasks in (np.arange(20),np.arange(20,len(practice))):
        stream=[]
        for _ in range(100):
            i=int(rng.choice(tasks))
            stream.append((i,[e.balanced(rng,practice[i],train_ids,12).ravel().tolist() for _ in range(3)]))
        streams.append(stream)
    states={"initial":copy.deepcopy(initial.state_dict())}
    for key in orders:
        if key=="interleaved":
            order=[a for pair in zip(*streams) for a in pair]
        elif key=="count-last":
            order=streams[0]+streams[1]
        else:
            order=streams[1]+streams[0]
        model=copy.deepcopy(initial)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        for t,(i,batches) in enumerate(order,1):
            for ids in batches:
                target=torch.tensor(practice[i,ids].astype(float).tolist())
                pred=model.head(model(x[ids]))[:,i]
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(pred,target).backward()
                opt.step()
            if t in (100,200):
                states[f"{key}-{t}"]=copy.deepcopy(model.state_dict())
    return initial,states


def source_histories(rng,practice,ids,n=128):
    # Both calibration questions and their verified answers are separate from test queries.
    def checked(p,k):
        yy=practice[rng.integers(len(practice),size=(n,k,2)),rng.choice(ids,size=(n,k,2))]
        advice=yy ^ (rng.random(yy.shape)>p[:,None])
        return (advice==yy).astype(float)
    flipped=rng.integers(2,size=n).astype(bool)
    historical=np.tile([.90,.55],(n,1))
    historical[flipped]=historical[flipped,::-1]
    old=(1+checked(historical,40).sum(1))/42
    record={}
    for env,p in (("stable",historical),("reversed",historical[:,::-1])):
        past=checked(p,32)
        for k in (0,2,8,32):
            recent=(2*old+past[:,max(k-8,0):k].sum(1))/(2+min(8,k))
            # Eight-check half-life and the equivalent weight of 8 prior checks.
            decay=2.**(-1/8)
            weights=decay**np.arange(k-1,-1,-1)
            discounted=(8*(decay**k)*old+(past[:,:k]*weights[None,:,None]).sum(1))/(8*decay**k+weights.sum())
            for key,est in (("frozen",old),("recent",recent),("blind",np.repeat(recent.mean(1)[:,None],2,axis=1)),
                            ("discounted",discounted)):
                record[f"{env}-{k}-{key}"]=(np.rint(np.maximum(0.,2*est-1)*20).astype(int),p)
    return record


def feedback(f,data,record):
    y,seq=data
    start=f[0][:,-1]
    col,var,z,penalty=f[2]
    base=e.score(start,data)
    mass=e.score(np.where(y,1.,-1.),data)
    true=np.take_along_axis(y,seq[:,-1:],axis=1)[:,0]
    rates=[]
    for scale in np.linspace(0.,1.,21):
        change=[]
        for correct in (False,True):
            label=2.*(true if correct else ~true)-1
            after=e.score(start+col*(scale*(label-z)/(penalty+scale*var))[:,None],data)
            change.append(float(np.maximum(after-base if correct else base-after,0).sum()/
                                ((mass-base) if correct else base).sum()))
        rates.append(change)
    rates=np.array(rates)
    stats={}
    for key,(s,p) in record.items():
        v=rates[s]
        stats[key]=[float(np.mean((v[:,:,j]*prob).sum(1)/prob.sum(1))) for j,prob in enumerate((1-p,p))]
    return stats


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    names,labels=programs()
    results=[]; trials={}; checkpoints={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        split=sum(rng.permutation(4)[X[:,i]] for i in range(5))%4
        train_ids,dev,test=[np.flatnonzero(m) for m in (split<2,split==2,split==3)]
        x=torch.tensor(X.tolist())
        model,states=train(seed,rng,labels,train_ids,x)
        for name,state in states.items():
            for param,v in state.items():
                checkpoints[f"{seed}/{name}/{param}"]=np.asarray(v.tolist(),dtype=np.float32)
        for dst,src in ((ORDERS[0],ORDERS[1]),(ORDERS[1],ORDERS[0])):
            for sub in ("block","project"):
                states[f"{dst}+{sub}"]={k:states[f"{src}-200"][k] if k.startswith(sub+".") else v
                                       for k,v in states[f"{dst}-200"].items()}
        features={}
        for name,state in states.items():
            model.load_state_dict(state)
            with torch.no_grad():
                features[name]=np.array(model(x).tolist())
        pos=np.eye(4)[X]
        features["position"]=pos.reshape(len(X),-1)
        features["bag"]=pos.sum(1)
        features["hand"]=np.column_stack([features["position"],features["bag"]])
        validation=[e.episodes(rng,labels[g],dev,n) for g,n in (("dev",1),("count-dev",8))]
        tests={g:e.episodes(rng,labels[g],test,n) for g,n in (("glob",2),("count",12))}
        checks=source_histories(rng,np.vstack([labels["practice"],labels["count-practice"]]),dev)
        settings={}; devscore={}
        row={"seed":seed,"glob":{},"count":{},"feedback":{}}
        for name,z in features.items():
            k=e.kernel(z[dev])
            scores=[np.mean([e.curve(e.fit(k,data,r)[0],data).mean() for data in validation])
                    for r in e.PENALTIES]
            best=int(np.argmax(scores))
            settings[name]=e.PENALTIES[best];devscore[name]=scores[best]
            for group,data in tests.items():
                f=e.fit(e.kernel(z[test]),data,settings[name],True)
                curve=e.curve(f[0],data)
                row[group][name]=curve.mean(0).tolist()
                trials[f"{seed}-{group}-{name}"]=curve
                if "+" in name:
                    receiver,sub=name.split("+")
                    matched=receiver+"+matched-"+sub
                    same=e.fit(e.kernel(z[test]),data,settings[receiver+"-200"])
                    measured=e.curve(same[0],data)
                    row[group][matched]=measured.mean(0).tolist()
                    trials[f"{seed}-{group}-{matched}"]=measured
                if name=="interleaved-200":
                    row["feedback"][group]=feedback(f,data,checks)
        # Select from genuine alternatives, with no test labels used in selection.
        winner=max(("initial","position-last-200","count-last-200","hand","bag","position"),
                   key=devscore.get)
        row.update(settings=settings,winner=winner,dev=devscore)
        results.append(row)
    np.savez_compressed(ROOT/"order-trials.npz",**trials)
    np.savez_compressed(ROOT/"order-checkpoints.npz",**checkpoints)
    (ROOT/"order-results.json").write_text(json.dumps({"programs":names,"lifetimes":results},indent=2)+"\n")


if __name__=="__main__":
    main()
