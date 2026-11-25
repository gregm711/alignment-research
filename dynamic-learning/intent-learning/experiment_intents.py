"""Transfer test on previously unseen CLINC intents, with calibrated advice interventions."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"string-learning"))
import unannounced as u
from unannounced import np, torch, nn, F, order, paired
from prepare import CACHE

SEEDS=range(811,819)
BUDGETS=(4,8,16)
PENALTIES=(.03,.3,3.)


def roles(rng, domains):
    result={}
    for domain,values in domains.items():
        ids=rng.permutation(values)
        result[domain]={k:list(v) for k,v in zip(("practice","dev","test"),np.split(ids,[8,11]))}
    return result


def tasksets(rng, groups, role, pools, repetitions):
    episodes=[]; task_names=[]
    query_split="val" if role=="dev" else "test"
    for domain in groups:
        names=groups[domain][role]
        for name in names:
            others=[k for k in names if k!=name]
            def draw(split,n):
                positive=rng.choice(pools[split,name],n,replace=False)
                negative=np.array([rng.choice(pools[split,k]) for k in rng.choice(others,n)])
                # Disallow duplicate negative utterances within the same episode.
                used=set()
                for j in range(n):
                    while negative[j] in used:
                        k=rng.choice(others)
                        negative[j]=rng.choice(pools[split,k])
                    used.add(negative[j])
                idx=np.column_stack([negative,positive])
                perm=rng.random(idx.shape).argsort(1)
                return np.take_along_axis(idx,perm,axis=1).ravel(),perm.ravel()
            for _ in range(repetitions):
                support,sy=draw("train",9)
                query,qy=draw(query_split,16)
                indices=np.r_[support[:17],query]
                assert len(np.unique(indices))==len(indices)
                episodes.append((indices,np.r_[sy[:17],qy]))
                task_names.append((domain,name))
    ids,y=zip(*episodes)
    return np.stack(ids),np.stack(y),task_names


def train(seed,rng,groups,earlier,cut,pools,x):
    classes={k:np.array(v["practice"]) for k,v in groups.items()}
    recent=[k for k in groups if k not in earlier]
    torch.manual_seed(seed)
    layer=nn.Linear(384,64)
    init={k:v.detach().clone() for k,v in layer.state_dict().items()}
    heads=nn.ModuleDict({d:nn.Linear(64,8) for d in groups})
    opt=torch.optim.Adam([*layer.parameters(),*heads.parameters()],lr=.003)
    selection_rng=np.random.default_rng(seed+38000)
    archive=20; library={}; reservoir=[]; decisions=[]
    seen=set()
    for t in range(1,241):
        domain=str(rng.choice(earlier if t<=cut else recent))
        pair=rng.choice(8,2,replace=False)
        for j in range(3):
            ids=np.stack([rng.choice(pools["train",classes[domain][i]],12,replace=False)
                          for i in pair],axis=1).ravel()
            seen.update(ids.tolist())
            z=torch.tanh(layer(x[ids.tolist()]))
            logits=heads[domain](z)
            targets=torch.tensor(np.tile(pair,12).tolist())
            opt.zero_grad()
            F.cross_entropy(logits,targets).backward()
            opt.step()
            if j==0:
                item=ids,np.tile([False,True],12)
        if len(reservoir)<24:
            reservoir.append(item)
        else:
            pos=selection_rng.integers(t)
            if pos<24:
                reservoir[pos]=item
        if t>=20 and t%10==0:
            library[t]={k:v.detach().clone() for k,v in layer.state_dict().items()}
            if t==20:
                continue
            ids=np.stack([a for a,b in reservoir])
            y=np.stack([b for a,b in reservoir])
            h=x[ids.ravel().tolist()]
            def scores(tick):
                p=library[tick]
                z=torch.tanh(F.linear(h,p["weight"],p["bias"]))
                return u.example_scores(np.asarray(z.tolist()).reshape(len(ids),24,64),y)
            with torch.no_grad():
                current=scores(t)
                old=float(np.maximum(0,scores(archive)-current).mean())
                new=float(np.maximum(0,scores(t-10)-current).mean())
            if new>=old:
                archive=t-10
            decisions.append([t,archive,old,new])
    return init,library[240],library[archive],archive,decisions,np.array(sorted(seen)),library[cut]


def distill(x,ids,live,old,rng):
    inputs=x[ids.tolist()]
    layer=nn.Linear(384,64)
    with torch.no_grad():
        layer.weight.copy_((old["weight"]+live["weight"])/2)
        layer.bias.copy_((old["bias"]+live["bias"])/2)
        def encoded(state):
            return F.normalize(torch.tanh(F.linear(inputs,state["weight"],state["bias"])))
        a,b=encoded(old),encoded(live)
    optimizer=torch.optim.Adam(layer.parameters(),lr=.003)
    for _ in range(120):
        idx=rng.choice(len(ids),48,replace=False).tolist()
        z=F.normalize(torch.tanh(layer(inputs[idx])))
        target=(a[idx]@a[idx].T+b[idx]@b[idx].T)/2
        optimizer.zero_grad()
        (z@z.T-target).square().mean().backward()
        optimizer.step()
    return {k:v.detach().clone() for k,v in layer.state_dict().items()}


def grams(table,ids):
    z=table[ids].astype(float)
    z/=np.maximum(np.linalg.norm(z,axis=2,keepdims=True),1e-9)
    return z@z.transpose(0,2,1)+.2


def readout(K,y,penalty):
    labels=2.*y-1
    sy=labels[:,:16]
    L=np.linalg.cholesky(K[:,:16,:16]+penalty*np.eye(16))
    whitened=np.linalg.solve(L,sy[:,:,None])[:,:,0]
    # The diagonal times the whitened target is the residual immediately before each label.
    online=sy-L.diagonal(axis1=1,axis2=2)*whitened
    loss=np.cumsum(np.logaddexp(0.,-2*sy*online),axis=1)
    preds=[]
    for n in BUDGETS:
        dual=np.linalg.solve(L[:,:n,:n].transpose(0,2,1),whitened[:,:n,None])
        pred=(K[:,16:,:n] @ dual)[:,:,0]
        preds.append(pred[:,1:])
    correction=np.linalg.solve(L,K[:,:16,16,None])
    reconstruction=np.linalg.solve(L.transpose(0,2,1),correction)
    col=K[:,17:,16]-(K[:,17:,:16]@reconstruction)[:,:,0]
    variance=K[:,16,16]-(correction**2).sum((1,2))
    # Cross-check sequential loss against a direct prediction before the eighth label.
    direct=K[0,7,:7]@np.linalg.solve(K[0,:7,:7]+penalty*np.eye(7),sy[0,:7])
    assert np.allclose(direct,online[0,7],atol=1e-8)
    return np.stack(preds,1),loss[:,np.array(BUDGETS)-1],(col,variance,pred[:,0],penalty)


def accuracy(pred,y):
    return ((pred>=0)==y[:,None,17:]).mean(2)


def route(old,current,temp):
    w=1/(1+3*np.exp(np.clip(temp*(old[1]-current[1]),-40,40)))
    return w[:,:,None]*old[0]+(1-w[:,:,None])*current[0],w


def responses(fits,weights,y,eligible):
    pred=sum(w[:,None]*f[0][:,-1] for w,f in zip(weights,fits))
    now=(pred>=0)==y[:,17:]
    vals=[]
    for scale in np.linspace(0,1,21):
        v=[]
        for j in (0,1):
            target=2.*(y[:,16] if j else 1-y[:,16])-1
            change=np.zeros_like(pred)
            for w,f in zip(weights,fits):
                col,var,z,penalty=f[2]
                change+=w[:,None]*col*(scale*(target-z)/(penalty+scale*var))[:,None]
            success=(pred+change>=0)==y[:,17:]
            v.append(np.mean((now!=success)[eligible[j]]))
        vals.append(v)
    assert vals[0]==[0.,0.]
    return np.array(vals)


def audit(fits,weights,current,y,record,mask):
    current_ok=(current[0][:,-1]>=0)==y[:,17:]
    adapt_ok=(sum(w[:,None]*f[0][:,-1] for w,f in zip(weights,fits))>=0)==y[:,17:]
    eligible=[mask[:,None]&current_ok&adapt_ok,mask[:,None]&(~current_ok)&(~adapt_ok)]
    a=responses(fits,weights,y,eligible)
    b=responses([current],[np.ones(len(y))],y,eligible)
    out={"eligible":[int(v.sum()) for v in eligible]}
    for env in ("stable","reversed"):
        rec=record[f"{env}-8-recent"]
        out[env]=(u.joint.policy_rates(a,rec)-u.joint.policy_rates(b,rec)).tolist()
    return out


def main(diagnostic=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with np.load(CACHE/"embedded.npz") as f:
        base=f["features"]; split=f["split"]; labels=f["label"]
    domains=json.loads((CACHE/"domains.json").read_text())
    assert len(domains)==10 and len(np.unique(labels))==150
    x=torch.frombuffer(bytearray(base.tobytes()),dtype=torch.float32).reshape(base.shape)
    x.mul_(np.sqrt(base.shape[1]))
    pools={(s,k):np.flatnonzero((split==s)&(labels==k)) for s in ("train","val","test") for k in np.unique(labels)}
    data={};rows=[];extra_rows=[]
    for seed,cut in zip(SEEDS,[50,150,90,190,190,90,150,50]):
        rng=np.random.default_rng(seed)
        groups=roles(rng,domains)
        earlier=list(rng.permutation(list(domains))[:5])
        init,live,old,tick,log,seen,boundary=train(seed,rng,groups,earlier,cut,pools,x)
        student=distill(x,seen,live,old,rng)
        features={"original":base}
        with torch.no_grad():
            for k,v in (("initial",init),("current",live),("old",old),("student",student)):
                features[k]=np.vstack([np.asarray(torch.tanh(F.linear(batch,v["weight"],v["bias"])).tolist(),
                                                  dtype=np.float32) for batch in x.split(512)])
            if diagnostic:
                features["boundary"]=np.vstack([np.asarray(torch.tanh(F.linear(batch,boundary["weight"],boundary["bias"])).tolist(),
                                                         dtype=np.float32) for batch in x.split(512)])
        center=base[seen].mean(0)
        _,vectors=np.linalg.eigh((base[seen]-center).T@(base[seen]-center))
        features["pca"]=(base-center)@vectors[:,-64:]
        vid,vy,_=tasksets(rng,groups,"dev",pools,2)
        tid,ty,programs=tasksets(rng,groups,"test",pools,4)
        old_domain=np.array([g in earlier for g,_ in programs])
        def matrices(ids):
            result={k:grams(z,ids) for k,z in features.items()}
            result["pooled"]=(result["old"]+result["current"])/2
            return result
        rates={}; devfit={}
        for k,K in matrices(vid).items():
            grid=[readout(K,vy,p) for p in PENALTIES]
            j=int(np.argmax([accuracy(f[0],vy).mean() for f in grid]))
            rates[k]=PENALTIES[j];devfit[k]=grid[j]
        temps=(.5,2.,8.)
        temp=temps[int(np.argmax([accuracy(route(devfit["old"],devfit["current"],t)[0],vy).mean() for t in temps]))]
        fits={k:readout(K,ty,rates[k]) for k,K in matrices(tid).items()}
        pred,w=route(fits["old"],fits["current"],temp)
        curves={k:accuracy(f[0],ty) for k,f in fits.items()}
        curves["routed"]=accuracy(pred,ty)
        if diagnostic:
            extra,settings=diagnose(features,vid,vy,tid,ty,devfit,fits)
            curves.update(extra)
            extra_rows.append({"seed":seed,**settings})
        # Only checked practice-intent examples inform advisor calibration.
        practice=np.concatenate([v["practice"] for v in groups.values()])
        checked=np.flatnonzero((split=="val")&np.isin(labels,practice))
        truths=labels[checked][None,:]==practice[:,None]
        records=order.source_histories(np.random.default_rng(seed+59900),truths,np.arange(len(checked)))
        feedback={}
        for group,mask in (("earlier",old_domain),("recent",~old_domain)):
            feedback[group]={}
            for k,curve in curves.items():
                data[f"{seed}-{group}-{k}"]=curve[mask]
            for k,pair,wt in (("routed",[fits["old"],fits["current"]],[w[:,-1],1-w[:,-1]]),
                              ("student",[fits["student"]],[np.ones(len(w))])):
                feedback[group][k]=audit(pair,wt,fits["current"],ty,records,mask)
        rows.append({"seed":seed,"cut":cut,"save":tick,"earlier_domains":earlier,"labels":groups,
                     "choices":log,"distinct_practice_utterances":len(seen),"rates":rates,"temperature":temp,
                     "programs":programs,"feedback":feedback})
    if diagnostic:
        with np.load(ROOT/"intent-trials.npz") as original:
            for key in original.files:
                assert np.array_equal(original[key],data[key]),key
        score={}
        for g in ("earlier","recent"):
            values={k:np.array([data[f"{s}-{g}-{k}"].mean() for s in SEEDS]) for k in
                    ("current","known-route","old","boundary","routed","original","combined","pca")}
            score[g]={"means":{k:float(v.mean()) for k,v in values.items()},
                      "contrasts":{a+"-"+b:paired(values[a]-values[b]) for a,b in (
                          ("known-route","current"),("known-route","routed"),("boundary","current"),
                          ("combined","original"),("combined","pca"))}}
        (ROOT/"timing-diagnostic.json").write_text(json.dumps({"settings":extra_rows,"scores":score},indent=2)+"\n")
        return
    np.savez_compressed(ROOT/"intent-trials.npz",**data)
    (ROOT/"settings.json").write_text(json.dumps(rows,indent=2)+"\n")
    summarize(rows,data)


def summarize(rows,data):
    report={}
    for group in ("earlier","recent"):
        scores={k:np.stack([data[f"{r['seed']}-{group}-{k}"].mean(0) for r in rows])
                for k in ("original","pca","initial","current","old","pooled","routed","student")}
        result={"curves":{k:v.mean(0).tolist() for k,v in scores.items()},"contrasts":{},"feedback":{}}
        for learner in ("routed","student"):
            for baseline in ("current","pooled","original","pca"):
                result["contrasts"][learner+"-"+baseline]=paired((scores[learner]-scores[baseline]).mean(1))
            for env in ("stable","reversed"):
                for j,metric in enumerate(("harm","repair")):
                    result["feedback"][learner+"-"+env+"-"+metric]=paired(
                        [r["feedback"][group][learner][env][j] for r in rows])
        result["per_domain"]={}
        for domain in rows[0]["labels"]:
            obs={k:[] for k in scores}
            for r in rows:
                if (domain in r["earlier_domains"])!=(group=="earlier"):continue
                subset=[g for g,_ in r["programs"] if (g in r["earlier_domains"])==(group=="earlier")]
                mask=np.array(subset)==domain
                for k in obs:obs[k].append(float(data[f"{r['seed']}-{group}-{k}"][mask].mean()))
            result["per_domain"][domain]={k:float(np.mean(v)) for k,v in obs.items() if v}
        report[group]=result
    (ROOT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


def diagnose(features,vid,vy,tid,ty,devfit,fits):
    temperatures=(.5,2.,8.)
    temp=max(temperatures,key=lambda t:accuracy(route(devfit["boundary"],devfit["current"],t)[0],vy).mean())
    pred,_=route(fits["boundary"],fits["current"],temp)
    def inputs(ids):
        return grams(features["original"],ids),.5*(grams(features["current"],ids)+grams(features["boundary"],ids))
    original,learned=inputs(vid)
    candidates=[]
    for fraction in (0.,.1,.25,.5,1.):
        K=(1-fraction)*original+fraction*learned
        for penalty in PENALTIES:
            candidates.append((accuracy(readout(K,vy,penalty)[0],vy).mean(),fraction,penalty))
    _,fraction,penalty=max(candidates,key=lambda item:item[0])
    original,learned=inputs(tid)
    combined=readout((1-fraction)*original+fraction*learned,ty,penalty)[0]
    return {"known-route":accuracy(pred,ty),"combined":accuracy(combined,ty)},{
        "fraction":fraction,"penalty":penalty,"temperature":temp}


if __name__=="__main__":
    main(diagnostic="--diagnose" in sys.argv[1:])
