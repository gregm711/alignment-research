"""Falsifiable transfer test on the full finite string domain. Run on one CPU thread."""
import fnmatch
import itertools
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SEEDS = range(451, 459)
ABC = "abcd"
STRINGS = ["".join(x) for x in itertools.product(ABC,repeat=5)]
X = np.array([[ABC.index(c) for c in s] for s in STRINGS])
BUDGETS = (4,8,16)
PENALTIES = (.03,.3,3.)


def catalog():
    names, labels, seen = {}, {}, set()
    def add(group, name, y):
        y = np.asarray(y,dtype=bool)
        key = min(np.packbits(y).tobytes(),np.packbits(~y).tobytes())
        if key in seen:
            return
        seen.add(key)
        names.setdefault(group,[]).append(name)
        labels.setdefault(group,[]).append(y)
    for i in range(5):
        for k,c in enumerate(ABC):
            add("practice",f"[{i}]={c}",X[:,i]==k)
    for group in ("dev","glob"):
        for i,j in itertools.combinations(range(5),2):
            for a,b in itertools.product(range(4), repeat=2):
                if group == "dev":
                    add(group,f"[{i}]={ABC[a]} or [{j}]={ABC[b]}",(X[:,i]==a)|(X[:,j]==b))
                else:
                    pattern=list("?????")
                    pattern[i],pattern[j]=ABC[a],ABC[b]
                    pattern="".join(pattern)
                    add(group,pattern,[fnmatch.fnmatchcase(s,pattern) for s in STRINGS])
    for a,b in itertools.permutations(ABC,2):
        add("count",f"count({a})>count({b})",[s.count(a)>s.count(b) for s in STRINGS])
    assert [len(names[k]) for k in ("practice","dev","glob","count")] == [20,160,160,12]
    return names,{k:np.stack(v) for k,v in labels.items()}


class StringEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.symbol=nn.Embedding(4,16)
        self.position=nn.Parameter(torch.randn(5,16))
        self.block=nn.TransformerEncoderLayer(16,2,32,dropout=0.,batch_first=True)
        self.project=nn.Sequential(nn.Flatten(),nn.Linear(80,32),nn.Tanh())
        self.head=nn.Linear(32,20)

    def forward(self,x):
        return self.project(self.block(self.symbol(x)+self.position))


def balanced(rng,y,ids,n):
    return np.stack([rng.choice(ids[y[ids]==v],n,replace=False) for v in (False,True)],axis=1)


def kernel(features):
    z=features.astype(float)
    z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-9)
    return z @ z.T + .2


def representations(seed,rng,labels,train_ids):
    torch.manual_seed(seed)
    models = [StringEncoder(),StringEncoder()]
    models[1].load_state_dict(models[0].state_dict())
    opts=[torch.optim.Adam(m.parameters(),lr=.003) for m in models]
    x=torch.tensor(X.tolist())
    def snapshot(model):
        with torch.no_grad():
            return kernel(np.array(model(x).tolist()))
    result={"initial":snapshot(models[0])}
    for t in range(1,101):
        task=rng.integers(len(labels))
        y=labels[task]
        for _ in range(3):
            idx=balanced(rng,y,train_ids,12).ravel()
            target=torch.tensor(y[idx].astype(float).tolist())
            for j,(model,opt) in enumerate(zip(models,opts)):
                opt.zero_grad()
                z=model(x[idx.tolist()])
                if j==0:
                    loss=F.binary_cross_entropy_with_logits(model.head(z)[:,task],target)
                else:
                    z=F.normalize(z)
                    direction=((2*target[:8,None]-1)*z[:8]).mean(0)
                    loss=F.binary_cross_entropy_with_logits(10*(z[8:]@direction),target[8:])
                loss.backward()
                opt.step()
        if t in (10,100):
            for model,name in zip(models,("supervised","episodic")):
                result[f"{name}-{t}"]=snapshot(model)
    raw=np.eye(4)[X]
    result["hand"]=kernel(np.column_stack([raw.reshape(len(X),-1),raw.sum(1)]))
    return result


def episodes(rng,labels,ids,repeats):
    truth=[]; sequences=[]
    for rule in labels:
        y=rule[ids]
        for _ in range(repeats):
            pairs=balanced(rng,y,np.arange(len(ids)),9)
            pairs=np.take_along_axis(pairs,rng.random(pairs.shape).argsort(1),axis=1)
            sequences.append(pairs.ravel()[:17])
            truth.append(y)
    return np.array(truth),np.array(sequences)


def score(pred, data):
    y,seq=data
    eligible=np.ones(y.shape,dtype=bool)
    np.put_along_axis(eligible,seq,False,axis=1)
    weight=.5*(eligible*y/(eligible*y).sum(1,keepdims=True)
               +eligible*(~y)/(eligible*(~y)).sum(1,keepdims=True))
    return ((pred>=0)==y)*weight


def fit(K,data,penalty,details=False):
    y,seq=data
    n=len(y)
    labels=2.*np.take_along_axis(y,seq,axis=1)-1
    # Predict every string, including the next (still unseen) verified demonstration.
    pred=np.zeros((n, K.shape[0]))
    inverse_col=[]
    evidence=np.zeros(n)
    preds,losses=[],[]
    for t in range(17):
        if t in BUDGETS:
            preds.append(pred.copy())
            losses.append(evidence.copy())
        idx=seq[:,t]
        z=pred[np.arange(n),idx]
        if t<16:
            evidence+=np.logaddexp(0,-2*labels[:,t]*z)
        # Rank-one recursive kernel ridge; columns encode orthogonalized examples.
        col=K[idx].copy()
        if inverse_col:
            prior=np.stack(inverse_col,1)
            coefficients=prior[np.arange(n),:,idx]
            col-=np.einsum("nk,nkm->nm",coefficients,prior)
        variance=col[np.arange(n),idx]
        assert np.min(variance)>-1.e-7
        if t==16:
            break
        denom=np.maximum(variance,0.)+penalty
        pred+=col*((labels[:,t]-z)/denom)[:,None]
        inverse_col.append(col/np.sqrt(denom)[:,None])
    if details:
        # Direct solution on one task verifies the recursive update and the exported readout.
        ids=seq[0,:16]
        dual=np.linalg.solve(K[np.ix_(ids,ids)]+np.eye(16)*penalty,labels[0,:16])
        assert np.allclose(dual @ K[ids],preds[-1][0],atol=1e-8)
    return np.stack(preds,1),np.stack(losses,1),(col,np.maximum(variance,0.),z,penalty)


def curve(pred,data):
    return np.stack([score(pred[:,j],data).sum(1) for j in range(len(BUDGETS))],1)


def mix(a,b,control):
    p,loss1,_=a
    q,loss2,_=b
    if isinstance(control,tuple):
        w=1/(1+np.exp(np.clip(control[0]*(loss1-loss2),-40,40)))
    else:
        w=np.full(loss1.shape,control)
    return w[:,:,None]*p+(1-w[:,:,None])*q,w


def trust_records(rng):
    ids=rng.permutation([.9,.55])
    old=rng.binomial(40,ids)
    historical=(1+old)/42
    # Each check represents a previous, independently executed string task instance.
    # Relabel physical source slots into historical-reliability order only for reporting.
    by_reliability=np.argsort(-ids)
    out={}
    for env,p in (("stable",ids),("reversed",ids[::-1])):
        hits=rng.random((32,2))<p
        for k in (0,2,8,32):
            recent=(2*historical+hits[max(0,k-8):k].sum(0))/(2+min(k,8))
            out[env,k]=(historical[by_reliability],recent[by_reliability],p[by_reliability])
    return out


def feedback(learners,data,w,record):
    y,seq=data
    pred=w[:,None]*learners[0][0][:,-1]+(1-w[:,None])*learners[1][0][:,-1]
    pre=score(pred,data)
    # Weight the query strings equally within true/false classes, just as for capability.
    mass=score(np.where(y,1.,-1.),data)
    true=np.take_along_axis(y,seq[:,-1:],axis=1)[:,0]
    result={}
    cache={}
    for (env,k),(historical,recent,p) in record.items():
        for name,rel in (("frozen",historical),("recent",recent),("blind",np.repeat(recent.mean(),2))):
            changes=[]
            for source,strength in enumerate(np.maximum(2*rel-1,0)):
                key=float(strength)
                if key not in cache:
                    effects=[]
                    for correct in (False,True):
                        ans=2.*(true if correct else ~true)-1
                        update=np.zeros_like(pred)
                        for factor,fit_result in zip((w,1-w),learners):
                            col,var,z,lam=fit_result[2]
                            update+=factor[:,None]*col*(strength*(ans-z)/(lam+strength*var))[:,None]
                        post=score(pred+update,data)
                        num=(np.maximum(post-pre,0) if correct else np.maximum(pre-post,0)).sum()
                        den=(mass-pre if correct else pre).sum()
                        effects.append(float(num/den))
                    cache[key]=effects
                changes.append(cache[key])
            harm,repair=np.array(changes).T
            result[f"{env}-{k}-{name}"]=[float(np.average(harm,weights=1-p)),float(np.average(repair,weights=p))]
    return result


def run():
    names,all_y=catalog()
    rows=[]
    raw={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        split=sum(rng.permutation(4)[X[:,i]] for i in range(5))%4
        train,dev,test=[np.flatnonzero(mask) for mask in (split<2,split==2,split==3)]
        kernels=representations(seed,rng,all_y["practice"],train)
        val=episodes(rng,all_y["dev"],dev,1)
        settings={};dev_scores={};vfits={}
        for key,full in kernels.items():
            K=full[np.ix_(dev,dev)]
            fs=[fit(K,val,r) for r in PENALTIES]
            means=[curve(f[0],val).mean() for f in fs]
            j=int(np.argmax(means))
            settings[key]=PENALTIES[j];dev_scores[key]=float(means[j]);vfits[key]=fs[j]
        learned=vfits["episodic-100"]; initial=vfits["initial"]
        for key,choices in (("fixed",(0.,.25,.5,.75,1.)),("adaptive",((.5,),(2.,),(8.,)))):
            vals=[curve(mix(learned,initial,c)[0],val).mean() for c in choices]
            j=int(np.argmax(vals));settings[key]=choices[j];dev_scores[key]=float(vals[j])
        winner=max(("supervised-100","hand","initial","fixed"),key=dev_scores.get)
        record=trust_records(rng)
        row={"seed":seed,"settings":settings,"dev_scores":dev_scores,"winner":winner}
        for group in ("glob","count"):
            data=episodes(rng,all_y[group],test,2 if group=="glob" else 12)
            fs={key:fit(full[np.ix_(test,test)],data,settings[key],True) for key,full in kernels.items()}
            scores={key:curve(f[0],data) for key,f in fs.items()}
            pair=(fs["episodic-100"],fs["initial"])
            for key in ("fixed","adaptive"):
                preds,w=mix(*pair,settings[key])
                scores[key]=curve(preds,data)
            scores["winner"]=scores[winner]
            row[group]={key:s.mean(0).tolist() for key,s in scores.items()}
            row[group]["mix"]=w.mean(0).tolist()
            row[group]["feedback"]=feedback(pair,data,w[:,-1],record)
            for key,s in scores.items():
                raw[f"{seed}-{group}-{key}"]=s
        rows.append(row)
    np.savez_compressed(ROOT/"string-trials.npz",**raw)
    (ROOT/"string-results.json").write_text(json.dumps({"programs":names,"lifetimes":rows},indent=2)+"\n")


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run()
