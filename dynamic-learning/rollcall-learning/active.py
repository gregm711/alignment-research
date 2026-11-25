"""Acquire verified ballots before replaying a disputed attention-weight edit."""
import json

from learn import ROOT, N, RollModel, fit, load, metrics, np, panel, reveal, templates, tensor, torch
from replay import adjust, posterior_step

POLICIES=("random","disputed","uncertain","panel")
READERS=("constant","before","after","template")
GROUPS=("all","unexpected","other")
BUDGETS=(1,3)


def update(w,yes,label):
    likelihood=np.where(label[:,None],yes,1-yes)
    v=w*likelihood
    return v/v.sum(1,keepdims=True)


def choose(policy,w,menu_prob,target_prob,used,remaining,catalog):
    n=len(w)
    means=(menu_prob*w[:,None,:]).sum(2)
    variance=means*(1-means)
    if policy=="random":
        score=np.broadcast_to(-np.arange(used.shape[1]),used.shape).astype(float).copy()
    elif policy=="uncertain":
        score=variance.copy()
    elif policy=="disputed":
        joint=(menu_prob*(w*target_prob)[:,None,:]).sum(2)
        cov=joint-means*(w*target_prob).sum(1,keepdims=True)
        score=cov**2/variance
    else:
        p=w@catalog
        joint=(menu_prob*w[:,None,:])@catalog
        cov=joint-means[:,:,None]*p[:,None,:]
        score=(cov**2*remaining[:,None,:]).sum(2)/variance
    assert score.shape==(n,8) and np.isfinite(score).all()
    score[used]=-np.inf
    return score.argmax(1)


def traces(y,menu,old,recent,base,err,policies=POLICIES):
    archive=old if recent is None else np.vstack([old,recent])
    complete=np.where(np.isfinite(archive),archive,base)
    prob=err+(1-2*err)*complete
    prior=np.full(len(old),1/len(old)) if recent is None else np.r_[np.full(len(old),.5/len(old)),np.full(len(recent),.5/len(recent))]
    traces={}
    for start in range(0,len(menu),32):
        ids=menu[start:start+32]
        vals=y[start:start+len(ids)]
        row=np.arange(len(ids))
        w=np.tile(prior,(len(ids),1))
        for j in range(4):
            w=update(w,prob[:,ids[:,j]].T,vals[row,ids[:,j]])
        target=prob[:,ids[:,4]].T
        early=(w*target).sum(1)
        cand=ids[:,5:]
        menu_prob=prob[:,cand].transpose(1,2,0)
        remaining=np.ones((len(ids),N),bool)
        np.put_along_axis(remaining,ids,False,1)
        for policy in policies:
            post=w.copy()
            used=np.zeros(cand.shape,bool)
            choices=[]
            for t in range(1,4):
                pick=choose(policy,post,menu_prob,target,used,remaining,prob)
                assert not used[row,pick].any()
                used[row,pick]=True
                member=cand[row,pick]
                choices.append(member)
                post=update(post,menu_prob[row,pick],vals[row,member])
                if t not in BUDGETS:
                    continue
                late=(post*target).sum(1)
                claimed=.1+.8*target
                actual=vals[row,ids[:,4]]
                direct=[post@complete]
                for label in (actual,1-actual):
                    direct.append(update(post,claimed,label)@complete)
                record=(np.column_stack(choices),early,late,np.stack(direct,axis=1))
                traces.setdefault((t,policy),[]).append(record)
    return {k:tuple(np.concatenate(a,axis=0) for a in zip(*records)) for k,records in traces.items()}


def neural_readers(model,base,y,menu,trace):
    chosen,early,late,direct=trace
    result={k:[] for k in READERS[:-1]}
    bias=tensor(np.log(base/(1-base)))
    with torch.no_grad():
        r,d=model.items()
        writer=model.writer
        for start in range(0,len(y),64):
            sl=slice(start,start+64)
            ids=menu[sl,:5]
            verified=chosen[sl]
            vals=y[sl]
            n=len(ids); row=np.arange(n)
            def step(state,idx,soft=None):
                i=idx.tolist()
                if soft is not None:
                    return posterior_step(writer,state,r[i],d[i],soft,bias[i,None])
                return writer.step(state,r[i],d[i],tensor(vals[row,idx]),bias=bias[i,None])[0]
            def finish(state):
                for i in verified.T:
                    state=step(state,i)
                pred=writer.score(r.expand(n,-1,-1,-1),state[0][:,:1],bias)[:,0].sigmoid()
                return np.asarray(pred.tolist())
            prefix=writer.start(n)
            for j in range(4):
                prefix=step(prefix,ids[:,j])
            omit=finish(prefix)
            actual=tensor(vals[row,ids[:,4]])
            forks={k:[omit] for k in result}
            for report in (actual,1-actual):
                for method,soft in (("constant",.1+.8*report),("before",adjust(tensor(early[sl]),report)),
                                    ("after",adjust(tensor(late[sl]),report))):
                    forks[method].append(finish(step(prefix,ids[:,4],soft)))
            for k,v in forks.items():
                result[k].append(np.stack(v,axis=1))
    out={k:np.concatenate(v) for k,v in result.items()}
    out["template"]=direct
    return out


def collect(pred,random,actual,prob,truth,obs,group):
    ok=(pred>=.5)==truth[:,None,:]
    ref=(random>=.5)==truth[:,None,:]
    mixed=((ok[:,1]*.9+ok[:,2]*.1)*obs).sum(1)/obs.sum(1)
    brier=((.9*(pred[:,1]-truth)**2+.1*(pred[:,2]-truth)**2)*obs).sum(1)/obs.sum(1)
    # Each selector has its own omission state; directly compare flips only on common eligible targets.
    bad=(~ok[:,0])&(~ref[:,0])&obs
    good=ok[:,0]&ref[:,0]&obs
    c=[mixed,brier,(prob-actual)**2,np.ones(len(group)),
       (ok[:,1]&bad).sum(1),(ref[:,1]&bad).sum(1),bad.sum(1),
       (~ok[:,2]&good).sum(1),(~ref[:,2]&good).sum(1),good.sum(1),
       (ok[:,1]&~ok[:,0]&obs).sum(1),(~ok[:,0]&obs).sum(1),
       (~ok[:,2]&ok[:,0]&obs).sum(1),(ok[:,0]&obs).sum(1)]
    return (np.stack(c,axis=1)*group[:,None]).reshape(-1,4,len(c)).sum(1)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results=[]; meta=[]
    old=load(116)
    for cong in (117,118):
        new=load(cong)
        rng=np.random.default_rng(7000+cong)
        (prev,_),(now,rolls)=panel(old,new,rng)
        old=new
        tr,dev,_=np.split(prev,[int(.8*len(prev)),int(.9*len(prev))])
        base=(np.nansum(tr,axis=0)+1)/(np.isfinite(tr).sum(0)+2)
        dv=reveal(dev,rng)
        _,err=min((metrics(templates(tr,base,e,dv),dv,1)[:,:,1].mean(),e) for e in (.03,.1,.2,.3))
        torch.manual_seed(8120+cong)
        model=RollModel()
        fit(model,tr,base,192,3000+cong,"learned",.003)
        fit(model,now[:32],base,32,4000+cong,"learned",.0008)
        y=np.repeat(now[288:544],4,axis=0)
        assert len(y)==1024
        menu=(rng.random(y.shape)+2*np.isnan(y)).argsort(1)[:,:13]
        assert np.isfinite(np.take_along_axis(y,menu,1)).all()
        obs=np.isfinite(y)
        np.put_along_axis(obs,menu,False,1)
        assert obs.sum(1).min()>=11
        y=np.nan_to_num(y)
        trace=traces(y,menu,tr,now[:32],base,err)
        actual=y[np.arange(len(y)),menu[:,4]]
        early=trace[1,"random"][1]
        unusual=np.where(actual,early,1-early)<.2
        groups=(np.ones(len(y),bool),unusual,~unusual)
        counts=[]; diversity={}
        for t in BUDGETS:
            arms=[]
            for policy in POLICIES:
                info=trace[t,policy]
                preds=neural_readers(model,base,y,menu,info)
                if policy=="random":
                    random=preds
                    np.testing.assert_array_equal(info[0],menu[:,5:5+t])
                arms.append([[collect(preds[m],random[m],actual,info[2],y,obs,g) for m in READERS] for g in groups])
                diversity[f"{t}-{policy}"]=float((info[0]!=trace[t,"random"][0]).any(1).mean())
            counts.append(arms)
        results.append(counts)
        meta.append(dict(congress=cong,motions=rolls[288:544].tolist(),unusual_orders=int(unusual.sum()),
                         unusual_motions=int(unusual.reshape(-1,4).any(1).sum()),changed_checks=diversity))
        print("finished",cong,flush=True)
    raw=np.array(results)
    np.testing.assert_allclose(raw[:,:,:,1:].sum(3),raw[:,:,:,0],atol=1e-8)
    np.savez_compressed(ROOT/"active-trials.npz",counts=raw)
    (ROOT/"active-settings.json").write_text(json.dumps(meta,indent=1)+"\n")


if __name__=="__main__":
    main()
