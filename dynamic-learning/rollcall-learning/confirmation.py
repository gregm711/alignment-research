"""Confirm the replay and rare-truth results on two later House sessions."""
import json

from learn import (ROOT, RollModel, fit, load, metrics, np, panel, reveal, templates, torch)
from replay import predictions

NAMES=("ordinary","constant","prospective","self-replay","template-replay","template-prospective","matrix-mean","output-mean","direct-template")
GROUPS=("all","surprising","usual","middle")


def per_motion(pred,y,obs,group):
    correct=(pred>=.5)==y
    wrong0=(~correct[0])&obs
    right0=correct[0]&obs
    mixed_accuracy=(np.sum(correct*obs,axis=2)/obs.sum(1))[[1,2]].T@[.9,.1]
    mixed_brier=((((pred-y)**2)*obs).sum(2)/obs.sum(1))[[1,2]].T@[.9,.1]
    events=[mixed_accuracy,mixed_brier,np.ones(len(group)),(correct[1]&wrong0).sum(1),
            wrong0.sum(1),((~correct[2])&right0).sum(1),right0.sum(1)]
    return (np.stack(events,axis=1)*group[:,None]).reshape(-1,4,7).sum(1)


def rates(counts):
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.stack([counts[...,0]/counts[...,2],counts[...,1]/counts[...,2],
                     counts[...,3]/counts[...,4],counts[...,5]/counts[...,6]],axis=-1)


def summarize(sessions):
    report={}
    for g,group in enumerate(GROUPS):
        raw=sessions[:,g]  # period x method x motion x metric
        summed=raw.sum(2)
        point=rates(summed.sum(0))
        # Give each session the same weight for mixed accuracy and Brier. Pool eligible feedback counts.
        point[:,:2]=rates(summed)[:,:,:2].mean(0)
        blocks=raw.reshape(2,len(NAMES),16,16,7).sum(3)
        rng=np.random.default_rng(719)
        draws=[]
        for _ in range(32):
            batch=[]
            for period in range(2):
                ids=rng.integers(16,size=(128,16))
                batch.append(blocks[period].transpose(1,0,2)[ids].sum(1))
            a=np.stack(batch)
            pooled=rates(a.sum(0))
            pooled[:,:,:2]=rates(a)[:,:,:,:2].mean(0)
            draws.append(pooled)
        draws=np.concatenate(draws)
        out={}
        for k,method in enumerate(NAMES):
            out[method]=dict(rates=point[k].tolist(),per_congress=rates(summed[:,k]).tolist())
            for key,ref in (("versus_constant",1),("versus_prospective_template",NAMES.index("template-prospective"))):
                diff=draws[:,k]-draws[:,ref]
                delta=point[k]-point[ref]
                out[method][key]=np.column_stack([delta,np.nanquantile(diff,[.025,.975],axis=0).T]).tolist()
            out[method]["valid_bootstrap_draws"]=np.isfinite(draws[:,k]).sum(0).tolist()
        report[group]=out
    return report


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    old=load(116)
    results=[]; metadata=[]
    for cong in (117,118):
        new=load(cong)
        rng=np.random.default_rng(7000+cong)
        (prev,_),(now,rolls)=panel(old,new,rng)
        old=new
        a,b=int(.8*len(prev)),int(.9*len(prev))
        tr,dev=prev[:a],prev[a:b]
        assert len(now)>=288
        base=(np.nansum(tr,axis=0)+1)/(np.isfinite(tr).sum(0)+2)
        dv=reveal(dev,rng)
        _,err=min((metrics(templates(tr,base,e,dv),dv,1)[:,:,1].mean(),e) for e in (.03,.1,.2,.3))
        torch.manual_seed(8120+cong)
        model=RollModel()
        fit(model,tr,base,192,3000+cong,"learned",.003)
        fit(model,now[:32],base,32,4000+cong,"learned",.0008)
        data=reveal(now[32:288],rng,4)
        preds,trust=predictions(model,tr,now[:32],base,err,data)
        groups=(np.ones(len(trust),bool),trust<.2,trust>.8,(trust>=.2)&(trust<=.8))
        y,_,_,mask=data
        res=np.array([[per_motion(preds[k],y,mask,g) for k in NAMES] for g in groups])
        np.testing.assert_allclose(res[1:].sum(0),res[0],atol=1e-9)
        results.append(res)
        metadata.append(dict(congress=cong,template_error=err,rollnumbers=rolls[32:288].tolist(),
                             orders=[int(g.sum()) for g in groups],
                             motions=[int(g.reshape(-1,4).any(1).sum()) for g in groups]))
        print("finished Congress",cong,flush=True)
    raw=np.array(results)
    np.savez_compressed(ROOT/"confirmation-trials.npz",counts=raw)
    out=dict(groups=GROUPS,methods=NAMES,metrics=["mixed_accuracy","mixed_brier","truthful_repair","false_harm"],
             metadata=metadata,results=summarize(raw))
    (ROOT/"confirmation.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
