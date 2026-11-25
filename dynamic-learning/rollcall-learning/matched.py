"""Practice the same weight writes on randomly or deliberately selected evidence."""
import copy
import json

from active import BUDGETS,GROUPS,collect,neural_readers,traces
from learn import ROOT,N,F,RollModel,load,metrics,np,panel,reveal,templates,tensor,torch
from replay import adjust,posterior_step

CHOICES=("random","disputed")


def menu_for(y,rng):
    menu=(rng.random(y.shape)+2*np.isnan(y)).argsort(1)[:,:13]
    observed=np.isfinite(y)
    assert np.isfinite(np.take_along_axis(y,menu,axis=1)).all()
    np.put_along_axis(observed,menu,False,axis=1)
    return np.nan_to_num(y),menu,observed


def sequence(model,bias,read,direction,y,ids,soft=None):
    n=len(ids)
    state=model.writer.start(n)
    for j,col in enumerate(ids.T):
        i=col.tolist()
        if j==4:
            state=posterior_step(model.writer,state,read[i],direction[i],soft,bias[i,None])
        else:
            label=tensor(y[np.arange(n),col])
            state,_=model.writer.step(state,read[i],direction[i],label,bias=bias[i,None])
    return state


def practice(models,data,archive,base,err,steps,seed,rate):
    rng=np.random.default_rng(seed)
    row=rng.integers(len(data),size=24*steps)
    y,menu,obs=menu_for(data[row],rng)
    info=traces(y,menu,archive,None,base,err,CHOICES)
    misreport=rng.random(len(y))<.1
    label=np.logical_xor(y[np.arange(len(y)),menu[:,4]],misreport)
    bias=tensor(np.log(base/(1-base)))
    for model,policy in zip(models,CHOICES):
        opt=torch.optim.Adam(model.parameters(),lr=rate)
        for t in range(steps):
            sl=slice(t*24,(t+1)*24)
            truth=tensor(y[sl]); mask=tensor(obs[sl])
            r,d=model.items()
            losses=[]
            for k in BUDGETS:
                selected,_,belief,_=info[k,policy]
                ids=np.column_stack([menu[sl,:5],selected[sl]])
                soft=adjust(tensor(belief[sl]),tensor(label[sl]))
                state=sequence(model,bias,r,d,y[sl],ids,soft)
                z=model.writer.score(r.expand(24,-1,-1,-1),state[0],bias)
                lp=F.logsigmoid((2*truth[:,None]-1)*z)*mask[:,None]
                den=mask.sum(1)
                losses.append((-lp[:,0].sum(-1)/den-.5*(torch.logsumexp(lp[:,1:].sum(-1),dim=1)-np.log(2))/den).mean())
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            opt.step()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results=[]; control=[]; saved={}; meta=[]
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
        first=RollModel(); models=(first,copy.deepcopy(first))
        cut=len(tr)//3
        practice(models,tr[cut:],tr[:cut],base,err,192,3000+cong,.003)
        practice(models,now[:32],tr,base,err,32,4000+cong,.0008)
        if len(now)<800:
            raise ValueError("Final 256 overlap earlier prospective tests")
        y,menu,obs=menu_for(np.repeat(now[-256:],4,axis=0),rng)
        trace=traces(y,menu,tr,now[:32],base,err,CHOICES)
        actual=y[np.arange(len(y)),menu[:,4]]
        early=trace[1,"random"][1]
        surprising=np.where(actual,early,1-early)<.2
        groups=(np.ones(len(y),bool),surprising,~surprising)
        session=[]; refs=[]
        for k in BUDGETS:
            settings=[]; controls=[]
            for policy in CHOICES:
                info=trace[k,policy]
                preds=[neural_readers(m,base,y,menu,info)["after"] for m in models]
                settings.append([[collect(v,preds[0],actual,info[2],y,obs,g) for v in preds] for g in groups])
                controls.append([collect(info[3],info[3],actual,info[2],y,obs,g) for g in groups])
                if k==3 and policy=="disputed":
                    with torch.no_grad():
                        net=models[1]
                        r,d=net.items(); bias=tensor(np.log(base/(1-base)))
                        ids=np.column_stack([menu[:1,:5],info[0][:1]])
                        state=sequence(net,bias,r,d,y[:1],ids,adjust(tensor(info[2][:1]),tensor(actual[:1])))
                        for name,param in net.state_dict().items():
                            saved[f"{cong}/model/{name}"]=np.asarray(param.tolist(),dtype=np.int64 if name=="tokens" else np.float32)
                        saved[f"{cong}/value"]=np.array(state[0][0,0].tolist(),np.float32)
                        saved[f"{cong}/bias"]=np.array(bias.tolist(),np.float32)
                        saved[f"{cong}/prediction"]=preds[1][0,1].copy()
            session.append(settings); refs.append(controls)
        results.append(session); control.append(refs)
        meta.append(dict(congress=cong,eligible=len(now),rollnumbers=rolls[-256:].tolist(),
                         unexpected_orders=int(surprising.sum()),unexpected_motions=int(surprising.reshape(-1,4).any(1).sum())))
        print("finished",cong,flush=True)
    a=np.array(results)
    np.testing.assert_allclose(a[:,:,:,1:].sum(3),a[:,:,:,0],atol=1e-8)
    np.savez_compressed(ROOT/"matched-trials.npz",counts=a,template=np.array(control))
    np.savez_compressed(ROOT/"matched-weights.npz",**saved)
    (ROOT/"matched-settings.json").write_text(json.dumps(meta,indent=1)+"\n")


if __name__=="__main__":
    main()
