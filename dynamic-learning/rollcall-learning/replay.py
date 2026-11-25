"""Use subsequent independently checked ballots to revisit one disputed weight update."""
import json

from learn import (ROOT, N, F, RollModel, load, neural, np, panel, reveal, tensor, torch)


def adjust(prob,report):
    odds_yes=torch.where(report==1,9.,1/9.)
    return (odds_yes*prob)/(odds_yes*prob+1-prob)


def template_evidence(old,recent,base,err,ids,sy):
    votes=np.where(np.isfinite(np.vstack([old,recent])),np.vstack([old,recent]),base)
    prob=err+(1-2*err)*votes
    prior=np.r_[np.full(len(old),.5/len(old)),np.full(len(recent),.5/len(recent))]
    w=np.broadcast_to(prior,(len(ids),len(prior))).copy()
    target=prob[:,ids[:,4]].T
    for j in (0,1,2,3,5,6,7):
        pj=prob[:,ids[:,j]].T
        w*=np.where(sy[:,j,None],pj,1-pj)
        if j==3:
            early=(w*target).sum(1)/w.sum(1)
    w/=w.sum(1,keepdims=True)
    belief=(w*target).sum(1)
    out=[w@votes]
    for label in (sy[:,4],1-sy[:,4]):
        claimed=.1+.8*target
        updated=w*np.where(label[:,None],claimed,1-claimed)
        out.append((updated@votes)/updated.sum(1,keepdims=True))
    return early,belief,np.stack(out)


def posterior_step(writer,state,r,d,y,bias):
    changed,_=writer.step(state,r,d,y,bias=bias)
    z=writer.score(r[:,None],state[0][:,1:],bias[:,:,None])[:,:,0]
    expected=y[:,None]*F.logsigmoid(z)+(1-y[:,None])*F.logsigmoid(-z)
    return changed[0],state[1]+expected[:,0]-expected[:,1]


def predictions(model,old,recent,base,err,data):
    _,all_ids,all_sy,_=data
    methods=("ordinary","constant","prospective","self-replay","template-replay","template-prospective","matrix-mean","output-mean","direct-template")
    scores={k:[] for k in methods}
    trust=[]; checked=[]
    bias=tensor(np.log(base/(1-base)))
    writer=model.writer
    with torch.no_grad():
        r,d=model.items()
        for start in range(0,len(all_ids),64):
            ids=all_ids[start:start+64]; sy=all_sy[start:start+64]; n=len(ids)
            def step(state,j,target=None):
                i=ids[:,j].tolist()
                if target is None:
                    return writer.step(state,r[i],d[i],tensor(sy[:,j]),bias=bias[i,None])[0]
                return posterior_step(writer,state,r[i],d[i],target,bias[i,None])
            def read(state):
                return writer.score(r.expand(n,-1,-1,-1),state[0][:,:1],bias)[:,0].sigmoid()
            prefix=writer.start(n)
            for j in range(4):
                prefix=step(prefix,j)
            omit=prefix
            for j in (5,6,7):
                omit=step(omit,j)
            p0,p1=[read(s)[torch.arange(n),ids[:,4].tolist()] for s in (prefix,omit)]
            early_template,p_template,direct=template_evidence(old,recent,base,err,ids,sy)
            actual=tensor(sy[:,4])
            trust.extend(torch.where(actual==1,p1,1-p1).tolist())
            checked.append(np.asarray(read(omit).tolist()))
            scores["direct-template"].append(direct)
            values={k:[checked[-1]] for k in methods[:-1]}
            completed=[]
            for bit in (0.,1.):
                state=step(prefix,4,torch.full_like(actual,bit))
                for j in (5,6,7):
                    state=step(state,j)
                completed.append(state)
            zero,one=completed
            for report in (actual,1-actual):
                belief=adjust(p1,report)
                v=torch.lerp(zero[0],one[0],belief[:,None,None,None,None])
                values["matrix-mean"].append(np.array(read((v,None)).tolist()))
                mean=torch.lerp(read(zero),read(one),belief[:,None])
                values["output-mean"].append(np.array(mean.tolist()))
                for k,soft in (("ordinary",report),("constant",.1+.8*report),
                               ("prospective",adjust(p0,report)),("self-replay",adjust(p1,report)),
                               ("template-replay",adjust(tensor(p_template),report)),("template-prospective",adjust(tensor(early_template),report))):
                    state=step(prefix,4,soft)
                    for j in (5,6,7):
                        state=step(state,j)
                    values[k].append(np.asarray(read(state).tolist()))
            for k,v in values.items():
                scores[k].append(np.stack(v))
    return {k:np.concatenate(v,axis=1) for k,v in scores.items()},np.array(trust)


def outcomes(pred,data,group):
    truth,_,_,observed=data
    truth=truth[group]
    obs=observed[group]
    pred=pred[:,group]
    equal=(pred>=.5)==truth
    bad=(~equal[0])&obs
    good=equal[0]&obs
    counts=[int((equal[1]&bad).sum()),int(bad.sum()),int((~equal[2]&good).sum()),int(good.sum())]
    denom=obs.sum(1)
    acc=(equal&obs).sum(2)/denom
    brier=(((pred-truth)**2)*obs).sum(2)/denom
    return dict(orders=len(denom),counts=counts,accuracy=acc.mean(1).tolist(),brier=brier.mean(1).tolist())


def main():
    torch.set_num_threads(1)
    records=json.loads((ROOT/"results.json").read_text())
    old=load(110)
    rows=[]
    with np.load(ROOT/"recent-weights.npz") as archive:
        for rec in records:
            cong=rec["congress"]
            new=load(cong)
            rng=np.random.default_rng(7000+cong)
            (prev,_),(now,rollids)=panel(old,new,rng)
            old=new
            tr,dev,_=np.split(prev,[int(.8*len(prev)),int(.9*len(prev))])
            base=(np.nansum(tr,axis=0)+1)/(np.isfinite(tr).sum(0)+2)
            reveal(dev,rng)
            data=reveal(now[32:288],rng,4)
            assert np.array_equal(rollids[32:288],rec["blocks"]["early"]["rollnumbers"])
            model=RollModel()
            prefix=f"{cong}/"
            model.load_state_dict({k[len(prefix):]:torch.tensor(archive[k].tolist())
                                   for k in archive.files if k.startswith(prefix)})
            preds,trust=predictions(model,tr,now[:32],base,rec["tuning"][0],data)
            # Exact labels should give the same ordinary predictions under either witness likelihood expression.
            original=neural(model,"learned",base,data)[8]
            np.testing.assert_allclose(preds["ordinary"][1],original,atol=2e-6,rtol=0)
            labels,_,_,mask=data
            acc=((((original>=.5)==labels)&mask).sum(1)/mask.sum(1)).mean()
            assert abs(acc-rec["blocks"]["early"]["curves"]["recent"][3][0])<1e-10
            groups=dict(all=np.ones(len(trust),bool),surprising=trust<.2,usual=trust>.8,middle=(trust>=.2)&(trust<=.8))
            row=dict(congress=cong,results={g:{k:outcomes(pred,data,selector) for k,pred in preds.items()}
                                            for g,selector in groups.items()})
            rows.append(row)
    (ROOT/"replay-trials.json").write_text(json.dumps(rows,indent=1)+"\n")
    report={}
    for g in rows[0]["results"]:
        by_method={}
        for k in rows[0]["results"][g]:
            data=[r["results"][g][k] for r in rows]
            counts=np.array([v["counts"] for v in data])
            acc=np.array([v["accuracy"] for v in data])
            loss=np.array([v["brier"] for v in data])
            by_method[k]=dict(mixed_accuracy=acc[:,1:]@[.9,.1],mixed_brier=loss[:,1:]@[.9,.1],counts=counts)
        out={}
        rng=np.random.default_rng(785)
        ix=rng.integers(len(rows),size=(8000,len(rows)))
        for k,v in by_method.items():
            c=v["counts"]; ref=by_method["constant"]; other=ref["counts"]
            totals=c.sum(0)
            rec=dict(accuracy=v["mixed_accuracy"].mean(),brier=v["mixed_brier"].mean(),
                     repair=totals[0]/totals[1],harm=totals[2]/totals[3],versus_constant={})
            for j,l in ((0,"repair"),(2,"harm")):
                value=lambda a,i:a[i,j].sum(-1)/a[i,j+1].sum(-1)
                diff=value(c,ix)-value(other,ix)
                rec["versus_constant"][l]=[rec[l]-other[:,j].sum()/other[:,j+1].sum(),*np.quantile(diff,[.025,.975])]
                rec["versus_constant"][l+"_by_transition"]=(c[:,j]/c[:,j+1]-other[:,j]/other[:,j+1]).tolist()
            rec["versus_constant"]["mixed_accuracy"]=(v["mixed_accuracy"]-ref["mixed_accuracy"]).tolist()
            out[k]=rec
        report[g]=out
    (ROOT/"replay-summary.json").write_text(json.dumps(report,indent=1)+"\n")


if __name__=="__main__":
    main()
