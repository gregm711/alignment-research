"""Joint capability and feedback evaluation on fresh histories."""
import json

import recovery
from recovery import e, np, torch, ROOT, paired


def responses(pair,weight,data,eligibility):
    y,seq=data
    pred=sum(w[:,None]*f[0][:,-1] for w,f in zip((weight,1-weight),pair))
    correct=(pred>=0)==y
    label=np.take_along_axis(y,seq[:,-1:],axis=1)[:,0]
    values=[]
    for scale in np.linspace(0,1,21):
        outcome=[]
        for truthful in (False,True):
            target=2.*(label if truthful else ~label)-1
            drift=np.zeros_like(pred)
            for w,f in zip((weight,1-weight),pair):
                col,var,z,penalty=f[2]
                drift+=w[:,None]*col*(scale*(target-z)/(penalty+scale*var))[:,None]
            after=(pred+drift>=0)==y
            mass=eligibility[int(truthful)]
            outcome.append(float(((after!=correct)*mass).sum()/mass.sum()))
        values.append(outcome)
    assert values[0]==[0.,0.]
    return np.array(values)


def policy_rates(values,record):
    strength,prob=record
    results=values[strength]
    return np.array([np.mean((results[:,:,j]*p).sum(1)/p.sum(1))
                     for j,p in enumerate((1-prob,prob))])


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    feedback={}
    def audit(seed,group,pair,w,data,records):
        y,_=data
        mass=e.score(np.where(y,1.,-1.),data)
        current=pair[1][0][:,-1]
        adaptive=w[:,None]*pair[0][0][:,-1]+(1-w[:,None])*current
        a=(adaptive>=0)==y
        c=(current>=0)==y
        shared=[mass*a*c,mass*(~a)*(~c)]
        self_only=[mass*a,mass*(~a)]
        adaptive_shared=responses(pair,w,data,shared)
        current_shared=responses(pair,np.zeros_like(w),data,shared)
        solo=responses(pair,w,data,self_only)
        result={"jointly_correct_mass":float(shared[0].sum()),
                "jointly_wrong_mass":float(shared[1].sum())}
        for env in ("stable","reversed"):
            recent=records[f"{env}-8-recent"]
            result[env]={
                "adaptive_minus_current":(policy_rates(adaptive_shared,recent)-policy_rates(current_shared,recent)).tolist(),
                "recent_minus_frozen":(policy_rates(solo,recent)-policy_rates(solo,records[f"{env}-8-frozen"])).tolist(),
                "recent_minus_blind":(policy_rates(solo,recent)-policy_rates(solo,records[f"{env}-8-blind"])).tolist()}
        feedback[f"{seed}-{group}"]=result
    raw,rows=recovery.run(range(601,609),audit)
    report={"by_lifetime_feedback":feedback}
    for index,which in enumerate(("first","last")):
        cap={k:[] for k in ("current","adaptive","hand","pooled")}
        deltas={}
        for row in rows:
            seed=row["seed"]
            groups=("count","glob") if row["sequence"]=="position-last" else ("glob","count")
            group=groups[index]
            for key in cap:
                cap[key].append(float(raw[f"{seed}-{group}-{key}"].mean()))
            for env in ("stable","reversed"):
                for contrast,vals in feedback[f"{seed}-{group}"][env].items():
                    for j,endpoint in enumerate(("harm","repair")):
                        deltas.setdefault(f"{env}-{contrast}-{endpoint}",[]).append(vals[j])
        report[which]={"accuracy":{k:float(np.mean(v)) for k,v in cap.items()},
                       "capability":{k:paired(np.subtract(cap["adaptive"],cap[k])) for k in ("current","hand","pooled")},
                       "feedback":{k:paired(v) for k,v in deltas.items()}}
    (ROOT/"joint-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for group in ("first","last"):
        r=report[group]
        for key,v in {**r["capability"],**r["feedback"]}.items():
            print(group,key,[round(100*z,2) for z in (v["mean"],*v["interval"])])


if __name__=="__main__":
    main()
