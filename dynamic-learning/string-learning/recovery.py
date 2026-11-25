"""Preserve one past projection and test learned task routing on fresh lifetimes."""
import json
import order
from order import e, np, torch, ROOT, X, nn
from analyse import paired

SEEDS=range(551,559)
GROUPS=("glob","count","xor","equal")


def labels_with_extra():
    names,labels=order.programs()
    def identity(y):
        return min(np.packbits(y).tobytes(),np.packbits(~y).tobytes())
    seen={identity(y) for mat in labels.values() for y in mat}
    for group in ("xor","equal"):
        functions=[]; descriptions=[]
        if group=="xor":
            for i,j in e.itertools.combinations(range(5),2):
                for a,b in e.itertools.product(range(4),repeat=2):
                    functions.append((X[:,i]==a)^(X[:,j]==b))
                    descriptions.append(f"[{i}]={e.ABC[a]} xor [{j}]={e.ABC[b]}")
        else:
            for a,b in e.itertools.combinations(e.ABC,2):
                functions.append(np.array([s.count(a)==s.count(b) for s in e.STRINGS]))
                descriptions.append(f"count({a})=count({b})")
        for y in functions:
            assert identity(y) not in seen
            seen.add(identity(y))
        names[group]=descriptions;labels[group]=np.stack(functions)
    return names,labels


def average_validation(kernels,data):
    settings={};scores={};fits={}
    for name,K in kernels.items():
        grid=[[e.fit(K,data_set,r) for data_set in data] for r in e.PENALTIES]
        vals=[np.mean([e.curve(f[0],d).mean() for f,d in zip(pair,data)]) for pair in grid]
        j=int(np.argmax(vals))
        settings[name]=e.PENALTIES[j];scores[name]=float(vals[j]);fits[name]=grid[j]
    for name,controls in (("fixed",(0.,.25,.5,.75,1.)),("adaptive",((.5,),(2.,),(8.,)))):
        vals=[]
        for c in controls:
            vals.append(np.mean([e.curve(e.mix(a,b,c)[0],d).mean()
                                 for a,b,d in zip(fits["old-projection"],fits["current"],data)]))
        j=int(np.argmax(vals))
        settings[name]=controls[j];scores[name]=float(vals[j])
    return settings,scores


def run(seeds=SEEDS,feedback_audit=None):
    names,labels=labels_with_extra()
    raw={};rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        seq="position-last" if seed%2 else "count-last"
        split=sum(rng.permutation(4)[X[:,i]] for i in range(5))%4
        train,dev,test=[np.flatnonzero(mask) for mask in (split<2,split==2,split==3)]
        x=torch.tensor(X.tolist())
        model,states=order.train(seed,rng,labels,train,x,orders=[seq])
        final=states[seq+"-200"]; old=states[seq+"-100"]
        selected={"initial":states["initial"],"current":final,
                  "old-projection":{k:old[k] if k.startswith("project.") else v for k,v in final.items()}}
        features={}
        for name,state in selected.items():
            model.load_state_dict(state)
            with torch.no_grad():
                features[name]=np.array(model(x).tolist())
        pos=np.eye(4)[X]
        features["hand"]=np.column_stack([pos.reshape(len(X),-1),pos.sum(1)])
        def kernels(ids):
            result={k:e.kernel(z[ids]) for k,z in features.items()}
            result["pooled"]=(result["current"]+result["old-projection"])/2
            return result
        devsets=[e.episodes(rng,labels[g],dev,n) for g,n in (("dev",1),("count-dev",8))]
        settings,scores=average_validation(kernels(dev),devsets)
        sets={g:e.episodes(rng,labels[g],test,2 if g in ("glob","xor") else 16) for g in GROUPS}
        if feedback_audit is not None:
            advice_records=order.source_histories(np.random.default_rng(60000+seed),
                np.vstack([labels["practice"],labels["count-practice"]]),dev)
        K=kernels(test)
        probs={}; predictions={}
        for g,data in sets.items():
            base={k:e.fit(mat,data,settings[k],True) for k,mat in K.items()}
            pair=base["old-projection"],base["current"]
            predictions[g]=pair[0][0],pair[1][0]
            values={k:v[0] for k,v in base.items()}
            values["old-matched"]=e.fit(K["old-projection"],data,settings["current"])[0]
            values["fixed"]=e.mix(*pair,settings["fixed"])[0]
            values["adaptive"],probs[g]=e.mix(*pair,settings["adaptive"])
            if feedback_audit is not None and g in ("count","glob"):
                feedback_audit(seed,g,pair,probs[g][:,-1],data,advice_records)
            for key,pred in values.items():
                raw[f"{seed}-{g}-{key}"]=e.curve(pred,data)
            raw[f"{seed}-{g}-probability"]=probs[g]
        for g,data in sets.items():
            p,q=predictions[g]
            pool=rng.integers(len(GROUPS),size=len(data[0]))
            w=np.empty_like(probs[g])
            for h,other in enumerate(GROUPS):
                mask=pool==h
                ids=rng.integers(len(probs[other]),size=mask.sum())
                if other==g:
                    ids=np.where(ids==np.flatnonzero(mask),(ids+1)%len(ids),ids)
                w[mask]=probs[other][ids]
            raw[f"{seed}-{g}-unrelated"]=e.curve(w[:,:,None]*p+(1-w[:,:,None])*q,data)
        rows.append(dict(seed=seed,sequence=seq,settings=settings,dev=scores,
                         extra_parameters=sum(v.numel() for k,v in old.items() if k.startswith("project."))))
    prefix="recovery" if feedback_audit is None else "joint"
    np.savez_compressed(ROOT/(prefix+"-trials.npz"),**raw)
    (ROOT/(prefix+"-settings.json")).write_text(json.dumps({"programs":names,"rows":rows},indent=2)+"\n")
    return raw, rows


def summarise(raw,rows):
    report={}
    groups=(*GROUPS,"first-family","last-family")
    methods=("current","initial","hand","old-projection","old-matched","pooled","fixed","adaptive","unrelated")
    for group in groups:
        values={k:[] for k in methods}; probabilities=[]
        for row in rows:
            seed=row["seed"]
            first,last=("count","glob") if row["sequence"]=="position-last" else ("glob","count")
            g={"first-family":first,"last-family":last}.get(group,group)
            for key in methods:
                values[key].append(raw[f"{seed}-{g}-{key}"].mean(0))
            probabilities.append(raw[f"{seed}-{g}-probability"].mean(0))
        vals={k:np.stack(v) for k,v in values.items()}
        result={"curves":{k:v.mean(0).tolist() for k,v in vals.items()},
                "old_weight":np.mean(probabilities,0).tolist(),"contrasts":{}}
        for l,r in (("adaptive","current"),("adaptive","pooled"),("adaptive","unrelated"),("adaptive","fixed"),
                    ("pooled","current"),("old-matched","current"),("adaptive","hand")):
            result["contrasts"][l+" minus "+r]=paired((vals[l]-vals[r]).mean(1))
        report[group]=result
    (ROOT/"recovery-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for g in groups:
        print(g, {k:round(100*np.mean(v),2) for k,v in report[g]["curves"].items()})


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    summarise(*run())
