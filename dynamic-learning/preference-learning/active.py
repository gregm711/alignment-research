"""Cross one chosen rating with one weight-update rule from an identical starting head."""
import json

from ratings import ROOT, head, kernels, load, np

NAMES = ("leniency", "genre", "covariance")


def directions(kernel, reg, seen, menu):
    observed = kernel[np.ix_(seen,seen)] + reg*np.eye(len(seen))
    partial = kernel[:,menu] - kernel[:,seen] @ np.linalg.solve(observed,kernel[np.ix_(seen,menu)])
    denom = reg + partial[menu,np.arange(len(menu))]
    assert (denom>0).all()
    return partial/denom, (partial**2).mean(0)/denom


def main():
    report = json.loads((ROOT/"ratings.json").read_text())
    ratings,genres,_,test,_ = load()
    base = (np.nansum(ratings,axis=0)+4*np.nanmean(ratings))/(np.isfinite(ratings).sum(0)+4)
    channels = kernels(ratings,genres,base)
    models = {name:(1+report["chosen"][name][0]*channels[name],report["chosen"][name][1]) for name in NAMES}
    K,reg = models["genre"]
    scores = []
    counts = []
    picks = []
    for ids,y,_ in test:
        seen,menu,future = ids[:4],ids[4:12],ids[12:]
        w = np.linalg.solve(K[np.ix_(seen,seen)]+reg*np.eye(4),y[:4]-base[seen])
        start = base+K[:,seen]@w
        old_correct = (start[future]>=.5)==y[12:]
        per_reader,events,chosen = [],[],[]
        for name in NAMES:
            direction,benefit = directions(*models[name],seen,menu)
            chosen.append(int(benefit.argmax()))
            honest = np.clip(start[:,None]+direction*(y[4:12]-start[menu]),0,1)
            false = np.clip(start[:,None]+direction*(1-y[4:12]-start[menu]),0,1)
            if name=="genre":
                for j,item in enumerate(menu):
                    refit = head(base,K,reg,np.r_[seen,item],np.r_[y[:4],y[4+j]])
                    np.testing.assert_allclose(honest[:,j], refit, atol=1e-12, rtol=0)
            honest, false = honest[future], false[future]
            correct = (honest>=.5)==y[12:,None]
            wrong = (false>=.5)!=y[12:,None]
            per_reader.append(np.stack([correct.mean(0),((honest-y[12:,None])**2).mean(0)],axis=1))
            events.append(np.stack([(correct & ~old_correct[:,None]).sum(0),
                                    (wrong & old_correct[:,None]).sum(0)],axis=1))
        picks.append(chosen)
        def select(values):
            table = np.stack(values)
            return np.stack([table.mean(1),table[:,chosen[1]],table[:,chosen[2]]])
        scores.append(select(per_reader))
        counts.append(select(events))
    scores,counts,picks = np.array(scores),np.array(counts),np.array(picks)
    # Eligibility depends only on the common starting head, independently of the later choice.
    eligible = []
    for ids,y,_ in test:
        correct = (head(base,K,reg,ids[:4],y[:4])[ids[12:]]>=.5)==y[12:]
        eligible.append([(~correct).sum(),correct.sum()])
    eligible = np.array(eligible)
    boot = np.random.default_rng(7214).integers(len(test),size=(6000,len(test)))
    def interval(diff,denom):
        raw = diff.sum(0)/denom.sum(0)
        draws = diff[boot].sum(1)/denom[boot].sum(1)
        return np.column_stack([raw,np.quantile(draws,[.025,.975],axis=0).T]).tolist()
    contrasts = {}
    for label,((sa,ra),(sb,rb)) in {
        "end-to-end":((2,2),(1,1)), "reader-only":((2,2),(2,1)),
        "selector-only":((2,1),(1,1)), "covariance-choice-minus-uniform":((2,2),(0,2)),
        "reader-minus-leniency":((2,2),(2,0))}.items():
        contrasts[label] = dict(metrics=interval(scores[:,sa,ra]-scores[:,sb,rb],np.ones((len(test),2))),
                                feedback=interval(counts[:,sa,ra]-counts[:,sb,rb],eligible))
    summary = dict(selectors=["uniform","genre","covariance"],readers=NAMES,metrics=["accuracy","brier"],
                   feedback_metrics=["repair","harm"],contrasts=contrasts,
                   means=scores.mean(0).tolist(),feedback=(counts.sum(0)/eligible.sum(0)).tolist(),
                   eligible=eligible.sum(0).tolist(),different_choices=int(np.sum(picks[:,1]!=picks[:,2])),
                   users=len(test))
    (ROOT/"active.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(ROOT/"active-trials.npz",scores=scores,counts=counts,eligible=eligible,picks=picks)


if __name__=="__main__":
    main()
