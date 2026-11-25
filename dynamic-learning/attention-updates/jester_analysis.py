"""Estimate uncertainty across held-out people, retaining training-seed spread separately."""
import json

from preferences import ROOT, np


def main():
    settings = json.loads((ROOT/"jester-settings.json").read_text())
    with np.load(ROOT/"jester-trials.npz") as a:
        names = ("signed","positive","covariance","leniency","fixed")
        scores = {k:a[k].mean(0) if a[k].ndim==4 else a[k] for k in names}
        neural = [a[k].mean((1,2)).tolist() for k in names[:2]]
        # The same people recur across seeds; average their counts before resampling people.
        counts = a["feedback"].mean(0)
    keys = []
    numerators,denominators = [],[]
    def append(key,num,den=None):
        keys.append(key)
        numerators.append(num)
        denominators.append(np.ones_like(num) if den is None else den)
    for model in names[:2]:
        for reference in dict.fromkeys((settings["comparator"],"signed","fixed")):
            if reference!=model:
                diff = (scores[model]-scores[reference]).mean(1)
                for j,metric in enumerate(("accuracy","brier")):
                    append(f"{model}-{reference}-{metric}",diff[:,j])
    for j,label in enumerate(("repair","harm","negative-repair-upward","negative-score-upward")):
        a,b = counts[:,:,2*j],counts[:,:,2*j+1]
        for i,name in enumerate(names[:2]):
            append(name+"-"+label,a[i],b[i])
        if j<2:
            assert np.array_equal(b[0],b[1])
            append("positive-minus-signed-"+label,a[1]-a[0],b[0])
    num,den = np.stack(numerators,axis=1),np.stack(denominators,axis=1)
    estimate = num.sum(0)/den.sum(0)
    rng = np.random.default_rng(1309)
    bootstrap = []
    for _ in range(40):
        ix = rng.integers(len(num),size=(50,len(num)))
        bootstrap.append(num[ix].sum(1)/den[ix].sum(1))
    bounds = np.quantile(np.concatenate(bootstrap),[.025,.975],axis=0)
    result = dict(settings=settings,curves={k:v.mean(0).tolist() for k,v in scores.items()},
                  seed_means=dict(zip(names[:2],neural)),
                  statistics=dict(zip(keys,np.column_stack([estimate,bounds.T]).tolist())))
    (ROOT/"jester-summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
