"""Recompute paired selector differences using the same resampled motion blocks."""
import json

from active import BUDGETS, GROUPS, POLICIES, READERS, ROOT, np


def aggregate(x):
    # session x draws x counts; weight the two sessions equally for average scores.
    with np.errstate(invalid="ignore",divide="ignore"):
        scores=(x[:,:,:3]/x[:,:,3:4]).mean(0)
        c=x.sum(0)
        return np.column_stack([scores,(c[:,4]-c[:,5])/c[:,6],(c[:,7]-c[:,8])/c[:,9],
                                c[:,10]/c[:,11],c[:,12]/c[:,13]])


def main():
    with np.load(ROOT/"active-trials.npz") as f:
        a=f["counts"]
    rng=np.random.default_rng(5821)
    ix=rng.integers(16,size=(2,4096,16))
    report={}
    for j,budget in enumerate(BUDGETS):
        for g,group in enumerate(GROUPS):
            cells={}
            for p,policy in enumerate(POLICIES):
                for m,method in enumerate(READERS):
                    v=a[:,j,p,g,m]
                    blocks=v.reshape(2,16,16,14).sum(2)
                    bootstrap=np.stack([b[i].sum(1) for b,i in zip(blocks,ix)])
                    vals=aggregate(v.sum(1)[:,None,:])[0]
                    draws=aggregate(bootstrap)
                    individual=[aggregate(row.sum(0)[None,None,:])[0].tolist() for row in v]
                    cells[policy,method]=(vals,draws,individual)
            out={}
            for (policy,method),(vals,draws,individual) in cells.items():
                ref,rs,_=cells["random",method]
                # Common eligibility contrasts occupy indices 3/4; direct rates use arm-specific baselines.
                dv=vals.copy(); dv[:3]-=ref[:3]
                ds=draws.copy(); ds[:,:3]-=rs[:,:3]
                d=ds[:,:5]
                timing=vals-cells[policy,"before"][0]
                td=draws-cells[policy,"before"][1]
                out[f"{policy}/{method}"]=dict(rates=vals[[0,1,2,5,6]].tolist(),
                    per_session=individual,
                    versus_random=np.column_stack([dv[:5],np.nanquantile(d,[.025,.975],axis=0).T]).tolist(),
                    after_minus_before=np.column_stack([timing[[0,5,6]],np.nanquantile(td[:,[0,5,6]],[.025,.975],axis=0).T]).tolist())
            report[f"{budget}-{group}"]=out
    result=dict(rates=["mixed accuracy","mixed Brier","disputed ballot Brier","repair","harm"],
                contrasts=["mixed accuracy","mixed Brier","disputed ballot Brier","common repair","common harm"],
                results=report)
    (ROOT/"active-summary.json").write_text(json.dumps(result,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
