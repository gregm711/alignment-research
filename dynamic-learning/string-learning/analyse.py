"""Summarize paired lifetime differences and program variation from the saved trials."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def paired(values):
    v=np.asarray(values)
    boot=np.random.default_rng(1501).choice(v,(10000,len(v))).mean(1)
    return dict(mean=float(v.mean()),interval=np.quantile(boot,[.025,.975]).tolist(),
                per_lifetime=v.tolist())


def main():
    report={}
    for study in ("string","order"):
        rows=json.loads((ROOT/(study+"-results.json")).read_text())["lifetimes"]
        summary={}
        report[study]=summary
        with np.load(ROOT/(study+"-trials.npz")) as a:
            for g in ("glob","count"):
                accuracy={}; trials={}
                for key in rows[0][g]:
                    if key in ("feedback","mix"):continue
                    vals=np.stack([a[f"{r['seed']}-{g}-{key}"] for r in rows])
                    trials[key]=vals
                    assert all(np.allclose(z.mean(0),r[g][key]) for z,r in zip(vals,rows))
                    per_program=vals.reshape(len(rows),160 if g=="glob" else 12,-1,3).mean((0,2,3))
                    accuracy[key]={"mean":float(vals.mean()),"curves":vals.mean((0,1)).tolist(),
                                   "program_quartiles":np.quantile(per_program,[.25,.5,.75]).tolist()}
                contrasts=([("episodic-100","initial"),("adaptive","winner"),("adaptive","fixed")]
                           if study=="string" else [("count-last-200","position-last-200"),
                               ("interleaved-200","winner"),("count-last-200","initial"),
                               ("position-last-200","initial")])
                for key in list(trials):
                    if "+matched-" in key:
                        contrasts.append((key,key.split("+")[0]+"-200"))
                if "winner" not in trials:
                    trials["winner"]=np.stack([trials[r["winner"]][j] for j,r in enumerate(rows)])
                    accuracy["winner"]={"mean":float(trials["winner"].mean())}
                result={"accuracy":accuracy,"contrasts":{},"feedback":{}}
                for l,r in contrasts:
                    result["contrasts"][l+" minus "+r]=paired((trials[l]-trials[r]).mean((1,2)))
                for env in ("stable","reversed"):
                    for k in (0,2,8,32):
                        for name in (("recent","blind") if study=="string" else ("recent","blind","discounted")):
                            prefix=f"{env}-{k}-"
                            def get(row):
                                return row[g]["feedback"] if study=="string" else row["feedback"][g]
                            delta=np.array([np.subtract(get(r)[prefix+name],get(r)[prefix+"frozen"]) for r in rows])
                            result["feedback"][prefix+name+" minus frozen"]={
                                "harm":paired(delta[:,0]),"repair":paired(delta[:,1])}
                summary[g]=result
    (ROOT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for study,groups in report.items():
        for group,r in groups.items():
            print(study,group,{k:round(100*v["mean"],1) for k,v in r["accuracy"].items()})
            for key,stat in r["contrasts"].items():
                print(key,[round(v*100,2) for v in (stat["mean"],*stat["interval"])])
            for k in (0,2,8,32):
                z=r["feedback"][f"reversed-{k}-recent minus frozen"]
                print("after reversal",k,[[round(x*100,2) for x in (v["mean"],*v["interval"])] for v in z.values()])


if __name__=="__main__":
    main()
