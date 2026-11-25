"""Summarize prospective accuracy, calibration, and corruption sensitivity by transition."""
import json

from learn import ROOT, np


def summarize(records,archive):
    out={}
    for block in ("early","late"):
        vals={k:np.stack([archive[f"{r['congress']}-{block}-{k}"].mean(0) for r in records])
              for k in records[0]["blocks"][block]["curves"]}
        vals["development-choice"]=np.stack([vals[r["reference"]][i] for i,r in enumerate(records)])
        effects={}
        for k in vals:
            v=(vals["recent"]-vals[k]).mean(1)
            effects[k]=dict(mean=v.mean(0).tolist(),by_transition=v.tolist(),
                            accuracy_curve=(vals["recent"]-vals[k])[:,:,0].mean(0).tolist())
        comparisons={}
        for k in ("frozen","fixed-recent"):
            counts=np.array([r["blocks"][block]["feedback"][k] for r in records])
            f={}
            for j,label in enumerate(("repair","harm")):
                a,b,den=counts[:,j].T
                rng=np.random.default_rng(619)
                boot=rng.integers(len(a),size=(8000,len(a)))
                draw=(a-b)[boot].sum(1)/den[boot].sum(1)
                f[label]=dict(eligible=int(den.sum()),rates=[a.sum()/den.sum(),b.sum()/den.sum()],
                              difference=(a-b).sum()/den.sum(),
                              transition_resampling_range=np.quantile(draw,[.025,.975]).tolist(),
                              by_transition=((a-b)/den).tolist())
            comparisons[k]=f
        out[block]=dict(curves={k:v.mean(0).tolist() for k,v in vals.items()},
                        recent_minus=effects,feedback=comparisons)
    return out


if __name__=="__main__":
    rec=json.loads((ROOT/"results.json").read_text())
    with np.load(ROOT/"vote-trials.npz") as saved:
        for r in rec:
            assert set(r["blocks"]["early"]["rollnumbers"]).isdisjoint(r["blocks"]["late"]["rollnumbers"])
            for block,b in r["blocks"].items():
                for k,c in b["curves"].items():
                    np.testing.assert_array_equal(saved[f"{r['congress']}-{block}-{k}"].mean(0),c)
        result=summarize(rec,saved)
    (ROOT/"summary.json").write_text(json.dumps(result,indent=1)+"\n")
