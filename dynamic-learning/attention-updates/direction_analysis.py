"""Aggregate directional interventions by training lifetime."""
import json

from direction import ROOT, np, p


def main():
    raw = json.loads((ROOT/"direction-trials.json").read_text())
    models = list(raw[0]["models"])
    cap = {k:np.array([r["models"][k]["capability"] for r in raw])[:,:,0].mean(-1) for k in models}
    observer = np.array([r["observer"] for r in raw]).mean(2)
    cap["observer"] = observer[:,:,1]
    cap["independent"] = observer[:,:,0]
    out = dict(capability={k:v.mean(0).tolist() for k,v in cap.items()},effects={},spill={},feedback={})
    for k in ("positive","test-only","signed"):
        for other in ("signed","positive-random","independent","observer"):
            out["effects"][k+" minus "+other] = [p.interval((cap[k]-cap[other])[:,j]) for j in range(3)]
    for rate in ("1","4"):
        info = {}
        for k in models:
            vals = [r["models"][k]["spill"][rate] for r in raw]
            agg = {field:sum(x[field] for x in vals) for field in vals[0] if field!="max_increase"}
            info[k] = dict(counts=agg, max_up=max(x["max_increase"] for x in vals),
                           new=agg["new"]/agg["was_correct"],
                           fixed=agg["fixed"]/max(1,agg["was_wrong"]),wrong=agg["after_wrong"]/agg["outside"])
        out["spill"][rate] = info
        fb = {k:np.array([r["models"][k]["feedback"][rate] for r in raw]) for k in models}
        out["feedback"][rate] = dict(means={k:v.mean(0).tolist() for k,v in fb.items()})
        for ctrl in ("signed","positive-random"):
            diff = fb["positive"]-fb[ctrl]
            out["feedback"][rate]["positive minus "+ctrl] = [
                [[[ [p.interval(diff[:,a,b,c,d,e]) for e in range(3)] for d in range(2)]
                  for c in range(2)] for b in range(2)] for a in range(2)]
    (ROOT/"direction-summary.json").write_text(json.dumps(out,indent=1)+"\n")


if __name__=="__main__":
    main()
