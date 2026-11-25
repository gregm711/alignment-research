"""Paired generalization, correction, and spillover comparisons for source-aware edits."""
import json

from checks import KINDS, ROOT, np, p


def main():
    rows = json.loads((ROOT/"check-trials.json").read_text())
    cap = np.array([[m["capability"] for m in r["models"]] for r in rows])[:,:,:,0]
    explicit = np.array([r["explicit"] for r in rows])
    full = np.array([[[m["results"][k] for k in KINDS] for m in r["models"]] for r in rows])
    out = dict(axes=["model","update","pairing","reverse","check_noise","truth_fork","peer_trigger_outside"],
               kinds=KINDS,means=full.mean(0).tolist(),curves=cap.mean(0).tolist(),
               explicit_curves=explicit.mean(0).tolist(),effects={},feedback={})
    for name,control in (("random",cap[:,1]),("independent",explicit[...,0]),("historical",explicit[...,1])):
        out["effects"][name] = [p.interval((cap[:,0]-control)[:,j].mean(1)) for j in range(3)]
    for j,kind in enumerate(KINDS):
        result = {}
        for g,group in enumerate(("matched","changed")):
            for v,order in enumerate(("report-first","check-first")):
                for n,noise in enumerate(("perfect","noisy")):
                    x = full[:,:,j,g,v,n]
                    result[f"{group}-{order}-{noise}"] = [[p.interval(x[:,0,f,k]-x[:,1,f,k])
                                                          for k in range(3)] for f in range(2)]
        out["feedback"][kind] = result
    (ROOT/"check-summary.json").write_text(json.dumps(out,indent=1)+"\n")


if __name__=="__main__":
    main()
