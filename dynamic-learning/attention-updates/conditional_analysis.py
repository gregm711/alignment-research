"""Cold-read the written weights from tokens and summarize paired effects by domain."""
import json
from conditional import ARCHIVE, METHODS, ROOT, SEEDS, Editor, Writer, np, p, torch


def main():
    torch.set_num_threads(1)
    rows = json.loads((ROOT/"conditional-trials.json").read_text())
    result = {}
    for condition in rows[0]["results"]:
        group = [r["results"][condition] for r in rows]
        values = {k:np.array([g[k]["peer"] for g in group]) for k in METHODS}
        out = dict(means={k:float(v.mean()) for k,v in values.items()},
                   contrasts={k:p.interval(values["learned"]-v) for k,v in values.items()},
                   signed={k:[p.interval(np.array([g[k]["sign"][j] for g in group])) for j in (0,1)]
                           for k in METHODS},
                   fork_accuracy={k:np.mean([g[k]["fork_accuracy"] for g in group],axis=0).tolist() for k in METHODS})
        if "eligible_counts" in group[0]:
            c = np.array([g["eligible_counts"] for g in group])
            out["feedback"] = {}
            rng = np.random.default_rng(618)
            boot = rng.integers(len(c),size=(8000,len(c)))
            for j,name in enumerate(("repairs","harm")):
                a,b,den = c[:,j].T
                ratio = lambda ix: ((a-b)[ix].sum(-1)/den[ix].sum(-1))
                out["feedback"][name] = dict(eligible=int(den.sum()),rates=[a.sum()/den.sum(),b.sum()/den.sum()],
                                             contrast=[ratio(np.arange(len(c))),*np.quantile(ratio(boot),[.025,.975])])
            by_truth = np.array([g["check_counts_by_truth"] for g in group])
            out["check_error"] = {}
            for err in (0.,.1):
                vals=(1-err)*by_truth[:,:,0]+err*by_truth[:,:,1]
                outcome={}
                for j,name in enumerate(("repair_after_false_report","damage_after_true_report")):
                    a,b,den=vals[:,j].T
                    delta=a-b
                    interval=np.quantile(delta[boot].sum(1)/den[boot].sum(1),[.025,.975])
                    outcome[name]=dict(rates=[a.sum()/den.sum(),b.sum()/den.sum()],
                                       difference=[delta.sum()/den.sum(),*interval])
                out["check_error"][str(err)] = outcome
        result[condition]=out
    with np.load(ARCHIVE) as a,torch.no_grad():
        errors=[]
        for seed in SEEDS:
            prefix=f"{seed}/language/"
            st={k[len(prefix):]:torch.tensor(a[k].tolist()) for k in a.files if k.startswith(prefix)}
            lang=Editor(len(st["word.weight"]))
            lang.load_state_dict(st)
            tokens=torch.tensor(a[f"{seed}/tokens"].tolist())
            raw=lang.inputs(tokens)
            reader=torch.tensor(a[f"{seed}/reader"].tolist())
            h=torch.nn.functional.normalize(torch.einsum("qhd,hdk->qhk",raw,reader),dim=-1)
            for mode in (0,1):
                key=f"{seed}-{mode}"
                v=torch.tensor(a[key+"-v"].tolist()); out=torch.tensor(a[key+"-out"].tolist())
                pred=(torch.einsum("qhd,hdk->qhk",h,v).tanh()*out).sum((-1,-2)).sigmoid()
                err=np.max(np.abs(pred.tolist()-a[key+"-p"]))
                assert err<2e-6 and np.array_equal(np.array(pred.tolist())>=.5,a[key+"-p"]>=.5)
                errors.append(float(err))
    result["cold_reload"] = dict(cases=len(errors),max_error=max(errors))
    (ROOT/"conditional-summary.json").write_text(json.dumps(result,indent=1)+"\n")


if __name__=="__main__":
    main()
