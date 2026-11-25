"""Audit genuine novelty by exhaustively comparing all 64 inputs; deduplicate per lifetime."""
import itertools
import json

import revising_updates as revision
from recognizing_change import ROOT, pilot, np, tasks, GROUPS
from analyse import estimate

SIGNS = list(itertools.product((-1, 1),repeat=3))
ALL_TRIPLES = np.array(list(itertools.combinations(range(12),3)))


def fingerprints(labels):
    return [bytes(row) for row in np.packbits(labels,axis=1)]


def majority_space(triples):
    space = set()
    for signs in SIGNS:
        space.update(fingerprints(((pilot.TRUTH[:,triples] * signs).sum(2)>0).T))
    return space


def aggregate():
    all_majorities = majority_space(ALL_TRIPLES)
    report = {}
    for cohort in ("change","revision"):
        early = cohort=="change"
        size = 192 if early else 160
        seeds = range(241,249) if early else revision.SEEDS
        groups = GROUPS if early else revision.GROUPS
        results = {g:[] for g in groups}
        with np.load(ROOT/(cohort+"-trials.npz")) as a:
            for seed in seeds:
                rng = np.random.default_rng(seed+(9000 if early else 11500))
                old,new = rng.permutation(pilot.TRIPLES),rng.permutation(pilot.SHIFT)
                exposed = majority_space(old[:16]) | majority_space(new[:10])
                if early:
                    y = a[f"{seed}-labels"]
                else:
                    mixed = np.array([t for t in itertools.combinations(range(12),3)
                                      if 0<np.isin(t,pilot.COMMON).sum()<3])
                    tasks(rng,[old[12:16],new[:10]],64)
                    y,order = tasks(rng,[old[16:],new[10:],old[16:],new[10:],mixed],size)
                    # Confirm reconstruction against the recorded zero-shot curve.
                    # With zero logits, the evaluator predicts True for every input.
                    start = np.take_along_axis(y, order[:,18:], axis=1).mean(1)
                    assert np.array_equal(start,a[f"{seed}-flat"][:,0])
                for i,g in enumerate(groups):
                    sl=slice(i*size,(i+1)*size)
                    keys=fingerprints(y[sl])
                    disallowed = exposed if i<2 else all_majorities
                    distinct, inv, freq = np.unique(np.array(keys,dtype='V8'),return_inverse=True,return_counts=True)
                    keep=np.array([k not in disallowed for k in keys])
                    weight=keep.astype(float)/freq[inv]
                    assert weight.sum()>0
                    w=weight/weight.sum()
                    row={"seed":seed,"unique_drawn":len(distinct),"exposure_overlaps":sum(k in exposed for k in keys),
                         "retained_draws":int(keep.sum()),"retained_rules":int(round(weight.sum()))}
                    names = (("adaptive","winner","uniform","shuffled") if early else
                             ("retroactive","winner","matched-retro","prospective","shuffled"))
                    row["accuracy"]={k:float(w @ a[f"{seed}-{k}"][sl,1:].mean(1)) for k in names}
                    if not early:
                        pre=a[f"{seed}-advice-pre"][sl]
                        for j,ep in enumerate(("harm","repair")):
                            eligible=pre if j==0 else ~pre
                            f=(.1,.45) if j==0 else (.9,.55)
                            diffs=[]
                            for name in ("calibrated","common"):
                                out=a[f"{seed}-advice-{name}"][sl,:,j]
                                diffs.append(np.average(out[eligible]!=pre[eligible,None],axis=0,weights=w[eligible]))
                            delta=np.subtract(*diffs)
                            row[ep]=[float(np.average(delta,weights=f)),float(np.average(delta,weights=f[::-1]))]
                    results[g].append(row)
        for g,rows in results.items():
            def difference(left,right):
                return estimate([row["accuracy"][left]-row["accuracy"][right] for row in rows])
            primary = "adaptive" if early else "retroactive"
            entry={"rows":rows,"vs_winner":difference(primary,"winner"),
                   "vs_shuffled":difference(primary,"shuffled")}
            if not early:
                entry["matched_vs_prospective"]=difference("matched-retro","prospective")
                for ep in ("harm","repair"):
                    entry[ep]={k:estimate([row[ep][j] for row in rows])
                               for j,k in enumerate(("stable","reversed"))}
            report[cohort+"-"+g]=entry
    (ROOT/"change-task-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    for cohort in ("change","revision"):
        key=cohort+"-new-conditional"
        entry=report[key]
        print(key,'unique retained per seed',[r["retained_rules"] for r in entry["rows"]])
        for name in ("vs_winner","vs_shuffled","matched_vs_prospective"):
            if name in entry:
                print(name,entry[name]["mean"],entry[name]["ci"])
        if "harm" in entry:
            for ep in ("harm","repair"):
                for condition in ("stable","reversed"):
                    z=entry[ep][condition]
                    print(ep,condition,z["mean"],z["ci"])


if __name__=="__main__":
    aggregate()
