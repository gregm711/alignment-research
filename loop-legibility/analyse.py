"""Part A analysis (CPU): accuracy by pass count, and probes for each step of the chain at each site.

    modal volume get jlens-inkling-cache loop-legibility/ results/vol/
    uvx --with torch --with scikit-learn python analyse.py      # writes results/partA-summary.json

A probe is a linear classifier over the candidate words, fitted on fact sheets 0-124 and tested on 125-249.
"decoy" is the same probe design aimed at the steps of another start animal on the same sheet.
"""
import collections
import json
import pathlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent / "results"
VOL = next(HERE.rglob("states.pt")).parent
rows = json.load(open(VOL / "partA.json"))["rows"]
saved = torch.load(VOL / "states.pt", map_location="cpu")
by = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["passes"])].append(r)  # same order as the saved states


def probe(X, labels, train):
    """Held-out accuracy of a linear classifier for `labels` from X."""
    y = np.array(labels)
    s = StandardScaler().fit(X[train])
    clf = LogisticRegression(C=0.01, max_iter=300).fit(s.transform(X[train]), y[train])
    return float((clf.predict(s.transform(X[~train])) == y[~train]).mean())


out = {}
for (model, k), rs in by.items():
    S = saved[f"{model}|{k}"]["states"].float().numpy()  # [runs, sites, d]
    M = saved[f"{model}|{k}"]["messages"]
    sites = {"before the loop": S[:, 0], **{f"end of pass {j}": S[:, j] for j in range(1, k + 1)}, "last layer": S[:, -1]}
    if M is not None:  # what the final pass receives: the last 12 message calls, averaged over calls
        sites["messages into the final pass"] = M[:, -12:].float().mean(1).flatten(1).numpy()
    for c in sorted({r["chain"] for r in rs}):
        idx = np.array([i for i, r in enumerate(rs) if r["chain"] == c])
        sub = [rs[i] for i in idx]
        train = np.array([r["sheet"] < 125 for r in sub])
        cell = {"n": len(sub), "top_word_right": float(np.mean([r["top1_right"] for r in sub])),
                "best_candidate_right": float(np.mean([r["pick"] == r["steps"][-1] for r in sub])),
                # when wrong among candidates, is the pick an earlier-stopped or decoy answer? (not computable for other kinds)
                "probe": {}}
        for name, X in sites.items():
            cell["probe"][name] = {f"step {j}" + (" (answer)" if j == c else ""): {
                "queried": round(probe(X[idx], [r["steps"][j] for r in sub], train), 3),
                "decoy": round(probe(X[idx], [r["decoy_steps"][j] for r in sub], train), 3)} for j in range(1, c + 1)}
        out[f"{model} | {k} passes | chain {c}"] = cell
        print(f"\n{model} | {k} passes | chain {c}: top word right {cell['top_word_right']:.2f}, best candidate right {cell['best_candidate_right']:.2f}")
        for name, steps in cell["probe"].items():
            print(f"  {name:<28}" + "  ".join(f"{s}: {v['queried']:.2f} (decoy {v['decoy']:.2f})" for s, v in steps.items()))
(HERE / "partA-summary.json").write_text(json.dumps(out, indent=1))

# ---- paired change in accuracy from 1 pass to 3 on the variable-loop checkpoint, resampling fact sheets ----
V, rng, extra = "TML-d24-varloop-research", np.random.default_rng(0), {}
for c in (2, 3, 4):
    a, b = ([r["pick"] == r["steps"][-1] for r in by[(V, k)] if r["chain"] == c] for k in (1, 3))
    d = np.array(b, float) - np.array(a, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    extra[f"chain {c}: best-candidate accuracy, 3 passes minus 1"] = [round(float(x), 3) for x in (d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975))]
# prediction 8: chain-3 items wrong at 1 pass and right at 3; is step 1 decodable at the end of pass 1?
rs1, rs3 = ([r for r in by[(V, k)] if r["chain"] == 3] for k in (1, 3))
idx = np.array([i for i, r in enumerate(by[(V, 3)]) if r["chain"] == 3])
X = saved[f"{V}|3"]["states"].float().numpy()[idx, 1]
train = np.array([r["sheet"] < 125 for r in rs3])
y = np.array([r["steps"][1] for r in rs3])
s = StandardScaler().fit(X[train])
hit = LogisticRegression(C=0.01, max_iter=300).fit(s.transform(X[train]), y[train]).predict(s.transform(X)) == y
gained = np.array([(a["pick"] != a["steps"][-1]) and (b["pick"] == b["steps"][-1]) for a, b in zip(rs1, rs3)])
extra["chain 3, right only at 3 passes, held-out: step 1 decoded at end of pass 1"] = f"{int(hit[gained & ~train].sum())}/{int((gained & ~train).sum())}"
print(json.dumps(extra, indent=1))
(HERE / "partA-summary.json").write_text(json.dumps({"cells": out, "extra": extra}, indent=1))

# ---- added after review: on held-out chain-3 sheets where the best candidate is right at 3 passes, is step 2 decoded? ----
sub3 = [r for r in by[(V, 3)] if r["chain"] == 3]
right = np.array([r["pick"] == r["steps"][-1] for r in sub3])
y2 = np.array([r["steps"][2] for r in sub3])
for name, j in (("end of pass 3", 3), ("last layer", -1)):
    X = saved[f"{V}|3"]["states"].float().numpy()[idx, j]
    s = StandardScaler().fit(X[train])
    hit = LogisticRegression(C=0.01, max_iter=300).fit(s.transform(X[train]), y2[train]).predict(s.transform(X)) == y2
    extra[f"chain 3, 3 passes, held-out sheets with the right answer: step 2 decoded at {name}"] = f"{int(hit[right & ~train].sum())}/{int((right & ~train).sum())}"
for c in (3, 4):
    extra[f"chain {c}: top word right at 1, 2, 3 passes"] = [sum(r["top1_right"] for r in by[(V, k)] if r["chain"] == c) for k in (1, 2, 3)]
print(json.dumps(extra, indent=1))
(HERE / "partA-summary.json").write_text(json.dumps({"cells": out, "extra": extra}, indent=1))
