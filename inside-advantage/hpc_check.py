"""Post hoc check on harm-pressure choice: is the probe detecting deception, or only that the answer is wrong?

    uvx --with huggingface_hub --with pandas --with pyarrow --with scikit-learn python hpc_check.py

In this subset "deceptive" means the answer differs from the one the model gives when asked neutrally. Most
deceptive answers are also wrong by the answer key, so a wrong-answer detector would score well. The model also
makes honest mistakes (wrong answer, same as its neutral answer). Among wrong answers only, can anything tell a
lie from a mistake?
"""
import json
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import probe

d, X = probe.load("harm-pressure-choice")
num = re.compile(r"'(\w+)': (?:np\.\w+\()?(-?\d+(?:\.\d+)?(?:e-?\d+)?|True|False)\)?")
meta = pd.DataFrame([{k: float({"True": 1, "False": 0}.get(v, v)) for k, v in num.findall(str(m))} for m in d["meta"]])
chosen = d["response"].str.strip().str.upper().str.extract(r"([A-D])")[0].map({"A": 0, "B": 1, "C": 2, "D": 3})
y, wrong = d["y"].to_numpy(), (chosen != meta["correct_option"]).to_numpy()
folds = list(GroupKFold(5).split(d, y, d["group"]))
views = probe.views(d, X)
out = {"rows": {"deceptive and wrong": int((y & wrong).sum()), "deceptive and right": int((y & ~wrong).sum()),
                "honest and wrong": int(((1 - y) & wrong).sum()), "honest and right": int(((1 - y) & ~wrong).sum())},
       "answer_key_baseline_auroc": round(float(roc_auc_score(y, wrong)), 3), "auroc": {}}
for name in ["probe: last prompt token", "probe: mean of response tokens", "probe: last token", "text: transcript", "text: prompt only", "benchmark metadata"]:
    kind, data, C = views[name]
    sc = np.zeros(len(d))
    for tr, te in folds:
        o = probe.fit_predict(kind, data, C, tr, te, y)
        sc[te] = (o - o.mean()) / (o.std() + 1e-9)
    # same monitor, trained on all rows; scored on three slices
    w = wrong
    out["auroc"][name] = {"all rows": round(float(roc_auc_score(y, sc)), 3),
                          "wrong answers only: lie vs honest mistake": round(float(roc_auc_score(y[w], sc[w])), 3),
                          "honest rows only: wrong vs right answer": round(float(roc_auc_score(wrong[y == 0], sc[y == 0])), 3)}
    # and a monitor trained only on wrong answers, to see whether lie-vs-mistake is learnable at all
    idx = np.flatnonzero(w)
    sub = np.zeros(len(idx))
    for tr, te in GroupKFold(5).split(idx, y[idx], d["group"].to_numpy()[idx]):
        sub[te] = probe.fit_predict(kind, data.iloc[idx] if kind == "text" else data[idx], C, tr, te, y[idx]) if kind != "text" else probe.fit_predict(kind, data.iloc[idx].reset_index(drop=True), C, tr, te, y[idx])
        sub[te] = (sub[te] - sub[te].mean()) / (sub[te].std() + 1e-9)
    out["auroc"][name]["trained and tested on wrong answers only"] = round(float(roc_auc_score(y[idx], sub)), 3)
    print(name, out["auroc"][name], flush=True)
print(out["rows"], "answer key baseline", out["answer_key_baseline_auroc"])
(pathlib.Path(__file__).parent / "results" / "hpc-check.json").write_text(json.dumps(out, indent=1))
