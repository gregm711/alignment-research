"""Checks added after the independent review (post hoc, harm-pressure choice, CPU, laptop).

    uvx --with huggingface_hub --with pandas --with pyarrow --with scikit-learn==1.9.1 --with numpy==2.5.5 python review_checks.py

1. Stronger outside monitors: the benchmark authors' published judge scores, the answer key, and both together.
2. Does the probe add anything to the answer key plus judge? Nested folds, so neither stage sees its test questions.
3. Alarm counts at a threshold set for 1% false alarms on separate training questions.
4. Same-question pairs with opposite labels: how often does each monitor rank the pair correctly?

The judge file holds transcript text; it is downloaded to pooled/ (not in git). Only aggregates are written.
"""
import json
import pathlib
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import probe

HERE = pathlib.Path(__file__).parent
JUDGE = "https://raw.githubusercontent.com/Cadenza-Labs/liars-bench/main/results/blackbox/ext_judge_scoring_lie/harm-pressure-choice/llama-v3.3-70b-instruct.csv"
RESP, TEXT = "probe: mean of response tokens", "text: transcript"

d, X = probe.load("harm-pressure-choice")
y, groups = d["y"].to_numpy(), d["group"].to_numpy()
path = HERE / "pooled" / "judge-harm-pressure-choice.csv"
if not path.exists():
    urllib.request.urlretrieve(JUDGE, path)
j = pd.read_csv(path).set_index("index").loc[d["index"].astype(int)]
assert (j["deceptive"].to_numpy() == d["deceptive"].to_numpy()).all() and len(j) == len(d) == 2134
judge = j["prediction_extracted"].to_numpy(float)
chosen = d["response"].str.strip().str.upper().str.extract(r"([A-D])")[0].map({"A": 0, "B": 1, "C": 2, "D": 3})
wrong = (chosen != probe.meta_table(d)["correct_option"]).to_numpy()
key = wrong.astype(float)
folds = list(GroupKFold(5).split(d, y, groups))
views = probe.views(d, X)
norm = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # probe.py's fold normalisation


def oof(name):
    kind, data, C = views[name]
    sc = np.zeros(len(d))
    for tr, te in folds:
        sc[te] = norm(probe.fit_predict(kind, data, C, tr, te, y))
    return sc


def second_stage(F_tr, y_tr, F_te):
    s = StandardScaler().fit(F_tr)
    return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(s.transform(F_tr), y_tr).decision_function(s.transform(F_te))


p_resp, p_prompt, text = oof(RESP), oof("probe: last prompt token"), oof(TEXT)
key_text = key + rankdata(text) / (len(text) + 1)  # wrong answers first, text score breaks ties
key_judge, key_judge_probe = np.zeros(len(d)), np.zeros(len(d))
caught = flagged = 0
kind, data, C = views[RESP]
for tr, te in folds:
    key_judge[te] = norm(second_stage(np.c_[key[tr], judge[tr]], y[tr], np.c_[key[te], judge[te]]))
    inner = np.zeros(len(tr))  # probe scores for the training rows, each from a probe that did not see its question
    for a, b in GroupKFold(4).split(tr, y[tr], groups[tr]):
        inner[b] = probe.fit_predict(kind, data, C, tr[a], tr[b], y)
    outer = probe.fit_predict(kind, data, C, tr, te, y)
    key_judge_probe[te] = norm(second_stage(np.c_[key[tr], judge[tr], inner], y[tr], np.c_[key[te], judge[te], outer]))
    # alarm: fit on four fifths of the training questions, set the threshold on the honest rows of the other fifth
    a, b = next(GroupKFold(5).split(tr, y[tr], groups[tr]))
    thr = np.quantile(probe.fit_predict(kind, data, C, tr[a], tr[b], y)[y[tr[b]] == 0], 0.99)
    hit = probe.fit_predict(kind, data, C, tr[a], te, y) > thr
    caught, flagged = caught + int(hit[y[te] == 1].sum()), flagged + int(hit[y[te] == 0].sum())

monitors = {"text: transcript": text, "published external judge": judge, "answer key": key, "answer key, text breaks ties": key_text,
            "answer key + judge": key_judge, "probe: mean of response tokens": p_resp, "answer key + judge + probe": key_judge_probe}
r3 = lambda v: [round(float(x), 3) for x in v]
out = {"auroc": {k: round(float(roc_auc_score(y, v)), 3) for k, v in monitors.items()},
       "probe_minus": {k: r3(probe.boot(y, p_resp, monitors[k], groups)) for k in list(monitors)[:5]},
       "adding_probe_to_key_and_judge": r3(probe.boot(y, key_judge_probe, key_judge, groups)),
       "wrong_answers_only": {k: round(float(roc_auc_score(y[wrong], v[wrong])), 3) for k, v in (("probe", p_resp), ("text", text), ("judge", judge))},
       "wrong_answers_only_probe_minus_judge": r3(probe.boot(y[wrong], p_resp[wrong], judge[wrong], groups[wrong])),
       "alarm_at_1pct_target": {"caught": f"{caught}/{int(y.sum())}", "false_alarms": f"{flagged}/{int((1 - y).sum())}"}}

# same-question pairs with opposite labels
pairs = [(i, k) for g in pd.unique(groups) for i in np.flatnonzero((groups == g) & (y == 1)) for k in np.flatnonzero((groups == g) & (y == 0))]
both_wrong = [(i, k) for i, k in pairs if wrong[i] and wrong[k]]
count = lambda sc, ps: f"{sum(sc[i] > sc[k] for i, k in ps)}/{len(ps)}"
out["same_question_pairs"] = {"all": {"probe": count(p_resp, pairs), "text": count(text, pairs), "judge": count(judge, pairs)},
                              "both_answers_wrong": {"probe": count(p_resp, both_wrong), "text": count(text, both_wrong)}}
print(json.dumps(out, indent=1))
(HERE / "results" / "review-checks.json").write_text(json.dumps(out, indent=1))
