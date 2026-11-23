"""Probe on public activations against outside baselines, on the same held-out questions (CPU, laptop).

    modal volume get inside-advantage / pooled/      # once, after pool.py
    uvx --with huggingface_hub --with pandas --with pyarrow --with scikit-learn python probe.py

Writes aggregate scores to results/probe-vs-text.json. No transcript text is saved.
"""
import collections
import json
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent
SUBSETS = ["harm-pressure-choice", "harm-pressure-knowledge-report", "instructed-deception", "convincing-game",
           "gender-secret", "insider-trading", "soft-trigger"]
POOLINGS = ["probe: mean of prompt tokens", "probe: last prompt token", "probe: mean of response tokens", "probe: last token"]
TEXT = ["text: prompt only", "text: response only", "text: transcript"]
C_PROBE = 0.01  # fixed before any scores; 8,192 standardised features and a few thousand transcripts


def load(subset):
    """Llama 3.3 70B transcripts of one subset, aligned with their pooled activations, with strict question groups."""
    z = np.load(next((HERE / "pooled").rglob(f"pooled-{subset}.npz")))
    d = pd.read_parquet(hf_hub_download("Cadenza-Labs/liars-bench", f"{subset}/test-00000-of-00001.parquet", repo_type="dataset")).iloc[z["row_idx"]].reset_index(drop=True)
    assert (d["deceptive"].to_numpy() == z["deceptive"]).all() and d["model"].str.contains("llama").all(), subset
    msgs = d["messages"].tolist()
    d["response"] = [m[-1]["content"] or "" for m in msgs]
    d["prompt"] = ["\n".join(f"[{x['role']}] {x['content']}" for x in m[:-1]) for m in msgs]
    toks = [set(re.findall(r"[a-z0-9]{3,}", next((x["content"] for x in m if x["role"] == "user"), "").lower())) for m in msgs]
    df = collections.Counter(w for t in toks for w in t)
    d["group"] = [" ".join(sorted(sorted(t, key=lambda w: (df[w], w))[:5])) for t in toks]
    d["y"] = d["deceptive"].astype(int)
    return d, z["x"].astype(np.float32)


def meta_features(d):
    if "meta" not in d.columns:
        return None
    num = re.compile(r"'(\w+)': (?:np\.\w+\()?(-?\d+(?:\.\d+)?(?:e-?\d+)?|True|False)\)?")  # numeric and yes/no fields only
    rows = [{k: float({"True": 1, "False": 0}.get(v, v)) for k, v in num.findall(str(m))} for m in d["meta"]]
    f = pd.DataFrame(rows).fillna(0.0)
    f = f.loc[:, f.nunique() > 1]
    return f.to_numpy() if f.shape[1] else None


def views(d, X):
    """name -> (fit_transform on train rows, transform on test rows) builders, as functions of row index arrays."""
    out = {}
    for j, name in enumerate(POOLINGS):
        out[name] = ("dense", X[:, j], C_PROBE)
    text = {"text: prompt only": d["prompt"], "text: response only": d["response"], "text: transcript": d["prompt"] + "\n[assistant] " + d["response"]}
    for name, col in text.items():
        out[name] = ("text", col, 1.0)
    out["length only"] = ("dense", np.log1p(np.c_[d["response"].str.len(), d["prompt"].str.len()]), 1.0)
    m = meta_features(d)
    if m is not None:
        out["benchmark metadata"] = ("dense", m, 1.0)
    return out


def fit_predict(kind, data, C, tr, te, y):
    if kind == "text":
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")  # one-letter answers count as words
        A, B = v.fit_transform(data.iloc[tr]), v.transform(data.iloc[te])
    else:
        s = StandardScaler().fit(data[tr])
        A, B = s.transform(data[tr]), s.transform(data[te])
    return LogisticRegression(C=C, max_iter=1000, class_weight="balanced").fit(A, y[tr]).decision_function(B)


def boot(y, a, b, groups, n=2000, seed=0):
    """95% interval for AUROC(a) - AUROC(b), resampling question groups."""
    rng, ids = np.random.default_rng(seed), collections.defaultdict(list)
    for i, g in enumerate(groups):
        ids[g].append(i)
    keys, vals = list(ids), []
    while len(vals) < n:
        idx = np.concatenate([ids[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    res, cache = {}, {}
    for s in SUBSETS:
        d, X = load(s)
        cache[s] = (d, X)
        y, strict = d["y"].to_numpy(), d["group"].nunique() >= 20
        folds = list(GroupKFold(5).split(d, y, d["group"]) if strict else StratifiedKFold(5, shuffle=True, random_state=0).split(d, y))
        oof = {}
        for name, (kind, data, C) in views(d, X).items():
            sc = np.zeros(len(d))
            for tr, te in folds:
                out = fit_predict(kind, data, C, tr, te, y)
                sc[te] = (out - out.mean()) / (out.std() + 1e-9)  # put folds on one scale before pooling them
            oof[name] = sc
        auc = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}
        best_in, best_out = max(POOLINGS, key=auc.get), max((k for k in auc if k not in POOLINGS), key=auc.get)
        groups = d["group"].to_numpy() if strict else np.arange(len(d)).astype(str)
        res[s] = {"n_deceptive": int(y.sum()), "n_honest": int((1 - y).sum()), "question_groups": int(d["group"].nunique()), "strict_folds": bool(strict),
                  "auroc": {k: round(v, 3) for k, v in auc.items()}, "best_probe": best_in, "best_outside": best_out,
                  "best_probe_minus_best_outside": [round(v, 3) for v in boot(y, oof[best_in], oof[best_out], groups)],
                  "response_mean_probe_minus_transcript_text": [round(v, 3) for v in boot(y, oof[POOLINGS[2]], oof["text: transcript"], groups)]}
        print(s, json.dumps(res[s]), flush=True)
    # transfer: fit on instructed deception, score every other subset
    (ds, Xs), ys = cache["instructed-deception"], cache["instructed-deception"][0]["y"].to_numpy()
    res["transfer_from_instructed_deception"] = {}
    for s in SUBSETS[:2] + SUBSETS[3:]:
        d, X = cache[s]
        y, out = d["y"].to_numpy(), {}
        for j, name in enumerate(POOLINGS):
            sc = StandardScaler().fit(Xs[:, j])
            clf = LogisticRegression(C=C_PROBE, max_iter=1000, class_weight="balanced").fit(sc.transform(Xs[:, j]), ys)
            out[name] = round(float(roc_auc_score(y, clf.decision_function(sc.transform(X[:, j])))), 3)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")  # one-letter answers count as words
        tx = lambda f: f["prompt"] + "\n[assistant] " + f["response"]
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(v.fit_transform(tx(ds)), ys)
        out["text: transcript"] = round(float(roc_auc_score(y, clf.decision_function(v.transform(tx(d))))), 3)
        res["transfer_from_instructed_deception"][s] = out
        print("transfer", s, out, flush=True)
    (HERE / "results" / "probe-vs-text.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    d0 = boot(y, y.astype(float), y.astype(float), np.arange(8).astype(str), n=50)
    assert d0[0] == 0.0 and d0[1] == 0.0
    main()
