"""Text-only baselines on Liars' Bench (CPU). Needs access to the gated dataset and a Hugging Face login.

    uvx --with huggingface_hub --with pandas --with pyarrow --with scikit-learn python textbase.py

Writes aggregate scores to results/liars-bench-text.json. No transcript text is saved.
"""
import hashlib
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).parent / "results" / "liars-bench-text.json"
SUBSETS = ["instructed-deception", "convincing-game", "insider-trading", "harm-pressure-choice",
           "harm-pressure-knowledge-report", "gender-secret", "soft-trigger"]
VIEWS = ["prompt only", "response only", "transcript", "length only"]
PAPER_MODELS = {"llama", "mistral", "gemma", "qwen2.5"}  # the four models in the paper


def base(model):
    m = model.lower()
    return next(k for k, pat in (("llama", "llama"), ("gemma", "gemma"), ("kimi", "kimi"), ("mistral", "mistral"),
                                 ("qwen2.5", "qwen-2.5"), ("qwen3.5", "qwen3")) if pat in m)


def load(subset, split="test"):
    d = pd.read_parquet(hf_hub_download("Cadenza-Labs/liars-bench", f"{subset}/{split}-00000-of-00001.parquet", repo_type="dataset"))
    msgs = d["messages"].tolist()
    assert all(m[-1]["role"] == "assistant" for m in msgs), subset
    d["response"] = [m[-1]["content"] or "" for m in msgs]
    d["prompt"] = ["\n".join(f"[{x['role']}] {x['content']}" for x in m[:-1]) for m in msgs]
    d["group"] = [hashlib.md5(next((x["content"] for x in m if x["role"] == "user"), "").encode()).hexdigest() for m in msgs]
    d["y"] = d["deceptive"].astype(int)
    return d[["model", "prompt", "response", "group", "y"]]


def featurise(view, train, others):
    """Fit on train, transform train and each frame in others."""
    if view == "length only":
        f = lambda d: csr_matrix(np.log1p(np.c_[d["response"].str.len(), d["prompt"].str.len()]))
        return f(train), [f(o) for o in others]
    text = {"prompt only": lambda d: d["prompt"], "response only": lambda d: d["response"],
            "transcript": lambda d: d["prompt"] + "\n[assistant] " + d["response"]}[view]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    return v.fit_transform(text(train)), [v.transform(text(o)) for o in others]


def fit_score(view, train, test, control):
    """Train on train plus half of the control conversations (all honest); threshold at 1% false positives on the other half."""
    half = len(control) // 2
    tr = pd.concat([train, control.iloc[:half]])
    Xtr, (Xte, Xc) = featurise(view, tr, [test, control.iloc[half:]])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr["y"])
    s, thr = clf.decision_function(Xte), np.quantile(clf.decision_function(Xc), 0.99)
    y = test["y"].to_numpy()
    return roc_auc_score(y, s), 0.5 * ((s[y == 1] > thr).mean() + (s[y == 0] <= thr).mean())


def main():
    alpaca = load("alpaca").assign(y=0)
    control = {b: g.sample(frac=1, random_state=0) for b, g in alpaca.groupby(alpaca["model"].map(base))}
    data = {s: load(s) for s in SUBSETS}
    res = {"within": {}, "transfer": {}}
    for s, d in data.items():
        for model, g in d.groupby("model"):
            if min(g["y"].sum(), (1 - g["y"]).sum()) < 30:
                continue
            g, b = g.reset_index(drop=True), base(model)
            grouped = g["group"].nunique() >= 20
            folds = list((GroupKFold(5).split(g, g["y"], g["group"]) if grouped else StratifiedKFold(5, shuffle=True, random_state=0).split(g, g["y"])))
            for view in VIEWS:
                sc = [fit_score(view, g.iloc[a], g.iloc[t], control[b]) for a, t in folds if g.iloc[t]["y"].nunique() == 2]
                res["within"].setdefault(s, {}).setdefault(view, {})[model] = [float(np.mean([x[0] for x in sc])), float(np.mean([x[1] for x in sc]))]
            res["within"][s].setdefault("folds kept whole first-user-messages together", {})[model] = bool(grouped)
            src = data["instructed-deception"]
            src = src[src["model"].map(base) == b]
            if s != "instructed-deception" and len(src):
                for view in VIEWS:
                    res["transfer"].setdefault(s, {}).setdefault(view, {})[model] = list(map(float, fit_score(view, src, g, control[b])))
            print(s, model, {v: [round(x, 2) for x in res["within"][s][v][model]] for v in VIEWS}, flush=True)
    summary = {}
    for kind in ("within", "transfer"):
        for s, views in res[kind].items():
            for view in VIEWS:
                per = views[view]
                paper = [v for m, v in per.items() if base(m) in PAPER_MODELS]
                summary.setdefault(kind, {}).setdefault(s, {})[view] = {
                    "auroc_mean_all_models": round(float(np.mean([v[0] for v in per.values()])), 3),
                    "balanced_accuracy_mean_all_models": round(float(np.mean([v[1] for v in per.values()])), 3),
                    "auroc_mean_paper_models": round(float(np.mean([v[0] for v in paper])), 3) if paper else None,
                    "balanced_accuracy_mean_paper_models": round(float(np.mean([v[1] for v in paper])), 3) if paper else None,
                    "models": len(per)}
    OUT.write_text(json.dumps({"summary": summary, "per_model": res}, indent=1))
    for kind in ("within", "transfer"):
        print(f"\n{kind}: AUROC / balanced accuracy at 1% false positives on control, mean over models")
        for s, views in summary[kind].items():
            print(f"  {s:<32}" + "  ".join(f"{v[:10]}: {views[v]['auroc_mean_all_models']:.2f}/{views[v]['balanced_accuracy_mean_all_models']:.2f}" for v in VIEWS))




def checks():
    """Two post hoc checks. (1) Stricter folds: transcripts sharing their five rarest prompt words stay together, so the
    same question under a different template cannot sit on both sides. (2) One classifier per model trained across the
    five subsets the paper's upper-bound probe used, plus control, scored on held-out transcripts of each."""
    import collections
    import re
    alpaca = load("alpaca").assign(y=0)
    control = {b: g.sample(frac=1, random_state=0) for b, g in alpaca.groupby(alpaca["model"].map(base))}
    out = {"strict_folds": {}, "one_classifier_five_subsets": {}}
    data = {}
    for s in SUBSETS:
        d = load(s)
        first_user = pd.read_parquet(hf_hub_download("Cadenza-Labs/liars-bench", f"{s}/test-00000-of-00001.parquet", repo_type="dataset"))["messages"].map(
            lambda m: next((x["content"] for x in m if x["role"] == "user"), ""))
        toks = [set(re.findall(r"[a-z0-9]{3,}", t.lower())) for t in first_user]
        df = collections.Counter(w for t in toks for w in t)
        d["group"] = [" ".join(sorted(sorted(t, key=lambda w: (df[w], w))[:5])) for t in toks]
        data[s] = d
        for view in ("prompt only", "transcript"):
            per = []
            for model, g in d.groupby("model"):
                if min(g["y"].sum(), (1 - g["y"]).sum()) < 30 or g["group"].nunique() < 20:
                    continue
                g = g.reset_index(drop=True)
                per.append(np.mean([fit_score(view, g.iloc[a], g.iloc[t], control[base(model)])[0]
                                    for a, t in GroupKFold(5).split(g, g["y"], g["group"]) if g.iloc[t]["y"].nunique() == 2]))
            out["strict_folds"].setdefault(s, {})[view] = [round(float(np.mean(per)), 3), len(per)] if per else None
        out["strict_folds"][s]["distinct_groups"] = int(d["group"].nunique())
        print("strict folds", s, out["strict_folds"][s], flush=True)
    five = ["harm-pressure-choice", "harm-pressure-knowledge-report", "convincing-game", "instructed-deception", "insider-trading"]
    pooled = pd.concat([data[s].assign(subset=s) for s in five])
    pooled["b"] = pooled["model"].map(base)
    for view in ("transcript", "response only", "prompt only"):
        scores = collections.defaultdict(list)
        for b, g in pooled.groupby("b"):
            g = g.reset_index(drop=True)
            for a, t in GroupKFold(5).split(g, g["y"], g["subset"] + g["group"]):
                tr = pd.concat([g.iloc[a], control[b].iloc[:len(control[b]) // 2]])
                Xtr, (Xte,) = featurise(view, tr, [g.iloc[t]])
                s_ = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr["y"]).decision_function(Xte)
                te = g.iloc[t].assign(score=s_)
                for sub, h in te.groupby("subset"):
                    if h["y"].nunique() == 2:
                        scores[sub, b].append(roc_auc_score(h["y"], h["score"]))
        per_subset = {sub: round(float(np.mean([np.mean(v) for (s2, _), v in scores.items() if s2 == sub])), 3) for sub in five}
        out["one_classifier_five_subsets"][view] = {**per_subset, "mean over the five": round(float(np.mean(list(per_subset.values()))), 3)}
        print("one classifier", view, out["one_classifier_five_subsets"][view], flush=True)
    (OUT.parent / "liars-bench-text-checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    checks() if "--checks" in __import__("sys").argv else main()
