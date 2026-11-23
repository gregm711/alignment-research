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


if __name__ == "__main__":
    main()
