"""Pooled, paired comparison of the attention classifier against output-only baselines on hidden cases (offline).

    python3 pooled.py        # reads results/phase3_scores-*-clean.json (written by monitor.py), writes results/pooled.json

All three models answer the same questions, so cases are not independent across models, and near-duplicate
questions share an answer. Each bootstrap draw resamples answer clusters once, applies that draw to every
model, takes the AUC difference inside each model, and averages the three differences.
"""
import json
import pathlib
import random
import sys

from monitor import auc
from scoring import words

HERE = pathlib.Path(__file__).parent / "results"
TAGS = {"Phi-4-mini": "", "Qwen2.5-7B": "-Qwen2.5-7B-Instruct", "granite-3.3-2b": "-granite-3.3-2b-instruct"}
FILLER = ("the", "a", "an", "according", "in", "it", "there", "based")  # same list as audit.py
CLF, COMPARE = "attention classifier", ("output top probability", "output plus length baseline")
DRAWS = 5000
SET = "new" if "new" in sys.argv else "clean"  # python3 pooled.py new: Phase 5 questions, writes results/pooled-new.json


def delta(rows, weights, other):
    pos = [(r[CLF], r[other]) for r in rows for _ in range(weights.get(r["cluster"], 0)) if r["hidden"]]
    neg = [(r[CLF], r[other]) for r in rows for _ in range(weights.get(r["cluster"], 0)) if not r["hidden"]]
    if not pos or not neg:
        return None
    return auc([p[0] for p in pos], [n[0] for n in neg]) - auc([p[1] for p in pos], [n[1] for n in neg])


def main():
    data = {m: json.loads((HERE / f"phase3_scores{t}-{SET}.json").read_text()) for m, t in TAGS.items()}
    for rows in data.values():
        for r in rows:
            r["cluster"] = " ".join(words(r["gold0"]))
            r["filler"] = (r["answer"].split() or [""])[0].lower().strip(",.") in FILLER
    subsets = {"all": lambda r: True, "no filler-word starts": lambda r: not r["filler"], "known closed-book only": lambda r: r["known"]}
    if SET == "new":  # every Phase 5 question is known; split by kind of question instead
        subsets = {"all": lambda r: True, "no filler-word starts": lambda r: not r["filler"], "who questions only": lambda r: r["kind"] == "who"}
    out = {}
    for sname, keep in subsets.items():
        sub = {m: [r for r in rows if keep(r)] for m, rows in data.items()}
        clusters = sorted({r["cluster"] for rows in sub.values() for r in rows})
        ones = dict.fromkeys(clusters, 1)
        out[sname] = {"n_hidden_vs_grounded": {m: [sum(r["hidden"] for r in rows), sum(not r["hidden"] for r in rows)] for m, rows in sub.items()}}
        for other in COMPARE:
            rng, draws, skipped = random.Random(0), [], 0
            while len(draws) < DRAWS:
                w = {}
                for c in rng.choices(clusters, k=len(clusters)):
                    w[c] = w.get(c, 0) + 1
                d = [delta(rows, w, other) for rows in sub.values()]
                if None in d:  # a draw with no hidden case for some model is redrawn
                    skipped += 1
                    continue
                draws.append(d)
            ci = lambda v: [sorted(v)[int(0.025 * DRAWS)], sorted(v)[int(0.975 * DRAWS)]]
            res = {m: [delta(rows, ones, other), *ci([d[i] for d in draws])] for i, (m, rows) in enumerate(sub.items())}
            res["pooled mean"] = [sum(v[0] for v in res.values()) / 3, *ci([sum(d) / 3 for d in draws])]
            res["redrawn"] = skipped
            out[sname][f"classifier minus {other}"] = res
            print(f"\n{sname}: classifier minus {other}   n = {out[sname]['n_hidden_vs_grounded']}")
            for k, v in res.items():
                if k != "redrawn":
                    print(f"  {k:<16} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    (HERE / ("pooled-new.json" if SET == "new" else "pooled.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    flat = [{"cluster": "a", "hidden": True, "x": 2.0, "y": 0.0}, {"cluster": "b", "hidden": False, "x": 1.0, "y": 1.0}]
    CLF_, CLF = CLF, "x"
    assert delta(flat, {"a": 1, "b": 2}, "y") == 1.0 and delta(flat, {"b": 3}, "y") is None
    CLF = CLF_
    main()
