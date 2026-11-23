"""Three post hoc checks suggested by the fourth outside review (offline, no GPU).

    uvx --with datasets --with modal --with torch python checks.py

1. Share of all NQ-Swap substituted passages that keep a word of an original answer.
2. The audit did two things: dropped leaky passages and re-split by answer. Each alone, for output confidence.
3. A baseline that knows only first-token confidence and the question word, fitted on the same visible
   training runs as the attention classifier, frozen, and compared with it on the Phase 5 questions.
"""
import copy
import hashlib
import json
import pathlib
import random
import shutil
import sys
import tempfile

import torch

import monitor
import pooled
from build import clean, label, questions
from scoring import leaky, words

HERE = pathlib.Path(__file__).parent / "results"
KINDS = ("who", "when", "where")


def check1():
    from datasets import load_dataset
    ds = load_dataset("pminervini/NQ-Swap", split="dev")
    n = sum(leaky(r["org_answer"], r["question"].strip(), clean(r["sub_context"])) for r in ds)
    print(f"1. substituted passages keeping a word of an original answer: {n}/{len(ds)} ({n / len(ds):.1%})")


def check2():
    items = {it["question"]: it for it in questions()}
    for name, tag in pooled.TAGS.items():
        base, res = json.loads((HERE / f"phase1{tag}.json").read_text()), {}
        for drop in (False, True):
            for resplit in (False, True):
                data = copy.deepcopy(base)
                for r in data["rows"]:
                    if drop:
                        r["subs"] = [s for s, src in zip(r["subs"], items[r["question"]]["subs"]) if not leaky(r["gold"], r["question"], src["passage"])]
                        r["label"] = label(r)
                    if resplit:
                        r["split"] = "train" if int(hashlib.md5(" ".join(words(r["gold"][0])).encode()).hexdigest(), 16) % 2 == 0 else "test"
                with tempfile.TemporaryDirectory() as d:  # run the unchanged monitor.py on this variant
                    d = pathlib.Path(d)
                    (d / "phase1.json").write_text(json.dumps(data))
                    shutil.copy(HERE / f"phase1{tag}_features.pt", d / "phase1_features.pt")
                    monitor.HERE, argv, sys.argv, out = d, sys.argv, ["monitor.py"], sys.stdout
                    sys.stdout = open("/dev/null", "w")
                    try:
                        monitor.main()
                    finally:
                        sys.stdout, sys.argv = out, argv
                    m = json.loads((d / "phase2_3.json").read_text())["phase3_hidden_on_test_questions"]
                res[(drop, resplit)] = (m["monitors"]["output top probability"]["auc"][0], m["monitors"]["attention classifier"]["auc"][0], m["n"])
        print(f"2. {name}: top probability / classifier AUC on hidden cases (n hidden, grounded)")
        for (drop, resplit), (a, c, n) in res.items():
            print(f"     drop leaky passages: {str(drop):<5} re-split by answer: {str(resplit):<5} -> {a:.3f} / {c:.3f}  {n}")
    monitor.HERE = HERE


def check3():
    kind = lambda q: [float(q.lower().split()[0] == k) for k in KINDS]
    data = {}
    for name, tag in pooled.TAGS.items():
        tr = [(s, r["question"]) for r in json.loads((HERE / f"phase1{tag}-clean.json").read_text())["rows"] if r["split"] == "train"
              for s in r["subs"] if s["outcome"] in ("memory", "passage")]
        y = torch.tensor([float(s["outcome"] == "memory") for s, _ in tr])
        mu = sum(s["p_top"] for s, _ in tr) / len(tr)
        sd = (sum((s["p_top"] - mu) ** 2 for s, _ in tr) / len(tr)) ** 0.5
        feats = lambda p, q: [(-p - -mu) / sd] + kind(q)
        X = torch.tensor([feats(s["p_top"], q) for s, q in tr])
        w, b = torch.zeros(X.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.01 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        rows = json.loads((HERE / f"phase3_scores{tag}-new.json").read_text())
        for r in rows:  # "output top probability" is stored as minus the probability
            r["question word + confidence"] = float(torch.tensor(feats(-r["output top probability"], r["question"])) @ w.detach() + b.detach())
            r["cluster"] = " ".join(words(r["gold0"]))
        data[name] = rows
        a = lambda k: monitor.auc([r[k] for r in rows if r["hidden"]], [r[k] for r in rows if not r["hidden"]])
        print(f"3. {name}: attention classifier {a('attention classifier'):.3f} | question word + confidence {a('question word + confidence'):.3f} "
              f"| weights (low confidence, who, when, where) {[round(float(v), 2) for v in w]}")
    clusters = sorted({r["cluster"] for rows in data.values() for r in rows})
    other, rng, draws = "question word + confidence", random.Random(0), []
    while len(draws) < pooled.DRAWS:
        wts = {}
        for c in rng.choices(clusters, k=len(clusters)):
            wts[c] = wts.get(c, 0) + 1
        d = [pooled.delta(rows, wts, other) for rows in data.values()]
        if None not in d:
            draws.append(sum(d) / 3)
    draws.sort()
    point = sum(pooled.delta(rows, dict.fromkeys(clusters, 1), other) for rows in data.values()) / 3
    print(f"3. pooled classifier minus question word + confidence: {point:+.3f} [{draws[int(0.025 * len(draws))]:+.3f}, {draws[int(0.975 * len(draws))]:+.3f}]")


if __name__ == "__main__":
    check1(), check2(), check3()
