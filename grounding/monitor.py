"""Phases 2 and 3 (offline, no GPU): build monitors from visible cases on training questions, freeze them,
and test on hidden ungrounded answers.

    python3 monitor.py            # reads results/phase1.json and results/phase1_features.pt, writes results/phase2_3.json
"""
import json
import pathlib
import random

import torch

HERE = pathlib.Path(__file__).parent / "results"
ANS, PAS, QUE = 0, 1, 2  # feature channels: attention on the answer's location, the passage, the question


def auc(pos, neg):
    pos, neg = torch.as_tensor(pos, dtype=torch.float64), torch.as_tensor(neg, dtype=torch.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()).mean())


def boot(pos, neg, n=2000):
    r, pos, neg, vals = random.Random(0), list(map(float, pos)), list(map(float, neg)), []
    if len(pos) < 5 or len(neg) < 5:
        return [auc(pos, neg), float("nan"), float("nan")]
    for _ in range(n):
        vals.append(auc(r.choices(pos, k=len(pos)), r.choices(neg, k=len(neg))))
    vals.sort()
    return [auc(pos, neg), vals[int(0.025 * n)], vals[int(0.975 * n)]]


def main():
    data = json.loads((HERE / "phase1.json").read_text())
    F = torch.load(HERE / "phase1_features.pt", map_location="cpu").float()  # [runs, layers, heads, 3]
    rows = data["rows"]
    L, H = F.shape[1:3]

    def sub_runs(split):  # visible cases: substituted-passage runs, labelled from the output alone
        out = [(s["feat"], s) for r in rows if r["split"] == split for s in r["subs"] if s["outcome"] in ("memory", "passage")]
        return out, torch.tensor([int(s["outcome"] == "memory") for _, s in out])

    def orig_runs(split, known_only=False):  # hidden cases: original-passage runs of labelled questions; all answers correct
        out = [(r["original"]["feat"], r["original"]) for r in rows if r["split"] == split and r["label"] in ("hidden_ungrounded", "grounded")
               and (r["known_closed_book"] or not known_only)]
        lab = [int(r["label"] == "hidden_ungrounded") for r in rows if r["split"] == split and r["label"] in ("hidden_ungrounded", "grounded")
               and (r["known_closed_book"] or not known_only)]
        return out, torch.tensor(lab)

    (tr, ytr), (te_vis, yv) = sub_runs("train"), sub_runs("test")
    Xtr = F[[i for i, _ in tr]]
    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())

    # --- fit on training questions' visible cases only ---
    def best_head(channel):
        scores = -Xtr[..., channel]  # low attention = suspicious
        a = torch.tensor([[auc(scores[ytr == 1, l, h], scores[ytr == 0, l, h]) for h in range(H)] for l in range(L)])
        l, h = divmod(int(a.argmax()), H)
        return l, h, float(a[l, h]), float(scores[ytr == 0, l, h].quantile(0.9))

    ev = best_head(ANS)
    pa = best_head(PAS)
    avg_thr = float((-Xtr[..., ANS].mean((1, 2)))[ytr == 0].quantile(0.9))
    ratio = lambda X: (X[..., PAS] / (X[..., PAS] + X[..., QUE] + 1e-6)).reshape(len(X), -1)  # Lookback-style share on the passage
    R = ratio(Xtr)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    w = torch.zeros(R.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)
    Z, yf = (R - mu) / sd, ytr.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yf) + 1.0 * (w ** 2).sum()  # strong L2: few positives
        loss.backward()
        return loss

    opt.step(closure)
    clf = lambda X: (((ratio(X) - mu) / sd) @ w + b).detach()
    clf_thr = float(clf(Xtr)[ytr == 0].quantile(0.9))
    conf_thr = {k: float(torch.tensor([sgn * s[k] for _, s in tr])[ytr == 0].quantile(0.9)) for k, sgn in (("p_top", -1), ("margin", -1), ("entropy", 1))}

    monitors = {
        "evidence head": (lambda X, runs: -X[:, ev[0], ev[1], ANS], ev[3]),
        "passage head": (lambda X, runs: -X[:, pa[0], pa[1], PAS], pa[3]),
        "all-heads average": (lambda X, runs: -X[..., ANS].mean((1, 2)), avg_thr),
        "attention classifier": (lambda X, runs: clf(X), clf_thr),
        "output top probability": (lambda X, runs: torch.tensor([-s["p_top"] for _, s in runs]), conf_thr["p_top"]),
        "output margin": (lambda X, runs: torch.tensor([-s["margin"] for _, s in runs]), conf_thr["margin"]),
        "output entropy": (lambda X, runs: torch.tensor([s["entropy"] for _, s in runs]), conf_thr["entropy"]),
    }

    def evaluate(runs, y):
        X, res = F[[i for i, _ in runs]], {}
        for name, (fn, thr) in monitors.items():
            sc = fn(X, runs)
            res[name] = {"auc": boot(sc[y == 1], sc[y == 0]),
                         "caught_at_frozen_threshold": f"{int((sc[y == 1] > thr).sum())}/{int(y.sum())}",
                         "false_alarms_at_frozen_threshold": f"{int((sc[y == 0] > thr).sum())}/{int((1 - y).sum())}"}
        return res

    out = {"training": {"visible_ungrounded_runs": n_pos, "passage_following_runs": n_neg,
                        "evidence_head": {"layer": ev[0], "head": ev[1], "train_auc": ev[2]},
                        "passage_head": {"layer": pa[0], "head": pa[1], "train_auc": pa[2]}},
           "phase2_visible_on_test_questions": {"n": [int(yv.sum()), int((1 - yv).sum())], "monitors": evaluate(te_vis, yv)}}
    for name, known in (("phase3_hidden_on_test_questions", False), ("phase3_control_known_questions_only", True)):
        runs, y = orig_runs("test", known)
        out[name] = {"n": [int(y.sum()), int((1 - y).sum())], "monitors": evaluate(runs, y)}
    (HERE / "phase2_3.json").write_text(json.dumps(out, indent=1))

    fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    print(f"fitted on {n_pos} visible ungrounded and {n_neg} passage-following runs from training questions")
    print(f"evidence head: layer {ev[0]} head {ev[1]} (train AUC {ev[2]:.3f}); passage head: layer {pa[0]} head {pa[1]} (train AUC {pa[2]:.3f})")
    for key, title in (("phase2_visible_on_test_questions", "PHASE 2  visible ungrounded vs passage-following, test questions"),
                       ("phase3_hidden_on_test_questions", "PHASE 3  hidden ungrounded vs grounded, test questions, original passage"),
                       ("phase3_control_known_questions_only", "CONTROL  same, only questions the model knows with no passage")):
        print(f"\n{title}  (n = {out[key]['n'][0]} vs {out[key]['n'][1]})")
        for name, m in out[key]["monitors"].items():
            print(f"  {name:<24} AUC {fmt(m['auc'])} | caught {m['caught_at_frozen_threshold']:>8} | false alarms {m['false_alarms_at_frozen_threshold']}")


if __name__ == "__main__":
    main()
