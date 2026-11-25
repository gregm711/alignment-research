"""Recompute all summaries from the recorded paired trials without training or PyTorch."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("experienced", "reset", "conventional", "sgd", "adam", "fresh-adam", "own-sgd",
         "own-adam", "shuffled", "matched-norm", "prior", "peer", "context")
BUDGETS = np.array([0, 1, 2, 4, 8])


def estimate(values):
    values = np.array(values)
    resamples = np.random.default_rng(1501).choice(values, (10000, len(values)))
    return {"mean": float(values.mean()), "ci": np.quantile(resamples.mean(1), [.025, .975]).tolist(),
            "by_seed": values.tolist()}


def feedback_rates(x, y):
    result = {}
    for endpoint, eligible, col in (("harm", True, 2), ("repair", False, 1)):
        mask = (x[:, 0] == eligible) & (y[:, 0] == eligible)
        count = int(mask.sum())
        left = int((x[mask, col] != eligible).sum())
        right = int((y[mask, col] != eligible).sum())
        result[endpoint] = [count, left, right]
    return result


def main():
    meta = json.loads((ROOT / "run.json").read_text())
    seeds = [row["seed"] for row in meta["seeds"]]
    with np.load(ROOT / "trials.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    report = {}
    for family in ("same", "shift"):
        report[family] = {}
        for t in (0, 1, 10, 100):
            def get(s, name):
                return data[f"{s}-{t}-{family}-{name}"]
            result = {"methods": {}, "paired": {}, "feedback": {}}
            primary = []
            for name in NAMES:
                curves = np.stack([get(s, name) for s in seeds])
                assert curves.shape == (len(seeds), 128, len(BUDGETS)) and np.isfinite(curves).all()
                avg = curves[:, :, 1:].mean(axis=(1, 2))
                if name == "experienced":
                    primary = avg
                hits = curves >= .75
                reached = hits.any(-1)
                budgets = BUDGETS[hits.argmax(-1)]
                result["methods"][name] = {
                    "curve": curves.mean(axis=(0, 1)).tolist(), "post_mean": float(avg.mean()),
                    "reached_75": int(reached.sum()), "trials": int(reached.size),
                    "mean_budget_among_reached": float(budgets[reached].mean()) if reached.any() else None}
                result["paired"][name] = estimate(primary - avg)
            for name in ("conventional", "reset", "prior", "matched-norm"):
                counts = [feedback_rates(get(s, "experienced-fb"), get(s, name + "-fb")) for s in seeds]
                result["feedback"][name] = {}
                for key in ("harm", "repair"):
                    rows = np.array([c[key] for c in counts])
                    n, x, y = rows.sum(0)
                    valid = rows[:, 0] > 0
                    result["feedback"][name][key] = {
                        "eligible": int(n), "experienced_count": int(x), "other_count": int(y),
                        "by_seed_counts": rows.tolist(),
                        "paired": estimate((rows[valid, 1] - rows[valid, 2]) / rows[valid, 0]) if valid.any() else None}
            report[family][str(t)] = result
        for name in ("conventional", "reset"):
            last = report[family]["100"]["paired"][name]["by_seed"]
            first = report[family]["0"]["paired"][name]["by_seed"]
            report[family]["gain-over-life-" + name] = estimate(np.subtract(last, first))
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for family in ("same", "shift"):
        for t in (0, 1, 10, 100):
            block = report[family][str(t)]
            print(family, t, "accuracy %", {k: round(100*v["post_mean"], 1)
                  for k, v in block["methods"].items()})
        print("checkpoint 100 paired points", {k: [round(100*x, 2) for x in (v["mean"], *v["ci"])]
              for k,v in block["paired"].items()})
        print("feedback vs conventional", block["feedback"]["conventional"])


if __name__ == "__main__":
    main()
