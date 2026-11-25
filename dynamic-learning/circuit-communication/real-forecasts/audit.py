"""Recompute decisions, causal prefix checks, and a no-increase comparison."""
import json

import numpy as np
import pandas as pd

from run import ARMS, CACHE, HERE, WARM, block_interval, load_panel, predict


def main():
    saved = json.loads((HERE / "results.json").read_text())
    dates = sorted(p.name[:10] for p in CACHE.glob("*flu_forecasts_data.csv"))
    truth = pd.read_csv(CACHE / "target-hospital-admissions.csv").pivot(
        index="date", columns="location_name", values="value")
    contrast, sizes, broken, fixed = [], [], 0, 0
    for i, bounds in enumerate((("2024-11", "2025-06"), ("2025-11", "2026-06"))):
        p = load_panel([d for d in dates if bounds[0] <= d <= bounds[1]], truth)
        predictions = {}
        for arm in ARMS:
            pred, _ = predict(p["q"], p["anchor"], p["weights"], arm)
            prefix, _ = predict(p["q"][:-3], p["anchor"][:-3], p["weights"], arm)
            assert np.array_equal(prefix, pred[:-3])
            correct = ((pred == p["gold"]) & p["mask"]).sum(axis=1).tolist()
            assert correct == saved["arms"][arm]["issue_correct"][i]
            predictions[arm] = pred
        gate = predictions["reactive"] == p["gold"]
        majority = predictions["majority"] == p["gold"]
        broken += int((p["mask"] & majority & ~gate).sum())
        fixed += int((p["mask"] & ~majority & gate).sum())
        null = ~p["gold"]
        contrast.append(((gate.astype(int) - null) * p["mask"]).sum(axis=1))
        sizes.append(p["mask"].sum(axis=1))
    out = dict(reactive_fixes_over_majority=fixed, reactive_breaks_over_majority=broken,
               reactive_minus_no_increase=[float(sum(x.sum() for x in contrast)/sum(x.sum() for x in sizes)),
                                            *block_interval(contrast,sizes)])
    (HERE / "audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
