"""Replay the learned receiver's evidence and test a fixed interval between refreshes."""
import json
from statistics import median

import numpy as np

from run import CHECK_ODDS, EVENTS, HERE, LOG_ODDS, decide, interval, route, timing, vote, worlds


def per_event(measured, n):
    return {key: median(t[key] for t in measured["trials"]) / (n * measured["repeats"])
            for key in ("cpu_joules", "wall_seconds") if measured["trials"][0][key] is not None}


def main():
    original = json.loads((HERE / "results.json").read_text())
    radius, threshold = original["configs"]["rolling"]
    summary = {}
    for seed in (19274, 56093):
        for drift in (False, True):
            data, truth = worlds(seed, 128, drift)
            gold = route(data, (truth > 0).astype(int))
            naive = vote(data, "vote", 0)
            def infer(period):
                return decide(data, "rolling", vote(data, "rolling", radius, period), threshold)
            regular, checks, _ = infer(1)
            equal_receipts = (naive*LOG_ODDS + (checks * data["check"])*CHECK_ODDS > 0).astype(int)
            no_check = route(data, (naive > 0).astype(int))
            inferred_no_check = route(data, (vote(data, "rolling", radius) > 0).astype(int))
            periodic, periodic_checks, _ = infer(8)
            assert not np.any(periodic_checks & (np.abs(vote(data, "rolling", radius, 8)) >= threshold))

            def difference(a, b):
                return interval(((a == gold).astype(float) - (b == gold)).mean(axis=1))

            label = f"{seed}-{'drift' if drift else 'fixed'}"
            times = {str(p): timing(lambda: infer(p)) for p in (1, 8)}
            summary[label] = dict(
                regular=float((regular == gold).mean()), periodic=float((periodic == gold).mean()),
                periodic_checks_per_event=periodic_checks.mean(),
                periodic_minus_full=difference(periodic, regular),
                rolling_minus_vote_same_receipts=difference(regular, route(data, equal_receipts)),
                rolling_minus_vote_no_checks=difference(inferred_no_check, no_check),
                per_event={p:per_event(t, len(gold)*EVENTS) for p,t in times.items()},
                timing=times)
    (HERE / "refresh-results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
