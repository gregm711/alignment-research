"""Recompute only networks whose putative common-origin group stops agreeing."""
import json
import numpy as np

from refresh import per_event
from run import EVENTS, HERE, N, WARM, decide, interval, neighbors, route, timing, vote, worlds


def gated(data, radius):
    readings = data["readings"]
    history = np.packbits(np.moveaxis(readings[:, :WARM] > 0, 1, -1), axis=-1)[:, :, 0]
    degrees = neighbors(history, radius)
    suspects = degrees >= 8
    flags = np.zeros(len(readings), dtype=np.uint8)
    age = np.zeros(len(readings), dtype=int)
    count = np.zeros(len(readings), dtype=int)
    margins = []
    for t in range(EVENTS):
        observation = readings[:, WARM + t]
        margins.append((observation / degrees).sum(axis=1))
        size = suspects.sum(axis=1)
        flagged = (size == 0) | (abs((observation * suspects).sum(axis=1)) < .75 * size)
        flags = ((flags << 1) | flagged) & 15
        age += 1
        history = (history << 1) | (observation > 0)
        trigger = (age >= 6) & (np.bitwise_count(flags) >= 2)
        if t + 1 < EVENTS and np.any(trigger):
            degrees[trigger] = neighbors(history[trigger], radius)
            suspects[trigger] = degrees[trigger] >= 8
            count[trigger] += 1
            age[trigger] = 0
            flags[trigger] = 0
    return np.stack(margins, axis=1), count


def main():
    radius, threshold = json.loads((HERE / "results.json").read_text())["configs"]["rolling"]
    summary = {}
    for drift in (False, True):
        data, truth = worlds(60914, 256, drift)
        gold = route(data, (truth > 0).astype(int))
        masks, rows = {}, {}
        for name, period in (("continuous", 1), ("periodic", 8), ("frozen", EVENTS), ("gated", None)):
            def infer():
                if name == "gated":
                    margins, count = gated(data, radius)
                else:
                    margins = vote(data, "frozen" if name == "frozen" else "rolling", radius, period)
                    count = np.full(len(truth), (EVENTS-1) // period)
                pred, checks, _ = decide(data, "rolling", margins, threshold)
                return pred, checks, count
            pred, checks, count = infer()
            right = pred == gold
            masks[name] = right
            cost = timing(infer)
            rows[name] = dict(accuracy=float(right.mean()), checks_per_event=checks.mean(),
                              post_setup_refreshes=count.mean(),
                              bits_per_event=N + N*WARM/EVENTS + 18 + 2*checks.mean(),
                              active_cells_per_event=N + N*WARM/EVENTS + 3 + checks.mean(),
                              timing=cost, per_event=per_event(cost, right.size))
        advantage = {k: interval((masks["gated"].astype(float) - value).mean(axis=1))
                     for k, value in masks.items() if k != "gated"}
        summary["drift" if drift else "fixed"] = dict(arms=rows, gated_minus=advantage)
    (HERE / "gate-results.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
