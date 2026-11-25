"""Fit the price of a check on independent simulated format changes."""
import json
from pathlib import Path

import numpy as np

from direct import Direct

MODEL = Path(__file__).with_name("query-model.json")


def features(learner, producer, x):
    if "n" in x:
        digits = x["n"]
        shape = [1, len(digits) / 12, len(set(digits)) / 10, int(digits.startswith("0"))]
    else:
        shape = [0, int(len(x["m"]) == 2), int(len(x["d"]) == 2), int(x["m"] == x["d"])]
        digits = "".join(x.values())
    uncertainty = 1 - learner.confidence(producer, x)
    return [1, uncertainty, *shape, *(uncertainty * value for value in shape),
            len(set(digits)) / len(digits)]


def train():
    rng = np.random.default_rng(713)
    rows, targets = [], []
    for episode in range(128):
        phone = episode % 2
        if phone:
            xs = [{"n": "".join(map(str, rng.integers(10, size=int(rng.integers(7, 12)))))}
                  for _ in range(23)]
        else:
            xs = [dict(y=str(rng.integers(1980, 2060)), m=str(rng.integers(1, 13)),
                       d=str(rng.integers(1, 29))) for _ in range(23)]

        def rule():
            sep = rng.choice(["/", "-", ".", " "])
            if phone:
                cut = int(rng.integers(2, 6))
                return lambda x: x["n"][:cut] + sep + x["n"][cut:]
            order = rng.choice(["ymd", "dmy", "mdy"])
            pad = int(rng.integers(1, 3))
            return lambda x: sep.join(x[key].zfill(pad) for key in order)

        old = rule()
        new = rule() if rng.random() < .5 else old
        learner = Direct(1)
        for x in xs[:9]:
            learner.write("sim", x, old(x))
        probes, hold = xs[9:15], xs[15:]
        state = learner.state.copy()
        before = sum(learner.predict("sim", x) == new(x) for x in hold) / len(hold)
        for x in probes:
            rows.append(features(learner, "sim", x))
            mismatch = learner.predict("sim", x) != new(x)
            if mismatch:
                learner.write("sim", x, new(x))
            after = sum(learner.predict("sim", h) == new(h) for h in hold) / len(hold)
            targets.append(after - before + .25 * mismatch * (before < 1))
            learner.state[:] = state
    x = np.array(rows)
    weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ targets)
    assert np.isfinite(weights).all() and len(targets) == 768
    MODEL.write_text(json.dumps(dict(seed=713, simulations=128, weights=list(weights)), indent=2) + "\n")


if __name__ == "__main__":
    train()
