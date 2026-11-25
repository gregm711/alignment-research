"""Infer shared origins from unlabeled histories, then route two private operations."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

N, WARM, EVENTS, SYMBOLS = 32, 8, 64, 16
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
LOG_ODDS = np.log(.62 / .38)
CHECK_ODDS = np.log(4)
ARMS = ("vote", "oracle", "frozen", "rolling", "check_only")


def worlds(seed, count, drift):
    rng = np.random.default_rng(seed)
    steps = WARM + EVENTS
    before = rng.random((count, N)).argsort(axis=1) < 20
    after = rng.random((count, N)).argsort(axis=1) < 20
    groups = np.repeat(before[:, None, :], steps, axis=1)
    if drift:
        groups[:, WARM + EVENTS // 2:] = after[:, None, :]
    truth = rng.choice([-1, 1], (count, steps))
    ancestor = truth * np.where(rng.random(truth.shape) < .62, 1, -1)
    private = truth[:, :, None] * np.where(rng.random(groups.shape) < .62, 1, -1)
    inherited = ancestor[:, :, None] * np.where(rng.random(groups.shape) < .96, 1, -1)
    readings = np.where(groups, inherited, private).astype(np.int8)
    truth = truth[:, WARM:]
    check = truth * np.where(rng.random(truth.shape) < .80, 1, -1)
    shape = (count, EVENTS, 2, SYMBOLS)
    perms = rng.random(shape).argsort(axis=-1).astype(np.uint8)
    arg = rng.integers(SYMBOLS, size=(count, EVENTS), dtype=np.uint8)
    return dict(readings=readings, groups=groups[:, WARM:], check=check,
                perms=perms, arg=arg), truth


def vote(data, arm, radius):
    readings = data["readings"]
    signals = readings[:, WARM:]
    if arm == "check_only":
        return np.zeros(signals.shape[:2])
    if arm == "vote":
        return signals.sum(axis=2)
    if arm == "oracle":
        group = data["groups"]
        return (signals * ~group).sum(axis=2) + np.sign((signals * group).sum(axis=2))

    history = np.zeros(readings.shape[::2], dtype=np.uint8)
    for t in range(WARM):
        history = (history << 1) | (readings[:, t] > 0)
    margins = []
    for t in range(EVENTS):
        if t == 0 or arm == "rolling":
            distance = POPCOUNT[history[:, :, None] ^ history[:, None, :]]
            duplicates = (distance <= radius).sum(axis=2)
        margins.append((signals[:, t] / duplicates).sum(axis=1))
        if arm == "rolling":
            history = (history << 1) | (signals[:, t] > 0)
    return np.stack(margins, axis=1)


def route(data, choice, sever=False):
    # Only the addressed module's output is exposed per call.
    a, b = np.indices(choice.shape, sparse=True)
    intermediate = data["perms"][a, b, choice, data["arg"]]
    return data["perms"][a, b, 1 - choice, data["arg"] if sever else intermediate]


def decide(data, arm, margins, threshold):
    checked = np.ones_like(margins, dtype=bool) if arm == "check_only" else abs(margins) < threshold
    logits = margins * LOG_ODDS
    logits[checked] += data["check"][checked] * CHECK_ODDS
    order = (logits > 0).astype(int)
    return route(data, order), checked, order


def train():
    configs = {}
    fixtures = [worlds(7119 + i, 24, i == 1) for i in range(2)]
    for arm in ARMS:
        best = None
        for radius in ((0, 1, 2) if arm in ("rolling", "frozen") else (0,)):
            margins = [vote(d, arm, radius) for d, _ in fixtures]
            for threshold in ((0,) if arm == "check_only" else (0, 2, 5)):
                utility = 0.
                for (data, truth), margin in zip(fixtures, margins):
                    pred, checks, _ = decide(data, arm, margin, threshold)
                    gold = route(data, (truth > 0).astype(int))
                    utility += (pred == gold).mean() - .1 * checks.mean()
                if best is None or utility > best[0]:
                    best = (utility, radius, threshold)
        configs[arm] = best[1:]
    return configs


def timing(fn):
    _, first = measure(fn)
    repeats = max(1, int(.20 / max(.0001, first["wall_seconds"])) + 1)
    def batch():
        for _ in range(repeats):
            fn()
    trials = [measure(batch)[1] for _ in range(3)]
    return dict(repeats=repeats, trials=trials)


def interval(diff):
    rng = np.random.default_rng(3017)
    bootstrap = diff[rng.integers(len(diff), size=(3000, len(diff)))].mean(axis=1)
    return [float(diff.mean()), *np.quantile(bootstrap, [.025, .975]).tolist()]


def main():
    cfg = train()
    report = dict(configs=cfg, panels={})
    for drift in (False, True):
        data, truth = worlds(19274, 128, drift)
        gold = route(data, (truth > 0).astype(int))
        rows, correct = {}, {}
        for arm, (radius, threshold) in cfg.items():
            def infer():
                return decide(data, arm, vote(data, arm, radius), threshold)
            pred, checked, order = infer()
            right = pred == gold
            correct[arm] = right
            severed = route(data, order, sever=True) == gold
            assert right.shape == (128, 64)
            contact = 0 if arm == "check_only" else N
            setup = N * WARM if arm in ("rolling", "frozen") else 0
            rows[arm] = dict(accuracy=float(right.mean()), bits_per_event=contact + 18
                             + 2*checked.mean() + setup/EVENTS,
                             cells_per_event=contact + 3 + checked.mean() + setup/EVENTS,
                             calibration_bits_per_network=setup, checks_per_event=checked.mean(),
                             bit_accuracy=float((order == (truth > 0)).mean()),
                             severed_accuracy=float(severed.mean()),
                             severing_loss=interval((right.astype(float)-severed).mean(axis=1)),
                             timing=timing(infer))
        gain = correct["rolling"].astype(float) - correct["vote"]
        drift_gain = correct["rolling"][:,40:].astype(float) - correct["frozen"][:,40:]
        report["panels"]["drift" if drift else "fixed"] = dict(
            policies=rows, rolling_vs_vote=interval(gain.mean(axis=1)),
            late_rolling_vs_frozen=interval(drift_gain.mean(axis=1)),
            worlds=len(gold), events_per_world=EVENTS)
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
