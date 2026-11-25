"""Evaluate frozen hidden-source voting on held-out chronological gas recordings."""
import ctypes
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

import numpy as np

from prepare import CACHE, PANELS, SHA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

MODES = dict(majority=-1, competence=-2, single=-3, reactive=-4, frozen=0)
MODES.update({f"clock-{p}": p for p in (1, 2, 4, 8, 16, 32, 64, 128, 256)})


def calibration(x, y):
    cuts, signs, accs = [], [], []
    for col in x.T:
        thresholds = np.quantile(col, np.arange(1, 40) / 40)
        positive = col[:, None] > thresholds
        scores = (positive[y].mean(axis=0) + (~positive[~y]).mean(axis=0)) / 2
        both = np.stack([scores, 1-scores])
        orient, idx = np.unravel_index(both.argmax(), both.shape)
        cuts.append(thresholds[idx])
        signs.append(1 if orient == 0 else -1)
        correct = positive[:, idx] if orient == 0 else ~positive[:, idx]
        accs.append(((correct[y].sum()+1)/(y.sum()+2) +
                     ((~correct[~y]).sum()+1)/((~y).sum()+2)) / 2)
    a = np.array(accs)
    return (np.array(cuts), np.array(signs, dtype=np.int8),
            np.log(a / (1-a)).clip(min=0), int(a.argmax()))


def build(folder):
    binary = Path(folder) / "receiver.so"
    subprocess.run(["cc", "-std=c99", "-O3", "-fPIC", "-shared",
                    str(HERE / "infer.c"), "-o", str(binary)], check=True)
    infer = ctypes.CDLL(str(binary)).infer
    ptr = np.ctypeslib.ndpointer
    infer.argtypes = [ctypes.c_int, ctypes.c_int, ptr(np.float64, flags="C_CONTIGUOUS"),
                      ptr(np.float64, ndim=1), ptr(np.int8, ndim=1), ptr(np.float64, ndim=1),
                      ctypes.c_int, ptr(np.uint8, ndim=1)]
    infer.restype = ctypes.c_int
    return infer


def self_check(infer):
    rng = np.random.default_rng(17)
    values = rng.normal(size=(108, 16))
    cut, sign, weight, best = calibration(values[:8], np.arange(8) % 2 == 0)
    bits = (values > cut) == (sign > 0)
    for mode in [-4, 1, 8, 128, 0]:
        pred = np.empty(100, dtype=np.uint8)
        calls = infer(mode, 100, values, cut, sign, weight, best, pred)
        count, flags, age, at = 0, 0, 0, 0
        for t in range(100):
            if t == at:
                h = bits[t:t+8]
                d = np.count_nonzero(h[:, :, None] != h[:, None, :], axis=0)
                degree = (d <= 1).sum(axis=1)
            v = 2 * bits[t+8].astype(int) - 1
            assert bool(pred[t]) == ((v * (720720 // degree)).sum() > 0)
            group = v[degree >= 8]
            bad = len(group) == 0 or abs(group.mean()) < .75
            flags = (flags*2 + bad) & 15
            age += 1
            if t < 99 and ((mode > 0 and age == mode) or
                           (mode == -4 and age >= 6 and flags.bit_count() >= 2)):
                count += 1
                age = flags = 0
                at = t+1
        assert count == calls


def energy(fn):
    start = time.perf_counter()
    for _ in range(4):
        fn()
    reps = max(1, min(1500, round(.075*4/(time.perf_counter()-start))))
    def loop():
        for _ in range(reps):
            fn()
    trials = [measure(loop)[1] for _ in range(3)]
    return dict(reps=reps, trials=trials)


def report(name, infer):
    raw = np.load(CACHE / (name + ".npy"), allow_pickle=False)
    cut = len(raw) // 4
    x = np.ascontiguousarray(raw[:, 3:])
    y = raw[:, 2] > 0
    fit = calibration(x[:cut], y[:cut])
    test, labels, ts = x[cut-8:], y[cut:], raw[cut:, 0]
    n = len(labels)
    block = np.floor((ts-ts[0])/300).astype(int)
    blocks = np.unique(block)
    assert np.array_equal(blocks, np.arange(len(blocks)))
    samples = np.random.default_rng(59071).integers(0, len(blocks), (2000, len(blocks)))
    switch_time = np.maximum.accumulate(np.where(np.diff(y.astype(int), prepend=int(y[0])) != 0,
                                                raw[:, 0], -np.inf))
    recent = (raw[:, 0] - switch_time < 30)[cut:]
    switches = np.flatnonzero(y[cut:] != y[cut-1:-1])
    def describe(a):
        return [float(z) for z in np.quantile(a, [0,.25,.5,.75,1])] if len(a) else []
    out = dict(calibration_rows=cut, rows=n, positives=int(labels.sum()),
               test_start_s=ts[0], test_end_s=ts[-1], sampling_period_s=describe(np.diff(ts)),
               binary_changes=len(switches), transition_events=int(recent.sum()),
               complete_dwell_s=describe(np.diff(ts[switches])), blocks=len(blocks),
               calibration=dict(cuts=fit[0].tolist(), signs=fit[1].tolist(),
                                competence_weights=fit[2].tolist(), single_sensor=fit[3]+1))
    results, boots, decisions = {}, {}, {}
    for arm in np.random.default_rng(621).permutation(list(MODES)):
        mode = MODES[arm]
        pred = np.empty(n, dtype=np.uint8)
        def compute():
            return infer(mode, n, test, *fit, pred)
        calls = compute()
        correct = pred == labels
        decisions[arm] = pred.copy()
        ben = energy(compute)
        trial = ben["trials"]
        cls = {}
        for k in [False, True]:
            mask = labels == k
            num = np.bincount(block, weights=correct & mask)[samples].sum(axis=1)
            den = np.bincount(block, weights=mask)[samples].sum(axis=1)
            cls[k] = num / den
        bacc = (correct[labels].mean()+correct[~labels].mean())/2
        boots[arm] = (cls[False]+cls[True])/2
        amortized = (median(t["cpu_joules"] for t in trial) / ben["reps"] / n * 1e6
                     if all(t["cpu_joules"] is not None for t in trial) else None)
        dynamic = mode >= 0 or mode == -4
        results[arm] = dict(accuracy=correct.mean(), balanced_accuracy=bacc,
                            conditional_accuracy=[correct[~labels].mean(),correct[labels].mean()],
                            near_change_accuracy=correct[recent].mean(),
                            refreshes_after_setup=calls, setup_refreshes=int(dynamic),
                            reports_per_event=1 if mode == -3 else 16,
                            historical_report_bits=128 if dynamic else 0, checks=0,
                            pairwise_comparisons=256*(calls+int(dynamic)),
                            microjoules_per_decision=amortized, timing=ben,
                            microseconds_per_decision=median(t["wall_seconds"] for t in trial)/ben["reps"]/n*1e6)
    # Descriptive after the frozen evaluation: identify which majority decisions reweighting reversed.
    votes = (test[8:] > fit[0]) == (fit[1] > 0)
    unanimous = (votes == votes[:, :1]).all(axis=1)
    majority_right = decisions["majority"] == labels
    reversed_majority = decisions["reactive"] != decisions["majority"]
    out["agreement_audit"] = dict(unanimous=int(unanimous.sum()),
        unanimous_correct=int((unanimous & majority_right).sum()),
        majority_errors_corrected=int((reversed_majority & ~majority_right).sum()),
        correct_majorities_lost=int((reversed_majority & majority_right).sum()))
    out["arms"] = results
    out["paired_gate_minus_other_balanced_accuracy"] = {
        arm: [results["reactive"]["balanced_accuracy"]-r["balanced_accuracy"],
              *np.quantile(boots["reactive"]-boots[arm], [.025,.975]).tolist()]
        for arm, r in results.items() if arm != "reactive"}
    return out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        runner = build(tmp)
        self_check(runner)
        panels = {k: report(k, runner) for k in PANELS}
    dest = Path(sys.argv[1]) if len(sys.argv)>1 else HERE / "results.json"
    dest.write_text(json.dumps(dict(source_sha256=SHA, panels=panels), indent=2, allow_nan=False) + "\n")
