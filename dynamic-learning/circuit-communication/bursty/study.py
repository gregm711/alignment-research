"""Replay a frozen disagreement gate on clustered changes and price the clock envelope."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'hidden-sources'))
import gate
from run import N, WARM, LOG_ODDS, CHECK_ODDS, neighbors, vote, worlds
from meter import measure

LENGTH = 1200
WORLDS = 96
PERIODS = [*range(1, 17), 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, LENGTH]
ARMS = ['majority', 'gate', *map(str, PERIODS)]


def fixture(seed):
    rng = np.random.default_rng(seed)
    group = np.zeros((WORLDS, LENGTH + WARM, N), dtype=bool)
    shifts = np.zeros((WORLDS, LENGTH), dtype=bool)
    for w in range(WORLDS):
        t = 0
        while t < LENGTH + WARM:
            for span in (WARM * (t == 0) + rng.integers(100, 501),
                         rng.integers(4, 17), rng.integers(4, 17)):
                if t >= LENGTH + WARM:
                    break
                if t:
                    shifts[w, t-WARM] = True
                group[w, t:t+span, rng.choice(N, 20, replace=False)] = True
                t += span
    truth = rng.choice([-1, 1], group.shape[:2]).astype(np.int8)
    ancestor = np.where(rng.random(truth.shape) < .62, truth, -truth)
    inherited = np.where(rng.random(group.shape) < .96, ancestor[..., None], -ancestor[..., None])
    private = np.where(rng.random(group.shape) < .62, truth[..., None], -truth[..., None])
    observations = np.where(group, inherited, private)
    truth = truth[:, WARM:]
    check = np.where(rng.random(truth.shape) < .8, truth, -truth)
    early = shifts.copy()
    for lag in range(1, 8):
        early[:, lag:] |= shifts[:, :-lag]
    assert (group.sum(axis=2) == 20).all() and early.sum(axis=1).min() > 0
    return observations, truth, check, shifts, early, ancestor[:, WARM:] != truth


def clock(readings, period):
    signals = readings[:, WARM:]
    count, length = signals.shape[:2]
    output = np.empty((count, length))
    histories = np.lib.stride_tricks.sliding_window_view(readings, WARM, axis=1)
    # Predetermined clocks can amortize dispatch across independent worlds and intervals.
    chunk = max(1, 2048 // count)
    for start in range(0, length, period * chunk):
        end = min(start + period * chunk, length)
        histories_here = histories[:, start:end:period]
        packed = np.packbits(histories_here > 0, axis=3)[..., 0]
        degrees = neighbors(packed.reshape(-1, N), 1).reshape(count, -1, N)
        part = signals[:, start:end]
        full = (end - start) // period
        if full:
            values = part[:, :full*period].reshape(count, full, period, N)
            output[:, start:start+full*period] = (values / degrees[:, :full, None]).sum(axis=-1).reshape(count, -1)
        if full * period < end - start:
            output[:, start+full*period:end] = (part[:, full*period:] / degrees[:, -1:]).sum(axis=-1)
    return output, np.full(count, (length-1)//period)


def infer(readings, arm):
    if arm == 'majority':
        return readings[:, WARM:].sum(axis=2), np.zeros(len(readings), dtype=int)
    if arm == 'gate':
        gate.EVENTS = readings.shape[1] - WARM
        return gate.gated({'readings': readings}, 1)
    return clock(readings, int(arm))


def selfcheck():
    data, _ = worlds(101, 4, True)
    for p in (1, 2, 8, 13, 500):
        assert np.array_equal(vote(data, 'rolling', 1, p), infer(data['readings'], str(p))[0])
    expected, counts = gate.gated(data, 1)
    actual, replay_counts = infer(data['readings'], 'gate')
    assert np.array_equal(actual, expected) and np.array_equal(counts, replay_counts)
    shortened = data['readings'][:, :WARM+23]
    assert np.array_equal(actual[:, :23], infer(shortened, 'gate')[0])
    budget = np.array([2.])
    expected_envelope = envelope(np.array([[.7], [.9], [.71]]), np.array([[1.], [3.], [2.]]), budget)
    assert np.allclose(expected_envelope, [.8])


def energy(readings, seed, reader=infer):
    rng = np.random.default_rng(seed)
    readings = readings[:48]
    observations = readings.shape[0] * (readings.shape[1]-WARM)
    raw = {arm: [] for arm in ARMS}
    for _ in range(3):
        for arm in rng.permutation(ARMS):
            def decision():
                return reader(readings, arm)[0] > 0
            _, pilot = measure(decision)
            repeats = max(1, int(.08 / max(.00001, pilot['wall_seconds'])) + 1)
            def repeat():
                for _ in range(repeats):
                    decision()
            _, measured = measure(repeat)
            measured['repeats'] = repeats
            raw[arm].append(measured)
    return {arm: dict(trials=trials, decisions=observations, microjoules=np.median(
        [m['cpu_joules']/m['repeats'] for m in trials])*1e6/observations if all(
            m['cpu_joules'] is not None for m in trials) else None,
        wall_us=np.median([m['wall_seconds']/m['repeats'] for m in trials])*1e6/observations)
            for arm, trials in raw.items()}


def envelope(accuracy, costs, budget):
    # Each column represents a bootstrap draw; rows represent policies.
    best = np.full(accuracy.shape[1], -np.inf)
    for i in range(len(costs)):
        best = np.maximum(best, np.where(costs[i] <= budget, accuracy[i], -np.inf))
    for i, j in itertools.combinations(range(len(costs)), 2):
        delta = costs[j] - costs[i]
        weight = (budget - costs[i]) / np.where(delta != 0, delta, 1)
        viable = (delta != 0) & (weight >= 0) & (weight <= 1)
        mixture = accuracy[i] + weight * (accuracy[j] - accuracy[i])
        best = np.maximum(best, np.where(viable, mixture, -np.inf))
    assert np.isfinite(best).all()
    return best


def one(seed):
    read, truth, check, shifts, early, wrong_source = fixture(seed)
    checks_gate = abs(infer(read, 'gate')[0]) < 2
    rows = {}
    correct, rebuilds = {}, {}
    for arm in ARMS:
        margins, count = infer(read, arm)
        base = ((margins > 0) == (truth > 0))
        checks = abs(margins) < 2
        def verified(receipts):
            return (margins*LOG_ODDS + receipts*check*CHECK_ODDS > 0) == (truth > 0)
        correct[arm] = base.mean(axis=1)
        rebuilds[arm] = count + int(arm != 'majority')
        rows[arm] = dict(accuracy=base.mean(), first_eight=base[early].mean(),
            other=base[~early].mean(), wrong_ancestor=base[wrong_source].mean(),
            own_verified=verified(checks).mean(), own_check_rate=checks.mean(),
            gate_receipts_verified=verified(checks_gate).mean(),
            rebuilds_per_world=rebuilds[arm].mean(),
            bits_per_decision=N*(1+(WARM/LENGTH if arm != 'majority' else 0)),
            active_relay_visits_per_decision=N*(1+(WARM/LENGTH if arm != 'majority' else 0)),
            world_accuracy=correct[arm].tolist(), world_rebuilds=rebuilds[arm].tolist())
    meters = energy(read, seed+201)
    clocks = list(map(str, PERIODS))
    index = np.random.default_rng(501).integers(WORLDS, size=(2500, WORLDS))
    def draws(xs):
        return np.r_[xs.mean(), xs[index].mean(axis=1)]
    acc = np.stack([draws(correct[arm]) for arm in clocks])
    output = {}
    for metric in ('rebuilds','microjoules'):
        if metric == 'microjoules' and any(meters[a][metric] is None for a in clocks + ['gate']):
            continue
        def cost(arm):
            if metric == 'rebuilds':
                return draws(rebuilds[arm])
            return np.full(acc.shape[1], meters[arm][metric])
        gain = draws(correct['gate']) - envelope(acc, np.stack([cost(c) for c in clocks]), cost('gate'))
        output[metric] = [gain[0], *np.quantile(gain[1:], [.025, .975])]
    return dict(arms=rows, costs=meters, gate_minus_clock_envelope=output,
                total_shifts=int(shifts.sum()), post_shift_events=int(early.sum()), wrong_ancestor_events=int(wrong_source.sum()))


if __name__ == '__main__':
    selfcheck()
    for seed in (764802, 764803):
        report = one(seed)
        (HERE / f'panel-{seed}.json').write_text(json.dumps(report, indent=2)+'\n')
