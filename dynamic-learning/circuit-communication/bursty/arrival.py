"""Meter clocks with one event available at a time."""
import json
import numpy as np
from study import HERE, N, PERIODS, WARM, WORLDS, clock, energy, envelope, fixture, infer, neighbors


def arrival(readings, arm):
    if arm == 'gate':
        return infer(readings, arm)
    count, steps = readings.shape[:2]
    length = steps - WARM
    margins = np.empty((count, length))
    if arm == 'majority':
        for t in range(length):
            margins[:, t] = readings[:, WARM+t].sum(axis=1)
        return margins, np.zeros(count)
    period = int(arm)
    histories = np.lib.stride_tricks.sliding_window_view(readings, WARM, axis=1)
    for t in range(length):
        if t % period == 0:
            # Histories is a zero-copy view; this slice ends immediately before the current sign.
            bits = np.packbits(histories[:, t] > 0, axis=2)[..., 0]
            degrees = neighbors(bits, 1)
        margins[:, t] = (readings[:, WARM+t] / degrees).sum(axis=1)
    return margins, np.full(count, (length-1)//period)


def main():
    summary = {}
    for seed in (764802,764803):
        read = fixture(seed)[0]
        for p in PERIODS:
            sample = read[:3,:WARM+150]
            assert np.array_equal(arrival(sample, str(p))[0], clock(sample, p)[0])
        meters = energy(read, seed+401, arrival)
        primary = json.loads((HERE / f'panel-{seed}.json').read_text())
        clocks = list(map(str, PERIODS))
        indices = np.random.default_rng(501).integers(WORLDS,size=(2500,WORLDS))
        def draws(arm):
            acc = np.array(primary['arms'][arm]['world_accuracy'])
            return np.r_[acc.mean(),acc[indices].mean(axis=1)]
        gain = None
        if all(meters[a]['microjoules'] is not None for a in clocks + ['gate']):
            costs = np.array([meters[a]['microjoules'] for a in clocks])[:,None]
            acc = np.stack([draws(a) for a in clocks])
            budget = np.full(len(indices)+1, meters['gate']['microjoules'])
            deltas = draws('gate') - envelope(acc, costs, budget)
            gain = [deltas[0], *np.quantile(deltas[1:], [.025,.975])]
        summary[seed] = dict(costs=meters, gate_minus_clock_envelope=gain)
    (HERE/'arrival-results.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
