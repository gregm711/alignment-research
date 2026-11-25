"""Test evidence-dependent weight updates and source calibration on unseen tool rules."""
import itertools
import json
from pathlib import Path

import pilot
import numpy as np
import torch
from analyse import estimate

ROOT = Path(__file__).resolve().parent
SEEDS = range(241, 249)
BUDGETS = (0, 4, 8, 16)
RATES = (.1, .4, 1.6)
GROUPS = ("old-majority", "new-majority", "old-conditional", "new-conditional")


def tasks(rng, pools, size):
    labels, orders = [], []
    for group, triples in enumerate(pools):
        selected = triples[rng.integers(len(triples), size=size)]
        # Each member of an unordered triple can serve as the conditional switch.
        selected = np.take_along_axis(selected, rng.random(selected.shape).argsort(1), axis=1)
        signs = rng.choice([-1, 1], size=selected.shape)
        values = pilot.TRUTH[:, selected] * signs
        if group < 2:
            output = values.sum(-1) > 0
        else:
            output = np.where(values[:, :, 0] > 0, values[:, :, 1], values[:, :, 2]) > 0
        order = rng.random((size, len(pilot.BITS))).argsort(1)
        for row, ids in enumerate(selected):
            remaining = order[row, 16:].copy()
            features = pilot.TRUTH[remaining][:, ids]
            left, right = np.argwhere(np.triu((features[:, None] == features[None, :]).all(-1), 1))[0]
            order[row, 16:18] = remaining[[left, right]]
            order[row, 18:] = np.delete(remaining, [left, right])
        labels.append(output.T)
        orders.append(order)
    order = np.vstack(orders)
    y = np.vstack(labels)
    assert np.all(np.diff(np.sort(order, axis=1), axis=1) == 1)
    assert np.array_equal(y[np.arange(len(y)), order[:, 16]], y[np.arange(len(y)), order[:, 17]])
    return y, order


def run_adapter(table, gate, dataset, config, donors):
    name, rate, control = config
    y, order = dataset
    n = len(y)
    rows = np.arange(n)
    # Only verified support is visible to any of the three online adapters.
    w = np.zeros((n, 12))
    plain, old = w.copy(), w.copy()
    state = pilot.fresh(n)
    evidence = np.zeros(n)
    curves, mixing, snapshots = [], [], []
    query = order[:, 18:]
    query_x = table[query]
    for step in range(17):
        if name in ("adaptive", "shuffled"):
            log_odds = evidence[donors] if name == "shuffled" else evidence
            mix = pilot.probability(control * log_odds)
        else:
            mix = np.full(n, control)
        direction = 1. + mix[:, None] * (gate - 1.)
        if step in BUDGETS:
            pred = (query_x * w[:, None]).sum(-1) >= 0
            curves.append((pred == y[rows[:, None], query]).mean(axis=1))
            snapshots.append(w.copy())
            mixing.append(mix.copy())
        if step == 16:
            break
        ids = order[:, step]
        x, label = table[ids], y[rows, ids].astype(float)
        if name in ("adaptive", "shuffled"):
            sign = 2 * label - 1
            # Out-of-sample log score difference, evaluated before either learns this label.
            evidence += (np.logaddexp(0., -sign * (plain*x).sum(1))
                         - np.logaddexp(0., -sign * (old*x).sum(1)))
            plain -= rate * pilot.gradient(plain, x, label)
            old -= rate * gate * pilot.gradient(old, x, label)
        w, state = pilot.advance(w, pilot.gradient(w, x, label), name, rate, state, direction)
    return np.stack(curves, 1), np.stack(mixing, 1), np.stack(snapshots, 1), direction


def feedback(table, dataset, w, direction, rate, reliability):
    labels, order = dataset
    rows = np.arange(len(labels))
    x, probe = table[order[:, 16]], table[order[:, 17]]
    y = labels[rows, order[:, 16]].astype(float)
    correct = ((w * probe).sum(1) >= 0) == y
    strength = np.maximum(0., 2 * reliability - 1)
    modes = {"calibrated": strength, "common": np.repeat(strength.mean(), 2),
             "swapped": strength[::-1], "ordinary": np.ones(2), "none": np.zeros(2)}
    result = {}
    for mode, scales in modes.items():
        by_source = []
        for scale in scales:
            trials = []
            for label in (1-y, y):
                updated = w - scale * rate * direction * pilot.gradient(w, x, label)
                after = ((updated * probe).sum(1) >= 0) == y
                trials.append(after)
            by_source.append(np.column_stack(trials))
        result[mode] = np.stack(by_source, axis=1)
    assert np.all(result["none"] == correct[:, None, None])
    return result


def linear_reference(table, data):
    labels, order = data
    # Forty verified inputs plus an intercept; the final 24 are held out.
    x = np.column_stack([table, np.ones(len(table))])[order]
    y = np.take_along_axis(labels, order, axis=1)
    train = x[:, :40]
    gram = train.transpose(0, 2, 1) @ train + .001 * np.eye(x.shape[-1])
    coef = np.linalg.solve(gram, train.transpose(0, 2, 1) @ (2.*y[:, :40, None]-1))
    return (((x[:, 40:] @ coef)[:, :, 0] >= 0) == y[:, 40:]).mean(1)


def donor_ids(rng, n):
    order = rng.permutation(n)
    donors = np.empty_like(order)
    donors[order] = np.roll(order, 1)
    assert np.all(donors != np.arange(n))
    return donors


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw, settings = {}, []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 9000)
        table, _ = pilot.feature_table(seed)
        triples = rng.permutation(pilot.TRIPLES)
        other = rng.permutation(pilot.SHIFT)
        gate = pilot.life(table, seed, "gate", pilot.BASE_RATE, triples[:12])[100][1]
        dev = tasks(rng, [triples[12:16], other[:10]], 64)
        test = tasks(rng, [triples[16:], other[10:], triples[16:], other[10:]], 192)
        donors = donor_ids(rng, len(dev[0]))
        options = {"uniform": [0.], "practiced": [1.], "fixed": [.25,.5,.75],
                   "adam": [0.], "adaptive": [.25,1.,4.]}
        chosen, scores = {}, {}
        for name, constants in options.items():
            configs = list(itertools.product([name], RATES, constants))
            values = [run_adapter(table, gate, dev, cfg, donors)[0][:, 1:].mean() for cfg in configs]
            best = int(np.argmax(values))
            chosen[name], scores[name] = configs[best], float(values[best])
        winner = max(("uniform","practiced","fixed","adam"), key=scores.get)
        chosen["shuffled"] = ("shuffled", *chosen["adaptive"][1:])
        chosen["winner"] = chosen[winner]
        donors = donor_ids(rng, len(test[0]))
        # Source slots are randomized within lifetime; all scoring restores reliability order.
        identities = rng.permutation([.9,.55])
        reliability = (rng.binomial(40, identities) + 1) / 42
        reliability = reliability[np.argsort(-identities)]
        settings.append({"seed":seed, "dev":scores, "configs":chosen, "winner":winner,
                         "source_estimates":reliability.tolist()})
        raw[f"{seed}-reference"] = linear_reference(table, test)
        for key, cfg in chosen.items():
            curve, mix, snapshots, direction = run_adapter(table, gate, test, cfg, donors)
            raw[f"{seed}-{key}"] = curve
            if key == "adaptive":
                raw[f"{seed}-mix"] = mix
                raw[f"{seed}-weights"] = snapshots
                raw[f"{seed}-features"] = table
                raw[f"{seed}-order"], raw[f"{seed}-labels"] = test[1],test[0]
                raw[f"{seed}-prior"] = gate
                # Reload using only weight snapshots and the encoder for every recorded budget.
                for j in range(len(BUDGETS)):
                    query = test[1][:, 18:]
                    logits = (snapshots[:, j, None] * table[query]).sum(-1)
                    expected = np.take_along_axis(test[0], query, axis=1)
                    assert np.array_equal(curve[:, j], ((logits >= 0) == expected).mean(1))
                for mode, outcomes in feedback(table, test, snapshots[:, -1], direction,
                                               cfg[1], reliability).items():
                    raw[f"{seed}-advice-{mode}"] = outcomes
    np.savez_compressed(ROOT / "change-trials.npz", **raw)
    (ROOT / "change-settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    summarise(raw)


def summarise(raw):
    report = {}
    for group, name in enumerate(GROUPS):
        sl = slice(192*group, 192*(group+1))
        def get(key):
            return np.stack([raw[f"{s}-{key}"][sl] for s in SEEDS])
        adaptive = get("adaptive")[:,:,1:].mean((1,2))
        result = {"accuracy":{}, "adaptive_minus":{}, "mix":get("mix").mean((0,1)).tolist(),
                  "reference":float(get("reference").mean()), "feedback":{}}
        for key in ("uniform","practiced","fixed","adam","adaptive","shuffled","winner"):
            curves = get(key)
            result["accuracy"][key] = curves.mean((0,1)).tolist()
            result["adaptive_minus"][key] = estimate(adaptive-curves[:,:,1:].mean((1,2)))
        pre = get("advice-none")[:,:,0,0]
        for mode in ("calibrated","common","swapped","ordinary","none"):
            out = get("advice-"+mode)
            stats = {}
            for j, (endpoint, weights) in enumerate([("harm",[.1,.45]),("repair",[.9,.55])]):
                eligible = pre if j==0 else ~pre
                events = (out[:,:,:,j] != pre[:,:,None]) & eligible[:,:,None]
                counts = events.sum(1)
                per_source = counts / eligible.sum(1)[:,None]
                stats[endpoint] = {"counts_by_source":counts.sum(0).tolist(),
                                   "eligible":int(eligible.sum()),
                                   "per_seed":np.average(per_source,axis=1,weights=weights).tolist()}
            result["feedback"][mode] = stats
        for endpoint in ("harm","repair"):
            result["feedback"]["calibrated_minus_common_"+endpoint] = estimate(np.subtract(
                result["feedback"]["calibrated"][endpoint]["per_seed"],
                result["feedback"]["common"][endpoint]["per_seed"]))
        report[name] = result
    for op in ("majority","conditional"):
        mix = np.stack([raw[f"{s}-mix"][:, -1].reshape(4,192).mean(1) for s in SEEDS])
        j = 0 if op == "majority" else 2
        report[op+"-recognition"] = estimate(mix[:,j]-mix[:,j+1])
    (ROOT / "change-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for name in GROUPS:
        r = report[name]
        print(name, 'adaptive minus winner', r["adaptive_minus"]["winner"]["mean"],
              'mix', r["mix"], 'reference',r["reference"])
        print('feedback',[(k,r["feedback"]["calibrated_minus_common_"+k]["mean"])
                          for k in ("harm","repair")])


if __name__ == "__main__":
    main()
