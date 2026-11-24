"""Replay the trained reader, enforce a bijection, and audit new episodes."""

import collections
import json
import pathlib

import numpy as np

from audit import N, PERMS, WORLDS

ROOT = pathlib.Path(__file__).parent / "results"
PERM = np.asarray(PERMS)
WORLD = np.asarray(WORLDS)
BINS = np.arange(N)


def softmax(array, axis):
    exp = np.exp(array - array.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def predict(logits, odds, ids, bins, joint):
    # dim order: episode, bin, axis, key
    world = WORLD[ids]
    pair = N * world[:, 0] + world[:, 1]
    scores = logits[pair]
    if joint:
        marg = np.empty_like(scores)
        for axis in (0, 1):
            # One legal permutation gives one key to each bin.
            joint_scores = scores[:, BINS, axis, PERM].sum(-1)
            weights = softmax(joint_scores, 1)
            for b in BINS:
                for k in BINS:
                    marg[:, b, axis, k] = weights[:, PERM[:, b] == k].sum(1)
        prob = marg
    else:
        prob = softmax(scores, 1)
    rows = np.arange(len(ids))

    def likelihood(axis, column):
        key = world[rows, axis, bins[:, column]]
        return prob[rows, :, axis, key]

    cc, sc = likelihood(0, 0), likelihood(1, 1)
    cq, sq = likelihood(0, 2), likelihood(1, 3)
    result = []
    for axis in (0, 1):
        label = bins[:, axis]
        # Within a feature, key-to-bin probabilities sum to one.
        wc, ws = np.exp(odds) * cc[rows, label], sc[rows, label]
        result.append((wc[:, None] * cq + ws[:, None] * sq).argmax(1))
    return np.stack(result, 1)


def optimal_uncued_sampling(words, ids, bins):
    # Query keys are independent of the clue here. No exclusion of destinations.
    states = np.asarray(words)[WORLD[:, 0] * N + WORLD[:, 1]]
    groups = collections.defaultdict(list)
    for i, state in enumerate(states):
        groups[tuple(state)].append(i)
    inv = WORLD.argsort(2)
    out = []
    for i, b in zip(ids, bins):
        candidates = inv[groups[tuple(states[i])]]
        c, s = WORLD[i]
        cb = candidates[:, 0, c[b[0]]]
        sb = candidates[:, 1, s[b[1]]]
        possible = cb != sb
        cq = candidates[:, 0, c[b[2]]]
        sq = candidates[:, 1, s[b[3]]]
        out.append(
            [
                np.bincount(
                    np.concatenate([cq[possible & (cb == lab)], sq[possible & (sb == lab)]])
                ).argmax()
                for lab in b[:2]
            ]
        )
    return np.array(out)


def main():
    # Use the frozen torch world split; derive fresh queries independently in NumPy.
    from study import TEST

    rng = np.random.default_rng(2401)
    ids = rng.choice(TEST.tolist(), size=1024)
    disjoint = PERM[rng.integers(len(PERM), size=len(ids)), :4]
    independent = disjoint.copy()
    independent[:, 2:] = rng.integers(N, size=(len(ids), 2))
    subset = np.array([len(set(row)) == 4 for row in independent])
    records = {}
    for folder in (ROOT, ROOT / "confirmation"):
        old = json.loads((folder / "episodes.json").read_text())
        for path in sorted(folder.glob("fork-*.json")):
            r = json.loads(path.read_text())
            logits = np.asarray(r["reader_logits"])
            # BLAS may differ by a few ULPs across rows. A single lookup per binary
            # word ensures the original pair index cannot reach this decoder.
            words = np.asarray(r["words"])
            canonical = np.zeros((2 ** r["bits"], 2, N))
            canonical[words] = logits
            assert np.max(np.abs(canonical[words] - logits)) < 1e-5
            logits = canonical[words]
            odds = r["log_odds"]
            # Float32 near ties can change argmax; require exact old decisions for every row.
            old_pred = predict(logits, odds, np.array(old["ids"]), np.array(old["bins"]), False)
            assert np.array_equal(old_pred, r["predictions"]), path
            out = {}
            for b, name in ((disjoint, "disjoint"), (independent, "independent")):
                predictions = {
                    p: predict(logits, odds, ids, b, p == "joint") for p in ("model", "joint")
                }
                if name == "independent":
                    predictions["optimal"] = optimal_uncued_sampling(r["words"], ids, b)
                out[name] = {p: v.tolist() for p, v in predictions.items()}
            records[path.stem] = out
    data = dict(
        ids=ids.tolist(),
        independent_bins=independent.tolist(),
        disjoint_bins=disjoint.tolist(),
        independent_disjoint_subset=int(sum(subset)),
        runs=records,
    )
    (ROOT / "decoder-control.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    for bits in (4, 6):
        for bias in (125, 500, 875):
            runs = [v for k, v in records.items() if k.startswith(f"fork-{bits}-{bias}-")]
            for selection in ("all", "four distinct"):
                mask = slice(None) if selection == "all" else subset
                summary = {}
                for p in ("model", "joint", "optimal"):
                    hit = np.array([np.asarray(r["independent"][p]) == independent[:, 2:] for r in runs])
                    summary[p] = (100 * hit[:, mask, :].mean(axis=(0, 1))).round(1).tolist()
                print(bits, bias, selection, summary)


if __name__ == "__main__":
    main()
