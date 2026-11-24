"""Causal prefix-swap audit using only the saved binary codebooks and trained readers."""

import json

import numpy as np

from decode import ROOT, WORLD, softmax


def responses(reader, odds, memory, keys, clues):
    table = reader[memory]
    scores = []
    for axis in (0, 1):
        pair = table[:, axis, keys[axis]].T
        exp = np.exp(pair - pair.max(1, keepdims=True))
        scores.append(exp / exp.sum(1, keepdims=True))
    color, shape = scores
    return np.array(
        [
            (np.exp(odds) * color[0, clue] * color[1] + shape[0, clue] * shape[1]).argmax()
            for clue in clues
        ]
    )


def main():
    from study import TEST

    directory = ROOT / "confirmation"
    episodes = json.loads((directory / "episodes.json").read_text())
    lookup = {tuple(world.flatten()): i for i, world in enumerate(WORLD)}
    test = set(TEST.tolist())
    results = {}
    for path in sorted(directory.glob("fork-4-*.json")):
        run = json.loads(path.read_text())
        words = np.array(run["words"])
        logits = np.array(run["reader_logits"])
        reader = np.zeros((2 ** run["bits"], 2, 5))
        reader[words] = logits
        assert np.max(np.abs(reader[words] - logits)) < 0.00001  # CPU kernels differ by a few ULPs
        odds = run["log_odds"]
        trials = {"target": [], "control": []}
        for row, (idx, b) in enumerate(zip(episodes["ids"], episodes["bins"])):
            prefix = WORLD[idx]
            (unseen,) = set(range(5)) - set(b)
            original = words[prefix[0] * 5 + prefix[1]]
            keys = ([prefix[0, b[0]], prefix[0, b[2]]], [prefix[1, b[1]], prefix[1, b[3]]])
            pred = responses(reader, odds, original, keys, b[:2])
            assert np.array_equal(pred, run["predictions"][row])
            for kind, source in [("target", b[3]), ("control", b[2])]:
                locations = [source, unseen]
                donor = prefix.copy()
                donor[1, locations] = donor[1, locations[::-1]]
                donor_id = lookup[tuple(donor.flatten())]
                if donor_id not in test:
                    continue
                patched = original.copy()
                patched[locations] = words[donor[0, locations] * 5 + donor[1, locations]]
                assert np.array_equal(patched, words[donor[0] * 5 + donor[1]])
                after = responses(reader, odds, patched, keys, b[:2])
                trials[kind].append(
                    dict(
                        row=row,
                        donor=donor_id,
                        invisible=bool(np.array_equal(original, patched)),
                        color_changed=bool(pred[0] != after[0]),
                        shape_changed=bool(pred[1] != after[1]),
                        original_shape_correct=bool(pred[1] == b[3]),
                        followed=bool(pred[1] == b[3] and after[1] == unseen),
                    )
                )
        results[path.stem] = trials
    (directory / "intervention.json").write_text(json.dumps(results, separators=(",", ":")) + "\n")
    eligible_now = {}
    for seed in (44, 55, 66):
        scored = {}
        ids = np.array(episodes["ids"])
        world = WORLD[ids]
        for bias in (125, 500, 875):
            run = json.loads((directory / f"fork-4-{bias}-{seed}.json").read_text())
            table = np.asarray(run["reader_logits"])[5 * world[:, 0] + world[:, 1]]
            odds = np.exp(run["log_odds"])
            color = softmax(table[:, :, 0, :], 1)
            shape = softmax(table[:, :, 1, :], 1)
            logits = odds * np.take_along_axis(color, world[:, 0, None, :], axis=2) + np.take_along_axis(
                shape, world[:, 1, None, :], axis=2
            )
            scored[bias] = (logits.argmax(1) == np.arange(5)).all(1)
        mask = np.logical_and.reduce(list(scored.values()))
        target = np.array(episodes["bins"])[:, 2:]
        group = {"episodes": np.flatnonzero(mask).tolist(), "accuracy": {}}
        for bias in scored:
            run = json.loads((directory / f"fork-4-{bias}-{seed}.json").read_text())
            hit = np.asarray(run["predictions"])[mask] == target[mask]
            group["accuracy"][bias] = hit.mean(0).tolist()
        eligible_now[seed] = group
        print("all correct before clue", seed, int(mask.sum()), group["accuracy"])
    (directory / "matched-current-behavior.json").write_text(
        json.dumps(eligible_now, separators=(",", ":")) + "\n"
    )
    for bias in (125, 500, 875):
        for kind in ("target", "control"):
            rows = [r for k, v in results.items() if f"-{bias}-" in k for r in v[kind]]
            print(
                bias,
                kind,
                len(rows) // 3,
                {
                    key: round(100 * np.mean([r[key] for r in rows]), 1)
                    for key in ("invisible", "color_changed", "shape_changed", "followed")
                },
            )


if __name__ == "__main__":
    main()
