"""Forecast per-world failure rates before disambiguation on the slot learners."""

import json

import numpy as np

from decode import BINS, N, PERM, ROOT, WORLD, softmax
from study import TRAIN, TEST


def inputs(run):
    rare = int(run["color_probability"] > 0.5)
    words = np.asarray(run["words"])
    lookup = np.zeros((2 ** run["bits"], 2, N))
    lookup[words] = np.asarray(run["reader_logits"])
    pairs = WORLD[:, 0] * N + WORLD[:, 1]
    table = lookup[words[pairs]]
    odds = np.exp(run["log_odds"])
    c = np.take_along_axis(softmax(table[:, :, 0, :], 1), WORLD[:, 0, None, :], axis=2)
    s = np.take_along_axis(softmax(table[:, :, 1, :], 1), WORLD[:, 1, None, :], axis=2)
    now = (odds * c + s) / (1 + odds)
    current = np.concatenate([now[:, BINS, BINS], now.max(1), (now.argmax(1) == BINS)], 1)
    individual = (c, s)[rare]
    inside = np.concatenate(
        [individual[:, BINS, BINS], individual.max(1), individual.argmax(1) == BINS], 1
    )
    # Include the raw prefix and the permutation-invariant pair co-occurrences.
    triples = np.eye(N * N)[pairs]
    prefix = np.column_stack([triples.reshape(len(pairs), -1), triples.sum(1)])
    error = np.zeros(len(WORLD))
    rehearsal = None
    for order in PERM:
        c0, s0, c1, s1 = order[:4]
        lab = order[rare]
        target = order[2 + rare]
        wc, ws = odds * c[:, lab, c0, None], s[:, lab, s0, None]
        posterior = (wc * c[:, :, c1] + ws * s[:, :, s1]) / (wc + ws)
        correct = posterior.argmax(1) == target
        error += ~correct
        if rehearsal is None:
            rehearsal = np.column_stack([correct, posterior[:, target], posterior.max(1)])
    probes = []
    for c0, s0 in [(0, 1), (2, 3)]:
        lab = (c0, s0)[rare]
        wc, ws = odds * c[:, lab, c0, None], s[:, lab, s0, None]
        for c1 in BINS:
            s1 = (c1 + 1) % N
            target = (c1, s1)[rare]
            posterior = (wc * c[:, :, c1] + ws * s[:, :, s1]) / (wc + ws)
            probes.extend([posterior.argmax(1) == target, posterior[:, target], posterior.max(1)])
    return prefix, current, inside, rehearsal, np.column_stack(probes), error / len(PERM)


def estimate(x, target, ids, test):
    mu = x[ids].mean(0)
    sd = x[ids].std(0)
    sd[sd < 1e-8] = 1
    x = (x - mu) / sd
    a = np.column_stack([np.ones(len(x)), x])
    train = a[ids]
    w = np.linalg.solve(train.T @ train + np.eye(a.shape[1]), train.T @ target[ids])
    return (a[test] @ w).clip(0, 1)


def main():
    folder = ROOT / "confirmation"
    ids, test = TRAIN[:2048].numpy(), TEST.numpy()
    scores = {}
    for path in sorted(folder.glob("fork-4-*.json")):
        run = json.loads(path.read_text())
        if run["color_probability"] == 0.5:
            continue
        prefix, now, inside, rehearsal, probes, y = inputs(run)
        ext = np.column_stack([prefix, now])
        preds = {"mean": np.full(len(test), y[ids].mean())}
        for name, x in [
            ("prefix", prefix),
            ("current", ext),
            ("rehearsal", np.column_stack([ext, rehearsal])),
            ("ten_probes", np.column_stack([ext, probes])),
            ("internal", np.column_stack([ext, inside])),
        ]:
            preds[name] = estimate(x, y, ids, test)
        mae = {p: float(np.abs(v - y[test]).mean()) for p, v in preds.items()}
        scores[path.stem] = dict(
            mae=mae,
            worst_change=int((y[test] * 120).round().max()),
            risk_counts=np.rint(120 * y[test]).astype(int).tolist(),
            forecast={p: v.round(6).tolist() for p, v in preds.items()},
        )
        print(path.stem, {p: round(100 * v, 2) for p, v in mae.items()})
    (folder / "monitor.json").write_text(
        json.dumps(dict(worlds=test.tolist(), runs=scores), separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
