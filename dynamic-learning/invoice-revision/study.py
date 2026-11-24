"""Fit the preregistered CPU study. Run with OPENBLAS_NUM_THREADS=1."""
from pathlib import Path

import numpy as np

from ledger import (FEATURES, ORDERS, PATH_ID, PATHS, QUERIES, SCOPES, generate,
                    posterior, receipts, relations, restore, store_state)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
POLICIES = ('fixed', 'random', 'learned', 'oracle')
WRITERS = ('explicit', 'dense', 'sparse')
SEEDS = range(4101, 4109)


def training_counts(data):
    counts = np.zeros((len(PATHS), 4))
    likelihood = np.ones((4, 3, 2))
    for j, order in enumerate(data['menus']):
        for w in range(4):
            path = receipts(data, j, w, order[:2])
            for q, bit in path:
                likelihood[w, q, bit] += 1
            for b in range(3):
                counts[PATH_ID[path[:b]], w] += 1
    return counts, likelihood[:, :, 1] / likelihood.sum(axis=2)


def best_fixed(dev):
    def grade(order):
        scores, errors = [], []
        for w in range(4):
            for budget in range(3):
                path = receipts(dev, 0, w, order[:budget])
                p = posterior(path, SCOPES)
                scores.append(int(p.argmax() == w))
                errors.append(((p - np.eye(4)[w])**2).sum())
        return sum(scores), -round(sum(errors), 10)
    # All development families have identical probe outcomes by role.
    assert np.all(np.take_along_axis(dev['truth'], dev['roles'][:, None, :3], axis=2)
                  == SCOPES[None, :, :3])
    return max(ORDERS, key=grade)


def entropy(p):
    pos = p > 0
    return -(p[pos] * np.log(p[pos])).sum()


def actions(policy, menu, world, likelihood, fixed):
    path = ()
    for step in range(2):
        if policy == 'oracle':
            q = 1 if not path else (2 if path[0][1] else 0)
        elif policy == 'fixed':
            q = fixed[step]
        elif policy == 'random':
            q = menu[step]
        else:
            p = posterior(path, likelihood)
            def risk(q):
                total = 0
                for probs in (likelihood[:, q], 1 - likelihood[:, q]):
                    joint = p * probs
                    total += joint.sum() * entropy(joint / joint.sum())
                return round(total, 12)
            q = min((a for a in menu if all(a != old for old, _ in path)), key=risk)
        path += ((q, int(SCOPES[world, q])),)
    return path


def fit(train, dev, seed, name):
    mask = np.ones((6, 3)) if name == 'dense' else np.eye(3).repeat(2, axis=0)
    def targets(counts):
        n = counts.sum(axis=1, keepdims=True)
        return counts @ SCOPES / np.maximum(1, n), n / n.sum() / 4
    y, freq = targets(train)
    val, val_freq = targets(dev)
    rng = np.random.default_rng(seed)
    initial = [rng.normal(0, .1, (6, 3)), rng.normal(0, .1, (4, 3)), np.zeros(4)]
    best = None
    for rate in (.01, .03):
        table, read, bias = params = [a.copy() for a in initial]
        moments = [np.zeros_like(a) for a in params]
        variances = [np.zeros_like(a) for a in params]
        for step in range(1, 401):
            states = FEATURES @ (table * mask)
            logits = states @ read.T + bias
            err = (1 / (1 + np.exp(-logits)) - y) * freq
            grads = [FEATURES.T @ (err @ read) * mask, err.T @ states, err.sum(axis=0)]
            for a, m, v, grad in zip(params, moments, variances, grads):
                m *= .9
                m += .1 * grad
                v *= .999
                v += .001 * grad**2
                a -= rate * (m / (1 - .9**step)) / (np.sqrt(v / (1 - .999**step)) + 1e-8)
        logits = FEATURES @ (table * mask) @ read.T + bias
        loss = ((np.logaddexp(0, logits) - val * logits) * val_freq).sum()
        if best is None or loss < best[0]:
            best = loss, rate, (table * mask).astype('f4'), read.astype('f4'), bias.astype('f4')
    return best


def run():
    dev = generate(20261125, 256)
    fixed = best_fixed(dev)
    dev_counts, _ = training_counts(dev)
    fits = []
    for seed in SEEDS:
        counts, likelihood = training_counts(generate(seed, 256))
        learned = [fit(counts, dev_counts, seed, name) for name in WRITERS[1:]]
        fits.append((likelihood, learned))
    # Fit and development decisions are complete before generating test ledgers.
    data = generate(20261126, 512)
    n = len(data['rows'])
    menu_index = [ORDERS.index(tuple(x)) for x in data['menus']]
    trace = np.empty((8, 4, 3, n, 4), dtype=np.uint8)
    probs = np.empty((8, 4, len(PATHS), 4))
    votes = np.zeros((8, 3, len(PATHS), 4), bool)
    lengths = np.zeros((8, 3, len(PATHS)), dtype=np.uint8)
    blobs = np.zeros((8, 3, len(PATHS), 12), dtype=np.uint8)
    params = []
    for f, (likelihood, learned) in enumerate(fits):
        params.append(np.array([np.r_[loss, rate, t.ravel(), r.ravel(), b]
                                for loss, rate, t, r, b in learned]))
        for pi, policy in enumerate(POLICIES):
            like = likelihood if policy == 'learned' else SCOPES
            probs[f, pi] = [posterior(path, like) for path in PATHS]
            by_menu = np.empty((6, 3, 4), dtype=np.uint8)
            for m, menu in enumerate(ORDERS):
                for w in range(4):
                    path = actions(policy, menu, w, likelihood, fixed)
                    for budget in range(3):
                        by_menu[m, budget, w] = PATH_ID[path[:budget]]
            trace[f, pi] = by_menu[menu_index].transpose(1, 0, 2)
        for wi, name in enumerate(WRITERS):
            t, r, b = (None, None, None) if not wi else learned[wi-1][2:]
            for i, path in enumerate(PATHS):
                if not posterior(path, SCOPES).any():
                    continue
                blob = store_state(name, 0, path, t)
                lengths[f, wi, i] = len(blob)
                blobs[f, wi, i, :len(blob)-2] = np.frombuffer(blob[2:], dtype='u1')
                _, votes[f, wi, i] = restore(name, blob, r, b)
    # Audit compact representation against actual per-ledger receipts and reads.
    for j, rows in enumerate(data['rows']):
        role = data['roles'][j]
        anchor = int(rows[role[0], 0])
        rel = relations(rows, anchor)
        assert np.array_equal(rel[role[QUERIES]], [1, 2, 3, 0, 1, 2, 3])
        for i in np.unique(trace[:, :, :, j]):
            path = PATHS[i]
            for w in range(4):
                if (trace[:, :, :, j, w] == i).any():
                    assert receipts(data, j, w, [q for q, _ in path]) == path
            for wi, name in enumerate(WRITERS):
                f = j % 8
                t, r, b = (None, None, None) if not wi else fits[f][1][wi-1][2:]
                blob = store_state(name, anchor, path, t)
                restored_id, choice = restore(name, blob, r, b)
                assert restored_id == anchor and np.array_equal(choice, votes[f, wi, i])
                assert len(blob) == lengths[f, wi, i]
                assert blob[2:] == blobs[f, wi, i, :len(blob)-2].tobytes()
    assert np.all(trace[:, 3, 2] == np.array([PATH_ID[actions('oracle', ORDERS[0], w, SCOPES, fixed)]
                                               for w in range(4)]))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'trials.npz', **data, trace=trace, probs=probs, votes=votes,
                        lengths=lengths, blobs=blobs, params=params,
                        likelihood=[x[0] for x in fits], fixed=fixed)


if __name__ == '__main__':
    run()
