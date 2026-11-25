"""Fit a policy-change prior from audit histories and intervene on later weight updates."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
from itertools import product
from pathlib import Path

import numpy as np
from prepare import CACHE

ROOT = Path(__file__).resolve().parent
MASK = np.array(list(product((0., 1.), repeat=6)))
LEFT, RIGHT = np.triu_indices(6, k=1)
FEATURE = np.column_stack([MASK, MASK[:, LEFT] == MASK[:, RIGHT]])
METHODS = ("independent", "coupled", "other-history", "empirical", "mixture", "oracle")
BUDGETS = (1, 2, 4, 8)
PAIRS = np.arange(6).reshape(3, 2)
OTHER = np.array([[1, 2], [3, 4], [5, 0]])


def normalize(x):
    return x / x.sum(axis=-1, keepdims=True)


def softmax(x):
    return normalize(np.exp(x - x.max(axis=-1, keepdims=True)))


def fit(history):
    rate = (history.sum(0) + 1) / (len(history) + 2)
    offset = np.log(rate / (1 - rate))
    independent = softmax(MASK @ offset)
    theta = np.r_[offset, np.zeros(15)]
    target = np.r_[history.mean(0), (history[:, LEFT] == history[:, RIGHT]).mean(0)]
    for _ in range(500):
        expected = softmax(FEATURE @ theta) @ FEATURE
        theta += .5 * (target - expected - .008 * theta)
        theta[6:] = np.maximum(theta[6:], 0)
    return independent, softmax(FEATURE @ theta), theta


def make_priors(history, pairs):
    independent, coupled, theta = fit(history)
    swapped = np.empty_like(history)
    swapped[:, OTHER.ravel()] = history[:, PAIRS.ravel()]
    other = fit(swapped)[1]
    powers = 2 ** np.arange(5, -1, -1)
    counts = np.bincount((history @ powers).astype(int), minlength=64)
    empirical = normalize(counts + 2 * independent)
    rate = history.mean()
    allowed = (MASK[:, pairs[:, 0]] == MASK[:, pairs[:, 1]]).all(axis=1)
    n = MASK.sum(1) / 2
    oracle = normalize(allowed * rate ** n * (1 - rate) ** (3 - n))
    return np.stack([independent, coupled, other, empirical, .5*(coupled+independent), oracle]), theta


def views(features, split, labels, names, rng):
    def draw(part, size):
        return np.stack([rng.choice(np.flatnonzero((split == part) & (labels == name)),
                                    size, replace=False) for name in names])
    centers = features[draw("train", 32)].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    valid = features[draw("val", 15)] @ centers.T
    temp = min((.025, .05, .1, .2), key=lambda t:
               -np.log(softmax(valid / t)[np.arange(6), :, np.arange(6)]).mean())
    indices = draw("test", 26)
    assert len(np.unique(indices)) == indices.size
    probs = softmax(features[indices] @ centers.T / temp)
    return probs[:, :12], probs[:, 12:], indices, temp


def capability(rng, priors, pairs, support, queries):
    n = 96
    output = np.empty((3, len(METHODS), len(BUDGETS)))
    for changes in range(3):
        chosen = rng.random((n, 3)).argsort(axis=1)[:, :changes]
        flips = np.zeros((n, 6), bool)
        flips[np.arange(n)[:, None, None], pairs[chosen]] = True
        old = rng.random((n, 6)).argsort(1) < 3
        truth = old ^ flips
        possible = (old[:, None, :] != MASK[None, :, :]).astype(float)
        post = np.broadcast_to(priors, (n, *priors.shape)).copy()
        for t in range(8):
            intent = rng.integers(6, size=n)
            x = support[intent, t]
            label = truth[np.arange(n), intent] ^ (rng.random(n) < .1)
            p = .1 + .8 * np.einsum("nd,nkd->nk", x, possible)
            post = normalize(post * np.where(label[:, None], p, 1-p)[:, None, :])
            if t+1 in BUDGETS:
                heads = post @ possible
                pred = np.einsum("dqf,nmf->nmdq", queries, heads) >= .5
                output[changes, :, BUDGETS.index(t+1)] = (pred == truth[:,None,:,None]).mean(axis=(0,2,3))
    return output


def feedback(rng, priors, support, queries):
    totals = np.zeros((4, 2, len(METHODS)))
    eligible = 0
    example = None
    for trial in range(120):
        old = rng.random(6).argsort() < 3
        trigger = trial % 6
        peer = trigger ^ 1
        possible = (old != MASK).astype(float)
        # Both forks receive the same report about the same request.
        report = ~old[trigger]
        x = support[trigger, trial % 12]
        prob_report_true = x @ possible.T
        if not report:
            prob_report_true = 1 - prob_report_true
        first = .1 + .8 * prob_report_true
        # Independent raters share the latent intent when labeling the very same text.
        twice = .9**2 * prob_report_true + .1**2 * (1-prob_report_true)
        verified_x = support[peer, (trial+1) % 12]
        p_peer = verified_x @ possible.T
        eligible_query = ((queries[peer] @ old >= .5) == old[peer])
        eligible += eligible_query.sum()
        for branch, truth in enumerate((~old[peer], old[peer])):
            verified = p_peer if truth else 1-p_peer
            for j,likelihood in enumerate((first, first, twice, twice*verified)):
                heads = normalize(priors * likelihood) @ possible
                changed = (heads @ queries[peer].T >= .5) != old[peer]
                totals[j,branch] += changed[:,eligible_query].sum(1)
                if trial == 0 and branch == 0 and j == 3:
                    example = dict(old=old.tolist(), trigger=trigger, peer=peer,
                                   report=bool(report), head=heads[1].tolist(),
                                   prediction=(heads[1] @ queries[peer].T >= .5).tolist())
    assert eligible and np.array_equal(totals[0], totals[1])
    assert np.array_equal(totals[:3,0], totals[:3,1])
    return totals / eligible, eligible, example


def interval(values):
    rng = np.random.default_rng(732)
    boot = rng.choice(values, size=(8000, len(values))).mean(1)
    return [float(values.mean()), *np.quantile(boot, [.025,.975]).tolist()]


def main():
    with np.load(CACHE / "embedded.npz") as file:
        features, split, labels = (file[k] for k in ("features", "split", "label"))
    domains = json.loads((CACHE / "domains.json").read_text())
    scores, feedback_scores, records = [], [], []
    for i,seed in enumerate(range(901,921)):
        rng = np.random.default_rng(seed)
        domain = list(domains)[i % 10]
        names = rng.choice(domains[domain], 6, replace=False)
        support, queries, indices, temp = views(features, split, labels, names, rng)
        history = np.zeros((48,6))
        times = np.flatnonzero(rng.random(48) >= .4)
        history[times[:,None], PAIRS[rng.integers(3,size=len(times))]] = 1
        scores.append([])
        for pairs in (PAIRS, OTHER):
            priors, theta = make_priors(history, pairs)
            scores[-1].append(capability(rng, priors, pairs, support, queries))
        fs, count, example = feedback(rng, make_priors(history, PAIRS)[0], support, queries)
        feedback_scores.append(fs)
        records.append(dict(seed=seed, domain=domain, intents=names.tolist(), temperature=temp,
                            test_indices=indices.tolist(), history=history.astype(int).tolist(),
                            theta=theta.tolist(), eligible=int(count), example=example))
    scores, feedback_scores = np.array(scores), np.array(feedback_scores)
    np.savez_compressed(ROOT/"policy-change-trials.npz", accuracy=scores, feedback=feedback_scores)
    report = dict(methods=METHODS, budgets=BUDGETS, accuracy=scores.mean(0).tolist(),
                  feedback=feedback_scores.mean(0).tolist(), records=records, contrasts={})
    for g, regime in enumerate(("familiar","changed-pairing")):
        for changes in range(3):
            vals = scores[:,g,changes].mean(-1)
            for m,method in enumerate(METHODS):
                report["contrasts"][f"{regime}-{changes}-{method}-minus-independent"] = interval(vals[:,m]-vals[:,0])
            report["contrasts"][f"{regime}-{changes}-coupled-minus-other-history"] = interval(vals[:,1]-vals[:,2])
    for j, evidence in enumerate(("one","duplicate","independent","verified")):
        for k, outcome in enumerate(("repair","harm")):
            report["contrasts"][evidence+"-"+outcome] = interval(feedback_scores[:,j,k,1]-feedback_scores[:,j,k,0])
    (ROOT/"policy-change.json").write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
