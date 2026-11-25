"""Learn a pixel update and a decision to replace an obsolete rule."""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BUDGETS = (1, 2, 4, 8)
BASELINES = ("old", "candidate", "wait", "prior20", "prior50", "prior80")
LEARNERS = ("change", "exception", "change_gate", "change_pixels")
SEEDS = range(721, 729)
HISTORIES = (.8, .2)


def images():
    path = Path(importlib.util.find_spec("sklearn").origin).parent / "datasets/data/digits.csv.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"
    rows = np.loadtxt(path, delimiter=",")
    x, y = rows[:, :-1], rows[:, -1].astype(int)
    x -= x[y < 6].mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.tensor(x.tolist(), dtype=torch.float32), [np.flatnonzero(y == d) for d in range(10)]


def episodes(x, banks, rng, pairs):
    n = len(pairs)
    support, query, support_bit = [], [], rng.integers(2, size=(n, 8))
    query_bit = np.tile(np.repeat([0, 1], 16), (n, 1))
    for i, pair in enumerate(pairs):
        perm = pair[::-1] if rng.integers(2) else pair
        ids = np.empty(8, int)
        q = []
        for bit, digit in enumerate(perm):
            where = support_bit[i] == bit
            draw = rng.choice(banks[digit], size=int(where.sum()) + 16, replace=False)
            ids[where] = draw[16:]
            q.extend(draw[:16])
        assert not set(ids).intersection(q)
        support.append(ids.tolist())
        query.append(q)
    mark = rng.integers(2, size=(n, 8))
    mark[:, 0] = 1 - support_bit[:, 0]
    query_mark = rng.integers(2, size=(n, 32))
    # Always use the same opening report in both worlds.
    return (x[torch.tensor(support)], torch.Tensor(support_bit.tolist()), torch.Tensor(mark.tolist()),
            x[torch.tensor(np.asarray(query).tolist())], torch.Tensor(query_bit.tolist()), torch.Tensor(query_mark.tolist()))


def weights(data, regime, log_step, gate, method):
    sx, identity, mark, _, _, _ = data
    n = len(sx)
    label = identity if regime == 0 else torch.cat([identity[:, :1], mark[:, 1:]], dim=1)
    candidate = torch.zeros_like(sx[:, 0])
    conflict = torch.zeros(n)
    evidence = torch.zeros(n)
    out = []
    for j in range(8):
        y, x = label[:, j], sx[:, j]
        before = (candidate*x).sum(1)
        if j:
            conflict = conflict + (y != mark[:, j]).float()
            evidence = evidence + F.logsigmoid((2*y-1)*before) + np.log(2.)
        candidate = candidate + log_step.exp()*(y-before.sigmoid())[:, None]*x
        if j+1 not in BUDGETS:
            continue
        if method == "learned":
            mix = (gate[0] + gate[1]*(j-conflict) + gate[2]*conflict + gate[3]*evidence).sigmoid()
        elif method == "old":
            mix = torch.zeros(n)
        elif method == "candidate":
            mix = torch.ones(n)
        elif method == "wait":
            mix = (conflict > 0).float()
        else:
            prior = int(method[5:])/100
            p = prior / (prior+(1-prior)*2**j)
            mix = torch.where(conflict > 0, 1., p)
        out.append(torch.cat([mix[:, None]*candidate, 3*(1-mix[:, None])], dim=1))
    return torch.stack(out, 1)


def query_features(data):
    return torch.cat([data[3], (2*data[5]-1)[:, :, None]], dim=-1)


def predict(data, log_step, gate, method):
    query = query_features(data)
    return torch.stack([weights(data, regime, log_step, gate, method) @ query.transpose(1, 2)
                        for regime in (0, 1)], dim=1)


def train(x, banks, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    start = (torch.randn(64)*.03 + np.log(4.), torch.tensor([0., -.3, .3, .1]))
    learners = [[p.clone().requires_grad_() for p in start] for _ in HISTORIES]
    opts = [torch.optim.Adam(params, lr=.045) for params in learners]
    pairs = list(itertools.combinations(range(6), 2))
    for _ in range(100):
        data = episodes(x, banks, rng, [pairs[i] for i in rng.integers(len(pairs), size=24)])
        target = torch.stack([data[4], data[5]], dim=1)[:, :, None]
        for params, opt, prob in zip(learners, opts, HISTORIES):
            z = predict(data, *params, "learned")
            loss = F.softplus((1-2*target)*z).mean((0, 2, 3))
            opt.zero_grad()
            (prob*loss[0] + (1-prob)*loss[1]).backward()
            opt.step()
    return [[v.detach() for v in model] for model in learners]


def metrics(data, z):
    pred = z >= 0
    target = torch.stack([data[4], data[5]], dim=1)[:, :, None].bool()
    correct = pred == target
    wrong_old = data[4] != data[5]
    repaired = (correct[:, 0] & wrong_old[:, None]).sum(-1)
    eligible = wrong_old.sum(-1)[:, None].expand_as(repaired)
    # Every query was correct under the old rule in the stable fork.
    return np.array(torch.stack([correct[:, 0].sum(-1), correct[:, 1].sum(-1),
                                 repaired, eligible], dim=-1).tolist(), dtype=np.uint8)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    x, banks = images()
    dev_pairs = list(itertools.combinations(range(6), 2))
    dev = episodes(x, banks, np.random.default_rng(698), dev_pairs*12)
    pairs = list(itertools.combinations(range(6, 10), 2))
    evaluations = [episodes(x, banks, np.random.default_rng(7810+i), [pair]*192) for i, pair in enumerate(pairs)]
    baseline_steps, dev_scores, raw, checkpoints = {}, {}, {}, {}
    for method in BASELINES:
        scores = []
        for step in (1, 4, 16, 64):
            counts = metrics(dev, predict(dev, torch.tensor(float(np.log(step))), None, method))
            scores.append(counts[:, :, :2].mean()/32)
        k = np.argmax(scores)
        baseline_steps[method], dev_scores[method] = (1,4,16,64)[k], scores[k]
        raw[method] = np.stack([metrics(d, predict(d, torch.tensor(float(np.log(baseline_steps[method]))), None, method))
                                for d in evaluations])
    for seed in SEEDS:
        change, stable = train(x, banks, seed)
        for name, (pixel, gate) in zip(LEARNERS, (change, stable, (stable[0], change[1]), (change[0], stable[1]))):
            scores = []
            for d in evaluations:
                z = predict(d, pixel, gate, "learned")
                scores.append(metrics(d, z))
            raw.setdefault(name, []).append(scores)
            checkpoints[f"{seed}/{name}/gate"] = gate.tolist()
            if name in LEARNERS[:2]:
                checkpoints[f"{seed}/{name}/log_pixel_step"] = pixel.tolist()
            if seed == SEEDS[0] and name == "change":
                d = tuple(v[:1] for v in evaluations[0])
                checkpoints["written"] = weights(d, 0, pixel, gate, "learned")[0, -1].tolist()
                checkpoints["queries"] = query_features(d)[0].tolist()
                checkpoints["logits"] = predict(d, pixel, gate, "learned")[0, 0, -1].tolist()
        print("completed paired seed", seed, flush=True)
    for name in raw:
        a = np.array(raw[name])
        assert (a <= 32).all() and (a[..., 2] <= a[..., 3]).all()
        first = a[..., 0, :].astype(int)
        # Identical first predictions imply this exact error accounting across forks.
        np.testing.assert_array_equal(first[..., 0], first[..., 1]-first[..., 3]+2*first[..., 2])
    for params in (change, stable):
        z = predict(evaluations[0], *params, "learned")
        torch.testing.assert_close(z[:, 0, 0], z[:, 1, 0])
    np.savez_compressed(ROOT/"trials.npz", **raw)
    meta = dict(budgets=BUDGETS, seeds=list(SEEDS), pairs=pairs, baseline_steps=baseline_steps,
                validation_balanced_accuracy=dev_scores, selected_baseline=max(dev_scores, key=dev_scores.get),
                columns=["change_correct", "stable_correct", "true_repaired", "old_errors"], checkpoints=checkpoints)
    (ROOT/"weights.json").write_text(json.dumps(meta, separators=(",", ":"), allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
