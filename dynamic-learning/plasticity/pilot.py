"""Run the preregistered CPU study and save the paired trials and gate trajectories."""
import itertools
import json
import os
from pathlib import Path
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(41, 49))
CHECKPOINTS = (0, 1, 10, 100)
BUDGETS = (0, 1, 2, 4, 8)
BITS = np.array(list(itertools.product((-1, 1), repeat=6)))
TRUTH = np.column_stack([BITS, BITS * np.roll(BITS, -1, axis=1)])
COMMON = np.array([0, 1, 2, 6, 7, 8])
OTHER = np.setdiff1d(np.arange(12), COMMON)
TRIPLES = np.array(list(itertools.combinations(COMMON, 3)))
TRIPLES = TRIPLES[np.random.default_rng(331).permutation(len(TRIPLES))]
TRAIN, DEV, TEST = np.split(TRIPLES, [12, 16])
SHIFT = np.array(list(itertools.combinations(OTHER, 3)))
RATES = (.1, .3, 1., 3.)
BASE_RATE = .3


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(2, 12)
        self.position = nn.Parameter(torch.randn(6, 12))
        self.block = nn.TransformerEncoderLayer(12, 2, 24, dropout=0., batch_first=True)
        self.readout = nn.Sequential(nn.Flatten(), nn.Linear(72, 48), nn.Tanh(),
                                     nn.Linear(48, 12), nn.Tanh())

    def forward(self, x):
        return self.readout(self.block(self.token(x) + self.position))


def feature_table(seed):
    torch.manual_seed(seed)
    model = Encoder()
    x = torch.tensor(((BITS + 1) // 2).tolist())
    y = torch.tensor(TRUTH.tolist(), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=.008)
    for _ in range(350):
        opt.zero_grad()
        (model(x) - y).square().mean().backward()
        opt.step()
    with torch.no_grad():
        table = np.asarray(model(x).tolist())
    return table, sum(p.numel() for p in model.parameters())


def tools(rng, triples, n, paired_probe=False):
    selected = triples[rng.integers(len(triples), size=n)]
    sign = rng.choice([-1, 1], size=(n, 3))
    votes = (TRUTH[:, selected] * sign).sum(-1).T
    labels = (votes > 0).astype(float)
    order = rng.random((n, len(BITS))).argsort(axis=1)
    if paired_probe:
        # Reserve a distinct input with identical executable features for the feedback probe.
        for row, ids in enumerate(selected):
            remaining = order[row, 8:].copy()
            features = TRUTH[remaining][:, ids]
            equal = (features[:, None] == features[None, :]).all(axis=-1)
            left, right = np.argwhere(np.triu(equal, 1))[0]
            order[row, 8:10] = remaining[[left, right]]
            order[row, 10:] = np.delete(remaining, [left, right])
        assert np.all(labels[np.arange(n), order[:, 8]] == labels[np.arange(n), order[:, 9]])
    return labels, order


def probability(z):
    return 1. / (1. + np.exp(-np.clip(z, -50, 50)))


def gradient(w, x, y):
    residual = probability((w * x).sum(axis=-1)) - y
    return residual[..., None] * x


def fresh(n):
    return np.zeros((n, 12)), np.zeros((n, 12)), 0


def advance(w, g, method, rate, state, gate):
    if method == "matched-norm":
        # Same step length as the experienced gate, ordinary gradient direction.
        ratio = np.linalg.norm(gate * g, axis=-1) / np.maximum(np.linalg.norm(g, axis=-1), 1e-12)
        return w - rate * ratio[..., None] * g, state
    if method == "adam":
        m, v, t = state
        t += 1
        m = .9 * m + .1 * g
        v = .999 * v + .001 * g ** 2
        step = (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
        return w - rate * step, (m, v, t)
    return w - rate * gate * g, state


def life(table, seed, method, rate, train=TRAIN, save_all=False):
    rng = np.random.default_rng(seed + 300)
    labels, order = tools(rng, train, 100)
    w = np.zeros((1, 12))
    state = fresh(1)
    log_gate = np.zeros(12)
    result = {}
    for t in range(101):
        if save_all or t in CHECKPOINTS:
            result[t] = (w.copy(), np.exp(log_gate), state)
        if t == 100:
            break
        for j in range(0, 8, 2):
            a, b = order[t, j:j + 2]
            gate = np.exp(log_gate)
            g = gradient(w, table[a], labels[t, a])
            w, state1 = advance(w, g, method, rate, state, gate)
            # Shadow Adam accumulates the actual gradients seen by the gated trajectory.
            if method == "gate":
                _, state1 = advance(w, g, "adam", rate, state, 1.)
            h = gradient(w, table[b], labels[t, b])
            if method == "gate":
                log_gate = np.clip(log_gate + .5 * rate * gate * (g * h)[0], -3, 3)
            w, state = advance(w, h, method, rate, state1, gate)
            if method == "gate":
                _, state = advance(w, h, "adam", rate, state1, 1.)
    return result


def evaluate(table, task, checkpoint, method, rate, gate=1., clear=False, paired_probe=False):
    labels, order = task
    rows = np.arange(len(labels))
    initial, _, state = checkpoint
    w = np.broadcast_to(initial, (len(labels), 12)).copy()
    if clear:
        state = fresh(len(labels))
    query = order[:, 9:]
    qx = table[query]
    target = labels[rows[:, None], query]
    curves = []
    for j in range(9):
        if j in BUDGETS:
            pred = (qx * w[:, None]).sum(axis=-1) >= 0
            curves.append((pred == target).mean(axis=1))
        i = order[:, j]
        x, y = table[i], labels[rows, i]
        g = gradient(w, x, y)
        if j == 0:
            first = g.copy()
        w, state = advance(w, g, method, rate, state, gate)
    # The ninth verified item is reserved for the paired feedback challenge.
    probe = table[order[:, 9]] if paired_probe else x
    before = ((w * probe).sum(-1) >= 0) == y
    def after(advice):
        updated, _ = advance(w, gradient(w, x, advice), method, rate, state, gate)
        return (((updated * probe).sum(-1) >= 0) == y)
    feedback = np.column_stack([before, after(y), after(1 - y)])
    return np.column_stack(curves), feedback, first


def context_curve(table, task):
    labels, order = task
    rows = np.arange(len(labels))[:, None]
    query = order[:, 9:]
    # A frozen linear kernel using all demonstrations as labeled context.
    kernels = table[query] @ table[order[:, :8]].transpose(0, 2, 1)
    votes = kernels * (2 * labels[rows, order[:, :8]] - 1)[:, None]
    return np.column_stack([((votes[:, :, :n].sum(-1) >= 0) == labels[rows, query]).mean(1)
                            for n in BUDGETS])


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    encoders = [feature_table(s) for s in SEEDS]
    lives = [life(table, s, "gate", BASE_RATE) for (table, _), s in zip(encoders, SEEDS)]
    records, meta = {}, []
    for idx, ((table, params), s) in enumerate(zip(encoders, SEEDS)):
        dev = tools(np.random.default_rng(1000 + s), DEV, 96)
        test = tools(np.random.default_rng(2000 + s), TEST, 128)
        shift = tools(np.random.default_rng(6000 + s), SHIFT, 128)
        practiced = lives[idx]
        info = {"seed": s, "feature_accuracy": float(((table > 0) == (TRUTH > 0)).mean()),
                "checkpoint": {}}
        # Trials can be reconstructed from the seed, including per-item predictions' targets.
        for t in CHECKPOINTS:
            cp = practiced[t]
            other_gates = [hist[t][1] for k, hist in enumerate(lives) if k != idx]
            peer = np.mean(other_gates, axis=0)
            shuffled = cp[1][np.random.default_rng(20 + s).permutation(12)]
            options = {}
            for name, method, clear in (("sgd", "sgd", False), ("adam", "adam", False),
                                        ("fresh-adam", "adam", True)):
                options[name] = [(method, r, 1., clear) for r in RATES]
            options["prior"] = [("sgd", r, np.where(np.isin(np.arange(12), COMMON), 1., low), False)
                                for r in RATES for low in (0., .1, .5)]
            chosen, scores = {}, {}
            for name, configs in options.items():
                values = [evaluate(table, dev, cp, *cfg)[0][:, 1:].mean() for cfg in configs]
                best = int(np.argmax(values))
                chosen[name], scores[name] = configs[best], float(values[best])
            conventional = max(("sgd", "adam", "fresh-adam"), key=scores.get)
            info["checkpoint"][t] = {"winner": conventional, "dev": scores,
                                     "rates": {k: v[1] for k, v in chosen.items()},
                                     "prior_low": float(np.min(chosen["prior"][2]))}
            configs = {**chosen, "conventional": chosen[conventional],
                       "experienced": ("sgd", BASE_RATE, cp[1], False),
                       "matched-norm": ("matched-norm", BASE_RATE, cp[1], False),
                       "reset": ("sgd", BASE_RATE, 1., False),
                       "shuffled": ("sgd", BASE_RATE, shuffled, False),
                       "peer": ("sgd", BASE_RATE, peer, False)}
            records[f"{s}-{t}-gate"] = cp[1]
            records[f"{s}-{t}-weights"] = cp[0]
            records[f"{s}-{t}-moments"] = np.stack(cp[2][:2])
            info["checkpoint"][t]["adam_steps"] = cp[2][2]
            for family, task in (("same", test), ("shift", shift)):
                prefix = f"{s}-{t}-{family}-"
                first, zero = None, None
                for name, cfg in configs.items():
                    curve, fb, g = evaluate(table, task, cp, *cfg)
                    if first is not None:
                        assert np.array_equal(first, g) and np.array_equal(zero, curve[:, 0])
                    first, zero = g, curve[:, 0]
                    records[prefix + name] = curve
                    records[prefix + name + "-fb"] = fb
                records[prefix + "context"] = context_curve(table, task)
                for name in ("sgd", "adam"):
                    r = chosen[name][1]
                    own = life(table, s, name, r)[t]
                    records[prefix + "own-" + name] = evaluate(table, task, own, name, r)[0]
        meta.append(info)
    np.savez_compressed(ROOT / "trials.npz", **records)
    output = {"seeds": meta, "encoder_parameters": params, "seconds": time.perf_counter() - start,
              "torch": torch.__version__, "numpy": np.__version__}
    (ROOT / "run.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Completed {len(SEEDS)} lifetimes in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
