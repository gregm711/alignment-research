"""Transfer and fingerprint controls for the forked-memory study."""

import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from study import N, ROOT, SPLIT, TRAIN, TEST, WORLDS, accuracy, sample


class GlobalMemory(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.record = nn.Embedding(N**3, 96)
        self.write = nn.Sequential(nn.Linear(96, 128), nn.Tanh(), nn.Linear(128, bits))
        self.read = nn.Sequential(nn.Linear(bits, 128), nn.Tanh(), nn.Linear(128, 2 * N * N))
        self.log_odds = nn.Parameter(torch.zeros(()))

    def codes(self, world):
        idx = world + (N**2 * torch.arange(N))
        soft = self.write(self.record(idx).sum(1) / N**0.5).tanh()
        return torch.where(soft >= 0, 1.0, -1.0) + (soft - soft.detach())

    def forward(self, world, bins, colors, shapes, fork):
        batch = torch.arange(len(world))
        logits = self.read(self.codes(world)).reshape(-1, 2, N, N).log_softmax(2)

        def score(axis, keys):
            return logits[batch, axis, :, keys]

        cc, sc = score(0, colors[:, 0]), score(1, shapes[:, 0])
        label = bins[batch, fork]
        gate = torch.stack([cc[batch, label] + self.log_odds, sc[batch, label]], 1).log_softmax(1)
        heads = torch.stack([score(0, colors[:, 1]), score(1, shapes[:, 1])], 1)
        return (gate[:, :, None] + heads).logsumexp(1)


def diagnostics(codes):
    worlds = WORLDS.numpy()
    truth = np.stack([(worlds // N).argsort(1), (worlds % N).argsort(1)], 1)
    _, group = np.unique(np.packbits(codes > 0, axis=1), axis=0, return_inverse=True)
    exact = np.zeros(2)
    for axis in (0, 1):
        for key in range(N):
            cell = group * N + truth[:, axis, key]
            histogram = np.bincount(cell, minlength=(int(group.max()) + 1) * N).reshape(-1, N)
            exact[axis] += histogram.max(1).sum() / (len(codes) * N)
    ids = SPLIT[:2048].numpy()
    test = TEST.numpy()
    design = np.column_stack([np.ones(len(codes)), codes])
    x = design[ids]
    y = np.eye(N)[truth[ids]].reshape(len(x), -1)
    weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
    guess = (design[test] @ weights).reshape(-1, 2, N, N).argmax(-1)
    return dict(
        exact=exact.tolist(),
        linear=(guess == truth[test]).mean(axis=(0, 2)).tolist(),
        distinct_states=len(set(group.tolist())),
    )


def main():
    torch.set_num_threads(1)
    dest = ROOT / "results" / "global"
    dest.mkdir(exist_ok=True)
    for bits in (4, 6):
        for p in sorted((ROOT / "results" / "confirmation").glob(f"fork-{bits}-*.json")):
            run = json.loads(p.read_text())
            words = np.asarray(run["words"])[WORLDS.numpy()]
            code = (((words[:, :, None] >> np.arange(bits)) & 1) * 2 - 1).reshape(len(words), -1)
            scores = diagnostics(code)
            assert np.allclose(scores["exact"], run["uniform_map_accuracy"])
            (dest / (p.stem + "-readout.json")).write_text(json.dumps(scores) + "\n")
    _, world, bins, colors, shapes = sample(TEST, 1024, torch.Generator().manual_seed(1602))
    target = bins[:, 2:].numpy()
    for bits in (30, 20):
        for seed in (77, 88, 99):
            for prob in (0.5, 0.125, 0.875):
                torch.manual_seed(seed)
                rng = torch.Generator().manual_seed(700 + seed)
                model = GlobalMemory(bits)

                def inspect():
                    with torch.no_grad():
                        codes = (
                            torch.cat([model.codes(w) for w in WORLDS.split(256)])
                            .numpy()
                            .astype(np.int8)
                        )
                    summary = diagnostics(codes)
                    with torch.no_grad():
                        preds = torch.stack(
                            [
                                model(
                                    world, bins, colors, shapes, torch.full((len(bins),), axis)
                                ).argmax(1)
                                for axis in (0, 1)
                            ],
                            1,
                        )
                    summary["fork_accuracy"] = accuracy(preds.numpy(), target)
                    return summary

                initial = inspect()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                for _ in range(6000):
                    _, w, b, c, s = sample(TRAIN, 96, rng)
                    f = (torch.rand(len(w), generator=rng) >= prob).long()
                    pred = model(w, b, c, s, f)
                    optimizer.zero_grad()
                    F.nll_loss(pred, b[torch.arange(len(w)), 2 + f]).backward()
                    optimizer.step()
                with torch.no_grad():
                    preds = torch.stack(
                        [
                            model(world, bins, colors, shapes, torch.full((len(bins),), axis)).argmax(1)
                            for axis in (0, 1)
                        ],
                        1,
                    )
                result = dict(
                    bits=bits,
                    seed=seed,
                    p=prob,
                    initial=initial,
                    final=inspect(),
                    accuracy=accuracy(preds.numpy(), target),
                    predictions=preds.tolist(),
                )
                path = dest / f"global-{bits}-{int(1000 * prob)}-{seed}.json"
                path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
                print(path.stem, result["accuracy"], result["final"], flush=True)


if __name__ == "__main__":
    main()
