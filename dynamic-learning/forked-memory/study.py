"""CPU study. Run python3 dynamic-learning/forked-memory/study.py from any directory."""

import argparse
import itertools
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N = 5
SEEDS = [11, 22, 33]
STEPS = 1600
ROOT = pathlib.Path(__file__).parent
PAIR = torch.arange(N * N)
PERMS = torch.tensor(list(itertools.permutations(range(N))))
WORLDS = (PERMS[:, None, :] * N + PERMS[None, :, :]).reshape(-1, N)
ORDERS = PERMS  # first four entries index the two distinct clue bins and two distinct query bins
SPLIT = torch.randperm(len(WORLDS), generator=torch.Generator().manual_seed(124))
TRAIN, TEST = SPLIT.split([11520, 2880])


def sample(pool, count, rng):
    ids = pool[torch.randint(len(pool), (count,), generator=rng)]
    world = WORLDS[ids]
    bins = ORDERS[torch.randint(len(ORDERS), (count,), generator=rng)][:, :4]
    values = world.gather(1, bins)
    colors = values[:, ::2] // N
    shapes = values[:, 1::2] % N
    return ids, world, bins, colors, shapes


class Learner(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2 * N, 48), nn.Tanh(), nn.Linear(48, bits))
        self.reader = nn.Sequential(nn.Linear(bits, 64), nn.Tanh(), nn.Linear(64, 2 * N))
        self.log_odds = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "pairs", torch.cat([F.one_hot(PAIR // N, N), F.one_hot(PAIR % N, N)], 1).float()
        )

    def codes(self):
        soft = self.encoder(self.pairs).tanh()
        hard = torch.where(soft >= 0, 1.0, -1.0)
        return hard + (soft - soft.detach())

    def forward(self, world, bins, colors, shapes, fork):
        # Only the discrete word for each bin reaches the reader. Caching the 25 possible
        # writes/reads during training is equivalent to applying the shared network per row.
        table = self.reader(self.codes()).reshape(N * N, 2, N)
        color, shape = table[world].unbind(2)

        def scores(matrix, keys):
            return matrix.gather(2, keys[:, None, :].expand(-1, N, -1)).log_softmax(1).unbind(2)

        c_clue, c_query = scores(color, colors)
        s_clue, s_query = scores(shape, shapes)
        clue_label = bins.gather(1, fork[:, None])
        selector = torch.cat(
            [c_clue.gather(1, clue_label) + self.log_odds, s_clue.gather(1, clue_label)], 1
        )
        query = torch.stack([c_query, s_query], 1)
        mixed = torch.logsumexp(selector.log_softmax(1)[:, :, None] + query, 1)
        return mixed, query, selector


def state_diagnostic(model):
    words = (model.codes().detach().numpy() > 0).astype(int) @ (2 ** np.arange(model.codes().shape[1]))
    worlds = WORLDS.numpy()
    _, groups = np.unique(words[worlds], axis=0, return_inverse=True)
    maxima = []
    for attrs in (worlds // N, worlds % N):
        total = 0
        for key in range(N):
            labels = (attrs == key).argmax(1)
            counts = np.bincount(groups * N + labels, minlength=(groups.max() + 1) * N).reshape(-1, N)
            total += counts.max(1).sum()
        maxima.append(total / (N * len(worlds)))
    return dict(words=words.tolist(), distinct_states=int(groups.max() + 1), uniform_map_accuracy=maxima)


def fit(seed, bits, color_probability, test_data):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(700 + seed)
    model = Learner(bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(STEPS):
        _, world, bins, colors, shapes = sample(TRAIN, 96, rng)
        fork = (torch.rand(len(world), generator=rng) >= color_probability).long()
        mixed, _, _ = model(world, bins, colors, shapes, fork)
        target = bins.gather(1, (2 + fork[:, None])).squeeze(1)
        optimizer.zero_grad()
        F.nll_loss(mixed, target).backward()
        optimizer.step()
    with torch.no_grad():
        _, world, bins, colors, shapes = test_data
        guesses, forced, routes = [], [], []
        for axis in (0, 1):
            mixed, query, selector = model(world, bins, colors, shapes, torch.full((len(world),), axis))
            guesses.append(mixed.argmax(1))
            forced.append(query[:, axis].argmax(1))
            routes.append(selector.argmax(1))
        result = state_diagnostic(model)
        result.update(
            reader_logits=model.reader(model.codes()).reshape(N * N, 2, N).tolist(),
            log_odds=model.log_odds.item(),
            predictions=torch.stack(guesses, 1).tolist(),
            forced=torch.stack(forced, 1).tolist(),
            routes=torch.stack(routes, 1).tolist(),
        )
        return result


def reference(test_data, arrival):
    # Uniform posterior over worlds, equiprobable forks, conditioned on the same
    # distinct-bin sampling rule as the neural evaluation. Resolve ties by bin ID.
    worlds = WORLDS.numpy()
    color_bin = (worlds // N).argsort(axis=1)
    shape_bin = (worlds % N).argsort(axis=1)
    ids, truth, bins, colors, shapes = (t.numpy() for t in test_data)
    ranks = {tuple(p): i for i, p in enumerate(PERMS.tolist())}
    out = {key: [] for key in ("color_only", "fifo_2", "fifo_3", "ranked_maps", "clue_only")}
    for i in range(len(ids)):
        c0, c1 = color_bin[:, colors[i]].T
        s0, s1 = shape_bin[:, shapes[i]].T
        eligible = (c0 != s0) & (c1 != s1) & (c0 != s1) & (s0 != c1)
        # The generation also requires c0 != c1 and s0 != s1, already implied by the keys.
        rowmatch = worlds == truth[i]
        packed_id = ranks[tuple(truth[i] // N)] * len(ranks) + ranks[tuple(truth[i] % N)]
        assert packed_id == ids[i] and packed_id < 2**14
        for name, seen in [
            ("clue_only", np.ones(len(worlds), dtype=bool)),
            ("color_only", (worlds // N == truth[i] // N).all(1)),
            ("fifo_2", rowmatch[:, arrival[i, -2:]].all(1)),
            ("fifo_3", rowmatch[:, arrival[i, -3:]].all(1)),
            ("ranked_maps", np.arange(len(worlds)) == packed_id),
        ]:
            predictions = []
            for label in bins[i, :2]:
                possible = eligible & seen
                targets = np.concatenate((c1[possible & (c0 == label)], s1[possible & (s0 == label)]))
                assert len(targets)
                predictions.append(int(np.bincount(targets, minlength=N).argmax()))
            out[name].append(predictions)
    assert np.array_equal(out["ranked_maps"], bins[:, 2:])
    return out


def accuracy(pred, target):
    hit = np.asarray(pred) == target
    return [*hit.mean(0).tolist(), hit.all(1).mean().item()]


def main(confirm=False):
    torch.set_num_threads(1)
    test = sample(TEST, 1024, torch.Generator().manual_seed(1602 if confirm else 602))
    ids, _, bins, _, _ = test
    target = bins[:, 2:].numpy()
    arrival = ORDERS[
        torch.randint(
            len(ORDERS), (len(ids),), generator=torch.Generator().manual_seed(1603 if confirm else 603)
        )
    ].numpy()
    assert not np.intersect1d(TRAIN.numpy(), ids.numpy()).size
    assert torch.all(bins.sort(1).values.diff(dim=1) > 0)
    refs = reference(test, arrival)
    print("baselines", {k: accuracy(v, target) for k, v in refs.items()}, flush=True)
    result_dir = ROOT / "results"
    if confirm:
        result_dir = result_dir / "confirmation"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "episodes.json").write_text(
        json.dumps(
            dict(ids=ids.tolist(), bins=bins.tolist(), arrival=arrival.tolist(), references=refs),
            separators=(",", ":"),
        )
        + "\n"
    )
    probabilities = (0.125, 0.5, 0.875) if confirm else (0.5, 0.875)
    seeds = [44, 55, 66] if confirm else SEEDS
    for bits, prob, seed in itertools.product((4, 6), probabilities, seeds):
        start = time.monotonic()
        result = fit(seed, bits, prob, test)
        result.update(
            seed=seed,
            bits=bits,
            color_probability=prob,
            accuracy=accuracy(result["predictions"], target),
        )
        path = result_dir / f"fork-{bits}-{int(1000 * prob)}-{seed}.json"
        path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        print(
            path.stem,
            result["accuracy"],
            result["uniform_map_accuracy"],
            round(time.monotonic() - start, 2),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    main(parser.parse_args().confirm)
