"""Run the local sleep-timing feasibility pilot with python3 dynamic-learning/pilot.py."""
import argparse
import itertools
import json
import pathlib
import platform
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 32
PAIRS = torch.tensor(list(itertools.combinations(range(8), 2)))
STEPS = 800
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5)
FIXED = ("first", "2", "3", "4", "5", "6", "last")
RESULTS = pathlib.Path(__file__).parent / "results"


def episodes(size, before, after, rng, conditional):
    windows = before + after
    machine = torch.randint(2, (size,), generator=rng)
    changed = torch.arange(size) % 2 == 0
    rules = torch.randint(2, (size, 2), generator=rng)
    latest = rules.clone()
    latest[torch.arange(size), machine] ^= changed.long()
    labels = torch.cat([rules[:, None].expand(-1, before, -1),
                        latest[:, None].expand(-1, after, -1)], dim=1)
    indices = torch.rand(size, len(PAIRS), generator=rng).argsort()[:, :2 * (windows + 1)]
    pairs = PAIRS[indices].reshape(size, windows + 1, 2, 2)
    flip = torch.rand(size, windows + 1, 2, 1, generator=rng) < 0.5
    pairs = torch.where(flip, pairs.flip(-1), pairs)
    support, query = pairs[:, :-1], pairs[:, -1]
    flipped_query = changed.clone()
    if conditional:
        rules = torch.randint(2, (size, 2, 2), generator=rng)
        new = rules.clone()
        entry = torch.randint(2, (size,), generator=rng)
        new[torch.arange(size), machine, entry] ^= changed.long()
        cue = (pairs[..., 0] >= 4).long()
        table = torch.cat([rules[:, None].expand(-1, before, -1, -1),
                           new[:, None].expand(-1, after + 1, -1, -1)], dim=1)
        choices = table.gather(-1, cue.unsqueeze(-1)).squeeze(-1)
        labels, latest = choices[:, :-1], choices[:, -1]
        flipped_query = changed & (cue[torch.arange(size), -1, machine] == entry)
    observed = support.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return support, observed, query, latest, changed, machine, flipped_query


class WindowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.symbol = nn.Embedding(8, WIDTH)
        self.machine = nn.Embedding(2, WIDTH)
        self.example = nn.Linear(4 * WIDTH, WIDTH)
        self.question = nn.Linear(3 * WIDTH, WIDTH)
        self.block = nn.TransformerEncoderLayer(WIDTH, 2, 2 * WIDTH, dropout=0,
                                                batch_first=True, norm_first=True)
        self.age = nn.Embedding(10, WIDTH)
        self.read = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.answer = nn.Linear(WIDTH, 2)

    def inputs(self, pairs):
        machines = self.machine.weight.expand(*pairs.shape[:-2], 2, WIDTH)
        return torch.cat([machines, self.symbol(pairs).flatten(-2)], dim=-1)

    def encode(self, support, observed):
        size, windows = observed.shape[:2]
        evidence = torch.cat([self.inputs(support), self.symbol(observed)], dim=-1)
        return self.block(self.example(evidence).reshape(-1, 2, WIDTH)).reshape(size, windows, 2, WIDTH)

    def predict(self, memory, query, slot=None):
        if slot is not None:
            batch = torch.arange(len(memory))
            memory = memory.clone()
            memory[batch, slot] = self.block(memory[batch, slot])
        windows = memory.size(1)
        age = self.age(torch.arange(windows - 1, -1, -1))[None, :, None]
        keys = (memory + age).flatten(1, 2)
        q = self.question(self.inputs(query))
        value, _ = self.read(q, keys, memory.flatten(1, 2), need_weights=False)
        return self.answer(value)

    def surprise(self, memory, support, observed):
        losses = torch.full(memory.shape[:2], -torch.inf)
        for w in range(1, memory.size(1)):
            logits = self.predict(memory[:, :w], support[:, w])
            labels = (support[:, w, :, 1] == observed[:, w]).long()
            losses[:, w] = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none").mean(1)
        return losses


def choose(policy, windows, losses, random_slot):
    if policy == "random":
        return random_slot
    if policy.startswith("loss>"):
        fired = losses > float(policy[5:])
        return torch.where(fired.any(1), fired.int().argmax(1), windows - 1)
    fixed = {"first": 0, "last": windows - 1}
    position = fixed[policy] if policy in fixed else min(int(policy) - 1, windows - 1)
    return torch.full_like(random_slot, position)


@torch.no_grad()
def evaluate(model, seed, counts, policies, conditional, size=96):
    rows = []
    rng = torch.Generator().manual_seed(seed)
    for before in (2, 3, 4, 5):
        for after in counts:
            support, observed, query, labels, changed, machine, flipped = episodes(size, before, after, rng, conditional)
            latest_choice = torch.zeros_like(labels)
            for demo, answer in zip(support.unbind(1), observed.unbind(1)):
                match = (demo[..., 0] >= 4) == (query[..., 0] >= 4)
                choice = (demo[..., 1] == answer).long()
                latest_choice = torch.where(match, choice, latest_choice) if conditional else choice
            if not conditional:
                assert torch.equal(latest_choice, labels)
            assert not ((support.sort(-1).values == query[:, None].sort(-1).values).all(-1)).any()
            memory = model.encode(support, observed)
            windows = memory.size(1)
            losses = model.surprise(memory, support, observed)
            random_slot = torch.randint(windows, (size,), generator=rng)
            slots = {}
            predictions = {"latest": latest_choice.tolist()}
            for policy in policies:
                slot = None if policy == "off" else choose(policy, windows, losses, random_slot)
                predictions[policy] = model.predict(memory, query, slot).argmax(-1).tolist()
                slots[policy] = [-1] * size if slot is None else slot.tolist()
            for i in range(size):
                rows.append(dict(before=before, after=after, changed=bool(changed[i]),
                                 machine=int(machine[i]), flipped_query=bool(flipped[i]), labels=labels[i].tolist(),
                                 predictions={p: v[i] for p, v in predictions.items()},
                                 slots={p: v[i] for p, v in slots.items()}))
    return rows


def correct(row, policy, other=False):
    m = row["machine"] ^ int(other)
    return int(row["predictions"][policy][m] == row["labels"][m])


def mean(values):
    return sum(values) / len(values)


def main(conditional):
    start = time.monotonic()
    offset = 100 if conditional else 0
    torch.set_num_threads(2)
    torch.manual_seed(71 + offset)
    rng = torch.Generator().manual_seed(72 + offset)
    lengths = random.Random(73 + offset)
    model = WindowModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(STEPS):
        before, after = lengths.randint(2, 5), lengths.choice((1, 2, 4))
        support, observed, query, labels, _, _, _ = episodes(64, before, after, rng, conditional)
        slot = torch.randint(before + after, (64,), generator=rng)
        logits = model.predict(model.encode(support, observed), query, slot)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    triggers = [f"loss>{v}" for v in THRESHOLDS]
    dev = evaluate(model, 74 + offset, (2,), [*FIXED, *triggers], conditional)
    dev_changed = [r for r in dev if r["changed"]]
    scores = {p: mean([correct(r, p) for r in dev_changed]) for p in [*FIXED, *triggers]}
    fixed = max(FIXED, key=scores.get)
    trigger = max(triggers, key=scores.get)
    policies = ["off", "random", fixed, trigger]
    rows = evaluate(model, 75 + offset, (1, 2, 4), policies, conditional)
    summary = {}
    primary = []
    for after in (1, 2, 4):
        subset = [r for r in rows if r["changed"] and r["after"] == after]
        summary[after] = {p: [mean([correct(r, p, other) for r in subset])
                              for other in (False, True)] for p in [*policies, "latest"]}
        if after == 2:
            primary = [correct(r, trigger) - correct(r, fixed) for r in subset]
    boot = random.Random(76)
    differences = sorted(mean(boot.choices(primary, k=len(primary))) for _ in range(4000))
    stable = [r for r in rows if not r["changed"]]
    affected = [r for r in rows if r["after"] == 2 and r["flipped_query"]]
    result = dict(fixed=fixed, trigger=trigger, development_scores=scores, accuracy_changed_unchanged=summary,
                  stable_off_accuracy=mean([correct(r, "off", m) for r in stable for m in (False, True)]),
                  stable_trigger_before_final=mean([r["slots"][trigger] < r["before"] + r["after"] - 1 for r in stable]),
                  mean_online_trigger_query_calls=mean([r["slots"][trigger] for r in rows]),
                  primary_gain=mean(primary), paired_bootstrap_95=[differences[99], differences[3899]],
                  flipped_query_n=len(affected),
                  flipped_query_accuracy={p: mean([correct(r, p) for r in affected]) for p in [*policies, "latest"]},
                  seconds=round(time.monotonic()-start, 2), steps=STEPS, torch=str(torch.__version__),
                  python=platform.python_version())
    print(json.dumps(result, indent=2), flush=True)
    RESULTS.mkdir(exist_ok=True)
    name = "conditional-rule-pilot" if conditional else "sleep-timing-pilot"
    (RESULTS / f"{name}.json").write_text(json.dumps(dict(summary=result, episodes=rows), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional", action="store_true", help="use two rules per machine")
    main(parser.parse_args().conditional)
