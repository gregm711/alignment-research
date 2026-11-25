"""Active induction and revision of binary feedback rules, using bounded state."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics

from study import measure

ROOT = Path(__file__).resolve().parent
WIDTHS = (8, 16, 32, 64)
BUDGETS = (8, 16, 32, 72)
CHECKS = (0, 1, 2, 4, 8)


def parity(mask, x):
    return (mask & x).bit_count() % 2


def advance(n, mask, x, steps):
    for _ in range(steps):
        x = (x >> 1) | (parity(mask, x) << (n-1))
    return x


def fixture(n, seed, triples):
    rng = random.Random(seed)
    def delta():
        if rng.random() < .75:
            return 1 << rng.randrange(n)
        if triples == "alias":
            while True:
                a, b, c, j = rng.sample(range(n), 4)
                d = a ^ b ^ c
                if d != j:
                    return sum(1 << i for i in (a, b, c, d, j))
        if triples:
            return sum(1 << i for i in rng.sample(range(n), 3))
        return rng.randrange(1, 1 << n)
    old = 0
    phases = []
    for _ in range(2):
        edit = delta()
        mask = old ^ edit
        witness = rng.getrandbits(n)
        while not parity(edit, witness):
            witness = rng.getrandbits(n)
        query = [rng.getrandbits(n) for _ in range(8)]
        phases.append((mask, (witness, parity(mask, witness)), query))
        old = mask
    return phases


class Equations:
    def __init__(self, n):
        self.rows = [0]*n
        self.labels = [0]*n
        self.touches = 0

    def add(self, x, y):
        while x:
            i = x.bit_length()-1
            self.touches += 1
            if not self.rows[i]:
                self.rows[i], self.labels[i] = x, y
                return
            x ^= self.rows[i]
            y ^= self.labels[i]
        assert y == 0

    def answer(self, prior=0):
        mask = prior
        for i, row in enumerate(self.rows):
            if row and parity(mask, row) != self.labels[i]:
                mask ^= 1 << i
        return mask


def infer(n, cap, limit, previous, witness, ask, seed, replay=None, targeted=False):
    equations = Equations(n)
    log = []
    def record(x, y):
        equations.add(x, y ^ parity(previous, x))
        log.append((x, y))
        return y ^ parity(previous, x)
    def probe(x):
        assert len(log) <= cap
        return record(x, ask(x))
    record(*witness)
    counts = Counter()
    candidate = None
    if replay is not None:
        for x, y in replay:
            record(x, y)
    elif limit is not None:
        index = 0
        for bit in range((n-1).bit_length()):
            width = 1 << bit
            stripe = (((1 << n)-1) // ((1 << width)+1)) << width
            index |= probe(stripe) << bit
        guess = 1 << index
        if all(parity(previous ^ guess, x) == y for x, y in log):
            rng = random.Random(seed)
            for check in range(min(limit, cap+1-len(log))):
                x = rng.getrandbits(n)
                if targeted and check == 0:
                    x = guess
                counts["checks"] += 1
                if probe(x) != parity(guess, x):
                    break
            else:
                candidate = previous ^ guess
        counts["proposals"] = 1
        counts["rejected"] = int(candidate is None)
    if candidate is None:
        if replay is None:
            for i in range(n):
                if len(log) > cap:
                    break
                if not equations.rows[i]:
                    probe(1 << i)
        candidate = previous ^ equations.answer()
    counts.update(probes=len(log)-1, row_touches=equations.touches,
                  transmitted_bits=len(log)*(n+1))
    return candidate, log[1:], counts


def run(worlds, n, cap, limits, seed, replay=None, targeted=False, infer_fn=infer):
    outputs = []
    for j, world in enumerate(worlds):
        old = 0
        phases = []
        for k, (secret, witness, xs) in enumerate(world):
            def oracle(x):
                return parity(secret, x)
            old, log, count = infer_fn(n, cap, limits[k], old, witness, oracle,
                                    seed+2*j+k, None if replay is None else replay[j][k], targeted)
            predictions = [advance(n, old, x, t) for x in xs for t in (3, 7)]
            phases.append((predictions, log, count))
        outputs.append(phases)
    return outputs


def grade(outputs, labels):
    return [[p[0] == truth for p, truth in zip(world, targets)]
            for world, targets in zip(outputs, labels)]


def dataset(n, base, size, triples):
    worlds = [fixture(n, base+100*n+i, triples) for i in range(size)]
    labels = [[[advance(n, mask, x, t) for x in xs for t in (3, 7)]
               for mask, _, xs in world] for world in worlds]
    return worlds, labels


def fit():
    policy = {}
    for n in WIDTHS:
        worlds, labels = dataset(n, 170000, 24, False)
        for cap in BUDGETS:
            limits = [0, 0]
            for k in range(2):
                choices = []
                for value in CHECKS:
                    limits[k] = value
                    out = run(worlds, n, cap, limits, 701000)
                    scores = grade(out, labels)
                    choices.append((sum(row[k] for row in scores),
                                    -sum(row[k][2]["probes"] for row in out), -value))
                limits[k] = -max(choices)[2]
            policy[f"{n}/{cap}"] = limits
    return policy


def main():
    eq = Equations(4)
    for x in (9, 7, 3, 1):
        eq.add(x, parity(13, x))
    assert eq.answer() == 13
    for n in WIDTHS:
        for i in range(n):
            one = 1 << i
            mask, _, _ = infer(n, 8, 0, 0, (one, 1), lambda x: parity(one, x), 0)
            assert mask == one
    policy, training = measure(fit)
    (ROOT / "feedback-policy.json").write_text(json.dumps(dict(policy=policy, training=training))+"\n")
    results = {}
    for base, family in [(270000, "mixed"), (370000, "three-tap")]:
        for n in WIDTHS:
            worlds, labels = dataset(n, base, 48, family == "three-tap")
            for cap in BUDGETS:
                schedules = {"fitted": policy[f"{n}/{cap}"], "basis": [None]*2,
                             **{f"fixed-{c}": [c]*2 for c in CHECKS}}
                fitted = run(worlds, n, cap, schedules["fitted"], base+910000)
                logs = [[phase[1] for phase in row] for row in fitted]
                schedules["replay"] = [None]*2
                keys = list(schedules)
                random.Random(base+n+cap).shuffle(keys)
                for key in keys:
                    def perform():
                        return run(worlds, n, cap, schedules[key], base+910000,
                                   logs if key == "replay" else None)
                    out, _ = measure(perform)
                    costs = []
                    for _ in range(3):
                        _, cost = measure(perform)
                        costs.append(cost)
                    pairs = grade(out, labels)
                    counts = sum((p[2] for row in out for p in row), Counter())
                    total = sum(map(all, pairs))
                    energy = [t["cpu_joules"] for t in costs if t["cpu_joules"] is not None]
                    results[f"{family}/{n}/{cap}/{key}"] = dict(
                        solved=total, phases=pairs, counts=dict(counts), costs=costs,
                        joules_per_solve=statistics.median(energy)/total if energy and total else None)
        print(f"{family} finished", flush=True)
    (ROOT / "feedback-results.json").write_text(json.dumps(results, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
