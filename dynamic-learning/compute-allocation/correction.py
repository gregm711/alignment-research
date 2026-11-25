"""Locate genuine coefficient changes by bisecting a contradicting observation."""
from collections import Counter
from itertools import permutations
import json
import random
import statistics

from feedback import BUDGETS, Equations, ROOT, dataset, grade, infer, measure, parity, run, advance

SEED = 740613


def correct(n, cap, flip_limit, previous, witness, ask, seed, replay, targeted, deduce=False):
    assert replay is None and not targeted
    eqs = Equations(n)
    history = [witness]
    eqs.add(*witness)
    candidate = previous
    counts = Counter()
    rng = random.Random(seed)

    def probe(x):
        assert len(history) <= cap
        if deduce:
            residual, implied = x, 0
            while residual:
                i = residual.bit_length()-1
                if not eqs.rows[i]:
                    break
                residual ^= eqs.rows[i]
                implied ^= eqs.labels[i]
            if residual == 0:
                counts["deductions"] += 1
                return implied
        y = ask(x)
        eqs.add(x, y)
        history.append((x, y))
        return y

    while len(history) <= cap:
        if deduce and all(eqs.rows):
            break
        bad = next((x for x, y in history if parity(candidate, x) != y), None)
        if bad is None:
            if all(eqs.rows) or counts["confirmations"] >= 4:
                break
            before = len(history)
            probe(rng.getrandbits(n))
            counts["checks"] += 1
            counts["confirmations"] += int(len(history) > before)
            continue
        if counts["flips"] == flip_limit:
            for i, row in enumerate(eqs.rows):
                if not row and len(history) <= cap:
                    probe(1 << i)
            break
        counts["confirmations"] = 0
        while bad & (bad-1) and len(history) <= cap:
            half = 0
            for _ in range(bad.bit_count() // 2):
                low = bad & -bad
                half |= low
                bad ^= low
            if probe(half) != parity(candidate, half):
                bad = half
        if bad & (bad-1):
            break
        candidate ^= bad
        counts["flips"] += 1
    candidate = eqs.answer(candidate)
    counts.update(probes=len(history)-1, row_touches=eqs.touches,
                  transmitted_bits=len(history)*(n+1))
    return candidate, history[1:], counts


def external(n, masks):
    rng = random.Random(SEED+n)
    pairs = list(permutations(masks, 2))
    worlds, labels = [], []
    for first, second in rng.sample(pairs, min(96, len(pairs))):
        order = rng.sample(range(n), n)
        old = 0
        phases, targets = [], []
        for raw in (first, second):
            mask = sum(((raw >> i) & 1) << j for i, j in enumerate(order))
            witness = rng.getrandbits(n)
            while not parity(old ^ mask, witness):
                witness = rng.getrandbits(n)
            xs = [rng.getrandbits(n) for _ in range(8)]
            phases.append((mask, (witness, parity(mask, witness)), xs))
            targets.append([advance(n, mask, x, t) for x in xs for t in (3, 7)])
            old = mask
        worlds.append(phases)
        labels.append(targets)
    return worlds, labels


def main():
    masks = json.loads((ROOT/"polynomials.json").read_text())["masks"]
    results = {}
    for domain in ("three", "dense", "catalogue"):
        for n in (8, 16, 32, 64):
            if domain == "catalogue":
                if len(masks.get(str(n), [])) < 2:
                    continue
                worlds, labels = external(n, masks[str(n)])
            else:
                worlds, labels = dataset(n, SEED+1000*(domain=="dense"), 96, domain=="three")
            for cap in BUDGETS:
                arms = [("basis", infer, None), ("fixed4", infer, 4),
                        ("split", correct, None), ("switch2", correct, 2)]
                random.Random(SEED+n+cap).shuffle(arms)
                for name, fn, limit in arms:
                    def perform():
                        return run(worlds, n, cap, [limit]*2, SEED+8300, infer_fn=fn)
                    rows, timing = measure(perform)
                    repeats = max(1, min(512, round(.035/timing["wall_seconds"])))
                    def batch():
                        for _ in range(repeats):
                            perform()
                    trials = []
                    for _ in range(3):
                        _, cost = measure(batch)
                        trials.append({k: v/repeats if v is not None else None for k, v in cost.items()})
                    pairs = grade(rows, labels)
                    solves = sum(map(all, pairs))
                    counts = sum((p[2] for row in rows for p in row), Counter())
                    energy = [t["cpu_joules"] for t in trials if t["cpu_joules"] is not None]
                    results[f"{domain}/{n}/{cap}/{name}"] = dict(solved=solves, phases=pairs,
                        counts=dict(counts), costs=trials, repeats=repeats,
                        joules_per_solve=statistics.median(energy)/solves if solves and energy else None)
        print(domain, flush=True)
    (ROOT/"correction-results.json").write_text(json.dumps(results,separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
