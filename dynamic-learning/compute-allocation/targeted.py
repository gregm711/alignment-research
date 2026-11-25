"""Paired trial of candidate-specific versus random verification."""
from collections import Counter
import json
import random
import statistics

from feedback import ROOT, BUDGETS, dataset, grade, measure, parity, run


def main():
    for n in (16, 32, 64):
        worlds, _ = dataset(n, 670000, 96, "alias")
        traps = 0
        for world in worlds:
            old = 0
            for mask, _, _ in world:
                change = old ^ mask
                if change.bit_count() > 1:
                    indices = [i for i in range(n) if parity(change, 1 << i)]
                    j = 0
                    for i in indices:
                        j ^= i
                    assert len(indices) == 5 and j in indices
                    traps += 1
                old = mask
        assert traps > 0
    results = {}
    for name, seed, kind in [("three", 470000, True), ("dense", 570000, False),
                             ("alias", 670000, "alias")]:
        for n in (16, 32, 64):
            worlds, labels = dataset(n, seed, 96, kind)
            for budget in BUDGETS:
                arms = [(False, None), (False, 0), (False, 1), (True, 1), (False, 4), (True, 4)]
                random.Random(seed+n+budget).shuffle(arms)
                for targeted, limit in arms:
                    def perform():
                        return run(worlds, n, budget, [limit]*2, seed+910000, targeted=targeted)
                    rows, timing = measure(perform)
                    repeats = max(1, round(.040/timing["wall_seconds"]))
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
                    key = f"{name}/{n}/{budget}/" + ("target" if targeted else "random") + str(limit)
                    results[key] = dict(solved=solves, phases=pairs, counts=dict(counts), costs=trials,
                                        repeats=repeats, joules_per_solve=statistics.median(energy)/solves
                                        if energy and solves else None)
        print(name, flush=True)
    (ROOT/"targeted-results.json").write_text(json.dumps(results, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
