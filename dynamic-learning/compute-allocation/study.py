"""Paired single-instance induction, composition and one-edit revision."""
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "circuit-communication"))
from meter import measure

P = 67
OPS = range(4)
DEPTHS = (2, 4, 6, 8)
BUDGETS = (128, 512, 2048, 8192)
SEEDS = {"table": 120310, "affine": 120819}


def execute(tables, program, x):
    for op in program:
        x = tables[op][x]
    return x


def world(family, depth, seed):
    rng = random.Random(seed)
    if family == "table":
        tables = [rng.sample(range(P), P) for _ in OPS]
    elif family == "affine":
        coeff = [(rng.randrange(1, P), rng.randrange(P)) for _ in OPS]
        tables = [[(a*x+b) % P for x in range(P)] for a, b in coeff]
    else:
        raise ValueError(family)
    program = tuple(rng.choices(OPS, k=depth))
    xs = rng.sample(range(P), 32)
    while True:
        i = rng.randrange(depth)
        new = program[:i] + ((program[i] + rng.randrange(1, 4)) % 4,) + program[i+1:]
        if any(execute(tables, program, x) != execute(tables, new, x) for x in xs[16:20]):
            break
    phases, truth = [], []
    for prog, offset in ((program, 0), (new, 16)):
        demos = [(x, execute(tables, prog, x)) for x in xs[offset:offset+4]]
        queries = xs[offset+4:offset+16]
        phases.append((demos, queries))
        truth.append([execute(tables, prog, x) for x in queries])
    return (tables, depth, phases), truth


class Exhausted(Exception):
    pass


class Search:
    def __init__(self, tables):
        self.tables = tables
        self.inverse = None
        self.count = Counter()

    def take(self, n):
        if n > self.left:
            raise Exhausted
        self.left -= n
        self.count["reads"] += n

    def verify(self, program, demos):
        self.count["candidates_checked"] += 1
        for x, y in demos:
            self.take(len(program))
            if execute(self.tables, program, x) != y:
                self.count["rejected"] += 1
                return False
        return True

    def paths(self, tables, start, depth):
        def visit(path, state):
            if len(path) == depth:
                yield path, state
                return
            for op, table in enumerate(tables):
                self.take(len(state))
                yield from visit(path + (op,), tuple(table[x] for x in state))
        return visit((), start)

    def find(self, method, depth, demos, seed, old):
        if old is not None:
            if self.verify(old, demos):
                return old
            for i in range(depth):
                for op in OPS:
                    if op != old[i]:
                        candidate = old[:i] + (op,) + old[i+1:]
                        if self.verify(candidate, demos):
                            return candidate
        if method == "random":
            rng = random.Random(seed)
            while True:
                candidate = tuple(rng.choices(OPS, k=depth))
                if self.verify(candidate, demos):
                    return candidate
        if method == "dfs":
            for candidate, (end,) in self.paths(self.tables, (demos[0][0],), depth):
                self.count["leaves"] += 1
                if end == demos[0][1] and self.verify(candidate, demos[1:]):
                    return candidate
            return None
        assert method == "join"
        if self.inverse is None:
            self.take(P * len(OPS))
            self.inverse = [[0] * P for _ in OPS]
            for table, inv in zip(self.tables, self.inverse):
                for x, y in enumerate(table):
                    inv[y] = x
            self.count["inverse_writes"] += P * len(OPS)
        split = depth // 2
        assert 4 ** split <= 256
        index = defaultdict(list)
        start, end = map(tuple, zip(*demos[:2]))
        for prefix, signature in self.paths(self.tables, start, split):
            index[signature].append(prefix)
            self.count["index_writes"] += 1
        for reverse, signature in self.paths(self.inverse, end, depth-split):
            self.count["index_probes"] += 1
            for prefix in index.get(signature, ()):
                candidate = prefix + reverse[::-1]
                if self.verify(candidate, demos[2:]):
                    return candidate
        return None


def affine(demos, queries, count):
    (x, y), (u, v) = demos[:2]
    a = (y-v) * pow((x-u) % P, -1, P) % P
    b = (y-a*x) % P
    count["modular_inverses"] += 1
    count["multiplications"] += 2
    for x, y in demos[2:]:
        count["multiplications"] += 1
        if (a*x+b) % P != y:
            return None
    count["multiplications"] += len(queries)
    return [(a*x+b) % P for x in queries]


def solve(public, method, budget, seed):
    tables, depth, phases = public
    search = Search(tables)
    old = None
    answers = []
    for phase, (demos, queries) in enumerate(phases):
        if method == "affine":
            answers.append(affine(demos, queries, search.count))
            continue
        reserve = len(queries) * depth
        search.left = budget - reserve
        try:
            old = search.find(method, depth, demos, seed+phase, old)
        except Exhausted:
            old = None
        answer = None
        if old is not None:
            search.left += reserve
            search.take(reserve)
            answer = [execute(tables, old, x) for x in queries]
        answers.append(answer)
    assert search.count["reads"] <= 2*budget
    return answers, search.count


def panel(family, depth):
    return [world(family, depth, SEEDS[family]+1000*depth+i) for i in range(24)]


def cell(worlds, method, budget):
    public = [data for data, _ in worlds]
    def run():
        return [solve(data, method, budget, 9317+2*i) for i, data in enumerate(public)]
    rows, initial = measure(run)
    repeats = max(1, min(256, round(0.025 / initial["wall_seconds"])))
    def batch():
        for _ in range(repeats):
            run()
    trials = []
    for _ in range(3):
        _, cost = measure(batch)
        trials.append({k: v/repeats if v is not None else None for k, v in cost.items()})
    passed = [[a == t for a, t in zip(ans, truth)] for (ans, _), (_, truth) in zip(rows, worlds)]
    solves = sum(map(all, passed))
    energy = [t["cpu_joules"] for t in trials if t["cpu_joules"] is not None]
    return dict(solved=solves, phases=passed, counts=[dict(c) for _, c in rows],
                repeats=repeats, cost_per_panel=trials,
                joules_per_solve=statistics.median(energy)/solves if energy and solves else None)


def main():
    results = {}
    for family in SEEDS:
        for depth in DEPTHS:
            worlds = panel(family, depth)
            for method in ("random", "dfs", "join", "affine"):
                if method == "affine" and family != "affine":
                    continue
                for budget in (BUDGETS[:1] if method == "affine" else BUDGETS):
                    results[f"{family}/{depth}/{method}/{budget}"] = cell(worlds, method, budget)
        print(f"{family} finished", flush=True)
    (ROOT / "results.json").write_text(json.dumps(results, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
