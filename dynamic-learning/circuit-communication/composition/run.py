"""Private permutation composition with source-conditioned routing."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

LAYERS = 5
SYMBOLS = 16
QUERIES = 240
PROTOTYPES = np.array([[1., 0.], [-.5, 3**.5/2], [-.5, -3**.5/2]])
METHODS = ("learned", "analytic", "central", "severed", "first")


def tables(rng, width):
    values = np.broadcast_to(np.arange(SYMBOLS), (LAYERS, width, SYMBOLS))
    return rng.permuted(values, axis=2)


def task(rng, width, depth):
    op = rng.integers(2, size=(LAYERS, QUERIES))
    op[depth - 1] = 2
    if depth < LAYERS:
        op[depth:] = rng.integers(3, size=(LAYERS - depth, QUERIES))
    query = (rng.integers(width, size=QUERIES), rng.integers(SYMBOLS, size=QUERIES),
             PROTOTYPES[op] + rng.normal(0, .30, size=(LAYERS, QUERIES, 2)))
    return query, op


def action(op, value):
    return np.where(op == 2, 2, op ^ (value % 2))


def features(tags):
    return np.column_stack((tags, np.ones(len(tags))))


def fit():
    data = [[], []]
    rng = np.random.default_rng(924053)
    for world in range(32):
        bank = tables(rng, 8)
        depth = world % 2 + 1
        (worker, value, tags), op = task(rng, 8, depth)
        for layer in range(depth):
            value = bank[layer, worker, value]
            target = action(op[layer], value)
            for parity in (0, 1):
                chosen = value % 2 == parity
                data[parity].append((features(tags[layer, chosen]), np.eye(3)[target[chosen]]))
            worker = (2 * worker + target) % 8
    weights = []
    for pairs in data:
        x, y = (np.concatenate(a) for a in zip(*pairs))
        # Equal action weights compensate for extra stop decisions on short training paths.
        counts = y.sum(axis=0)
        scale = 1 / np.sqrt(y @ counts)
        weights.append(np.linalg.lstsq(x * scale[:, None], y * scale[:, None], rcond=None)[0])
    return np.array(weights)


def follow(bank, query, router, weights=None, op=None, record=False):
    width = bank.shape[1]
    entry, initial, tags = query
    cells = entry.copy()
    value = initial.copy()
    counts = np.zeros_like(value)
    active = np.arange(len(value))
    path = np.full((LAYERS, len(value)), -1) if record else None
    for layer in range(LAYERS):
        if record:
            path[layer, active] = cells[active]
        current = bank[layer, cells[active], value[active]]
        value[active] = current
        counts[active] += 1
        if router == "first" or layer == LAYERS - 1:
            break
        visible = tags[layer, active]
        if router == "gold":
            move = action(op[layer, active], current)
        elif router == "analytic":
            move = action((visible @ PROTOTYPES.T).argmax(axis=1), current)
        elif router == "learned":
            scores = np.einsum("nf,nfc->nc", features(visible), weights[current % 2])
            move = scores.argmax(axis=1)
        else:
            raise ValueError(router)
        passed = move != 2
        active = active[passed]
        if not len(active):
            break
        cells[active] = (2 * cells[active] + move[passed]) % width
    return value, counts, path


def replay(bank, query, path, counts, replacement):
    value = query[1].copy()
    for layer, visits in enumerate(path):
        active = np.where(visits >= 0)[0]
        if not len(active):
            break
        args = value[active].copy()
        if replacement is not None and layer > 0:
            cut = counts[active] == layer + 1
            args[cut] = replacement[active[cut]]
        value[active] = bank[layer, visits[active], args]
    return value


def check():
    rng = np.random.default_rng(0)
    bank = tables(rng, 8)
    query, op = task(rng, 8, 5)
    answer, counts, path = follow(bank, query, "gold", op=op, record=True)
    assert np.all(counts == 5) and np.all((path >= 0).sum(0) == counts)
    assert np.array_equal(replay(bank, query, path, counts, None), answer)
    # Two executions with the same first four private tables and different final functions.
    other = bank.copy()
    other[-1] = (other[-1] + 1) % SYMBOLS
    altered, _, route = follow(other, query, "gold", op=op, record=True)
    assert np.array_equal(path, route) and (altered != answer).all()


def benchmark(bank, query, router, weights):
    def batch():
        return follow(bank, query, router, weights)[0]
    batch()
    t = time.perf_counter()
    for _ in range(10):
        batch()
    repeats = max(10, min(12000, int(1.5 / max(time.perf_counter() - t, .00001))))

    def work():
        for _ in range(repeats):
            batch()

    return dict(queries_per_interval=repeats * len(query[0]),
                readings=[measure(work)[1] for _ in range(3)])


def main(output):
    check()
    weights, training_cost = measure(fit)
    raws, timings, summaries = [], [], []
    for width in (8, 32, 128):
        for depth in (1, 3, 5):
            rows = []
            for world in range(12):
                # Reuse the same private world across depths, with independent query streams.
                bank = tables(np.random.default_rng([924054, width, world]), width)
                rng = np.random.default_rng([924055, width, depth, world])
                query, op = task(rng, width, depth)
                gold, _, _ = follow(bank, query, "gold", op=op)
                guess, count, path = follow(bank, query, "learned", weights, record=True)
                ordinary, simple_count, _ = follow(bank, query, "analytic")
                first, one, _ = follow(bank, query, "first")
                broken = replay(bank, query, path, count, rng.integers(SYMBOLS, size=QUERIES))
                assert np.array_equal(guess, replay(bank, query, path, count, None))
                estimates = (guess, ordinary, ordinary, broken, first)
                contacts = (count, simple_count, simple_count, count, one)
                row = dict(world=world, width=width, depth=depth, queries=QUERIES, arms={})
                for method, pred, visits in zip(METHODS, estimates, contacts):
                    messages = 0 if method == "central" else 2 * int(visits.sum())
                    bits = messages * (4 + (LAYERS * width).bit_length())
                    row["arms"][method] = dict(correct=int((pred == gold).sum()),
                                               worker_lookups=int(visits.sum()), messages=messages,
                                               bits=bits, verification_checks=0)
                rows.append(row)
            raws.extend(rows)
            means = np.array([[r["arms"][m]["correct"] / QUERIES for m in METHODS] for r in rows])
            boot = np.random.default_rng(924056).integers(len(rows), size=(10000, len(rows)))
            def interval(compare):
                diff = means[:, 0] - means[:, METHODS.index(compare)]
                return [float(diff.mean()), *np.quantile(diff[boot].mean(1), [.025, .975]).tolist()]
            summaries.append(dict(width=width, depth=depth, accuracy=dict(zip(METHODS, means.mean(0))),
                                  learned_minus_severed=interval("severed"), learned_minus_analytic=interval("analytic"),
                                  learned_bits_per_query=np.mean([r["arms"]["learned"]["bits"]/QUERIES for r in rows]),
                                  learned_cells_per_query=np.mean([r["arms"]["learned"]["worker_lookups"]/QUERIES for r in rows])))
            for method in (("analytic", "learned") if (width.bit_length()+depth)//2 % 2 else ("learned", "analytic")):
                timings.append(dict(width=width, depth=depth, method=method, **benchmark(bank, query, method, weights)))
    output.write_text(json.dumps(dict(queries_per_world=QUERIES, worlds_per_width=12, training_cost=training_cost,
                                      learned_parameters=int(weights.size), summaries=summaries, worlds=raws,
                                      timings=timings), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json")
