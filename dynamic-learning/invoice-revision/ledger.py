"""Generated invoices, observable relations, and exact calendar and evidence controls."""
from datetime import date, timedelta
from itertools import permutations, product
import struct

import numpy as np


# Canonical relation order: focal invoice, its supplier, its importer, other.
SCOPES = np.tri(4, 4, -1, dtype=bool)
ORDERS = list(permutations(range(3)))
QUERIES = [6, 7, 8, 0, 3, 4, 5]
PATHS = [()] + [
    tuple(zip(order, bits)) for size in (1, 2)
    for order in permutations(range(3), size) for bits in product(range(2), repeat=size)
]
PATH_ID = {path: i for i, path in enumerate(PATHS)}
FEATURES = np.zeros((len(PATHS), 6))
for i, path in enumerate(PATHS):
    for probe, bit in path:
        FEATURES[i, 2 * probe + bit] = 1


def relations(rows, anchor):
    focal, = rows[rows[:, 0] == anchor]
    return np.select([rows[:, 0] == anchor, rows[:, 1] == focal[1],
                      rows[:, 2] == focal[2]], [0, 1, 2], default=3)


def generate(seed, count):
    rng = np.random.default_rng(seed)
    records, roles, menus, outcomes, answers = [], [], [], [], []
    for _ in range(count):
        ids = rng.choice(65536, 15, replace=False)
        supplier = ids[9:13][[0, 0, 1, 0, 2, 3, 0, 1, 3]]
        importer = ids[13:][[0, 0, 0, 0, 0, 1, 0, 0, 1]]
        raw = np.array([rng.choice(np.arange(1, 13), 2, replace=False) for _ in range(9)])
        # Include one duplicate raw date that belongs to a different invoice.
        raw[3] = raw[0]
        installed = rng.integers(2)
        terms = rng.choice([7, 14, 30, 45], 9)
        year = rng.integers(2020, 2031)
        by_role = np.column_stack([ids[:9], supplier, importer, raw, terms,
                                   np.full(9, year), np.full(9, installed)])
        order = rng.permutation(9)
        rows = by_role[order].astype(np.int32)
        role = np.argsort(order)
        menu = rng.permutation(3)
        rel = relations(rows, ids[0])
        # Generate world truth by applying its actual identifier predicate.
        reverse = np.array([np.zeros(9, bool), rows[:, 0] == ids[0],
                            rows[:, 1] == supplier[0], rows[:, 2] == importer[0]])
        assert np.array_equal(reverse, SCOPES[:, rel])
        assert np.array_equal(reverse[:, role[:3]], SCOPES[:, :3])
        assert not set(role[QUERIES[:3] + QUERIES[4:]]) & set(role[:3])
        calendar = np.zeros((9, 2, 2), dtype=np.int32)
        for i, (_, _, _, a, b, days, y, mode) in enumerate(rows.tolist()):
            for swap in (0, 1):
                month, day = (b, a) if mode ^ swap else (a, b)
                parsed = date(y, month, day)
                calendar[i, swap] = [parsed.toordinal(), (parsed + timedelta(days=days)).toordinal()]
        assert np.all(calendar[:, 0] != calendar[:, 1])
        # The visible ledger, focal ID, menu and cached values share one backing
        # instance across the four forks. Only reference answers are forked.
        records.append(rows)
        roles.append(role)
        menus.append(menu)
        outcomes.append(reverse)
        answers.append(calendar)
    return dict(rows=np.array(records), roles=np.array(roles, dtype=np.uint8),
                menus=np.array(menus, dtype=np.uint8), truth=np.array(outcomes),
                calendar=np.array(answers))


def receipts(data, ledger, world, order):
    indices = data['roles'][ledger]
    return tuple((q, int(data['truth'][ledger, world, indices[q]])) for q in order)


def posterior(path, likelihood):
    mass = np.ones(4)
    for q, bit in path:
        mass *= likelihood[:, q] if bit else 1 - likelihood[:, q]
    return mass / mass.sum() if mass.sum() else np.zeros(4)


def store_state(name, anchor, path, weights=None):
    header = struct.pack('<H', anchor)
    if name == 'explicit':
        code = sum((bit + 1) * 3**q for q, bit in path)
        return header + bytes([code])
    state = np.zeros(3, dtype='<f4')
    selected = sorted(q for q, _ in path)
    for q, bit in path:
        if name == 'dense':
            state += weights[2*q + bit]
        else:
            state[q] += weights[2*q + bit, q]
    if name == 'dense':
        return header + state.tobytes()
    return header + bytes([sum(1 << q for q in selected)]) + state[selected].tobytes()


def restore(name, blob, readout=None, bias=None):
    anchor, = struct.unpack('<H', blob[:2])
    if name == 'explicit':
        code = blob[2]
        possible = [w for w in range(4) if all(
            code // 3**q % 3 in (0, 1 + int(q < w)) for q in range(3))]
        assert possible
        return anchor, np.arange(4) < possible[0]
    if name == 'dense':
        state = np.frombuffer(blob, dtype='<f4', offset=2)
    else:
        state = np.zeros(3, dtype=np.float32)
        occupied = [q for q in range(3) if blob[2] & (1 << q)]
        state[occupied] = np.frombuffer(blob, dtype='<f4', offset=3)
    return anchor, readout @ state + bias > 0
