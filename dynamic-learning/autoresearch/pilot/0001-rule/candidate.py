from itertools import product


ACTION_ROWS = tuple((0, *bits) for bits in product((0, 1), repeat=3))[1:] + ((1, 1, 1, 1),)
ACTION_WEIGHTS = (1,) * 7 + (2,)


def _label(row, scope):
    return row[scope - 1] if scope else 0


def _beliefs(observation):
    history = observation['history']
    source = observation['source']
    endorses = observation['endorses']
    weights = [0.0] * 5
    for reliabilities in ((0.85, 0.55), (0.55, 0.85)):
        likelihood = 1.0
        for count, rate in zip(history, reliabilities):
            likelihood *= rate ** count * (1 - rate) ** (8 - count)
        trust = 0.8 * reliabilities[source] + 0.2 * reliabilities[1 - source]
        for scope in range(5):
            weights[scope] += likelihood * (trust if bool(scope) == endorses else 1 - trust)
    for index, bit in observation['receipts']:
        row = observation['menu'][index]
        for scope in range(5):
            if _label(row, scope) != bit:
                weights[scope] = 0.0
    return tuple(weights)


def _reversal_mass(weights, row):
    return sum(weight for scope, weight in enumerate(weights) if _label(row, scope))


def forecast(observation, row):
    weights = _beliefs(observation)
    return _reversal_mass(weights, row) / sum(weights)


def _terminal(weights):
    total = sum(weights)
    loss = 0.0
    memory = 0
    for bit, (row, importance) in enumerate(zip(ACTION_ROWS, ACTION_WEIGHTS)):
        reverse = _reversal_mass(weights, row)
        keep = total - reverse
        loss += importance * min(reverse, keep)
        if reverse > keep:
            memory |= 1 << bit
    return loss, memory


def _plan(weights, menu, available, remaining):
    loss, memory = _terminal(weights)
    best_loss, best_cost = loss, 0.0
    action = bytes([memory]).hex()
    total = sum(weights)
    if remaining == 0 or loss == 0:
        return best_loss, best_cost, action

    tolerance = 1e-12 * total
    for index in available:
        row = menu[index]
        branches = tuple(
            tuple(weight if _label(row, scope) == bit else 0.0
                  for scope, weight in enumerate(weights))
            for bit in (0, 1)
        )
        if not all(any(branch) for branch in branches):
            continue
        rest = tuple(other for other in available if other != index)
        left, right = (_plan(branch, menu, rest, remaining - 1) for branch in branches)
        probe_loss = left[0] + right[0]
        probe_cost = total + left[1] + right[1]
        if (probe_loss < best_loss - tolerance or
                (abs(probe_loss - best_loss) <= tolerance and probe_cost < best_cost - tolerance)):
            best_loss, best_cost, action = probe_loss, probe_cost, index
    return best_loss, best_cost, action


def step(observation):
    menu = observation['menu']
    checked = {index for index, _ in observation['receipts']}
    available = tuple(index for index in range(len(menu)) if index not in checked)
    remaining = max(0, 2 - len(observation['receipts']))
    return _plan(_beliefs(observation), menu, available, remaining)[2]


def answer(memory_hex, row, old):
    if not memory_hex:
        return 0
    same_invoice, supplier, region, importer = row
    if same_invoice:
        bit = 7
    else:
        pattern = 4 * supplier + 2 * region + importer
        if pattern == 0:
            return 0
        bit = pattern - 1
    return (int(memory_hex, 16) >> bit) & 1