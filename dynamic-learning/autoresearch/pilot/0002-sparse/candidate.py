"""Price receipts against expected balanced decision error."""
from functools import lru_cache

PROBE_COST = 0.0125
# Equal weight for fresh and booked sets. The booked set assigns one fifth
# of its mass to the reported invoice. Other patterns are uniform.
ROW_MASS = (0.1125,) * 8 + (0.1,)
# The assumed reversal prevalence is 0.35 under the uniform scope prior.
REPAIR_WEIGHT = 0.5 / 0.35
RETENTION_WEIGHT = 0.5 / 0.65


def row_index(row):
    same_invoice, supplier, region, importer = row
    if same_invoice:
        return 8
    return supplier + 2 * region + 4 * importer


# Bit s is set when scope s predicts a reversal for this row pattern.
LABEL_MASKS = tuple(pattern << 2 for pattern in range(8)) + (30,)


def posterior(observation):
    first, second = observation['history']
    first_high = 0.85 ** first * 0.15 ** (8 - first)
    first_high *= 0.55 ** second * 0.45 ** (8 - second)
    second_high = 0.55 ** first * 0.45 ** (8 - first)
    second_high *= 0.85 ** second * 0.15 ** (8 - second)
    initial_high = first_high / (first_high + second_high)
    if observation['source'] == 1:
        initial_high = 1 - initial_high
    trust = 0.55 + 0.30 * (0.20 + 0.60 * initial_high)
    endorse = observation['endorses']
    weights = tuple(trust if bool(scope) == endorse else 1 - trust
                    for scope in range(5))
    possible = 31
    for index, bit in observation['receipts']:
        mask = LABEL_MASKS[row_index(observation['menu'][index])]
        possible &= mask if bit else 31 ^ mask
    return weights, possible


def mass(weights, scopes):
    return sum(weight for scope, weight in enumerate(weights)
               if scopes & (1 << scope))


def forecast(observation, row):
    weights, possible = posterior(observation)
    mask = LABEL_MASKS[row_index(row)]
    return mass(weights, possible & mask) / mass(weights, possible)


def step(observation):
    weights, possible = posterior(observation)
    receipts = observation['receipts']
    checked = {index for index, _ in receipts}
    # Keep the first index for each distinct label pattern.
    probes = {}
    for index, row in enumerate(observation['menu']):
        if index not in checked:
            probes.setdefault(LABEL_MASKS[row_index(row)], index)

    @lru_cache(maxsize=None)
    def total(scopes):
        return mass(weights, scopes)

    @lru_cache(maxsize=None)
    def terminal(scopes):
        loss = 0.0
        actions = 0
        for pattern, (mask, frequency) in enumerate(zip(LABEL_MASKS, ROW_MASS)):
            keep = REPAIR_WEIGHT * total(scopes & mask)
            reverse = RETENTION_WEIGHT * total(scopes & (31 ^ mask))
            if reverse < keep:
                actions |= 1 << pattern
            loss += frequency * min(keep, reverse)
        return loss, actions

    @lru_cache(maxsize=None)
    def plan(scopes, depth):
        best_cost, _ = terminal(scopes)
        best_index = None
        if depth:
            price = PROBE_COST * total(scopes)
            for mask, index in probes.items():
                positive = scopes & mask
                negative = scopes & (31 ^ mask)
                if not positive or not negative:
                    continue
                cost = price + plan(positive, depth - 1)[0]
                cost += plan(negative, depth - 1)[0]
                if cost < best_cost - 1e-12:
                    best_cost, best_index = cost, index
        return best_cost, best_index

    _, index = plan(possible, max(0, 2 - len(receipts)))
    if index is not None:
        return index
    _, actions = terminal(possible)
    size = (actions.bit_length() + 7) // 8
    return actions.to_bytes(size, 'little').hex()


def answer(memory_hex, row, old):
    actions = int.from_bytes(bytes.fromhex(memory_hex), 'little')
    return (actions >> row_index(row)) & 1
