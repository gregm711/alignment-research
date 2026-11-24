"""Infer consistent rule scopes, using the reporter as a soft prior."""
from math import log

MODE = 'random'


def labels(row):
    return (0, *row)


def beliefs(obs):
    weights = [1.] * 5
    trust = (1 + obs['history'][obs['source']]) / 10
    for scope in range(5):
        weights[scope] *= trust if bool(scope) == obs['endorses'] else 1-trust
        if any(labels(obs['menu'][i])[scope] != y for i, y in obs['receipts']):
            weights[scope] = 0
    return weights


def forecast(obs, row):
    if MODE == 'keep':
        return .05
    weights = beliefs(obs)
    return sum(w*y for w,y in zip(weights,labels(row))) / sum(weights)


def step(obs):
    checked = {i for i, _ in obs['receipts']}
    weights = beliefs(obs)
    if MODE != 'keep' and len(checked) < 2 and sum(w > 0 for w in weights) > 1:
        options = [i for i in range(len(obs['menu'])) if i not in checked]
        if MODE == 'random':
            return options[0]  # The menu is shuffled independently in each incident.
        def information(i):
            p = sum(w*y for w,y in zip(weights, labels(obs['menu'][i]))) / sum(weights)
            return -sum(v*log(v) for v in (p,1-p) if v > 0)
        probe = max(options,key=information)
        if information(probe) > 0:
            return probe
    if MODE == 'keep':
        return ''
    return bytes([max(range(5),key=weights.__getitem__)]).hex()


def answer(memory_hex, row, old):
    return labels(row)[int(memory_hex,16)] if memory_hex else 0
