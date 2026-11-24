"""A paired transit decision. Public features stop at the router boundary."""
from itertools import product
import random

import numpy as np


ACTIONS = ('stop', 'plan', 'ask', 'both')
FEATURES = ('alert_stages', 'other_stages', 'gap', 'probability', 'delay',
            'abs_expected_gap', 'probe_value', 'correction_gap')


def line(rng, shifted):
    stages = rng.randint(8, 14) if shifted else rng.randint(4, 9)

    def edge():
        cost = rng.randint(1, 9)
        if shifted and rng.random() < .15:
            cost += 10
        return cost

    return [[(edge(), edge()) for _ in range(3)] for _ in range(stages)]


def greedy(graph):
    lane = total = spread = 0
    for stage in graph:
        stay, move = stage[lane]
        total += min(stay, move)
        spread += abs(stay - move)
        if move < stay:
            lane = (lane + 1) % 3
    return [1, len(graph), total, spread]


def shortest(graph):
    remaining = [0, 0, 0]
    for stage in reversed(graph):
        remaining = [min(stay + remaining[i], move + remaining[(i+1) % 3])
                     for i, (stay, move) in enumerate(stage)]
    return remaining[0]


def maps(start, count, shifted=False):
    observed, exact, alerts, probabilities, delays = [], [], [], [], []
    for seed in range(start, start + count):
        rng = random.Random(seed)
        lines = [line(rng, shifted) for _ in range(2)]
        observed.append([greedy(g) for g in lines])
        exact.append([shortest(g) for g in lines])
        alerts.append(rng.randrange(2))
        probabilities.append(rng.choice((.1, .3, .65, .9) if shifted else (.15, .35, .5, .8)))
        delays.append(rng.choice((2, 5, 9, 16) if shifted else (3, 7, 12, 20)))
    return (np.array(observed), np.array(exact), np.array(alerts),
            np.array(probabilities), np.array(delays))


def costs(data, beta, raw=False):
    observed, exact, alert, p, delay = data
    stages, path = observed[:, :, 1], observed[:, :, 2]
    cheap = path if raw else np.clip(observed @ beta, stages, path)
    i = np.arange(len(p))
    shock = (np.arange(2) == alert[:, None]) * delay[:, None]
    future = np.stack((exact, exact + shock), axis=1)
    weights = np.column_stack((1-p, p))
    reference = future.min(axis=2)
    excess, success, choices = [], [], []
    for act in ACTIONS:
        base = exact if act in ('plan', 'both') else cheap
        if act in ('ask', 'both'):
            predictions = np.stack((base, base + shock), axis=1)
        else:
            predictions = np.repeat((base + p[:, None]*shock)[:, None, :], 2, axis=1)
        choice = predictions.argmin(axis=2)
        realized = np.take_along_axis(future, choice[:, :, None], axis=2)[:, :, 0]
        regret = realized - reference
        excess.append((regret*weights).sum(axis=1))
        success.append(((regret == 0)*weights).sum(axis=1))
        choices.append(choice)
    gap = cheap[i, alert] - cheap[i, 1-alert]
    advantage = np.minimum(gap + p*delay, 0) - (
        (1-p)*np.minimum(gap, 0) + p*np.minimum(gap + delay, 0))
    explicit_value = ((cheap + p[:, None]*shock).min(axis=1)
                      - (1-p)*cheap.min(axis=1) - p*(cheap + shock).min(axis=1))
    assert np.allclose(advantage, explicit_value) and np.min(advantage) > -1e-10
    correction = path - cheap
    x = np.column_stack((stages[i, alert], stages[i, 1-alert], gap, p, delay,
                         abs(gap + p*delay), advantage, correction[i, alert]-correction[i, 1-alert]))
    excess = np.array(excess).T
    assert np.all(excess >= 0) and np.all(excess[:, 3] == 0)
    return dict(x=x, cheap=cheap, excess=excess, score=excess+[0,.5,.5,1],
                success=np.array(success).T, choices=np.stack(choices, axis=1),
                edges=stages.sum(axis=1))


def control():
    for seed in range(25):
        graph = line(random.Random(seed), False)
        paths = []
        for actions in product((0, 1), repeat=len(graph)):
            lane = travel = 0
            for stage, move in zip(graph, actions):
                travel += stage[lane][move]
                lane = (lane + move) % 3
            paths.append(travel)
        assert shortest(graph) == min(paths) <= greedy(graph)[2]
