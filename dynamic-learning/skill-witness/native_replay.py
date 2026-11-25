"""Isolate durable memory on identical acquired synthetic evidence."""
import json

import numpy as np

from native_model import HERE, K, load_model, stream_step


def score(model, state, future, world):
    p = model.prior(state)
    mass = np.stack([(p * (future == label)).sum(1) for label in range(6)])
    return (mass.argmax(0) == future[:, world]).mean()


def main():
    model = load_model()
    rng = np.random.default_rng(912571)
    size = 384
    worlds = rng.integers(K, size=size)
    retained = np.stack([model.initial() for _ in range(size)])
    scores = np.empty((size, 3, 3, 3))
    changed = np.zeros((size, 3), dtype=bool)
    for rev in range(4):
        old = worlds
        worlds, records, reported = stream_step(rng, worlds)
        carried = np.stack([model.begin(state) for state in retained])
        if rev:
            changed[:, rev - 1] = worlds != old
        for j in range(size):
            query, future = records[j, :6], records[j, 6:]
            states = [carried[j].copy(), model.initial(), carried[(j + 1) % size].copy()]
            used = []
            for t in range(3):
                if rev:
                    scores[j, rev - 1, t] = [score(model, s, future, worlds[j]) for s in states]
                if t < 2:
                    q = model.choose(states[0], query, used)
                    used.append(q)
                    mask = query[q] == query[q, reported[j, q]]
                    states = [model.update(s, mask) for s in states]
            retained[j] = states[0]
    res = {}
    idx = np.random.default_rng(718).integers(size, size=(4000, size))
    for key, mask in (('all', np.ones_like(changed)), ('changed', changed), ('unchanged', ~changed)):
        total = mask.sum()
        means = scores[mask].mean(0)
        gaps = ((scores[..., 0] - scores[..., 1]) * mask[:, :, None]).sum(1)
        boot = gaps[idx].sum(1) / mask.sum(1)[idx].sum(1)[:, None]
        res[key] = dict(n=int(total), by_check_and_arm=means.tolist(), carry_minus_reset_bounds=np.quantile(boot, [.025,.975], axis=0).tolist())
    out = dict(seed=912571, stream_count=size, checks=[0,1,2], arms=['retained','reset','transplanted'], results=res)
    (HERE / 'native-transplants.json').write_text(json.dumps(out, indent=2)+'\n')
    np.savez_compressed(HERE / 'native-transplants.npz', accuracy=scores, actual_change=changed)


if __name__ == '__main__':
    main()
