"""Evaluate a frozen learner on externally defined binary languages."""
from itertools import permutations
import json
import random
from study import Machine, ROOT, phase, random_word, sweep


def external_phase(m, rng):
    data = phase(m, rng)
    if data is not None:
        return data, False
    # Reach rare periodic positives while keeping the fitted data interface identical.
    guards = {}
    while len(guards) < 12:
        w = random_word(rng,rng.randint(8,12))
        guards[w] = m.predict(w)
    pools = [{},{}]
    for n in range(14,41):
        for k in range(1,6):
            for pat in range(1<<k):
                motif = format(pat,f"0{k}b")
                word = int("1"+(motif*n)[:n],2)
                y = m.predict(word)
                if len(pools[y]) < 8:
                    pools[y][word] = y
    assert all(len(p)==8 for p in pools)
    items = list(pools[0].items())+list(pools[1].items())
    rng.shuffle(items)
    words,truth = zip(*items)
    return ((m.table(),tuple(guards.items()),words),truth), True


def corpus():
    defs=json.loads((ROOT/"tomita-machines.json").read_text())["machines"]
    machines=[Machine(1,[[1<<s for s in col] for col in spec["steps"]],spec["output"],True)
              for spec in defs]
    rng=random.Random(991501)
    worlds=[]
    repairs=0
    for a,b in permutations(machines,2):
        assert a.table()!=b.table()
        pair=[]
        for m in (a,b):
            p, repaired=external_phase(m,rng)
            pair.append(p)
            repairs+=repaired
        worlds.append((tuple(p[0] for p in pair),tuple(p[1] for p in pair),(a,b)))
    print("balanced sampling repairs",repairs,flush=True)
    return worlds,0


if __name__=="__main__":
    sweep([("tomita",corpus())],"tomita-results.json")
