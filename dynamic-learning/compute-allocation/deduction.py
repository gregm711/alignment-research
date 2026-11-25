"""Frozen switch policy and a counterfactual comparison of internal deductions."""
from collections import Counter
from functools import partial
import json
import random
import statistics
import sys

from correction import correct, external
from feedback import BUDGETS, ROOT, WIDTHS, advance, dataset, grade, infer, measure, parity, run

deduce = partial(correct, deduce=True)
LIMITS = (0, 2, None)


def route(n, cap, limit, old, *args):
    if isinstance(limit, dict):
        limit = limit[key(n, cap, old)]
    if limit == 0:
        return infer(n, cap, None, old, *args)
    return deduce(n, cap, limit, old, *args)


def key(n, cap, old):
    return f"{n}/{cap}/{int(old.bit_count() > 3)}"


def train():
    policy = {}
    for n in WIDTHS:
        panels = [dataset(n, 810000, 24, True), dataset(n, 820000, 24, False)]
        for cap in BUDGETS:
            scores = {}
            for idx, limit in enumerate(LIMITS):
                for worlds, labels in panels:
                    for j, world in enumerate(worlds):
                        old = 0
                        for k, (mask, witness, xs) in enumerate(world):
                            pred, _, cnt = route(n, cap, limit, old, witness, lambda x: parity(mask, x),
                                                  691000+2*j+k, None, False)
                            outputs = [advance(n, pred, x, t) for x in xs for t in (3,7)]
                            pair = scores.setdefault((key(n,cap,old), idx), [0,0])
                            pair[0] += outputs == labels[j][k]
                            pair[1] -= cnt["probes"]
                            old = mask
            for old in (0, 15):
                state = key(n, cap, old)
                winner = max(range(3), key=lambda i: (*scores.get((state,i), (0,0)), -i))
                policy[state] = LIMITS[winner]
    return policy


def evaluate(domains):
    policy = json.loads((ROOT/"deduction-policy.json").read_text())["policy"]
    results = {}
    for domain in domains:
        for n in WIDTHS:
            if domain == "reveng":
                masks = json.loads((ROOT/"reveng-polynomials.json").read_text())["masks"].get(str(n),[])
                if len(masks) < 2:
                    continue
                worlds, labels = external(n, masks)
            else:
                worlds, labels = dataset(n, 910000 if domain == "three" else 920000, 96, domain=="three")
            for cap in BUDGETS:
                seed = 853100
                fitted = run(worlds,n,cap,[policy]*2,seed,infer_fn=route)
                logs = [[phase[1] for phase in row] for row in fitted]
                arms = [("basis",infer,None),("split",correct,None),("deduce",deduce,None),
                        ("switch2",route,2),("fitted",route,policy),("replay",infer,None)]
                random.Random(7634+n+cap).shuffle(arms)
                for name, fn, limit in arms:
                    def perform():
                        return run(worlds,n,cap,[limit]*2,seed,logs if name=="replay" else None,infer_fn=fn)
                    rows, timing = measure(perform)
                    repeats = max(1,min(512,round(.035/timing["wall_seconds"])))
                    def batch():
                        for _ in range(repeats):
                            perform()
                    costs = []
                    for _ in range(3):
                        _, cost = measure(batch)
                        costs.append({k:v/repeats if v is not None else None for k,v in cost.items()})
                    pairs = grade(rows, labels)
                    solved = sum(map(all,pairs))
                    counts = sum((p[2] for row in rows for p in row), Counter())
                    energy = [t["cpu_joules"] for t in costs if t["cpu_joules"] is not None]
                    results[f"{domain}/{n}/{cap}/{name}"] = dict(solved=solved,phases=pairs,counts=dict(counts),
                        costs=costs,repeats=repeats,joules_per_solve=statistics.median(energy)/solved
                        if solved and energy else None)
        print(domain,flush=True)
    (ROOT/("deduction-"+"-".join(domains)+".json")).write_text(json.dumps(results,separators=(",",":"))+"\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        fitted, cost = measure(train)
        (ROOT/"deduction-policy.json").write_text(json.dumps(dict(policy=fitted,training=cost))+"\n")
    else:
        assert sys.argv[1:] in (["three","dense"], ["reveng"])
        evaluate(sys.argv[1:])
