"""Chronological evaluation. Only paid receipts enter a learner."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from direct import Direct
from panel import _input_key, _rank_key
from query import MODEL, features

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/alignment-research/skill-witness"


def split(panel):
    inputs = panel["inputs"]
    keys = {i: _input_key(x) for i, x in inputs.items()}
    public, reserved = {}, set()
    for name, series in sorted(panel["producers"].items()):
        first = next(v for v in series if v is not None)
        ids = sorted(first, key=lambda i: _rank_key(keys[i]))
        assert len(set(keys[i] for i in ids)) == len(ids)
        public[name] = ids[:10]
        reserved.update(keys[i] for i in public[name])
    return public, reserved, keys


def run(panel, project, arm, k):
    start = time.monotonic()
    explicit = arm in ("ordinary", "codegen")
    if explicit:
        from comparator import Comparator
        agent = Comparator(model=arm == "codegen")
    else:
        agent = Direct(32 if arm == "wide" else 8)
    weights = np.array(json.loads(MODEL.read_text())["weights"])
    public, reserved, keys = split(panel)
    rows, previous = [], {}
    for t, release in enumerate(panel["releases"]):
        for name, series in sorted(panel["producers"].items()):
            labels = series[t]
            if labels is None:
                continue
            qs = public[name]
            hs = sorted(i for i in labels if keys[i] not in reserved)
            assert len(qs) == 10 and hs and not set(qs).intersection(hs)
            xs = [panel["inputs"][i] for i in qs]
            setup = name not in previous
            checks, receipts, writes, accepted = [], [], 0, False
            if setup:
                checks = list(range(len(qs)))
                receipts = [(x, labels[i]) for x, i in zip(xs, qs)]
                if explicit:
                    agent.learn(name, receipts, intro=True)
                else:
                    for x, y in receipts:
                        agent.write(name, x, y)
                writes = len(receipts)
            else:
                for _ in range(k):
                    available = [j for j in range(len(qs)) if j not in checks]
                    if explicit:
                        j = agent.choose(name, xs, checks)
                    elif arm == "ordered":
                        j = available[0]
                    else:
                        scores = [np.dot(features(agent, name, x), weights) for x in xs]
                        if arm == "swapped":
                            scores = np.roll(scores, 3)
                        j = max(available, key=lambda n: scores[n])
                    assert j in available
                    forecast = agent.predict(name, xs[j])
                    checks.append(j)
                    value = labels[qs[j]]
                    receipts.append((xs[j], value))
                    if forecast == value:
                        accepted = True
                        break
                    if explicit:
                        agent.learn(name, receipts, failed_idx=j)
                    elif arm != "frozen":
                        agent.write(name, xs[j], value)
                    writes += arm != "frozen"
            gold = [labels[i] for i in hs]
            pred = [agent.predict(name, panel["inputs"][i]) for i in hs]
            correct = sum(a == b for a, b in zip(pred, gold))
            success = correct == len(hs)
            eligible = broken = 0
            changed = False
            if not setup:
                old_ids, old_gold, old_pred = previous[name]
                assert old_ids == hs
                changed = old_gold != gold
                for og, op, g, p in zip(old_gold, old_pred, gold, pred):
                    if og == op == g:
                        eligible += 1
                        broken += p != g
            previous[name] = (hs, gold, pred)
            rows.append(dict(producer=name, time=t, release=release, setup=setup,
                             checks=len(checks), queries=checks, writes=writes,
                             success=success, correct=correct, total=len(hs),
                             previously_correct=eligible, broken=broken, accepted=accepted,
                             false_accept=accepted and not success, changed=changed))
    memory = agent.bytes() if explicit else agent.state.nbytes
    result = dict(project=project, arm=arm, k=k, rows=rows, state_bytes=memory,
                query_weight_bytes=0 if explicit or arm == "ordered" else weights.nbytes,
                model_calls=getattr(agent,"calls",0), model_tokens=getattr(agent,"tokens",0),
                duration_seconds=round(time.monotonic()-start,2))
    if arm == "codegen":
        result["codegen_cost"] = agent.codegen_cost()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("project", choices=["dates","phones"])
    p.add_argument("arm", choices=["native","ordered","frozen","wide","swapped","ordinary","codegen"])
    p.add_argument("k", type=int, choices=[1,2,3])
    args = p.parse_args()
    data = json.loads((CACHE / (args.project + ".json")).read_text())
    out = run(data, args.project, args.arm, args.k)
    dest = HERE / "results" / f"{args.project}-{args.arm}-{args.k}.json"
    dest.parent.mkdir(exist_ok=True)
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(out, separators=(",", ":"))+"\n")
    temp.replace(dest)
    print(dest.name, out["duration_seconds"])
