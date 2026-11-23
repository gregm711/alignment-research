"""Ouro analysis (CPU): written reasoning at 2 and 4 loops on the same fact sheets.

    modal volume get loop-legibility ouro.json results/ouro.json
    python3 ouro_analyse.py          # writes results/ouro-summary.json
"""
import collections
import json
import pathlib
import random
import re

from tasks import items

HERE = pathlib.Path(__file__).parent / "results"
import sys

NAME, SEED = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("ouro", 5)  # e.g. python3 ouro_analyse.py ouro-hard 8
rows = json.load(open(HERE / f"{NAME}.json"))["rows"]
CHAINS = sorted({r["chain"] for r in rows})
pairs = {(c, it["sheet"]): it["pairs"] for c in CHAINS for it in items(SEED, 40, c)}


def strict(r):
    """Step j counts if one sentence of the thinking holds both the previous step's word and this one's,
    after copies of the fact sheet's own rows are removed."""
    body = r["text"].partition("</think>")[0].lower()
    for p in pairs[(r["chain"], r["sheet"])]:
        body = body.replace(p.lower(), " ")
    sents = re.split(r"[.!?\n]+", body)
    has = lambda s, w: re.search(rf"\b{w}\b", s)
    return [any(has(s, a) and has(s, b) for s in sents) for a, b in zip(r["steps"][:-2], r["steps"][1:-1])]


def boot(vals, n=2000):
    rng = random.Random(0)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return [round(sum(vals) / len(vals), 3), round(bs[int(0.025 * n)], 3), round(bs[int(0.975 * n)], 3)]


out = {}
for chain in CHAINS:
    by = collections.defaultdict(dict)
    for r in rows:
        if r["chain"] == chain:
            by[r["sheet"]][r["loops"]] = {**r, "strict": strict(r)}
    both = [s for s in by.values() if len(s) == 2]
    cell = {"sheets": len(both)}
    for k in (2, 4):
        rs = [s[k] for s in both]
        cell[f"{k} loops"] = {"correct": sum(r["correct"] for r in rs), "finished": sum(r["finished"] for r in rs),
                              "mean_thinking_tokens": round(sum(r["thinking_tokens"] for r in rs) / len(rs), 1),
                              "steps_named": f"{sum(sum(r['named']) for r in rs)}/{sum(len(r['named']) for r in rs)}",
                              "steps_named_strict": f"{sum(sum(r['strict']) for r in rs)}/{sum(len(r['strict']) for r in rs)}"}
    for label, sel in (("all sheets", both), ("sheets correct and finished at both", [s for s in both if all(s[k]["correct"] and s[k]["finished"] for k in (2, 4))])):
        if sel:
            mean2 = sum(s[2]["thinking_tokens"] for s in sel) / len(sel)
            cell[label] = {"n": len(sel), "thinking_tokens_4_minus_2": boot([s[4]["thinking_tokens"] - s[2]["thinking_tokens"] for s in sel]),
                           "as_share_of_2_loop_mean": round(sum(s[4]["thinking_tokens"] - s[2]["thinking_tokens"] for s in sel) / len(sel) / mean2, 3),
                           "strict_steps_named_4_minus_2": boot([sum(s[4]["strict"]) / len(s[4]["strict"]) - sum(s[2]["strict"]) / len(s[2]["strict"]) for s in sel]),
                           "correct_4_minus_2": boot([s[4]["correct"] - s[2]["correct"] for s in sel])}
    out[f"chain {chain}"] = cell
print(json.dumps(out, indent=1))
(HERE / f"{NAME}-summary.json").write_text(json.dumps(out, indent=1))
