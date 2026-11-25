"""Fresh lifetimes, equal rate selection, and feedback transfer to a distinct input."""
import json
from pathlib import Path

import pilot
import numpy as np
import torch
from analyse import estimate, feedback_rates

ROOT = Path(__file__).resolve().parent
RATES = (.03, .1, .3, 1., 3., 10.)
SEEDS = range(141, 149)
NAMES = ("gate", "uniform", "adam", "prior", "conventional", "context")


def run():
    meta, records = [], {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 4000)
        triples = rng.permutation(pilot.TRIPLES)
        train, dev, test = np.split(triples, [12, 16])
        assert set(map(tuple, train)).isdisjoint(map(tuple, test))
        table, _ = pilot.feature_table(seed)
        history = pilot.life(table, seed, "gate", pilot.BASE_RATE, train, save_all=True)
        records[f"{seed}-weights"] = np.vstack([cp[0] for cp in history.values()])
        records[f"{seed}-gates"] = np.stack([cp[1] for cp in history.values()])
        validation = pilot.tools(rng, dev, 128)
        tasks = {"same": pilot.tools(rng, test, 160, True),
                 "shift": pilot.tools(rng, pilot.SHIFT, 160, True)}
        for t in (0, 100):
            w, gate, state = history[t]
            options = {"gate": [("sgd", r, gate) for r in RATES],
                       "uniform": [("sgd", r, 1.) for r in RATES],
                       "adam": [("adam", r, 1.) for r in RATES],
                       "prior": [("sgd", r, np.where(np.isin(np.arange(12), pilot.COMMON), 1., low))
                                 for r in RATES for low in (0., .1, .5)]}
            for boundary, weights in (("carry", w), ("clear", w * 0)):
                cp = weights, gate, state
                chosen, scores = {}, {}
                for name, configs in options.items():
                    values = [pilot.evaluate(table, validation, cp, *cfg, clear=True)[0][:, 1:].mean()
                              for cfg in configs]
                    best = int(np.argmax(values))
                    chosen[name], scores[name] = configs[best], float(values[best])
                winner = max(("uniform", "adam"), key=scores.get)
                chosen["conventional"] = chosen[winner]
                meta.append({"seed": seed, "t": t, "boundary": boundary, "winner": winner,
                             "rates": {k: v[1] for k, v in chosen.items()}})
                for family, task in tasks.items():
                    prefix = f"{seed}-{t}-{boundary}-{family}-"
                    first = None
                    for name, cfg in chosen.items():
                        curve, fb, g = pilot.evaluate(table, task, cp, *cfg, clear=True, paired_probe=True)
                        if first is not None:
                            assert np.array_equal(g, first)
                        first = g
                        records[prefix + name] = curve
                        records[prefix + name + "-fb"] = fb
                    records[prefix + "context"] = pilot.context_curve(table, task)
    np.savez_compressed(ROOT / "followup-trials.npz", **records)
    (ROOT / "followup-rates.json").write_text(json.dumps(meta, indent=2) + "\n")


def summarise():
    with np.load(ROOT / "followup-trials.npz") as raw:
        data = {key: raw[key] for key in raw.files}
    result = {}
    for family in ("same", "shift"):
        result[family] = {}
        for t in (0, 100):
            for boundary in ("carry", "clear"):
                def get(seed, name):
                    return data[f"{seed}-{t}-{boundary}-{family}-{name}"]
                score = {name: np.stack([get(s, name) for s in SEEDS]) for name in NAMES}
                avg = {name: arr[:, :, 1:].mean((1, 2)) for name, arr in score.items()}
                delta = {name: estimate(avg["gate"] - arr) for name, arr in avg.items()}
                fb = [feedback_rates(get(s, "gate-fb"), get(s, "conventional-fb")) for s in SEEDS]
                outcome = {}
                for k in ("harm", "repair"):
                    counts = np.array([row[k] for row in fb])
                    valid = counts[:, 0] > 0
                    outcome[k] = {"counts": counts.tolist(), "pooled": counts.sum(0).tolist(),
                                  "paired": estimate((counts[valid, 1] - counts[valid, 2]) / counts[valid, 0])}
                result[family][f"{t}-{boundary}"] = {
                    "curves": {name: arr.mean((0, 1)).tolist() for name, arr in score.items()},
                    "accuracy": {name: float(arr.mean()) for name, arr in avg.items()},
                    "gate_minus": delta, "feedback_vs_conventional": outcome}
                if t == 100:
                    early = result[family][f"0-{boundary}"]["gate_minus"]["uniform"]["by_seed"]
                    result[family][f"100-{boundary}"]["lifetime_gain_vs_uniform"] = estimate(
                        np.subtract(delta["uniform"]["by_seed"], early))
    (ROOT / "followup-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for family, blocks in result.items():
        for name, block in blocks.items():
            print(family, name, {k: round(100 * v, 2) for k, v in block["accuracy"].items()})
            if name.startswith("100"):
                print("gate minus conventional", block["gate_minus"]["conventional"])
                print("feedback", {k: (v["pooled"], v["paired"]["ci"])
                                   for k, v in block["feedback_vs_conventional"].items()})


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run()
    summarise()
