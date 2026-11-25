"""Measure propagation of a unit-length weight update across all equivalent input pairs."""
from pathlib import Path
import json

import pilot
import numpy as np
import torch
from analyse import estimate

ROOT = Path(__file__).resolve().parent


def transfer(table, gate, triples):
    direction = table * gate
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    influence = direction @ table.T
    means = []
    for triple in triples:
        features = pilot.TRUTH[:, triple]
        equivalent = (features[:, None] == features[None, :]).all(-1)
        np.fill_diagonal(equivalent, False)
        means.append(influence[equivalent].mean())
    return float(np.mean(means))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    differences = {"same": [], "shift": []}
    rows = []
    with np.load(ROOT / "followup-trials.npz") as archive:
        for seed in range(141, 149):
            table, _ = pilot.feature_table(seed)
            triples = np.random.default_rng(seed + 4000).permutation(pilot.TRIPLES)
            history = pilot.life(table, seed, "gate", pilot.BASE_RATE, triples[:12])
            gate = archive[f"{seed}-gates"][-1]
            assert np.allclose(history[100][1], gate)
            assert np.allclose(history[100][0], archive[f"{seed}-weights"][-1])
            for family, masks in (("same", triples[16:]), ("shift", pilot.SHIFT)):
                plain = transfer(table, np.ones(12), masks)
                learned = transfer(table, gate, masks)
                differences[family].append(learned - plain)
                rows.append(dict(seed=seed, family=family, ordinary=plain, learned=learned))
    result = {"units": "logit change per unit L2 weight step toward feedback",
              "rows": rows, "learned_minus_ordinary": {k: estimate(v) for k, v in differences.items()}}
    (ROOT / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["learned_minus_ordinary"])


if __name__ == "__main__":
    main()
