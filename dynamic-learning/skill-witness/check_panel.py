"""Run the episode adapter's causal and split invariants on synthetic fixtures only."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from panel import episodes, _input_key


def fixture(phone):
    inputs, producers = {}, {}
    for j, producer in enumerate(("alpha", "beta")):
        rows = {}
        for i in range(36):
            # Share field values across producers, keeping provider-specific input IDs.
            key = f"{producer}-{(31 * i + j) % 37}"
            if phone:
                n = str(1730000000 + i * 8207)
                fields, value = {"n": n}, f"({n[:3]}) {n[3:6]}-{n[6:]}"
            else:
                d, m, y = i % 27 + 1, i % 12 + 1, 1990 + i
                fields, value = (
                    {"y": str(y), "m": str(m), "d": str(d)},
                    f"{d:02}/{m:02}/{y}",
                )
            inputs[key] = fields
            rows[key] = value
        producers[producer] = [rows, None if j else rows, rows.copy()]
    return {
        "project": "synthetic",
        "inputs": inputs,
        "producers": producers,
        "releases": ["a", "b", "c"],
    }


def test_panel(path, phone):
    data = fixture(phone)
    path.write_text(json.dumps(data))
    before = list(episodes(path))
    assert [x[4] for x in before] == ["b", "c", "c"]
    reserved, graded = set(), set()
    for name, init_n, candidates, idx, release, qids, tids, pred, gold in before:
        assert init_n == len(qids) == 10
        assert len(tids) == 26 and pred.shape == (36, 8) and gold.shape == (36,)
        assert len(candidates) == len(set(map(tuple, pred.T))) == 8
        assert np.array_equal(pred[:, 0], gold)
        assert json.loads(json.dumps(candidates)) == candidates
        reserved.update(_input_key(data["inputs"][i]) for i in qids)
        graded.update(_input_key(data["inputs"][i]) for i in tids)
        # Corrupt every unbought calibration label and every subsequent outcome.
        for i in tids:
            data["producers"][name][0][i] = "corrupt"
        data["producers"][name][idx] = {i: "counterfactual" for i in qids + tids}
    assert not reserved & graded
    path.write_text(json.dumps(data))
    after = list(episodes(path))
    for old, new in zip(before, after, strict=True):
        assert old[:7] == new[:7] and np.array_equal(old[7], new[7])
        assert np.all(new[8] == "counterfactual")


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        for phone in (False, True):
            test_panel(Path(temp) / "synthetic.json", phone)
    print(
        "Synthetic date and phone causality, cost, disjointness, and alias checks passed."
    )
