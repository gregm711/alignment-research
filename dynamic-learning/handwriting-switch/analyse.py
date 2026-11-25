"""Pair all comparisons over the same lifetimes and held-out handwriting episodes."""
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("change_accuracy", "false_harm", "true_repair", "reach_80", "balanced_accuracy")


def summaries(counts, seed_weights, episode_weights):
    # seed x digit pair x episode x budget x count
    a = counts.astype(float)
    changed, stable, repaired, eligible = a.transpose(4, 0, 1, 2, 3)
    measures = np.stack([changed/32, 1-stable/32, repaired, eligible,
                         np.maximum.accumulate(changed >= 26, axis=-1), (changed+stable)/64], axis=-1)
    def ratios(v):
        assert (v[..., 3] > 0).all()
        return np.stack([v[..., 0], v[..., 1], v[..., 2]/v[..., 3], v[..., 4], v[..., 5]], axis=-1)
    pair_values = ratios(measures.mean((0,2)))
    mean = ratios(measures.mean((0,1,2)))
    nseeds, npairs, n, nb, nm = measures.shape
    boot = np.zeros((len(seed_weights), nb, nm))
    for p, ew in enumerate(episode_weights):
        flat = measures[:, p].transpose(1, 0, 2, 3).reshape(n, -1)
        selected = (ew @ flat).reshape(-1, nseeds, nb, nm)
        sw = seed_weights if nseeds > 1 else np.ones((len(seed_weights), 1))
        boot += np.einsum("ds,dsbm->dbm", sw, selected) / npairs
    return mean, ratios(boot), pair_values


def main():
    meta = json.loads((ROOT/"weights.json").read_text())
    ck = meta["checkpoints"]
    # Read the stored classifier directly with NumPy, without loading the learning code.
    z = np.array(ck["queries"]) @ ck["written"]
    np.testing.assert_allclose(z, ck["logits"], atol=2e-5, rtol=0)
    np.testing.assert_array_equal(z >= 0, np.array(ck["logits"]) >= 0)
    rng = np.random.default_rng(4971)
    reps = 2048
    sw = rng.multinomial(8, np.full(8, 1/8), size=reps)/8
    ew = [rng.multinomial(192, np.full(192, 1/192), size=reps)/192 for _ in meta["pairs"]]
    cells = {}
    with np.load(ROOT/"trials.npz") as f:
        for method in f:
            a = f[method]
            cells[method] = summaries(a if a.ndim == 5 else a[None], sw, ew)
    effects = {}
    for method, ref in (("change","exception"), ("change_gate","exception"), ("change_pixels","exception"),
                        ("change","change_gate"), ("exception","prior20"), ("change","prior80"),
                        ("change",meta["selected_baseline"]), ("exception",meta["selected_baseline"])):
        delta, draws, pair = [v-w for v,w in zip(cells[method], cells[ref])]
        effects[f"{method}/{ref}"] = dict(mean=np.column_stack([delta.mean(0),
            np.quantile(draws.mean(1), [.025, .975], axis=0).T]).tolist(),
            curves=np.stack([delta, *np.quantile(draws, [.025,.975], axis=0)], axis=-1).tolist(),
            by_pair=pair.mean(1).tolist())
    out = dict(metrics=NAMES, budgets=meta["budgets"], bootstrap=reps, cold_reload_max_error=float(np.max(abs(z-ck["logits"]))),
               methods={k:v[0].tolist() for k,v in cells.items()}, contrasts=effects)
    (ROOT/"summary.json").write_text(json.dumps(out, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
