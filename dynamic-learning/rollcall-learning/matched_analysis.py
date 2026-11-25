"""Estimate the paired interaction between training and test-time verification."""
import json

from active_analysis import BUDGETS, GROUPS, ROOT, aggregate, np
from learn import RollModel, torch
from matched import CHOICES


def resample(counts, indices):
    blocks = counts.reshape(2, 16, 16, 14).sum(2)
    sampled = np.stack([period[i].sum(1) for period, i in zip(blocks, indices)])
    return aggregate(counts.sum(1)[:, None])[0], aggregate(sampled)


def interval(point, draws):
    finite = draws[np.isfinite(draws)]
    assert len(finite) and np.isfinite(point)
    return [float(point), *np.quantile(finite, [.025, .975]).tolist(), len(finite)]


def reload_weights():
    errors = {}
    with np.load(ROOT / "matched-weights.npz") as weights, torch.no_grad():
        for period in (117, 118):
            model = RollModel()
            prefix = f"{period}/model/"
            model.load_state_dict({k[len(prefix):]: torch.tensor(weights[k].tolist())
                                   for k in weights if k.startswith(prefix)})
            read, _ = model.items()
            value = torch.tensor(weights[f"{period}/value"].tolist())[None, None]
            bias = torch.tensor(weights[f"{period}/bias"].tolist())
            prob = model.writer.score(read[None], value, bias)[0, 0].sigmoid()
            pred = np.asarray(prob.tolist())
            reference = weights[f"{period}/prediction"]
            np.testing.assert_allclose(pred, reference, atol=2e-6, rtol=0)
            np.testing.assert_array_equal(pred >= .5, reference >= .5)
            errors[period] = float(np.abs(pred - reference).max())
    return errors


def main():
    torch.set_num_threads(1)
    accuracy, brier, disputed, repair, harm, own_repair, own_harm = range(7)
    names = ("accuracy", "brier", "disputed_brier", "repair_difference_common",
             "harm_difference_common", "repair", "harm")
    with np.load(ROOT / "matched-trials.npz") as data:
        counts, template = data["counts"], data["template"]
    assert counts.shape == (2, 2, 2, 3, 2, 256, 14)
    indices = np.random.default_rng(713).integers(16, size=(2, 4096, 16))
    report = {}
    for j, budget in enumerate(BUDGETS):
        for g, group in enumerate(GROUPS):
            cells, effects, main, summaries = {}, {}, {}, {}
            for p, selection in enumerate(CHOICES):
                for m, training in enumerate(CHOICES):
                    raw = counts[:, j, p, g, m]
                    val, draws = resample(raw, indices)
                    cells[p, m] = (val, draws)
                    summaries[f"train-{training}/test-{selection}"] = {
                        "rates": {names[k]: float(val[k]) for k in (accuracy, brier, disputed, own_repair, own_harm)},
                        "accuracy_by_congress": [float(s[:, 0].sum() / s[:, 3].sum()) for s in raw]}
                v, d = cells[p, 1]
                ref, rd = cells[p, 0]
                effects[selection] = {names[k]: interval(v[k] - ref[k], d[:, k] - rd[:, k])
                                      for k in (accuracy, brier)}
                effects[selection].update({names[k]: interval(v[k], d[:, k]) for k in (repair, harm)})
                tv, _ = resample(template[:, j, p, g], indices)
                summaries[f"template/test-{selection}"] = {names[k]: float(tv[k]) for k in (accuracy, brier, disputed)}
            interaction = {}
            for k in (accuracy, brier):
                v0, d0 = cells[0, 0]
                v1, d1 = cells[0, 1]
                v2, d2 = cells[1, 0]
                v3, d3 = cells[1, 1]
                interaction[names[k]] = interval((v3[k]-v2[k])-(v1[k]-v0[k]),
                                                  (d3[:, k]-d2[:, k])-(d1[:, k]-d0[:, k]))
                main[names[k]] = {
                    "training": interval(((v3[k]-v2[k])+(v1[k]-v0[k]))/2,
                                         ((d3[:, k]-d2[:, k])+(d1[:, k]-d0[:, k]))/2),
                    "selection": interval(((v3[k]-v1[k])+(v2[k]-v0[k]))/2,
                                          ((d3[:, k]-d1[:, k])+(d2[:, k]-d0[:, k]))/2)}
            report[f"{budget}-{group}"] = dict(cells=summaries, targeted_training_minus_random=effects,
                                              interaction=interaction, main_effects=main)
    out = dict(interval_fields=["estimate", "lower", "upper", "valid_draws"],
               reload_max_abs_error=reload_weights(), results=report)
    (ROOT / "matched-summary.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
