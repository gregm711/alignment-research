"""Grade only after the two receivers have predicted using identical observations."""
import json

import numpy as np

from run import CHECK_ODDS, HERE, LOG_ODDS, decide, route, vote, worlds


def evaluate(seed, n):
    audit = {}
    fixture, truth = worlds(seed, n, False, audit=audit)
    observed = {k: fixture[k] for k in ("readings", "check", "arg", "perms")}
    origin = vote(observed, "rolling", 1)
    naive = vote(observed, "vote", 0)
    target = route(observed, (truth > 0).astype(int))
    rng = np.random.default_rng(1670)
    worlds_sampled = rng.integers(n, size=(3000, n))
    results = {}
    for label, threshold in [("matched_checks", 2), ("no_checks", 0)]:
        final, checked, order = decide(observed, "rolling", origin, threshold)
        blind_order = (naive * LOG_ODDS + checked * observed["check"] * CHECK_ODDS > 0).astype(int)
        blind_final = route(observed, blind_order)
        subsets = {}
        for name, mask in [("wrong", audit["wrong_ancestor"]), ("right", ~audit["wrong_ancestor"]),
                           ("all", np.ones(truth.shape, dtype=bool))]:
            scores = {}
            for key, ours, other, gold in [("bit", order, blind_order, (truth > 0).astype(int)),
                                           ("task", final, blind_final, target)]:
                denominator = mask.sum(axis=1)
                ours_correct = ((ours == gold) & mask).sum(axis=1)
                other_correct = ((other == gold) & mask).sum(axis=1)
                diff = (ours_correct-other_correct)[worlds_sampled].sum(axis=1)
                diff = diff / denominator[worlds_sampled].sum(axis=1)
                scores[key] = dict(aware=ours_correct.sum()/mask.sum(), blind=other_correct.sum()/mask.sum(),
                                   delta_ci95=np.quantile(diff, [.025, .975]).tolist())
            subsets[name] = dict(events=int(mask.sum()), checks=int(checked[mask].sum()), **scores)
        assert subsets["right"]["events"] + subsets["wrong"]["events"] == subsets["all"]["events"]
        results[label] = subsets
    return results


if __name__ == "__main__":
    results = {str(seed): evaluate(seed, n) for seed, n in [(19274, 128), (764582, 256)]}
    old = json.loads((HERE / "results.json").read_text())["panels"]["fixed"]["policies"]["rolling"]
    assert results["19274"]["matched_checks"]["all"]["task"]["aware"] == old["accuracy"]
    (HERE / "ancestor-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results["764582"]["matched_checks"])
