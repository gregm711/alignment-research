"""Compare checking a contradicted habit with observing its next independent instance."""
import json

from learn import N, ROOT, load, np, panel

METHODS = ("old", "report", "recheck", "next", "recheck_only", "next_only", "beta_recheck", "beta_next")
STRENGTHS = (1, 2, 4, 8, 16, 32, 64)


def episodes(old, new, congress):
    (past, _), (present, _) = panel(old, new, np.random.default_rng(7000 + congress))
    rows = []
    for member in range(N):
        ids = (member + np.arange(4)) % N
        def codes(matrix):
            x = matrix[:, ids]
            return np.where(np.isfinite(x).all(1), np.nan_to_num(x[:, 1:]) @ [1, 2, 4], -1)
        then, now = codes(past), codes(present)
        for pattern in range(8):
            history = past[then == pattern, member]
            if len(history) < 20:
                continue
            majority = history.mean() >= .5
            exceptions = (history != majority).sum()
            if exceptions > .2 * len(history):
                continue
            prior = (exceptions + 1) / (len(history) + 2)
            for start in range(0, len(present), 64):
                ids_now = start + np.flatnonzero(now[start:start+64] == pattern)
                seq = (present[ids_now[:len(ids_now)//6*6], member] != majority).reshape(-1, 6)
                for group in seq:
                    rows.append((congress, start//64, prior, group[0], group[1], group[2:].sum()))
    assert rows
    return np.array(rows).T


def cells(data, method, report):
    _, _, prior, first, second, _ = data
    band = np.digitize(prior, [.05, .15])
    claim = 0 if method.endswith("_only") else report
    check = 0 if method == "report" else (first if method.startswith("recheck") else second).astype(int)
    return 4 * band + 2 * claim + check


def tables(train):
    fitted = {}
    for method in ("report", "recheck", "next", "recheck_only", "next_only"):
        total = np.zeros(12)
        deviations = np.zeros(12)
        for data in train:
            for report in (0, 1):
                weight = np.where(data[3] == report, .9, .1)
                idx = cells(data, method, report)
                total += np.bincount(idx, weights=weight, minlength=12)
                deviations += np.bincount(idx, weights=weight*data[5]/4, minlength=12)
        fitted[method] = total, deviations
    return fitted


def beta(data, method, strength):
    _, _, prior, first, second, _ = data
    if method == "beta_recheck":
        return (strength * prior + first) / (strength + 1)
    expected_first = .9 * prior / (.9 * prior + .1 * (1-prior))
    return (strength * prior + second + expected_first) / (strength + 2)


def brier(prob, fraction):
    return fraction*(1-prob)**2 + (1-fraction)*prob**2


def fit(train):
    strengths = {}
    for method in ("beta_next", "beta_recheck"):
        def loss(s):
            return np.mean([np.average(brier(beta(data, method, s), data[5]/4),
                                       weights=np.where(data[3], .9, .1)) for data in train])
        strengths[method] = min(STRENGTHS, key=loss)
    return tables(train), strengths


def predict(data, fitted, strengths):
    prior = data[2]
    out = {"old": prior}
    for method, (total, deviations) in fitted.items():
        idx = cells(data, method, 1)
        prob = (32*prior + deviations[idx]) / (32+total[idx])
        fast_log_odds = np.log(prob / (1-prob))
        out[method] = 1 / (1+np.exp(-fast_log_odds))
    for method, strength in strengths.items():
        out[method] = beta(data, method, strength)
    # A verified original ballot makes its noisy copy conditionally uninformative.
    # Small differences between these empirical estimates can still arise from shrinkage.
    return out


def summarize(data, predictions, draws, group):
    _, block, _, first, _, count = data
    block = block.astype(int)
    fraction = count / 4
    w = np.where(first, .9, .1)
    if group == "true":
        w = w * first
    elif group == "false":
        w = w * (1-first)
    def bundle(x):
        v = np.bincount(block, weights=x, minlength=draws.shape[1])
        return np.r_[v.sum(), v[draws].sum(1)]
    denom = bundle(w)
    assert (denom > 0).all()
    values = {"persistent_fraction": bundle(w*(count >= 3))/denom,
              "false_fraction": bundle(w*(1-first))/denom}
    for method, prob in predictions.items():
        values[method + "/brier"] = bundle(w*brier(prob, fraction))/denom
        values[method + "/accuracy"] = bundle(w*np.where(prob >= .5, fraction, 1-fraction))/denom
        values[method + "/forecast"] = bundle(w*prob)/denom
    return values


def feedback(data, pred):
    first, count = data[3], data[5]
    baseline, full = pred["next_only"] >= .5, pred["next"] >= .5
    right = np.where(baseline, count, 4-count)
    switched = baseline != full
    return np.array([((4-right)*first*switched).sum(), ((4-right)*first).sum(),
                     (right*(1-first)*switched).sum(), (right*(1-first)).sum()], dtype=int)


def main():
    rng = np.random.default_rng(29071)
    old = load(110)
    archive = []
    for congress in range(111, 119):
        new = load(congress)
        archive.append(episodes(old, new, congress))
        old = new
    fitted, strengths = fit(archive[:6])
    output = {"strength": strengths, "episodes_by_congress": [len(d.T) for d in archive]}
    by_group = {name: [] for name in ("all", "true", "false")}
    event_counts = []
    raw = {}
    for data in archive[6:]:
        congress = int(data[0, 0])
        pred = predict(data, fitted, strengths)
        nblocks = int(data[1].max()) + 1
        indices = rng.integers(nblocks, size=(4096, nblocks))
        for name, results in by_group.items():
            results.append(summarize(data, pred, indices, name))
        event_counts.append(feedback(data, pred))
        raw[str(congress)] = np.vstack([data, *[pred[m] for m in METHODS]])
    output["raw_rows"] = ["congress", "block", "prior", "first_deviation", "second_deviation",
                          "future_deviations_out_of_four", *METHODS]
    output["feedback_by_congress"] = [a.tolist() for a in event_counts]
    output["feedback_columns"] = ["true_repairs", "eligible_true_errors", "false_harms", "eligible_false_correct"]
    for group, (a, b) in by_group.items():
        merged = {key: (a[key]+b[key])/2 for key in a}
        result = {key: float(v[0]) for key, v in merged.items()}
        for ref in ("recheck", "beta_next", "next_only", "old", "report"):
            key, cmp = "next/brier", ref+"/brier"
            diff = merged[key] - merged[cmp]
            result["next_minus_"+ref] = dict(estimate=float(diff[0]),
                interval=np.quantile(diff[1:], [.025, .975]).tolist(),
                congress=[float(x[key][0] - x[cmp][0]) for x in (a,b)])
        output[group] = result
    assert sum(event_counts)[1] and sum(event_counts)[3]
    np.savez_compressed(ROOT/"persistence-trials.npz", **raw)
    (ROOT/"persistence.json").write_text(json.dumps(output, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
