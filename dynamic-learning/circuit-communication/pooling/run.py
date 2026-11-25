"""Sequential four-cell packets on a planted common-source / independent sensor panel."""
import gzip
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

TRUE = np.array([.5, .57, .90])
CHECK = math.log(.82 / .18)
SEEDS = 20
BITS_PRICE = .00015
CHECK_PRICE = .12
BLOCK = 4
REGIMES = ["local", "weak", "copies", "wrong_copy"]
METRICS = ["accuracy", "nll", "brier", "bits", "active", "checks", "messages", "objective"]


def generate(n, per, seed):
    rng = np.random.default_rng(seed)
    count = 3 * per
    kind = np.repeat(np.arange(3), per)
    truth = rng.choice([-1, 1], size=count)
    tag = np.ones((count, n), dtype=np.int8)
    tag[:per] = 0
    tag[np.arange(per), rng.integers(n, size=per)] = 2
    source = rng.random((count, n)).argsort(axis=1)
    reading = np.where(rng.random((count, n)) < TRUE[tag], 1, -1) * truth[:, None]
    first = np.ones_like(tag, dtype=bool)
    wrong = np.zeros(count, dtype=bool)
    for row in range(2 * per, count):
        copies = np.sort(rng.choice(n, 20, replace=False))
        original = copies[0]
        source[row, copies] = source[row, original]
        reading[row, copies] = reading[row, original]
        first[row, copies[1:]] = False
        wrong[row] = reading[row, original] != truth[row]
    check = np.where(rng.random(count) < .82, truth, -truth)
    return dict(source=source, tag=tag, reading=reading, check=check,
                truth=truth, kind=kind, wrong=wrong, first=first)


def weights(train):
    same = np.ones(3)
    total = np.full(3, 2.)
    for data in train:
        for t in range(3):
            mask = (data["tag"] == t) & data["first"]
            total[t] += mask.sum()
            same[t] += ((data["reading"] == data["truth"][:, None]) & mask).sum()
    return np.log(same / (total - same))


def per_episode(data, logit, bits, active, checked, messages):
    y = data["truth"]
    loss = np.logaddexp(0, -y * logit)
    accuracy = ((logit > 0) == (y > 0)).astype(float)
    accuracy[logit == 0] = .5
    error_probability = np.exp(-np.logaddexp(0, y * logit))
    objective = loss + BITS_PRICE * bits + CHECK_PRICE * checked
    return np.stack([accuracy, loss, error_probability ** 2, bits, active, checked, messages, objective], axis=-1)


def infer(obs, w, repeat, policy, replay=False):
    """Inputs available at purchase only. Live receivers deduplicate observed messages."""
    minimum, fraction, stop, check_limit = policy
    tags, signals, origins = obs["tag"], obs["reading"], obs["source"]
    batch, n = tags.shape
    limit = BLOCK * math.ceil(n * fraction / BLOCK)
    seen = np.zeros((batch, n), dtype=bool)
    posterior = np.zeros(batch)
    blind = np.zeros(batch)
    active = np.zeros(batch, dtype=int)
    bits = np.zeros(batch, dtype=int)
    messages = np.zeros(batch, dtype=int)
    rows = np.arange(batch)
    for col in range(0, limit, BLOCK):
        if not len(rows):
            break
        active[rows] += BLOCK
        bits[rows] += 2 * BLOCK
        for j in range(col, col + BLOCK):
            senders = rows[tags[rows, j] >= minimum]
            origin = origins[senders, j]
            sign_weight = signals[senders, j] * w[tags[senders, j]]
            posterior[senders] += sign_weight * np.where(seen[senders, origin], repeat, 1.)
            if replay:
                blind[senders] += sign_weight
            seen[senders, origin] = True
            messages[senders] += 1
            bits[senders] += 3 + (n - 1).bit_length()
        rows = rows[np.abs(posterior[rows]) < stop]
    checked = np.zeros(batch, dtype=bool)
    checked[rows[np.abs(posterior[rows]) < check_limit]] = True
    posterior[checked] += CHECK * obs["check"][checked]
    bits[checked] += 2
    if replay:
        blind[checked] += CHECK * obs["check"][checked]
    return (posterior, bits, active, checked, messages), blind


def observable(data):
    return {k: data[k] for k in ("tag", "reading", "source", "check")}


def fit_policy(train, w, repeat):
    # The receipt-count branch is identical for every check threshold; the small grid is cheap.
    best = (float("inf"), None)
    for gate, frac, stop, check in itertools.product((0, 1, 2), (.25, .5, 1.), (1.3, 2.4, 1000.), (0., .8, 1.8)):
        policy = (gate, frac, stop, check)
        score = 0.
        for data in train:
            result, _ = infer(observable(data), w, repeat, policy)
            score += per_episode(data, *result)[:, -1].mean()
        if score < best[0]:
            best = (float(score), policy)
    return best[1]


def summarize(values):
    return np.mean(values, axis=0).tolist()


def evaluate(models):
    raw = {}
    for n in (24, 48, 96):
        rows = []
        for s in range(SEEDS):
            data = generate(n, 128, 8101 + s + 100 * n)
            obs = observable(data)
            masks = [data["kind"] == k for k in range(3)] + [data["wrong"]]
            arms = {}
            for arm, (w, repeat, policy) in models.items():
                results, blind = infer(obs, w, repeat, policy, replay=arm == "learned")
                perf = per_episode(data, *results)
                arms[arm] = [summarize(perf[m]) for m in masks]
                if arm == "learned":
                    replay = per_episode(data, blind, *results[1:])
                    assert np.array_equal(replay[data["kind"] != 2], perf[data["kind"] != 2])
                    arms["replay_blind"] = [summarize(replay[m]) for m in masks]
            rows.append(dict(seed=8101 + s + 100 * n, counts=[int(m.sum()) for m in masks], arms=arms))
        raw[str(n)] = rows
    return raw


def ci(values, draws):
    x = np.array(values)
    return [float(x.mean()), *np.quantile(x[draws].mean(axis=1), [.025, .975]).tolist()]


def report(raw):
    draws = np.random.default_rng(883).integers(SEEDS, size=(4000, SEEDS))
    out = {}
    for n, rows in raw.items():
        arms = {arm: np.array([r["arms"][arm] for r in rows]) for arm in rows[0]["arms"]}
        learned = arms["learned"]
        contrasts = {
            "blind_minus_learned_copy_nll": arms["replay_blind"][:, 2, 1] - learned[:, 2, 1],
            "blind_minus_learned_wrong_copy_nll": arms["replay_blind"][:, 3, 1] - learned[:, 3, 1],
            "blind_minus_learned_weak_nll": arms["replay_blind"][:, 1, 1] - learned[:, 1, 1],
            "learned_minus_exact_accuracy": (learned - arms["exact_broadcast"])[:, :3, 0].mean(axis=1),
            "learned_bits_saved_fraction": 1 - learned[:, :3, 3].mean(axis=1) / arms["exact_broadcast"][:, :3, 3].mean(axis=1),
            "analytic_minus_learned_objective": (arms["analytic_policy"] - learned)[:, :3, -1].mean(axis=1),
        }
        out[n] = dict(means={k: summarize(v) for k, v in arms.items()}, contrasts={k: ci(v, draws) for k, v in contrasts.items()})
    return out


def timings(models):
    obs = observable(generate(48, 512, 99001))
    result = {}
    for arm, (w, repeat, policy) in models.items():
        infer(obs, w, repeat, policy)
        loops = 36
        def work():
            for _ in range(loops):
                infer(obs, w, repeat, policy)
        _, measured = measure(work)
        result[arm] = dict(**measured, episodes=loops * len(obs["tag"]), repetitions=loops)
    return result


def main():
    train = [generate(n, 256, 101 + i) for i, n in enumerate((24, 48))]
    full = (0, 1., 1000., 0.)
    learned = weights(train)
    exact = np.log(TRUE / (1 - TRUE))
    def training_loss(r):
        return sum(np.logaddexp(0, -d["truth"] * infer(observable(d), learned, r, full)[0][0]).mean() for d in train)
    repeat = min((0., .25, 1.), key=training_loss)
    models = {
        "learned": (learned, repeat, fit_policy(train, learned, repeat)),
        "analytic_policy": (exact, 0., fit_policy(train, exact, 0.)),
        "exact_broadcast": (exact, 0., full),
        "naive_broadcast": (exact, 1., full),
        "strong_only": (exact, 0., (2, 1., 1000., 0.)),
    }
    truth = generate(24, 4, 1)
    assert all(len(np.unique(row)) == 5 for row in truth["source"][-4:])
    assert np.all(truth["first"][-4:].sum(axis=1) == 5)
    exact_pred = infer(observable(truth), exact, 0., full)[0][0]
    assert np.allclose(exact_pred, (truth["reading"] * exact[truth["tag"]] * truth["first"]).sum(axis=1))
    raw = evaluate(models)
    with gzip.GzipFile(str(HERE / "seed-results.json.gz"), "wb", mtime=0) as f:
        f.write(json.dumps(raw, separators=(",", ":")).encode())
    out = dict(metrics=METRICS, regimes=REGIMES, models={a: [w.tolist(), r, p] for a, (w, r, p) in models.items()},
               summary=report(raw), timing=timings(models))
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
