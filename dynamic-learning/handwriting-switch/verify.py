"""Test provisional learning and selective verification on previously unused writers."""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
DIGEST = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
POLICIES = ("random", "confident", "disagree")
GATES = ("both", "one", "last_only")
NOISE = ("independent", "perfect", "shared")
METRICS = ("change_accuracy", "stable_harm", "true_repair", "change_release", "stable_release")


def corpus():
    path = Path.home()/".cache/alignment-research/optdigits/original.zip"
    if not path.exists():
        with urlopen(URL, timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DIGEST:
            raise ValueError("UCI archive changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DIGEST
    with ZipFile(path) as z:
        new = np.loadtxt(z.open("optdigits.tra"), delimiter=",")
        old = np.loadtxt(z.open("optdigits.tes"), delimiter=",")
    bundled = Path(importlib.util.find_spec("sklearn").origin).parent/"datasets/data/digits.csv.gz"
    np.testing.assert_array_equal(old, np.loadtxt(bundled, delimiter=","))
    x = new[:, :-1] - old[old[:, -1] < 6, :-1].mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, new[:, -1]


def episodes(x, y, pair, seed):
    rng = np.random.default_rng(seed)
    n = 256
    bits = rng.integers(2, size=(n, 17))
    marks = rng.integers(2, size=bits.shape)
    marks[:, 0] = 1-bits[:, 0]
    ids = np.empty(bits.shape, int)
    query = np.empty((n, 32), int)
    qmark = rng.integers(2, size=query.shape)
    for i in range(n):
        order = pair[::-1] if rng.integers(2) else pair
        for b, digit in enumerate(order):
            take = bits[i] == b
            sample = rng.choice(np.flatnonzero(y == digit), int(take.sum())+16, replace=False)
            ids[i, take] = sample[16:]
            query[i, 16*b:16*b+16] = sample[:16]
        assert len(set(ids[i]) | set(query[i])) == 49
    return x[ids], bits, marks, x[query], np.tile(np.repeat([0, 1], 16), (n, 1)), qmark


def write(v, x, y, step):
    p = 1/(1+np.exp(-np.clip((v*x).sum(1), -60, 60)))
    return v + step*(y-p)[:, None]*x


def acquire(v, x, mark, policy):
    scores = np.einsum("nd,nkd->nk", v, x)
    if policy == "disagree":
        scores = scores*(1-2*mark)
    elif policy == "confident":
        scores = np.abs(scores)
    elif policy == "random":
        return np.zeros(len(x), int)
    else:
        raise ValueError(policy)
    return scores.argmax(1)


def paths(data, step, policy):
    x, bit, mark, q, label, qmark = data
    n = len(x)
    rows = np.arange(n)
    wrong = label != qmark
    initial = write(np.zeros_like(x[:, 0]), x[:, 0], bit[:, 0], step)
    # noise x gate x episode x counts: change successes, stable mistakes, repairs,
    # eligible errors, change releases, stable releases
    counts = np.zeros((3, 3, n, 6))
    for flips in itertools.product((0,1), repeat=2):
        k = sum(flips)
        mass = np.array([.1**k*.9**(2-k), float(k == 0), (.9, 0, .1)[k]])
        for changed in (True, False):
            v = initial
            disagreements = np.zeros(n, int)
            for t, flip in enumerate(flips):
                menu = slice(1+8*t, 9+8*t)
                j = 1+8*t+acquire(v, x[:, menu], mark[:, menu], policy)
                old = mark[rows, j]
                seen = (bit[rows, j] if changed else old) ^ flip
                item = x[rows, j]
                disagreements += seen != old
                v = write(v, item, seen, step)
            warmed = np.einsum("nd,nqd->nq", v, q) >= 0
            cold = np.einsum("nd,nqd->nq", write(np.zeros_like(v), item, seen, step), q) >= 0
            for g, gate in enumerate(GATES):
                release = disagreements >= (1 if gate == "one" else 2)
                pred = np.where(release[:, None], cold if gate=="last_only" else warmed, qmark)
                if changed:
                    success = pred == label
                    c = np.stack([success.sum(1), (wrong&success).sum(1), wrong.sum(1), release], 1)
                    for i, col in enumerate((0, 2, 3, 4)):
                        counts[:,g,:,col] += mass[:, None]*c[:, i]
                else:
                    counts[:,g,:,1] += mass[:, None]*(pred != qmark).sum(1)
                    counts[:,g,:,5] += mass[:, None]*release
    np.testing.assert_allclose(counts[:, 0, :, 5], np.broadcast_to([[.01], [0.], [.1]], (3, n)))
    return counts


def rates(a):
    return np.stack([a[..., 0]/32, a[..., 1]/32, a[..., 2]/a[..., 3], a[..., 4], a[..., 5]], axis=-1)


def report(raw):
    rng = np.random.default_rng(43519)
    ix = [rng.multinomial(256, np.full(256, 1/256), size=2048)/256 for _ in range(6)]
    cells, summaries = {}, {}
    for key, a in raw.items():
        # pair x episode x counts; all arms reuse these episode bootstrap weights.
        avg = a.mean((0,1))
        draws = sum(index @ values for index, values in zip(ix, a))/6
        v, ds = rates(avg), rates(draws)
        cells[key] = (v, ds)
        summaries[key] = np.column_stack([v, np.quantile(ds,[.025,.975],axis=0).T]).tolist()
    diffs = {}
    for noise in NOISE:
        key = f"scalar/disagree/{noise}/both"
        for other in (f"scalar/random/{noise}/both", f"scalar/confident/{noise}/both",
                      f"scalar/disagree/{noise}/one", f"scalar/disagree/{noise}/last_only",
                      f"learned/disagree/{noise}/both"):
            a, boot = [x-y for x,y in zip(cells[key],cells[other])]
            diffs[f"{key} minus {other}"] = np.column_stack([a, np.quantile(boot,[.025,.975],axis=0).T]).tolist()
    return dict(metrics=METRICS, cells=summaries, contrasts=diffs)


def main():
    meta = json.loads((ROOT/"weights.json").read_text())
    scalar = meta["baseline_steps"][meta["selected_baseline"]]
    steps = [np.exp(meta["checkpoints"][f"{s}/change/log_pixel_step"]) for s in meta["seeds"]]
    x, y = corpus()
    raw = {}
    for p, pair in enumerate(meta["pairs"]):
        d = episodes(x, y, pair, 43170+p)
        for policy in POLICIES:
            for method, gains in (("scalar", [scalar]), ("learned", steps)):
                counts = sum(paths(d, gain, policy) for gain in gains)/len(gains)
                for i, noise in enumerate(NOISE):
                    for j, gate in enumerate(GATES):
                        raw.setdefault(f"{method}/{policy}/{noise}/{gate}", []).append(counts[i,j])
    raw = {k: np.stack(v) for k,v in raw.items()}
    np.savez_compressed(ROOT/"verification-trials.npz", **raw)
    (ROOT/"verification.json").write_text(json.dumps(report(raw), indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
