"""Replay published original-order judge ballots without calling any models."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

REV = "900616bff90b6c6c8e1681f7d079250637c55992"
URL = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/" + REV + "/Dataset/LLMBar/"
CACHE = Path.home() / ".cache/alignment-research/llmbar-ballots" / REV
MODELS = ("ChatGPT", "ChatGPT-0301", "Falcon", "GPT-4", "LLaMA2", "PaLM2")
METHODS = ("Vanilla", "Vanilla_NoRules", "CoT")
SUBSETS = ("Natural", *("Adversarial/" + s for s in ("GPTInst", "GPTOut", "Manual", "Neighbor")))
ARMS = ("majority", "competence", "best", "frozen", "exact", "gate", *(str(2**i) for i in range(7)))


def fetch(path):
    dest = CACHE / path
    if not dest.exists():
        body = urlopen(URL + path, timeout=30).read()
        json.loads(body)
        dest.parent.mkdir(exist_ok=True, parents=True)
        temp = dest.with_suffix(".tmp")
        temp.write_bytes(body)
        temp.replace(dest)
    return json.loads(dest.read_bytes())


def load(subset):
    items = fetch(subset + "/dataset.json")
    def channel(pair):
        model, prompt = pair
        out = fetch(f"{subset}/evaluators/{model}/{prompt}/result.json")
        assert len(out) == len(items)
        votes = []
        for x, row in zip(items, out):
            assert all(x[k] == row[k] for k in ("input", "output_1", "output_2", "label"))
            winner = row["results"][0]["swap = False"]["winner"]
            votes.append({"1": 1, "2": -1}.get(str(winner), 0))
        return votes
    with ThreadPoolExecutor(8) as pool:
        raw = list(pool.map(channel, [(m,p) for m in MODELS for p in METHODS]))
    keys = [sha256(x["input"].strip().encode()).hexdigest() for x in items]
    pairs = [tuple(sorted((x["output_1"].strip(), x["output_2"].strip()))) for x in items]
    order = sorted(range(len(items)), key=lambda i: sha256(("llmbar-v1"+keys[i]).encode()).digest())
    y = np.array([1 if str(x["label"]) == "1" else -1 for x in items])
    assert all(str(x["label"]) in ("1", "2") for x in items)
    return np.array(raw, dtype=np.int8).T[order], y[order], [keys[i] for i in order], [pairs[i] for i in order]


def predict(arm, training, train_y, x):
    train = training[-8:]
    majority = np.where(x.sum(axis=1) >= 0, 1, -1)
    if arm == "majority":
        return majority, 0
    if arm in ("best", "competence"):
        match = training == train_y[:, None]
        # An abstaining individual falls back to the training panel majority.
        if arm == "best":
            fallback = np.where(training.sum(axis=1)>=0, 1, -1) == train_y
            idx = np.argmax(np.where(training==0, fallback[:, None], match).sum(axis=0))
            return np.where(x[:, idx]==0, majority, x[:, idx]), 0
        prob = (match.sum(axis=0)+1)/((training != 0).sum(axis=0)+2)
        weights = np.maximum(np.log(prob/(1-prob)), 0)
        return np.where(x @ weights >= 0, 1, -1), 0
    histories = np.concatenate((train, x))
    refresh = True
    flags = age = count = 0
    preds = []
    for t, row in enumerate(x):
        if refresh:
            h = histories[t:t+8].T
            same = (h[:, None, :] == h[None, :, :]) & (h[:, None, :] != 0)
            near = same.sum(axis=2) >= (8 if arm == "exact" else 7)
            np.fill_diagonal(near, True)
            degree = near.sum(axis=1)
            suspects = degree >= 8
            count += 1
            age = flags = 0
        preds.append(1 if (row/degree).sum() >= 0 else -1)
        age += 1
        if arm == "gate":
            suspect = row[suspects]
            flagged = not len(suspect) or abs(suspect.sum()) < .75*len(suspect)
            flags = ((flags << 1) | int(flagged)) & 15
            refresh = age>=6 and flags.bit_count()>=2
        elif arm == "frozen":
            refresh = False
        else:
            refresh = (t+1) % (1 if arm == "exact" else int(arm)) == 0
    return np.array(preds), count


def main():
    tr, ty, tkeys, tpairs = load("Natural")
    test = []
    train_keys, train_pairs = set(tkeys), set(tpairs)
    for sub in SUBSETS[1:]:
        x, y, keys, pairs = load(sub)
        keep = np.array([k not in train_keys and p not in train_pairs for k,p in zip(keys,pairs)])
        test.append((x[keep], y[keep], list(np.array(keys)[keep])))
    keys = [k for _,_,ks in test for k in ks]
    ids, units = np.unique(keys, return_inverse=True)
    draw = np.random.default_rng(83504).integers(len(ids),size=(4000,len(ids)))
    mass = np.bincount(units)[draw].sum(axis=1)

    saved = {}
    def run(origin, arm):
        cols = list(range(tr.shape[1])) + ([] if origin is None else [origin*len(METHODS)]*20)
        mat = tr[:,cols]
        result = [predict(arm, mat, ty, x[:,cols]) for x,_,_ in test]
        masks = [(out == y) for (out,_),(_,y,_) in zip(result,test)]
        return np.concatenate(masks), [int(m.sum()) for m in masks], sum(n for _,n in result)
    for condition, origins in [("native", [None]), ("copied", list(range(len(MODELS))))]:
        scores = {}
        baseline = np.mean([run(origin, "majority")[0] for origin in origins], axis=0)
        for arm in ARMS:
            comparisons = [run(origin, arm) for origin in origins]
            correct = np.stack([v[0] for v in comparisons]).mean(axis=0)
            gain = np.bincount(units, weights=correct-baseline)[draw].sum(axis=1)/mass
            scores[arm] = dict(accuracy=correct.mean(),delta_ci95=np.quantile(gain,[.025,.975]).tolist(),
                               subset_correct=np.mean([v[1] for v in comparisons],axis=0).tolist(),
                               per_origin_accuracy=[float(v[0].mean()) for v in comparisons],
                               builds=float(np.mean([v[2] for v in comparisons])))
        saved[condition]=scores
    out=dict(revision=REV, test_n=list(map(lambda v:len(v[1]),test)), unique_instructions=len(ids),
             train_n=len(ty), abstentions=sum(int((x==0).sum()) for x,_,_ in test), scores=saved)
    fallback = np.where(tr.sum(axis=1)>=0, 1, -1) == ty
    best_index = int(np.argmax(np.where(tr==0, fallback[:, None], tr==ty[:,None]).sum(axis=0)))
    out["best_calibration_judge"] = [MODELS[best_index // len(METHODS)], METHODS[best_index % len(METHODS)]]
    # The archival item order is arbitrary. Reorder within each named subset.
    rng = np.random.default_rng(11397)
    order_scores = []
    for _ in range(16):
        hit = {"1": 0, "8": 0, "gate": 0}
        for x, y, _ in test:
            ix = rng.permutation(len(y))
            for arm in hit:
                pred, _ = predict(arm, tr, ty, x[ix])
                hit[arm] += int((pred == y[ix]).sum())
        order_scores.append(hit)
    out["alternative_order_correct"] = order_scores
    # Charge the original code path including re-reading in-memory calibration ballots.
    out["receiver_cost"] = {}
    for arm in ("majority", "best", "1", "8", "gate"):
        def once():
            for x, _, _ in test:
                predict(arm, tr, ty, x)
        _, warm = measure(once)
        repeats = max(1, round(.18 / warm["wall_seconds"]))
        def batch():
            for _ in range(repeats):
                once()
        out["receiver_cost"][arm] = dict(events=repeats*len(keys), trials=[measure(batch)[1] for _ in range(3)])
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == "__main__":
    main()
