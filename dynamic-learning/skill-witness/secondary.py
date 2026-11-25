"""Compare native and analytic online program-bank readers with equal paid observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from native_model import DRIFT, K, NOISE, WEIGHTS, load_model
from panel import episodes

HERE = Path(__file__).resolve().parent
ARMS = ('native', 'analytical', 'replay', 'anchor')


def scores(weights, outputs, target):
    # At each input pick the output with the highest combined supporting mass.
    eq = outputs[:, :, None] == outputs[:, None, :]
    predicted = outputs[np.arange(len(outputs)), (eq @ weights).argmax(axis=1)]
    good = predicted == target
    return np.array([good.mean(), float(good.all())])


def read(p, match):
    if not match.any():
        return p.copy()
    lik = (1. - NOISE) * match + NOISE * match.mean()
    a = p * lik
    return a / a.sum()


def choose(p, keys, used):
    mass = (keys[:, :, None] == keys[:, None, :]) @ p
    entropy = -(np.log(mass.clip(1e-12)) * p).sum(1)
    entropy[used] = -np.inf
    return int(entropy.argmax())


def prior_summary(arr):
    return arr.mean(axis=0).tolist()


def run(path):
    raw = json.loads(path.read_text())
    releases = raw['producers']
    model = load_model()
    producers = {}
    anchor_prior = np.eye(K)[0]
    for name, nq, bank, j, _, qids, tids, pred, gold in episodes(path):
        panels = producers.get(name)
        if panels is None:
            first = next(row for row in releases[name] if row is not None)
            native = model.initial()
            analytic = np.full(K, 1 / K)
            for i, key in enumerate(qids):
                good = pred[i] == first[key]
                native = model.update(native, good)
                analytic = read(analytic, good)
            panels = {'bank_bytes':len(json.dumps(bank, ensure_ascii=False, separators=(',', ':')).encode()),
                      'states':[native, analytic, analytic.copy()],
                      'prev':np.array([first[i] for i in tids], dtype=object),
                      'scores':[], 'oracle':[], 'changed':[], 'miss':[0, 0, 0], 'different':0}
            producers[name] = panels
        native, analytic, replay = panels['states']
        native = model.begin(native)
        analytic = (1 - DRIFT) * analytic + DRIFT / K
        replay = (1 - DRIFT) * replay + DRIFT / K
        used_n, used_a = [], []
        query_pred, audit = pred[:nq], pred[nq:]
        truth = gold[nq:]
        result = np.empty((len(ARMS), 3, 2))
        for t in range(3):
            for a, p in enumerate((model.prior(native), analytic, replay, anchor_prior)):
                result[a, t] = scores(p, audit, truth)
            if t == 2:
                break
            qn = model.choose(native, query_pred, used_n)
            qa = choose(analytic, query_pred, used_a)
            panels['different'] += int(qn != qa)
            used_n.append(qn)
            used_a.append(qa)
            mn, ma = query_pred[qn] == gold[qn], query_pred[qa] == gold[qa]
            panels['miss'][0] += int(not mn.any())
            panels['miss'][1] += int(not ma.any())
            panels['miss'][2] += int(not mn.any())
            native = model.update(native, mn)
            analytic = read(analytic, ma)
            replay = read(replay, mn)
        correct = audit == truth[:, None]
        panels['oracle'].append([correct.mean(0).max(), correct.all(0).any(), correct.any(1).mean()])
        panels['scores'].append(result)
        panels['states'] = [native, analytic, replay]
        panels['changed'].append(bool((truth != panels['prev']).any()))
        panels['prev'] = truth.copy()
    if not producers:
        raise ValueError('No later releases')
    avgs, limits, changed, stable, counts, details = [], [], [], [], [], {}
    for name, panels in producers.items():
        outcomes = np.array(panels['scores'])
        flags = np.array(panels['changed'])
        avgs.append(outcomes.mean(0))
        limits.append(np.mean(panels['oracle'], axis=0))
        for mask, dest in ((flags, changed), (~flags, stable)):
            if mask.any():
                dest.append(outcomes[mask].mean(0))
        counts.append(len(flags))
        details[name] = dict(revisions=len(flags), changed=int(flags.sum()),
                             average=avgs[-1].tolist(), oracle=limits[-1].tolist())
    avgs = np.array(avgs)
    rng = np.random.default_rng(7901)
    idx = rng.integers(len(counts), size=(4000, len(counts)))
    ci = {}
    for i in (1, 2):
        gap = avgs[:, 0, 2] - avgs[:, i, 2]
        ci[ARMS[i]] = np.stack([gap.mean(0), *np.quantile(gap[idx].mean(1), [.025, .975], axis=0)]).tolist()
    return dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), producers=len(counts),
                releases=sum(counts), graded_change_releases=sum(p['changed'] for p in details.values()),
                paid_calls_per_active_arm=10 * len(counts) + 2 * sum(counts),
                total_program_bytes=sum(p['bank_bytes'] for p in producers.values()),
                bank_miss_receipts=np.sum([p['miss'] for p in producers.values()], axis=0).tolist(),
                native_and_analytical_check_ids_different=sum(p['different'] for p in producers.values()),
                scores=prior_summary(avgs), oracle=prior_summary(np.array(limits)),
                changed_scores=prior_summary(np.array(changed)) if changed else None,
                stable_scores=prior_summary(np.array(stable)) if stable else None,
                producers_with_changes=len(changed), native_minus_at_two=ci, by_producer=details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    summary = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                   arms=ARMS, checks=[0,1,2], metrics=['row_accuracy','whole_revision'],
                   oracle_order=['best_single_row_accuracy','any_program_perfect','pointwise_row_accuracy'],
                   ci_order=['difference','paired_2.5_percentile','paired_97.5_percentile'],
                   panels={p.stem:run(p.expanduser()) for p in args.files})
    (HERE / 'secondary-results.json').write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':'))+'\n')
    for key, panel in summary['panels'].items():
        print(key, {k:v for k,v in panel.items() if k not in ('by_producer',)})
