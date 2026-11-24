"""Develop routers, then evaluate previously frozen selections on fresh paired maps."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from transit import ACTIONS, FEATURES, control, costs, maps

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
sys.path.insert(0, str(HERE.parent))
from orchestrator import ask, save


def read(path):
    return json.loads(path.read_text())


def tree(x, loss, depth):
    totals = loss.sum(axis=0)
    best = ACTIONS[int(totals.argmin())]
    best_loss = totals.min()
    split = None
    if depth:
        for j, col in enumerate(x.T):
            for cut in np.unique(np.quantile(col, [.25,.5,.75])):
                left = col <= cut
                if min(left.sum(), (~left).sum()) < 32:
                    continue
                value = loss[left].sum(0).min() + loss[~left].sum(0).min()
                if value < best_loss - 1e-9:
                    best_loss = value
                    split = j, float(cut), left
    if split is None:
        return best
    j, cut, left = split
    return [FEATURES[j], cut, tree(x[left],loss[left],depth-1), tree(x[~left],loss[~left],depth-1)]


def predict(t, x, depth=0):
    if isinstance(t,str) and t in ACTIONS[:3]:
        return np.full(len(x),ACTIONS.index(t))
    if not isinstance(t,list) or len(t)!=4 or depth>=3 or t[0] not in FEATURES:
        raise ValueError('Malformed router tree')
    if type(t[1]) not in (int,float) or not np.isfinite(t[1]):
        raise ValueError('Threshold must be a finite number')
    return np.where(x[:,FEATURES.index(t[0])]<=t[1],
                    predict(t[2],x,depth+1),predict(t[3],x,depth+1))


def action_values(values, actions):
    return values[np.arange(len(actions)),actions]


def mean_score(t, d):
    return float(action_values(d['score'],predict(t,d['x'])).mean())


def breakdown(t, d):
    acts = predict(t,d['x'])
    regret = action_values(d['score'],acts) - d['score'][:,:3].min(axis=1)
    worst = np.argsort(-regret)[:8]
    return dict(tree=t, mean_cost=mean_score(t,d), actions=dict(Counter(ACTIONS[i] for i in acts)),
                high_regret_examples=[dict(features=d['x'][i].round(3).tolist(),
                                           costs=d['score'][i,:3].round(3).tolist()) for i in worst])


def develop():
    if (OUT/'frozen.json').exists():
        raise ValueError('The development selections are already frozen')
    OUT.mkdir(exist_ok=True)
    control()
    train = maps(110000,1536)
    beta = np.linalg.lstsq(train[0].reshape(-1,4),train[1].ravel(),rcond=None)[0]
    tr = costs(train,beta)
    dev = costs(maps(210000,384),beta)
    fitted = [tree(tr['x'],tr['score'][:,:3],depth) for depth in (1,2,3)]
    t = min(fitted,key=lambda item: mean_score(item,dev))
    simple = [['probe_value',.5,fallback,'ask'] for fallback in ('stop','plan')]
    rule = min(simple,key=lambda item:mean_score(item,dev))
    fixed = min(ACTIONS[:3],key=lambda item:mean_score(item,dev))
    candidates = {'fitted':t}
    feedback = [breakdown(v,dev) for v in [fixed,rule,t]]
    for n in (1,2):
        path = OUT/f'proposal-{n}.json'
        if not path.exists():
            prompt = ((HERE/'README.md').read_text().split('## Outcomes')[0]
                      + '\nReturn a depth <=3 decision tree as JSON serialized inside the tree string. '
                      'Leaves are the strings stop, plan, ask. Nodes are [feature,threshold,left,right], '
                      'left means <=. Gap is cheap time alerted minus cheap time other; correction_gap '
                      'is greedy minus cheap on alerted line minus the corresponding other correction. '
                      'These are the only deployable features, in this order: '+str(FEATURES)
                      +'\nDevelopment summaries, with action costs ordered stop/plan/ask:\n'
                      +json.dumps(feedback)+'\nUse no tools. Propose one plausible improvement. '
                      'Specify a measurable prediction and a failure criterion.')
            save(path, ask(prompt, ('hypothesis','prediction','failure','tree')))
        result = read(path)
        try:
            candidate = json.loads(result['tree'])
            evidence = breakdown(candidate,dev)
            candidates[f'codex{n}'] = candidate
        except (ValueError,TypeError) as error:
            evidence = {'error':str(error)}
        feedback.append(evidence)
    best = min(candidates,key=lambda key:mean_score(candidates[key],dev))
    save(OUT/'development.json',dict(tree_grid=fitted, reports=feedback))
    save(OUT/'frozen.json',dict(beta=beta.tolist(),fixed=fixed,cheap=rule,trees=candidates,selected=best))
    print('Frozen:',best,fixed,rule,flush=True)


def interval(delta):
    rng = np.random.default_rng(52)
    draws = rng.choice(delta, size=(2000,len(delta)),replace=True).mean(axis=1)
    return [float(delta.mean()),*np.quantile(draws,[.025,.975]).tolist()]


def evaluate(tag, start, frozen, raw):
    data = maps(start,1024,shifted=tag=='shift')
    d = costs(data,np.array(frozen['beta']),raw)
    x = d['x']
    decisions = {act:np.full(len(x),i) for i,act in enumerate(ACTIONS)}
    for name,t in dict(fixed=frozen['fixed'],cheap_rule=frozen['cheap'],**frozen['trees']).items():
        decisions[name]=predict(t,x)
    selected = decisions[frozen['selected']]
    decisions.update(selected=selected, shuffled=np.random.default_rng(51).permutation(selected),
                     oracle=d['score'][:,:3].argmin(axis=1))
    summary, scores = {}, {}
    for name, acts in decisions.items():
        score = action_values(d['score'],acts)
        scores[name]=score
        travel=action_values(d['excess'],acts)
        probes=np.isin(acts,[2,3])
        plan=np.isin(acts,[1,3])
        summary[name]=dict(cost=float(score.mean()), travel_regret=float(travel.mean()),
                           optimal_line=float(action_values(d['success'],acts).mean()),
                           probes=float(probes.mean()), plans=float(plan.mean()),
                           base_edge_reads=float((2*d['edges']).mean()),
                           extra_edge_evaluations=float((6*d['edges']*plan).mean()),
                           repriced={str(v):float((travel+.5*plan+v*probes).mean()) for v in (.25,2.)})
    summary['paired_savings_vs_selected']={name:interval(scores[name]-scores['selected'])
                                          for name in ['fixed','cheap_rule','shuffled','oracle']}
    summary['policy_decisions']=dict(Counter(ACTIONS[a] for a in selected))
    summary['oracle_decisions']=dict(Counter(ACTIONS[a] for a in decisions['oracle']))
    if not raw:
        with gzip.open(OUT/f'{tag}-maps.jsonl.gz','wt') as file:
            for i in range(len(x)):
                row=dict(seed=start+i,features=x[i].tolist(), exact=data[1][i].tolist(),
                         cheap=d['cheap'][i].tolist(),alert=int(data[2][i]),
                         routes_by_action_and_delay=d['choices'][i].tolist(),
                         expected_costs=d['score'][i].tolist(),
                         decisions={k:ACTIONS[v[i]] for k,v in decisions.items()})
                file.write(json.dumps(row,separators=(',',':'))+'\n')
    return summary


def test():
    control()
    frozen=read(OUT/'frozen.json')
    data={tag:{mode:evaluate(tag,seed,frozen,mode=='raw') for mode in ['corrected','raw']}
          for tag,seed in [('standard',310000),('shift',410000)]}
    data['frozen_sha256']=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest()
    save(OUT/'summary.json',data)
    for tag in ('standard','shift'):
        print(tag,data[tag]['corrected']['paired_savings_vs_selected'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['develop','test'])
    args=p.parse_args()
    if args.phase=='develop':
        develop()
    else:
        test()
