"""Develop, freeze and score the paired lifetime experiment locally."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from learner import METHODS, CUTS, COUNT, History, deploy, features, replay, teacher
from propose import DEFAULT
from world import case

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
METRICS = ['correct_of_96','examples','expansions','support_program_calls','active_programs','actions']
CONDITIONS = ['clean','misleading','stale']


def save_json(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def config(name):
    return json.loads((OUT/name).read_text())


def threshold(scores):
    reached = scores>=87
    return np.where(reached.any(axis=-1),reached.argmax(axis=-1),scores.shape[-1])


def train_history(arm, settings, events, vectors, shuffle):
    base = arm.split('_')[0]
    store = History(base,settings)
    episodes = [event for event in events if event[1] is not None]
    if arm.endswith('_reset'):
        episodes = []
    labels = [event[1] for event in episodes]
    if arm.endswith('_shuffle'):
        order = np.random.default_rng(shuffle).permutation(len(labels))
        labels = [labels[k] for k in order]
    h,logits = vectors['practice']
    for (idx,_),label in zip(episodes,labels):
        store.save(h[idx],logits[idx],label)
    return store


def lifetime(seed, domain, settings, winner=None, dev=False):
    start = time.process_time()
    cache = np.load(OUT/'encoder.npz')
    vectors = features(cache)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(4)
    swap = rng.choice(4,2,replace=False)
    stale = perm.copy()
    stale[swap] = stale[swap[::-1]]
    practice = []
    practice_cost = []
    total_probes = total_work = saves = tokens = 0
    for i in range(CUTS[-1]):
        idx,target,mat = case(rng,'practice',int(rng.integers(4)),perm)
        ids,label,work = teacher(target,mat,practice=True)
        practice.append((idx,label))
        total_probes += len(ids)
        total_work += work
        saves += int(label is not None)
        tokens += cache['practice_tokens'][idx]
        practice_cost.append((total_probes,total_work,saves,tokens))
    cases = []
    for g in rng.permutation(np.tile(np.arange(4),2)):
        idx,target,mat = case(rng,domain,g,perm)
        alt = int(4*stale[g]+target%4)
        cases.append((idx,target,alt,mat))
    keys = list(settings)
    if winner is not None:
        keys += [winner+'_reset',winner+'_shuffle']
    cuts = [96] if dev else CUTS
    cns = 1 if dev else len(CONDITIONS)
    axes = (len(cuts),len(keys),8)
    common = np.zeros(axes+(7,),dtype=np.uint8)
    native = np.zeros(axes+(cns,3,6),dtype=np.uint16)
    matched = np.zeros(axes+(cns,2),dtype=np.uint8)
    logs = np.full(axes+(cns,6),-1,dtype=np.int8)
    memory = np.zeros((len(cuts),len(keys)),dtype=np.int32)
    cpu = np.zeros((len(cuts),len(keys),2))
    for ci,cut in enumerate(cuts):
        stores = []
        for k,key in enumerate(keys):
            t = time.process_time()
            store = train_history(key,settings[key.split('_')[0]],practice[:cut],vectors,seed+10000)
            cpu[ci,k,0] = time.process_time()-t
            memory[ci,k] = store.bytes()
            stores.append(store)
        for ti,(idx,target,alt,mat) in enumerate(cases):
            common_ids,_,_ = teacher(target,mat)
            x,logits = (arr[idx] for arr in vectors[domain])
            ps = []
            for k,store in enumerate(stores):
                t = time.process_time()
                prior = store.prior(x,logits)
                common[ci,k,ti] = replay(prior,target,mat,common_ids)
                ps.append(prior)
                for c in range(cns):
                    actual = alt if c==2 else target
                    snapshots,trace = deploy(prior,store.settings,actual,mat,corrupt=c==1)
                    native[ci,k,ti,c] = snapshots
                    logs[ci,k,ti,c,:len(trace)]=trace
                cpu[ci,k,1] += time.process_time()-t
            if winner is not None:
                wi = keys.index(winner)
                for c in range(cns):
                    trace = logs[ci,wi,ti,c]
                    trace = trace[trace>=0]
                    for k,prior in enumerate(ps):
                        scores = replay(prior,alt if c==2 else target,mat,trace,corrupt=c==1)
                        matched[ci,k,ti,c]=scores[-1],len(trace)
    return dict(common=common,native=native,matched=matched,traces=logs,memory=memory,cpu=cpu,
                practice=np.array([[0,0,0,0] if n==0 else practice_cost[n-1] for n in cuts]),
                targets=np.array([[i,t,a] for i,t,a,_ in cases]),
                seconds=np.array(time.process_time()-start))


def batch(domain,start,count,settings,winner=None,dev=False,workers=4):
    run = partial(lifetime,domain=domain,settings=settings,winner=winner,dev=dev)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run,range(start,start+count),chunksize=2))
    return {k:np.stack([row[k] for row in rows]) for k in rows[0]}


def develop():
    if (OUT/'frozen.json').exists():
        raise RuntimeError('Settings already frozen')
    variants = {'default':DEFAULT}
    for i in (1,2):
        obj=config(f'proposal-{i}.json')
        if 'settings' in obj:
            variants[f'codex{i}'] = obj['settings']
    losses = {}
    selected = {}
    for variant,settings in variants.items():
        data=batch('dev',610000,16,{key:settings for key in METHODS},dev=True)
        np.savez_compressed(OUT/f'dev-{variant}.npz',**data)
        ex=threshold(data['common']).mean(axis=(0,1,3))/7
        fails=(data['native'][...,0]<87).mean(axis=(0,1,3,4,5))
        losses[variant] = dict(zip(METHODS,((ex+fails)/2).tolist()))
    for key in METHODS:
        selected[key]=min(variants,key=lambda v:losses[v][key])
    ranked=sorted(METHODS,key=lambda k:losses[selected[k]][k])
    frozen=dict(variants=selected,settings={k:variants[selected[k]] for k in METHODS},
                winner=ranked[0],control=next(k for k in ranked if k!='weights'),
                writable='weights',development_loss=losses, ranking=ranked,
                files={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in
                       ['study.py','learner.py','world.py','encode.py','results/encoder.npz']})
    save_json('frozen.json',frozen)
    print(frozen,flush=True)


def test(domain):
    frozen = config('frozen.json')
    for p,v in frozen['files'].items():
        assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==v, p
    start = {'number':710000,'text':810000}[domain]
    data = batch(domain,start,96,frozen['settings'],frozen['winner'])
    path=OUT/f'{domain}-lifetimes.npz'
    np.savez_compressed(path,**data)
    print(path, path.stat().st_size, 'bytes',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('develop','number','text'))
    args = parser.parse_args()
    if args.phase=='develop':
        develop()
    else:
        test(args.phase)
