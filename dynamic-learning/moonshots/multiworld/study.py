"""Matched four-byte memories and one-probe policies on paired diagnostic futures."""
from argparse import ArgumentParser
from bisect import bisect_right
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
from time import process_time

import numpy as np

NAMES = ('branches', 'particles', 'marginals', 'fifo', 'exact', 'map', 'uniform')
POLICIES = ('decision', 'disagreement', 'replay', 'no_probe')
TRUTH = np.array([[bin(i & mask).count('1') % 2 for i in range(32)]
                  for mask in range(1, 32)], dtype=bool)
UNIFORM = np.full(32, 1/32)
CHOOSE = [[comb(i,k) if i>=k else 0 for i in range(67)] for k in range(7)]


def likelihood(reports):
    hits = (TRUTH[reports//2] == (reports%2)[:,None]).sum(axis=0)
    weights = 7.0**hits
    return weights/weights.sum()


def pack(values, width):
    packed = sum(int(v) << (i*width) for i,v in enumerate(values))
    return packed.to_bytes(4, 'little')


def unpack(data, width, count):
    n = int.from_bytes(data,'little')
    return np.array([(n >> (i*width)) & ((1<<width)-1) for i in range(count)])


def archive(reports, p):
    top = np.argsort(-p, kind='stable')[:4]
    weights = np.maximum(1,np.rint(7*p[top]/p[top[0]])).astype(int)
    particles = np.searchsorted(p.cumsum(), (np.arange(6)+.5)/6)
    marginals = TRUTH[[0,1,3,7,15]] @ p
    rank = sum(CHOOSE[i+1][int(r)+i] for i,r in enumerate(sorted(reports)))
    return [pack(top+32*weights,8), pack(particles,5), pack(np.rint(63*marginals),6),
            pack(reports[-5:],6), pack([rank],32), pack(top[:1],5), bytes(4)]


def recover(data, name):
    if name == 'exact':
        rank = int.from_bytes(data, 'little')
        reports = []
        for k in range(6,0,-1):
            index = bisect_right(CHOOSE[k],rank)-1
            rank -= CHOOSE[k][index]
            reports.append(index-k+1)
        assert rank==0 and all(0<=r<62 for r in reports)
        return likelihood(np.array(reports))
    if name == 'fifo':
        p = likelihood(unpack(data,6,5))
    elif name == 'marginals':
        m = unpack(data,6,5)/63
        p = np.where(TRUTH[[0,1,3,7,15]],m[:,None],1-m[:,None]).prod(axis=0)
    elif name == 'branches':
        entries = unpack(data,8,4)
        p = np.bincount(entries%32,weights=entries//32,minlength=32)
    elif name in ('map','particles'):
        p = np.bincount(unpack(data,5,1 if name=='map' else 6),minlength=32)
    else:
        assert name=='uniform'
        return UNIFORM
    return .99*p/p.sum()+.01*UNIFORM


def joint_distribution(weights, target, probes):
    # Dimensions: proposed test, test result, service decision.
    high = TRUTH[probes] @ (weights*(TRUTH[target]))
    mass = TRUTH[probes] @ weights
    total_high = TRUTH[target] @ weights
    joint = np.empty((len(probes),2,2))
    joint[:,1,1] = high
    joint[:,1,0] = mass-high
    joint[:,0,1] = total_high-high
    joint[:,0,0] = weights.sum()-mass-(total_high-high)
    return joint


def first_max(values):
    # All exact ties should have the same winner after float matrix products.
    return int(np.argmax(np.round(values,12)))


def evaluate_case(seed, extra_writer=None, extra_reader=None):
    rng = np.random.default_rng(seed)
    world = rng.integers(32)
    sensors = rng.integers(31,size=6)
    readings = TRUTH[sensors,world] ^ (rng.random(6)<.125)
    reports = sensors*2+readings
    selected = rng.integers(6)
    calibration = sensors[selected]
    target,*menu = rng.choice(np.delete(np.arange(31),calibration),6,replace=False).tolist()
    probes = np.asarray(menu)
    p = likelihood(reports)
    blobs = archive(reports,p)
    decoded = [recover(b,k) for k,b in zip(NAMES,blobs)]
    if extra_writer is not None:
        extra = extra_writer(reports)
        blobs += extra
        decoded += [extra_reader(i,b) if extra_reader else recover(b,'branches') for i,b in enumerate(extra)]
    assert all(len(b)==4 for b in blobs) and np.allclose(decoded[NAMES.index('exact')],p)
    scores = np.zeros((len(decoded),len(POLICIES)))
    contradictory = np.zeros_like(scores)
    disagree_choices = np.zeros(len(decoded))
    bad_mass = 0.
    for bit in (0,1):
        admitted = TRUTH[calibration]==bit
        true_joint = joint_distribution(p*admitted,target,probes)
        oracle_query = first_max(true_joint.max(axis=2).sum(axis=1))
        fork_mass = p[admitted].sum()
        bad = bit!=readings[selected]
        bad_mass += fork_mass*bad
        for i,prior in enumerate(decoded):
            q = prior*admitted
            assumed = joint_distribution(q/q.sum(),target,probes)
            decision = first_max(assumed.max(axis=2).sum(axis=1))
            disagreement = first_max(-abs(assumed[:,1].sum(axis=1)-.5))
            disagree_choices[i] += (decision!=disagreement)*fork_mass
            predictions = np.round(assumed,12).argmax(axis=2)
            recorded = true_joint[np.arange(len(probes))[:,None],np.arange(2),predictions].sum(axis=1)
            votes = assumed[0].sum(axis=0)
            without = true_joint[0].sum(axis=0)[first_max(votes)]
            values = np.array([recorded[decision], recorded[disagreement], recorded[oracle_query], without])
            scores[i] += values
            contradictory[i] += values*bad
    assert np.all(scores<=scores[NAMES.index('exact'),0]+1e-10)
    return dict(seed=seed, reports=reports.tolist(), calibration=int(calibration), target=target, menu=menu,
                states=[b.hex() for b in blobs], accuracy=scores.tolist(), contradiction_correct=contradictory.tolist(),
                contradiction_mass=bad_mass, different_question_mass=disagree_choices.tolist())


def run(path):
    assert comb(67,6)<=2**32 and len(set(map(tuple,TRUTH)))==31
    start = process_time()
    rows = [evaluate_case(seed) for seed in range(947000,948024)]
    elapsed = process_time()-start
    contents = ''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows).encode()
    path.mkdir(parents=True,exist_ok=True)
    (path/'cases.jsonl.gz').write_bytes(gzip.compress(contents,mtime=0))
    a = np.array([r['accuracy'] for r in rows])
    c = np.array([r['contradiction_correct'] for r in rows])
    mass = np.array([r['contradiction_mass'] for r in rows])
    different = np.array([r['different_question_mass'] for r in rows])
    rng = np.random.default_rng(947)
    draws = rng.integers(len(a), size=(4000,len(a)))
    def interval(d):
        return [float(d.mean()),*np.quantile(d[draws].mean(1),[.025,.975]).tolist()]
    summary = dict(memory_bytes=4, exact_code_bits=(comb(67,6)-1).bit_length(), cases=len(a),
                   sha256_uncompressed=hashlib.sha256(contents).hexdigest(), names=NAMES, policies=POLICIES,
                   accuracy=a.mean(0).tolist(), contradictory_accuracy=(c.sum(0)/mass.sum()).tolist(),
                   contradictory_mass=float(mass.mean()), different_question_mass=different.mean(0).tolist(),
                   paired_intervals=dict(branches_minus_marginals_oracle_query=interval(a[:,0,2]-a[:,2,2]),
                                         branches_decision_minus_disagreement=interval(a[:,0,0]-a[:,0,1]),
                                         branches_minus_exact=interval(a[:,0,0]-a[:,4,0]),
                                         exact_decision_minus_disagreement=interval(a[:,4,0]-a[:,4,1])))
    (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['paired_intervals']),f'CPU seconds: {elapsed:.2f}')


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    run(parser.parse_args().out)
