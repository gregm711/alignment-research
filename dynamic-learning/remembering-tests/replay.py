"""Replay industrial test verdicts with an observation boundary at each test choice."""
import csv
from collections import defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
REV = '85bae8b6dd58ef7d49c4c15690ba628a5187d4f7'
HASH = {'paintcontrol':'cdf4c0c2bca580acaf7981f2608258cdf480ca821c88aa2eca6c2333c5c6b11a',
        'iofrol':'4efd94e87d4d3dca26970db5d4a5d7b0db75e5eb0c0cc1cc2807e5ad93a627b4'}
METHODS = ('frequency','latest','retrieve','diagnostic','random_first','equal','market','uniform','oracle')
BUDGETS = (1,3,8)


def rows(system, first_only=False):
    path = Path.home()/'.cache'/'alignment-research'/'ci-question-pilot'/f'{system}.csv'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        data = urlopen(f'https://raw.githubusercontent.com/yourkevin/atcs-data/{REV}/{system}.csv',
                       timeout=60).read()
        assert hashlib.sha256(data).hexdigest()==HASH[system]
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==HASH[system]
    cycles = defaultdict(dict)
    duplicates = 0
    with path.open() as file:
        for row in csv.DictReader(file,delimiter=';'):
            build,name,y = int(row['Cycle']),row['Name'],int(row['Verdict'])
            assert y in (0,1)
            before = cycles[build].get(name)
            duplicates += int(before is not None)
            if before is None or not first_only:
                cycles[build][name] = max(before or 0,y)
    return cycles,duplicates


def diagnostic(values,seen):
    count = seen.T@seen
    sums = values.T@seen
    mean = sums/np.maximum(count,1)
    joint = (values.T@values)/np.maximum(count,1)
    cov = joint-mean*mean.T
    var = mean*(1-mean)
    gain = np.where(count>=4, cov**2/np.maximum(var,1e-10)*count/max(1,len(values)), 0)
    np.fill_diagonal(gain,0)
    return gain.sum(1)


def mixture(freq,latest,recent,conditional,shares):
    return np.sum(shares*np.column_stack((freq,latest,recent,conditional)),axis=1)


def order(method,values,seen,freq,last,recent,probe_value,shares,rng,read):
    weights = np.ones(len(values))
    used = np.zeros(len(freq),bool)
    seq = []
    for k in range(min(8,len(freq))):
        if method=='frequency':
            score = freq
        elif method=='latest':
            score = last
        elif method=='diagnostic' and k==0:
            score = probe_value
        elif method=='random_first' and k==0:
            score = rng.random(len(freq))
        else:
            conditional = (1+weights@values)/(2+weights@seen)
            if method in ('equal','market'):
                score = mixture(freq,last,recent,conditional,shares if method=='market' else 0.25)
            else:
                score = conditional
        # Stable ordering breaks exact ties by frequency and then lexical test identity.
        i = int(np.lexsort((np.arange(len(freq)),-freq,-score,used))[0])
        assert not used[i]
        used[i] = True
        result = read(i)
        seq.append(i)
        weights *= np.where((seen[:,i]>0)&(values[:,i]!=result),.1,1.)
    return seq


def bootstrap(numer,denom,draws):
    total = denom[draws].sum(1)
    keep = total>0
    if not keep.any() or not denom.sum():
        return None
    samples = numer[draws].sum(1)[keep]/total[keep]
    return [float(numer.sum()/denom.sum()),*np.quantile(samples,[.025,.975]).tolist()]


def replay(system, first_only=False):
    cycles,duplicates = rows(system,first_only)
    builds = sorted(cycles)
    wealth = {}
    records = []
    for j,b in enumerate(builds):
        names = sorted(cycles[b])
        current = np.array([cycles[b][n] for n in names])
        past = [cycles[i] for i in builds[max(0,j-32):j]]
        raw = np.array([[p.get(n,-1) for n in names] for p in past],dtype=float).reshape(len(past),len(names))
        seen = (raw>=0).astype(float)
        values = np.maximum(raw,0)
        freq = (1+values.sum(0))/(2+seen.sum(0))
        recent = (1+values[-4:].sum(0))/(2+seen[-4:].sum(0))
        last = np.full(len(names),.5)
        for row in raw:
            last = np.where(row<0,last,.1+.8*row)
        shares = np.array([wealth.get(n,[.25]*4) for n in names])
        experts = np.column_stack((freq,last,recent,freq))
        if j>=len(builds)//3:
            diags = diagnostic(values,seen)
            n,f = len(names),int(sum(current))
            records_for_build = {}
            for method in METHODS[:-2]:
                seq=order(method,values,seen,freq,last,recent,diags,shares,
                          np.random.default_rng(809+b),current.__getitem__)
                finds=np.cumsum(current[seq])
                records_for_build[method]=[int(finds[min(k,n)-1]) for k in BUDGETS]
            records_for_build['uniform']=[min(k,n)*f/n for k in BUDGETS]
            records_for_build['oracle']=[min(k,f) for k in BUDGETS]
            hit={m:[int(x>0) for x in v] for m,v in records_for_build.items()}
            hit['uniform']=[1-comb(n-f,min(k,n))/comb(n,min(k,n)) for k in BUDGETS]
            bits={}
            for policy,p in [('market',(shares*experts).sum(1)),('equal',experts.mean(1))]:
                bits[policy]=float(-np.log2(np.where(current,p,1-p)).mean())
            records.append(dict(cycle=b,tests=n,failures=f,found=records_for_build,hit=hit,bits=bits))
        factor = np.where(current[:,None],experts,1-experts)
        after = shares*factor
        after /= after.sum(1,keepdims=True)
        log_wealth = np.log(shares)+np.log(factor)
        alternative = np.exp(log_wealth-np.logaddexp.reduce(log_wealth,1)[:,None])
        assert np.allclose(after,alternative) and np.allclose(shares.sum(1),1)
        wealth.update(zip(names,.95*after+.05/4))
    return records,duplicates,len(builds)


def report(records):
    n=len(records)
    rng=np.random.default_rng(803)
    starts=rng.integers(0,n-10+1,size=(3000,(n+9)//10))
    draws=(starts[:,:,None]+np.arange(10)).reshape(len(starts),-1)[:,:n]
    f=np.array([r['failures'] for r in records])
    h={m:np.array([r['hit'][m] for r in records]) for m in METHODS}
    found={m:np.array([r['found'][m] for r in records]) for m in METHODS}
    baseline=max(('frequency','latest'),key=lambda m:h[m][:,1].sum())
    pairs=[('retrieve',baseline),('diagnostic','retrieve'),('diagnostic','random_first'),('market','equal')]
    return dict(scored_cycles=n,failing_cycles=int(sum(f>0)),failures=int(f.sum()),best_fixed=baseline,
                hit_on_failing={m:(v.sum(0)/(f>0).sum()).tolist() for m,v in h.items()},
                share_of_failures={m:(v.sum(0)/f.sum()).tolist() for m,v in found.items()},
                paired_three_check_gain={a+'-'+b:bootstrap(h[a][:,1]-h[b][:,1],f>0,draws) for a,b in pairs},
                bits_saved=bootstrap(np.array([r['bits']['equal']-r['bits']['market'] for r in records]),
                                     np.ones(n),draws))


if __name__=='__main__':
    for name in HASH:
        rec,dup,total=replay(name)
        out=dict(total_cycles=total,duplicates=dup,summary=report(rec),cycles=rec)
        (HERE/f'{name}.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
        print(name,json.dumps(out['summary']),flush=True)
