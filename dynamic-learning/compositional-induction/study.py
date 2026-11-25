"""Infer executable rules from examples and test on new nested expressions."""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/brendenlake/MLC/34837ccb06c855bb29558ad99bb1cef70debd559/data_human/few_shot/val_gold/mini_scan.txt"
SHA = "e0cbe17dc429e42a7743926d6dbf804f85f13768311f44e9396142a85916e4b7"
POLICIES = ("random","active","easy")
PATTERNS = [p for length in range(1,5) for p in itertools.product((0,1),repeat=length)]


def dataset():
    p = Path.home()/".cache/alignment-research/miniscan-public.txt"
    if not p.exists():
        with urlopen(URL,timeout=30) as response:
            b = response.read()
        assert hashlib.sha256(b).hexdigest()==SHA
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b)
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest()==SHA
    raw = b.decode().split("*GRAMMAR*")[0].split("*SUPPORT*")[1]
    out = []
    for group in raw.split("*QUERY*"):
        rows = []
        for line in group.strip().splitlines():
            x,y = line.removeprefix("IN: ").split(" OUT: ")
            rows.append((tuple(x.split()),tuple(y.split())))
        out.append(rows)
    return out


def anonymize(support,query,rng):
    def mapping(symbols,prefix):
        tokens = sorted(set(symbols))
        codes = [prefix+str(i) for i in range(len(tokens))]
        rng.shuffle(codes)
        return dict(zip(tokens,codes))
    inp = mapping((t for x,_ in support for t in x),"i")
    out = mapping((t for _,y in support for t in y),"o")
    def convert(rows):
        return [(tuple(inp[t] for t in x),tuple(out[t] for t in y)) for x,y in rows]
    return convert(support),convert(query)


def conventions(ops):
    return [(order,(a,b)) for order in itertools.permutations(ops) for a,b in itertools.product((0,1),repeat=2)]


def execute(x,rule):
    primitives,templates,unary,binary,order,side = rule
    for op in order:
        if op==unary:
            if x[-1]==op and len(x)>1:
                return execute(x[:-1],rule)*len(templates[op])
        elif op in x:
            pos = x.index(op) if side[binary.index(op)] else len(x)-1-x[::-1].index(op)
            if pos==0 or pos==len(x)-1:
                raise ValueError("Incomplete binary expression")
            children = execute(x[:pos],rule),execute(x[pos+1:],rule)
            return sum((children[k] for k in templates[op]),())
    if all(t in primitives for t in x):
        return sum((primitives[t] for t in x),())
    raise ValueError("Unparsed expression")


def induce(rows):
    prim = {x[0]:y for x,y in rows if len(x)==1}
    options = {}
    unary = None
    for x,y in rows:
        if len(x)==2 and x[0] in prim and x[1] not in prim:
            unary = x[1]
            valid = {(0,)*n for n in range(1,5) if prim[x[0]]*n==y}
            options[unary] = options.get(unary,valid)&valid
        if len(x)==3 and x[0] in prim and x[2] in prim and x[1] not in prim:
            op = x[1]
            args = (prim[x[0]],prim[x[2]])
            valid = {p for p in PATTERNS if sum((args[i] for i in p),())==y}
            options[op] = options.get(op,valid)&valid
    binary = sorted(set(options)-{unary})
    assert unary is not None and len(binary)==2
    ops = [unary,*binary]
    rules = []
    for choices in itertools.product(*(sorted(options[k],key=lambda x:(len(x),x)) for k in ops)):
        templates = dict(zip(ops,choices))
        for order,side in conventions(ops):
            rule = (prim,templates,unary,binary,order,side)
            if all(execute(x,rule)==y for x,y in rows):
                rules.append(rule)
    return rules


def distance(a,b):
    row = list(range(len(b)+1))
    for i,x in enumerate(a):
        nxt = [i+1]
        for j,y in enumerate(b):
            nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row = nxt
    return row[-1]


def official():
    s,q = anonymize(*dataset(),random.Random(60514))
    local = [(x,y) for x,y in s if len(x)<=3]
    start = time.process_time()
    base,rules = induce(local),induce(s)
    fit_ms = 1000*(time.process_time()-start)
    def predict(bank,x):
        return execute(x,bank[0]) if bank else ()
    result = []
    for x,y in q:
        assert x not in {a for a,_ in s}
        _,copy = min(s,key=lambda row:distance(row[0],x))
        result.append(dict(x=x,gold=y,copy=copy,copy_possible=y in {b for _,b in s},
                           atomic=predict(base,x),full=predict(rules,x)))
    return dict(support=len(s),local=len(local),candidates=[len(base),len(rules)],
                fit_cpu_ms=fit_ms,queries=result)


def world(rng):
    tokens = [f"i{k}" for k in range(7)]
    rng.shuffle(tokens)
    inputs,ops = tokens[:4],tokens[4:]
    colors = [(f"o{k}",) for k in range(4)]
    rng.shuffle(colors)
    prim = dict(zip(inputs,colors))
    unary,*binary = ops
    valid = [p for p in PATTERNS if set(p)=={0,1} and p!=p[::-1]]
    forms = [(0,)*rng.choice([2,3]),*rng.sample(valid,2)]
    order,side = rng.choice(conventions(ops))
    true = (prim,dict(zip(ops,forms)),unary,binary,order,side)
    atomic = [(t,) for t in inputs]+[(inputs[0],unary),(inputs[0],binary[0],inputs[1]),(inputs[2],binary[1],inputs[3])]
    start = [(x,execute(x,true)) for x in atomic]
    seen = set(atomic)
    def draw():
        while True:
            seq = [rng.choice(inputs)]
            for k in range(rng.choice([2,3])):
                op = rng.choice(ops)
                if op==unary:
                    seq.append(op)
                elif rng.randrange(2):
                    seq += [op,rng.choice(inputs)]
                else:
                    seq = [rng.choice(inputs),op]+seq
            x = tuple(seq)
            if x not in seen:
                seen.add(x)
                return x
    menus = [[draw() for _ in range(8)] for _ in range(2)]
    return start,menus,[draw() for _ in range(24)],true


def trial(rng):
    support,menus,query,truth = world(rng)
    hypotheses = induce(support)
    assert len(hypotheses)==24
    items = sum(menus,[])+query
    given = [execute(x,truth) for x in items]
    predictions = [[execute(x,r) for x in items] for r in hypotheses]
    assert given in predictions
    scores = {}
    for policy in POLICIES:
        active = list(range(len(hypotheses)))
        seen = {y for _,y in support}
        curve = []
        for t in range(3):
            proposed = predictions[active[0]][16:]
            accuracy = sum(x==y for x,y in zip(proposed,given[16:]))/len(query)
            ceiling = sum(y in seen for y in given[16:])/len(query)
            curve.append([accuracy,float(accuracy==1),len(active)/24,ceiling])
            if t==2:
                break
            def diversity(j):
                counts = {}
                for i in active:
                    y=predictions[i][j]
                    counts[y]=counts.get(y,0)+1
                return 1-sum((v/len(active))**2 for v in counts.values())
            ids = range(8*t,8*t+8)
            k = max(ids,key=diversity) if policy=="active" else min(ids,key=diversity) if policy=="easy" else ids.start
            active = [i for i in active if predictions[i][k]==given[k]]
            seen.add(given[k])
        scores[policy] = curve
    return [scores[k] for k in POLICIES]


def main():
    public = official()
    begin = time.process_time()
    a = np.array([trial(random.Random(770131+i)) for i in range(96)])
    elapsed = time.process_time()-begin
    rng = np.random.default_rng(770132)
    draws = rng.integers(96,size=(4096,96))
    diff = a[:,1]-a[:,0]
    summary = np.stack([diff.mean(0),*np.quantile(diff[draws].mean(1),[.025,.975],axis=0)],axis=-1)
    report = dict(official=public,policies=POLICIES,metrics=["exact_output","all_24","fraction_rules","copy_ceiling"],
                  means=a.mean(0).tolist(),active_minus_random=summary.tolist(),trials=a.tolist(),
                  simulated_cpu_seconds=elapsed)
    (ROOT/"results.json").write_text(json.dumps(report,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
