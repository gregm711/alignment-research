"""One scoped correction in a finite language of 4-bit transforms."""
import csv
import gzip
import io
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
SCOPES = ('none', 'local', 'group', 'shared')
MODES = ('program', 'ordinal', 'literal', 'in_place')
WORDS = tuple(word for size in range(4) for word in product(range(4), repeat=size))


def execute(code, x):
    for op in code:
        if op == 0:
            x = (x << 1 & 15) | (x >> 3)
        elif op == 1:
            x = (x + 1) % 16
        elif op == 2:
            x ^= 3
        else:
            x = 3*x % 16
    return x


@lru_cache(None)
def truth(code):
    return tuple(execute(code, x) for x in range(16))


CANONICAL = {}
for word in WORDS:
    CANONICAL.setdefault(truth(word), word)
RULES = list(CANONICAL)
CODES = list(CANONICAL.values())
TABLES = np.array(RULES)
SHORT = [i for i, code in enumerate(CODES) if len(code) <= 2]
LONG = [i for i, code in enumerate(CODES) if len(code) == 3]


def bindings(scope, site, base):
    if scope == 0:
        return set()
    if scope == 1:
        return {site}
    if scope == 2:
        return set(range(site//4*4, site//4*4+4))
    return {s for s, rule in enumerate(base) if rule == base[site]}


def infer(base, site, receipts):
    where, xs, ys = np.array(receipts).T
    best = (-1, 0, 0)
    for scope in range(4):
        affected = np.isin(where, list(bindings(scope, site, base)))
        successes = (TABLES[:, xs[affected]] == ys[affected]).sum(axis=1)
        fixed = TABLES[np.array(base)[where[~affected]], xs[~affected]]
        scores = successes + (fixed == ys[~affected]).sum()
        rule = int(scores.argmax())
        best = max(best, (int(scores[rule]), -scope, -rule))
    return -best[1], -best[2]


def write(mode, site, scope, rule, receipts):
    blob = bytearray([site, scope, len(receipts)])
    for client,x,y in receipts:
        blob.extend((client, x*16+y))
    if scope:
        if mode in ('program','in_place'):
            blob.extend(CODES[rule])
        elif mode == 'ordinal':
            blob.append(WORDS.index(CODES[rule]))
        else:
            table = RULES[rule]
            blob.extend(table[i]*16+table[i+1] for i in range(0,16,2))
    return bytes(blob)


class Executable:
    def __init__(self, code):
        self.code = code

    def __getitem__(self, x):
        return execute(self.code, x)


def restore(mode, base, blob):
    site, scope, count = blob[:3]
    offset = 3+2*count
    evidence = [(blob[i], blob[i+1]//16, blob[i+1]%16) for i in range(3,offset,2)]
    changed = bindings(3 if mode=='in_place' and scope else scope, site, base)
    code = blob[offset:]
    installed = [RULES[b] for b in base]
    if scope:
        if mode == 'literal':
            table = tuple(y for b in code for y in divmod(b,16))
        elif mode == 'ordinal':
            table = truth(WORDS[code[0]])
        else:
            table = Executable(code)
        for c in changed:
            installed[c] = table
    return tuple(installed), changed, evidence


def materialize(policy, row):
    first, second, x = row
    return policy[second][policy[first][x]]


def dependency_index(rows):
    lists = [bytearray() for _ in range(16)]
    for i,(a,b,_) in enumerate(rows):
        for site in {a,b}:
            lists[site].append(i)
    blob = b''.join(bytes([len(ids)])+ids for ids in lists)
    offset = 0
    decoded = []
    while offset<len(blob):
        size = blob[offset]
        decoded.append(blob[offset+1:offset+1+size])
        offset += size+1
    assert offset==len(blob) and len(decoded)==16
    return blob, decoded


def run(seed):
    rng = random.Random(seed)
    a,b = rng.sample(SHORT,2)
    groups = [a,a,b,b]
    rng.shuffle(groups)
    base = [groups[s//4] for s in range(16)]
    site = rng.randrange(16)
    dirty = rng.choice(LONG)
    others = bindings(2,site,base)-{site}
    peers = bindings(3,site,base)-bindings(2,site,base)
    probes = [(s,x) for s in (site,rng.choice(sorted(others)),rng.choice(sorted(peers)))
              for x in rng.sample(range(16),4)]
    rows = [(rng.randrange(16),rng.randrange(16),rng.randrange(16)) for _ in range(128)]
    old = tuple(RULES[b] for b in base)
    cache = tuple(materialize(old,row) for row in rows)
    index, lists = dependency_index(rows)
    trials = []
    for cause in range(4):
        true_sites = bindings(cause,site,base)
        actual = tuple(RULES[dirty] if s in true_sites else old[s] for s in range(16))
        receipts = [(s,x,actual[s][x]) for s,x in probes]
        scope,rule = infer(base,site,receipts)
        expected = tuple(materialize(actual,row) for row in rows)
        exact_receipts = [(s,x,actual[s][x]) for s in sorted({s for s,_ in probes}) for x in range(16)]
        full_scope,full_rule = infer(base,site,exact_receipts)
        assert full_scope==cause and (not cause or full_rule==dirty)
        for mode in MODES:
            blob = write(mode,site,scope,rule,receipts)
            new,changed,readback = restore(mode,base,blob)
            assert readback==receipts
            fresh = tuple(tuple(output[x] for x in range(16)) for output in new)
            if mode!='in_place':
                assert all(fresh[s]==(RULES[rule] if s in bindings(scope,site,base) else old[s]) for s in range(16))
            affected = {i for s in changed for i in lists[s]}
            scanned = {i for i,(a,b,_) in enumerate(rows) if a in changed or b in changed} if changed else set()
            assert affected==scanned
            repaired=list(cache)
            for i in affected:
                repaired[i]=materialize(new,rows[i])
            assert repaired==[materialize(new,row) for row in rows]
            previous_ok=np.equal(old,actual)
            fresh_ok=np.equal(fresh,actual)
            old_book=np.equal(cache,expected)
            correct_book=np.equal(repaired,expected)
            trials.append(dict(seed=seed, cause=SCOPES[cause], mode=mode, inferred=SCOPES[scope],
                site=site, true_rule=dirty, inferred_rule=rule, correction_hex=blob.hex(),
                correct_policy=int(np.all(fresh_ok)),fresh_ok=int(fresh_ok.sum()),
                fresh_harmed=int((previous_ok & ~fresh_ok).sum()),fresh_initial_ok=int(previous_ok.sum()),
                fresh_missed=int((~previous_ok & ~fresh_ok).sum()),booked_ok=int(correct_book.sum()),
                booked_harmed=int((old_book & ~correct_book).sum()),booked_missed=int((~old_book & ~correct_book).sum()),
                correction_bytes=len(blob),index_bytes=len(index),dependency_lookups=len(changed),
                indexed_rows=len(affected),metadata_scan_rows=len(rows) if changed else 0,
                cache_reads=len(affected),cache_writes=sum(p!=q for p,q in zip(repaired,cache))))
    return trials


def main():
    seed_rows = [run(seed) for seed in range(2200000,2200512)]
    all_rows = [r for fork in seed_rows for r in fork]
    with gzip.GzipFile(HERE/'trials.csv.gz','wb',mtime=0) as raw:
        with io.TextIOWrapper(raw,encoding='utf-8',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    summary = {}
    for mode in MODES:
        rs = [r for r in all_rows if r['mode']==mode]
        summary[mode] = {k:mean(r[k] for r in rs) for k in
            ('correct_policy','correction_bytes','index_bytes','indexed_rows','dependency_lookups','metadata_scan_rows',
             'cache_reads','cache_writes','fresh_harmed','fresh_missed','booked_harmed','booked_missed')}
        summary[mode]['fresh_accuracy']=mean(r['fresh_ok']/256 for r in rs)
        summary[mode]['booked_accuracy']=mean(r['booked_ok']/128 for r in rs)
        summary[mode]['by_scope']={}
        for scope in SCOPES:
            subset=[r for r in rs if r['cause']==scope]
            summary[mode]['by_scope'][scope]=dict(
                exact=mean(r['correct_policy'] for r in subset),
                retention=mean(1-r['fresh_harmed']/r['fresh_initial_ok'] for r in subset),
                correction_bytes=mean(r['correction_bytes'] for r in subset))
    diffs=[]
    for fork in seed_rows:
        local={r['mode']:1-r['fresh_harmed']/r['fresh_initial_ok'] for r in fork if r['cause']=='local'}
        diffs.append(local['program']-local['in_place'])
    rng=np.random.default_rng(2200)
    bs=rng.choice(diffs,(4000,len(diffs))).mean(1)
    summary['local_retention_difference']=[mean(diffs),*np.quantile(bs,[.025,.975])]
    summary['language']={'syntactic':len(WORDS),'distinct':len(RULES),'old':len(SHORT),'replacements':len(LONG)}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for mode in MODES:
        print(mode, {k:v for k,v in summary[mode].items() if k!='by_scope'})
    print(summary['local_retention_difference'], summary['language'])


if __name__=='__main__':
    main()
