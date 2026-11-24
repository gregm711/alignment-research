"""Learn a search order on SyGuS support rows, score independently authored BUSTLE queries."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
from statistics import median
from urllib.request import urlopen

REV = 'c43cb523fa9887513fb18bf088eb588f3fec5835'
DIGEST = {'new':'908f224cf8a2f904e057e1baf805c20b61c4eb40f4f7bad9b8d98b6d8920623e',
          'sygus':'e16b74d241acfb7d29a68f543148f360a8c0af177b44ff8dc9f8d6ebcf6ba4ca'}
HERE = Path(__file__).resolve().parent
SEPS = ('', ' ', '-', '_', '.', '/', ',', ':', '@', ', ')
UNARY = {
    'lower':str.lower, 'upper':str.upper, 'title':str.title, 'strip':str.strip,
    'digits':lambda x: ''.join(c for c in x if c.isdigit()),
    'letters':lambda x: ''.join(c for c in x if c.isalpha()),
    'alnum':lambda x: ''.join(c for c in x if c.isalnum()),
    'first':lambda x:x[:1], 'last':lambda x:x[-1:],
    'dropfirst':lambda x:x[1:], 'droplast':lambda x:x[:-1],
}
for size in (2,3,4):
    UNARY[f'prefix{size}'] = lambda x,n=size:x[:n]
    UNARY[f'suffix{size}'] = lambda x,n=size:x[-n:]
for separator in (' ','.','-','/','_',',',':','@'):
    for pos in (0,1,-1):
        def token(x, sep=separator, at=pos):
            parts=x.split(sep)
            return parts[at] if -len(parts)<=at<len(parts) else ''
        UNARY[f'token{separator}{pos}']=token

def load(kind):
    cache=Path.home()/'.cache'/'alignment-research'/f'crossbeam-{REV[:8]}'
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{kind}.json'
    if not path.exists():
        url=f'https://raw.githubusercontent.com/google-research/crossbeam/{REV}/crossbeam/data/{kind}/{kind}_benchmarks.json'
        path.write_bytes(urlopen(url,timeout=20).read())
    content=path.read_bytes()
    assert sha256(content).hexdigest()==DIGEST[kind]
    return json.loads(content)


def examples(rows):
    result={}
    for row in rows:
        x=tuple(row['inputs'])
        y=row['output']
        if x in result and result[x]!=y:
            raise ValueError('inconsistent labels')
        result[x]=y
    return result


@lru_cache(None)
def grammar(arity):
    # (left column, left transform chain, right column, right transform, separator)
    out={0:[],1:[],2:[],3:[]}
    compact=('', 'first','last','lower','upper','title','strip')
    for i in range(arity):
        out[0].append((i,(),-1,'',''))
        for f in UNARY:
            out[1].append((i,(f,),-1,'',''))
        for f,g in product(UNARY,repeat=2):
            out[2].append((i,(f,g),-1,'',''))
    for i,j,sep,f,g in product(range(arity),range(arity),SEPS,compact,compact):
        out[1+bool(f)+bool(g)].append((i,(f,) if f else (),j,g,sep))
    return tuple(code for group in out.values() for code in group)


@lru_cache(100000)
def mapped(x,chain):
    for f in chain:
        x=UNARY[f](x)
    return x


def execute(p,x):
    i,left,j,right,sep=p
    value=mapped(x[i],left)
    if j>=0:
        value+=sep+mapped(x[j],(right,) if right else ())
    return value


def select(order, support):
    for count,code in enumerate(order,1):
        if all(execute(code,x)==out for x,out in support.items()):
            return code,count
    return None,len(order)


def shuffled_prefix(archive, arity):
    standard=grammar(arity)
    eligible=set(standard)
    groups={}
    for p in standard:
        shape=(len(p[1]),p[2]>=0,bool(p[3]))
        groups.setdefault(shape,[]).append(p)
    rng=random.Random(321+arity)
    for group in groups.values():
        rng.shuffle(group)
    result=[]
    for p in archive:
        if p in eligible:
            result.append(groups[(len(p[1]),p[2]>=0,bool(p[3]))].pop())
    return result


def train(tasks):
    learned=Counter()
    all_x=set()
    supports=set()
    for t in tasks:
        s=examples(t['trainExamples'])
        if not s:
            continue
        supports.add(frozenset(s.items()))
        all_x.update(s)
        p,_=select(grammar(len(next(iter(s)))),s)
        if p is not None:
            learned[p]+=1
    return [p for p,_ in learned.most_common()],all_x,supports,learned


def main():
    train_tasks,target_tasks=load('sygus'),load('new')
    prior,seen,supports,counts=train(train_tasks)
    budgets=(64,512,1000000)
    trials=[]
    excluded={}
    for t in target_tasks:
        name=t['name']
        try:
            s=examples(t['trainExamples'])
            e=examples(t['testExamples'])
            examples(t['trainExamples']+t['testExamples'])
        except ValueError:
            excluded[name]='inconsistent examples'
            continue
        q={x:y for x,y in e.items() if x not in s and x not in seen}
        if not s or not q or frozenset(s.items()) in supports:
            excluded[name]='empty support, empty independent test or duplicated source support'
            continue
        n=len(next(iter(s)))
        base=grammar(n)
        assert len(base)<budgets[-1] and all(len(x)==n for x in [*s,*q])
        allowed=set(base)
        prefix=[p for p in prior if p in allowed]
        assert len(set(prefix))==len(prefix)
        shuf=shuffled_prefix(prior,n)
        row=dict(name=name,support=len(s),published_test=len(e),independent=len(q),removed=len(e)-len(q))
        for key,head in [('frozen',[]),('archive',prefix),('shuffled',shuf)]:
            head_set=set(head)
            order=tuple(head)+tuple(p for p in base if p not in head_set)
            assert len(order)==len(allowed)
            # Only support enters selection. The three capped searches are prefixes of this same search.
            p,k=select(order,s)
            success=p is not None and all(execute(p,x)==y for x,y in q.items())
            row[key]=dict(program=p,count=k,heldout_correct=success,
                          solved=[int(success and k<=budget) for budget in budgets])
        p,_=select(base,{**s,**q})
        row['oracle']=p is not None
        trials.append(row)
    summary=dict(source_tasks=len(train_tasks),source_solved=sum(counts.values()),
                 distinct_archived=len(prior),targets=len(target_tasks),eligible=len(trials),
                 oracle=sum(r['oracle'] for r in trials),excluded=excluded,budgets=[64,512,'exhaustive'],
                 total_independent_rows=sum(r['independent'] for r in trials),
                 removed_test_rows=sum(r['removed'] for r in trials))
    for key in ('frozen','archive','shuffled'):
        summary[key]=dict(solved=[sum(r[key]['solved'][i] for r in trials) for i in range(3)],
                          median_proposals=median(r[key]['count'] for r in trials))
    summary['archive_at_512']={key:{
        'wins':sum(r['archive']['solved'][1]>r[key]['solved'][1] for r in trials),
        'losses':sum(r['archive']['solved'][1]<r[key]['solved'][1] for r in trials)}
        for key in ('frozen','shuffled')}
    out=dict(summary=summary,archive=[dict(code=p,source_fits=counts[p]) for p in prior],trials=trials)
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
