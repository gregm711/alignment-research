"""Learn to buy diagnostic continuations; every learner receives the same oracle."""
import os
for var in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from study import Machine, SUFFIXES, certificate, concat, measure, realization

S = tuple(range(1, 16))
WORDS = {s: tuple(SUFFIXES[p][s-1] for p in range(1,32)) for s in S}
MASKS = {s: sum(1<<w for w in WORDS[s]) for s in S}
CAPS = (64,128,192,255)


def reduce_col(v, pivots):
    while v and v.bit_length() in pivots:
        v ^= pivots[v.bit_length()]
    return v


def add_col(v, pivots):
    v = reduce_col(v,pivots)
    if v:
        pivots[v.bit_length()] = v
    return bool(v)


class Evidence:
    def __init__(self, table):
        self._device = table.__getitem__
        self.known = {}
        self.mask = 0
        self.wire = 0

    def ask(self, w):
        if w not in self.known:
            self.known[w] = self._device(w)
            self.mask |= 1 << w
            self.wire += w.bit_length()  # input length plus answer bit
        return self.known[w]

    def column(self, s, n):
        return sum(self.ask(w)<<i for i,w in enumerate(WORDS[s][:n]))

    def cost(self, s):
        return (MASKS[s] & ~self.mask).bit_count()


def features(candidates, scout, chosen, evidence, scout_basis):
    previous = [scout[s] for s in chosen]
    result = []
    for s in candidates:
        c = scout[s]
        nearest = min(((c^p).bit_count() for p in previous), default=7)
        f = [float(bool(reduce_col(c,scout_basis))), nearest/7, c.bit_count()/7,
             evidence.cost(s)/31, (s.bit_length()-1)/3, len(chosen)/15, len(scout_basis)/7]
        f.extend((c>>i)&1 for i in range(7))
        f.extend(int(s==i) for i in S)
        result.append(f)
    return np.asarray(result,dtype=np.float32)


def world(rng, n, family):
    if family == "linear":
        columns=[[rng.randrange(1,1<<n) for _ in range(n)] for _ in range(2)]
        a=Machine(rng.randrange(1,1<<n),columns,rng.randrange(1,1<<n))
        new=[col[:] for col in columns]
        new[rng.randrange(2)][rng.randrange(n)] ^= 1<<rng.randrange(n)
        return a,Machine(a.start,new,a.output)
    cols=[[1<<rng.randrange(n) for _ in range(n)] for _ in range(2)]
    a=Machine(1,cols,rng.randrange(1,(1<<n)-1),True)
    new=[c[:] for c in cols]
    action,i=rng.randrange(2),rng.randrange(n)
    new[action][i]=1<<((new[action][i].bit_length()-1+rng.randrange(1,n))%n)
    return a,Machine(1,new,a.output,True)


def train_set(nworlds):
    rng=random.Random(371901)
    X,Y=[],[]
    for _ in range(nworlds):
        a,_=world(rng,rng.choice((3,4,5,6,7,8)),rng.choice(("linear","states")))
        tab=a.table()
        labels={s:sum(tab[w]<<i for i,w in enumerate(WORDS[s])) for s in S}
        for trajectory in range(3):
            e=Evidence(tab)
            scout={s:e.column(s,7) for s in S}
            basis,small,chosen={},{},[]
            candidates=list(S)
            while candidates:
                vals=[int(bool(reduce_col(labels[s],basis))) for s in candidates]
                X.append(features(candidates,scout,chosen,e,small))
                Y.extend(vals)
                if not chosen:
                    s=1
                elif trajectory==0:
                    s=candidates[0]
                elif trajectory==1:
                    s=rng.choice(candidates)
                else:
                    s=max(candidates,key=lambda c:(bool(reduce_col(scout[c],small)),-e.cost(c),-c))
                e.column(s,31)
                add_col(labels[s],basis)
                add_col(scout[s],small)
                chosen.append(s)
                candidates.remove(s)
                if not any(vals):
                    break
    return np.concatenate(X),np.array(Y,dtype=np.float32)


def train(x,y,width):
    rng=np.random.default_rng(371901+width)
    w=rng.normal(0,.15,(x.shape[1],width)).astype("f")
    b=np.zeros(width,"f")
    v=rng.normal(0,.15,width).astype("f")
    bias=np.zeros(1,"f")
    for _ in range(12):
        order=rng.permutation(len(x))
        for i in range(0,len(order),192):
            ids=order[i:i+192]
            batch=x[ids]
            h=np.tanh(batch@w+b)
            err=(1/(1+np.exp(-(h@v+bias)))-y[ids])/len(ids)
            grad_h=(err[:,None]*v)*(1-h*h)
            v-=.12*(h.T@err)
            bias-=.12*err.sum()
            w-=.12*(batch.T@grad_h)
            b-=.12*grad_h.sum(axis=0)
    return (w,b,v,bias)


def fit(table,guards,cap,arm,weights,counts):
    e=Evidence(table)
    preview = arm in ("preview","analytic","neural")
    scout={s:e.column(s,7) for s in S} if preview else {}
    open_set=set(S) if arm!="expand" else {1}
    chosen=[]
    h=[0]*31
    pivots,small={},{}
    def finish(m):
        assert len(e.known)<=cap
        counts.update(observations=len(e.known)+len(guards),
                      wire_bits=e.wire+sum(w.bit_length() for w,_ in guards))
        return m
    if arm=="full" and cap<255:
        return finish(None)
    while True:
        allowed=sorted(s for s in open_set if e.cost(s)+len(e.known)<=cap)
        if not allowed: return finish(None)
        if not chosen:
            s=1
        elif arm=="analytic":
            s=max(allowed,key=lambda j:(bool(reduce_col(scout[j],small)),-e.cost(j),-j))
        elif arm=="neural":
            x=features(allowed,scout,chosen,e,small)
            w,b,v,bias=weights
            s=allowed[int(np.argmax(np.tanh(x@w+b)@v+bias))]
        else:
            s=allowed[0]
        open_set.remove(s)
        col=e.column(s,31)
        informative=add_col(col,pivots)
        if preview:
            add_col(scout[s],small)
        if arm=="expand" and (s==1 or informative):
            open_set.update(c for c in (concat(2,s),concat(3,s)) if c<=15)
        for i in range(31):
            h[i] |= ((col>>i)&1)<<len(chosen)
        chosen.append(s)
        if arm=="full":
            if open_set: continue
        elif not informative and s!=1:
            continue
        counts["attempts"]+=1
        m=realization(h,15,"discover")
        if m is not None:
            observed=m.table()
            if all(observed[w]==y for w,y in e.known.items()):
                for w,y in guards:
                    counts["guard_checks"]+=1
                    if m.predict(w)!=y: break
                else:
                    return finish(m)
        counts["rejected"]+=1


def save_training():
    summary={}
    for n in (48,192):
        (x,y),label_cost=measure(lambda:train_set(n))
        for w in (8,32):
            params,cost=measure(lambda:train(x,y,w))
            name=f"w{w}-n{n}"
            summary[name]={"weights":[p.tolist() for p in params],"fit_cost":cost,
                           "teacher_cost":label_cost,"examples":int(len(y))}
    (ROOT/"weights.json").write_text(json.dumps(summary,separators=(",",":"))+"\n")


if __name__=="__main__":
    save_training()
