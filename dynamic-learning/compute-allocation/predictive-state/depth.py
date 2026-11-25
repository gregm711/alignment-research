"""Composition of learned operators on exponentially long, explicitly shared programs."""
from collections import Counter
import json
import random
import statistics

from study import ROOT, Machine, certificate, fit, fixtures, linear, measure, parity

DEPTHS = (6,24,96)
BUDGETS = (512,4096,32768)


def independent(rows):
    basis={}
    for x in rows:
        while x and x.bit_length() in basis:
            x ^= basis[x.bit_length()]
        if not x: return False
        basis[x.bit_length()]=x
    return True


def systems(n):
    rng=random.Random(919221+n)
    def pairs():
        while True:
            cols=[]
            for _ in (0,1):
                while True:
                    c=[rng.randrange(1,1<<n) for _ in range(n)]
                    if independent(c):
                        cols.append(c)
                        break
            a=Machine(rng.randrange(1,1<<n),cols,rng.randrange(1,1<<n))
            new=[c[:] for c in cols]
            i,j=rng.sample(range(n),2)
            op=rng.randrange(2)
            new[op]=[x ^ (((x>>i)&1)<<j) for x in new[op]]
            yield a,Machine(a.start,new,a.output)
    from itertools import islice
    return [v for v,_ in islice(fixtures(pairs(),933001+n),12)]


def instance(seed):
    rng=random.Random(seed)
    levels=[None,[(0,0),(0,1),(1,0),(1,1)]]
    for _ in range(2,max(DEPTHS)+1):
        levels.append([divmod(x,4) for x in rng.sample(range(16),4)])
    def short():
        return tuple(rng.choices((0,1),k=rng.randint(2,7)))
    return levels,[(short(),i%4,short()) for i in range(16)]


def walk(m,x,seq):
    for a in seq:
        x=m.step(x,a)
    return x


def reference(m,levels,queries,depth):
    # Independent column product: direct bit scan; also compare against literal expansion at depth 6.
    def apply(cols,x):
        out=0
        for i,v in enumerate(cols):
            if x & (1<<i): out ^= v
        return out
    transforms=m.columns
    for pairs in levels[1:depth+1]:
        transforms=[[apply(transforms[j],v) for v in transforms[i]] for i,j in pairs]
    ys=[parity(walk(m,apply(transforms[i],walk(m,m.start,pre)),post)&m.output)
        for pre,i,post in queries]
    if depth==6:
        words=[(0,),(1,)]
        for pairs in levels[1:7]:
            words=[words[i]+words[j] for i,j in pairs]
        assert ys==[parity(walk(m,m.start,pre+words[i]+post)&m.output) for pre,i,post in queries]
    return ys


class End(Exception):
    pass


def execute(m,levels,queries,d,arm,budget):
    used=0
    peak=0
    def charge(n=1):
        nonlocal used
        if used+n>budget:
            raise End
        used+=n
    try:
        if arm=="memo":
            cache={}
            def call(k,i,x):
                nonlocal peak
                key=(k,i,x)
                if key not in cache:
                    charge()
                    if k:
                        l,r=levels[k][i]
                        y=call(k-1,r,call(k-1,l,x))
                    else:
                        y=m.step(x,i)
                    cache[key]=y
                    peak=max(peak,len(cache))
                return cache[key]
        else:
            dense=arm=="dense"
            if dense:
                charge(2*(1<<m.width))
                block=[[m.step(x,a) for x in range(1<<m.width)] for a in (0,1)]
            else:
                block=m.columns
            for pairs in levels[1:d+1]:
                nxt=[]
                for l,r in pairs:
                    charge(len(block[l]))
                    if dense:
                        nxt.append([block[r][v] for v in block[l]])
                    else:
                        nxt.append([linear(block[r],v) for v in block[l]])
                peak=max(peak,(len(block)+len(nxt))*len(block[0]))
                block=nxt
            def call(k,i,x):
                return block[i][x] if dense else linear(block[i],x)
        ys=[]
        for pre,i,post in queries:
            charge(len(pre)+len(post)+1)
            ys.append(parity(walk(m,call(d,i,walk(m,m.start,pre)),post)&m.output))
        return ys,used,peak
    except End:
        return None,used,peak


def main():
    summaries={}
    for n in (4,8):
        cases=systems(n)
        dag=[instance(919221+100*n+i) for i in range(len(cases))]
        certified=[]
        for public,_,ref in cases:
            cert=[]
            for (table,guards,_),hidden in zip(public,ref):
                m=fit(table,guards,15,"discover",Counter())
                cert.append(m is not None and certificate(m,hidden) is None)
            certified.append(cert)
        for depth in DEPTHS:
            targets=[]
            shortcut=[]
            for (_,_,hidden),(levels,queries) in zip(cases,dag):
                ys=[reference(h,levels,queries,depth) for h in hidden]
                targets.append(ys)
                skip=[[parity(walk(h,h.start,pre+post)&h.output) for pre,_,post in queries] for h in hidden]
                six=[reference(h,levels,queries,6) for h in hidden]
                shortcut.append([skip==ys,six==ys, all(len(set(p))==1 for p in ys)])
            for cap in BUDGETS:
                for arm in ("memo","basis","dense"):
                    def run():
                        episodes=[]
                        work=Counter()
                        for (public,_,_),(levels,queries) in zip(cases,dag):
                            row=[]
                            for table,guards,_ in public:
                                m=fit(table,guards,15,"discover",work)
                                if m is None:
                                    row.append(None)
                                else:
                                    pred,used,peak=execute(m,levels,queries,depth,arm,cap)
                                    work["work"]+=used
                                    work["peak_entries"]=max(work["peak_entries"],peak)
                                    row.append(pred)
                            episodes.append(row)
                        return episodes,work
                    (pred,cnt),_=measure(run)
                    scored=[p==t for p,t in zip(pred,targets)]
                    for p,t,cert in zip(pred,targets,certified):
                        for yp,yt,ok in zip(p,t,cert):
                            if yp is not None and ok: assert yp==yt
                    costs=[]
                    for _ in range(3):
                        _,t=measure(run)
                        costs.append(t)
                    summaries[f"{n}/{depth}/{cap}/{arm}"]=dict(solved=sum(scored),phases_certified=certified,
                        accepted=scored,counts=dict(cnt),costs=costs,shortcut=shortcut)
            print(n,depth,flush=True)
    (ROOT/"depth-results.json").write_text(json.dumps(summaries,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
