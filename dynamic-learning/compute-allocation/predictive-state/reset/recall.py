"""Fixed-byte retention, validation, and refitting after a session reset."""
from collections import Counter
import json
import random
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"coordinate-policy"))
from evaluate import modular
from policy import world
from study import Machine, certificate, concat, fit, measure, realization

CAPS=(8,16,24,32)
LAYOUT={}
for length in range(4,8):
    for u in range(1,length):
        v=length-u-1
        LAYOUT[length,u]=[[concat(p,s) for s in range(1,2**(v+1))]
                          for p in range(1,2**(u+2))]


def write(table,guards,cap,arm):
    if arm=="cold":
        return b""
    if arm=="raw":
        return sum(table[w]<<(w-1) for w in range(1,min(256,8*cap+1))).to_bytes(cap,"little")
    m=fit(table,guards,15,"discover",Counter())
    if m:
        d=m.width
        if d>0 and 2*d*(d+1)<=8*(cap-1):
            z=0
            for value in [m.start,m.output,*m.columns[0],*m.columns[1]]:
                z=(z<<d)|value
            return bytes([d])+z.to_bytes(cap-1,"little")
    return bytes([0])+write(table,guards,cap-1,"raw")


def read(payload,arm):
    if arm=="cold":
        return None,{}
    data=payload
    if arm!="raw":
        d,data=data[0],data[1:]
        if d:
            assert 2*d*(d+1)<=8*len(data)
            z=int.from_bytes(data,"little")
            vals=[(z>>(d*i))&((1<<d)-1) for i in reversed(range(2+2*d))]
            return Machine(vals[0],[vals[2:2+d],vals[2+d:]],vals[1]),{}
    z=int.from_bytes(data,"little")
    return None,{w:(z>>(w-1))&1 for w in range(1,min(256,8*len(data)+1))}


def recover(ask,old,guards):
    """Try all layouts for each complete depth, validating against every available bit."""
    evidence=dict(old)
    for length in range(4,8):
        bound=1<<(length+1)
        for w in range(1,bound):
            if w not in evidence:
                evidence[w]=ask(w)
        for u in sorted(range(1,length),key=lambda v:abs(2*v+1-length)):
            grid=LAYOUT[length,u]
            h=[sum(evidence[w]<<i for i,w in enumerate(row)) for row in grid]
            m=realization(h,15,"discover",2**(u+1)-1)
            if m is None: continue
            tab=m.table()
            if all(tab[w]==y for w,y in evidence.items()) and all(m.predict(w)==ask(w) for w in guards):
                return m
    return None


def recall(payload,arm,teacher,checks,guards,words):
    obs={}
    def ask(w):
        if w not in obs:
            obs[w]=teacher(w)
        return obs[w]
    m,old=read(payload,arm)
    initial=m is not None
    trusted=False
    if m is not None:
        if all(m.predict(w)==ask(w) for w in (*checks,*guards)):
            trusted=True
        else: m=None
    elif old:
        if not all(old[w]==ask(w) for w in checks):
            old={}
    if m is None:
        m=recover(ask,old,guards)
        if m is None and old:
            m=recover(ask,{},guards)
    yp=tuple(m.predict(w) for w in words) if m is not None else None
    return m,yp,dict(observations=len(obs),wire_bits=sum(w.bit_length() for w in obs),
                     stored_operator=initial,reused_operator=trusted)


def dataset(n,family):
    rng=random.Random((711903 if family=="modular" else 611903)+n)
    out=[]
    while len(out)<16:
        a,b=modular(rng,n) if family=="modular" else world(rng,n,family)
        t,u=a.table(),b.table()
        if t==u or any(len(set(val[1:]))==1 for val in (t,u)):continue
        def longs(k,low,high):
            ws={}
            while len(ws)<k:
                length=rng.randint(low,high)
                ws[(1<<length)|rng.getrandbits(length)]=None
            return tuple(ws)
        first=tuple((w,a.predict(w)) for w in longs(12,8,12))
        checks=tuple(rng.sample(range(8,56),4))
        out.append((a,b,t,first,checks,longs(12,8,12),longs(16,14,40)))
    return out


def timed(fn):
    result,t=measure(fn)
    repeats=max(1,min(256,round(.025/t["wall_seconds"])))
    def batch():
        for _ in range(repeats): fn()
    costs=[]
    for _ in range(3):
        _,t=measure(batch)
        costs.append({k:v/repeats if v is not None else None for k,v in t.items()})
    return result,costs


def main():
    for family in ("linear","states","modular"):
        summary={}
        for size in (4,6,8):
            cases=dataset(size,family)
            initial=[]
            for a,_,t,gs,*_ in cases:
                m=fit(t,gs,15,"discover",Counter())
                initial.append(m is not None and certificate(m,a) is None)
            for cap in CAPS:
                for arm in ("cold","raw","operator"):
                    if arm=="cold" and cap!=8:continue
                    def saving():
                        return [write(c[2],c[3],cap,arm) for c in cases]
                    payloads,costs=timed(saving)
                    assert all(len(x)==(0 if arm=="cold" else cap) for x in payloads)
                    key=f"{size}/{cap}/{arm}"
                    summary[key]=dict(initial=initial,write_cost=costs)
                    for state in (0,1):
                        def reading():
                            return [recall(p,arm,c[state].predict,*c[4:]) for p,c in zip(payloads,cases)]
                        rows,costs=timed(reading)
                        exact=[]
                        counter=[]
                        stats=[]
                        for p,(m,ys,counts),c in zip(payloads,rows,cases):
                            actual=c[state]
                            ce=certificate(m,actual) if m is not None else None
                            ok=m is not None and ce is None
                            assert not ok or ys==tuple(actual.predict(w) for w in c[-1])
                            if arm=="operator":
                                # Frozen interpreter accepts precisely the same serialized parameters.
                                other,pred,log=recall(bytes(bytearray(p)),"state",actual.predict,*c[4:])
                                assert pred==ys and log==counts and (m is None)==(other is None)
                                assert m is None or certificate(m,other) is None
                            exact.append(ok)
                            counter.append(ce)
                            stats.append(counts)
                        summary[key][("stable","changed")[state]]=dict(exact=exact,counter=counter,
                                      counts=stats,read_cost=costs)
            print(family,size,flush=True)
        (ROOT/f"{family}-results.json").write_text(json.dumps(summary,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
