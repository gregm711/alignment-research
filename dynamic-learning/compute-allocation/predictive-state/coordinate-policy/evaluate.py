"""Paired certified deployment trials, with the judge outside the energy meter."""
from policy import CAPS, ROOT, Machine, Counter, certificate, fit, json, measure, np, random, world
from study import fit as full_fit
from math import gcd
import statistics


def modular(rng, size):
    n={4:5,6:7,8:11}[size]
    radix=rng.choice([a for a in range(2,n) if gcd(a,n)==1])
    start=1<<rng.randrange(n)
    out=rng.randrange(1,(1<<n)-1)
    cols=[[1<<((radix*x+a)%n) for x in range(n)] for a in (0,1)]
    m=Machine(start,cols,out,True)
    return m,Machine(start,cols,out^(1<<rng.randrange(n)),True)


def panel(size,family):
    rng=random.Random((571901 if family=="modular" else 471901)+size)
    cases=[]
    while len(cases)<24:
        pair=modular(rng,size) if family=="modular" else world(rng,size,family)
        t,u=(m.table() for m in pair)
        if t==u or any(len(set(table[1:]))==1 for table in (t,u)):
            continue
        phases=[]
        for m,tab in zip(pair,(t,u)):
            guards={}
            while len(guards)<12:
                n=rng.randint(8,12)
                w=(1<<n)|rng.getrandbits(n)
                guards[w]=m.predict(w)
            words=tuple((1<<n)|rng.getrandbits(n) for n in (rng.randint(14,40) for _ in range(16)))
            phases.append((tab,tuple(guards.items()),words))
        cases.append((phases,pair))
    return cases


def run_cell(public,cap,arm,weights):
    counts=Counter()
    out=[]
    for phases in public:
        record=[]
        for tab,guards,words in phases:
            if arm=="full":
                assert cap==255
                counts["observations"]+=255+len(guards)
                counts["wire_bits"]+=sum(w.bit_length() for w in range(1,256))+sum(w.bit_length() for w,_ in guards)
                m=full_fit(tab,guards,15,"discover",counts)
            else:
                m=fit(tab,guards,cap,arm,weights,counts)
            record.append((m,tuple(m.predict(w) for w in words) if m is not None else None))
        out.append(record)
    return out,counts


def main():
    params={k:tuple(np.asarray(p,dtype="f") for p in v["weights"])
            for k,v in json.loads((ROOT/"weights.json").read_text()).items()}
    results={}
    for family in ("linear","states","modular"):
        for size in (4,6,8):
            cases=panel(size,family)
            public=[p for p,_ in cases]
            for cap in CAPS:
                arms=["short","preview","analytic","expand",*params]
                if cap==255: arms.append("full")
                records={}
                def perform(name):
                    return run_cell(public,cap,"neural" if name in params else name,params.get(name))
                for arm in arms:
                    (out,cnt),t=measure(lambda:perform(arm))
                    scores=[]
                    accepted=[]
                    for row,(phases,hidden) in zip(out,cases):
                        passed=[]
                        exists=[]
                        for (m,yp),(_,_,words),actual in zip(row,phases,hidden):
                            ok=m is not None and certificate(m,actual) is None
                            assert not ok or yp==tuple(actual.predict(w) for w in words)
                            passed.append(ok)
                            exists.append(m is not None)
                        scores.append(passed)
                        accepted.append(exists)
                    records[arm]=dict(scores=scores,accepted=accepted,counts=dict(cnt),costs=[],
                                      repeats=max(1,min(32,round(.020/t["wall_seconds"]))))
                for trial in range(3):
                    random.Random(size+cap+trial).shuffle(arms)
                    for arm in arms:
                        n=records[arm]["repeats"]
                        def batch():
                            for _ in range(n): perform(arm)
                        _,t=measure(batch)
                        records[arm]["costs"].append({k:v/n if v is not None else None for k,v in t.items()})
                results[f"{family}/{size}/{cap}"]=records
            print(family,size,flush=True)
        (ROOT/f"{family}-results.json").write_text(json.dumps({k:v for k,v in results.items()
            if k.startswith(family+"/")},separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
