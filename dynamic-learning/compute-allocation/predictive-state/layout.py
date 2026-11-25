"""Compare several compatible observation-table shapes on the same evidence."""
from collections import Counter
import json
import random
import statistics
from study import ROOT, Machine, certificate, concat, fit, generated, measure
from tomita import corpus

LAYOUTS = [(3,3),(4,2),(5,1),(6,0),(2,4),(1,5),(0,6)]
INDICES = [([[concat(p,s) for s in range(1,2**(v+1))]
             for p in range(1,2**(u+2))],2**(u+1)-1) for u,v in LAYOUTS]


def state_fit(table,guards,counts):
    for words,limit in INDICES:
        counts["layouts"] += 1
        h=[sum(table[w]<<i for i,w in enumerate(row)) for row in words]
        reps={}
        for p,obs in enumerate(h[:limit],1):
            reps.setdefault(obs,p)
        index={r:1<<i for i,r in enumerate(reps)}
        try:
            cols=[[index[h[2*p+a-1]] for p in reps.values()] for a in (0,1)]
        except KeyError:
            continue
        m=Machine(index[h[0]],cols,sum(index[r] for r in reps if r&1),True)
        if m.table()[1:]==table[1:] and all(m.predict(w)==y for w,y in guards):
            return m
    return None


def evaluate(cases,arm):
    out=[]
    counts=Counter()
    for inputs,_,_ in cases:
        row=[]
        for table,guards,words in inputs:
            m=None
            if arm=="cascade":
                for short in ("affine","quadratic"):
                    m=fit(table,guards,15,short,counts)
                    if m: break
            if m is None and arm in ("states","cascade"):
                m=state_fit(table,guards,counts)
            if m is None and arm in ("discover","cascade"):
                m=fit(table,guards,15,"discover",counts)
            row.append((m,tuple(m.predict(w) for w in words) if m else None))
        out.append(row)
    return out,counts


def main():
    summary={}
    for name,(cases,_) in [("tomita",corpus()),("rank6",generated(6)),("polynomial",generated(0))]:
        arms=["discover","states","cascade"]
        records={}
        for a in arms:
            (rows,cnt),cost=measure(lambda:evaluate(cases,a))
            solved=0
            for row,(_,truth,hidden) in zip(rows,cases):
                valid=all(m and certificate(m,h) is None for (m,_),h in zip(row,hidden))
                assert not valid or all(ys==label for (_,ys),label in zip(row,truth))
                solved+=bool(valid)
            records[a]=dict(solved=solved,counts=dict(cnt),costs=[],repeats=max(1,round(.035/cost["wall_seconds"])))
        for trial in range(7):
            random.Random(53901+trial).shuffle(arms)
            for a in arms:
                reps=records[a]["repeats"]
                def batch():
                    for _ in range(reps):
                        evaluate(cases,a)
                _,t=measure(batch)
                records[a]["costs"].append({k:v/reps if v is not None else None for k,v in t.items()})
        summary[name]=records
    (ROOT/"layout-results.json").write_text(json.dumps(summary,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
