"""Controlled simulator and grader, including two scopes absent from the solver menu."""
from collections import Counter
import json
import random

from revision import CAPS, KNOWN, ROOT, Machine, certificate, combine, explicit, fit, measure, panel, polynomial, revise, word

SCOPES=(*KNOWN,"odd","subsequence")


def pattern_machine(name,s):
    n=len(s)
    if name in ("whole","prefix"):
        size=n+2
    elif name=="odd":
        size=2*n
    else:
        size=n+1
    cols=[[],[]]
    output=0
    for i in range(size):
        if (i>=n if name=="odd" else i==n):
            output|=1<<i
        for bit in (0,1):
            b=str(bit)
            if name in ("whole","prefix"):
                if i==n and name=="prefix": j=n
                elif i<n and s[i]==b: j=i+1
                else: j=n+1
            elif name=="subsequence":
                j=i+int(i<n and s[i]==b)
            elif name=="contains" and i==n:
                j=n
            else:
                past=s[:i%n if name=="odd" else i]+b
                if name=="odd":
                    hit=past==s
                    j=max(k for k in range(n) if past.endswith(s[:k]))
                    j+=n*int((i>=n)^hit)
                else:
                    j=max(k for k in range(n+1) if past.endswith(s[:k]))
            cols[bit].append(1<<j)
    return Machine(1,cols,output,True)


def test_scope(k,s,t):
    if k in KNOWN: return explicit(k,s,t)
    if k=="odd": return sum(t.startswith(s,i) for i in range(len(t)))%2
    chars=iter(t)
    return all(c in chars for c in s)


def main():
    results={}
    rng=random.Random(991743)
    for n in (15,31):
        motifs=[]
        while len(motifs)<6:
            s=format(rng.getrandbits(n),f"0{n}b")
            if s not in motifs and all(s[:i]!=s[-i:] for i in range(1,n)):
                motifs.append(s)
        hidden_old=[polynomial(3,2+i%6) for i in range(6)]
        old_tables=[h.table() for h in hidden_old]
        def fit_old():
            return [fit(t,(),15,"discover",Counter()) for t in old_tables]
        old,basecost=measure(fit_old)
        assert all(certificate(a,b) is None for a,b in zip(old,hidden_old))
        results[f"{n}/old_fit"]=basecost
        for name in SCOPES:
            deltas=[pattern_machine(name,s) for s in motifs]
            hidden=[combine(a,b) for a,b in zip(hidden_old,deltas)]
            for s,d,new,prev in zip(motifs,deltas,hidden,hidden_old):
                assert new.table()==prev.table() and new.predict(word(s))!=prev.predict(word(s))
                for t in panel(s)+[s+s+"1",s+"11"+s,s[:n//2]+"101"+s[n//2:]]:
                    assert d.predict(word(t))==test_scope(name,s,t)
            for cap in CAPS:
                for arm in ("discover","explicit","cascade"):
                    def run():
                        rows=[]
                        for base,new,s in zip(old,hidden,motifs):
                            w=word(s)
                            rows.append(revise(base,s,new.predict(w),new.predict,cap,arm,pattern_machine))
                        return rows
                    rows,_=measure(run)
                    costs=[]
                    for _ in range(3):
                        _,time=measure(run)
                        costs.append(time)
                    exact=[]
                    counter=[]
                    for (m,_),h in zip(rows,hidden):
                        c=certificate(m,h) if m else None
                        exact.append(m is not None and c is None)
                        counter.append(c)
                    results[f"{n}/{name}/{cap}/{arm}"]=dict(exact=exact,witnesses=counter,
                          counts=[dict(c) for _,c in rows],costs=costs)
            print(n,name,flush=True)
    (ROOT/"changed-results.json").write_text(json.dumps(results,separators=(",",":"))+"\n")


if __name__=="__main__":
    main()
