"""Learn a change from a single failure and paid queries to the changed device."""
from collections import Counter
from functools import lru_cache
import json
import random

from study import Machine, ROOT, bits, certificate, concat, fit, measure, polynomial

KNOWN=("whole","prefix","suffix","contains")
CAPS=(32,256,2048,8192)


def word(s):
    return int("1"+s,2)


def combine(a,b):
    cols=[list(a.columns[i])+[v<<a.width for v in b.columns[i]] for i in (0,1)]
    return Machine(a.start | b.start<<a.width,cols,a.output | b.output<<a.width)


def explicit(name,pattern,s):
    if name=="whole": return s==pattern
    if name=="prefix": return s.startswith(pattern)
    if name=="suffix": return s.endswith(pattern)
    if name=="contains": return pattern in s
    raise ValueError(name)


def panel(s):
    mid=len(s)//2
    return list(dict.fromkeys(("",s,"0"+s,"1"+s,s+"0",s+"1","0"+s+"1","1"+s+"0",
         s+s,s*3,s+"10"+s,s[::-1],s[:-1],s[1:],s[:mid]+s[mid+1:],
         s[:mid]+"01"+s[mid:],s[:mid]+"10"+s[mid:],
         "".join(c+"0" for c in s),"10"+s[::2]+"01")))


def learn(witness,read):
    prefix=[witness[:i] for i in range(len(witness)+1)]
    suffix=[witness[i:] for i in range(len(witness),-1,-1)]
    def row(p):
        return sum(read(p+s)<<i for i,s in enumerate(suffix))
    basis={}
    chosen=[]
    def reduce(x):
        coords=0
        while x and x.bit_length() in basis:
            v,y=basis[x.bit_length()]
            x ^= v
            coords ^= y
        return x,coords
    for p in prefix:
        v,y=reduce(row(p))
        if v:
            basis[v.bit_length()]=(v,y ^ (1<<len(chosen)))
            chosen.append(p)
    cols=[[],[]]
    for bit in (0,1):
        for p in chosen:
            v,y=reduce(row(p+str(bit)))
            if v: return None
            cols[bit].append(y)
    return Machine(reduce(row(""))[1],cols,sum(read(p)<<i for i,p in enumerate(chosen)))


class Exhausted(Exception):
    pass


def revise(old,witness,new_bit,device,cap,arm,named):
    # Only device() sees the changed world. The witness and incumbent are public.
    outcomes={witness:new_bit ^ old.predict(word(witness))}
    counts=Counter()
    def read(s):
        if s not in outcomes:
            if counts["probes"]==cap: raise Exhausted
            w=word(s)
            outcomes[s]=device(w)^old.predict(w)
            counts["probes"]+=1
            counts["bits_sent_and_received"]+=len(s)+1
        return outcomes[s]
    guards=panel(witness)
    def valid(m):
        if m is None: return False
        for s in guards:
            counts["checks"]+=1
            if m.predict(word(s))!=read(s):
                return False
        return True
    try:
        if arm!="discover":
            options=list(KNOWN)
            while len(options)>1:
                def split(s):
                    yes=sum(explicit(k,witness,s) for k in options)
                    return min(yes,len(options)-yes)
                q=max(guards,key=split)
                if not split(q): break
                y=read(q)
                options=[k for k in options if explicit(k,witness,q)==y]
            for k in options:
                m=named(k,witness)
                if valid(m): return combine(old,m),counts
            counts["rejected_menu"]+=1
        if arm!="explicit":
            m=learn(witness,read)
            if valid(m): return combine(old,m),counts
            counts["rejected_basis"]+=1
    except Exhausted:
        counts["exhausted"]+=1
    return None,counts
