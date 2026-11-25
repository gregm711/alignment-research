"""Identify a recurrent machine from finite observations and audit all-string equivalence."""
from collections import Counter, deque
from itertools import permutations
from pathlib import Path
import json
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "circuit-communication"))
from meter import measure

CAPS = (2, 4, 8, 15)
SHORT = range(1, 256)


def bits(word):
    return bin(word)[3:]


def parity(x):
    return x.bit_count() & 1


def linear(columns, state):
    value = 0
    while state:
        bit = state & -state
        value ^= columns[bit.bit_length()-1]
        state ^= bit
    return value


class Machine:
    def __init__(self, start, columns, output, ordinary=False):
        self.start, self.columns, self.output = start, columns, output
        self.ordinary = ordinary
        self.width = len(columns[0])

    def step(self, state, action):
        cols = self.columns[action]
        if self.ordinary:
            return cols[state.bit_length()-1]
        return linear(cols, state)

    def predict(self, word):
        s = self.start
        for b in bits(word):
            s = self.step(s, int(b))
        return parity(s & self.output)

    def table(self):
        states = [0, self.start]
        for i in range(1, 128):
            states.extend((self.step(states[i], 0), self.step(states[i], 1)))
        return bytes(parity(s & self.output) for s in states)


def certificate(a, b):
    """Reachable paired span suffices by linearity; a returned integer encodes a witness."""
    queue = deque([(a.start | b.start << a.width, 1)])
    basis = {}
    mask = (1 << a.width)-1
    while queue:
        state, word = queue.popleft()
        x, y = state & mask, state >> a.width
        if parity(x & a.output) != parity(y & b.output):
            assert a.predict(word) != b.predict(word)
            return word
        v = state
        while v and v.bit_length() in basis:
            v ^= basis[v.bit_length()]
        if v:
            basis[v.bit_length()] = v
            for act in (0, 1):
                queue.append((a.step(x, act) | b.step(y, act) << a.width, 2*word+act))
    return None


def features(state, act):
    z, o = (state >> 1) & 1, (state >> 2) & 1
    if act == 0:
        return state ^ ((state & 1)<<1 | z<<3 | o<<5)
    return state ^ ((state & 1)<<2 | z<<4 | o<<6)


def polynomial(width, output):
    cols = [[features(1<<i, a) & ((1<<width)-1) for i in range(width)] for a in (0,1)]
    return Machine(1, cols, output)


def equations(rows, width):
    basis = {}
    for x, y in rows:
        while x and x.bit_length() in basis:
            v, w = basis[x.bit_length()]
            x, y = x^v, y^w
        if x:
            basis[x.bit_length()] = x, y
        elif y:
            return None
    result = 0
    for i in sorted(basis):
        x, y = basis[i]
        if parity(x & result) != y:
            result |= 1 << (i-1)
    return result


def concat(p, s):
    n = s.bit_length()-1
    return (p << n) | (s ^ (1<<n))


# Fixed input layout and feature semantics contain no learned labels.
SUFFIXES = [[concat(p,s) for s in range(1,16)] for p in range(32)]
FEATURES = [0,1]
for parent in range(1,128):
    FEATURES.extend(features(FEATURES[parent], a) for a in (0,1))


def rows(table):
    return [sum(table[w] << i for i,w in enumerate(words)) for words in SUFFIXES[1:]]


def realization(h, cap, mode):
    if mode == "states":
        reps = {}
        for p, row in enumerate(h[:15], 1):
            reps.setdefault(row, p)
        index = {row: 1<<i for i,row in enumerate(reps)}
        try:
            cols = [[index[h[2*p+a-1]] for p in reps.values()] for a in (0,1)]
        except KeyError:
            return None
        output = sum(index[row] for row in reps if row & 1)
        return Machine(index[h[0]],cols,output,True)
    basis, reps = {}, []
    def reduce(row):
        code = 0
        while row and row.bit_length() in basis:
            v, w = basis[row.bit_length()]
            row ^= v
            code ^= w
        return row, code
    for p, row in enumerate(h[:15],1):
        rem, code = reduce(row)
        if rem:
            if len(reps) >= cap:
                return None
            basis[rem.bit_length()] = rem, code ^ (1 << len(reps))
            reps.append(p)
    cols = [[],[]]
    for a in (0,1):
        for p in reps:
            row = h[2*p+a-1]
            if mode == "scramble":
                row = (row >> 1) | ((row & 1) << 14)
            rem, code = reduce(row)
            if rem:
                return None
            cols[a].append(code)
    return Machine(reduce(h[0])[1],cols,sum((h[p-1]&1)<<i for i,p in enumerate(reps)))


ORDERS = {"discover": ("discover",), "states": ("states",), "scramble": ("scramble",),
          "affine": ("affine",), "quadratic": ("quadratic",),
          "cascade": ("affine","quadratic","states","discover"),
          "first": ("discover","affine","quadratic","states")}


def fit(table, guards, cap, arm, counts):
    h = None
    for proposal in ORDERS[arm]:
        counts["attempts"] += 1
        if proposal in ("affine","quadratic"):
            width = 3 if proposal == "affine" else 7
            mask = (1<<width)-1
            coef = equations(((FEATURES[w]&mask,table[w]) for w in SHORT),width)
            model = polynomial(width,coef) if coef is not None else None
        else:
            if h is None:
                h = rows(table)
            model = realization(h,cap,proposal)
        if model is not None:
            counts["table_checks"] += 1
            if model.table()[1:] == table[1:]:
                for w,y in guards:
                    counts["guard_checks"] += 1
                    if model.predict(w) != y:
                        break
                else:
                    counts["accept_"+proposal] += 1
                    counts["fitted_dimensions"] += model.width
                    return model
        counts["rejections"] += 1
    return None


def random_word(rng, n):
    mode = rng.randrange(5)
    if mode == 0:
        s = str(rng.randrange(2))*n
    elif mode == 1:
        motif = format(rng.randrange(8),"03b")
        s = (motif*n)[:n]
    elif mode == 2:
        s = "".join(rng.choice("01")*rng.randint(1,6) for _ in range(n))[:n]
    else:
        s = format(rng.getrandbits(n),f"0{n}b")
    return int("1"+s,2)


def phase(machine, rng):
    table = machine.table()
    guards = {}
    while len(guards)<12:
        w=random_word(rng,rng.randint(8,12))
        guards[w]=machine.predict(w)
    pools = [{},{}]
    for _ in range(6000):
        w=concat(random_word(rng,rng.randint(7,20)),random_word(rng,rng.randint(7,20)))
        y=machine.predict(w)
        if len(pools[y])<8:
            pools[y][w]=y
        if all(len(p)==8 for p in pools):
            items=list(pools[0].items())+list(pools[1].items())
            rng.shuffle(items)
            words,truth=zip(*items)
            return (table,tuple(guards.items()),words),truth
    return None


def fixtures(pairs, seed):
    rng=random.Random(seed)
    result=[]
    rejects=0
    for a,b in pairs:
        if a.table()==b.table() or any(len(set(m.table()[1:]))==1 for m in (a,b)):
            rejects+=1
            continue
        phases=[phase(m,rng) for m in (a,b)]
        if None in phases:
            rejects+=1
            continue
        result.append((tuple(p[0] for p in phases),tuple(p[1] for p in phases),(a,b)))
        yield result[-1],rejects


def generated(d, n=24):
    rng=random.Random(814103+d)
    def pairs():
        while True:
            if d==0:
                width=rng.choice((3,7))
                c=rng.randrange(2,1<<width)
                yield polynomial(width,c),polynomial(width,c^(1<<rng.randrange(width)))
            else:
                cols=[[rng.randrange(1,1<<d) for _ in range(d)] for _ in (0,1)]
                a=Machine(rng.randrange(1,1<<d),cols,rng.randrange(1,1<<d))
                changed=[r[:] for r in cols]
                changed[rng.randrange(2)][rng.randrange(d)] ^= 1<<rng.randrange(d)
                yield a,Machine(a.start,changed,a.output)
    from itertools import islice
    cases=list(islice(fixtures(pairs(),819103+d),n))
    return [v for v,_ in cases],cases[-1][1]


def cell(cases, cap, arm):
    public=[c[0] for c in cases]
    def run():
        answers=[]
        counts=Counter()
        for episode in public:
            row=[]
            for table,guards,words in episode:
                m=fit(table,guards,cap,arm,counts)
                row.append((m,tuple(m.predict(w) for w in words) if m else None))
            answers.append(row)
        return answers,counts
    (out,counts),cost=measure(run)
    repeats=max(1,min(256,round(.018/cost["wall_seconds"])))
    def batch():
        for _ in range(repeats):
            run()
    trials=[]
    for _ in range(3):
        _,t=measure(batch)
        trials.append({k:v/repeats if v is not None else None for k,v in t.items()})
    def audit():
        phases=[]
        for row,(_,truth,ref) in zip(out,cases):
            item=[]
            for (m,pred),ys,hidden in zip(row,truth,ref):
                counter=certificate(m,hidden) if m else None
                exact=m is not None and counter is None
                assert not exact or pred==ys
                item.append(dict(sample=pred==ys,exact=exact,witness=counter,
                                 width=m.width if m else None))
            phases.append(item)
        return phases
    phases,audit_cost=measure(audit)
    return dict(phases=phases,counts=dict(counts),costs=trials,audit_cost=audit_cost)


def sweep(groups,name):
    data={}
    for group,(cases,rejected) in groups:
        for cap in CAPS:
            for arm in ORDERS:
                if arm in ("affine","quadratic","states","scramble") and cap!=15:
                    continue
                data[f"{group}/{cap}/{arm}"]=cell(cases,cap,arm)
        print(group,len(cases),"fixture rejections",rejected,flush=True)
        data[f"{group}/rejections"]=rejected
    (ROOT/name).write_text(json.dumps(data,separators=(",",":"))+"\n")


def checks():
    for p in range(1,25):
        for s in range(1,25):
            assert bits(concat(p,s))==bits(p)+bits(s)
    for width in (3,7):
        model=polynomial(width,3)
        assert model.table()[1:]==bytes(model.predict(w) for w in SHORT)
        assert certificate(model,model) is None
        assert certificate(model,polynomial(width,2)) is not None
        assert equations([(1,0),(1,1)],1) is None


if __name__=="__main__":
    checks()
    sweep(((str(d),generated(d)) for d in (2,4,6,8,0)),"generated.json")
