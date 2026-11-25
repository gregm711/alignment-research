"""Check a finite inductive certificate independently of the learner and simulator."""
import json
from pathlib import Path


def verify(record):
    a,b=record["systems"]
    widths=[len(m["ops"][0]) for m in (a,b)]
    n,k=widths
    for m,w in zip((a,b),widths):
        assert len(m["ops"])==2 and all(len(row)==w for row in m["ops"])
        assert all(0<=v<1<<w for row in m["ops"] for v in row)
        assert all(0<=m[key]<1<<w for key in ("output","start"))
    output=a["output"] | (b["output"]<<n)
    start=a["start"] | (b["start"]<<n)
    pivots={}
    def remainder(x):
        for bit in sorted(pivots,reverse=True):
            if x & (1<<bit):
                x ^= pivots[bit]
        return x
    basis=record["invariant_basis"]
    for v in basis:
        assert 0<v<1<<(n+k) and (v & output).bit_count()%2==0
        r=remainder(v)
        assert r
        pivots[r.bit_length()-1]=r
    assert remainder(start)==0
    for action in (0,1):
        op=a["ops"][action]+[v<<n for v in b["ops"][action]]
        for v in basis:
            after=0
            for i,image in enumerate(op):
                if v & (1<<i): after ^= image
            assert remainder(after)==0


if __name__=="__main__":
    if not __debug__:
        raise SystemExit("Run with Python assertions enabled.")
    proof=json.loads(Path(__file__).with_name("example-proof.json").read_text())
    for rec in proof["phases"]:
        verify(rec)
    print("Verified initial agreement, closure under both actions and equal outputs in both phases.")
