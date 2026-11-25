"""Fixed-capacity online pointer readout. Persistent state is exactly one float tensor."""
from functools import lru_cache
from hashlib import blake2s
import string

import numpy as np

WIDTH = 16
STEPS = 32
# Each copy action addresses a digit at a fixed position, with a choice of output alphabet.
DIGITS = ("0123456789", "٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹", "०१२३४५६७८९", "০১২৩৪৫৬৭৮৯", "๐๑๒๓๔๕๖๗๘๙")
LITERALS = string.printable[:95] + "\u00a0\u200e\u200f\u202f年月日년월일"
COPY = WIDTH * len(DIGITS)
END = COPY + len(LITERALS)
CLASSES = END + 1


def _hash(text):
    return int.from_bytes(blake2s(text.encode(),digest_size=4).digest(),"little")


@lru_cache(maxsize=512)
def _input(source):
    if len(source) > WIDTH or any(c not in DIGITS[0] for c in source):
        raise ValueError("expected at most sixteen decimal digits")
    symbols = [""] * COPY + list(LITERALS) + [""]
    valid = np.ones(CLASSES,dtype=bool)
    for alphabet,chars in enumerate(DIGITS):
        for pos in range(WIDTH):
            token = alphabet*WIDTH+pos
            if pos < len(source):
                symbols[token] = chars[int(source[pos])]
            else:
                valid[token] = False
    return tuple(symbols),valid


class Direct:
    def __init__(self, slots=8):
        if slots < 1:
            raise ValueError("positive slot count required")
        # Twelve fixed feature rows per cell: shared, numeric length/shape, and a prefix feature.
        self.state = np.zeros((slots,12,STEPS,CLASSES),dtype=np.float32)

    def _read(self, producer, fields):
        if "n" in fields:
            source = fields["n"]
            shape = len(source)%8
            prefix = source[:2]
        else:
            source = fields["y"]+fields["m"].zfill(2)+fields["d"].zfill(2)
            shape = 2*(len(fields["m"])==2)+(len(fields["d"])==2)
            prefix = "dates"
        symbols,valid = _input(source)
        slot = _hash(str(producer)) % len(self.state)
        ids = (0,1+shape,9+_hash(prefix)%3)
        logits = self.state[slot,ids[0]] + .65*self.state[slot,ids[1]] + .25*self.state[slot,ids[2]]
        # A small fixed copy-through prior gives every fresh address a defined forecast.
        for pos in range(len(source)):
            logits[pos,pos]+=.1
        logits[len(source):,END]+=.1
        logits[:,~valid]=-1e9
        return slot,ids,symbols,logits

    def predict(self, producer, fields):
        _,_,symbols,logits = self._read(producer,fields)
        out=[]
        for token in logits.argmax(axis=1):
            if token==END:
                break
            out.append(symbols[token])
        return "".join(out)

    def confidence(self, producer, fields):
        _,_,_,scores=self._read(producer,fields)
        best = scores.argmax(axis=1)
        stops=np.flatnonzero(best==END)
        size=int(stops[0])+1 if len(stops) else STEPS
        top = np.partition(scores[:size],-2,axis=1)[:,-2:]
        gap=float((top[:,1]-top[:,0]).min())
        return float(-np.expm1(-gap))

    def write(self, producer, fields, gold):
        slot,ids,symbols,logits = self._read(producer,fields)
        # Ambiguous digit alignments share supervision; later distinct inputs resolve them.
        count=min(len(gold)+1,STEPS)
        z=logits[:count]
        p=np.exp(z-z.max(axis=1,keepdims=True))
        error=-p/p.sum(axis=1,keepdims=True)
        for pos,ch in enumerate(gold[:count]):
            matches=[j for j in range(COPY) if symbols[j]==ch]
            if not matches and ch in LITERALS:
                matches=[COPY+LITERALS.index(ch)]
            if not matches:
                error[pos]=0
                continue
            error[pos,matches]+=1/len(matches)
        if len(gold)<STEPS:
            error[-1,END]+=1
        for feature,weight in zip(ids,(1.,.65,.25)):
            row = self.state[slot,feature,:count]
            row *= .92
            row += np.float32(2*weight)*error


if __name__=="__main__":
    learner=Direct()
    original=learner.state.copy()
    for n in ("1234567","7651243","4126375"):
        learner.write("demo",{"n":n},n[:3]+"-"+n[3:])
    assert learner.predict("demo",{"n":"7321564"})=="732-1564"
    assert 0 <= learner.confidence("demo",{"n":"7321564"}) <= 1
    assert np.count_nonzero(np.any(learner.state!=original,axis=(1,2,3)))==1
    assert Direct(32).state.shape[0]==32
