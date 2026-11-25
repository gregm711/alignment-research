"""Two demonstrations identify a private permutation, then test a fresh input."""
from itertools import permutations
from pathlib import Path
import json
import random
import sys
import time

HERE = Path(__file__).resolve().parent
PERMS = tuple(permutations(range(4)))


def permute(s, perm):
    return "".join(s[i] for i in perm)


def tasks(seed=776031, count=24):
    if count % 4:
        raise ValueError("use a multiple of four worlds for joint branch counterbalancing")
    rng = random.Random(seed)
    assignments = [(i%2, (i//2)%2) for i in range(count)]
    rng.shuffle(assignments)
    result = []
    for world, letter_ids in enumerate(assignments):
        symbols = rng.sample("ABCDEFGHJKLMNPRSTUVWXYZ", 10)
        positions = [(row, col) for row in (0,1) for col in (0,1)]
        rng.shuffle(positions)
        first = "".join(symbols[row] for row,col in positions)
        clue = "".join(symbols[2+col] for row,col in positions)
        query = "".join(symbols[4+(row^col)] for row,col in positions)
        qa = "".join(symbols[6:])
        p0 = rng.choice(PERMS)
        p1 = tuple(positions.index((positions[i][0], 1-positions[i][1])) for i in p0)
        first_out = permute(first,p0)
        assert first_out == permute(first,p1)
        intermediate = [permute(query,p) for p in (p0,p1)]
        q = rng.choice([p for p in PERMS if all(permute(s,p)!=s for s in intermediate)])
        qb = permute(qa,q)
        for depth in (1,2):
            def apply(s, p):
                mid = permute(s,p)
                return permute(mid,q) if depth == 2 else mid

            outcomes = [apply(query,p) for p in (p0,p1)]
            target = "P" if depth == 1 else "P, then pass P's output through Q"
            for fork,p in enumerate((p0,p1)):
                clue_out = permute(clue,p)
                candidates1 = [k for k in PERMS if permute(first,k)==first_out]
                candidates2 = [k for k in PERMS if permute(clue,k)==clue_out]
                both = set(candidates1) & set(candidates2)
                assert len(candidates1) == len(candidates2) == 4 and both == {p}
                assert [k for k in PERMS if permute(qa,k)==qb] == [q]
                for known in (candidates1,candidates2):
                    assert set(outcomes) <= {apply(query,k) for k in known}
                options = [outcomes[fork],outcomes[1-fork]]
                if letter_ids[fork]:
                    options.reverse()
                assert len(set(options)) == 2 and outcomes[fork] == options[letter_ids[fork]]
                intro = ("Machines P and Q each reorder the four positions of any input string "
                         "in a fixed order. Letters themselves never change.\n")
                old = f"P example: {first} -> {first_out}\n"
                rest = (f"Q example: {qa} -> {qb}\nNew P example: {clue} -> {clue_out}\n"
                        f"Start with {query}. Run it through {target}. What is the final output?\n"
                        f"A. {options[0]}\nB. {options[1]}\nAnswer with just A or B.")
                # Transfer the earliest lexicographically compatible mapping from the later example.
                guess = apply(query,candidates2[0])
                nearest = min(range(2), key=lambda j:sum(a!=b for a,b in zip(guess,options[j])))
                exact = apply(query, next(iter(both)))
                result.append(dict(world=world,depth=depth,fork=fork,answer="AB"[letter_ids[fork]],
                                   prompt=intro+old+rest,clue_only=intro+rest,
                                   matched="AB"[nearest],enumerated="AB"[options.index(exact)]))
    return result


def summary(rows):
    import numpy as np
    data = {}
    for depth in (1,2):
        subs = [r for r in rows if r["depth"]==depth]
        exacts = {arm:np.array([r[arm]==r["answer"] for r in subs]).reshape(-1,2)
                  for arm in ("full","partial","matched","enumerated")}
        score = {arm:dict(each=xs.sum(0).tolist(),both=int(xs.all(1).sum()),n=len(xs))
                 for arm,xs in exacts.items()}
        rng = np.random.default_rng(11429)
        draws = rng.integers(len(subs)//2,size=(5000,len(subs)//2))
        diff = (exacts["full"].astype(float)-exacts["partial"]).mean(1)
        score["paired_gain"] = [float(diff.mean()),*np.quantile(diff[draws].mean(1),[.025,.975]).tolist()]
        data[str(depth)] = score
    return data


def main():
    sys.path.insert(0,str(HERE.parent))
    from reader_mlx import MODEL, REVISION, Reader
    t0 = time.perf_counter()
    reader = Reader(MODEL,REVISION,batch_size=8)
    load = time.perf_counter()-t0
    rows = tasks()
    timings = {}
    for arm,prompt in (("full","prompt"),("partial","clue_only")):
        start = time.perf_counter()
        predicted = reader.predict_many([r[prompt] for r in rows])
        timings[arm] = time.perf_counter()-start
        for r, pred in zip(rows,predicted,strict=True):
            r[arm] = pred
    scores = summary(rows)
    out = dict(seed=776031, checkpoint=MODEL, revision=REVISION, load_seconds=load,
               inference_seconds=timings, summary=scores, rows=rows)
    (HERE/"results.json").write_text(json.dumps(out,indent=2)+"\n")
    print(scores, timings)


if __name__ == "__main__":
    main()
