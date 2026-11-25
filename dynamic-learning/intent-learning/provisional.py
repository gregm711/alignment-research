"""Write an addition to a text classifier and verify it on selected requests."""
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
from prepare import CACHE

ROOT = Path(__file__).resolve().parent
POLICIES = ("random", "addition", "opening")
SOURCES = ("same", "second", "uniform", "perfect")
GATES = ("both", "one", "always")
NAMES = ("change_accuracy", "new_recall", "new_repair", "stable_harm", "old_accuracy",
         "change_release", "stable_release", "change_selected_error", "stable_selected_error",
         "change_menu_error", "stable_menu_error", "old_change_accuracy", "stable_accuracy")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def metric(z, pools, practice):
    rng = np.random.default_rng(4181)
    same = np.zeros(z.shape[1])
    different = same.copy()
    for _ in range(8192):
        names = practice[rng.integers(len(practice))]
        a, b = rng.choice(names, 2, replace=False)
        i, j = rng.choice(pools["train", a], 2, replace=False)
        k = rng.choice(pools["train", b])
        same += (z[i]-z[j])**2
        different += (z[i]-z[k])**2
    gains = np.clip((different/8192+.0005)/(same/8192+.0005), .25, 4.)
    return (gains/gains.mean()).astype(np.float32)


def assignments(z, pools, names, count, seed):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < count:
        intents = rng.choice(names, 4, replace=False)
        blocks = [rng.permutation(pools["train", i]) for i in intents]
        old_proto = unit(np.stack([z[blocks[i][:16]].mean(0) for i in (0, 2, 3)]))
        def accepted(ids):
            return (z[ids] @ old_proto.T).argmax(-1) == 0
        # Menus and verifier banks are disjoint from the eligible opening pool.
        eligible = blocks[1][24:][~accepted(blocks[1][24:])]
        if len(eligible) == 0:
            continue
        opening = rng.choice(eligible)
        ids, latent = [], []
        for t in range(2):
            seq = np.stack([b[20+2*t:22+2*t] for b in blocks]).ravel()
            order = rng.permutation(8)
            ids.extend(seq[order]); latent.extend(np.repeat(np.arange(4),2)[order])
        query = np.concatenate([rng.choice(pools["test", i],16,replace=False) for i in intents])
        allids = np.r_[opening, ids, query]
        assert len(np.unique(allids)) == 81
        verifiers = []
        for start in (16,18):
            centers = unit(np.stack([z[b[start:start+2]].mean(0) for b in blocks]))
            verifiers.append((z[ids] @ centers.T).argmax(-1))
        cases.append((allids, accepted(allids), latent, verifiers))
    return tuple(np.stack(items) for items in zip(*cases))


def kernels(z, data, gains):
    ids = data[0]
    features = z[ids]
    return (features[:, :17]*gains) @ features.transpose(0,2,1)


def pick(policy, score, old, opening):
    if policy == "random":
        return np.zeros(len(old), int)
    rank = opening if policy == "opening" else score
    return np.where(old, -np.inf, rank).argmax(1)


def evaluate(data, kernel, step, bias, policy, source):
    ids, old, latent, verifiers = data
    n = len(ids)
    ix = np.arange(n)
    qold = old[:,17:]
    qclass = np.repeat(np.arange(4), 16)
    result = np.zeros((3,n,15))
    # Opening is genuinely positive in the change world, fabricated in the stable world.
    opening = kernel[:,0]
    initial = -bias + step/(1+np.exp(-bias))*opening
    for world, boundary in (("change",1), ("stable",0)):
        true = latent <= boundary
        qtrue = qclass <= boundary
        for flips in (itertools.product((0,1), repeat=2) if source == "uniform" else [(0,0)]):
            score = initial.copy()
            confirms = np.zeros(n, int)
            errors = np.zeros(n)
            menus = np.zeros(n)
            mass = np.ones(n)
            for t,flip in enumerate(flips):
                start = 8*t
                menu = slice(1+start,9+start)
                j = start + pick(policy, score[:,menu], old[:,menu], opening[:,menu])
                checked = true[ix,j]
                source_ids = verifiers[:,t if source == "second" else 0]
                answers = source_ids <= boundary
                rate = (answers[:, start:start+8] != true[:, start:start+8]).mean(1)
                if source == "uniform":
                    mass *= rate if flip else 1-rate
                    seen = checked ^ flip
                elif source == "perfect":
                    seen = checked
                    rate = np.zeros(n)
                else:
                    seen = answers[ix,j]
                menus += rate/2
                errors += (seen != checked)/2
                evidence = seen & ~old[ix,j+1]
                confirms += evidence
                predicted = 1/(1+np.exp(-np.clip(score[ix,j+1], -60,60)))
                score += step*(evidence-predicted)[:,None]*kernel[ix,j+1]
            for g,gate in enumerate(GATES):
                release = confirms >= (2,1,0)[g]
                out = qold | (release[:,None] & (score[:,17:] >= 0))
                if world == "change":
                    missing = ~qold[:,16:32]
                    vals = {0:(out==qtrue).mean(1), 1:out[:,16:32].mean(1),
                            2:(out[:,16:32]&missing).sum(1), 3:missing.sum(1),
                            7:release, 9:errors, 11:menus, 13:(qold==qtrue).mean(1)}
                else:
                    ok = qold==qtrue
                    vals = {4:((out!=qtrue)&ok).sum(1), 5:ok.sum(1), 6:ok.mean(1),
                            8:release, 10:errors, 12:menus, 14:(out==qtrue).mean(1)}
                for k,val in vals.items():
                    result[g,:,k] += mass*val
    return result


def rates(a):
    return np.stack([a[...,0],a[...,1],a[...,2]/a[...,3],a[...,4]/a[...,5],
                     *[a[...,i] for i in (6,7,8,9,10,11,12,13,14)]], axis=-1)


def summarize(raw, eval_domains):
    rng = np.random.default_rng(59943)
    weights = [rng.multinomial(128, np.full(128,1/128),size=2048)/128 for _ in eval_domains]
    draws, cells = {}, {}
    for key, a in raw.items():
        point = rates(a.mean((0,1)))
        boot = rates(sum(w@v for w,v in zip(weights,a))/len(weights))
        draws[key] = point,boot
        cells[key] = dict(interval=np.column_stack([point, np.quantile(boot,[.025,.975],axis=0).T]).tolist(),
                          domains=rates(a.mean(1)).tolist())
    contrasts = {}
    for source in SOURCES:
        ref = f"scalar/addition/{source}/both"
        for other in (f"scalar/random/{source}/both",f"scalar/opening/{source}/both",
                      f"diagonal/addition/{source}/both", "scalar/addition/uniform/both",
                      "scalar/addition/second/both"):
            v,d = (x-y for x,y in zip(draws[ref],draws[other]))
            contrasts[ref+" minus "+other] = np.column_stack([v,np.quantile(d,[.025,.975],axis=0).T]).tolist()
    primary = raw["scalar/addition/same/both"]
    n = primary.shape[0]*primary.shape[1]
    zero_event = None
    if not primary[:,:,4].any():
        # Bonferroni: upper chance of any harmful episode, lower baseline accuracy.
        # Fixed strata have equal count; AM-GM bounds the all-zero likelihood.
        upper = 1-.025**(1/n)
        lower = primary[:,:,5].mean()/64 - np.sqrt(np.log(1/.025)/(2*n))
        zero_event = dict(episodes=n, upper_harm=upper/lower)
    return dict(metrics=NAMES,cells=cells,contrasts=contrasts,zero_event_bound=zero_event)


def main():
    with np.load(CACHE/"embedded.npz") as f:
        x, split, label = (f[k] for k in ("features","split","label"))
    assert hashlib.sha256(x.tobytes()).hexdigest() == "baa29a91f5588105d8b58e0617c1f9e2a243c064860dd70aa154520e36c19ffa"
    domains = json.loads((CACHE/"domains.json").read_text())
    practice, test = sorted(domains)[::2], sorted(domains)[1::2]
    names = [i for d in practice for i in domains[d]]
    z = unit(x-x[(split=="train") & np.isin(label,names)].mean(0))
    pools = {(s,i):np.flatnonzero((split==s)&(label==i)) for s in ("train","test") for i in set(label)}
    gains = {"scalar":1., "diagonal":metric(z,pools,[domains[d] for d in practice])}
    dev = tuple(np.concatenate(a) for a in zip(*[assignments(z,pools,domains[d],24,7501+j) for j,d in enumerate(practice)]))
    settings = {}
    for writer, gain in gains.items():
        kernel = kernels(z,dev,gain)
        grid = []
        for step,bias in itertools.product((2.,6.,16.),(.75,1.5,3.)):
            a = evaluate(dev,kernel,step,bias,"addition","same")[0]
            # The released head can also repair initial stable false negatives.
            # Recover absolute stable accuracy from its predictions below in the stored counts.
            utility = (a[:,0] + a[:,14]).mean()/2
            grid.append((utility,step,bias))
        _,step,bias = max(grid, key=lambda item:item[0])
        settings[writer] = dict(step=step,bias=bias,grid=grid)
    raw = {}
    for j,d in enumerate(test):
        data = assignments(z,pools,domains[d],128,88501+j)
        for writer,gain in gains.items():
            kernel = kernels(z,data,gain)
            for policy,src in itertools.product(POLICIES,SOURCES):
                a = evaluate(data,kernel,settings[writer]["step"],settings[writer]["bias"],policy,src)
                for g,gate in enumerate(GATES):
                    raw.setdefault(f"{writer}/{policy}/{src}/{gate}",[]).append(a[g])
    raw = {k:np.stack(v) for k,v in raw.items()}
    np.savez_compressed(ROOT/"provisional-trials.npz",**raw)
    report = summarize(raw,test)
    report.update(practice=practice, test=test, settings=settings,
                  step_metric=gains["diagonal"].tolist())
    (ROOT/"provisional.json").write_text(json.dumps(report,indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
