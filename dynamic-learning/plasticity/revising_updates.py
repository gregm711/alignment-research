"""Paired test of prospective and retrospective uses of exactly the same evidence."""
import itertools
import json

import recognizing_change as change
from recognizing_change import pilot, np, torch, ROOT
from analyse import estimate

SEEDS = range(341, 349)
GROUPS = (*change.GROUPS, "mixed-conditional")


def branches(table, diagonals, data, rate):
    y, order = data
    rows = np.arange(len(y))
    w = np.zeros((len(y), 3, 12))
    loss = np.zeros((len(y), 3))
    stored, scores = [w.copy()], [loss.copy()]
    for step in range(16):
        ids = order[:, step]
        x, label = table[ids, None], y[rows, ids, None].astype(float)
        z = (w * x).sum(2)
        loss += np.logaddexp(0, (1 - 2 * label) * z)
        w -= rate * diagonals * pilot.gradient(w, x, label)
        stored.append(w.copy())
        scores.append(loss.copy())
    return np.stack(stored), np.stack(scores)


def mixture(scores, temp):
    centered = scores - scores.min(axis=2, keepdims=True)
    exp = np.exp(-temp * centered)
    return exp / exp.sum(2, keepdims=True)


def curves(table, data, snapshots):
    labels, order = data
    idx = order[:, 18:]
    y = np.take_along_axis(labels, idx, axis=1)
    x = table[idx]
    return np.column_stack([(((x * w[:, None]).sum(2) >= 0) == y).mean(1) for w in snapshots])


def prospective(table, ds, diagonals, posterior, rate):
    y, order = ds
    rows = np.arange(len(y))
    w = np.zeros((len(y), 12))
    snapshots = [w.copy()]
    for step in range(16):
        i = order[:, step]
        w -= rate * (posterior[step] @ diagonals) * pilot.gradient(w, table[i], y[rows,i])
        if step + 1 in change.BUDGETS:
            snapshots.append(w.copy())
    return snapshots


def ridge(table, ds, penalty):
    labels, order = ds
    x = table[order[:, :16]]
    y = 2.*np.take_along_axis(labels, order[:, :16], axis=1)-1
    snapshots = [np.zeros((len(y),12))]
    for n in change.BUDGETS[1:]:
        a = x[:,:n]
        gram = a.transpose(0,2,1) @ a + penalty*np.eye(12)
        snapshots.append(np.linalg.solve(gram, a.transpose(0,2,1) @ y[:,:n,None])[:,:,0])
    return snapshots


def evaluate(table, ds, diagonals, cache, name, rate, parameter, donors=None):
    if name == "ridge":
        return curves(table, ds, ridge(table, ds, parameter)), None
    weights, scores = cache[rate]
    p = mixture(scores, parameter)
    if donors is not None:
        p = p[:,donors]
    if name == "prospective":
        snapshots = prospective(table, ds, diagonals, p, rate)
    elif name in ("practiced","flat","reciprocal"):
        branch = ("practiced","flat","reciprocal").index(name)
        snapshots = weights[list(change.BUDGETS),:,branch]
    else:
        snapshots = (p[list(change.BUDGETS),:,:,None] * weights[list(change.BUDGETS)]).sum(2)
    return curves(table, ds, snapshots), (p[-1], weights[-1], snapshots[-1])


def advice(table, data, state, diag, rate, reliability):
    p, experts, w = state
    labels, order = data
    rows = np.arange(len(labels))
    x, probe = table[order[:,16]], table[order[:,17]]
    y = labels[rows,order[:,16]].astype(float)
    pre = ((probe*w).sum(1) >= 0) == y
    strength = np.maximum(0., 2*reliability-1)
    result = {"pre":pre}
    for name, scales in (("calibrated", strength),("common", np.repeat(strength.mean(),2))):
        by_source = []
        for scale in scales:
            outcomes = []
            for label in (1-y,y):
                g = pilot.gradient(experts,x[:,None],label[:,None])
                updated = (p[:,:,None]*(experts-rate*scale*diag*g)).sum(1)
                outcomes.append(((probe*updated).sum(1)>=0) == y)
            by_source.append(np.column_stack(outcomes))
        result[name] = np.stack(by_source,1)
    return result


def run():
    raw, choices = {}, []
    for seed in SEEDS:
        rng = np.random.default_rng(seed+11500)
        table, _ = pilot.feature_table(seed)
        old, new = rng.permutation(pilot.TRIPLES), rng.permutation(pilot.SHIFT)
        mixed = np.array([ids for ids in itertools.combinations(range(12),3)
                          if 0 < np.isin(ids,pilot.COMMON).sum() < 3])
        gate = pilot.life(table,seed,"gate",pilot.BASE_RATE,old[:12])[100][1]
        d = np.vstack([gate,np.ones(12),1/gate])
        d /= np.sqrt((d*d).mean(1,keepdims=True))
        dev = change.tasks(rng,[old[12:16],new[:10]],64)
        test = change.tasks(rng,[old[16:],new[10:],old[16:],new[10:],mixed],160)
        standard = ("practiced","flat","reciprocal","equal","ridge")
        candidate = {name:[(r,0.) for r in change.RATES] for name in standard[:-1]}
        candidate["ridge"] = [(0.,p) for p in (.01,.3,3.,12.)]
        for name in ("prospective","retroactive"):
            candidate[name] = list(itertools.product(change.RATES,(.25,1.,4.)))
        dev_cache = {r:branches(table,d,dev,r) for r in change.RATES}
        configs, scores = {}, {}
        for name, grid in candidate.items():
            vals = [evaluate(table,dev,d,dev_cache,name,*cfg)[0][:,1:].mean() for cfg in grid]
            best = int(np.argmax(vals))
            configs[name], scores[name] = grid[best], float(vals[best])
        winner = max(standard,key=scores.get)
        choices.append({"seed":seed,"winner":winner,"config":configs,"dev":scores})
        test_cache = {r:branches(table,d,test,r) for r in change.RATES}
        donors = change.donor_ids(rng,len(test[0]))
        for key in (*configs,"matched-retro","shuffled","winner"):
            name = {"matched-retro":"retroactive","shuffled":"retroactive","winner":winner}.get(key,key)
            cfg = configs["prospective"] if key == "matched-retro" else configs[name]
            curve, state = evaluate(table,test,d,test_cache,name,*cfg,donors if key=="shuffled" else None)
            raw[f"{seed}-{key}"] = curve
            if key == "retroactive":
                ids = rng.permutation([.9,.55])
                reliability = ((rng.binomial(40,ids)+1)/42)[np.argsort(-ids)]
                choices[-1]["reliability"] = reliability.tolist()
                for mode, value in advice(table,test,state,d,cfg[0],reliability).items():
                    raw[f"{seed}-advice-{mode}"] = value
                raw[f"{seed}-posterior"],raw[f"{seed}-weights"] = state[0],state[2]
    np.savez_compressed(ROOT/"revision-trials.npz",**raw)
    (ROOT/"revision-settings.json").write_text(json.dumps(choices,indent=2)+"\n")
    summarise(raw)


def summarise(raw):
    report = {}
    for j, group in enumerate(GROUPS):
        def get(key):
            return np.stack([raw[f"{s}-{key}"][j*160:(j+1)*160] for s in SEEDS])
        names = ("retroactive","matched-retro","prospective","shuffled","winner","ridge","flat",
                 "practiced","reciprocal","equal")
        score = {name:get(name)[:,:,1:].mean((1,2)) for name in names}
        result = {"accuracy":{k:float(v.mean()) for k,v in score.items()},
                  "curves":{k:get(k).mean((0,1)).tolist() for k in names},
                  "retro_minus":{k:estimate(score["retroactive"]-v) for k,v in score.items()},
                  "matched_retro_minus_prospective":estimate(score["matched-retro"]-score["prospective"]),
                  "posterior":get("posterior").mean((0,1)).tolist(),"feedback":{}}
        pre = get("advice-pre")
        for jlabel,endpoint in enumerate(("harm","repair")):
            eligible = pre if endpoint=="harm" else ~pre
            counts = {}
            for mode in ("calibrated","common"):
                events = (get("advice-"+mode)[:,:,:,jlabel] != pre[:,:,None]) & eligible[:,:,None]
                counts[mode] = events.sum(1)
            rates = (counts["calibrated"]-counts["common"])/eligible.sum(1)[:,None]
            freq = [.1,.45] if endpoint=="harm" else [.9,.55]
            result["feedback"][endpoint] = {
                "stable":estimate(np.average(rates,axis=1,weights=freq)),
                "reversed":estimate(np.average(rates,axis=1,weights=freq[::-1])),
                "eligible":int(eligible.sum()),
                "counts":{k:v.sum(0).tolist() for k,v in counts.items()}}
        report[group]=result
        print(group, {k:round(100*v,2) for k,v in result["accuracy"].items()})
    (ROOT/"revision-summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run()
