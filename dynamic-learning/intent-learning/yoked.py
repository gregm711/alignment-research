"""Cross the selector of a check with the learner that interprets that same check."""
import json

import verify_change as v
from policy_change import CACHE, MASK, OTHER, PAIRS, ROOT, np
import policy_change as p

CONDITIONS = ("accurate", "noisy-calibrated", "noisy-overtrusted")


def measure(rng, priors, support, query, batch, probe=False):
    old, truth, trigger, report, menu = batch
    n = len(old)
    row = np.arange(n)
    possible = (old[:,None,:] != MASK).astype(float)
    x = support[trigger,0]
    yes = .1 + .8 * np.einsum("nd,nkd->nk", x, possible)
    post = p.normalize(priors[None,[0,4],:] * np.where(report[:,None],yes,1-yes)[:,None,:])
    classes = menu[:,0]
    yes = support[classes,1] @ possible.transpose(0,2,1)
    _, risk = v.risks(post,possible,yes)
    choices = risk.argmin(-1)
    flipped = rng.random((n//2 if probe else n,6)) < .1
    if probe:
        flipped = np.tile(flipped,(2,1))
        assert np.array_equal(*np.split(choices,2))
        old_label = old[row,trigger^1]
        q = query[trigger^1]
        eligible = ((q @ old[:,:,None]).squeeze(-1) >= .5) == old_label[:,None]
    cells = []
    for j,condition in enumerate(CONDITIONS):
        checked = classes[row[:,None],choices]
        returns = truth[row[:,None],checked].copy()
        if j:
            returns ^= flipped[row[:,None],checked]
        chance = yes[row[:,None],choices]
        if condition == "noisy-calibrated":
            chance = .1 + .8 * chance
        evidence = np.where(returns[:,:,None],chance,1-chance)
        # Index one selects the check. Index two selects the reader of the same answer.
        heads = p.normalize(post[:,None,:,:] * evidence[:,:,None,:]) @ possible[:,None,:,:]
        if probe:
            changed = (np.einsum("nsrd,nqd->nsrq",heads,q) >= .5) != old_label[:,None,None,None]
            eligible_changes = changed & eligible[:,None,None,:]
            cells.append([a.sum((0,3))/b.sum() for a,b in zip(np.split(eligible_changes,2),np.split(eligible,2))])
        else:
            pred = np.einsum("nsrd,cqd->nsrcq",heads,query) >= .5
            cells.append((pred == truth[:,None,None,:,None]).mean((0,3,4)))
    return np.array(cells)


def main():
    with np.load(CACHE/"embedded.npz") as f:
        features,split,labels = (f[k] for k in ("features","split","label"))
    domains = json.loads((CACHE/"domains.json").read_text())
    all_cells = []
    for i,seed in enumerate(range(1001,1021)):
        rng = np.random.default_rng(seed)
        names = rng.choice(domains[list(domains)[i%10]],6,replace=False)
        support,query,_,_ = p.views(features,split,labels,names,rng)
        history = np.zeros((48,6))
        changed = np.flatnonzero(rng.random(48)>=.4)
        history[changed[:,None],PAIRS[rng.integers(3,size=len(changed))]] = 1
        priors,_ = p.make_priors(history,PAIRS)
        capability = [measure(rng,priors,support,query,v.stream(rng,96,pairs,2)) for pairs in (PAIRS,OTHER)]
        feedback = measure(rng,priors,support,query,v.stream(rng,120,PAIRS,1,True),True)
        all_cells.append(np.concatenate([np.stack(capability,axis=1),feedback],axis=1))
    cells = np.array(all_cells)
    report = dict(conditions=CONDITIONS, outcomes=("familiar","changed","repair","harm"),
                  selectors=("independent","mixture"),readers=("independent","mixture"),
                  seeds=list(range(1001,1021)),means=cells.mean(0).tolist(),contrasts={})
    for j,name in enumerate(CONDITIONS):
        for k,metric in enumerate(report["outcomes"]):
            a = cells[:,j,k]
            for label,diff in (("reader-at-independent-choice",a[:,0,1]-a[:,0,0]),
                               ("reader-at-mixture-choice",a[:,1,1]-a[:,1,0]),
                               ("choice-with-independent-reader",a[:,1,0]-a[:,0,0]),
                               ("choice-with-mixture-reader",a[:,1,1]-a[:,0,1]),
                               ("end-to-end",a[:,1,1]-a[:,0,0])):
                report["contrasts"][f"{name}-{metric}-{label}"] = p.interval(diff)
    np.savez_compressed(ROOT/"yoked-trials.npz",cells=cells)
    (ROOT/"yoked.json").write_text(json.dumps(report,indent=1)+"\n")


if __name__ == "__main__":
    main()
