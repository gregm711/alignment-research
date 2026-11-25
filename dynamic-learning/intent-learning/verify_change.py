"""Select a verified request before deciding how broadly to change the routing weights."""
import json

import policy_change as p
from policy_change import CACHE, MASK, OTHER, PAIRS, ROOT, np

PRIORS = ("independent", "coupled", "mixture")
POLICIES = ("random", "uncertainty", "expected-error", "recheck")
BUDGETS = (1, 2, 4)


def stream(rng, n, pairs, count, probes=False):
    old = rng.random((n, 6)).argsort(1) < 3
    trigger = rng.integers(6, size=n)
    true = old.copy()
    noise = rng.random(n) < .1
    menu = rng.random((n, 4, 6)).argsort(-1)
    if probes:
        true[np.arange(n), trigger] ^= True
        true[np.arange(n), trigger ^ 1] ^= True
        # Branches face identical reports and fresh candidate texts.
        return (np.tile(old, (2,1)), np.vstack([true, old]), np.tile(trigger, 2),
                np.tile(~old[np.arange(n),trigger], 2), np.tile(menu, (2,1,1)))
    indices = rng.random((n, 3)).argsort(1)[:,:count]
    true[np.arange(n)[:,None,None], pairs[indices]] ^= True
    return old, true, trigger, true[np.arange(n),trigger] ^ noise, menu


def risks(post, possible, yes):
    forecast = post @ yes.transpose(0,2,1)
    joint = (post[:,:,None,:] * yes[:,None,:,:]) @ possible[:,None,:,:]
    current = (post @ possible)[:,:,None,:]
    negative = current - joint
    risk = (np.minimum(joint, forecast[:,:,:,None]-joint)
            + np.minimum(negative, 1-forecast[:,:,:,None]-negative)).sum(-1)
    return forecast, risk


def evaluate(priors, support, query, batch, probes):
    old, truth, trigger, report, menu = batch
    n = len(old)
    row = np.arange(n)
    possible = (old[:,None,:] != MASK).astype(float)
    first = support[trigger,0]
    yes = .1 + .8 * np.einsum("nd,nkd->nk", first, possible)
    likelihood = np.where(report[:,None], yes, 1-yes)
    prior = priors[[0,1,4]]
    component = .5 * prior[1] / prior[2]
    post = p.normalize(prior[None,:,:] * likelihood[:,None,:]).repeat(len(POLICIES), axis=1)
    scores, confidence, pick = [], [], None
    modes = np.arange(post.shape[1])
    mix = slice(2*len(POLICIES), None)
    confidence.append((post[:,mix] @ component).mean(0))
    if probes:
        q = query[trigger ^ 1]
        old_label = old[row, trigger ^ 1]
        eligible = (np.einsum("nqd,nd->nq", q, old) >= .5) == old_label[:,None]
    for t in range(4):
        classes = menu[:,t]
        x = support[classes,t+1]
        # Verified probability for every candidate under every possible policy.
        yes = x @ possible.transpose(0,2,1)
        forecast, risk = risks(post, possible, yes)
        choice = np.zeros((n,len(modes)), dtype=int)
        choice[:,1::4] = np.abs(forecast[:,1::4]-.5).argmin(-1)
        choice[:,2::4] = risk[:,2::4].argmin(-1)
        choice[:,3::4] = ((x-first[:,None,:])**2).sum(-1).argmin(-1)[:,None]
        selected = classes[row[:,None],choice]
        if pick is None:
            pick = selected.copy()
        obs = truth[row[:,None],selected]
        evidence = yes[row[:,None], choice]
        post = p.normalize(post*np.where(obs[:,:,None],evidence,1-evidence))
        if t+1 not in BUDGETS:
            continue
        heads = post @ possible
        confidence.append((post[:,mix] @ component).mean(0))
        if probes:
            changed = ((heads @ q.transpose(0,2,1) >= .5) != old_label[:,None,None]) & eligible[:,None,:]
            half = n//2
            scores.append(np.stack([a.sum((0,2)) / b.sum() for a,b in
                                    zip(np.split(changed,2),np.split(eligible,2))]))
        else:
            pred = np.einsum("nmd,cqd->nmcq", heads, query) >= .5
            scores.append((pred == truth[:,None,:,None]).mean(axis=(0,2,3)))
    if probes:
        assert np.array_equal(pick[:n//2],pick[n//2:])
    assert np.allclose(post.sum(-1),1)
    return np.stack(scores), np.stack(confidence), pick


def main():
    with np.load(CACHE/"embedded.npz") as data:
        features, split, labels = (data[k] for k in ("features","split","label"))
    domains = json.loads((CACHE/"domains.json").read_text())
    accuracy, trust, feedback, records = [], [], [], []
    for i,seed in enumerate(range(951,971)):
        rng = np.random.default_rng(seed)
        domain = list(domains)[i%10]
        names = rng.choice(domains[domain], 6, replace=False)
        support, query, _, temp = p.views(features, split, labels, names, rng)
        history = np.zeros((48,6))
        times = np.flatnonzero(rng.random(48)>=.4)
        history[times[:,None],PAIRS[rng.integers(3,size=len(times))]] = 1
        priors,_ = p.make_priors(history,PAIRS)
        by_regime, confidence, first_choices = [], [], []
        for pairs in (PAIRS,OTHER):
            by_count, confidences = [], []
            for count in range(3):
                batch = stream(rng,48,pairs,count)
                scores, levels, picks = evaluate(priors,support,query,batch,False)
                by_count.append(scores)
                confidences.append(levels)
                first_choices.append(picks.tolist())
            by_regime.append(by_count)
            confidence.append(confidences)
        batch = stream(rng,120,PAIRS,1,True)
        f,_,picks = evaluate(priors,support,query,batch,True)
        accuracy.append(by_regime)
        trust.append(confidence)
        feedback.append(f)
        records.append(dict(seed=seed,domain=domain,intents=names.tolist(),temperature=temp,
                            peer_check_fraction=(picks==(batch[2]^1)[:,None]).mean(0).tolist(),
                            probe_trigger=batch[2][:120].tolist(), probe_first=picks[:120].tolist(),
                            first_choices=first_choices))
    acc = np.array(accuracy)
    fb = np.array(feedback)
    conf = np.array(trust)
    np.savez_compressed(ROOT/"verification-trials.npz", accuracy=acc, feedback=fb,
                        confidence=conf, first_choices=[r.pop("first_choices") for r in records],
                        probe_first=[r.pop("probe_first") for r in records])
    result = dict(priors=PRIORS,policies=POLICIES,budgets=BUDGETS,
                  accuracy=acc.mean(0).tolist(),feedback=fb.mean(0).tolist(),
                  confidence=conf.mean(0).tolist(),records=records,contrasts={})
    tests = result["contrasts"]
    for j,name in enumerate(("familiar","changed")):
        for count in range(3):
            vals = acc[:,j,count].mean(1)
            for mode,label in ((2,"independent-active"),(8,"mixture-random"),(9,"mixture-uncertainty"),
                               (11,"mixture-recheck"),(6,"coupled-active")):
                tests[f"{name}-{count}-minus-{label}"] = p.interval(vals[:,10]-vals[:,mode])
    for j,n in enumerate(BUDGETS):
        for k,label in enumerate(("repair","harm")):
            tests[f"{n}-{label}"] = p.interval(fb[:,j,k,10]-fb[:,j,k,2])
    for count in range(3):
        tests[f"familiar-minus-changed-confidence-{count}"] = p.interval(conf[:,0,count,-1,2]-conf[:,1,count,-1,2])
    (ROOT/"verification.json").write_text(json.dumps(result,indent=1)+"\n")


if __name__ == "__main__":
    main()
