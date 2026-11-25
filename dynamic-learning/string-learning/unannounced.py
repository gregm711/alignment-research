"""One archived projection selected using verified, previously observed practice episodes."""
import copy
import itertools
import json

import joint
import order
import recovery
from order import e, np, torch, nn, F, ROOT, X
from analyse import paired

SEEDS = tuple(range(701, 717))
GROUPS = ("glob", "count", "xor", "equal")
TIMES = tuple(range(20, 240, 10))


def schedule():
    entries = list(itertools.product(("count", "glob"), (50, 90, 150, 190))) * 2
    indices = np.random.default_rng(933).permutation(len(entries))
    return [entries[i] for i in indices]


def example_scores(features, targets):
    """Eight balanced supports and 16 held-out practice examples per reservoir entry."""
    z = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-9)
    support = z[:, :8]
    gram = support @ support.transpose(0, 2, 1) + .2 + .3 * np.eye(8)
    dual = np.linalg.solve(gram, (2. * targets[:, :8, None] - 1))
    query = z[:, 8:] @ support.transpose(0, 2, 1) + .2
    return (((query @ dual)[:, :, 0] >= 0) == targets[:, 8:]).mean(1)


def hidden(model, x):
    return model.block(model.symbol(x) + model.position).flatten(1)


def select(model, x, library, time, archive, reservoir):
    ids = np.stack([sample for sample, _ in reservoir])
    labels = np.stack([label for _, label in reservoir])
    with torch.no_grad():
        h = hidden(model, x)
        scores = {}
        for tick in (time, archive, time - 10):
            state = library[tick]
            projected = torch.tanh(F.linear(h, state["1.weight"], state["1.bias"]))
            scores[tick] = example_scores(np.array(projected.tolist())[ids], labels)
    def utility(tick):
        return float(np.maximum(0, scores[tick] - scores[time]).mean())
    incumbent, challenger = utility(archive), utility(time-10)
    selected = time-10 if challenger >= incumbent else archive
    return selected, [time, archive, incumbent, time-10, challenger, selected]


def practice(seed, earlier, boundary, labels):
    rng = np.random.default_rng(seed)
    split = sum(rng.permutation(4)[X[:, i]] for i in range(5)) % 4
    ids = np.flatnonzero(split < 2)
    x = torch.tensor(X.tolist())
    rules = np.vstack([labels["practice"], labels["count-practice"]])
    domains = {"glob": np.arange(20), "count": np.arange(20, len(rules))}
    later = "glob" if earlier == "count" else "count"
    torch.manual_seed(seed)
    model = e.StringEncoder()
    model.head = nn.Linear(32, len(rules))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    reservoir_rng = np.random.default_rng(seed + 31000)
    library, reservoir, log = {}, [], []
    archive = 20
    observed = np.zeros(len(X), dtype=bool)
    cuts = (boundary,) if isinstance(boundary, int) else boundary
    for t in range(1, 241):
        block = sum(t > cut for cut in cuts)
        task = rng.choice(domains[earlier if block % 2 == 0 else later])
        for batch in range(3):
            samples = e.balanced(rng, rules[task], ids, 12).ravel()
            observed[samples] = True
            y = rules[task, samples]
            if batch == 0:
                item = samples, y
            targets = torch.tensor(y.astype(float).tolist())
            logits = model.head(model(x[samples.tolist()]))[:, task]
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(logits, targets).backward()
            optimizer.step()
        if len(reservoir) < 24:
            reservoir.append(item)
        else:
            replace = reservoir_rng.integers(t)
            if replace < len(reservoir):
                reservoir[replace] = item
        if t >= 20 and t % 10 == 0:
            library[t] = copy.deepcopy(model.project.state_dict())
            if t >= 30:
                archive, decision = select(model, x, library, t, archive, reservoir)
                log.append(decision)
    assert archive in TIMES and all(d[-1] < d[0] for d in log)
    return model, library, split, archive, log, observed


def independent_programs(labels, test):
    def identity(rule):
        y = rule[test]
        return min(np.packbits(y).tobytes(), np.packbits(~y).tobytes())
    seen = {identity(rule) for key in ("practice", "count-practice", "dev", "count-dev")
            for rule in labels[key]}
    result = {}
    for group in GROUPS:
        kept = []
        for i, rule in enumerate(labels[group]):
            key = identity(rule)
            positives = int(rule[test].sum())
            # Nine eligible examples plus at least one independent query in each class.
            if key not in seen and min(positives, len(test)-positives) >= 10:
                kept.append(i)
                seen.add(key)
        result[group] = kept
    return result


def compare_advice(pair, old_weight, data, records):
    y, _ = data
    mass = e.score(np.where(y, 1., -1.), data)
    cur = pair[1][0][:, -1]
    adapt = old_weight[:, None]*pair[0][0][:, -1] + (1-old_weight[:, None])*cur
    a, c = (adapt >= 0) == y, (cur >= 0) == y
    eligible = [mass*a*c, mass*(~a)*(~c)]
    aa = joint.responses(pair, old_weight, data, eligible)
    cc = joint.responses(pair, np.zeros_like(old_weight), data, eligible)
    return {env: (joint.policy_rates(aa, records[f"{env}-8-recent"]) -
                  joint.policy_rates(cc, records[f"{env}-8-recent"])).tolist()
            for env in ("stable", "reversed")}


def evaluate(seed, history, labels, times, groups=GROUPS, check_feedback=True):
    model, library, split, _, _, _ = history
    rng = np.random.default_rng(seed + 25000)
    dev, test = [np.flatnonzero(split == k) for k in (2, 3)]
    selected = independent_programs(labels, test)
    validations = [e.episodes(rng, labels[g], dev, n) for g,n in (("dev",1), ("count-dev",8))]
    sets = {g:e.episodes(rng, labels[g][selected[g]], test, 2 if g in ("glob","xor") else 12)
            for g in GROUPS}
    record = order.source_histories(np.random.default_rng(seed+54000),
                                     np.vstack([labels["practice"],labels["count-practice"]]), dev)
    pos = np.eye(4)[X]
    hand = np.column_stack([pos.reshape(len(X), -1), pos.sum(1)])
    with torch.no_grad():
        h = hidden(model, torch.tensor(X.tolist()))
        def project(t):
            s = library[t]
            return np.array(torch.tanh(F.linear(h, s["1.weight"], s["1.bias"])).tolist())
        current = project(240)
    def kernels(z, ids):
        c, a = e.kernel(current[ids]), e.kernel(z[ids])
        return {"current":c, "old-projection":a, "pooled":.5*(a+c), "hand":e.kernel(hand[ids])}
    cache = {}
    advice = {}
    for tick in set(times.values()):
        with torch.no_grad():
            z = project(tick)
        settings, _ = recovery.average_validation(kernels(z,dev), validations)
        K = kernels(z,test)
        output = {}
        for g,data in sets.items():
            if g not in groups:
                continue
            fs = {k:e.fit(mat,data,settings[k]) for k,mat in K.items()}
            pair = fs["old-projection"],fs["current"]
            even, even_w = e.mix(*pair, settings["adaptive"])
            # Bayes' rule with 1:3 prior odds for the archived representation.
            w = even_w / (even_w + 3*(1-even_w))
            pred = w[:,:,None]*pair[0][0] + (1-w[:,:,None])*pair[1][0]
            result = {k:e.curve(fs[k][0],data) for k in ("current","pooled","hand")}
            result.update(adaptive=e.curve(pred,data),even=e.curve(even,data),old_weight=w)
            output[g]=result
            if check_feedback and tick == times["selected"] and g in ("glob","count"):
                advice[g] = compare_advice(pair,w[:,-1],data,record)
        cache[tick]=output
    return {k:cache[tick] for k,tick in times.items()}, advice, selected


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, labels=recovery.labels_with_extra()
    design=schedule()
    histories = [practice(seed, earlier, boundary, labels)
                 for seed,(earlier,boundary) in zip(SEEDS,design)]
    rows, trials = [], {}
    for i,(seed,(earlier,boundary),history) in enumerate(zip(SEEDS,design,histories)):
        times = {"selected":history[3], "fixed":120, "boundary":boundary,
                 "random":int(np.random.default_rng(seed+4400).choice(TIMES)),
                 "yoked":histories[(i+1)%len(histories)][3]}
        outputs, feedback, kept = evaluate(seed, history, labels, times)
        rows.append({"seed":seed,"earlier":earlier,"boundary":boundary,"times":times,
                     "log":history[4],"feedback":feedback,"programs":kept})
        for method,result in outputs.items():
            for group,values in result.items():
                keys=values if method=="selected" else ["adaptive"]
                for key in keys:
                    trials[f"{seed}-{method}-{group}-{key}"]=values[key]
    np.savez_compressed(ROOT/"unannounced-trials.npz",**trials)
    (ROOT/"unannounced-settings.json").write_text(json.dumps(rows,indent=2)+"\n")
    summarise(rows,trials)


def summarise(rows,trials):
    output={}
    for group in ("earlier","recent","xor","equal"):
        columns={}; advice={}
        for row in rows:
            earlier=row["earlier"]
            recent="count" if earlier=="glob" else "glob"
            g={"earlier":earlier,"recent":recent}.get(group,group)
            for policy in row["times"]:
                keys=("adaptive","even","current","pooled","hand","old_weight") if policy=="selected" else ("adaptive",)
                for k in keys:
                    col=k if policy=="selected" else policy
                    columns.setdefault(col,[]).append(trials[f"{row['seed']}-{policy}-{g}-{k}"].mean(0))
            if g in row["feedback"]:
                for env,(harm,repair) in row["feedback"][g].items():
                    advice.setdefault(env+"-harm",[]).append(harm)
                    advice.setdefault(env+"-repair",[]).append(repair)
        arrays={k:np.array(v) for k,v in columns.items()}
        output[group]={"curves":{k:v.mean(0).tolist() for k,v in arrays.items()},
                       "contrasts":{k:paired((arrays["adaptive"]-v).mean(1)) for k,v in arrays.items() if k!="old_weight"},
                       "feedback":{k:paired(v) for k,v in advice.items()}}
        for key in ("current","yoked"):
            v=(arrays["adaptive"]-arrays[key]).mean(1)
            for timing,mask in (("early",[r["boundary"]<120 for r in rows]),("late",[r["boundary"]>120 for r in rows])):
                output[group][timing+"-vs-"+key]=paired(v[mask])
    (ROOT/"unannounced-summary.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()
