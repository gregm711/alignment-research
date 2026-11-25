"""Consolidate two representation geometries and export single-projection learners."""
import json

import unannounced as u
from unannounced import e, joint, recovery, order, np, torch, nn, F, ROOT, X, paired

SEEDS = tuple(range(751, 767))


def design():
    entries = [(first, cuts) for first in ("count", "glob")
               for cuts in ((60,), (180,), (60,180), (90,150))] * 2
    return [entries[i] for i in np.random.default_rng(7519).permutation(len(entries))]


def project(h, state):
    return torch.tanh(F.linear(h, state["1.weight"], state["1.bias"]))


def consolidate(h, stored, live, train, batches, historical):
    layer = nn.Linear(stored["1.weight"].shape[1], stored["1.weight"].shape[0])
    inputs = h[train.tolist()].detach()
    with torch.no_grad():
        layer.weight.copy_((stored["1.weight"] + live["1.weight"]) / 2)
        layer.bias.copy_((stored["1.bias"] + live["1.bias"]) / 2)
        a = F.normalize(project(inputs, stored))
        b = F.normalize(project(inputs, live))
        target = (a @ a.T + b @ b.T)/2 if historical else b @ b.T
    optimizer = torch.optim.Adam(layer.parameters(), lr=.003)
    for idx in batches:
        sample = torch.tanh(layer(inputs[idx]))
        z = F.normalize(sample)
        loss = (z @ z.T - target[idx][:,idx]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"1.weight":layer.weight.detach().clone(),"1.bias":layer.bias.detach().clone()}


def matched_advice(student, live, data, records):
    y, _ = data
    mass = e.score(np.where(y,1.,-1.),data)
    a = (student[0][:,-1]>=0)==y
    b = (live[0][:,-1]>=0)==y
    shared = [mass*a*b,mass*(~a)*(~b)]
    pair = student, live
    n = len(y)
    s = joint.responses(pair,np.ones(n),data,shared)
    c = joint.responses(pair,np.zeros(n),data,shared)
    return {env:(joint.policy_rates(s,records[f"{env}-8-recent"]) -
                 joint.policy_rates(c,records[f"{env}-8-recent"])).tolist()
            for env in ("stable","reversed")}


def normalized(z):
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-9)


def export_readout(export, seed, group, z, data, penalty, all_pred):
    labels, seq = data
    support = seq[0,:16]
    features = normalized(z)
    sx = features[support]
    dual = np.linalg.solve(sx @ sx.T + .2 + penalty*np.eye(16), 2.*labels[0,support]-1)
    w, bias = sx.T @ dual, .2*dual.sum()
    assert np.allclose(features @ w + bias, all_pred[0,-1], atol=1e-8)
    prefix = f"{seed}/readout/{group}/"
    export[prefix+"weight"], export[prefix+"bias"] = w, bias


def run(seeds=SEEDS, prefix="consolidation"):
    _, labels = recovery.labels_with_extra()
    raw, rows, export, checks = {}, [], {}, {}
    for seed,(first,cuts) in zip(seeds,design()):
        model, library, split, tick, log, seen = u.practice(seed, first, cuts, labels)
        dev, test = [np.flatnonzero(split==k) for k in (2,3)]
        train = np.flatnonzero(seen)
        assert set(train).isdisjoint(test) and set(train).isdisjoint(dev)
        rng = np.random.default_rng(seed+20000)
        with torch.no_grad():
            h = u.hidden(model,torch.tensor(X.tolist())).detach()
        last, old = library[240], library[tick]
        mean = {key:(v+old[key])/2 for key,v in last.items()}
        batches = [rng.choice(len(train),48,replace=False).tolist() for _ in range(120)]
        trained = consolidate(h,old,last,train,batches,True)
        sham = consolidate(h,old,last,train,batches,False)
        states = {"student":trained,"current":last,"mean":mean,"self":sham,"old":old}
        z = {k:np.array(project(h,v).tolist()) for k,v in states.items()}
        pos = np.eye(4)[X]
        z["hand"] = np.column_stack([pos.reshape(len(X),-1),pos.sum(1)])
        def kernels(ids):
            result = {k:e.kernel(v[ids]) for k,v in z.items()}
            result["teacher"] = (result.pop("old")+result["current"])/2
            return result
        devK,testK = kernels(dev),kernels(test)
        valid = [e.episodes(rng,labels[g],dev,n) for g,n in (("dev",1),("count-dev",8))]
        rates = {}
        for key,K in devK.items():
            vals = [np.mean([e.curve(e.fit(K,d,r)[0],d).mean() for d in valid]) for r in e.PENALTIES]
            rates[key]=e.PENALTIES[int(np.argmax(vals))]
        saved = {k:v for k,v in model.state_dict().items() if not k.startswith(("head.","project."))}
        saved.update({"project."+k:v for k,v in trained.items()})
        for key,v in saved.items():
            export[f"{seed}/encoder/{key}"]=np.asarray(v.tolist(),dtype=np.float32)
        checks[f"{seed}-test"]=test
        source=order.source_histories(np.random.default_rng(seed+54100),
                                       np.vstack([labels["practice"],labels["count-practice"]]),dev)
        kept=u.independent_programs(labels,test)
        feedback={}
        for group in u.GROUPS:
            ds=e.episodes(rng,labels[group][kept[group]],test,2 if group in ("glob","xor") else 12)
            fits={k:e.fit(K,ds,rates[k],True) for k,K in testK.items()}
            for k,fit in fits.items():
                raw[f"{seed}-{group}-{k}"]=e.curve(fit[0],ds)
            if group in ("count","glob"):
                export_readout(export,seed,group,z["student"][test],ds,rates["student"],fits["student"][0])
                checks[f"{seed}-{group}"]=fits["student"][0][0,-1]
                feedback[group]=matched_advice(fits["student"],fits["current"],ds,source)
        recent = first if len(cuts)==2 else ("count" if first=="glob" else "glob")
        # For an unrestricted rank-32 representation, spectral tail is a lower bound on RMS error.
        eig = np.linalg.eigvalsh(testK["teacher"]-.2)
        rank_floor = np.linalg.norm(eig[:-32])/len(test)
        errors={k:float(np.sqrt(np.mean((K-testK["teacher"])**2))) for k,K in testK.items()}
        rows.append({"seed":seed,"initial_family":first,"recent":recent,"cuts":cuts,"archive":tick,
                     "decisions":log,"train_strings":len(train),"rates":rates,"feedback":feedback,
                     "rmse":errors,"rank_floor":float(rank_floor),"programs":kept})
    np.savez_compressed(ROOT/(prefix+"-trials.npz"),**raw,**checks)
    np.savez_compressed(ROOT/(prefix+"-models.npz"),**export)
    (ROOT/(prefix+"-settings.json")).write_text(json.dumps(rows,indent=2)+"\n")
    return rows,raw


def verify_export(prefix="consolidation"):
    with np.load(ROOT/(prefix+"-models.npz")) as models, np.load(ROOT/(prefix+"-trials.npz")) as trials:
        seeds=sorted({int(k.split("/")[0]) for k in models.files})
        for seed in seeds:
            p=f"{seed}/encoder/"
            model=e.StringEncoder()
            state={k[len(p):]:torch.tensor(models[k].tolist()) for k in models.files if k.startswith(p)}
            missing,unexpected=model.load_state_dict(state,strict=False)
            assert set(missing)=={"head.weight","head.bias"} and not unexpected
            ids=trials[f"{seed}-test"]
            with torch.no_grad():
                z=normalized(np.array(model(torch.tensor(X[ids].tolist())).tolist()))
            for g in ("count","glob"):
                p=f"{seed}/readout/{g}/"
                logits=z @ models[p+"weight"] + models[p+"bias"]
                assert np.allclose(logits,trials[f"{seed}-{g}"],atol=1e-6),seed
                assert np.array_equal(logits>=0,trials[f"{seed}-{g}"]>=0)
    return len(seeds)*2


def summarise(rows,raw,prefix="consolidation"):
    report={}
    for group in ("absent","recent","xor","equal"):
        scores={}; feedback={}
        for row in rows:
            recent=row["recent"]
            absent="count" if recent=="glob" else "glob"
            g={"absent":absent,"recent":recent}.get(group,group)
            for k in row["rates"]:
                scores.setdefault(k,[]).append(raw[f"{row['seed']}-{g}-{k}"].mean(0))
            if g in row["feedback"]:
                for env,(harm,repair) in row["feedback"][g].items():
                    feedback.setdefault(env+"-harm",[]).append(harm)
                    feedback.setdefault(env+"-repair",[]).append(repair)
        s={k:np.stack(v) for k,v in scores.items()}
        report[group]={"curves":{k:v.mean(0).tolist() for k,v in s.items()},
                       "contrasts":{k:paired((s["student"]-v).mean(1)) for k,v in s.items()},
                       "feedback":{k:paired(v) for k,v in feedback.items()}}
        for switches in (1,2):
            mask=[len(r["cuts"])==switches for r in rows]
            report[group][f"switches-{switches}"]=paired((s["student"]-s["current"]).mean(1)[mask])
    (ROOT/(prefix+"-summary.json")).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summarise(*run())
    print("Cold-reloaded task readouts:",verify_export())
