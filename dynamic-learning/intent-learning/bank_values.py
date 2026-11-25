"""Confirm the full-gradient reference on external, disjoint banking intent quartets."""
import csv
import hashlib
import itertools
from urllib.request import urlopen

from protect_values import Protected, ROOT, json, np, rates, torch, truth, counts
from prepare import CACHE
from value_history import NoOpening

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BLOBS = {"test":"799687a8367359432985b8b13d85a2baf73f92dd","train":"98e2543cf482d0dca7bfb175ebe35d98efad95be"}
METHODS = ("full","ordinary","ordinary_high","matched","donor","ridge","anchored_ridge")


def corpus():
    path = CACHE.parent/"banking77"
    path.mkdir(exist_ok=True)
    pool,seen,omitted = {},set(),{}
    for split,digest in BLOBS.items():
        target = path/f"{split}.csv"
        if not target.exists():
            url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data/{split}.csv"
            with urlopen(url,timeout=30) as f:
                data = f.read()
            assert hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==digest
            target.write_bytes(data)
        data = target.read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==digest
        omitted[split] = 0
        with target.open() as f:
            for row in csv.DictReader(f):
                text = row["text"]; key = " ".join(text.casefold().split())
                if key in seen:
                    omitted[split]+=1
                    continue
                seen.add(key)
                pool.setdefault((split,row["category"]),[]).append(text)
    names = sorted(k[1] for k in pool if k[0]=="train")
    assert len(names)==77 and all(("test",i) in pool for i in names)
    return pool,names,omitted


@torch.no_grad()
def cache_intents(model,names):
    if not hasattr(model,"cached_train"):
        model.cached_train = {}
    texts = [t for name in names for t in model.pool["train",name] if t not in model.cached_train]
    model.reset()
    for j in range(0,len(texts),32):
        batch = texts[j:j+32]
        enc = model.tok.encode_batch(batch)
        ids = torch.tensor([v.ids for v in enc])
        mask = torch.tensor([v.attention_mask for v in enc])
        hidden = model.model(ids,attention_mask=mask).last_hidden_state
        z = torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True))
        model.cached_train.update(zip(batch,z.unbind()))


@torch.no_grad()
def geometry(model,case,anchors):
    z = torch.cat([model.embeddings(case,slice(None)),model.embeddings(anchors,slice(None))])
    return (z@z.T).numpy(),case[-1].numpy()


def ridge(geom,anchored,penalty,magnitude):
    kernel,initial = geom
    n = len(initial)
    q = initial[4:]
    result = []
    for genuine in (True,False):
        target = truth(genuine).bool()
        init = torch.tensor(q>=0) == target
        curve = [counts(init,init)]
        labels = np.array([1, int(not genuine), int(genuine), int(not genuine)])
        for budget in (1,4):
            ids = np.arange(budget)
            errors = magnitude*(2*labels[:budget]-1)-initial[ids]
            if anchored:
                ids = np.r_[ids,np.arange(n,n+12)]
                errors = np.r_[errors,np.zeros(12)]
            dual = np.linalg.solve(kernel[np.ix_(ids,ids)]+penalty*np.eye(len(ids)),errors)
            pred = q+kernel[4:n,ids]@dual
            curve.append(counts(torch.tensor(pred>=0)==target,init))
        result.append(curve)
    return np.array(result)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Protected()
    rng = np.random.default_rng(390609)
    grid = list(itertools.product((.01,.1,1.),(1.,3.,8.)))
    tuning = np.zeros((2,len(grid)))
    for d in sorted(model.domains)[::2]:
        for _ in range(4):
            c,hist = model.episode(d,"val",rng,include_anchors=True)
            geom = geometry(model,c,hist)
            for anchored in (0,1):
                for j,params in enumerate(grid):
                    tuning[anchored,j] += ridge(geom,anchored,*params)[:,1:,-1].mean()/(32*20)
    params = [grid[i] for i in tuning.argmax(1)]
    model.pool,names,duplicates = corpus()
    rng = np.random.default_rng(390611)
    perm = rng.permutation(names)
    quartets = perm[:76].reshape(19,4)
    cache_intents(model,quartets[-1])
    _,base = model.episode(None,"test",rng,True,intents=quartets[-1])
    _,donor = model.memory(base)
    results,completed,omitted = [],[],[]
    skipped = model.skipped
    for j,roles in enumerate(quartets):
        cache_intents(model,roles)
        group = []
        for i in range(6):
            try:
                case,anchors = model.episode(None,"test",rng,True,intents=roles)
            except NoOpening:
                omitted.append(dict(index=j,completed_repeats=i))
                print("opening cap",j+1,i,flush=True)
                break
            _,basis = model.memory(anchors)
            geom = geometry(model,case,anchors)
            settings = [("full",basis,.1),("ordinary",None,.03),("ordinary",None,.1),
                        ("full_matched",basis,.1),("full",donor,.1)]
            arms = [model.evaluate(case,(mode,b),lr) for mode,b,lr in settings]
            arms.extend(ridge(geom,a,*params[a]) for a in (0,1))
            group.append(arms)
        if len(group)!=6:
            continue
        completed.append(j)
        donor = basis
        results.append(group)
        print("finished quartet",j+1,flush=True)
    a = np.array(results)
    n = len(completed)
    assert a.shape==(n,6,7,2,3,7)
    np.testing.assert_array_equal(a[:,:,:,:,0],np.broadcast_to(a[:,:,:1,:,0],a[:,:,:,:,0].shape))
    np.savez_compressed(ROOT/"bank-value-trials.npz",counts=a)
    grouped = a.mean(1)
    point = rates(grouped.mean(0))
    w = np.random.default_rng(390612).multinomial(n,np.full(n,1/n),size=4096)/n
    b = rates(np.tensordot(w,grouped,axes=(1,0)))
    effects = {}
    for i,m in enumerate(METHODS[1:],1):
        delta,ds = point[0]-point[i], b[:,0]-b[:,i]
        effects[m] = np.stack([delta,*np.quantile(ds,[.025,.975],axis=0)],axis=-1).tolist()
    out = dict(methods=METHODS,metrics=["new_accuracy","familiar_accuracy","repairs","harm","overall_accuracy"],
               worlds=["true","false"],steps=[0,1,4],quartets=quartets.tolist(),unused=perm[-1],
               duplicates=duplicates,completed=completed,omitted=omitted,skipped=model.skipped-skipped,ridge_grid=grid,ridge_params=params,
               tuning=tuning.tolist(),means=point.tolist(),quartet_means=rates(grouped).tolist(),full_minus=effects)
    (ROOT/"bank-values.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
