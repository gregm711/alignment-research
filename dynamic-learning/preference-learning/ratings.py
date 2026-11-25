"""Predict a new user's real ratings and intervene on a single observed rating."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path.home()/".cache/alignment-research/movielens/ml-100k.zip"
SHA = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
BUDGETS = (1,2,4,8)


def load():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True,exist_ok=True)
        with urlopen("https://files.grouplens.org/datasets/movielens/ml-100k.zip",timeout=60) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != SHA:
            raise ValueError("Unexpected MovieLens archive digest")
        CACHE.write_bytes(archive)
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == SHA
    with ZipFile(CACHE) as z:
        ratings = np.loadtxt(z.open("ml-100k/u.data"),dtype=int)
        genres = {int(row[0]):np.array(row[-19:],float) for row in
                  (line.rstrip().split(b"|") for line in z.open("ml-100k/u.item"))}
    people = np.unique(ratings[:,0])
    first = np.array([ratings[ratings[:,0]==i,3].min() for i in people])
    assert len(people)==943 and len(ratings)==100000
    order = people[np.lexsort((people,first))]
    hist,dev,test = np.split(order,[int(.6*len(order)),int(.77*len(order))])
    dev_start = first[people==dev[0]].item()
    test_start = first[people==test[0]].item()
    train = ratings[np.isin(ratings[:,0],hist) & (ratings[:,3]<dev_start)]
    counts = np.bincount(train[:,1])
    films = np.lexsort((np.arange(len(counts)),-counts))[:160]
    assert counts[films].min()>0 and train[:,3].max()<dev_start
    index = np.full(max(genres)+1,-1)
    index[films] = np.arange(len(films))
    matrix = np.full((len(hist),160),np.nan)
    rows = {user:j for j,user in enumerate(hist)}
    for user,film,rating,_ in train:
        if index[film]>=0:
            matrix[rows[user],index[film]] = float(rating>=4)
    def tasks(users, cutoff):
        found = []
        for user in users:
            rr = ratings[(ratings[:,0]==user) & (index[ratings[:,1]]>=0) & (ratings[:,3]<cutoff)]
            if len(rr)<20:
                continue
            rr = rr[np.lexsort((rr[:,1],rr[:,3]))]
            ids = index[rr[:,1]]
            assert len(np.unique(ids)) == len(rr)
            found.append((ids,(rr[:,2]>=4).astype(float),rr[7,3]==rr[8,3]))
        return found
    return matrix,np.stack([genres[f] for f in films]),tasks(dev,test_start),tasks(test,np.inf),dict(
        historical_ratings=len(train),historical_catalog_ratings=int(np.isfinite(matrix).sum()),
        development_start=int(dev_start),test_start=int(test_start),split_sizes=[len(hist),len(dev),len(test)])


def normalized(kernel):
    scale = np.sqrt(np.maximum(np.diag(kernel),1e-12))
    return kernel/scale[:,None]/scale[None,:]


def kernels(matrix,genres,base):
    observed = np.isfinite(matrix)
    residual = np.nan_to_num(matrix-base)
    n = observed.astype(float)
    covariance = residual.T@residual / (n.T@n + 12)
    values,vectors = np.linalg.eigh(covariance)
    covariance = normalized((vectors*values.clip(0))@vectors.T)
    perm = np.random.default_rng(2305).permutation(len(base))
    _,_,vt = np.linalg.svd(residual,full_matrices=False)
    items = vt[:12].T.copy()
    penalty = 4*np.eye(12)
    def update(mask,target,fixed):
        result = np.empty((len(target),12))
        for i,row in enumerate(mask):
            x = fixed[row]
            result[i] = np.linalg.solve(x.T@x+penalty, x.T@target[i,row])
        return result
    for _ in range(12):
        user = update(observed,residual,items)
        items = update(observed.T,residual.T,user)
    factors = items@items.T
    return dict(fixed=np.zeros_like(covariance),leniency=np.zeros_like(covariance),
                genre=normalized(genres@genres.T),covariance=covariance,
                scrambled=covariance[np.ix_(perm,perm)],collaborative=factors/np.diag(factors).mean())


def head(base, kernel, reg, ids, y):
    if reg is None:
        return base
    w = np.linalg.solve(kernel[np.ix_(ids,ids)]+reg*np.eye(len(ids)),y-base[ids])
    return np.clip(base+kernel[:,ids]@w,0,1)


def evaluate(tasks,base,kernel,reg):
    rows = []
    for ids,y,_ in tasks:
        future = ids[8:]
        rows.append([[( (prob>=.5)==y[8:] ).mean(), ((prob-y[8:])**2).mean()]
                     for n in BUDGETS for prob in [head(base,kernel,reg,ids[:n],y[:n])[future]]])
    return np.array(rows)


def tune(tasks,base,channels):
    chosen = {}
    for method,extra in channels.items():
        if method=="fixed":
            chosen[method] = (0,None)
            continue
        candidates = []
        for strength in ((0,) if method=="leniency" else (0,.25,1,4)):
            for reg in (.25,1,4,16):
                result = evaluate(tasks,base,1+strength*extra,reg)
                candidates.append((result[:,:,1].mean(),strength,reg))
        _,s,r = min(candidates)
        chosen[method] = (s,r)
    return chosen


def run():
    matrix,genre,dev,test,metadata = load()
    assert dev and test
    base = (np.nansum(matrix,axis=0)+4*np.nanmean(matrix))/(np.isfinite(matrix).sum(0)+4)
    channels = kernels(matrix,genre,base)
    chosen = tune(dev,base,channels)
    models = {k:(1+s*channels[k],r) for k,(s,r) in chosen.items()}
    comparator = min(("leniency","genre"),key=lambda k:evaluate(dev,base,*models[k])[:,:,1].mean())
    scores = {k:evaluate(test,base,*m) for k,m in models.items()}
    counts = []
    for ids,y,_ in test:
        def preds(k):
            K,r = models[k]
            branches = [y[:4],y[:5],np.r_[y[:4],1-y[4]]]
            return np.stack([head(base,K,r,ids[:len(b)],b)[ids[8:]]>=.5 for b in branches])
        a,b = preds("covariance"),preds(comparator)
        row = []
        for j in (1,2):
            eligible = (a[0]==y[8:]) & (b[0]==y[8:]) if j==2 else (a[0]!=y[8:]) & (b[0]!=y[8:])
            row.append([np.sum(eligible & (v[j]!=v[0])) for v in (a,b)] + [eligible.sum()])
        counts.append(row)
    counts = np.array(counts)
    rng = np.random.default_rng(6713)
    bootstrap = rng.integers(len(test),size=(6000,len(test)))
    def stats(value,resamples):
        return [float(value), *np.quantile(resamples,[.025,.975]).tolist()]
    contrasts = {}
    for k in models:
        diff = (scores["covariance"]-scores[k]).mean(1)
        contrasts[k] = [stats(diff[:,j].mean(),diff[bootstrap,j].mean(1)) for j in range(2)]
    feedback = {}
    for j,k in enumerate(("repair","harm")):
        c = counts[:,j]
        num = c[:,0]-c[:,1]
        den = c[:,2]
        feedback[k] = dict(rates=(c[:,:2].sum(0)/den.sum()).tolist(),eligible=int(den.sum()),
                           difference=stats(num.sum()/den.sum(),num[bootstrap].sum(1)/den[bootstrap].sum(1)))
    report = dict(metadata=metadata,eligible_development=len(dev),eligible_test=len(test),
                  tied_boundaries=int(sum(x[2] for x in test)),chosen=chosen,comparator=comparator,
                  metrics=["accuracy","brier"],budgets=BUDGETS,curves={k:v.mean(0).tolist() for k,v in scores.items()},
                  covariance_minus=contrasts,feedback=feedback)
    np.savez_compressed(ROOT/"rating-trials.npz",**scores,feedback_counts=counts)
    (ROOT/"ratings.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    run()
