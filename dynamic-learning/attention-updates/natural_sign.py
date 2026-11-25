"""Select conditional rating reversals on development people and verify on two cohorts."""
import hashlib
import itertools
import json
import sys
from urllib.request import urlopen
from zipfile import ZipFile

from preferences import ANCHOR, CACHE, ROOT, SOURCE, TARGET, inputs, np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact

SELECTION = ROOT/"natural-sign-selection.json"


def second_cohort():
    import xlrd
    name = "jester_dataset_1_2.zip"
    path = CACHE/name
    digest = "4d312fd824d8c8cb2c9c952a7b1fb7c00d1f5fd3ece370f7285aa701d371748b"
    if not path.exists():
        with urlopen(SOURCE+name,timeout=60) as f:
            payload = f.read()
        if hashlib.sha256(payload).hexdigest()!=digest:
            raise ValueError("Unexpected second ratings archive")
        path.write_bytes(payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with ZipFile(path) as z:
        book = xlrd.open_workbook(file_contents=z.read("jester-data-2.xls"),on_demand=True)
        sheet = book.sheet_by_index(0)
        ratings = np.array([sheet.row_values(i,1,101) for i in range(sheet.nrows)],dtype=np.float32)
        book.release_resources()
    assert ratings.shape==(23500,100)
    obs = ratings!=99
    assert abs(ratings[obs]).max()<=10
    keep = obs[:,ANCHOR].all(1)&(obs[:,TARGET].sum(1)>=10)
    return np.where(obs[keep],ratings[keep]>0,np.nan)


def table(y,a,b,x,targets):
    out = []
    for first in (0,1):
        for answer in (0,1):
            rows = (y[:,a]==first)&(y[:,b]!=first)&(y[:,x]==answer)
            values = y[rows][:,targets]
            out.append(np.stack([np.isfinite(values).sum(0),np.nansum(values,axis=0)]))
    return np.stack(out).reshape(2,2,2,len(targets)).astype(int)


def ranks(cells):
    n = cells[:,:,0]
    prob = (cells[:,:,1]+.5)/(n+1)
    diff = prob[:,1]-prob[:,0]
    se = np.sqrt((prob*(1-prob)/(n+1)).sum(1))
    scores = np.stack([np.minimum(diff[0]-1.64*se[0],-diff[1]-1.64*se[1]),
                       np.minimum(-diff[0]-1.64*se[0],diff[1]-1.64*se[1])])
    return scores


def collect(y,item):
    a,b,x,t = item["indices"]
    return table(y,a,b,x,[t])[:,:,:,0]


def select(train,dev):
    best = {}
    for a,b in itertools.combinations(ANCHOR,2):
        for x in np.setdiff1d(ANCHOR,[a,b]):
            counts = table(train,a,b,x,TARGET)
            score = ranks(counts)
            for j,t in enumerate(TARGET):
                if counts[:,:,0,j].min()<100:
                    continue
                sign = int(score[:,j].argmax())
                value = float(score[sign,j])
                if (x,t) not in best or value>best[x,t]["train_score"]:
                    best[x,t] = dict(indices=list(map(int,[a,b,x,t])),sign=sign,train_score=value)
    candidates = sorted(best.values(),key=lambda v:-v["train_score"])[:30]
    for v in candidates:
        c = collect(dev,v)
        v["dev_score"] = float(ranks(c)[v["sign"]])
    return sorted(candidates,key=lambda v:-v["dev_score"])[:5]


def evaluate(train,people,item):
    c = collect(people,item)
    n,success = c[:,:,0],c[:,:,1]
    prop = success/n
    d = (prop[:,1]-prop[:,0])*([1,-1] if item["sign"]==0 else [-1,1])
    ps = []
    for group in range(2):
        counts = np.c_[success[group],n[group]-success[group]]
        direction = "greater" if group==item["sign"] else "less"
        ps.append(float(fisher_exact(counts[::-1],alternative=direction).pvalue))
    fit = collect(train,item)
    y = fit[:,:,1].ravel()+.5
    trials = fit[:,:,0].ravel()+1
    design = np.array([[1,g,x] for g in (0,1) for x in (0,1)])
    def objective(w):
        z = design@w
        return (np.logaddexp(0,z)*trials-y*z).sum(), design.T@(expit(z)*trials-y)
    opt = minimize(objective,np.zeros(3),jac=True,method="BFGS",tol=1e-6)
    assert np.linalg.norm(opt.jac)<1e-4
    def brier(probs):
        return float(((1-probs)**2*success+probs**2*(n-success)).sum()/n.sum())
    return dict(counts=c.tolist(),oriented_effects=d.tolist(),p=max(ps),one_sided=ps,
                brier_additive=brier(expit(design@opt.x).reshape(2,2)),brier_interaction=brier((y/trials).reshape(2,2)))


def main():
    _,((train,_),(dev,_),(test,_)) = inputs()
    if sys.argv[1:]==["--select"]:
        SELECTION.write_text(json.dumps(select(train,dev),indent=2)+"\n")
        return
    items = json.loads(SELECTION.read_text())
    result = {}
    for name,people in (("first-test",test),("second-release",second_cohort())):
        rows = [evaluate(train,people,it) for it in items]
        order = np.argsort([r["p"] for r in rows])
        floor = 0.
        for j,idx in enumerate(order):
            floor = max(floor,min(1.,rows[idx]["p"]*(len(rows)-j)))
            rows[idx]["adjusted_p"] = floor
            rows[idx]["passes"] = bool(min(rows[idx]["oriented_effects"])>.03 and floor<.05)
        result[name] = dict(users=len(people),cases=rows)
    (ROOT/"natural-sign-results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
