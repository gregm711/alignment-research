"""Screen direction reversals on previous decisions and test locked cases one Congress later."""
import json
import sys

from learn import ROOT, cached, csv, load, np, panel
from scipy.stats import fisher_exact

SELECTION=ROOT/"sign-selection.json"


def cells(y,quad):
    a,b,x,t=y[:,quad].T
    use=(a!=b)&np.isfinite(a+b+x+t)
    code=(2*a[use]+x[use]).astype(int)
    return np.stack([np.bincount(code,minlength=4),np.bincount(code,weights=t[use],minlength=4)],axis=-1).astype(int).reshape(2,2,2)


def statistics(c):
    n,s=c[:,:,0],c[:,:,1]
    prop=(s+.5)/(n+1)
    delta=prop[:,1]-prop[:,0]
    se=np.sqrt((prop*(1-prop)/(1+n)).sum(1))
    return np.array([min(delta[i]-1.64*se[i],-delta[1-i]-1.64*se[1-i]) for i in (0,1)])


def choose(tr,dev,live,rng):
    candidates={}
    samples=set()
    while len(samples)<4096:
        a,b,x,t=map(int,rng.choice(live,4,replace=False))
        samples.add((min(a,b),max(a,b),x,t))
    for quad in sorted(samples):
        c=cells(tr,quad)
        if c[:,:,0].min()<16:
            continue
        ranks=statistics(c); sign=int(ranks.argmax()); value=float(ranks[sign])
        key=quad[2:]
        if key not in candidates or value>candidates[key]["train_score"]:
            candidates[key]=dict(quad=list(quad),sign=sign,train_score=value)
    shortlist=sorted(candidates.values(),key=lambda v:-v["train_score"])[:30]
    eligible=[]
    for v in shortlist:
        c=cells(dev,v["quad"])
        if c[:,:,0].min()>=5:
            eligible.append(dict(**v,development_score=float(statistics(c)[v["sign"]])))
    return sorted(eligible,key=lambda v:-v["development_score"])[:3]


def effect(c,sign):
    n,s=c[:,:,0],c[:,:,1]
    if (n==0).any():
        return None
    q=s/n
    return ((q[:,1]-q[:,0])*([1,-1] if sign==0 else [-1,1])).tolist()


def main():
    selecting=sys.argv[1:]==["--select"]
    items={} if selecting else json.loads(SELECTION.read_text())
    old=load(110)
    tests=[]
    for cong in range(111,117):
        new=load(cong)
        rng=np.random.default_rng(7000+cong)
        chosen=rng.choice(np.flatnonzero(np.isfinite(old[2]).mean(0)>=.9),64,replace=False)
        with cached(f"HS{cong}_members.csv","members").open() as f:
            official={int(float(r["icpsr"])) for r in csv.DictReader(f) if r["chamber"]=="House"}
        incumbents=np.flatnonzero(np.isin(old[0][chosen],list(official)))
        (prev,_),(now,_)=panel(old,new,np.random.default_rng(7000+cong))
        old=new
        if selecting:
            tr,dev=np.split(prev,[int(.8*len(prev))])
            items[str(cong)]=choose(tr,dev,incumbents,np.random.default_rng(cong+71000))
            continue
        for item in items[str(cong)]:
            test=now[32:]
            c=cells(test,item["quad"])
            d=effect(c,item["sign"])
            ps=[]
            for j in (0,1):
                n,s=c[j,:,0],c[j,:,1]
                ps.append(float(fisher_exact(np.c_[s,n-s][::-1],alternative="greater" if j==item["sign"] else "less").pvalue))
            tests.append(dict(congress=cong,counts=c.tolist(),oriented=d,
                              halves=[effect(cells(x,item["quad"]),item["sign"]) for x in np.array_split(test,2)],
                              p=max(ps) if d is not None else 1.))
    if selecting:
        SELECTION.write_text(json.dumps(items,indent=1)+"\n")
        return
    order=np.argsort([r["p"] for r in tests]); floor=0.
    for j,i in enumerate(order):
        r=tests[i]
        floor=max(floor,min(1.,(len(tests)-j)*r["p"]))
        r["adjusted_p"]=floor
        r["passes"]=bool(r["oriented"] is not None and min(r["oriented"])>.03 and floor<.05)
    (ROOT/"sign-results.json").write_text(json.dumps(tests,indent=1)+"\n")


if __name__=="__main__":
    main()
