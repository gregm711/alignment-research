"""Retrospectively audit temporal testing for repeated and reversed voting patterns."""
import json

from learn import ROOT, load, np, panel


def distances(history,query):
    known=np.isfinite(history).astype(float)
    positive=np.nan_to_num(history).astype(float)
    km=np.isfinite(query).astype(float)
    qm=np.nan_to_num(query).astype(float)
    both=km@known.T
    different=(qm@known.T+km@positive.T-2*qm@positive.T)
    eligible=both>=24
    assert eligible.any(1).all()
    denom=np.maximum(both,1)
    ordinary=np.where(eligible,different/denom,np.inf)
    reversed_distance=np.where(eligible,np.minimum(different,both-different)/denom,np.inf)
    return ordinary.min(1),reversed_distance.min(1)


def main():
    rec=json.loads((ROOT/"results.json").read_text())
    output=[]
    old=load(110)
    with np.load(ROOT/"vote-trials.npz") as saved:
        for r in rec:
            cong=r["congress"]
            new=load(cong)
            (prev,_),(now,rolls)=panel(old,new,np.random.default_rng(7000+cong))
            old=new
            past=np.vstack([prev[:r["eligible"][0]],now[:32]])
            row=dict(congress=cong)
            for block,sl in (("early",slice(32,288)),("late",slice(-256,None))):
                assert np.array_equal(rolls[sl],r["blocks"][block]["rollnumbers"])
                ds=distances(past,now[sl])
                curves={k:saved[f"{cong}-{block}-{k}"].mean(1) for k in ("recent","frozen","template-recent")}
                entry={}
                for name,d in zip(("direct","either-sign"),ds):
                    far=np.zeros(len(d),bool)
                    far[np.lexsort((np.arange(len(d)),-d))[:len(d)//4]]=True
                    groups=dict(exact=d==0,near=d<=.05,distant=d>=.15,farthest_quarter=far)
                    entry[name]=dict(distances=d.tolist(),groups={
                        k:dict(n=int(mask.sum()),metrics={m:v[mask].mean(0).tolist() for m,v in curves.items()} if mask.any() else {})
                        for k,mask in groups.items()})
                row[block]=entry
            output.append(row)
    (ROOT/"novelty.json").write_text(json.dumps(output,indent=1)+"\n")


if __name__=="__main__":
    main()
