"""Cross each selection time with every trained trajectory on the two primary groups."""
import json

import unannounced as u
from unannounced import np, torch, ROOT, paired


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, labels=u.recovery.labels_with_extra()
    rows=json.loads((ROOT/"unannounced-settings.json").read_text())
    times=[r["times"]["selected"] for r in rows]
    unique=sorted(set(times))
    scores={group:np.zeros((len(rows),len(unique))) for group in ("earlier","recent")}
    with np.load(ROOT/"unannounced-trials.npz") as primary:
        for i,row in enumerate(rows):
            seed=row["seed"]
            history=u.practice(seed,row["earlier"],row["boundary"],labels)
            assert history[3]==times[i] and history[4]==row["log"]
            policies={str(t):t for t in unique}
            outputs,_,_=u.evaluate(seed,history,labels,policies,("glob","count"),False)
            for group,array in scores.items():
                earlier=row["earlier"]
                g=earlier if group=="earlier" else ("glob" if earlier=="count" else "count")
                for j,t in enumerate(unique):
                    value=outputs[str(t)][g]["adaptive"]
                    if t==times[i]:
                        assert np.array_equal(value,primary[f"{seed}-selected-{g}-adaptive"])
                    array[i,j]=value.mean()
        report={"timestamps":unique}
        for group,array in scores.items():
            gaps={}; efficiency=[]
            for i,row in enumerate(rows):
                chosen=array[i,unique.index(times[i])]
                other=[j for j in range(len(rows)) if j!=i]
                same=[j for j in other if rows[j]["earlier"]==row["earlier"]]
                identical=[j for j in same if rows[j]["boundary"]==row["boundary"]]
                for key,ids in (("all_other",other),("same_first",same),("same_design",identical)):
                    gaps.setdefault(key,[]).append(chosen-np.mean([array[i,unique.index(times[j])] for j in ids]))
                best=int(np.argmax(array[other].mean(0)))
                gaps.setdefault("best_other_scores",[]).append(chosen-array[i,best])
                g=row["earlier"] if group=="earlier" else ("glob" if row["earlier"]=="count" else "count")
                key=f"{row['seed']}-selected-{g}-"
                efficiency.append(float(primary[key+"adaptive"][:,1].mean()-primary[key+"current"][:,2].mean()))
            report[group]={"score_matrix":array.tolist(),"gaps":{k:paired(v) for k,v in gaps.items()},
                           "adaptive8_minus_current16":paired(efficiency)}
    (ROOT/"timing-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print("All selected timestamps, decision traces, and primary task scores reproduced.")


if __name__=="__main__":
    main()
