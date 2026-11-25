"""Intervene on the sign of attention inputs and attribute new errors to one weight edit."""
import copy
import json

from checks import (A, B, ROOT, WIDTH, Editor, corpus, evaluate, explicit, fork_features, forks,
                    logits, np, p, pretrain, test_set, torch, train, write)
from torch.nn import functional as F


class NonnegativeEditor(Editor):
    def inputs(self,ids):
        return F.normalize(super().inputs(ids).relu(),dim=-1)


def spill(net,f,rate):
    x,q,trigger,pos = f
    peer = trigger ^ 1
    n = len(trigger)
    one = torch.ones(n)
    before = write(x[trigger.tolist(),pos.tolist()],one,net.value.expand(n,-1,-1,-1),net.read)
    after = before+rate*(write(x[peer.tolist(),((pos+1)%9).tolist()],1-one,before,net.read)-before)
    z0,z1 = [logits(q.expand(n,-1,-1,-1),v,net.read) for v in (before,after)]
    outside = (np.arange(6)[None,:]!=trigger[:,None]) & (np.arange(6)[None,:]!=peer[:,None])
    outside = torch.tensor(np.repeat(outside,9,axis=1).tolist())
    was_correct = (z0<0)&outside
    was_wrong = (z0>=0)&outside
    return dict(new=int(((z1>=0)&was_correct).sum()),was_correct=int(was_correct.sum()),
                fixed=int(((z1<0)&was_wrong).sum()),was_wrong=int(was_wrong.sum()),
                after_wrong=int(((z1>=0)&outside).sum()),outside=int(outside.sum()),
                max_increase=float((z1-z0).max()))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domains = json.loads((p.CACHE/"domains.json").read_text())
    rows = []
    for i,seed in enumerate(range(1201,1211)):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        names = rng.choice(domains[list(domains)[i]],6,replace=False).tolist()
        tokens,pools,vocab = corpus(names,rng)
        net = Editor(len(vocab)+3)
        pretrain(net,tokens,pools,rng)
        posnet = NonnegativeEditor(len(vocab)+3)
        posnet.load_state_dict(net.state_dict())
        signed,history = train(copy.deepcopy(rng),net,tokens,pools)
        positive,other = train(rng,posnet,tokens,pools)
        assert np.array_equal(history,other)
        posnet.load_state_dict(signed[0].state_dict())
        cases = test_set(np.random.default_rng(seed+5000),pools)
        with torch.no_grad():
            row = dict(seed=seed,intents=names,observer=explicit(net,tokens,cases,history),models={})
            for key,m in zip(("signed","random","positive","positive-random","test-only"),[*signed,*positive,posnet]):
                f = fork_features(m,tokens,pools["test"])
                result = dict(capability=evaluate(m,tokens,cases)[0],feedback={},spill={})
                for rate in (1,4):
                    result["feedback"][rate] = np.array([[forks(m,f,pairs,"gain",rate,rev) for rev in (False,True)]
                                                        for pairs in (A,B)]).tolist()
                    result["spill"][rate] = spill(m,f,rate)
                    if isinstance(m,NonnegativeEditor):
                        delta = result["spill"][rate]
                        assert delta["new"]==0 and delta["max_increase"]<2e-6
                row["models"][key] = result
        rows.append(row)
        print("seed",seed,"finished",flush=True)
    (ROOT/"direction-trials.json").write_text(json.dumps(rows,indent=1)+"\n")


if __name__=="__main__":
    main()
