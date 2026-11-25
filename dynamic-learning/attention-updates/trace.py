"""Recover one ordinary-step false alarm from the serialized model and attribute its logit change."""
import json

from study import B, ROOT, Editor, corpus, logits, np, p, test_set, torch, words, write


def main():
    torch.set_num_threads(1)
    raw = json.loads((ROOT/"attention-results.json").read_text())
    source = json.loads((p.CACHE/"data_full.json").read_text())
    domains = json.loads((p.CACHE/"domains.json").read_text())
    with np.load(ROOT/"attention-checkpoints.npz") as archive, torch.no_grad():
        for r in raw:
            seed = r["seed"]
            rng = np.random.default_rng(seed)
            names = rng.choice(domains[r["domain"]],6,replace=False).tolist()
            assert names==r["intents"]
            tokens,pools,vocab = corpus(names,rng)
            support,query,_,_ = test_set(np.random.default_rng(seed+5000),pools)
            state = {k.split("/",1)[1]:torch.tensor(archive[k].tolist()) for k in archive.files
                     if k.startswith(f"{seed}/") and k.split("/",1)[1] not in ("inputs","logits","trigger")}
            net = Editor(len(vocab)+3)
            net.load_state_dict(state)
            # The exported incident always begins with a contrary report about type zero.
            peer = int(B[np.any(B==0,axis=1)].sum())
            first,check = int(support[0,0]),int(support[peer,1])
            assert np.array_equal(tokens[first].tolist(),archive[f"{seed}/trigger"])
            s = net.inputs(tokens[[check]])
            ids = query.ravel()
            q = net.inputs(tokens[ids.tolist()])
            before = net.value[None]
            after = write(s,torch.zeros(1),before,net.read)
            z0,z1 = [logits(q[None],v,net.read)[0] for v in (before,after)]
            cls = np.repeat(np.arange(6),12)
            crossed = ((z0<0)&(z1>=0)).tolist()
            hits = np.flatnonzero(np.array(crossed)&~np.isin(cls,[0,peer]))
            if not len(hits):
                continue
            j = int(hits[0])
            def utterance(idx,label):
                wanted = tokens[idx].tolist()
                matches = []
                for line,c in source["test"]:
                    if c!=label:
                        continue
                    seq = [2]+[vocab.get(w,1) for w in words(line)[:19]]
                    if seq+[0]*(20-len(seq))==wanted:
                        matches.append(line)
                assert len(matches)==1, "Ambiguous displayed text"
                return matches[0]
            def parts(v):
                a = torch.einsum("hd,hdk->hk",q[j],v[0]).tanh()
                return (a*net.read.reshape(2,-1)).sum(-1)
            components = parts(after)-parts(before)
            torch.testing.assert_close(components.sum(),z1[j]-z0[j],atol=1e-6,rtol=0)
            result = dict(seed=seed,selection="first exported incident in seed order with an unrelated default-step flip, then first such query",
                          trigger=[names[0],utterance(first,names[0])],check=[names[peer],utterance(check,names[peer])],
                          query=[names[cls[j]],utterance(ids[j],names[cls[j]])],before=float(z0[j]),after=float(z1[j]),
                          probabilities=[float(z0[j].sigmoid()),float(z1[j].sigmoid())],
                          head_dots=(q[j]*s[0]).sum(-1).tolist(),head_logit_deltas=components.tolist())
            (ROOT/"example.json").write_text(json.dumps(result,indent=2)+"\n")
            return
    raise AssertionError("No unrelated default-step crossing in the ten exported incidents")


if __name__=="__main__":
    main()
