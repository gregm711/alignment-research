"""Keep every within-head block of past gradient memory fixed and alter the cross terms."""
from bank_values import cache_intents, NoOpening, Protected, ROOT, corpus, json, np, rates, torch
from protect_values import pack


class Signed(Protected):
    @torch.no_grad()
    def write(self, grad, memory, lr, matched=False):
        mode, basis, signs = memory
        g = pack(grad,self.h)
        q = basis.reshape(self.h,-1,basis.shape[1])
        flat = g.reshape(self.h,-1)
        coefficients = (q.transpose(1,2) @ flat[:,:,None]).squeeze(-1)
        intact = flat - .95*(q @ coefficients.sum(0))
        if mode=="intact":
            delta = intact
        else:
            if mode=="signed":
                total = (signs[:,None]*coefficients).sum(0)
                removed = signs[:,None]*(q @ total)
            else:
                removed = (q @ coefficients[:,:,None]).squeeze(-1)
            delta = flat-.95*removed
            delta *= intact.norm()/delta.norm()
        delta = delta.reshape(self.params[0].shape[0],-1)
        self.params[0].add_(delta[:,:-1],alpha=-lr)
        self.params[1].add_(delta[:,-1],alpha=-lr)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Signed()
    model.pool,names,_ = corpus()
    roles = np.random.default_rng(390611).permutation(names)[:76].reshape(19,4)
    r = np.random.default_rng(780351)
    patterns = torch.tensor(np.stack([r.permutation(np.repeat([-1.,1.],6)) for _ in range(4)]),dtype=torch.float32)
    assert len({tuple(x) for x in patterns.tolist()}) == 4
    rng = np.random.default_rng(780352)
    result,completed,omitted = [],[],[]
    for j,quartet in enumerate(roles):
        cache_intents(model,quartet)
        group = []
        for _ in range(3):
            try:
                case,anchors = model.episode(None,"test",rng,True,intents=quartet)
            except NoOpening:
                omitted.append(dict(index=j,completed_repeats=len(group)))
                break
            _,basis = model.memory(anchors)
            settings = [("intact",basis,None),("block",basis,None)]
            settings.extend(("signed",basis,s) for s in patterns)
            group.append([model.evaluate(case,mem,.1) for mem in settings])
        if len(group)!=3:
            continue
        completed.append(j)
        result.append(group)
        print("finished",j+1,flush=True)
    a = np.array(result)
    n = len(completed)
    assert a.shape==(n,3,6,2,3,7)
    np.testing.assert_array_equal(a[:,:,:,:,0],np.broadcast_to(a[:,:,:1,:,0],a[:,:,:,:,0].shape))
    np.savez_compressed(ROOT/"head-sign-trials.npz",counts=a,signs=patterns.numpy())
    means = a.mean(1)
    groups = np.concatenate([means[:,:2],means[:,2:].mean(1,keepdims=True)],1)
    p = rates(groups.mean(0))
    w = np.random.default_rng(780353).multinomial(n,np.full(n,1/n),size=4096)/n
    b = rates(np.tensordot(w,groups,axes=(1,0)))
    effects = {}
    for i,m in enumerate(("block","signed"),1):
        v,ds = p[0]-p[i], b[:,0]-b[:,i]
        effects[m] = np.stack([v,*np.quantile(ds,[.025,.975],axis=0)],axis=-1).tolist()
    data = dict(methods=["intact","block","signed"],worlds=["true","false"],steps=[0,1,4],
                metrics=["new_accuracy","familiar_accuracy","repairs","harm","accuracy"],means=p.tolist(),
                intact_minus=effects,patterns=patterns.tolist(),skipped=model.skipped,completed=completed,omitted=omitted,quartets=roles.tolist())
    (ROOT/"head-signs.json").write_text(json.dumps(data,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
