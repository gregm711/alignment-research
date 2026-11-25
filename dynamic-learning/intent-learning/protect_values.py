"""Use old logit gradients to reparameterize attention before a new task arrives."""
import json

from value_history import Experiment, F, METRICS, ROOT, counts, np, rates, torch, truth

METHODS = ("local", "ordinary", "matched", "donor", "full")
RATES = (0., .001, .003, .01, .03, .1)


def pack(parts, heads):
    w,b = parts
    return torch.cat((w,b[:,None]),dim=1).reshape(heads,-1,w.shape[1]+1)


class Protected(Experiment):
    def memory(self, anchors):
        examples = []
        for i in range(len(anchors[0])):
            g = torch.autograd.grad(self.read(anchors,slice(i,i+1)).sum(),self.params)
            flat = pack(g,self.h).flatten()
            examples.append(F.normalize(flat,dim=0))
        old = torch.stack(examples)
        by_head = old.reshape(len(examples),self.h,-1,self.start[0].shape[1]+1)
        covariance = torch.einsum("ahik,ahjk->hij",by_head,by_head)
        _,eig = torch.linalg.eigh(covariance)
        basis,r = torch.linalg.qr(old.T,mode="reduced")
        assert r.diagonal().abs().min() > 1e-6
        return eig[:,:,-2:].contiguous(),basis

    @torch.no_grad()
    def write(self, grad, memory, lr, matched=False):
        method, basis = memory
        if method == "ordinary":
            for p,g in zip(self.params,grad):
                p.add_(g,alpha=-lr)
            return
        matrix = pack(grad,self.h)
        if method in ("full","full_matched"):
            along = (basis @ (basis.T @ matrix.flatten())).reshape_as(matrix)
        else:
            along = basis @ (basis.transpose(1,2) @ matrix)
        filtered = matrix - .95*along
        if method in ("matched","full_matched"):
            filtered = matrix*(filtered.norm()/matrix.norm())
        step = filtered.reshape(self.start[0].shape[0],-1)
        self.params[0].add_(step[:,:-1],alpha=-lr)
        self.params[1].add_(step[:,-1],alpha=-lr)

    def literal(self, case, basis, lr):
        self.reset()
        g = self.gradient(case,slice(0,1),torch.ones(1))
        self.write(g,("local",basis),lr)
        with torch.no_grad():
            ref = self.read(case,slice(None))
            self.reset()
            projector = basis @ basis.transpose(1,2)
            eye = torch.eye(basis.shape[1])
            enlarge = eye+(1/np.sqrt(.05)-1)*projector
            shrink = eye+(np.sqrt(.05)-1)*projector
            value = (enlarge @ pack(self.params,self.h)).reshape(self.start[0].shape[0],-1)
            self.params[0].copy_(value[:,:-1])
            self.params[1].copy_(value[:,-1])
            out = self.last.attention.output.dense.weight
            saved = out.clone()
            blocks = out.reshape(out.shape[0],self.h,-1).transpose(0,1)
            out.copy_((blocks @ shrink).transpose(0,1).reshape_as(out))
            before = self.read(case,slice(None))
        g = self.gradient(case,slice(0,1),torch.ones(1))
        self.write(g,("ordinary",None),lr)
        with torch.no_grad():
            post = self.read(case,slice(None))
            out.copy_(saved)
            self.reset()
        errors = [float((before-case[-1]).abs().max()),float((post-ref).abs().max())]
        assert max(errors)<1e-4
        torch.testing.assert_close(post>=0,ref>=0)
        return errors


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Protected()
    practice,test = sorted(model.domains)[::2], sorted(model.domains)[1::2]
    rng = np.random.default_rng(313151)
    grid = np.zeros((len(METHODS),len(RATES)))
    results,equality,sample = [],[],{}
    for split,domains in (("val",practice),("test",test)):
        for d in domains:
            _,before = model.episode(d,split,rng,include_anchors=True)
            donor,_ = model.memory(before)
            domain = []
            for i in range(4 if split=="val" else 12):
                case,anchors = model.episode(d,split,rng,include_anchors=True)
                local,full = model.memory(anchors)
                memories = (local,None,local,donor,full)
                if split == "test" and i == 0:
                    equality.append(model.literal(case,local,lr=.03))
                    sample[d] = local.numpy()
                entries = []
                for j,(m,basis) in enumerate(zip(METHODS,memories)):
                    options = RATES if split == "val" else (RATES[grid[j].argmax()],)
                    for r,lr in enumerate(options):
                        if lr == 0:
                            base = (case[-1][4:]>=0)
                            rows = []
                            for genuine in (True,False):
                                ok = base == truth(genuine).bool()
                                rows.append([counts(ok,ok)]*3)
                            score = np.array(rows)
                        else:
                            score = model.evaluate(case,(m,basis),lr)
                        if split=="val":
                            grid[j,r] += score[:,1:,-1].mean()/(32*20)
                        else:
                            entries.append(score)
                donor = local
                if split=="test":
                    domain.append(entries)
            if split=="test":
                results.append(domain)
            print("finished",split,d,flush=True)
    a = np.array(results)
    assert a.shape == (5,12,5,2,3,7)
    np.testing.assert_array_equal(a[:,:,:,:,0],np.broadcast_to(a[:,:,:1,:,0],a[:,:,:,:,0].shape))
    np.savez_compressed(ROOT/"protected-value-trials.npz",counts=a,**sample)
    bootrng = np.random.default_rng(313152)
    w = [bootrng.multinomial(12,[1/12]*12,size=2048)/12 for _ in test]
    p = rates(a.mean((0,1)))
    b = rates(sum(np.tensordot(v,row,axes=(1,0)) for v,row in zip(w,a))/5)
    effects = {}
    for j,m in enumerate(METHODS[1:],1):
        delta,draws = p[0]-p[j],b[:,0]-b[:,j]
        effects[m] = np.stack([delta,*np.quantile(draws,[.025,.975],axis=0)],axis=-1).tolist()
    out = dict(methods=METHODS,steps=[0,1,4],worlds=["true","false"],metrics=METRICS,
               test=test,equality=equality,grid=RATES,tuning=grid.tolist(),
               selected={m:RATES[row.argmax()] for m,row in zip(METHODS,grid)},skipped=model.skipped,
               means=p.tolist(),domain_means=rates(a.mean(1)).tolist(),local_minus=effects)
    (ROOT/"protected-values.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
