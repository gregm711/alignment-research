"""Compare first-step logit drift after equally long attention value updates."""
from protect_values import METHODS, Protected, ROOT, json, np, pack, rates, torch, truth


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = json.loads((ROOT/"protected-values.json").read_text())
    lr = report["selected"]["local"]
    model = Protected()
    rng = np.random.default_rng(313151)
    for d in sorted(model.domains)[::2]:
        for _ in range(5):
            model.episode(d,"val",rng,include_anchors=True)
    samples,matched_four = [],[]
    with np.load(ROOT/"protected-value-trials.npz") as raw:
        expected = raw["counts"]
    for j,d in enumerate(report["test"]):
        _,anchor = model.episode(d,"test",rng,include_anchors=True)
        donor,_ = model.memory(anchor)
        subset,matched = [],[]
        for i in range(12):
            case,anchor = model.episode(d,"test",rng,include_anchors=True)
            local,_ = model.memory(anchor)
            g = model.gradient(case,slice(0,1),torch.ones(1))
            v = pack(g,model.h)
            projection = local @ (local.transpose(1,2) @ v)
            removed = 1-((v-.95*projection).norm()/v.norm()).square().item()
            overlap = (local.transpose(1,2)@donor).square().sum((1,2)).mean().item()/2
            movements = []
            for m in ("local","matched"):
                model.reset()
                model.write(g,(m,local),lr)
                with torch.no_grad():
                    delta = model.read(case,slice(4,None))-case[-1][4:]
                    distance = (sum((p-s).square().sum() for p,s in zip(model.params,model.start))).sqrt().item()
                movements.append([delta[:16].square().mean().sqrt().item(),
                                  (delta[16:]*(2*truth(True)[16:]-1)).mean().item(),distance])
            np.testing.assert_allclose(movements[0][2],movements[1][2],rtol=1e-4,atol=1e-8)
            model.reset()
            eq = model.evaluate(case,("matched",local),lr)
            np.testing.assert_array_equal(eq[:,0],expected[j,i,0,:,0])
            if report["selected"]["matched"]==lr:
                np.testing.assert_array_equal(eq,expected[j,i,METHODS.index("matched")])
            subset.append([removed,overlap,*movements[0],*movements[1]])
            matched.append(eq)
            donor = local
        samples.append(subset);matched_four.append(matched)
    a = np.array(samples)
    m = np.array(matched_four)
    t = expected[:,:,0]
    rng = np.random.default_rng(313152)
    w = [rng.multinomial(12,[1/12]*12,size=2048)/12 for _ in samples]
    def boot(v):
        return sum(np.tensordot(x,y,axes=(1,0)) for x,y in zip(w,v))/5
    draw = boot(a)
    delta = rates(t.mean((0,1)))-rates(m.mean((0,1)))
    ds = rates(boot(t))-rates(boot(m))
    out = dict(columns=["fraction_gradient_energy_removed","subspace_overlap","local_familiar_rms",
                        "local_new_signed","local_step_norm","matched_familiar_rms","matched_new_signed","matched_step_norm"],
               means=a.mean((0,1)).tolist(), trials=a.tolist(),
               movement_differences=np.stack([a.mean((0,1))[2:4]-a.mean((0,1))[5:7],
                   *np.quantile(draw[:,2:4]-draw[:,5:7],[.025,.975],axis=0)],axis=-1).tolist(),
               local_minus_same_rate_matched=np.stack([delta,*np.quantile(ds,[.025,.975],axis=0)],axis=-1).tolist())
    (ROOT/"value-diagnostic.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
