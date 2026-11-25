"""Recompute paired effects and independently validate serialized attention updates."""
import json

from study import ROOT, TIMES, Editor, np, p, torch, logits, write


def verify():
    with np.load(ROOT/"attention-checkpoints.npz") as archive:
        worst = 0.
        for seed in range(1101,1111):
            state = {k.split("/",1)[1]:torch.tensor(archive[k].tolist())
                     for k in archive.files if k.startswith(f"{seed}/")}
            ids = state.pop("inputs").long()
            expected = state.pop("logits")
            trigger = state.pop("trigger").long()[None]
            net = Editor(len(state["word.weight"]))
            net.load_state_dict(state)
            with torch.no_grad():
                actual = net(ids)
                torch.testing.assert_close(actual,expected,rtol=0,atol=2e-6)
                assert torch.equal(actual>=0,expected>=0)
                worst = max(worst,float((actual-expected).abs().max()))
                h = net.inputs(trigger).detach()
            # Validate first and nonzero-state edits against the derivative of binary log loss.
            for initial in (torch.zeros_like(net.value),net.value):
                v = initial.detach().clone().requires_grad_()
                z = logits(h[:,None],v[None],net.read)[0,0]
                label = torch.tensor(1. if not initial.any() else 0.)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z,label)
                grad, = torch.autograd.grad(loss,v)
                ref = v-4*grad
                fast = write(h,label[None],v[None],net.read)[0]
                torch.testing.assert_close(fast,ref,rtol=0,atol=2e-6)
                if not initial.any():
                    torch.testing.assert_close(fast,net.value,rtol=0,atol=2e-6)
                    with torch.no_grad():
                        net.value.zero_()
                        assert torch.eq(net(ids),-1).all()
                        net.value.copy_(state["value"])
    return worst


def main():
    torch.set_num_threads(1)
    raw = json.loads((ROOT/"attention-results.json").read_text())
    out = {"semantic_accuracy":p.interval(np.array([r["semantic"] for r in raw])),
           "max_export_logit_error":verify(),"checkpoints":{}}
    for t in TIMES:
        records = [r["checkpoints"][str(t)] for r in raw]
        scores = np.array([[m["scores"] for m in row] for row in records])
        observer = np.array([r["reference"] for r in raw])
        feedback = np.array([[m["feedback"] for m in row] for row in records])
        influence = np.array([[m["influence"] for m in row] for row in records])
        # Each history is evaluated on both pairings. Always give the matched arm its current regime.
        a,b = (1,0) if t>48 else (0,1)
        matched = (scores[:,0,:,a]+scores[:,1,:,b])/2
        wrong = (scores[:,1,:,a]+scores[:,0,:,b])/2
        random = scores[:,2].mean(2)
        baseline = observer.mean(2)
        fb = (feedback[:,0,a]+feedback[:,1,b])/2
        fr = feedback[:,2].mean(1)
        assert np.array_equal(fb[:,0,0],fb[:,0,1])
        shift = (influence[:,0,a]-influence[:,0,b]+influence[:,1,b]-influence[:,1,a])/2
        out["checkpoints"][t] = dict(
            curves={k:v.mean(0).tolist() for k,v in (("matched",matched),("other",wrong),("random",random),("observer",baseline))},
            effects={k:[p.interval((matched-v)[:,j].mean(1)) for j in range(3)]
                     for k,v in (("history-swap",wrong),("random",random),("observer",baseline))},
            influence=p.interval(shift),unrelated=float(influence[:,:2,2].mean()),
            feedback=fb.mean(0).tolist(),random_feedback=fr.mean(0).tolist(),
            feedback_difference=[[p.interval(fb[:,j,k]-fr[:,j,k]) for k in range(2)] for j in range(3)])
    (ROOT/"summary.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
