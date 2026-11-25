"""Measure how an exactly function-preserving attention rewrite changes the next SGD step."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import BertModel
from prepare import CACHE, DATA, MODEL, fetch

ROOT = Path(__file__).resolve().parent
GAINS = ("change", "exception", "ordinary", "permuted", "matched")
METRICS = ("new_accuracy", "familiar_accuracy", "truth_or_alternate_repairs", "harm", "accuracy")
MODEL_SHA = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def backbone():
    path = CACHE.parent/"minilm-float"
    path.mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        target = path/name
        if not target.exists():
            with urlopen(MODEL+name, timeout=90) as response:
                data = response.read()
            if name.endswith("safetensors") and hashlib.sha256(data).hexdigest() != MODEL_SHA:
                raise ValueError("Unexpected MiniLM weights")
            target.write_bytes(data)
    assert hashlib.sha256((path/"config.json").read_bytes()).hexdigest() == "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"
    assert hashlib.sha256((path/"model.safetensors").read_bytes()).hexdigest() == MODEL_SHA
    m = BertModel.from_pretrained(path,local_files_only=True,add_pooling_layer=False).eval()
    m.requires_grad_(False)
    return m


class NoOpening(RuntimeError):
    pass


class Experiment:
    def __init__(self):
        self.model = backbone()
        self.last = self.model.encoder.layer[-1]
        v = self.last.attention.self.value
        self.params = (v.weight, v.bias)
        self.start = [p.detach().clone() for p in self.params]
        for p in self.params:
            p.requires_grad_(True)
        self.h = self.last.attention.self.num_attention_heads
        self.tok = Tokenizer.from_file(str(fetch(MODEL,"tokenizer.json")))
        self.tok.enable_truncation(max_length=128)
        self.tok.enable_padding()
        self.domains = json.loads(fetch(DATA,"domains.json").read_text())
        self.pool = {}
        corpus = json.loads(fetch(DATA,"data_full.json").read_text())
        seen = set()
        for split in ("test","val","train"):
            for text,intent in corpus[split]:
                key = " ".join(text.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                self.pool.setdefault((split,intent),[]).append(text)
        self.skipped = 0

    @torch.no_grad()
    def reset(self):
        for p,w in zip(self.params,self.start):
            p.copy_(w)

    @torch.no_grad()
    def episode(self, domain, split, rng, include_anchors=False, intents=None):
        self.reset()
        for attempt in range(1024):
            a,b,c,d = rng.choice(self.domains[domain],4,replace=False) if intents is None else intents
            def pick(intent, n, group="train"):
                return list(rng.choice(self.pool[group,intent],n,replace=False))
            texts = pick(a,6)+pick(b,6)+pick(c,17)+pick(d,2)
            for i in (a,b,c,d):
                texts += pick(i,8,split)
            assert len(texts) == len(set(texts)) == 63
            if intents is not None and hasattr(self,"cached_train"):
                emb = torch.stack([self.cached_train[t] for t in texts[:28]])
                head = 8*(F.normalize(emb[:6].mean(0),dim=0)-F.normalize(emb[6:12].mean(0),dim=0))
                if (emb[12:] @ head).min().item() > 1e-5:
                    self.skipped += 1
                    continue
            pack = self.tok.encode_batch(texts)
            ids = torch.tensor([v.ids for v in pack])
            mask = torch.tensor([v.attention_mask for v in pack])
            outputs = self.model(ids, attention_mask=mask, output_hidden_states=True)
            def pool(x):
                return F.normalize((x*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True))
            embeds = pool(outputs.last_hidden_state)
            head = 8*(F.normalize(embeds[:6].mean(0),dim=0)-F.normalize(embeds[6:12].mean(0),dim=0))
            logits = embeds @ head
            eligible = torch.where(logits[12:28] < 0)[0].tolist()
            if not eligible:
                self.skipped += 1
                continue
            indices = [12+int(rng.choice(eligible)),29,28,30,*range(31,63)]
            hidden = outputs.hidden_states[-2][indices].contiguous()
            mask = mask[indices]
            ext = self.model.get_extended_attention_mask(mask,mask.shape)
            case = (hidden,mask,ext,head,logits[indices])
            torch.testing.assert_close(self.read(case,slice(None)),case[-1],atol=4e-5,rtol=1e-5)
            if include_anchors:
                old_mask = torch.tensor([v.attention_mask for v in pack[:12]])
                anchors = (outputs.hidden_states[-2][:12].clone(),old_mask,
                           self.model.get_extended_attention_mask(old_mask,old_mask.shape),
                           head,logits[:12])
                return case,anchors
            return case
        raise NoOpening("No rejected opening in 1024 proposals")

    def embeddings(self, case, ix):
        hidden,mask,ext,_,_ = case
        x = self.last(hidden[ix],attention_mask=ext[ix])[0]
        z = F.normalize((x*mask[ix,:,None]).sum(1)/mask[ix].sum(1,keepdim=True))
        return z

    def read(self, case, ix):
        return self.embeddings(case,ix) @ case[3]

    def gradient(self, case, ix, labels):
        z = self.read(case,ix)
        return torch.autograd.grad(F.binary_cross_entropy_with_logits(z,labels),self.params)

    def agreement(self, case):
        g = self.gradient(case,slice(0,1),torch.ones(1))
        rows = []
        for genuine in (True,False):
            q = self.gradient(case,slice(4,None),truth(genuine))
            rows.append(sum((a*b).reshape(self.h,-1).sum(1) for a,b in zip(g,q)).numpy())
        return np.stack(rows)

    @torch.no_grad()
    def write(self, grad, gain, lr, matched=False):
        scaled = [(a.reshape(self.h,-1)*gain[:,None]).reshape_as(a) for a in grad]
        if matched:
            ratio = (sum(v.square().sum() for v in scaled)/sum(v.square().sum() for v in grad)).sqrt()
            scaled = [ratio*a for a in grad]
        for p,delta in zip(self.params,scaled):
            p.add_(delta,alpha=-lr)

    def evaluate(self, case, gain, lr, matched=False):
        rows = []
        initial = case[-1][4:] >= 0
        for genuine in (True,False):
            self.reset()
            ys = [1, int(not genuine),int(genuine),int(not genuine)]
            target = truth(genuine).bool()
            correct0 = initial == target
            curve = [counts(correct0,correct0)]
            for i,y in enumerate(ys):
                g = self.gradient(case,slice(i,i+1),torch.tensor([float(y)]))
                self.write(g,gain,lr,matched)
                if i in (0,3):
                    with torch.no_grad():
                        correct = (self.read(case,slice(4,None)) >= 0) == target
                    curve.append(counts(correct,correct0))
            rows.append(curve)
        self.reset()
        return np.array(rows)

    def literal_check(self, case, gain, lr):
        self.reset()
        expected = self.gradient(case,slice(0,1),torch.ones(1))
        self.write(expected,gain,lr)
        with torch.no_grad():
            reference = self.read(case,slice(None))
            self.reset()
            scale = gain.rsqrt().repeat_interleave(self.params[1].numel()//self.h)
            output = self.last.attention.output.dense.weight
            original_output = output.clone()
            self.params[0].mul_(scale[:,None])
            self.params[1].mul_(scale)
            output.div_(scale[None,:])
            pre = self.read(case,slice(None))
        grad = self.gradient(case,slice(0,1),torch.ones(1))
        self.write(grad,torch.ones_like(gain),lr)
        with torch.no_grad():
            post = self.read(case,slice(None))
            output.copy_(original_output)
            self.reset()
        a = float((pre-case[-1]).abs().max())
        b = float((post-reference).abs().max())
        assert max(a,b)<1e-4
        torch.testing.assert_close(post>=0,reference>=0)
        return a,b


def truth(genuine):
    return torch.tensor([1.,0.,float(genuine),float(not genuine)]).repeat_interleave(8)


def counts(correct,correct0):
    wrong = ~correct0
    return [correct[16:].sum().item(),correct[:16].sum().item(),(correct&wrong).sum().item(),
            wrong.sum().item(),(~correct&correct0).sum().item(),correct0.sum().item(),correct.sum().item()]


def rates(a):
    return np.stack([a[...,0]/16,a[...,1]/16,a[...,2]/a[...,3],a[...,4]/a[...,5],a[...,6]/32],axis=-1)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    exp = Experiment()
    practice,test = sorted(exp.domains)[::2],sorted(exp.domains)[1::2]
    rng = np.random.default_rng(27711)
    agree = np.stack([exp.agreement(exp.episode(d,"val",rng)) for d in practice for _ in range(8)])
    gains = {}
    for name,p in (("change",.8),("exception",.2)):
        score = (p*agree[:,0]+(1-p)*agree[:,1]).mean(0)
        gain = torch.ones(exp.h)
        low,_,high = np.split(np.argsort(score),3)
        gain[low.tolist()] = .25
        gain[high.tolist()] = 4
        gains[name] = gain
    gains["ordinary"] = torch.ones(exp.h)
    gains["permuted"] = gains["change"][rng.permutation(exp.h).tolist()]
    gains["matched"] = gains["change"]
    lrs = (.03,.1,.3,1.)
    grid = np.zeros(len(lrs))
    for d in practice:
        for _ in range(4):
            case = exp.episode(d,"val",rng)
            for i,lr in enumerate(lrs):
                for m in ("ordinary","change"):
                    grid[i] += exp.evaluate(case,gains[m],lr)[:,1:,-1].mean()/(32*40)
    lr = lrs[grid.argmax()]
    print("tuned",lr,grid.tolist(),flush=True)
    results = []
    equality = []
    for d in test:
        domain = []
        for i in range(12):
            case = exp.episode(d,"test",rng)
            if i == 0:
                equality.append(exp.literal_check(case,gains["change"],lr))
            domain.append([exp.evaluate(case,gains[m],lr,m=="matched") for m in GAINS])
        results.append(domain)
        print("finished",d,flush=True)
    a = np.array(results)
    np.testing.assert_array_equal(a[:,:,:,:,0],np.broadcast_to(a[:,:,:1,:,0],a[:,:,:,:,0].shape))
    np.savez_compressed(ROOT/"value-history-trials.npz",counts=a,practice_agreements=agree)
    bootrng = np.random.default_rng(27712)
    w = [bootrng.multinomial(12,[1/12]*12,size=2048)/12 for _ in test]
    boot = sum(np.tensordot(v,b,axes=(1,0)) for v,b in zip(w,a))/5
    p,b = rates(a.mean((0,1))), rates(boot)
    effects = {}
    for j,name in enumerate(GAINS[1:],1):
        delta,draw = p[0]-p[j],b[:,0]-b[:,j]
        effects[name] = np.stack([delta,*np.quantile(draw,[.025,.975],axis=0)],axis=-1).tolist()
    out = dict(methods=GAINS, metrics=METRICS, worlds=["true","false"],steps=[0,1,4],
               practice=practice,test=test,gains={k:v.tolist() for k,v in gains.items()}, skipped=exp.skipped,
               learning_rate=lr, tuning=grid.tolist(), equality=equality, means=p.tolist(),
               domain_means=rates(a.mean(1)).tolist(),change_minus=effects)
    (ROOT/"value-history.json").write_text(json.dumps(out,indent=1,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
