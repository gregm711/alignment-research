"""Differentiate through edits to an attention value matrix; intercede on practice history."""
import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"intent-learning"))
import policy_change as p

A,B = p.PAIRS,p.OTHER
TIMES = (0,48,60,96)
BUDGETS = (1,2,4,8)
WIDTH = 16


def words(text):
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]",text.lower())


def corpus(names,rng):
    data = json.loads((p.CACHE/"data_full.json").read_text())
    pools = {}
    seen = set()
    by_split = {}
    for split in ("test","val","train"):
        by_split[split] = [[] for _ in names]
        for text,name in data[split]:
            if name not in names:
                continue
            key = " ".join(text.lower().split())
            if key not in seen:
                by_split[split][names.index(name)].append(words(text))
                seen.add(key)
    freq = Counter(w for rows in by_split["train"] for sentence in rows for w in sentence)
    vocab = {w:i+3 for i,w in enumerate(sorted(freq))}
    tokens = []
    for part,classes in by_split.items():
        pools[part] = []
        for examples in classes:
            ids = []
            for example in examples:
                seq = [2]+[vocab.get(w,1) for w in example[:19]]
                ids.append(len(tokens))
                tokens.append(seq+[0]*(20-len(seq)))
            pools[part].append(rng.permutation(ids))
    return torch.tensor(tokens),pools,vocab


class Editor(nn.Module):
    def __init__(self,nwords):
        super().__init__()
        self.word = nn.Embedding(nwords,WIDTH,padding_idx=0)
        self.position = nn.Parameter(torch.randn(20,WIDTH)*.1)
        self.block = nn.TransformerEncoderLayer(WIDTH,2,32,dropout=0,batch_first=True)
        self.intents = nn.Linear(WIDTH,6)
        self.key = nn.Linear(WIDTH,WIDTH)
        self.query = nn.Linear(WIDTH,WIDTH)
        self.register_buffer("read",F.normalize(torch.linspace(-1,1,WIDTH),dim=0))
        self.register_buffer("value",torch.zeros(2,WIDTH,WIDTH//2))

    def context(self,ids):
        return self.block(self.word(ids)+self.position,src_key_padding_mask=(ids==0))

    def inputs(self,ids):
        h = self.context(ids)
        keys = self.key(h).reshape(len(h),20,2,-1)
        query = self.query(h[:,0]).reshape(len(h),2,-1)
        scores = (keys*query[:,None]).sum(-1)/(WIDTH//2)**.5
        weights = scores.masked_fill(ids[:,:,None]==0,-1e4).softmax(dim=1)
        return F.normalize(torch.einsum("bth,btd->bhd",weights,h),dim=-1)

    def forward(self,ids):
        h = self.inputs(ids)
        return (torch.einsum("bhd,hdk->bhk",h,self.value).tanh().flatten(1)@self.read)-1


def logits(h,value,read,bias=-1):
    hidden = torch.einsum("bnhd,bhdk->bnhk",h,value).tanh()
    return hidden.flatten(2)@read+bias


def write(h,y,value,read,bias=-1):
    activation = torch.einsum("bhd,bhdk->bhk",h,value).tanh()
    prediction = (activation.flatten(1)@read+bias).sigmoid()
    delta = (y-prediction)[:,None,None]*(1-activation.square())*read.reshape(2,-1)
    return value+4*h[:,:,:,None]*delta[:,:,None,:]


def pretrain(net,tokens,pools,rng):
    opt = torch.optim.Adam(net.parameters(),lr=.004)
    for _ in range(200):
        labels = rng.integers(6,size=32)
        ids = [int(rng.choice(pools["train"][c])) for c in labels]
        loss = F.cross_entropy(net.intents(net.context(tokens[ids])[:,0]),torch.tensor(labels.tolist()))
        opt.zero_grad()
        loss.backward()
        opt.step()


def packed(net,tokens,ids):
    flat = tokens[ids.ravel().tolist()]
    return net.inputs(flat).reshape(*ids.shape,2,WIDTH)


def train_step(net,opt,tokens,support,query,label,qtruth):
    n,k = support.shape
    x = packed(net,tokens,np.column_stack([support,query]))
    value = net.value.expand(n,-1,-1,-1)
    y = torch.tensor(label.tolist(),dtype=torch.float32)
    target = torch.tensor(qtruth.tolist(),dtype=torch.float32)
    losses = []
    for j in range(k):
        value = write(x[:,j],y[:,j],value,net.read)
        if j+1 in BUDGETS:
            losses.append(F.binary_cross_entropy_with_logits(logits(x[:,k:],value,net.read),target))
    opt.zero_grad()
    torch.stack(losses).mean().backward()
    opt.step()


def observer(prob,ids,labels,qids):
    masks = p.MASK
    rate = .85/3
    prior = rate**masks.sum(1)*(1-rate)**(6-masks.sum(1))
    weights = np.tile(prior,(len(ids),1))
    result = []
    for j in range(ids.shape[1]):
        yes = .1+.8*prob[ids[:,j]]@masks.T
        weights = p.normalize(weights*np.where(labels[:,j,None],yes,1-yes))
        if j+1 in BUDGETS:
            result.append((weights@masks)@prob[qids].T>=.5)
    return np.array(result)


def test_set(rng,pools):
    support = np.stack([a[:12] for a in pools["test"]])
    query = np.stack([a[12:24] for a in pools["test"]])
    assert len(np.unique(np.r_[support.ravel(),query.ravel()]))==144
    cases = []
    for nchange in range(3):
        n = 96
        sc = rng.integers(6,size=(n,8))
        idx = support[sc,np.arange(8)]
        order = rng.random((n,3)).argsort(1)[:,:nchange]
        noise = rng.random((n,8))<.1
        group = []
        for pairs in (A,B):
            mask = np.zeros((n,6),bool)
            mask[np.arange(n)[:,None,None],pairs[order]] = True
            group.append((idx,np.take_along_axis(mask,sc,axis=1)^noise,np.repeat(mask,12,axis=1)))
        cases.append(group)
    tr = np.tile(np.arange(6),12)
    pos = np.repeat(np.arange(12),6)
    check_noise = rng.random(len(tr))<.1
    return support,query,cases,(tr,pos,check_noise)


def evaluate(net,tokens,cases):
    support,query,groups,probe = cases
    ids = np.r_[support.ravel(),query.ravel()]
    h = torch.zeros(len(tokens),2,WIDTH)
    h[ids.tolist()] = net.inputs(tokens[ids.tolist()])
    qh = h[query.ravel().tolist()]
    scores = []
    for group in groups:
        row = []
        for idx,labels,truth in group:
            v = net.value.expand(len(idx),-1,-1,-1)
            curve = []
            for j in range(8):
                v = write(h[idx[:,j].tolist()],torch.tensor(labels[:,j].tolist(),dtype=torch.float32),v,net.read)
                if j+1 in BUDGETS:
                    z = logits(qh.expand(len(idx),-1,-1,-1),v,net.read)
                    curve.append(float(((z>=0)==torch.tensor(truth.tolist())).float().mean()))
            row.append(curve)
        scores.append(row)
    trigger,pos,noise = probe
    n = len(trigger)
    value = write(h[support[trigger,pos].tolist()],torch.ones(n),net.value.expand(n,-1,-1,-1),net.read)
    shifts = logits(qh.expand(n,-1,-1,-1),value,net.read).reshape(n,6,12)+1
    influence,feedback = [],[]
    for pairs in (A,B):
        peer = dict([*pairs.tolist(),*pairs[:,::-1].tolist()])
        target = np.array([peer[c] for c in trigger])
        row = torch.arange(n)
        influence.append(float(shifts[row,target.tolist()].mean()))
        keys = h[query[target].tolist()]
        check = h[support[target,(pos+1)%12].tolist()]
        rates = []
        for noisy in (False,True):
            forks = []
            for positive in (True,False):
                y = torch.tensor((np.full(n,positive)^(noise & noisy)).tolist())
                updated = write(check,y.float(),value,net.read)
                forks.append(float((logits(keys,updated,net.read)>=0).float().mean()))
            rates.append(forks)
        first = float((logits(keys,value,net.read)>=0).float().mean())
        feedback.append([[first,first],*rates])
    others = [sorted(set(range(6))-{int(c),int(A.ravel()[c^1]),int(dict([*B.tolist(),*B[:,::-1].tolist()])[c])}) for c in trigger]
    unrelated = float(shifts[torch.arange(n)[:,None],torch.tensor(others)].mean())
    return scores,[*influence,unrelated],feedback,value[0],support[trigger[0],pos[0]]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domains = json.loads((p.CACHE/"domains.json").read_text())
    rows, exported = [],{}
    for i,seed in enumerate(range(1101,1111)):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        domain = list(domains)[i]
        names = rng.choice(domains[domain],6,replace=False).tolist()
        tokens,pools,vocab = corpus(names,rng)
        cases = test_set(np.random.default_rng(seed+5000),pools)
        net = Editor(len(vocab)+3)
        pretrain(net,tokens,pools,rng)
        with torch.no_grad():
            ids = np.r_[cases[0].ravel(),cases[1].ravel()]
            prob = np.zeros((len(tokens),6))
            prob[ids] = np.array(net.intents(net.context(tokens[ids.tolist()])[:,0]).softmax(-1).tolist())
            semantic = float((prob[ids].argmax(1)==np.tile(np.repeat(np.arange(6),12),2)).mean())
            ref = [[[float((pred==truth).mean()) for pred in observer(prob,idx,y,cases[1].ravel())]
                    for idx,y,truth in group] for group in cases[2]]
        nets = [copy.deepcopy(net) for _ in range(3)]
        opts = [torch.optim.Adam(m.parameters(),lr=.002) for m in nets]
        row = dict(seed=seed,domain=domain,intents=names,semantic=semantic,reference=ref,checkpoints={})
        for t in range(97):
            if t in TIMES:
                with torch.no_grad():
                    checkpoint = []
                    for j,m in enumerate(nets):
                        scores,inf,fb,v,trigger_id = evaluate(m,tokens,cases)
                        checkpoint.append(dict(scores=scores,influence=inf,feedback=fb))
                        if t==96 and j==0:
                            for name,param in m.state_dict().items():
                                saved = v if name=="value" else param
                                exported[f"{seed}/{name}"] = np.asarray(saved.tolist(),dtype=np.float32)
                            qids = cases[1].ravel()
                            exported[f"{seed}/inputs"] = np.array(tokens[qids.tolist()].tolist())
                            exported[f"{seed}/trigger"] = np.array(tokens[int(trigger_id)].tolist())
                            exported[f"{seed}/logits"] = np.array(logits(m.inputs(tokens[qids.tolist()])[None],v[None],m.read)[0].tolist())
                    row["checkpoints"][t] = checkpoint
            if t==96:
                break
            sc = rng.integers(6,size=(8,4))
            qc = np.tile(np.arange(6),(8,1))
            support = np.array([[rng.choice(pools["train"][c][:48]) for c in row_c] for row_c in sc])
            query = np.array([[rng.choice(pools["train"][c][48:]) for c in row_c] for row_c in qc])
            active = rng.random(8)<.85
            which = rng.integers(3,size=8)
            random_pairs = rng.random((8,6)).argsort(1).reshape(8,3,2)
            noise = rng.random(sc.shape)<.1
            for j,(m,opt) in enumerate(zip(nets,opts)):
                pair = random_pairs[np.arange(8),which] if j==2 else (A,B)[j^(t>=48)][which]
                mask = np.zeros((8,6),bool)
                mask[np.arange(8)[:,None],pair] = active[:,None]
                train_step(m,opt,tokens,support,query,np.take_along_axis(mask,sc,axis=1)^noise,mask)
        rows.append(row)
        print("seed",seed,"finished",flush=True)
    (ROOT/"attention-results.json").write_text(json.dumps(rows,indent=1)+"\n")
    np.savez_compressed(ROOT/"attention-checkpoints.npz",**exported)


if __name__=="__main__":
    main()
