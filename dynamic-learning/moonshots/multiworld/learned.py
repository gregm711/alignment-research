"""Fit four soft categorical slots and score their serialized hard states after restart."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from study import NAMES, TRUTH, evaluate_case, pack


class Writer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(62,64), nn.Tanh(), nn.Linear(64,132))

    def forward(self,x):
        v = self.layers(x)
        return F.log_softmax(v[:,:128].reshape(-1,4,32),dim=2), F.log_softmax(v[:,128:],dim=1)


def batches(seed):
    rng = np.random.default_rng(seed)
    for _ in range(800):
        world = rng.integers(32,size=128)
        sensors = rng.integers(31,size=(128,6))
        bits = TRUTH[sensors,world[:,None]] ^ (rng.random((128,6))<.125)
        x = np.zeros((128,62),np.float32)
        np.add.at(x,(np.arange(128)[:,None],2*sensors+bits),1)
        yield torch.from_numpy(x), torch.from_numpy(world)


def train(seed, penalty):
    torch.manual_seed(seed)
    writer = Writer()
    opt = torch.optim.Adam(writer.parameters(),lr=.004)
    for x,world in batches(seed):
        slots,weights = writer(x)
        logp = torch.logsumexp(slots+weights.unsqueeze(2),dim=1)
        loss = F.nll_loss(logp,world)
        if penalty:
            probabilities = slots.exp()
            overlap = sum((probabilities[:,i]*probabilities[:,j]).sum(1).mean()
                          for i in range(4) for j in range(i))
            loss = loss+penalty*overlap/6
        opt.zero_grad()
        loss.backward()
        opt.step()
    return writer.eval()


def main(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    writers = [train(seed,penalty) for penalty in (0.,.1) for seed in (401,402,403)]

    @torch.no_grad()
    def write(reports):
        x = torch.tensor(np.bincount(reports,minlength=62)[None],dtype=torch.float32)
        blobs = []
        for writer in writers:
            slots,weights = writer(x)
            worlds = slots[0].argmax(1).numpy()
            relative = torch.exp(weights[0]-weights.max())
            quantized = np.maximum(1,np.rint(7*relative.numpy())).astype(int)
            blobs.append(pack(worlds+32*quantized,8))
        return blobs

    rows = [evaluate_case(seed,write) for seed in range(952000,952512)]
    scores = np.array([r['accuracy'] for r in rows])[:,:,0]
    learnt = scores[:,len(NAMES):].reshape(-1,2,3)
    distinct = np.array([[len({b%32 for b in bytes.fromhex(blob)}) for blob in r['states'][len(NAMES):]]
                         for r in rows]).reshape(-1,2,3)
    rng = np.random.default_rng(947)
    draws = rng.integers(512,size=(4000,512))
    def ci(delta):
        return [delta.mean(),*np.quantile(delta[draws].mean(1),[.025,.975])]
    report = dict(torch=str(torch.__version__),cases=len(rows),accuracy=learnt.mean(0).tolist(),
                  mean_distinct_modes=distinct.mean(0).tolist(), analytic_branches=float(scores[:,0].mean()),
                  diversity_gain=ci(learnt[:,1,:].mean(1)-learnt[:,0,:].mean(1)),
                  analytical_advantage=ci(scores[:,0]-learnt[:,1,:].mean(1)))
    out.mkdir(exist_ok=True,parents=True)
    data = ''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode()
    (out/'learned-cases.jsonl.gz').write_bytes(gzip.compress(data,mtime=0))
    (out/'learned-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    main(p.parse_args().out)
