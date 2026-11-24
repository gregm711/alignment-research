"""Compare independently trained four-slot writers with a general four-byte learned code."""
import argparse
import gzip
import json
from pathlib import Path
from time import process_time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learned import batches, train
from study import NAMES, TRUTH, UNIFORM, archive, evaluate_case, first_max, joint_distribution, likelihood, pack, recover


class BinaryWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(62,121),nn.Tanh(),nn.Linear(121,32))
        self.decoder = nn.Linear(32,32)


def train_binary(seed):
    torch.manual_seed(seed)
    net = BinaryWriter()
    opt = torch.optim.Adam(net.parameters(),lr=.004)
    for x,world in batches(seed):
        soft = net.encoder(x).sigmoid()
        bits = (soft>=.5).float()+(soft-soft.detach())
        loss = F.cross_entropy(net.decoder(bits),world)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def write(net,reports):
    x = torch.from_numpy(np.bincount(reports,minlength=62)[None].astype(np.float32))
    if isinstance(net,BinaryWriter):
        bits = (net.encoder(x)[0]>=0).numpy()
        return np.packbits(bits,bitorder='little').tobytes()
    slots,gate = net(x)
    worlds = slots[0].argmax(1).numpy()
    counts = np.maximum(1,np.rint(7*(gate[0]-gate.max()).exp().numpy())).astype(int)
    return pack(worlds+32*counts,8)


def read(net,blob):
    if not isinstance(net,BinaryWriter):
        return recover(blob,'branches')
    bits = np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little').astype(np.float32)
    p = net.decoder(torch.from_numpy(bits)).softmax(0).numpy()
    return .99*p+.01*UNIFORM


def calibration(nets):
    seeds = np.arange(954000,954128)
    cases = [evaluate_case(s) for s in seeds]
    rng = np.random.default_rng(948)
    values = {k:[] for k in ('slots','binary','exact')}
    for _ in range(3):
        for mode in rng.permutation(list(values)):
            start = process_time()
            for case in cases:
                reports = np.asarray(case['reports'])
                if mode=='exact':
                    # Exact implementation needs only the rank, not the other writers.
                    from math import comb
                    rank = sum(comb(int(v)+i,i+1) for i,v in enumerate(sorted(reports)))
                    restored = recover(rank.to_bytes(4,'little'),'exact')
                else:
                    net = nets[3 if mode=='slots' else 6]
                    restored = read(net,write(net,reports))
                mask = TRUTH[case['calibration']]
                q = restored*mask
                j = joint_distribution(q/q.sum(),case['target'],np.array(case['menu']))
                first_max(j.max(2).sum(1))
            values[mode].append(1e6*(process_time()-start)/len(cases))
    return values


def main(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    nets, timing = [], []
    for method in ('plain','diverse','binary'):
        for seed in (401,402,403):
            start = process_time()
            nets.append(train_binary(seed) if method=='binary' else train(seed,.1 if method=='diverse' else 0))
            timing.append(process_time()-start)
    params = [sum(p.numel() for p in n.parameters()) for n in nets]
    assert params==[12612]*6+[12583]*3
    with torch.no_grad():
        def writer(reports):
            return [write(net,reports) for net in nets]
        def reader(i,blob):
            return read(nets[i],blob)
        rows = [evaluate_case(seed,writer,reader) for seed in range(953000,953512)]
        clock = calibration(nets)
    a = np.array([r['accuracy'] for r in rows])
    learned = a[:,len(NAMES):].reshape(-1,3,3,4)
    draw = np.random.default_rng(947).integers(512,size=(4000,512))
    def interval(delta):
        return [float(delta.mean()),*np.quantile(delta[draw].mean(1),[.025,.975])]
    comparisons = {}
    for j in (0,2):
        contrasts = learned[:,:,:,j].mean(2)
        comparisons[['own_probe','replay'][j//2]] = {
            'diverse_minus_binary':interval(contrasts[:,1]-contrasts[:,2]),
            'exact_minus_diverse':interval(a[:,4,j]-contrasts[:,1])}
    summary = dict(torch=str(torch.__version__),names=['plain','diverse','binary'],seeds=[401,402,403],
                   parameters=params,training_cpu_seconds=timing,inference_cpu_us=clock,
                   multiply_adds={'slots':62*64+64*132,'binary':62*121+121*32+32*32},
                   accuracy=learned.mean(0).tolist(),analytic_accuracy=a[:,:len(NAMES)].mean(0).tolist(),comparisons=comparisons)
    out.mkdir(exist_ok=True,parents=True)
    raw = ''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode()
    (out/'bottleneck-cases.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    (out/'bottleneck-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['comparisons']),json.dumps(clock))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    main(parser.parse_args().out)
