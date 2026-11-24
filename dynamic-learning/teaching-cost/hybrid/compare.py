"""Split an unchanged capacity between two frozen memory mechanisms."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bits import Bounded, measure
from analyse import interval
from learner import History, features, teacher
from study import OUT, config, threshold
from world import case

KEYS = ('episodes32','weights64','episodes16','weights32','hybrid')


def evaluate(domain, start):
    vec = features(np.load(OUT/'encoder.npz'))
    settings = config('frozen.json')['settings']
    bits = np.empty((96,8,5,2,6))
    scores = np.empty((96,8,5,2,7),dtype=np.uint8)
    for i in range(96):
        rng = np.random.default_rng(start+i)
        perm = rng.permutation(4)
        archive = Bounded(settings['neighbor']['mix'])
        full = History('weights',settings['weights'])
        half = History('weights',settings['weights'])
        half.delta = half.delta.astype(np.float32)
        for _ in range(96):
            idx,target,mat = case(rng,'practice',rng.integers(4),perm)
            _,field,_ = teacher(target,mat,practice=True)
            if field is not None:
                for store in (archive,full,half):
                    store.save(vec['practice'][0][idx],vec['practice'][1][idx],field)
        short = Bounded(settings['neighbor']['mix'])
        short.keys,short.labels = archive.keys[-16:].copy(),archive.labels[-16:].copy()
        assert short.keys.nbytes+short.labels.nbytes+half.delta.nbytes<=4128
        assert half.delta.dtype==np.float32 and full.delta.nbytes==4128
        for t,g in enumerate(rng.permutation(np.tile(np.arange(4),2))):
            idx,target,mat = case(rng,domain,g,perm)
            trace,_,_ = teacher(target,mat)
            x,logits = (v[idx] for v in vec[domain])
            ps = [s.prior(x,logits) for s in (archive,full,short,half)]
            ps.append((ps[2]+ps[3])/2)
            for k,p in enumerate(ps):
                assert np.isclose(p.sum(),1)
                for corrupt in (0,1):
                    bits[i,t,k,corrupt],scores[i,t,k,corrupt]=measure(p,mat,target,trace,corrupt)
    np.savez_compressed(HERE/f'{domain}.npz', bits=bits,correct=scores)
    examples=threshold(scores[:,:,:,0]).mean(1)
    return {k:dict(examples=interval(examples[:,j]),
                   savings_over_this_arm=interval(examples[:,j]-examples[:,-1]),
                   bits=interval(bits[:,:,j,0].mean((1,2))),
                   accuracy_after_wrong=interval(scores[:,:,j,1,-1].mean(1)/96))
            for j,k in enumerate(KEYS)}


if __name__=='__main__':
    results={d:evaluate(d,s) for d,s in [('number',1210000),('text',1310000)]}
    (HERE/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for d,r in results.items():
        print(d,[(k,v['savings_over_this_arm']) for k,v in r.items()])
