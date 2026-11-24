"""Prospective comparison of prediction loss and teaching cost with matched writable bytes."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

from analyse import interval
from learner import History, COUNT, correct, features, observe, softmax, teacher
from study import OUT, config, save_json, threshold
from world import case

KEYS = ('frozen','kernel','neighbor','weights','bounded')


class Bounded:
    def __init__(self, mix):
        self.mix = mix
        self.keys = np.empty((0,128), dtype=np.int8)
        self.labels = np.empty(0,dtype=np.uint8)

    def save(self,x,logits,field):
        h=x[:-1]
        key=np.rint(127*h/np.linalg.norm(h)).astype(np.int8)
        self.keys = np.vstack((self.keys[-31:],key))
        self.labels = np.append(self.labels[-31:],np.uint8(field.argmax()))
        assert self.keys.nbytes+self.labels.nbytes<=4128

    def prior(self,x,logits):
        if not len(self.labels):
            return softmax(logits)
        values=self.keys.astype(float)
        sim=(values@x[:-1])/np.linalg.norm(values,axis=1)
        nearest=self.labels[sim.argmax()]
        p=(1-self.mix)*softmax(logits)
        p[nearest]+=self.mix
        return p


def measure(prior, matrix, target, trace, corrupt):
    w=np.repeat(prior/4,4)
    scores=[]
    bits=[]
    for i,idx in enumerate(trace):
        scores.append(correct(w,matrix[:,COUNT:],matrix[target,COUNT:]))
        answer=int(matrix[target,idx]) ^ int(corrupt and i==0)
        likelihood=np.where(matrix[:,idx]==answer,.95,.05)
        prob=w@likelihood
        assert .05-1e-8<=prob<=.95+1e-8
        bits.append(-np.log2(prob))
        w=observe(w,matrix[:,:COUNT],idx,answer)
        w/=w.sum()
    scores.append(correct(w,matrix[:,COUNT:],matrix[target,COUNT:]))
    return bits,scores


def one(seed,domain):
    settings=config('frozen.json')['settings']
    stores=[History(k,settings[k]) for k in KEYS[:-1]]+[Bounded(settings['neighbor']['mix'])]
    vec=features(np.load(OUT/'encoder.npz'))
    rng=np.random.default_rng(seed)
    perm=rng.permutation(4)
    for _ in range(96):
        idx,target,matrix=case(rng,'practice',rng.integers(4),perm)
        _,field,_=teacher(target,matrix,practice=True)
        if field is not None:
            x,logits=(v[idx] for v in vec['practice'])
            for store in stores:
                store.save(x,logits,field)
    bits=np.empty((8,len(KEYS),2,6))
    correct_counts=np.empty((8,len(KEYS),2,7),dtype=np.uint8)
    for ti,g in enumerate(rng.permutation(np.tile(np.arange(4),2))):
        idx,target,mat=case(rng,domain,g,perm)
        trace,_,_=teacher(target,mat)
        x,logits=(v[idx] for v in vec[domain])
        for k,store in enumerate(stores):
            prior=store.prior(x,logits)
            for c in (0,1):
                bits[ti,k,c],correct_counts[ti,k,c]=measure(prior,mat,target,trace,bool(c))
    return bits,correct_counts


def main():
    summary={}
    for domain,start in [('number',1010000),('text',1110000)]:
        with ProcessPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(partial(one,domain=domain),range(start,start+96),chunksize=2))
        bits,counts = (np.stack(x) for x in zip(*rows))
        np.savez_compressed(OUT/f'{domain}-bits.npz',bits=bits,correct=counts)
        examples=threshold(counts[:,:,:,0]).mean(1)
        loss=bits[:,:,:,0].mean((1,3))
        clean_recovery=bits[:,:,:,0,1:].mean((1,3))
        recovery=bits[:,:,:,1,1:].mean((1,3))
        final=counts[:,:,:,1,-1].mean(1)/96
        half=(examples[:,0]-examples[:,-1])-.5*(examples[:,0]-examples[:,2])
        summary[domain]=dict(half_gain_margin=interval(half),
                            weights_minus_bounded_examples=interval(examples[:,3]-examples[:,4]),
                            weights_minus_bounded_bits=interval(loss[:,3]-loss[:,4]),
                            methods={key:dict(clean_bits=interval(loss[:,k]),examples=interval(examples[:,k]),
                                              recovery_bits=interval(recovery[:,k]),
                                              increase_after_bad_first=interval(recovery[:,k]-clean_recovery[:,k]),
                                              accuracy_after_bad_first=interval(final[:,k]))
                                     for k,key in enumerate(KEYS)})
        print(domain,summary[domain],flush=True)
    save_json('bits-summary.json',summary)


if __name__=='__main__':
    main()
