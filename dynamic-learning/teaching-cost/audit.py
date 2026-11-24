"""Independently re-execute every test target and check sample counts, replay, and overlap."""
import numpy as np
from learner import CUTS, METHODS, History, features
from study import OUT, config, save_json, threshold
from world import TABLES, WORDS, case


def run(domain, seed, archive, frozen, vectors):
    keys = [*METHODS,frozen['winner']+'_reset',frozen['winner']+'_shuffle']
    rng=np.random.default_rng(seed)
    perm=rng.permutation(4)
    swap=rng.choice(4,2,replace=False)
    stale=perm.copy()
    stale[swap]=stale[swap[::-1]]
    # Independently recover the actual practice labels, then the learned priors.
    receipts=[]
    for _ in range(96):
        idx,t,m=case(rng,'practice',int(rng.integers(4)),perm)
        odds=np.ones(16)
        selected=[]
        for step in range(6):
            post=odds/odds.sum()
            field=post.reshape((4,4)).sum(1)
            if len(selected)>=2 and field.max()>=.85:
                receipts.append((idx,field))
                break
            if step==5:
                break
            value=np.abs(post@m[:,:16]-.5)
            value[selected]=np.inf
            j=int(value.argmin())
            selected.append(j)
            odds*=np.where(m[:,j]==m[t,j],.95,.05)
    stores=[]
    for k in keys:
        method=k.split('_')[0]
        store=History(method,frozen['settings'][method])
        labels=[y for _,y in receipts]
        if k.endswith('_shuffle'):
            labels=[labels[i] for i in np.random.default_rng(seed+10000).permutation(len(labels))]
        if not k.endswith('_reset'):
            for (idx,_),label in zip(receipts,labels):
                store.save(vectors['practice'][0][idx],vectors['practice'][1][idx],label)
        stores.append(store)
    # Validate the complete execution tables on the original numeric / character representation.
    for v in range(8):
        for k in range(4):
            if domain=='number':
                actual=sum(int(bit) for bit in format(v^k,'03b'))>=2
            else:
                s=WORDS[v]
                actual=(s[0]==s[1],s[0]==s[2],s[1]==s[2],len([x for x in s if x=='b'])%2==1)[k]
            assert actual==TABLES[domain][k,v]
    masks=[]
    for ti,g in enumerate(rng.permutation(np.tile(np.arange(4),2))):
        idx,t,mat=case(rng,domain,g,perm)
        a=4*stale[g]+t%4
        assert list(archive['targets'][ti])==[idx,t,a]
        masks.append(t%4)
        reference=np.ones(16)
        probes=[]
        for _ in range(6):
            diff=np.abs((reference/reference.sum()) @ mat[:,:16]-.5)
            diff[probes]=np.inf
            q=int(diff.argmin())
            probes.append(q)
            reference*=np.where(mat[:,q]==mat[t,q],.95,.05)
        for ki,store in enumerate(stores):
            p=np.repeat(store.prior(vectors[domain][0][idx],vectors[domain][1][idx])/4,4)
            for step in range(7):
                accuracy=int(np.sum((p@mat[:,16:]>=.5)==mat[t,16:]))
                assert archive['common'][-1,ki,ti,step]==accuracy
                if step<6:
                    q=probes[step]
                    p*=np.where(mat[:,q]==mat[t,q],.95,.05)
                    p/=p.sum()
    return masks


def main():
    frozen=config('frozen.json')
    vectors=features(np.load(OUT/'encoder.npz'))
    output={}
    for domain,start in [('number',710000),('text',810000)]:
        with np.load(OUT/f'{domain}-lifetimes.npz') as z:
            arrays={k:z[k] for k in ['common','targets']}
        masks=[]
        for i in range(96):
            row={k:v[i] for k,v in arrays.items()}
            masks.append(run(domain,start+i,row,frozen,vectors))
        overlaps=[i for i,r in enumerate(TABLES[domain]) if
                  any(np.array_equal(1-r,old) for old in TABLES['practice'])]
        keep=~np.isin(masks,overlaps)
        examples=threshold(arrays['common'])
        neighbor=METHODS.index('neighbor')
        weight=METHODS.index('weights')
        delta=examples[:,0,neighbor]-examples[:,-1,neighbor]
        delta_weight=examples[:,-1,weight]-examples[:,-1,neighbor]
        def paired(d):
            sums=(keep*d).sum(1)
            counts=keep.sum(1)
            draws=np.random.default_rng(90).integers(0,96,(4000,96))
            q=sums[draws].sum(1)/counts[draws].sum(1)
            return [sums.sum()/counts.sum(),*np.quantile(q,[.025,.975]).tolist()]
        output[domain]=dict(overlapping_complement_indices=overlaps,
                            remaining_tasks=int(keep.sum()),neighbor_gain_after_exclusion=paired(delta),
                            neighbor_over_weights_after_exclusion=paired(delta_weight))
    save_json('audit.json',output)
    print(output)


if __name__=='__main__':
    main()
