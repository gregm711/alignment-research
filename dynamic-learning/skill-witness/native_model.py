"""Fixed-state, meta-trained revision reader and check selector. Inference requires only NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / 'native-weights.npz'
K = 8
DRIFT = .22
NOISE = .08
FEATURES = 5


def probability(state):
    a = state.astype(np.float32)
    a = np.exp(a - a.max(axis=-1, keepdims=True))
    return a/a.sum(axis=-1, keepdims=True)


def partition_features(p, eq):
    mass = eq @ p
    splits = 1.-eq
    future = (splits.sum(0)-splits)/max(1,len(eq)-1)
    pairs = np.outer(p,p)
    return np.column_stack((-(np.log(mass.clip(1e-9))*p).sum(-1)/np.log(K),
                            (splits*pairs).sum((1,2)), (splits*pairs*future).sum((1,2)),
                            1.-mass.max(-1), (1/eq.sum(-1)).sum(-1)/K))


class NativeModel:
    def __init__(self, params):
        self.params = params

    def initial(self):
        return np.zeros(K,dtype=np.float16)

    def begin(self,state):
        drift = float(self.params['drift'])
        p = (1.-drift)*probability(state) + drift/K
        a = np.log(p)
        return (a-a.mean()).astype(np.float16)

    def prior(self,state):
        return probability(state)

    def update(self,state,match,q_features=None):
        match=np.asarray(match,dtype=bool)
        if match.shape!=(K,) or np.asarray(state).shape!=(K,):
            raise ValueError('expected eight slots and eight compatibility bits')
        selected = match if match.sum()<=K//2 else ~match
        new = np.asarray(state,dtype=np.float16).copy()
        if selected.any():
            p = self.prior(new)
            x = np.column_stack((p, np.full(K,p@match), np.full(K,match.mean()),
                                 np.full(K,-(p*np.log(p.clip(1e-9))).sum()/np.log(K))))
            d=self.params
            hidden=np.tanh(x@d['w']+d['bias'])
            amount = np.logaddexp(0.,hidden@d['v']+d['intercept'])
            new[selected] = np.clip(new[selected]+(1 if match.sum()<=4 else -1)*amount[selected],-24,24)
        return new

    def choose(self,state,signatures,used=()):
        signatures=np.asarray(signatures)
        if signatures.ndim!=2 or signatures.shape[1]!=K:
            raise ValueError('expected one row per probe and eight hypotheses')
        unavailable=np.zeros(len(signatures),dtype=bool)
        supplied=np.asarray(list(used))
        if supplied.dtype==bool:
            if supplied.shape!=unavailable.shape:
                raise ValueError('wrong probe mask length')
            unavailable[:]=supplied
        elif supplied.size:
            unavailable[supplied.astype(int)]=True
        ids=np.flatnonzero(~unavailable)
        if not len(ids):
            raise ValueError('no available checks')
        S=signatures[ids]
        f=partition_features(self.prior(state), S[:,:,None]==S[:,None,:])
        return int(ids[np.argmax(f@self.params['query'])])


def load_model(path=WEIGHTS):
    with np.load(path, allow_pickle=False) as p:
        return NativeModel({name:p[name] for name in p.files})


def stream_step(rng,world):
    b=len(world)
    world=np.where(rng.random(b)<DRIFT,rng.integers(K,size=b),world)
    # Different tasks emphasize different families of output distinctions.
    prototypes=rng.integers(0,6,(b,4,K)) % rng.integers(2,5,(b,4,1))
    probs=rng.dirichlet([.65]*4,size=b)
    ix=(rng.random((b,14,1))>probs.cumsum(1)[:,None,:]).sum(2)
    rows=prototypes[np.arange(b)[:,None],ix]
    noisy=np.where(rng.random((b,6))<NOISE,rng.integers(K,size=(b,6)),world[:,None])
    return world,rows,noisy


def train(steps=180):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(271)
    rng=np.random.default_rng(272)
    w=nn.Parameter(torch.randn(4,10)*.12)
    bias=nn.Parameter(torch.zeros(10))
    v=nn.Parameter(torch.randn(10)*.12)
    intercept=nn.Parameter(torch.tensor(1.))
    drift_logit=nn.Parameter(torch.tensor(-1.))
    query=nn.Parameter(torch.tensor([1.,1.,1.,0.,0.]))
    opt=torch.optim.Adam([w,bias,v,intercept,drift_logit,query],lr=.022)
    idx=torch.arange(32)
    tic=time.process_time()
    losses=[]
    for step in range(steps):
        world=rng.integers(K,size=32)
        s=torch.zeros((32,K))
        loss=0.
        for revision in range(4):
            world,rs,wrong=stream_step(rng,world)
            rows=torch.tensor(rs.tolist())
            p=(1-drift_logit.sigmoid())*s.softmax(-1)+drift_logit.sigmoid()/K
            s=p.log()-p.log().mean(1,keepdim=True)
            queries=rows[:,:6]
            eq=(queries[:,:,:,None]==queries[:,:,None,:]).float()
            answers=queries[idx[:,None],torch.arange(6),torch.tensor(wrong.tolist())]
            matches=(queries==answers[:,:,None])
            future=(rows[:,6:,:]==rows[idx[:,None],torch.arange(6,14),torch.tensor(world.tolist())[:,None]][:,:,None]).float()
            loss=loss-.2*((future*p[:,None,:]).sum(-1).clamp(min=1e-9).log()).mean()
            used=torch.zeros((32,6),dtype=torch.bool)
            for probe in range(2):
                p=s.softmax(-1)
                mass=(eq*p[:,None,None,:]).sum(-1)
                split=1.-eq
                pairs=p[:,:,None]*p[:,None,:]
                left=(~used).float()[:,:,None,None]
                overlap=((split*left).sum(1,keepdim=True)-split)/(5-probe)
                f=torch.stack((-(mass.clamp(min=1e-9).log()*p[:,None,:]).sum(-1)/np.log(K),
                               (split*pairs[:,None]).sum((-1,-2)),
                               (split*pairs[:,None]*overlap).sum((-1,-2)),
                               1-mass.max(-1).values,eq.sum(-1).reciprocal().sum(-1)/K),dim=-1)
                policy=(10*(f@query)).masked_fill(used,-1e8).softmax(1)
                m=matches.float()
                pp=p[:,None,:].expand_as(m)
                x=torch.stack((pp,(m*pp).sum(-1,keepdim=True).expand_as(m),
                               m.mean(-1,keepdim=True).expand_as(m),
                               (-(pp*pp.clamp(min=1e-9).log()).sum(-1,keepdim=True)/np.log(K)).expand_as(m)),dim=-1)
                amt=torch.nn.functional.softplus(torch.tanh(x@w+bias)@v+intercept)
                signed=torch.where(m.sum(-1,keepdim=True)<=K//2,m,m-1)
                branches=(s[:,None,:]+signed*amt).clamp(-24,24)
                # Straight-through rounding reproduces the precision of the deployed memory.
                branches=branches+(branches.half().float()-branches).detach()
                out=branches.softmax(-1)
                future_p=(out[:,:,None,:]*future[:,None,:,:]).sum(-1)
                costs=-future_p.clamp(min=1e-9).log().mean(-1)
                loss=loss+(policy*costs).sum(-1).mean()
                pick=torch.multinomial(policy.detach(),1).squeeze(-1)
                s=branches[idx,pick]
                used=used.clone()
                used[idx,pick]=True
        loss=loss/8
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    params={k:np.asarray(val.detach().tolist(),dtype=np.float32) for k,val in
            dict(w=w,bias=bias,v=v,intercept=intercept,drift=drift_logit.sigmoid(),query=query).items()}
    np.savez(WEIGHTS,**params)
    return dict(cpu_seconds=time.process_time()-tic,steps=steps,training_streams=steps*32,
                first20_loss=np.mean(losses[:20]).item(),last20_loss=np.mean(losses[-20:]).item(),
                weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                learned_floats=sum(x.size for x in params.values()),params_bytes=sum(x.nbytes for x in params.values()))


def run_streams(model):
    rng=np.random.default_rng(48109)
    n=384
    arms=('native','no_write','uniform_query','bayes_entropy')
    scores=np.zeros((n,len(arms),3))
    worlds=rng.integers(K,size=n)
    states=[[model.initial() for _ in range(n)] for _ in arms]
    queries_changed=0
    spent=time.process_time()
    for revision in range(4):
        worlds,records,reported=stream_step(rng,worlds)
        for j,(world,rows,answers) in enumerate(zip(worlds,records,reported)):
            probes,future=rows[:6],rows[6:]
            baseline_queries=[]
            for a,arm in enumerate(arms):
                prev=states[a][j]
                if arm=='bayes_entropy':
                    state=np.log((1-DRIFT)*probability(prev)+DRIFT/K)
                else:
                    state=model.begin(prev)
                used=[]
                for t in range(3):
                    p=probability(state)
                    values=np.stack([(p*(future==c)).sum(1) for c in range(6)])
                    scores[j,a,t]+=(values.argmax(0)==future[:,world]).mean()/4
                    if t==2:
                        break
                    if arm=='bayes_entropy':
                        f=partition_features(p,probes[:,:,None]==probes[:,None,:])[:,0]
                        f[used]=-np.inf
                        query=int(np.argmax(f))
                    else:
                        query=model.choose(model.initial() if arm=='uniform_query' else state,probes,used)
                    if arm=='native':
                        baseline_queries.append(query)
                    if arm=='uniform_query':
                        queries_changed+=int(query!=baseline_queries[t])
                    used.append(query)
                    match=(probes[query]==probes[query,answers[query]])
                    if arm=='bayes_entropy':
                        state+=np.log((1-NOISE)*match+NOISE*match.mean())
                    elif arm!='no_write':
                        state=model.update(state,match)
                states[a][j]=state
    boot=np.random.default_rng(865).integers(0,n,(4000,n))
    gaps={}
    for a in range(1,len(arms)):
        diff=scores[:,0,2]-scores[:,a,2]
        gaps[arms[a]]=[diff.mean().item(),*np.quantile(diff[boot].mean(1),[.025,.975]).tolist()]
    return dict(streams=n,revisions=4,checks=[0,1,2],arms=arms,accuracy=scores.mean(0).tolist(),
                native_minus_two_checks=gaps,query_changes_vs_uniform_state=queries_changed,
                compared_query_choices=n*4*2,eval_cpu_seconds=time.process_time()-spent),scores


def checks(model):
    rng=np.random.default_rng(493)
    for _ in range(64):
        s=rng.normal(size=K).astype(np.float16)
        sig=rng.integers(5,size=(6,K))
        mask=sig[0]==sig[0,0]
        saved=model.update(s,mask)
        assert saved.nbytes==16 and (saved!=s).sum()<=4
        assert np.array_equal(model.update(s,np.zeros(K)),s)
        assert np.array_equal(model.update(s,np.ones(K)),s)
        restart=np.frombuffer(saved.tobytes(),dtype=np.float16)
        assert np.array_equal(model.prior(restart),model.prior(saved))
        assert model.choose(restart,sig,[0])==model.choose(saved,100-sig,[0])
        perm=rng.permutation(K)
        assert np.allclose(model.update(s[perm],mask[perm]),saved[perm],atol=.002)
        assert model.choose(restart,sig,[0])!=0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('train','check'))
    args=parser.parse_args()
    if args.phase=='train':
        training=train()
        model=load_model()
        checks(model)
        evaluation,scores=run_streams(model)
        np.savez_compressed(HERE/'native-streams.npz',scores=scores)
        (HERE/'native-summary.json').write_text(json.dumps(dict(training=training,evaluation=evaluation),indent=2)+'\n')
        print(training,evaluation)
    else:
        checks(load_model())
        print('Synthetic state, query and permutation checks passed.')
