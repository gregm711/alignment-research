"""Reconstruct partial historical votes with learned attention writes and statistical controls."""
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"attention-updates"))
from conditional import Editor, F, Writer, nn, np, tensor, torch

CACHE = Path.home()/".cache/alignment-research/house-votes"
N = 64
BUDGETS = (1,2,4,8)


def cached(name,section):
    sha = json.loads((ROOT/"sources.json").read_text())[name]
    path = CACHE/name
    if not path.exists():
        CACHE.mkdir(parents=True,exist_ok=True)
        with urlopen(f"https://voteview.com/static/data/out/{section}/"+name,timeout=60) as resp:
            payload = resp.read()
        if hashlib.sha256(payload).hexdigest()!=sha:
            raise ValueError(f"Source digest changed: {name}")
        path.write_bytes(payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    return path


def load(cong):
    path = cached(f"H{cong}_votes.csv","votes")
    roster = cached(f"HS{cong}_members.csv","members")
    with roster.open() as f:
        house = [int(float(r["icpsr"])) for r in csv.DictReader(f) if r["chamber"]=="House"]
    raw = np.loadtxt(path,delimiter=",",skiprows=1,usecols=(2,3,4))
    assert np.isfinite(raw).all() and np.equal(raw,np.floor(raw)).all()
    raw = raw.astype(np.int64)
    # Source is House-specific. Exclude any separate presidential pseudo-record.
    with path.open() as f:
        rows = csv.reader(f)
        next(rows)
        is_house = np.fromiter((r[1]=="House" for r in rows),bool,count=len(raw))
    raw = raw[is_house & np.isin(raw[:,1],house)]
    rolls,rid = np.unique(raw[:,0],return_inverse=True)
    members,mid = np.unique(raw[:,1],return_inverse=True)
    matrix = np.full((len(rolls),len(members)),np.nan,dtype=np.float32)
    cast = np.isin(raw[:,2],[1,6])
    keys = rid[cast]*len(members)+mid[cast]
    assert len(np.unique(keys))==len(keys)
    matrix[rid[cast],mid[cast]] = raw[cast,2]==1
    return members,rolls,matrix


def panel(old,new,rng):
    cand = np.flatnonzero(np.isfinite(old[2]).mean(0)>=.9)
    chosen = rng.choice(cand,N,replace=False)
    people = list(old[0][chosen])
    def extract(data):
        lookup = {m:i for i,m in enumerate(data[0])}
        y = np.full((len(data[1]),N),np.nan,dtype=np.float32)
        for i,m in enumerate(people):
            if m in lookup:
                y[:,i] = data[2][:,lookup[m]]
        mask = np.isfinite(y).sum(1)>=24
        return y[mask],data[1][mask]
    return extract(old),extract(new)


def reveal(y,rng,reps=1):
    true = np.repeat(y,reps,axis=0)
    order = (rng.random(true.shape)+2*np.isnan(true)).argsort(axis=1)[:,:8]
    support = np.take_along_axis(true,order,axis=1)
    known = np.isfinite(true)
    np.put_along_axis(known,order,False,axis=1)
    assert np.isfinite(support).all() and (known.sum(1)>=16).all()
    return np.nan_to_num(true),order,support,known


class RollModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Editor(N+3)
        self.writer = Writer()
        tokens = torch.zeros(N,20,dtype=torch.long)
        tokens[:,0] = 2
        tokens[:,1] = torch.arange(N)+3
        self.register_buffer("tokens",tokens)

    def items(self):
        return self.writer.features(self.encoder.inputs(self.tokens))


def fit(model,y,base,steps,seed,kind,rate):
    opt = torch.optim.Adam(model.parameters(),lr=rate)
    rng = np.random.default_rng(seed)
    bias = tensor(np.log(base/(1-base)))
    for _ in range(steps):
        batch,ids,sy,mask = reveal(y[rng.integers(len(y),size=24)],rng)
        truth,obs = tensor(batch),tensor(mask)
        r,d = model.items()
        state = model.writer.start(len(ids))
        losses = []
        for j in range(8):
            i = ids[:,j].tolist()
            state,_ = model.writer.step(state,r[i],d[i],tensor(sy[:,j]),kind,bias=bias[i,None])
            if j+1 in BUDGETS:
                z = model.writer.score(r.expand(len(ids),-1,-1,-1),state[0],bias)
                logp = F.logsigmoid((2*truth[:,None]-1)*z)*obs[:,None]
                count = obs.sum(-1)
                member_loss = -logp[:,0].sum(-1)/count
                witness_loss = -(torch.logsumexp(logp[:,1:].sum(-1),dim=1)-np.log(2))/count
                losses.append((member_loss+.5*witness_loss).mean())
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        opt.step()


def neural(model,kind,base,data):
    _,ids,sy,_ = data
    bias = tensor(np.log(base/(1-base)))
    result = {k:[] for k in (*BUDGETS,5,"false")}
    with torch.no_grad():
        r,d = model.items()
        for start in range(0,len(ids),64):
            n = len(ids[start:start+64])
            state = model.writer.start(n)
            def forecast(s):
                z = model.writer.score(r.expand(n,-1,-1,-1),s[0][:,:1],bias)
                return np.array(z[:,0].sigmoid().tolist(),dtype=np.float32)
            for j in range(8):
                i = ids[start:start+n,j].tolist()
                label = tensor(sy[start:start+n,j])
                if j==4:
                    fork,_ = model.writer.step(state,r[i],d[i],1-label,kind,bias=bias[i,None])
                    result["false"].append(forecast(fork))
                state,_ = model.writer.step(state,r[i],d[i],label,kind,bias=bias[i,None])
                if j+1 in BUDGETS or j==4:
                    result[j+1].append(forecast(state))
    return {k:np.concatenate(v) for k,v in result.items()}


def templates(history,base,error,data,extra=None):
    _,ids,sy,_ = data
    rows = history if extra is None else np.vstack([history,extra])
    prior = np.ones(len(rows))
    if extra is not None:
        prior[:len(history)] *= .5/len(history)
        prior[len(history):] *= .5/len(extra)
    complete = np.where(np.isfinite(rows),rows,base).astype(float)
    chances = error+(1-2*error)*complete
    result = {n:[] for n in BUDGETS}
    for start in range(0,len(ids),64):
        ii = ids[start:start+64]; y = sy[start:start+64]
        w = np.tile(prior,(len(ii),1))
        for j in range(8):
            yes = chances[:,ii[:,j]].T
            w *= np.where(y[:,j,None],yes,1-yes)
            if j+1 in BUDGETS:
                result[j+1].append((w@complete)/w.sum(1,keepdims=True))
    return {k:np.concatenate(v) for k,v in result.items()}


def kernel(history,base):
    obs = np.isfinite(history).astype(float)
    r = np.nan_to_num(history-base)
    c = r.T@r/(obs.T@obs+12)
    vals,vec = np.linalg.eigh(c)
    c = (vec*vals.clip(0))@vec.T
    scale = np.sqrt(np.maximum(np.diag(c),1e-12))
    return c/(scale[:,None]*scale)


def ridge(base,K,reg,data):
    _,ids,sy,_ = data
    result = {}
    for j in BUDGETS:
        i = ids[:,:j]
        w = np.linalg.solve(K[i[:,:,None],i[:,None,:]]+reg*np.eye(j),(sy[:,:j]-base[i])[:,:,None])
        result[j]=(base+(K[:,i].transpose(1,0,2)@w).squeeze(2)).clip(0,1)
    return result


def metrics(pred,data,reps):
    y,_,_,mask=data
    den=mask.sum(1)
    rows=[]
    for j in BUDGETS:
        v=pred[j]
        acc=(((v>=.5)==y)&mask).sum(1)/den
        brier=(((v-y)**2)*mask).sum(1)/den
        rows.append(np.stack([acc,brier],axis=1))
    return np.stack(rows,axis=1).reshape(-1,reps,4,2).mean(1)


def compact(pred):
    return np.stack([pred[k]>=.5 for k in (4,5,"false")])


def feedback(a,b,data):
    y,_,_,mask=data
    result=[]
    for k in (1,2):
        eligible = ((a[0]!=y)&(b[0]!=y) if k==1 else (a[0]==y)&(b[0]==y))&mask
        result.append([int((eligible&(v[k]!=v[0])).sum()) for v in (a,b)]+[int(eligible.sum())])
    return result


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records=[]; saved={}; checkpoints={}
    old=load(110)
    for cong in range(111,117):
        new=load(cong)
        rng=np.random.default_rng(7000+cong)
        (prev,_),(now,rollids)=panel(old,new,rng)
        old=new
        tr,dev,reserved=np.split(prev,[int(.8*len(prev)),int(.9*len(prev))])
        assert len(reserved)>=32 and len(now)>=544
        base=(np.nansum(tr,axis=0)+1)/(np.isfinite(tr).sum(0)+2)
        K=kernel(tr,base)
        dv=reveal(dev,rng)
        rank=lambda pred:metrics(pred,dv,1)[:,:,1].mean()
        template_loss,err=min((rank(templates(tr,base,err,dv)),err) for err in (.03,.1,.2,.3))
        ridge_loss,s,reg=min((rank(ridge(base,1+s*K,reg,dv)),s,reg) for s in (.25,1,4) for reg in (.25,1,4,16))
        pop_loss=rank(dict.fromkeys(BUDGETS,np.broadcast_to(base,dv[0].shape)))
        ref=min(((template_loss,"template-recent"),(ridge_loss,"covariance"),(pop_loss,"prior")))[1]
        torch.manual_seed(8120+cong)
        init=RollModel()
        dynamic=copy.deepcopy(init)
        fixed=copy.deepcopy(init)
        fit(dynamic,tr,base,192,3000+cong,"learned",.003)
        fit(fixed,tr,base,192,3000+cong,"clamped",.003)
        recent=copy.deepcopy(dynamic)
        old_extra=copy.deepcopy(dynamic)
        fixed_recent=copy.deepcopy(fixed)
        for model,history,kind in ((recent,now[:32],"learned"),(old_extra,reserved[-32:],"learned"),
                                   (fixed_recent,now[:32],"clamped"),(init,now[:32],"learned")):
            fit(model,history,base,32,4000+cong,kind,.0008)
        for name,param in recent.state_dict().items():
            checkpoints[f"{cong}/{name}"] = np.asarray(param.tolist(),dtype=np.int64 if name=="tokens" else np.float32)
        record=dict(congress=cong,eligible=[len(tr),len(dev),len(reserved),len(now)],reference=ref,
                    tuning=[err,s,reg],blocks={})
        for block,sl in (("early",slice(32,288)),("late",slice(-256,None))):
            data=reveal(now[sl],rng,4)
            runs=dict()
            runs["prior"]=dict.fromkeys(BUDGETS,np.broadcast_to(base,data[0].shape))
            runs["template-old"]=templates(tr,base,err,data)
            runs["template-recent"]=templates(tr,base,err,data,now[:32])
            runs["covariance"]=ridge(base,1+s*K,reg,data)
            curves={k:metrics(v,data,4) for k,v in runs.items()}
            before={}
            for k,model,kind in (("frozen",dynamic,"learned"),("fixed-old",fixed,"clamped"),
                                 ("recent",recent,"learned"),("old-extra",old_extra,"learned"),
                                 ("fixed-recent",fixed_recent,"clamped"),("recent-clamped",recent,"clamped"),
                                 ("fresh",init,"learned")):
                pred=neural(model,kind,base,data)
                curves[k]=metrics(pred,data,4)
                if k in ("frozen","recent","fixed-recent"):
                    before[k]=compact(pred)
            record["blocks"][block]=dict(curves={k:v.mean(0).tolist() for k,v in curves.items()},
                feedback={k:feedback(before["recent"],before[k],data) for k in ("frozen","fixed-recent")},
                rollnumbers=rollids[sl].tolist())
            for k,v in curves.items():
                saved[f"{cong}-{block}-{k}"]=v
        records.append(record)
        print("finished Congress",cong,flush=True)
    (ROOT/"results.json").write_text(json.dumps(records,indent=1)+"\n")
    np.savez_compressed(ROOT/"vote-trials.npz",**saved)
    np.savez_compressed(ROOT/"recent-weights.npz",**checkpoints)


if __name__=="__main__":
    main()
