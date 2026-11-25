"""Infer how a correction generalizes and change only the direction of the next weight write."""
import copy
import json

from study import ROOT, WIDTH, Editor, F, corpus, nn, np, p, pretrain, torch

SEEDS = range(1501,1511)
METHODS = ("learned","clamped","swapped","fixed","empirical","structure")
PARITIES = ("familiar","novel")
ARCHIVE = ROOT/"conditional-weights.npz"


def tensor(x,dtype=torch.float32):
    return torch.tensor(np.asarray(x).tolist(),dtype=dtype)


def gather(y,c):
    return np.take_along_axis(y,c,axis=1)


def episodes(rng,pool,n,parity):
    # Independent wording draws for support, check and query; no query wording is a support candidate.
    length = min(len(x) for x in pool)
    width = min(length//2,40)
    bank = np.stack([x[:width] for x in pool])
    other = np.stack([x[width:2*width] for x in pool])
    order = rng.random((n,3)).argsort(1)
    first = 2*order+rng.integers(2,size=(n,3))
    trigger = first[:,2]
    classes = np.column_stack([first[:,0],first[:,0]^1,first[:,1],first[:,1]^1,trigger,trigger])
    slots = rng.integers(width,size=(n,6))
    slots[:,5] = (slots[:,4]+rng.integers(1,width,size=n))%width
    ids = bank[classes,slots]
    query = other[np.arange(6),rng.integers(width,size=(n,6))]
    bits = rng.integers(2,size=(n,3))
    bits[:,2] = bits[:,0]^bits[:,1]^parity
    mode = rng.integers(2,size=n)
    truth = np.empty((n,6),int)
    truth[:,::2] = bits
    truth[:,1::2] = bits^mode[:,None]
    labels = gather(truth,classes)
    return dict(ids=ids,query=query,classes=classes,truth=truth,labels=labels,mode=mode)


class Writer(nn.Module):
    def __init__(self):
        super().__init__()
        identity = torch.eye(WIDTH).repeat(2,1,1)
        self.reader = nn.Parameter(identity.clone())
        self.directions = nn.Parameter(identity.repeat(2,1,1,1)+.25*torch.randn(2,2,WIDTH,WIDTH))
        self.sharpness = nn.Parameter(torch.tensor(.5))
        self.register_buffer("out",F.normalize(torch.linspace(-1,1,WIDTH),dim=0).reshape(2,-1))

    def features(self,x):
        read = F.normalize(torch.einsum("...hd,hdk->...hk",x,self.reader),dim=-1)
        directions = F.normalize(torch.einsum("...hd,ehdk->...ehk",x,self.directions),dim=-1)
        return read,directions

    def score(self,read,v,bias=0):
        # read [batch, queries, heads, width]; v [batch, readers, heads, width, values].
        a = torch.einsum("bqhd,behdk->beqhk",read,v)
        return (a.tanh()*self.out).sum((-2,-1))+bias

    def start(self,n):
        return torch.zeros(n,3,2,WIDTH,WIDTH//2),torch.zeros(n)

    def step(self,state,read,directions,y,kind="learned",override=None,bias=0):
        v,evidence = state
        g = (evidence*(F.softplus(self.sharpness)+.1)).sigmoid()
        if kind=="clamped":
            g = torch.full_like(g,.5)
        if override is not None:
            g = override
        mix = directions[:,0]*g[:,None,None]+directions[:,1]*(1-g[:,None,None])
        keys = torch.cat([mix[:,None],directions],dim=1)
        a = torch.einsum("bhd,behdk->behk",read,v).tanh()
        pred = (a*self.out).sum((-2,-1))+bias
        delta = (y[:,None]-pred.sigmoid())[:,:,None,None]*(1-a.square())*self.out
        updated = v+4*keys[:,:,:,:,None]*delta[:,:,:,None,:]
        # This report's evidence is available for the following write.
        logp = F.logsigmoid((2*y[:,None]-1)*pred[:,1:])
        return (updated,evidence+logp[:,0]-logp[:,1]),g


def history_loss(model,x,batch,cal,noise,kind):
    sx,sd = model.features(x[batch["ids"].tolist()])
    qr,_ = model.features(x[batch["query"].tolist()])
    y = tensor(batch["labels"]^noise)
    truth = tensor(batch["truth"])
    v = model.start(len(y))
    positions = [*range(cal*2),4,5]
    losses = []
    for j in positions:
        v,_ = model.step(v,sx[:,j],sd[:,j],y[:,j],kind)
        if j>=4:
            z = model.score(qr,v[0])
            main = F.binary_cross_entropy_with_logits(z[:,0],truth)
            # Fit the two candidate rules from outcomes without a regime label.
            lps = F.logsigmoid((2*truth[:,None]-1)*z[:,1:]).sum(2)
            mixture = -(torch.logsumexp(lps,dim=1)-np.log(2)).mean()/6
            losses.append(main+.5*mixture)
    return torch.stack(losses).mean()


def text_likelihood(prob,ids,y,error):
    yes = error+(1-2*error)*(prob[ids]@p.MASK.T)
    return np.where(y[:,:,None],yes,1-yes).clip(1e-8,1).prod(1)


def fit_history(rng,net,x,prob,pools):
    fixed = copy.deepcopy(net)
    opts = [torch.optim.Adam(m.parameters(),lr=.008) for m in (net,fixed)]
    evidence = []
    for t in range(256):
        batch = episodes(rng,pools["train"],32,0)
        cal = 1+t%2
        noise = np.zeros_like(batch["labels"])
        noise[:,4] = rng.random(32)<.1
        noise[:,5] = rng.random(32)<.02
        y = batch["labels"]^noise
        ev = text_likelihood(prob,batch["query"],batch["truth"],0)
        for js,err in ((list(range(2*cal)),0),([4],.1),([5],.02)):
            ev *= text_likelihood(prob,batch["ids"][:,js],y[:,js],err)
        evidence.append(ev)
        for model,opt,kind in zip((net,fixed),opts,("learned","clamped")):
            opt.zero_grad()
            history_loss(model,x,batch,cal,noise,kind).backward()
            opt.step()
    ev = np.concatenate(evidence)
    prior = np.ones(64)/64
    for _ in range(64):
        updated = p.normalize(.5+p.normalize(ev*prior).sum(0))
        if np.max(abs(updated-prior))<1e-8:
            break
        prior = updated
    return fixed,prior


def visible_queries(pool):
    size = min(min(map(len,pool))//2,40)
    return np.stack([ids[size:2*size] for ids in pool])


def neural(model,x,b,query,cal,kind):
    n = len(b["truth"])
    sr,sd = model.features(x[b["ids"].tolist()])
    qr,_ = model.features(x[query.ravel().tolist()])
    qr = qr.expand(n,-1,-1,-1)
    y = tensor(b["labels"])
    def prefix(flip):
        state = model.start(n)
        for j in range(cal*2):
            obs = 1-y[:,j] if flip and j%2 else y[:,j]
            state,_ = model.step(state,sr[:,j],sd[:,j],obs,kind)
        return state
    state = prefix(False)
    forced = None
    if kind=="swapped":
        _,forced = model.step(prefix(True),sr[:,4],sd[:,4],y[:,4])
    values = []
    probs = []
    def read(st):
        return model.score(qr,st[0][:,:1])[:,0].sigmoid().reshape(n,6,-1)
    for obs in (y[:,4],1-y[:,4]):
        edited,g = model.step(state,sr[:,4],sd[:,4],obs,kind,override=forced)
        probs.append([read(edited)])
        values.append(edited[0][:,0])
        for check in (y[:,5],1-y[:,5]):
            checked,_ = model.step(edited,sr[:,5],sd[:,5],check,kind)
            probs[-1].append(read(checked))
    return np.array(torch.stack([torch.stack(row) for row in probs]).tolist()),values,g,qr


def statistic(prob,b,query,cal,prior):
    post = p.normalize(prior*text_likelihood(prob,b["ids"][:,:2*cal],b["labels"][:,:2*cal],0))
    def learn(w,j,labels,err):
        return p.normalize(w*text_likelihood(prob,b["ids"][:,j:j+1],labels[:,None],err))
    def output(w):
        return ((w@p.MASK)@prob[query.ravel()].T).reshape(len(w),6,-1)
    arms = []
    for obs in (b["labels"][:,4],1-b["labels"][:,4]):
        new = learn(post,4,obs,.1)
        arms.append([output(new),*[output(learn(new,5,c,.02)) for c in (b["labels"][:,5],1-b["labels"][:,5])]])
    return np.array(arms)


def metrics(probs,b):
    n = len(b["truth"])
    tr = b["classes"][:,4]
    row = np.arange(n)
    peer = tr^1
    unseen = b["classes"][:,2]
    target = b["truth"][row,peer,None]
    partner = probs[:,:,row,peer]
    right = (partner>=.5)==target
    accurate = right.mean((2,3))
    overlap = []
    for branch in (1,0):
        eligible = ~right[branch,0] if branch else right[branch,0]
        events = right[branch,1:] if branch else ~right[branch,1:]
        overlap.append((eligible,events))
    signed = (partner[0,0]-partner[1,0])*(2*b["labels"][:,4,None]-1)
    untouched = ((probs[0,0,row,unseen]>=.5)==b["truth"][row,unseen,None]).mean()
    return dict(peer=float(accurate[0,0]),false_report=float(accurate[1,0]),fork_accuracy=accurate.tolist(),
                sign=[float(signed[b["mode"]==g].mean()) for g in (0,1)],other=float(untouched)),overlap


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domains = json.loads((p.CACHE/"domains.json").read_text())
    relation = (p.MASK[:,::2]!=p.MASK[:,1::2])
    oracle = p.normalize(np.all(relation==relation[:,:1],axis=1).astype(float))
    results,archive = [],{}
    for i,seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        names = rng.choice(domains[list(domains)[i]],6,replace=False).tolist()
        tokens,pools,vocab = corpus(names,rng)
        language = Editor(len(vocab)+3)
        pretrain(language,tokens,pools,rng)
        with torch.no_grad():
            x = torch.cat([language.inputs(chunk) for chunk in tokens.split(96)])
            prob = np.array(torch.cat([language.intents(language.context(chunk)[:,0]).softmax(-1) for chunk in tokens.split(96)]).tolist())
        model = Writer()
        fixed,prior = fit_history(rng,model,x,prob,pools)
        rec = dict(seed=seed,intents=names,results={})
        query = visible_queries(pools["test"])
        with torch.no_grad():
            for name,param in language.state_dict().items():
                archive[f"{seed}/language/{name}"] = np.array(param.tolist(),np.float32)
            archive[f"{seed}/reader"] = np.array(model.reader.tolist(),np.float32)
            archive[f"{seed}/tokens"] = np.array(tokens[query.ravel().tolist()].tolist(),np.int64)
            for parity,family in enumerate(PARITIES):
                b = episodes(rng,pools["test"],144,parity)
                assert not np.intersect1d(query,b["ids"]).size
                for cal in (0,1,2):
                    group = {}
                    overlaps = {}
                    for method in METHODS:
                        if method in ("empirical","structure"):
                            preds = statistic(prob,b,query,cal,prior if method=="empirical" else oracle)
                        else:
                            m = fixed if method=="fixed" else model
                            kind = "clamped" if method=="fixed" else method
                            preds,values,gate,qr = neural(m,x,b,query,cal,kind)
                            if method=="learned" and cal==1 and parity==1:
                                for g in (0,1):
                                    idx = int(np.flatnonzero(b["mode"]==g)[0])
                                    key = f"{seed}-{g}"
                                    archive[key+"-v"] = np.array(values[0][idx].tolist(),np.float32)
                                    archive[key+"-out"] = np.array(model.out.tolist(),np.float32)
                                    archive[key+"-p"] = preds[0,0,idx].ravel()
                                assert torch.eq(m.score(qr[:1],torch.zeros_like(values[0][:1,None])),0).all()
                        group[method],overlaps[method] = metrics(preds,b)
                    if cal==1:
                        counts,by_return = [],[]
                        for j in (0,1):
                            learned,control = overlaps["learned"][j],overlaps["fixed"][j]
                            eligible = learned[0]&control[0]
                            cases = [[int((v[1][k]&eligible).sum()) for v in (learned,control)]
                                     +[int(eligible.sum())] for k in (0,1)]
                            counts.append(cases[j])
                            by_return.append(cases)
                        group["eligible_counts"] = counts
                        group["check_counts_by_truth"] = by_return
                    rec["results"][family+str(cal)] = group
        results.append(rec)
        print("finished",seed,flush=True)
    (ROOT/"conditional-trials.json").write_text(json.dumps(results,indent=1)+"\n")
    np.savez_compressed(ARCHIVE,**archive)


if __name__=="__main__":
    main()
