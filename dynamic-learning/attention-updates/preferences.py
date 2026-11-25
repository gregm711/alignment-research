"""Transfer a signed or monotone attention edit to actual preference judgments."""
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from direction import NonnegativeEditor
from study import BUDGETS, ROOT, Editor, F, logits, np, torch, words, write

CACHE = Path.home()/".cache/alignment-research/jester"
SOURCE = "https://raw.githubusercontent.com/BerkeleyAutomation/Jester-Frontend/4449c6e7f4a8d3644308f5c5edc7e11404ea808b/dataset/"
FILES = {"jester_dataset_1_1.zip":"cb7841756ded125b15f3d95c9f8241224065e96354c132a46ae4214deeab7615",
         "jester_dataset_1_joke_texts.zip":"874a97180ac49a1d815ab6829fd9411021cdd25a92557dc39af46a26d5d1a9da"}
ANCHOR = np.array([5,7,8,13,15,16,17,18,19,20])-1
TARGET = np.setdiff1d(np.arange(100),ANCHOR)


def inputs():
    import xlrd
    CACHE.mkdir(parents=True,exist_ok=True)
    for name,digest in FILES.items():
        path = CACHE/name
        if not path.exists():
            with urlopen(SOURCE+name,timeout=60) as f:
                payload = f.read()
            if hashlib.sha256(payload).hexdigest()!=digest:
                raise ValueError("Unexpected data from Jester source")
            path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with ZipFile(CACHE/"jester_dataset_1_1.zip") as z:
        book = xlrd.open_workbook(file_contents=z.read("jester-data-1.xls"),on_demand=True)
        sheet = book.sheet_by_index(0)
        values = np.array([sheet.row_values(i,1,101) for i in range(sheet.nrows)],dtype=np.float32)
        book.release_resources()
    assert values.shape==(24983,100)
    observed = values!=99
    assert np.max(np.abs(values[observed]))<=10
    keep = observed[:,ANCHOR].all(1)&(observed[:,TARGET].sum(1)>=10)
    y = np.where(observed[keep],values[keep]>0,np.nan).astype(np.float32)
    vocab, tokens = {},[]
    with ZipFile(CACHE/"jester_dataset_1_joke_texts.zip") as z:
        for i in range(100):
            text = z.read(f"jokes/init{i+1}.html").decode("latin1")
            body = re.search(r"<!--\s*begin of joke\s*-->(.*?)<!--\s*end of joke\s*-->",text,re.I|re.S)
            if body is None:
                raise ValueError(f"Missing joke text {i+1}")
            seq = [2,3+i]
            for w in words(html.unescape(re.sub(r"<[^>]*>"," ",body[1])))[:18]:
                seq.append(vocab.setdefault(w,103+len(vocab)))
            tokens.append(seq+[0]*(20-len(seq)))
    rng = np.random.default_rng(88419)
    y = y[rng.permutation(len(y))]
    menu = ANCHOR[rng.random((len(y),len(ANCHOR))).argsort(1)]
    cuts = [int(.6*len(y)),int(.8*len(y))]
    return (torch.tensor(tokens),len(vocab)+103),list(zip(np.split(y,cuts),np.split(menu,cuts)))


def tensor(a):
    return torch.tensor(a.tolist(),dtype=torch.float32)


def fit(net,tokens,base,data,seed):
    y,menu = data
    bias = tensor(np.log(base/(1-base)))
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(),lr=.003)
    for _ in range(256):
        users = rng.integers(len(y),size=48)
        items = np.take_along_axis(menu[users],rng.random((48,10)).argsort(1),axis=1)[:,:4]
        support = tensor(y[users[:,None],items])
        truth = y[users[:,None],TARGET]
        observed = tensor(np.isfinite(truth))
        targets = tensor(np.nan_to_num(truth))
        h = net.inputs(tokens)
        q = h[TARGET.tolist()].expand(len(users),-1,-1,-1)
        v = net.value.expand(len(users),-1,-1,-1)
        losses = []
        for j in range(4):
            idx = items[:,j].tolist()
            v = write(h[idx],support[:,j],v,net.read,bias[idx])
            if j+1 in BUDGETS:
                score = logits(q,v,net.read,bias[TARGET.tolist()])
                loss = F.binary_cross_entropy_with_logits(score,targets,reduction="none")
                losses.append(((loss*observed).sum(1)/observed.sum(1)).mean())
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        opt.step()


def predictions(net,h,base,data,scale):
    y,menu = data
    bias = tensor(np.log(base/(1-base)))
    result = []
    for start in range(0,len(y),128):
        m = menu[start:start+128]
        support = tensor(np.take_along_axis(y[start:start+128],m,axis=1))
        v = net.value.expand(len(m),-1,-1,-1)
        q = h[TARGET.tolist()].expand(len(m),-1,-1,-1)
        def forecast(state):
            return logits(q,state,net.read,bias[TARGET.tolist()]).sigmoid()
        batch = {}
        for j in range(8):
            ids = m[:,j].tolist()
            def edit(label):
                return v+scale*(write(h[ids],label,v,net.read,bias[ids])-v)
            if j==4:
                batch["false"] = forecast(edit(1-support[:,j]))
            v = edit(support[:,j])
            if j+1 in BUDGETS or j==4:
                batch[j+1] = forecast(v)
        result.append({k:np.asarray(v.tolist(),dtype=np.float32) for k,v in batch.items()})
    return {k:np.concatenate([b[k] for b in result]) for k in result[0]}


def classic(base,data,kernel,reg):
    y,menu = data
    out = {}
    for n in BUDGETS:
        m = menu[:,:n]
        err = np.take_along_axis(y,m,axis=1)-base[m]
        w = np.linalg.solve(kernel[m[:,:,None],m[:,None,:]]+reg*np.eye(n),err[:,:,None])
        out[n] = (base[TARGET]+(kernel[TARGET[None,:,None],m[:,None,:]]@w).squeeze(-1)).clip(0,1)
    return out


def metrics(pred,y):
    truth = y[:,TARGET]
    obs = np.isfinite(truth)
    den = obs.sum(1)
    vals = []
    for n in BUDGETS:
        acc = (((pred[n]>=.5)==truth)&obs).sum(1)/den
        brier = np.nansum((pred[n]-truth)**2,axis=1)/den
        vals.append(np.stack([acc,brier],axis=1))
    return np.stack(vals,axis=1)


def feedback(pairs,data):
    y,menu = data
    truth = y[:,TARGET]
    seen = np.isfinite(truth)
    negative = np.take_along_axis(y,menu[:,4:5],axis=1)==0
    pre = np.stack([p[4]>=.5 for p in pairs])
    good = ((pre==truth)&seen).all(0)
    bad = ((pre!=truth)&seen).all(0)
    events = []
    for i,pred in enumerate(pairs):
        true = pred[5]>=.5
        false = pred["false"]>=.5
        wrong_low = (truth==1)&~pre[i]&negative
        events.append(np.stack([(bad&(true==truth)).sum(1),bad.sum(1),
                                (good&(false!=truth)).sum(1),good.sum(1),
                                (wrong_low&true).sum(1),wrong_low.sum(1),
                                ((pred[5]>pred[4]+1e-6)&negative&seen).sum(1),(negative&seen).sum(1)],axis=1))
    return np.stack(events)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    (tokens,nwords),(tr,dev,test) = inputs()
    train_y = tr[0]
    base = (np.nansum(train_y,axis=0)+1)/(np.isfinite(train_y).sum(0)+2)
    obs = np.isfinite(train_y).astype(float)
    residual = np.nan_to_num(train_y-base)
    cov = residual.T@residual/(obs.T@obs+12)
    vals,vec = np.linalg.eigh(cov)
    cov = (vec*vals.clip(0))@vec.T
    cov /= np.sqrt(np.diag(cov)[:,None]*np.diag(cov)[None,:])
    conventional,details = {},{}
    for method,strengths in (("leniency",(0,)),("covariance",(0,.25,1,4))):
        candidates = []
        for s in strengths:
            for r in (.25,1,4,16):
                score = metrics(classic(base,dev,1+s*cov,r),dev[0])[:,:,1].mean()
                candidates.append((score,s,r))
        loss,s,r = min(candidates)
        details[method] = [loss,s,r]
        conventional[method] = metrics(classic(base,test,1+s*cov,r),test[0])
    comparator = min(details,key=lambda k:details[k][0])
    fixed = np.broadcast_to(base[TARGET],(len(test[0]),len(TARGET)))
    conventional["fixed"] = metrics(dict.fromkeys(BUDGETS,fixed),test[0])
    scores,counts,settings = [],[],[]
    for seed in range(1301,1305):
        preds,ms,ss = [],[],[]
        for cls in (Editor,NonnegativeEditor):
            torch.manual_seed(seed)
            net = cls(nwords)
            fit(net,tokens,base,tr,seed)
            with torch.no_grad():
                h = net.inputs(tokens)
                scale = min((.5,1.,2.),key=lambda a:metrics(predictions(net,h,base,dev,a),dev[0])[:,:,1].mean())
                pred = predictions(net,h,base,test,scale)
            preds.append(pred)
            ms.append(metrics(pred,test[0]))
            ss.append(scale)
        scores.append(ms)
        counts.append(feedback(preds,test))
        settings.append(ss)
        print("seed",seed,"finished",flush=True)
    scores,counts = np.array(scores),np.array(counts)
    assert counts[:,1,:,4].sum()==0 and counts[:,1,:,6].sum()==0
    np.savez_compressed(ROOT/"jester-trials.npz",signed=scores[:,0],positive=scores[:,1],
                        feedback=counts,**conventional)
    meta = dict(split=[len(x[0]) for x in (tr,dev,test)],settings=settings,reference_settings=details,
                comparator=comparator,negative_checks=int((np.take_along_axis(test[0],test[1][:,4:5],axis=1)==0).sum()))
    (ROOT/"jester-settings.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    main()
