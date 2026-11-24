"""Prospective replay of each project's first selected test on archived Jenkins builds."""
import csv
import gzip
import hashlib
import heapq
import io
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path.home()/'.cache/alignment-research/ci-question-pilot'
META = 'https://raw.githubusercontent.com/lrtsuser/LRTS/2c1b68e9b1d4ec6c4f304869e36af2e3168da889/artifact/metadata/dataset.csv'
MD5 = {'karaf':'6f0d45a08644d19594692c11e5508300','log4j':'a62ec9e74fdcf53ab7a0ffeb3412a315'}
LABEL = {'PASSED':0,'FIXED':0,'FAILED':1,'REGRESSION':1,'SKIPPED':-1}
METHODS = ('market','equal','frequency','latest','recent','pr_latest','shrink')


def metadata():
    with urlopen(META,timeout=30) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest()=='49570f8dca753c385f25346f8aaaded3a841b0e67dabf36fbb67260a79b02987'
    builds = {project:{} for project in MD5}
    for r in csv.DictReader(io.StringIO(raw.decode())):
        if r['project'] not in builds:
            continue
        records = builds[r['project']]
        key = (r['pr_name'],int(r['build_id']))
        start,duration = float(r['build_timestamp']),float(r['build_duration'])
        assert start > 0 and duration > 0
        times = (start,start+duration)
        assert key not in records or records[key]==times
        records[key]=times
    return builds,hashlib.sha256(raw).hexdigest()


def ingested(project, meta):
    path = CACHE / f'{project}.zip'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(f'https://zenodo.org/records/15603448/files/{project}.zip?download=1',timeout=90) as f:
            raw=f.read()
        assert hashlib.md5(raw).hexdigest()==MD5[project]
        path.write_bytes(raw)
    assert hashlib.md5(path.read_bytes()).hexdigest()==MD5[project]
    counts=Counter(metadata_builds=len(meta))
    ids={}
    builds=[]
    matched=set()
    with ZipFile(path) as outer:
        for entry in outer.namelist():
            if not entry.endswith('.zip'):
                continue
            counts['archive_reports']+=1
            match=re.search(r'/PR/(PR-[^/]+)/testReport_build(\d+)\.json\.zip$',entry)
            if not match:
                counts['other_path']+=1
                continue
            key=(match[1],int(match[2]))
            if key not in meta:
                counts['archive_only']+=1
                continue
            assert key not in matched
            matched.add(key)
            with ZipFile(io.BytesIO(outer.read(entry))) as z:
                (member,)=[v for v in z.namelist() if v.endswith('.json')]
                doc=json.loads(z.read(member))
            values={}
            mixed=set()
            for suite in doc['suites']:
                stage=tuple(suite.get('enclosingBlockNames') or ())
                for c in suite['cases']:
                    label=LABEL[c['status'].strip().upper()]
                    identity=(stage,c['className'],c['name'])
                    idx=ids.setdefault(identity,len(ids))
                    counts['raw_tests']+=1
                    counts['skipped_occurrences']+=int(label<0)
                    if idx in values:
                        counts['duplicate_occurrences']+=1
                        if values[idx]!=label:
                            mixed.add(idx)
                    else:
                        values[idx]=label
            counts['duplicate_groups_mixed']+=len(mixed)
            if not values:
                counts['empty_reports']+=1
                continue
            start,end=meta[key]
            builds.append((start,end,key,values))
    counts['metadata_only']=len(meta)-len(matched)
    counts['joined']=len(matched)
    counts['nonempty']=len(builds)
    counts['identities']=len(ids)
    names={i:k for k,i in ids.items()}
    return sorted(builds,key=lambda b:(b[0],b[2])),names,dict(counts)


def prior(history, indices):
    # Ordinary test history contains executed cases, coded as -1 where unknown.
    hist=np.array([a[indices] for a in history],dtype=np.int8).reshape(len(history),len(indices))
    valid=hist>=0
    fails=(hist==1)
    f=(1+fails.sum(0))/(2+valid.sum(0))
    recent=(1+fails[-4:].sum(0))/(2+valid[-4:].sum(0))
    last=np.full(len(indices),.5)
    for row in hist:
        mask=row>=0
        last[mask]=.1+.8*row[mask]
    return f,recent,last,valid.sum(0),fails.sum(0)


def replay(builds,names):
    n=len(names)
    pending=[]
    archive=[]
    same_pr={}
    wealth=np.full((n,4),.25)
    records=[]
    for seq,(start,end,key,truth) in enumerate(builds):
        while pending and pending[0][0]<start:
            completed,_,old_pr,indices,actual,experts=heapq.heappop(pending)
            assert completed<start
            known=actual>=0
            ids,y,v=indices[known],actual[known],experts[known]
            summary=np.full(n,-1,dtype=np.int8)
            summary[ids]=y
            archive.append(summary)
            if len(archive)>32:
                archive.pop(0)
            same_pr.setdefault(old_pr,{}).update(zip(ids,y))
            posterior=wealth[ids]*np.where(y[:,None],v,1-v)
            wealth[ids]=.95*posterior/posterior.sum(1)[:,None]+.05/4
        indices=np.array(sorted(truth,key=names.__getitem__))
        f,recent,last,count,pos=prior(archive,indices)
        all_seen=sum(np.count_nonzero(row>=0) for row in archive)
        all_fail=sum(np.count_nonzero(row==1) for row in archive)
        base=(1+all_fail)/(2+all_seen)
        shrunk=(pos+2*base)/(count+2)
        pr=key[0]
        pr_last=last.copy()
        own=same_pr.get(pr,{})
        for i,idx in enumerate(indices):
            if idx in own:
                pr_last[i]=.1+.8*own[idx]
        experts=np.column_stack((f,last,recent,f))
        scores=(np.sum(wealth[indices]*experts,axis=1),experts.mean(1),f,last,recent,pr_last,shrunk)
        choices=[np.lexsort((np.arange(len(indices)),-f,-p))[0] for p in scores]
        y=np.array([truth[i] for i in indices],dtype=np.int8)
        if seq>=len(builds)//3:
            checked=y>=0
            bits={method:float(-np.log2(np.where(y[checked],s[checked],1-s[checked])).mean())
                  for method,s in zip(METHODS,scores)} if checked.any() else {}
            hits={m:int(y[c]==1) for m,c in zip(METHODS,choices)}
            fails=int(np.sum(y==1))
            hits.update(uniform=fails/len(y),oracle=int(fails>0))
            records.append(dict(pr=pr,build=key[1],start=start,end=end,n=len(y),fail=fails,
                                skipped=int((~checked).sum()),prior_pr=pr in same_pr,hits=hits,bits=bits,
                                picked={m:int(indices[c]) for m,c in zip(METHODS,choices)}))
        heapq.heappush(pending,(end,seq,pr,indices,y,experts))
    return records


def analyse(records):
    pr=np.array([r['pr'] for r in records])
    latent=np.array([r['fail']>0 for r in records])
    repeated=np.array([r['prior_pr'] for r in records])
    keys,ids=np.unique(pr,return_inverse=True)
    n=len(records)
    assert n>=10
    rng=np.random.default_rng(802)
    cluster=rng.integers(0,len(keys),(4000,len(keys)))
    block=(rng.integers(0,n-9,(4000,(n+9)//10))[:,:,None]+np.arange(10)).reshape(4000,-1)[:,:n]
    def bounds(x,y):
        xn=np.bincount(ids,weights=x,minlength=len(keys))
        yn=np.bincount(ids,weights=y,minlength=len(keys))
        def calc(a,b,draw):
            denom=b[draw].sum(1)
            values=a[draw].sum(1)[denom>0]/denom[denom>0]
            return np.quantile(values,[.025,.975]).tolist() if len(values) else None
        return dict(point=float(x.sum()/y.sum()) if y.sum() else None,
                    pr_cluster=calc(xn,yn,cluster),consecutive10=calc(x,y,block))
    hits={k:np.array([r['hits'][k] for r in records]) for k in (*METHODS,'uniform','oracle')}
    present=np.array([bool(r['bits']) for r in records])
    bits={k:np.array([r['bits'].get(k,0.) for r in records]) for k in METHODS}
    paired=hits['market']-hits['equal']
    return dict(scored=n,failing=int(latent.sum()),prior_pr_failing=int(sum(repeated&latent)),
                unique_prs=len(keys),failing_prs=len(set(pr[latent])),one_check={
                    m:dict(hits=float(x.sum()),all=float(x.mean()),failing=float(x.sum()/latent.sum()) if latent.any() else None,
                           same_pr_failing=float(x[repeated].sum()/(repeated&latent).sum()) if (repeated&latent).any() else None)
                    for m,x in hits.items()},market_minus_equal=bounds(paired,latent),
                discordant=dict(market=int(sum(paired>0)),equal=int(sum(paired<0))),
                equal_minus_market_bits=bounds(bits['equal']-bits['market'],present),
                bits={k:float(b.sum()/present.sum()) for k,b in bits.items()})


def self_check():
    # Flip outcomes on a long-running build and on the decision's own build.
    # Both changes must leave first choices on the overlapping decision identical.
    names={0:'a',1:'b'}
    def choices(changed):
        b=[(0,1,('p',0),{0:0,1:1}), (2,20,('p',1),{0:changed,1:1-changed}),
           (3,4,('p',2),{0:1-changed,1:changed})]
        return replay(b,names)[-1]['picked']
    assert choices(0)==choices(1)
    assert choices(0)['pr_latest']==1


def main():
    self_check()
    meta,digest=metadata()
    for project in MD5:
        builds,names,counts=ingested(project,meta[project])
        records=replay(builds,names)
        report=dict(project=project,metadata_sha256=digest,counts=counts,summary=analyse(records))
        (HERE/f'{project}-summary.json').write_text(json.dumps(report,indent=2)+'\n')
        with (HERE/f'{project}-builds.json.gz').open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
                f.write(json.dumps(records,separators=(',',':')).encode())
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
