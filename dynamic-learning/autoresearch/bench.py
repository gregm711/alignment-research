"""Host-side generator and scoring. Candidates see only observations through a fresh container."""
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
IMAGE = 'python:3.12-alpine'
MAXIMIZE = ('affected','unaffected','false_alarm','fresh','booked')
MINIMIZE = ('probes','bytes','bits_per_receipt')
FEATURES = [[0,*bits] for bits in itertools.product(range(2),repeat=3)]


def episodes(seed, count):
    rng = random.Random(seed)
    cases = []
    for i in range(count):
        scope = i % 5
        reliability = rng.sample([.85,.55],2)
        history = [sum(rng.random()<p for _ in range(8)) for p in reliability]
        if rng.random() < .2:
            reliability.reverse()
        source = rng.randrange(2)
        endorses = bool(scope) == (rng.random() < reliability[source])
        focal = [1,1,1,1]
        menu = [focal]+rng.choices(FEATURES,k=4)
        rng.shuffle(menu)
        queries = [(row,flag) for flag in (False,True) for row in rng.sample(FEATURES,4)]
        queries.append((focal,True))
        obs = dict(menu=menu,history=history,source=source,endorses=endorses,receipts=[])
        cases.append((scope,obs,queries))
    rng.shuffle(cases)
    return cases


def value(scope,row):
    return (0,*row)[scope]


def image_id():
    return subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],
                                   text=True,timeout=10).strip()


def batch(candidate, image, stage, items):
    name = 'revision-' + uuid.uuid4().hex
    cmd = ['docker','run','--rm','--name',name,'-i','--network=none','--read-only',
           '--cap-drop=ALL','--security-opt=no-new-privileges','--user=65534:65534',
           '--pids-limit=16','--memory=128m','--cpus=1','--log-driver=none',
           '-v',f'{candidate.resolve()}:/task/candidate.py:ro',
           '-v',f'{HERE / "worker.py"}:/task/worker.py:ro',image,'python','-IB','/task/worker.py']
    # Bound the candidate's output inside the container as well as its wall time.
    code = ("import resource,runpy,sys;resource.setrlimit(resource.RLIMIT_FSIZE,(4194304,4194304));"
            "sys.stdout=open('/tmp/result','w');runpy.run_path('/task/worker.py');"
            "sys.stdout.close();sys.__stdout__.write(open('/tmp/result').read())")
    cmd[cmd.index('--read-only'):cmd.index('--read-only')] = ['--tmpfs','/tmp:rw,noexec,nosuid,size=6m']
    cmd[-2:] = ['-IB','-c',code]
    try:
        with tempfile.TemporaryFile() as err:
            result = subprocess.run(cmd,input=json.dumps([stage,items]).encode(),stdout=subprocess.PIPE,
                                    stderr=err,timeout=30)
            if result.returncode:
                err.seek(0)
                raise ValueError('Candidate execution failed: '+err.read(2048).decode(errors='replace'))
        out = json.loads(result.stdout)
        if not isinstance(out,list) or len(out)!=len(items):
            raise ValueError('Wrong number of candidate outputs')
        return out
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
        raise


def evaluate(candidate,seed,count,image,examples=True):
    cases=episodes(seed,count)
    memory=[None]*count
    for turn in range(3):
        active=[i for i in range(count) if memory[i] is None]
        if not active:
            break
        decisions=batch(candidate,image,'step',[cases[i][1] for i in active])
        for i,decision in zip(active,decisions):
            scope,obs,_=cases[i]
            if isinstance(decision,str):
                if len(decision)>64 or bytes.fromhex(decision).hex()!=decision:
                    raise ValueError('State must be canonical hex, at most 32 bytes')
                memory[i]=decision
            else:
                if (type(decision) is not int or turn==2 or not 0<=decision<len(obs['menu'])
                        or any(j==decision for j,_ in obs['receipts'])):
                    raise ValueError('Invalid or repeated probe')
                obs['receipts'].append([decision,value(scope,obs['menu'][decision])])
    # Evaluate next-observation forecasts under a common, teacher-forced schedule.
    # Each depth runs separately so a later input cannot disclose an earlier target.
    losses=[]
    for depth in range(3):
        views=[]
        truths=[]
        for scope,obs,_ in cases:
            menu=obs['menu']
            visible=dict(obs,receipts=[[q,value(scope,menu[q])] for q in range(depth)])
            views.append([visible,menu[depth]])
            truths.append(value(scope,menu[depth]))
        forecasts=batch(candidate,image,'forecast',views)
        bits=0.
        for y,p in zip(truths,forecasts):
            if type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1:
                raise ValueError('Probability must be finite and in [0,1]')
            p=max(1e-6,min(1-1e-6,p))
            bits-=math.log2(p if y else 1-p)
        losses.append(bits/count)
    predictions=batch(candidate,image,'answer',[(m,c[2]) for m,c in zip(memory,cases)])
    totals={k:[0,0] for k in MAXIMIZE}
    misses=[]
    for (scope,obs,queries),votes,m in zip(cases,predictions,memory):
        if not isinstance(votes,list) or len(votes)!=len(queries):
            raise ValueError('Wrong prediction count')
        for (row,old),v in zip(queries,votes):
            if type(v) is not int or v not in (0,1):
                raise ValueError('Predictions must be binary integers')
            y=value(scope,row)
            groups=['affected' if y else 'unaffected','booked' if old else 'fresh']
            if not scope:
                groups.append('false_alarm')
            for k in groups:
                totals[k][0]+=int(y==v)
                totals[k][1]+=1
            if v!=y and examples and len(misses)<4:
                misses.append(dict(scope=scope,observed=obs,row=row,old=old,predicted=v,correct=y,state=m))
    metrics={k:n/d for k,(n,d) in totals.items()}
    metrics.update(probes=sum(len(obs['receipts']) for _,obs,_ in cases)/count,
                   bytes=sum(len(m)//2 for m in memory)/count)
    metrics['bits_per_receipt']=sum(losses)/len(losses)
    return dict(metrics=metrics,counts=totals,episodes=count,prequential_bits=losses,failures=misses)
