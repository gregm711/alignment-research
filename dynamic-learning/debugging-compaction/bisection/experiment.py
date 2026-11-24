"""Fork a debugging decision after an audit invalidates a worker's results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from restore import TOKENIZER_SHA
from screen import MODEL,answer,save

WORKERS=('red','blue')
OFFSETS=(2,0,3,1,1,2,0,3)
SYS='You investigate regressions across software revisions. Test results switch once from failing to passing.'
INT={'type':'integer','minimum':0,'maximum':12}
INTERVAL={'type':'object','properties':{'last_failure':INT,'first_pass':INT},
          'required':['last_failure','first_pass'],'additionalProperties':False}
GUESS={'type':'object','properties':{'first_pass':INT},'required':['first_pass'],'additionalProperties':False}


def scenario(case,index,correction):
    truth=case['position']
    low=truth-OFFSETS[index]-1
    high=low+4
    other=low+1+(OFFSETS[index]+1+index%3)%4
    assert 0<=low<truth<=high<=12 and other!=truth
    tight=WORKERS[(index+int(correction))%2]
    trusted=WORKERS[1-WORKERS.index(tight)] if correction else tight
    rows=[]
    for worker in WORKERS:
        precise=worker==tight
        boundary=other if worker!=trusted else truth
        points=([0,max(0,boundary-3),boundary-1,boundary,min(12,boundary+2),12] if precise
                else [0,max(0,low-2),low,high,min(12,high+2),12])
        for requested in points:
            served=min(12,max(0,requested+truth-boundary))
            rows.append((worker,requested,case['measurements'][served]['passed']))
    random.Random(941+index*2+int(correction)).shuffle(rows)
    return rows,trusted


def bounds(rows):
    return max(i for _,i,passed in rows if not passed),min(i for _,i,passed in rows if passed)


def search(case,rows,trusted,interval):
    left,right=interval
    valid=type(left) is int and type(right) is int and 0<=left<right<=12
    if not valid:left,right=0,12
    checks=[]
    for _ in range(2):
        if right-left<=1:break
        probe=(left+right)//2
        passed=case['measurements'][probe]['passed']
        checks.append((probe,passed))
        if passed:right=probe
        else:left=probe
    final=(left+1+right)//2
    ledger={(0,False),(12,True),*checks}
    ledger.update((i,passed) for worker,i,passed in rows if worker==trusted)
    certified=(final-1,False) in ledger and (final,True) in ledger
    correct=final==case['position']
    assert not certified or correct
    return dict(valid=valid,checks=[list(pair) for pair in checks],nomination=final,correct=correct,certified=certified)


def pad(tok,text,limit=96):
    ids=lambda s:tok.encode(s,add_special_tokens=False).ids
    original=ids(text.strip())
    for length in range(min(limit,len(original)),-1,-1):
        note=tok.decode(original[:length],skip_special_tokens=False)+'\nPadding:'
        size=len(ids(note))
        if size<=limit:
            padded=note+' .'*(limit-size)
            assert len(ids(padded))==limit
            return padded
    raise ValueError('padding marker exceeds budget')


def object_answer(raw):
    try:
        value=json.loads(raw)
        return value if isinstance(value,dict) else {}
    except ValueError:
        return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--recount',action='store_true')
    parser.add_argument('--model',default=MODEL)
    args=parser.parse_args()
    measurements=json.loads((HERE/'measurements.json').read_text())
    path=HERE/(args.model.replace(':','-')+'.json')
    d=json.loads(path.read_text()) if path.exists() else {'episodes':{}}
    if not args.check:
        from tokenizers import Tokenizer
        f=Path(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
        assert hashlib.sha256(f.read_bytes()).hexdigest()==TOKENIZER_SHA
        tok=Tokenizer.from_file(str(f))
        if not args.recount:
            with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
                digest=next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
            assert d.setdefault('model_digest',digest)==digest
    for i,(name,case) in enumerate(measurements.items()):
        if not case['eligible']:continue
        for correction in (False,True):
            rows,trusted=scenario(case,i,correction)
            pooled=bounds(rows)
            by={w:bounds([r for r in rows if r[0]==w]) for w in WORKERS}
            other=next(w for w in WORKERS if w!=trusted)
            assert pooled[1]-pooled[0]==1
            assert (pooled[1]!=case['position'])==correction
            assert search(case,rows,trusted,by[trusted])['certified']
            assert search(case,rows,trusted,pooled)['correct']==(not correction)
            assert search(case,rows,trusted,by[other])['correct']==(not correction)
            if args.check:continue
            key=name+('/correct' if correction else '/confirm')
            episode=d['episodes'].setdefault(key,{'trials':{}})
            log='\n'.join(f'{w} checked revision {j}: '+('PASS' if p else 'FAIL') for w,j,p in rows)
            issue=(f'The regression test for {name} changed from FAIL to PASS exactly once in revisions 0 through 12. '
                   'Independent release checks establish that 0 fails and 12 passes. Find the first passing revision.')
            if 'first' not in episode:
                assert not args.recount
                episode['first']=answer(args.model,SYS,issue+'\nWorker logs:\n'+log+
                    '\nBased on the current observations, what is your provisional first passing revision?',24,reply_format=GUESS)
                save(path,d)
            guess=object_answer(episode['first']['text']).get('first_pass')
            if 'summary' not in episode:
                assert not args.recount
                episode['summary']=answer(args.model,SYS,issue+'\n'+log+f'\nProvisional answer: {guess}. '
                    'Write a debugging handoff describing the experiments, results and current conclusion in at most 96 tokens.',96)
                save(path,d)
            def table(swapped):
                records=[]
                for w in WORKERS:
                    l,h=by[WORKERS[1-WORKERS.index(w)] if swapped else w]
                    records.append(f'{w}: last FAIL {l}, first PASS {h}.')
                return '\n'.join(records)
            notes={'full':log,'ordinary':pad(tok,episode['summary']['text']),'attributed':pad(tok,table(False)),
                   'swapped':pad(tok,table(True)),'draft':'No retained test log.'}
            if 'notes' not in episode:episode['notes']=notes
            assert notes==episode['notes']
            lengths=[]
            for arm,note in notes.items():
                prompt=(issue+f'\nYour provisional diagnosis was revision {guess}.\nHandoff:\n{note}\n\n'
                        f'Checkout audit: {other} reused cached results from other revisions. Discard its measurements. '
                        f'{trusted} ran the advertised source. Use its observations and the independent endpoints. '
                        'Choose the tightest supported search interval: give the latest known failing revision and earliest known passing revision. '
                        'The bisection tool will run up to two new clean midpoint tests in that interval.')
                fingerprint=hashlib.sha256(prompt.encode()).hexdigest()
                if arm not in episode['trials']:
                    assert not args.recount
                    t=answer(args.model,SYS,prompt,40,reply_format=INTERVAL)
                    t['prompt_sha256']=fingerprint
                    episode['trials'][arm]=t
                t=episode['trials'][arm]
                assert t['prompt_sha256']==fingerprint
                choice=object_answer(t['text'])
                score=search(case,rows,trusted,(choice.get('last_failure'),choice.get('first_pass')))
                if 'score' not in t:
                    assert not args.recount
                    t['score']=score
                    save(path,d)
                assert score==t['score']
                if arm in ('ordinary','attributed','swapped'):lengths.append(t['input_tokens'])
            if args.model==MODEL:assert len(set(lengths))==1,(key,lengths)
            print(key,guess,{a:t['score']['certified'] for a,t in episode['trials'].items()},flush=True)
    if args.check:
        print('All eligible cases have the required causal fork and certified deterministic recovery.')
    else:
        for branch in ('correct','confirm'):
            es=[v for k,v in d['episodes'].items() if k.endswith('/'+branch)]
            print(branch,'certified', {a:sum(e['trials'][a]['score']['certified'] for e in es) for a in es[0]['trials']})


if __name__=='__main__':
    main()
