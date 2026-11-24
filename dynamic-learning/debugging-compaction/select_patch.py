"""Select a freely generated patch by running an example retained at handoff."""
import argparse
from dataclasses import replace
import hashlib
import json
import urllib.request

from cases import CASES
from restore import WRITER_PATH
from screen import HERE, MODEL, NAMES, answer, function, materials, outcome, save, score


def inspect_pool(case, pool, notes, expected):
    candidates=[]
    for src in pool:
        trial=score(case,src,expected[0])
        outputs=trial.get('outputs')
        target=[outputs==e for e in expected]
        checks={}
        for k in ('observed','common','swapped'):
            call,first,second=notes[k].splitlines()[:3]
            arg=json.loads(call.removeprefix('Archived call repair(')[:-1])
            res=score(replace(case,common=[arg],hidden=[]),src,[])
            actual=outcome(res['outputs'][0]) if 'outputs' in res else None
            checks[k]=dict(actual=actual,passed=[actual==first.removeprefix('alder returned '),
                                                 actual==second.removeprefix('birch returned ')])
        candidates.append(dict(score=trial,solves=target,checks=checks))
    selections={}
    for order,indices in [('forward',list(range(len(pool)))),('reverse',list(reversed(range(len(pool)))))]:
        for j,name in enumerate(NAMES):
            for arm in ('first','common','observed','swapped','oracle'):
                eligible=[i for i in indices if (arm=='first' or
                          (candidates[i]['solves'][j] if arm=='oracle' else candidates[i]['checks'][arm]['passed'][j]))]
                chosen=next(iter(eligible or indices),None)
                selections[f'{order}/{name}/{arm}']=dict(eligible=eligible,chosen=chosen,
                    passed=chosen is not None and candidates[chosen]['solves'][j])
    return candidates,selections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recount',action='store_true')
    parser.add_argument('--model',default=MODEL)
    args=parser.parse_args()
    path=HERE/'results'/('select-patch-'+args.model.replace(':','-')+'.json')
    d=json.loads(path.read_text()) if path.exists() else {'cases':{}}
    if not args.recount:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            digest=next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        assert d.setdefault('model_digest',digest)==digest
    for i,c in enumerate(CASES):
        frozen=json.loads((WRITER_PATH/(c.name+'.json')).read_text())
        notes=json.loads((HERE/'results'/'restore-qwen2.5-coder-7b'/(c.name+'.json')).read_text())['notes']
        shared='\n'.join(line.removesuffix(' on both') for line in frozen['history'].splitlines() if line.endswith(' on both'))
        prompt=(f'Function purpose: {c.purpose}\nCurrent buggy checkout:\n{function(c.source)}\n'
                f'Smoke examples all supported versions agree on:\n{shared}\n'
                'Write four plausible complete replacements for repair(x). Consider different realistic conventions '
                'for ambiguous edge cases. Each candidate must handle the shared smoke cases, use only Python '
                'builtins and sequence/string methods, and implement a distinct plausible behavior. '
                'Return a JSON array of four plain Python function strings.')
        h=hashlib.sha256(prompt.encode()).hexdigest()
        if c.name not in d['cases']:
            assert not args.recount
            trial=answer(args.model,'You propose candidate Python compatibility repairs.',prompt,1400,
                         reply_format={'type':'array','items':{'type':'string'},'minItems':4,'maxItems':4})
            trial['prompt_sha256']=h
            try:
                pool=json.loads(trial['text'])
                if not isinstance(pool,list) or len(pool)!=4 or not all(isinstance(x,str) for x in pool):
                    pool=[]
            except ValueError:
                pool=[]
            trial['pool']=pool
            d['cases'][c.name]=trial
            save(path,d)
        t=d['cases'][c.name]
        assert t['prompt_sha256']==h
        _,expected=materials(c,i)
        analysis,selections=inspect_pool(c,t['pool'],notes,expected)
        if 'selections' not in t:
            assert not args.recount
            t.update(analysis=analysis,selections=selections)
            save(path,d)
        assert (t['analysis'],t['selections'])==(analysis,selections)
        print(c.name,len(t['pool']),flush=True)
    for order in ('forward','reverse'):
        for route in NAMES:
            for arm in ('first','common','observed','swapped','oracle'):
                key=f'{order}/{route}/{arm}'
                print(key,[c.name for c in CASES if d['cases'][c.name]['selections'][key]['passed']])


if __name__=='__main__':
    main()
