"""Run or rescore the local, paired debugging-compaction screen. See README.md."""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import urllib.request

from cases import CASES
from worker import evaluate

HERE = pathlib.Path(__file__).parent
MODEL = 'qwen2.5-coder:7b'
LIMIT = 96
NAMES = ('alder', 'birch')


def function(body):
    return 'def repair(x):\n'+textwrap.indent(body, '    ')


def display(obj):
    return json.dumps(obj, separators=(',',':'),ensure_ascii=True)


def outcome(obj):
    return display(obj['value']) if 'value' in obj else 'raises '+obj['exception']


def materials(case, index):
    refs = [function(case.left),function(case.right)]
    if index%2:
        refs.reverse()
    shared = case.common + case.hidden
    expected = [evaluate(src, copy.deepcopy(shared)) for src in refs]
    bad = evaluate(function(case.source), copy.deepcopy(shared))
    assert all(v!=bad for v in expected) and expected[0]!=expected[1]
    assert expected[0][:len(case.common)]==expected[1][:len(case.common)]
    prefix = [f'Compatibility incident: {case.name}. Two supported consumer builds have independent archived behavior. A drop-in Python function is being prepared for the build named by the pending route check.',
              'Engineer: The route dashboard currently points to alder. Treat alder behavior as the working diagnosis while the trace team checks the actual request. Do not combine the two builds.',
              'Worktree source and function purpose will be available after handoff.',
              'Probe 1: smoke fixtures were replayed against both archived consumers.']
    for x in case.common:
        prefix.append(f'  repair({display(x)}) -> {outcome(evaluate(refs[0],[copy.deepcopy(x)])[0])} on both')
    for j in (1,0):
        prefix.append(f'Probe {3-j}: archive replay from {NAMES[j]}:')
        for x in case.captures:
            prefix.append(f'  repair({display(x)}) -> {outcome(evaluate(refs[j],[copy.deepcopy(x)])[0])}')
    prefix.append('Probe 4: current checkout on the same capture inputs:')
    for x in case.captures:
        prefix.append(f'  repair({display(x)}) -> {outcome(evaluate(function(case.source),[copy.deepcopy(x)])[0])}')
    prefix.append('Engineer: Source and smoke cases above are from the same checkout. No patch applied. Current diagnosis remains an alder compatibility regression. Await request routing trace.')
    return '\n'.join(prefix),expected


def post(path,body):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+path,data=json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=220) as resp:
        return json.load(resp)


def answer(model,system,user,n):
    r=post('chat',dict(model=model,stream=False,keep_alive='15m',
        options=dict(temperature=0,seed=1103,num_ctx=4096,num_predict=n),
        messages=[dict(role='system',content=system),dict(role='user',content=user)]))
    return dict(text=r['message']['content'],input_tokens=r.get('prompt_eval_count'),output_tokens=r.get('eval_count'),
                duration_ms=round(r['total_duration']/1e6),finish=r.get('done_reason'))


def score(case,raw,expected):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',raw,re.S|re.I)
    src=(blocks[0] if blocks else raw).strip()
    try:
        proc=subprocess.run([sys.executable,'-I','-S',str(HERE/'worker.py')],
            input=json.dumps({'source':src,'inputs':case.common+case.hidden}),text=True,
            cwd='/tmp',capture_output=True,timeout=4,env={'PATH':'/usr/bin:/bin'})
        res=json.loads(proc.stdout)
    except (ValueError,subprocess.TimeoutExpired) as e:
        res={'error':type(e).__name__}
    return dict(passed=res.get('outputs')==expected,**res)


def repair_prompt(case, name, note):
    return f'Debugging handoff:\n{note}\n\nFresh routing test: the affected request definitely used {name}. Match that build exactly.\nPurpose: {case.purpose}\nCurrent source:\n{function(case.source)}\nWrite the complete replacement function repair(x). Use only Python builtins and standard sequence/string methods. No imports or commentary.'


REPAIR_SYSTEM = 'You maintain a Python compatibility layer. Return one Python function. Hidden regression tests will execute it.'


def save(path,record):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        for i,c in enumerate(CASES):
            _,expect=materials(c,i)
            for j,body in enumerate([c.left,c.right]):
                which=j^(i%2)
                assert score(c,function(body),expect[which])['passed']
                assert not score(c,function(body),expect[1-which])['passed']
        print('All 16 references pass their branch and fail the opposite branch.')
        return
    dest=HERE/'results'/args.model.replace(':','-')
    if not args.recount:
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
        dest.mkdir(parents=True,exist_ok=True)
        tags=json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5))
        digest=next(v['digest'] for v in tags['models'] if v['name']==args.model)
    report={}
    for i,case in enumerate(CASES):
        prefix,expected=materials(case,i)
        path=dest/(case.name+'.json')
        data=json.loads(path.read_text()) if path.exists() else dict(case=case.name,history=prefix,notes={},trials={})
        assert data['history']==prefix
        if not args.recount:
            assert data.get('model_digest',digest)==digest
            data['model_digest']=digest
            for policy, instruction in [
                ('ordinary','Write a concise debugging handoff for the next engineer. Summarize progress, current diagnosis, and the next action.'),
                ('evidence','Write a concise debugging handoff. Preserve evidence for both possible diagnoses, including conflicting observed input/output behavior and which build exhibited it.')]:
                if policy not in data['notes']:
                    result=answer(args.model,'You summarize engineering sessions. The full trace disappears after this handoff.',
                                  prefix+'\n\n'+instruction+' Budget: at most 96 tokens. Output only the handoff.',LIMIT)
                    ids=tok.encode(result['text'],add_special_tokens=False).ids[:LIMIT]
                    result.update(retained_tokens=len(ids),note=tok.decode(ids,skip_special_tokens=False))
                    data['notes'][policy]=result
                    save(path,data)
        policies=['full','ordinary','evidence','none']
        for j,name in enumerate(NAMES):
            for policy in policies:
                key=f'{name}-{policy}'
                if args.recount or key in data['trials']:
                    trial=data['trials'][key]
                    assert score(case,trial['text'],expected[j])==trial['score']
                else:
                    note=prefix if policy=='full' else '(No debugging history retained.)' if policy=='none' else data['notes'][policy]['note']
                    prompt=repair_prompt(case,name,note)
                    trial=answer(args.model,REPAIR_SYSTEM,prompt,220)
                    trial['score']=score(case,trial['text'],expected[j])
                    trial['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
                    data['trials'][key]=trial
                    save(path,data)
                report.setdefault(key,[]).append(int(trial['score']['passed']))
        print(case.name,{k:v[-1] for k,v in report.items()},flush=True)
    print({k:sum(v) for k,v in report.items()})


if __name__=='__main__':
    main()
