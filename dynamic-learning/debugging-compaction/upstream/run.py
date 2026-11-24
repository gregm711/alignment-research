"""Run or rescore the historical repair panel against the pinned upstream classes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

from panel import CASES, blob, neighbor, replacement, run, section, test_information

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from incident import extracted
from restore import TOKENIZER_SHA
from screen import MODEL, answer, save

HERE=Path(__file__).resolve().parent
SYSTEM='You maintain a Python iterator library. Diagnose the reported behavior and repair the affected function.'
CODE='Return the whole replacement function in a single Python code block. Keep its existing name, interface and other behavior. You may omit its docstring. The original module imports will remain available.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    args=parser.parse_args()
    dest=HERE/'results'/args.model.replace(':','-')
    if not args.recount:
        from tokenizers import Tokenizer
        path=Path(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==TOKENIZER_SHA
        tok=Tokenizer.from_file(str(path))
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            digest=next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        dest.mkdir(parents=True,exist_ok=True)
    for c in CASES:
        path=dest/(c.func+'.json')
        data=json.loads(path.read_text()) if path.exists() else {'revision':c.rev,'trials':{}}
        assert data['revision']==c.rev
        if not args.recount:
            assert data.setdefault('model_digest',digest)==digest
        original=section(blob(c.rev+'^','more_itertools/more.py'),c.func)
        examples,report=test_information(c)
        start=f'Report: {c.symptom}\nAffected function: {c.func}\nParent checkout:\n{original}'
        options={'existing_tests':examples,'nearby_source':neighbor(c)}
        if 'tool' not in data:
            assert not args.recount
            q=answer(args.model,SYSTEM,start+'\nChoose one inspection: existing_tests or nearby_source.',
                     20,reply_format={'type':'object','properties':{'tool':{'enum':list(options)}},
                                      'required':['tool'],'additionalProperties':False})
            data['selection']=q
            data['tool']=json.loads(q['text'])['tool']
            save(path,data)
        tool=data['tool']
        observation=options[tool]
        history=f"Initial report: {c.symptom}\nTOOL {tool}:\n{observation}"
        if 'first' not in data:
            assert not args.recount
            first=answer(args.model,SYSTEM,start+'\nInspection result:\n'+observation+
                         '\nState your leading diagnosis in one sentence before the code. '+CODE,650)
            first['score']=run(c,first['text'])
            data['first']=first
            save(path,data)
        first=data['first']
        if args.recount:
            assert run(c,first['text'])==first['score']
        history+='\nProvisional diagnosis: '+first['text'].partition('```')[0].strip()
        if 'notes' not in data:
            assert not args.recount
            ordinary=answer(args.model,'You write concise debugging handoffs.',
                            history+'\nProposed edit:\n'+first['text']+'\nSummarize the progress, current diagnosis and next action in at most 96 tokens.',96)
            note=tok.decode(tok.encode(ordinary['text'],add_special_tokens=False).ids[:96])
            data['summary']=ordinary
            data['notes']={'full':history,'ordinary':note,'extract':extracted(tok,[{'probe':tool,'observation':observation}],note),
                           'none':'No earlier debugging narrative.'}
            save(path,data)
        try:
            current=replacement(first['text'],c)
        except (ValueError,SyntaxError):
            current='No parseable edit applied. The checkout still contains the original function.'
        for arm,note in data['notes'].items():
            prompt=(f'Handoff:\n{note}\nOriginal issue: {c.symptom}\nParent source:\n{original}\n'
                    f'Current candidate:\n{current}\nThe reporter has now supplied the actual regression test:\n{report}\n'
                    f'Review the candidate against this evidence and revise it if necessary. {CODE}')
            fingerprint=hashlib.sha256(prompt.encode()).hexdigest()
            if arm not in data['trials']:
                assert not args.recount
                trial=answer(args.model,SYSTEM,prompt,650)
                trial['score']=run(c,trial['text'])
                trial['prompt_sha256']=fingerprint
                data['trials'][arm]=trial
                save(path,data)
            else:
                trial=data['trials'][arm]
                assert trial['prompt_sha256']==fingerprint
                if args.recount:
                    assert run(c,trial['text'])==trial['score']
        print(c.func,first['score']['passed'],{k:v['score']['passed'] for k,v in data['trials'].items()},flush=True)


if __name__=='__main__':
    main()
