"""Run the stored debugging observation on a frozen candidate and allow one new patch."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

from cases import CASES
from restore import TOKENIZER_SHA, WRITER_PATH
from screen import HERE, MODEL, NAMES, REPAIR_SYSTEM, answer, materials, outcome, save, score

ARMS = {'run':'observed','unrun':'observed','common':'common','swap':'swapped'}


def make_prompts(case, name, candidate, notes, tok):
    prompts = {}
    for arm, key in ARMS.items():
        row, alder, birch = notes[key].splitlines()[:3]
        assert row.startswith('Archived call repair(') and row.endswith(')')
        arg = json.loads(row[len('Archived call repair('):-1])
        expected = dict(zip(NAMES,[alder.removeprefix('alder returned '),birch.removeprefix('birch returned ')]))[name]
        if arm == 'unrun':
            report = 'Check has not been run.'
        else:
            execution = score(replace(case,common=[arg],hidden=[]), candidate, [])
            if 'outputs' in execution:
                actual = outcome(execution['outputs'][0])
                report = f"Observed {actual}. Check {'PASS' if expected==actual else 'FAIL'}."
            else:
                report = f"Check could not execute: {execution['error']}."
        prompts[arm]=(f'The affected request used {name}.\nPurpose: {case.purpose}\n'
                      f'Current candidate:\n{candidate}\nRetained archive check for {name}: {row}. '
                      f'Expected {expected}. {report}\nToken padding:')
    ending='\nReturn the complete replacement function repair(x). It must match the affected build on new inputs. Use only Python builtins and standard sequence/string methods. No imports or commentary.'
    count=lambda s: len(tok.encode(s,add_special_tokens=False).ids)
    budget=max(count(p+ending) for p in prompts.values())
    for arm,p in prompts.items():
        prompt=p+' .'*(budget-count(p+ending))+ending
        assert count(prompt)==budget
        prompts[arm]=prompt
    return prompts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recount',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer=Path(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest()==TOKENIZER_SHA
    tok=Tokenizer.from_file(str(tokenizer))
    output=HERE/'results'/'feedback-qwen2.5-coder-7b.json'
    data=json.loads(output.read_text()) if output.exists() else {'trials':{}}
    if not (args.recount or args.check):
        from urllib.request import urlopen
        with urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as response:
            digest=next(m['digest'] for m in json.load(response)['models'] if m['name']==MODEL)
        assert data.setdefault('model_digest',digest)==digest
    for i,c in enumerate(CASES):
        original=json.loads((WRITER_PATH/(c.name+'.json')).read_text())
        if 'model_digest' in data:
            assert data['model_digest']==original['model_digest']
        notes=json.loads((HERE/'results'/'restore-qwen2.5-coder-7b'/(c.name+'.json')).read_text())['notes']
        _,expected=materials(c,i)
        for j,name in enumerate(NAMES):
            starting=original['trials'][name+'-ordinary']['text']
            blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',starting,re.S|re.I)
            candidate=(blocks[0] if blocks else starting).strip()
            prompts=make_prompts(c,name,candidate,notes,tok)
            if args.check:
                continue
            lengths=set()
            for arm,prompt in prompts.items():
                key=f'{c.name}/{name}/{arm}'
                fingerprint=hashlib.sha256(prompt.encode()).hexdigest()
                if key not in data['trials']:
                    assert not args.recount
                    trial=answer(MODEL,REPAIR_SYSTEM,prompt,220)
                    trial.update(prompt_sha256=fingerprint,score=score(c,trial['text'],expected[j]))
                    data['trials'][key]=trial
                    save(output,data)
                t=data['trials'][key]
                assert fingerprint==t['prompt_sha256']
                assert t['score']==score(c,t['text'],expected[j])
                lengths.add(t['input_tokens'])
            assert len(lengths)==1, (c.name,name,lengths)
        print(c.name,flush=True)
    if not args.check:
        for name in NAMES:
            for arm in ARMS:
                print(name,arm,[c.name for c in CASES if data['trials'][f'{c.name}/{name}/{arm}']['score']['passed']])


if __name__=='__main__':
    main()
