"""Can the reader recover the attributed observation when no patch is needed?"""
import argparse
import hashlib
import json
import urllib.request

from cases import CASES
from restore import WRITER_PATH, observations
from screen import HERE, MODEL, NAMES, answer, display, materials, outcome, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--recount', action='store_true')
    parser.add_argument('--novel', action='store_true')
    args = parser.parse_args()
    prefix = 'transfer-' if args.novel else 'readout-'
    path = HERE/'results'/(prefix+args.model.replace(':','-')+'.json')
    data = json.loads(path.read_text()) if path.exists() else {'trials':{}}
    if not args.recount:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as resp:
            digest = next(m['digest'] for m in json.load(resp)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest',digest)==digest
    system = 'Read the archive observations. Return a JSON object with choice A or B.'
    schema = {'type':'object','required':['choice'],'properties':{'choice':{'enum':['A','B']}},
              'additionalProperties':False}
    for i,c in enumerate(CASES):
        filename = c.name+'.json'
        original = json.loads((WRITER_PATH/filename).read_text())
        notes = json.loads((HERE/'results'/'restore-qwen2.5-coder-7b'/filename).read_text())['notes']
        call,a,b = observations(original['history'])['observed'].splitlines()
        options = [a.removeprefix('alder returned '),b.removeprefix('birch returned ')]
        if args.novel:
            _,expected=materials(c,i)
            evaluated=next(j for j in range(len(c.common),len(c.common)+len(c.hidden))
                           if expected[0][j]!=expected[1][j])
            arg=c.hidden[evaluated-len(c.common)]
            assert arg not in c.captures+c.common
            call='Archived call repair('+display(arg)+')'
            options=[outcome(expected[b][evaluated]) for b in (0,1)]
        assert options[0]!=options[1]
        for route,name in enumerate(NAMES):
            for arm in ('full','ordinary','observed','swapped','none'):
                for flip in (0,1):
                    key=f'{c.name}/{name}/{arm}/{flip}'
                    first=(i+flip)%2
                    expected='AB'[int(route!=first)]
                    prompt=(f"Handoff:\n{notes[arm]}\n\nConfirmed route: {name}. "
                            f"What did the {name} archive return for {call.removeprefix('Archived call ')}?\n"
                            f"A: {options[first]}\nB: {options[1-first]}")
                    if args.novel:
                        prompt=prompt.replace(f'What did the {name} archive return for ',f'Infer what the {name} build would return on this new input: ')
                    digest_prompt=hashlib.sha256(prompt.encode()).hexdigest()
                    if key not in data['trials']:
                        assert not args.recount
                        trial=answer(args.model,system,prompt,18,reply_format=schema)
                        trial.update(prompt_sha256=digest_prompt,expected=expected)
                        data['trials'][key]=trial
                        save(path,data)
                    trial=data['trials'][key]
                    assert trial['prompt_sha256']==digest_prompt and trial['expected']==expected
        print(c.name,flush=True)
    for name in NAMES:
        for arm in ('full','ordinary','observed','swapped','none'):
            counts=[0,0]
            for c in CASES:
                correct=[]
                for flip in (0,1):
                    t=data['trials'][f'{c.name}/{name}/{arm}/{flip}']
                    try:
                        choice=json.loads(t['text'])['choice']
                    except (ValueError,KeyError,TypeError):
                        choice=None
                    correct.append(choice==t['expected'])
                counts[0]+=all(correct)
                counts[1]+=not any(correct)
            print(name,arm,'both correct / both opposite',*counts)


if __name__=='__main__':
    main()
