"""Can the reader recover the attributed observation when no patch is needed?"""
import argparse
import hashlib
import json
import urllib.request

from cases import CASES
from restore import WRITER_PATH, observations
from screen import HERE, MODEL, NAMES, answer, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--recount', action='store_true')
    args = parser.parse_args()
    path = HERE/'results'/('readout-'+args.model.replace(':','-')+'.json')
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
