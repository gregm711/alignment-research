"""Restore one attributed observation to a compacted handoff. Protocol in README."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from cases import CASES
from screen import HERE, MODEL, NAMES, REPAIR_SYSTEM, answer, materials, repair_prompt, save, score

WRITER_PATH = HERE / 'results' / MODEL.replace(':', '-')
TOKENIZER_SHA = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'


def observations(history):
    archives = {name: {} for name in (*NAMES, 'common')}
    section = None
    for line in history.splitlines():
        if line.startswith('Probe'):
            section = next((name for name in NAMES if f'from {name}:' in line), None)
            if line.startswith('Probe 1:'):
                section = 'common'
        elif section and line.startswith('  repair('):
            arg, output = line.strip().removesuffix(' on both').split(' -> ')
            archives[section][arg] = output
    a, b = (archives[name] for name in NAMES)
    key = next(key for key in a if a[key] != b[key])
    shared, value = next(iter(archives['common'].items()))
    def row(arg, x, y):
        return f'Archived call {arg}\nalder returned {x}\nbirch returned {y}\n'
    return {'observed': row(key, a[key], b[key]), 'swapped': row(key, b[key], a[key]),
            'common': row(shared, value, value)}


def replace_start(tok, note, record):
    ids = lambda text: tok.encode(text, add_special_tokens=False).ids
    budget = len(ids(note))
    original = ids(note)
    assert len(ids(record)) <= budget
    for count in range(budget, -1, -1):
        candidate = record + tok.decode(original[:count], skip_special_tokens=False)
        if (len(ids(candidate)) == budget and
                len(ids('handoff:\n'+candidate+'\n\nFresh routing')) ==
                len(ids('handoff:\n'+note+'\n\nFresh routing'))):
            return candidate
    for count in range(budget, 0, -1):
        candidate = record + tok.decode(original[-count:], skip_special_tokens=False)
        if len(ids(candidate)) == budget and len(ids('handoff:\n'+candidate+'\n\nFresh routing')) == len(ids('handoff:\n'+note+'\n\nFresh routing')):
            return candidate
    raise ValueError('cannot exactly match ordinary handoff length')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--recount', action='store_true')
    args = parser.parse_args()
    from tokenizers import Tokenizer
    path = Path(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TOKENIZER_SHA
    tok = Tokenizer.from_file(str(path))
    if not (args.check or args.recount):
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5) as stream:
            digest = next(m['digest'] for m in json.load(stream)['models'] if m['name'] == args.model)
    destination = HERE / 'results' / ('restore-' + args.model.replace(':','-'))
    for index, case in enumerate(CASES):
        frozen = json.loads((WRITER_PATH / (case.name + '.json')).read_text())
        history, expected = materials(case, index)
        assert frozen['history'] == history
        note = frozen['notes']['ordinary']['note']
        notes = {k: replace_start(tok, note, v) for k,v in observations(history).items()}
        notes.update(full=history, ordinary=note, evidence=frozen['notes']['evidence']['note'],
                     none='(No debugging history retained.)')
        output = destination / (case.name + '.json')
        data = json.loads(output.read_text()) if output.exists() else {'notes': notes, 'trials': {}}
        assert data['notes'] == notes
        if not (args.check or args.recount):
            destination.mkdir(parents=True, exist_ok=True)
            assert data.setdefault('model_digest', digest) == digest
        for j, name in enumerate(NAMES):
            base_prompt = repair_prompt(case, name, note)
            for arm, saved_note in notes.items():
                key = f'{name}-{arm}'
                prompt = repair_prompt(case, name, saved_note)
                fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
                if arm in ('observed','swapped','common'):
                    assert len(tok.encode(prompt).ids) == len(tok.encode(base_prompt).ids), (case.name, arm)
                elif args.model == MODEL:
                    assert fingerprint == frozen['trials'][key]['prompt_sha256']
                    continue
                if args.check:
                    continue
                if key not in data['trials'] and not args.recount:
                    trial = answer(args.model, REPAIR_SYSTEM, prompt, 220)
                    trial['prompt_sha256'] = fingerprint
                    trial['score'] = score(case, trial['text'], expected[j])
                    trial['opposite_passed'] = trial['score'].get('outputs') == expected[1-j]
                    data['trials'][key] = trial
                    save(output, data)
                else:
                    trial = data['trials'][key]
                    assert fingerprint == trial['prompt_sha256']
                    assert trial['score'] == score(case, trial['text'], expected[j])
                    assert trial['opposite_passed'] == (trial['score'].get('outputs') == expected[1-j])
        print(case.name, {k:t['score']['passed'] for k,t in data['trials'].items()}, flush=True)


if __name__ == '__main__':
    main()
