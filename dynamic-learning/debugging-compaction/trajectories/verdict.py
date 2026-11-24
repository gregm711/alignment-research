"""Can a compact handoff distinguish two real runs of the same reproducer?"""
import argparse
import hashlib
import json
import re
import urllib.request

from fork import HERE, MODEL, answer, content, load_messages, native_tokenizer, save
from bisection.experiment import pad


def decision_prompt(issue, command, note):
    return (f'Issue:\n{issue}\nThe ticket was closed because {command} exited 0. '
            f'The current checkout was independently tested.\nHandoff:\n{note}\n\n'
            'Audit the closure for the behavior demonstrated by this reproducer. '
            'Close if the reported behavior now meets the requirement. Reopen if it still fails.')


def decision(model, prompt, context):
    schema = {'type':'object','properties':{'decision':{'enum':['close','reopen']}},
              'required':['decision'],'additionalProperties':False}
    return answer(model, 'Review the status of a mathematical software bug.', prompt, 20, reply_format=schema, context=context)


def output_lines(report):
    match = re.fullmatch(r'\s*<returncode>0</returncode>\s*<output>(.*?)</output>\s*', report, re.S)
    assert match
    lines = iter(match[1].splitlines())
    kept = []
    for line in lines:
        if 'DeprecationWarning:' in line:
            assert next(lines).startswith('  ')
        elif line.strip():
            kept.append(line)
    return kept


def excerpt(tok, lines):
    kept = []
    for line in reversed(lines):
        candidate = '\n'.join([line, *kept]) + '\nPadding:'
        if len(tok.encode(candidate, add_special_tokens=False).ids) > 96:
            break
        kept.insert(0, line)
    return pad(tok, '\n'.join(kept))


def demonstrated(case, lines):
    if case.endswith('-20826'):
        outputs = '\n'.join(lines)
        axes = re.findall(r'Axis (\d):\n  X tick labels visible: (\[.*?\])\n  Y tick labels visible: (\[.*?\])', outputs)
        assert [i for i,_,_ in axes]==list('0123')
        return [('True' in x,'True' in y) for _,x,y in axes]==[(False,True),(False,False),(True,True),(True,False)]
    if case.endswith('-9288'):
        values = [float(line.split(' = ')[1]) for line in lines if 'kmeans.inertia_ = ' in line]
        assert len(values)==4
        return max(values)-min(values)<1e-8
    if case.endswith('-11310'):
        return any(line.startswith('Refit time: ') and float(line.split(': ')[1])>0 for line in lines)
    if case.endswith('-13124'):
        runs = re.split(r'shuffle = True, Random seed = [12]', '\n'.join(lines))
        assert len(runs)==3
        pairs = [set(re.findall(r'Test: \[\s*(\d+)\s+(\d+)\s*\]', run)) for run in runs[1:]]
        assert all(len(group)==10 for group in pairs)
        return pairs[0]!=pairs[1]
    if case.endswith('-12419'):
        return 'Total sum of all elements: n' in lines
    if case.endswith('-13615'):
        return r'Complement(a, b) = {x, y} \ Interval(-10, 10)' in lines
    assert case.endswith('-13757')
    return {"x*Poly(x) = Poly(x**2, x, domain='ZZ')",
            "S(-2)*Poly(x) = Poly(-2*x, x, domain='ZZ')"}.issubset(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--recount', action='store_true')
    parser.add_argument('--independent', action='store_true')
    args = parser.parse_args()
    tok = native_tokenizer(args.model)
    root = HERE / 'results'
    name = 'independent' if args.independent else 'verdict'
    path = root / (name + '-' + args.model.replace(':', '-') + '.json')
    data = json.loads(path.read_text()) if path.exists() else {'runs': {}}
    manifest = 'independent.json' if args.independent else 'screen.json'
    rows = [r for r in json.loads((HERE/manifest).read_text()) if r['eligible']]
    notes = {} if args.independent else json.loads((root/'handoffs.json').read_text())
    context = 8192 if args.independent else 4096
    if not (args.recount or args.check):
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5) as stream:
            digest = next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest', digest)==digest
    for row in rows:
        case = row['instance']
        messages = load_messages(row)
        issue = content(messages[1]).split('<pr_description>',1)[1].split('</pr_description>',1)[0].strip()
        indices = ('broken_observation','verified_observation') if args.independent else ('fresh_observation','verification')
        command = row.get('command','python reproduce_issue.py')
        for state, field in zip(('reopen','close'),indices):
            index = row[field]
            assert re.findall(r'```(?:sh|bash)\n(.*?)\n```', content(messages[index-1]), re.S)==[command]
            report = content(messages[index])
            lines = output_lines(report)
            assert demonstrated(case, lines)==(state=='close')
            output_note = excerpt(tok, lines)
            if args.check:
                print(case, state, 'complete lines retained', len(output_note.split('\nPadding:')[0].splitlines()))
                continue
            run = data['runs'].setdefault(case+'/'+state, {'trials': {}})
            prior = '' if args.independent else f"Earlier investigation:\n{notes[case]['ordinary']['note']}\n"
            writer = (f"Original issue:\n{issue}\n{prior}"
                      f"A colleague ran {command} on the current checkout.\n{report}\n"
                      "Write a debugging handoff. Summarize progress, results and the next step in at most 96 tokens.")
            writer_hash = hashlib.sha256(writer.encode()).hexdigest()
            if 'summary' not in run:
                assert not args.recount
                run['summary'] = answer(args.model, 'Write short handoffs for a debugging session.', writer, 96, context=context)
                run['writer_sha256'] = writer_hash
                save(path, data)
            assert run['writer_sha256']==writer_hash
            assert len(tok.encode(writer,add_special_tokens=False).ids)<=run['summary']['input_tokens']<context-96
            arms = {'full':report, 'ordinary':pad(tok, run['summary']['text']), 'excerpt':output_note, 'exit':'No output retained.'}
            for arm, note in arms.items():
                prompt = decision_prompt(issue,command,note)
                fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
                if arm not in run['trials']:
                    assert not args.recount
                    result = decision(args.model,prompt,context)
                    result.update(note=note, prompt_sha256=fingerprint)
                    run['trials'][arm] = result
                    save(path, data)
                result = run['trials'][arm]
                assert result['note']==note and result['prompt_sha256']==fingerprint
                assert len(tok.encode(prompt,add_special_tokens=False).ids)<=result['input_tokens']<context-20
            trials = run['trials']
            assert trials['ordinary']['input_tokens']==trials['excerpt']['input_tokens'], (case,state)
        if not args.check:
            paired = [data['runs'][case+'/'+state]['trials']['exit'] for state in ('reopen','close')]
            assert paired[0]['prompt_sha256']==paired[1]['prompt_sha256']
    if not args.check:
        for arm in ('full','ordinary','excerpt','exit'):
            solved = []
            for row in rows:
                case = row['instance']
                if all(json.loads(data['runs'][case+'/'+s]['trials'][arm]['text'])['decision']==s for s in ('reopen','close')):
                    solved.append(case)
            print(arm,solved)


if __name__=='__main__':
    main()
