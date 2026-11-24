"""Keep all four worker-count measurements within the original memory budget."""
import argparse
from collections import defaultdict
import hashlib
import json
import re
import urllib.request

from verdict import HERE, MODEL, content, decision, decision_prompt, load_messages, native_tokenizer, output_lines, pad, save


def grouped(lines):
    groups = defaultdict(list)
    for line in lines:
        match = re.fullmatch(r'\(n_jobs=(\d+)\) kmeans\.inertia_ = ([\d.]+)', line)
        assert match, line
        groups[match[2]].append(int(match[1]))
    table = '\n'.join(f'n_jobs {",".join(map(str,workers))}: {value}' for value,workers in groups.items())
    return 'Measured kmeans.inertia_\n' + table, len(groups)


def unpack(table):
    rows = {}
    for line in table.splitlines()[1:]:
        workers,value = line.removeprefix('n_jobs ').split(': ')
        for raw in workers.split(','):
            worker = int(raw)
            assert worker not in rows
            rows[worker] = f'(n_jobs={worker}) kmeans.inertia_ = {value}'
    assert sorted(rows)==[1,2,3,4]
    return [rows[i] for i in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--recount',action='store_true')
    args = parser.parse_args()
    tok = native_tokenizer(args.model)
    model = args.model.replace(':','-')
    root = HERE/'results'
    path = root/('inertia-'+model+'.json')
    data = json.loads(path.read_text()) if path.exists() else {'trials':{}}
    row = next(r for r in json.loads((HERE/'independent.json').read_text()) if r['instance'].endswith('-9288'))
    messages = load_messages(row)
    issue = content(messages[1]).split('<pr_description>',1)[1].split('</pr_description>',1)[0].strip()
    if not args.check:
        baseline = json.loads((root/('independent-'+model+'.json')).read_text())
        if not args.recount:
            with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
                digest = next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
            assert digest==baseline['model_digest']
            assert data.setdefault('model_digest',digest)==digest
    for state,field in [('reopen','broken_observation'),('close','verified_observation')]:
        lines = output_lines(content(messages[row[field]]))
        table,count = grouped(lines)
        assert unpack(table)==lines and count==(2 if state=='reopen' else 1)
        for arm,summary in [('groups',table),('count',table+f'\nDistinct measured inertias: {count}.'),
                            ('wrong',table+f'\nDistinct measured inertias: {3-count}.')]:
            note = pad(tok,summary)
            assert note.partition('\nPadding:')[0]==summary
            if args.check:continue
            prompt = decision_prompt(issue,row['command'],note)
            fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
            key = state+'/'+arm
            if key not in data['trials']:
                assert not args.recount
                trial = decision(args.model,prompt,8192)
                trial.update(note=note,prompt_sha256=fingerprint)
                data['trials'][key] = trial
                save(path,data)
            trial = data['trials'][key]
            reference = baseline['runs'][row['instance']+'/'+state]['trials']['excerpt']
            assert trial['input_tokens']==reference['input_tokens']
            assert trial['note']==note and trial['prompt_sha256']==fingerprint
            print(key,json.loads(trial['text'])['decision'])
    if args.check:print('Reconstructed all four measurements exactly from both 96-token notes.')


if __name__=='__main__':
    main()
