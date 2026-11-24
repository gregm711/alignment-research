"""Apply four anonymous reader decisions per arm to six measured version histories."""
import argparse
import hashlib
import json
import urllib.request

from experiment import GUESS, HERE, INTERVAL, MODEL, SYS, WORKERS, answer, bounds, native_tokenizer, object_answer, pad, save, scenario, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--recount', action='store_true')
    args = parser.parse_args()
    cases = list(json.loads((HERE/'measurements.json').read_text()).values())
    measured = {case['position']:case for case in cases}
    worlds = json.loads((HERE/'identity-results.json').read_text())['worlds']
    # Independently check the width shortcut on the original, unequal-width benchmark.
    for i, case in enumerate(cases):
        for correction in (False, True):
            rows, trusted = scenario(case, i, correction)
            intervals = [bounds([r for r in rows if r[0]==w]) for w in WORKERS]
            assert search(case, rows, trusted, max(intervals,key=lambda v:v[1]-v[0]))['certified']
    path = HERE / ('matched-'+args.model.replace(':','-')+'.json')
    data = json.loads(path.read_text()) if path.exists() else {'logs':{}}
    tok = native_tokenizer(args.model)
    if not (args.check or args.recount):
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            digest = next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest',digest)==digest
    outcomes = []
    issue = 'Versions 0 through 12 have one transition from FAIL to PASS. Independent checks establish that 0 fails and 12 passes.'
    for left_worker in WORKERS:
        right_worker = next(w for w in WORKERS if w!=left_worker)
        rows = [(left_worker,2,False),(right_worker,5,False),(left_worker,6,True),(right_worker,9,True)]
        log = '\n'.join(f'{w} checked revision {v}: '+('PASS' if p else 'FAIL') for w,v,p in rows)
        entry = data['logs'].setdefault(left_worker,{'audits':{}})
        if not args.check and 'summary' not in entry:
            assert not args.recount
            entry['first'] = answer(args.model,SYS,issue+'\n'+log+'\nWhat is your provisional first passing revision?',24,reply_format=GUESS)
            guess = object_answer(entry['first']['text'])['first_pass']
            entry['summary'] = answer(args.model,SYS,issue+'\n'+log+f'\nYour provisional diagnosis was revision {guess}. '
                                      'Write a debugging handoff describing the experiments, results and current conclusion in at most 96 tokens.',96)
            save(path,data)
        for trusted in WORKERS:
            stale = next(w for w in WORKERS if w!=trusted)
            eligible = sorted(int(t) for t in worlds if (int(t)<6)==(trusted==left_worker))
            target = bounds([r for r in rows if r[0]==trusted])
            assert len(eligible)==3 and target[1]-target[0]==4
            for t in eligible:
                witness = worlds[str(t)]['witnesses']
                assert [(v['requested'],v['passed']) for v in witness]==[(v,p) for _,v,p in rows]
                assert all((v['worker']=='blue')==(w==trusted) for v,(w,_,_) in zip(witness,rows))
                assert search(measured[t],rows,trusted,target)['certified']
            if args.check:continue
            guess = object_answer(entry['first']['text'])['first_pass']
            def table(workers):
                return '\n'.join(f'{w}: last FAIL {l}, first PASS {h}.' for w,(l,h) in zip(workers,[(2,6),(5,9)]))
            notes = {'full':log,'ordinary':pad(tok,entry['summary']['text']),
                     'attributed':pad(tok,table((left_worker,right_worker))), 'swapped':pad(tok,table((right_worker,left_worker))),
                     'anonymous':pad(tok,table(('unknown','unknown')))}
            trials = entry['audits'].setdefault(trusted,{})
            for arm,note in notes.items():
                prompt = (issue+f'\nYour provisional diagnosis was revision {guess}.\nHandoff:\n{note}\n\n'
                          f'Checkout audit: {stale} reused cached results from other revisions. Discard its measurements. '
                          f'{trusted} ran the advertised source. Choose the latest known failing revision and earliest known passing revision. '
                          'A bisection tool will run at most two clean midpoint tests within your interval.')
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                if arm not in trials:
                    assert not args.recount
                    trials[arm] = answer(args.model,SYS,prompt,40,reply_format=INTERVAL)
                    trials[arm].update(note=note,prompt_sha256=digest)
                trial = trials[arm]
                assert trial['note']==note and trial['prompt_sha256']==digest
                choice = object_answer(trial['text'])
                interval = (choice.get('last_failure'),choice.get('first_pass'))
                scores = {str(t):search(measured[t],rows,trusted,interval) for t in eligible}
                if 'scores' not in trial:
                    assert not args.recount
                    trial['scores'] = scores
                    save(path,data)
                assert trial['scores']==scores
                outcomes.append((arm,left_worker,interval==target,scores,guess))
            assert len({t['input_tokens'] for a,t in trials.items() if a!='full'})==1, (left_worker,trusted)
    if args.check:
        print('Verified the width shortcut on 16 forks and 12 anonymous matched replays.')
        return
    for arm in notes:
        selected=[row for row in outcomes if row[0]==arm]
        assert len(selected)==4
        both=sum(all(r[2] for r in selected if r[1]==w) for w in WORKERS)
        cert=sum(s['certified'] for _,_,_,scores,_ in selected for s in scores.values())
        wrong=[s for _,_,_,scores,guess in selected for t,s in scores.items() if int(t)!=guess]
        print(arm,'exact',sum(r[2] for r in selected),'/4; both audits',both,'/2; certified',cert,
              '/12; certified after wrong first guess',sum(s['certified'] for s in wrong),'/',len(wrong))
    print('Provisional guesses by assignment', {w:object_answer(e['first']['text']) for w,e in data['logs'].items()})


if __name__=='__main__':
    main()
