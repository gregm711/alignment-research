"""Apply one agent-authored edit after a revision-aware regression-test handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from cases_panel import CASES
from panel import blob, replacement, run, section

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from bisection.experiment import pad
from restore import native_tokenizer
from screen import MODEL, answer, save

EDIT = {'type':'object','properties':{'old':{'type':'string'},'new':{'type':'string'}},
        'required':['old','new'],'additionalProperties':False}
INSTRUCTIONS = ('Maintain the Python function and preserve its existing interface. To edit, supply one exact, '
                'unique substring from the displayed source as old and its replacement as new. '
                'Keep indentation on lines after the first. To keep the current code, use two empty strings.')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def apply_edit(source, response, case):
    try:
        edit = json.loads(response)
        old,new = edit['old'],edit['new']
        if old==new=='':return source,'keep'
        if not isinstance(old,str) or not isinstance(new,str) or not old or source.count(old)!=1:
            return source,'unmatched'
        result = source.replace(old,new)
        replacement(result,case)
        return result,'edit' if result!=source else 'unchanged'
    except (ValueError,KeyError,TypeError,SyntaxError):
        return source,'invalid'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--recount',action='store_true')
    args = parser.parse_args()
    path = HERE/(args.model.replace(':','-')+'.json')
    data = json.loads(path.read_text()) if path.exists() else {'calls':{},'cases':{}}
    tok = native_tokenizer(args.model)
    if not (args.check or args.recount):
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            digest = next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest',digest)==digest

    def call(prompt,summary=False):
        system = 'Write a handoff for the engineer who resumes this investigation.' if summary else INSTRUCTIONS
        key = sha(system+'\n'+prompt)
        if key not in data['calls']:
            assert not args.recount
            data['calls'][key] = answer(args.model,system,prompt,96 if summary else 512,
                                        reply_format=None if summary else EDIT,context=8192)
            save(path,data)
        result = data['calls'][key]
        assert len(tok.encode(prompt,add_special_tokens=False).ids)<=result['input_tokens']<8192-512
        return key,result

    for index,case in enumerate(CASES):
        parent,fixed = [section(blob(r,'more_itertools/more.py'),case.func) for r in (case.rev+'^',case.rev)]
        report = section(blob(case.rev,'tests/test_more.py'),case.suite+'.'+case.report)
        if args.check:
            for source,expected in [(parent,False),(fixed,True)]:
                assert run(case,source,focal_only=True)=={'named_passed':expected}
                score = run(case,source,audit=True)
                assert score['passed']==score['audit']['named_passed']==score['withheld_passed']==expected
            print(case.func,'controls passed',flush=True)
            continue
        episode = data['cases'].setdefault(case.func,{'scores':{},'measurements':{},'branches':{}})
        observed = {}
        scores = {}

        def evaluate(source,focal):
            key = sha(source)
            cache = observed if focal else scores
            stored = episode['measurements' if focal else 'scores']
            if key not in cache:
                if key not in stored or args.recount:
                    result = run(case,source,focal_only=focal,audit=not focal)
                    if focal:
                        if 'named_passed' not in result:
                            raise RuntimeError(f'Could not measure {case.func} {key}: {result}')
                        result = result['named_passed']
                    if key in stored:assert stored[key]==result,(case.func,key,stored[key],result)
                    else:
                        assert not args.recount
                        stored[key] = result
                        save(path,data)
                cache[key] = stored[key]
            return cache[key]

        task = f'Issue: {case.symptom}\nFunction: {case.func}'
        first_id,first = call(task+'\nCurrent source:\n'+parent+'\nSupply your proposed repair.')
        own,action = apply_edit(parent,first['text'],case)
        initial = dict(call=first_id,action=action,source_sha256=sha(own))
        if 'first' not in episode:
            assert not args.recount
            episode['first']=initial
        assert episode['first']==initial
        assert not evaluate(parent,True) and evaluate(fixed,True)
        for start,source in [('own',own),('parent',parent),('fixed',fixed)]:
            current = sha(source)[:12]
            passed = evaluate(source,True)
            opposite = parent if passed else fixed
            old = sha(opposite)[:12]
            assert current!=old
            ledger = {current:'PASS' if passed else 'FAIL',old:'FAIL' if passed else 'PASS'}
            order = [old,current] if index%2 else [current,old]
            receipts = '\n'.join(f'{tag}: {ledger[tag]}' for tag in order)
            history = (f'Issue: {case.symptom}\nTest: {case.report}\nRecorded test outcomes by function digest:\n{receipts}\n'
                       f'The deployment tracker provisionally identifies {old} as the current function. Await an independent checkout check.')
            if start=='own':
                summary_id,summary = call(history+'\nSummarize progress, current diagnosis and next action in at most 96 tokens.',True)
                if 'summary' not in episode:
                    assert not args.recount
                    episode['summary']=summary_id
                assert episode['summary']==summary_id
                notes = {'full':history,'ordinary':pad(tok,summary['text'])}
            else:notes = {}
            for name,tag in [('stale',old),('verified',current)]:
                short = receipts+'\nCached result for current function: '+ledger[tag]
                notes[name] = pad(tok,short)
                assert notes[name].partition('\nPadding:')[0]==short
            counts = []
            before = evaluate(source,False)
            for name,note in notes.items():
                prompt = (task+f'\nHandoff:\n{note}\n\nIndependent checkout check: current function digest is {current}.\n'
                          f'Current source:\n{source}\nReporter supplied this regression test:\n{report}\n'
                          'Review the handoff against this checkout. Keep the function or give one repair.')
                key,response = call(prompt)
                after,action = apply_edit(source,response['text'],case)
                scored = evaluate(after,False)
                record = dict(call=key,note=note,action=action,before=sha(source),after=sha(after),foil=sha(opposite))
                branch = start+'/'+name
                if branch not in episode['branches']:
                    assert not args.recount
                    episode['branches'][branch]=record
                    save(path,data)
                assert record==episode['branches'][branch]
                if name in ('stale','verified'):counts.append(response['input_tokens'])
                print(case.func,branch,action,before['passed'],'->',scored['passed'],flush=True)
            assert len(set(counts))==1,(case.func,start,counts)


if __name__=='__main__':
    main()
