"""Compare a verified failure bit with the same bit and an actual test traceback."""
import argparse
from dataclasses import replace
import json
import urllib.request

from compatibility import PROBES
from experiment import CASES,HERE,MODEL,answer,apply_edit,blob,native_tokenizer,pad,replacement,run,save,section,sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    args = parser.parse_args()
    tag = args.model.replace(':','-')
    base = json.loads((HERE/(tag+'.json')).read_text())
    path = HERE/('trace-'+tag+'.json')
    data = json.loads(path.read_text()) if path.exists() else {'cases':{},'calls':{}}
    if not args.recount:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            digest = next(m['digest'] for m in json.load(stream)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest',digest)==base['model_digest']==digest
    tok = native_tokenizer(args.model)
    for case in CASES:
        parent = section(blob(case.rev+'^','more_itertools/more.py'),case.func)
        test = section(blob(case.rev,'tests/test_more.py'),case.suite+'.'+case.report)
        first = base['cases'][case.func]['first']
        source,_ = apply_edit(parent,base['calls'][first['call']]['text'],case)
        assert sha(source)==first['source_sha256']
        observed = run(case,source,focal_only=True,include_trace=True)
        status = 'Current test: '+('PASS' if observed['named_passed'] else 'FAIL')
        assert bool(observed['trace'])!=observed['named_passed']
        init = dict(source_sha256=sha(source),observed=observed,arms={})
        if case.func not in data['cases']:
            assert not args.recount
            data['cases'][case.func] = init
        record = data['cases'][case.func]
        assert record['source_sha256']==sha(source) and record['observed']==observed
        trace_ids = tok.encode(observed['trace'].strip(),add_special_tokens=False).ids
        detail = '\n'+tok.decode(trace_ids[-78:],skip_special_tokens=False) if trace_ids else ''
        sizes = []
        for arm,summary in [('status',status),('trace',status+detail)]:
            note = pad(tok,summary)
            assert note.partition('\nPadding:')[0]==summary
            prompt = (f'Issue: {case.symptom}\nCurrent function:\n{source}\nThe reporter supplied this actual regression test:\n{test}\n'
                      f'Execution handoff:\n{note}\nUse the observed result to review your current function. '
                      'Return KEEP if it is correct, otherwise give the complete replacement function in a Python code block. '
                      'Preserve other behavior and the original function signature. You may omit the docstring.')
            key = sha(prompt)
            if key not in data['calls']:
                assert not args.recount
                data['calls'][key] = answer(args.model,'You maintain a Python iterator library.',prompt,900,context=8192)
                save(path,data)
            trial = data['calls'][key]
            sizes.append(trial['input_tokens'])
            assert len(tok.encode(prompt,add_special_tokens=False).ids)<=sizes[-1]<8192-900
            candidate = source
            if trial['text'].strip()=='KEEP':action='keep'
            else:
                try:
                    candidate = replacement(trial['text'],case)
                    action = 'edit' if candidate!=source else 'unchanged'
                except (SyntaxError,ValueError):
                    action = 'unusable'
            score = run(case,candidate,audit=True)
            extra = run(replace(case,withheld=PROBES[case.func]),candidate).get('withheld_passed',False)
            result = dict(call=key,note=note,action=action,source_sha256=sha(candidate),registered=score,compatible=extra)
            if arm not in record['arms']:
                assert not args.recount
                record['arms'][arm] = result
                save(path,data)
            assert record['arms'][arm]==result
            print(case.func,arm,action,score['passed'] and extra,flush=True)
        assert sizes[0]==sizes[1],(case.func,sizes)


if __name__=='__main__':
    main()
