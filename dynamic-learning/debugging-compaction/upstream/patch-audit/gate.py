"""Replay an action gate using only a current named-test result, then grade separately."""
import argparse
import json

from experiment import CASES,HERE,MODEL,apply_edit,blob,run,save,section,sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    args = parser.parse_args()
    tag = args.model.replace(':','-')
    data = json.loads((HERE/(tag+'.json')).read_text())
    results = {}
    for case in CASES:
        row = data['cases'][case.func]
        parent,fixed = [section(blob(r,'more_itertools/more.py'),case.func) for r in (case.rev+'^',case.rev)]
        own,_ = apply_edit(parent,data['calls'][row['first']['call']]['text'],case)
        cached = {}
        def named(source):
            key = sha(source)
            if key not in cached:
                cached[key] = run(case,source,focal_only=True)['named_passed']
            return cached[key]
        for start,source in [('own',own),('parent',parent),('fixed',fixed)]:
            branch = row['branches'][start+'/verified']
            assert sha(source)==branch['before']
            old_pass = named(source)
            assert old_pass==row['measurements'][sha(source)]
            chosen = source
            retest = False
            if not old_pass:
                candidate,_ = apply_edit(source,data['calls'][branch['call']]['text'],case)
                assert sha(candidate)==branch['after']
                retest = candidate!=source
                if retest and named(candidate):
                    chosen = candidate
            # Only now inspect the withheld evaluation of the selected artifact.
            result = row['scores'][sha(chosen)]['passed']
            results.setdefault(start,{})[case.func] = dict(before=sha(source),chosen=sha(chosen),
                generated=not old_pass,retested=retest,passed=result,false_accept=named(chosen) and not result)
    path = HERE/('gate-'+tag+'.json')
    if args.recount:assert results==json.loads(path.read_text())
    else:save(path,results)
    for name,rows in results.items():
        print(name,{field:sum(r[field] for r in rows.values()) for field in ('passed','generated','retested','false_accept')})


if __name__=='__main__':
    main()
