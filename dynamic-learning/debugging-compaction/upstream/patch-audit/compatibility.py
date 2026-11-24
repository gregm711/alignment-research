"""Check preservation of existing behavior that the registered grade did not test."""
import argparse
from dataclasses import replace
import json

from experiment import CASES,HERE,MODEL,apply_edit,blob,run,save,section,sha


PROBES = {
    'sliced': """for seq in ('', 'ab', 'abcdef', tuple(range(5))):
    for strict in (False, True):
        assert list(mi.sliced(seq,0,strict=strict))==[]
    for n in (1,2,3,8):
        assert list(mi.sliced(seq,n))==[seq[i:i+n] for i in range(0,len(seq),n)]""",
    'chunked_even': """for size in range(13):
    seq=list(range(size))
    for n in (1,2,3,5,12):
        groups=(size+n-1)//n
        width,remainder=divmod(size,groups) if groups else (0,0)
        expected=[]; pos=0
        for j in range(groups):
            end=pos+width+int(j<remainder)
            expected.append(seq[pos:end]); pos=end
        assert list(mi.chunked_even(seq,n))==expected
        assert list(mi.chunked_even(iter(seq),n))==expected""",
    'bucket.__init__': """for validator in (None,lambda key: key==1):
    b=mi.bucket(iter([0,1,2,3,4]),key=lambda x:x%2,validator=validator)
    assert list(b[1])==[1,3]
    assert list(b[0])==([0,2,4] if validator is None else [])
assert list(mi.bucket([],key=str))==[]""",
    'iterate': """from itertools import islice
for start in (-2,0,7):
    assert list(islice(mi.iterate(lambda x:x+3,start),5))==[start+3*i for i in range(5)]
def fail(n): raise ValueError('original')
g=mi.iterate(fail,None)
assert next(g) is None
try: next(g)
except ValueError as error: assert str(error)=='original'
else: raise AssertionError('callback exception disappeared')"""
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    args = parser.parse_args()
    tag = args.model.replace(':','-')
    data = json.loads((HERE/(tag+'.json')).read_text())
    result = {}
    for case in CASES:
        row = data['cases'][case.func]
        parent,fixed = [section(blob(r,'more_itertools/more.py'),case.func) for r in (case.rev+'^',case.rev)]
        own,_ = apply_edit(parent,data['calls'][row['first']['call']]['text'],case)
        starts = {'parent':parent,'fixed':fixed,'own':own}
        candidates = {sha(s):s for s in starts.values()}
        for name,branch in row['branches'].items():
            source = starts[name.split('/')[0]]
            candidate,_ = apply_edit(source,data['calls'][branch['call']]['text'],case)
            assert sha(candidate)==branch['after']
            candidates[sha(candidate)] = candidate
        assert set(candidates)==set(row['scores'])
        outcomes = {}
        for key,candidate in candidates.items():
            score = run(replace(case,withheld=PROBES[case.func]),candidate)
            outcomes[key] = score.get('withheld_passed',False)
        assert outcomes[sha(parent)] and outcomes[sha(fixed)],case.func
        result[case.func] = outcomes
        for name,branch in row['branches'].items():
            if row['scores'][branch['after']]['passed'] and not outcomes[branch['after']]:
                print('Previously passing candidate fails additional checks:',case.func,name)
    path = HERE/('compatibility-'+tag+'.json')
    if args.recount:assert result==json.loads(path.read_text())
    else:save(path,result)


if __name__=='__main__':
    main()
