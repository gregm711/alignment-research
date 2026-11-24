"""Load historical functions by revision and execute their real regression classes."""
import ast
from dataclasses import dataclass
import functools
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap

IMAGE = 'python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89'
REPO = Path(os.environ.get('MORE_ITERTOOLS_REPO','/tmp/alignment-more-itertools'))


@dataclass(frozen=True)
class Case:
    rev: str
    func: str
    suite: str
    report: str
    symptom: str
    withheld: str


CASES = [
    Case('71e070b35d8c136484e91189719c2a4eccd36fc3','all_unique','AllUniqueTests','test_falsey_key_is_honoured',
         'A caller that supplies a key gets a uniqueness result inconsistent with the normalized records.',
         """class K:
    def __len__(self): return 0
    def __call__(self, x): return x % 3
assert not mi.all_unique(iter([4, 7]), key=K())
assert mi.all_unique(iter([4, 8]), key=K())"""),
    Case('ce676d24136e6efa932df8a4f3bf4801b89e1dd0','split_before','SplitBeforeTest','test_empty_collection',
         'The splitter adds a group in an edge case involving a bounded number of splits.',
         """for m in (0,1,2,-1):
    assert list(mi.split_before(iter([]), bool, maxsplit=m)) == []
assert list(mi.split_before(iter([2, 4]), bool, maxsplit=0)) == [[2, 4]]"""),
    Case('e6ce1b0f65a57c459fc5004ad84158447b65b0ac','peekable._get_slice','PeekableTests','test_slicing_negative_step_omitted_stop',
         'A client reads a small slice through peekable and finds that far more source items were consumed.',
         """it = iter(range(80))
p = mi.peekable(it)
assert p[7::-3] == [7, 4, 1]
assert next(it) == 8
assert next(p) == 0
p = mi.peekable(range(10))
assert p[-2::-3] == list(range(10))[-2::-3]"""),
    Case('6b1907de293a655fb77881bf81948f95fd15bd7c','seekable.peek','SeekableTest','test_maxlen_zero_peek_does_not_drop_items',
         'A client loses the first source value after looking ahead in a seekable iterator.',
         """s = mi.seekable(iter('qrs'),maxlen=0)
assert s.peek() == s.peek() == 'q'
assert bool(s)
assert list(s) == ['q','r','s']
assert mi.seekable([],maxlen=0).peek(97) == 97"""),
    Case('edb3346f835ca917efbfda5e2d6664ab952da369','numeric_range.__reversed__','NumericRangeTests','test_empty_reversed',
         'Reverse iteration over a numeric range disagrees with reversing the materialized sequence.',
         """from decimal import Decimal
for args in [(7,2,1),(1,9,-2),(2,2), (Decimal('0'),Decimal('1'),Decimal('0.25'))]:
    x=mi.numeric_range(*args)
    assert list(reversed(x)) == list(x)[::-1]"""),
    Case('f51a53bfd2fe9504063a33ef5f4a73e30d82d0e2','interleave_evenly','InterleaveEvenlyTests','test_no_iterables',
         'An interleaving call raises unexpectedly when there is nothing to emit.',
         """for seq in ([], ()):
    assert list(mi.interleave_evenly(seq)) == []
    assert list(mi.interleave_evenly(seq, lengths=())) == []
assert list(mi.interleave_evenly([[3]])) == [3]"""),
    Case('0e6acdf9b60765ecf9634d6f5c132ac1bebc616b','chunked','ChunkedTests','test_negative',
         'The chunk-size validation disagrees with the neighboring slicing API.',
         """for n in (-2,-9):
    for strict in (True,False):
        try: list(mi.chunked(iter([1,4]),n,strict=strict))
        except ValueError as e: assert 'n must be at least 0' in str(e)
        else: raise AssertionError('negative chunk size accepted')
assert list(mi.chunked(range(5),None)) == [[0,1,2,3,4]]"""),
    Case('7847d4324a581da8d0919734a8e50b2a6573b780','value_chain','ValueChainTests','test_type_error_while_iterating',
         'Flattening an argument sometimes yields the iterable object itself after some of its elements.',
         """def fail():
    yield 7
    raise TypeError('original error')
g=mi.value_chain(fail(),'done')
assert next(g)==7
try: next(g)
except TypeError as e: assert str(e)=='original error'
else: raise AssertionError('lost iterator failure')
assert list(mi.value_chain([2], 4))==[2,4]""")
]


@functools.lru_cache(None)
def blob(rev, path):
    return subprocess.check_output(['git','-C',str(REPO),'show',rev+':'+path],text=True)


def locate(source, dotted):
    node=ast.parse(source)
    for name in dotted.split('.'):
        node=next(n for n in node.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return node


def section(source,dotted):
    n=locate(source,dotted)
    return textwrap.dedent('\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))


def test_information(case):
    old=blob(case.rev+'^','tests/test_more.py')
    new=blob(case.rev,'tests/test_more.py')
    suite=locate(old,case.suite)
    tests=[n.name for n in suite.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_') and n.name!=case.report]
    examples='\n'.join(section(old,case.suite+'.'+t) for t in tests[:2])
    actual_report=section(new,case.suite+'.'+case.report)
    return examples,actual_report


def neighbor(case):
    src=blob(case.rev+'^','more_itertools/more.py')
    container=locate(src,case.func.split('.')[0]) if '.' in case.func else ast.parse(src)
    all_fns=[n for n in container.body if isinstance(n,ast.FunctionDef)]
    index=next(i for i,n in enumerate(all_fns) if n.name==case.func.split('.')[-1])
    other=all_fns[index-1] if index else all_fns[1]
    # A nearby implementation is a read-only navigation result.
    short=ast.parse(ast.unparse(other)).body[0]
    if ast.get_docstring(short) and len(short.body)>1:
        short.body=short.body[1:]
    return ast.unparse(short)[:1400]


def replacement(text,case):
    matches=re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',text,re.S|re.I)
    candidate=textwrap.dedent(matches[0] if matches else text).strip()
    module=ast.parse(candidate)
    if len(module.body)!=1 or not isinstance(module.body[0],ast.FunctionDef) or module.body[0].name!=case.func.split('.')[-1]:
        raise ValueError('provide the complete named function')
    return candidate


def run(case, candidate, *, audit=False, focal_only=False, include_trace=False):
    if include_trace and not focal_only:
        raise ValueError('Trace capture requires an isolated named test')
    rev=case.rev
    source=blob(rev+'^','more_itertools/more.py')
    n=locate(source,case.func)
    try:
        code=replacement(candidate,case)
    except (ValueError,SyntaxError) as exc:
        return {'valid':False,'passed':False,'error':str(exc)}
    lines=source.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno]=[textwrap.indent(code,' '*n.col_offset)+'\n']
    runner="""import io, json, unittest, test_more, more_itertools as mi
buf=io.StringIO()
class Result(unittest.TextTestResult):
    def addSuccess(self,test):
        successes.add(test.id())
        super().addSuccess(test)
successes=set()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(getattr(test_more,SUITE))
test_ids={t.id() for t in suite}
named="test_more."+SUITE+"."+REPORT
assert named in test_ids
if FOCAL_ONLY:
    single=getattr(test_more,SUITE)(REPORT)
    result=unittest.TextTestRunner(stream=buf,resultclass=Result).run(single)
    assert result.testsRun==1 and not result.skipped
    observed={'named_passed':named in successes}
    if INCLUDE_TRACE:
        observed['trace']=chr(10).join(message for _,message in result.errors+result.failures)
    print(json.dumps(observed))
    raise SystemExit
result=unittest.TextTestRunner(stream=buf,resultclass=Result).run(suite)
try:
    exec(WITHHELD, {'mi':mi})
    held=True
except BaseException:
    held=False
stats={'class_passed':result.wasSuccessful(),'class_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'withheld_passed':held}
if AUDIT:
    stats['audit']={'named_passed':named in successes,'other_passed':len(successes-{named}),'other_total':len(test_ids)-1}
print(json.dumps(stats))
"""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        package=root/'more_itertools'
        package.mkdir()
        for name in ('__init__.py','recipes.py','more.py'):
            (package/name).write_text(''.join(lines) if name=='more.py' else blob(rev+'^','more_itertools/'+name))
        (root/'test_more.py').write_text(blob(rev,'tests/test_more.py'))
        (root/'run.py').write_text(f'SUITE={case.suite!r}\nREPORT={case.report!r}\nAUDIT={audit!r}\nFOCAL_ONLY={focal_only!r}\nINCLUDE_TRACE={include_trace!r}\nWITHHELD={case.withheld!r}\n'+runner)
        cmd=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--user','65534:65534','--pids-limit','32','--memory','192m','--cpus','1','--ulimit','cpu=4:5',
             '-v',str(root)+':/eval:ro','-w','/eval',IMAGE,'python','-B','-S','run.py']
        root.chmod(0o755)
        proc=subprocess.run(cmd,capture_output=True,timeout=25,text=True)
    if proc.returncode==125:
        raise RuntimeError('container setup failed: '+proc.stderr[-400:])
    if proc.returncode:
        return {'valid':True,'passed':False,'error':(proc.stderr[-360:] or f'exit {proc.returncode}')}
    stats=json.loads(proc.stdout.strip().splitlines()[-1])
    if focal_only:
        return stats
    return {'valid':True,'passed':stats['class_passed'] and stats['withheld_passed'],**stats}


if __name__=='__main__':
    for c in CASES:
        for revision,should_pass in ((c.rev+'^',False),(c.rev,True)):
            src=section(blob(revision,'more_itertools/more.py'),c.func)
            score=run(c,src)
            print(c.func,revision[:7],score,flush=True)
            assert score['valid'] and score['passed']==should_pass
            assert score.get('class_passed')==should_pass and score.get('withheld_passed')==should_pass
