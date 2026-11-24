"""Execute a public regression at each commit in eight short, linear histories."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'upstream'))
from panel import CASES, IMAGE, REPO, blob

HEAD='aa9977698a5e6ae15d3fdfac08a5fc5621c4e179'
POSITIONS=(7,4,9,6,8,3,5,10)


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True)


def history(fix,position):
    before=git('rev-list','--first-parent','--max-count='+str(position),fix+'^').splitlines()[::-1]
    rows=[r.split() for r in git('rev-list','--parents','--topo-order','--ancestry-path',fix+'..'+HEAD).splitlines()]
    children=defaultdict(list)
    depth={HEAD:0}
    for commit,*parents in rows:
        for parent in parents:
            children[parent].append(commit)
            depth[parent]=max(depth.get(parent,0),1+depth[commit])
    path=[*before,fix]
    while len(path)<13 and children[path[-1]]:
        path.append(max(children[path[-1]],key=lambda c:(depth[c],c)))
    assert len(before)==position
    return path


RUN_TEST="""import json, sys, unittest
sys.path.insert(0,'.')
import test_more
suite=unittest.TestSuite([getattr(test_more,sys.argv[1])(sys.argv[2])])
result=unittest.TestResult()
suite.run(result)
failures=result.errors+result.failures
print(json.dumps(dict(passed=result.wasSuccessful(),run=result.testsRun,skipped=len(result.skipped),
                      issue=failures[0][1].strip().splitlines()[-1] if failures else None)))
"""


def measure(case,position):
    path=history(case.rev,position)
    row=dict(fix=case.rev,commits=path,position=position)
    if len(path)!=13:
        return dict(row,eligible=False,reason='short path')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        root.chmod(0o755)
        (root/'test_more.py').write_text(blob(case.rev,'tests/test_more.py'))
        (root/'run.py').write_text(RUN_TEST)
        for i,rev in enumerate(path):
            package=root/str(i)/'more_itertools'
            package.mkdir(parents=True)
            for name in ('__init__.py','recipes.py','more.py'):
                (package/name).write_text(blob(rev,'more_itertools/'+name))
        # The image sees only these public snapshots and the registered test.
        code="""import json,os,subprocess,sys
for i in range(13):
    r=subprocess.run([sys.executable,'-B','-S','/snap/run.py',*sys.argv[1:]],cwd='/snap/'+str(i),
                     capture_output=True,text=True,timeout=6)
    print(r.stdout.strip() if r.returncode==0 else json.dumps({'error':r.stderr[-400:]}),flush=True)
"""
        cmd=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
             '--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','32','--memory','192m',
             '--cpus','1','--ulimit','cpu=6:8','-v',str(root)+':/snap:ro','-w','/snap',IMAGE,'python','-B','-S',
             '-c',code,case.suite,case.report]
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=100,check=True)
    values=[json.loads(s) for s in proc.stdout.splitlines()]
    assert len(values)==13
    row['measurements']=values
    outcomes=[v.get('passed') for v in values]
    good=(outcomes==[False]*position+[True]*(13-position)
          and all(v.get('run')==1 and v.get('skipped')==0 for v in values)
          and all(v['issue']==values[position-1]['issue'] for v in values[:position]))
    return dict(row,eligible=good,reason=None if good else 'transition or failure signature differs from parent/fix')


if __name__=='__main__':
    out={}
    for case,position in zip(CASES,POSITIONS):
        out[case.func]=measure(case,position)
        print(case.func,out[case.func]['eligible'],out[case.func]['reason'],flush=True)
    (HERE/'measurements.json').write_text(json.dumps(out,indent=2)+'\n')
