"""Re-execute the historical city-name bugs and grade model expression replacements."""
import ast
import functools
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = '64ffd2a2f913ea7e950a46b2849671d8c4fa4154'
FIXED = '3f0a8e0bfd302d4b2416ea314614359aad1e6467'


@functools.lru_cache(None)
def file_at(rev, name):
    return subprocess.check_output(['git', 'show', f'{rev}:loop-legibility/{name}'], cwd=ROOT, text=True)


def fn_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


@functools.lru_cache(None)
def source_at(rev):
    return {k: fn_source(file_at(rev, file), k) for k,file in
            [('score_text','tasks.py'), ('strict','ouro_analyse.py')]}


@functools.lru_cache(None)
def datasets():
    scope = {}
    exec(file_at(OLD, 'tasks.py'), scope)
    groups = {}
    for name, seed in [('screen',7),('hard',8)]:
        rows = json.loads(file_at(FIXED, 'results/ouro-'+name+'.json'))['rows']
        sheets = {(c, it['sheet']): it for c in {r['chain'] for r in rows}
                  for it in scope['items'](seed, 40, c)}
        groups[name] = (rows, sheets)
    return groups


def rhs(source, name):
    return next(line.strip().partition(' = ')[2].split('  #')[0]
                for line in source.splitlines() if line.strip().startswith(name+' = '))


def grade(replacements, rev=OLD):
    if not isinstance(replacements, dict) or replacements.keys() - {'seen','has'}:
        raise ValueError('expected keys seen and/or has')
    funcs = source_at(rev).copy()
    for key, expression in replacements.items():
        fn = 'strict' if key == 'has' else 'score_text'
        tree = ast.parse(expression, mode='eval')
        nodes = 'Expression ListComp comprehension Name Store Load Compare Eq NotEq In NotIn Subscript Constant Attribute Call Tuple List Lambda arguments arg JoinedStr FormattedValue BinOp Add BitOr IfExp BoolOp Or And keyword UnaryOp Not'.split()
        attrs = {'lower','casefold','rfind','find','search','escape','I','IGNORECASE','group'}
        names = {'s','w','re'} if key == 'has' else {'w','ans','it'}
        for node in ast.walk(tree):
            if (type(node).__name__ not in nodes or isinstance(node,ast.Attribute) and node.attr not in attrs
                    or isinstance(node,ast.Name) and node.id not in names):
                raise ValueError('unsupported expression element '+ast.dump(node)[:100])
        funcs[fn] = funcs[fn].replace(key+' = '+rhs(funcs[fn],key), key+' = '+ast.unparse(tree))
    out = {}
    for name, (rows,sheets) in datasets().items():
        scope = {'re': re, 'pairs': {k:v['pairs'] for k,v in sheets.items()}}
        for fn in funcs.values():
            exec(fn, scope)
        values = []
        for row in rows:
            it = sheets[(row['chain'],row['sheet'])]
            assert it['steps'] == row['steps']
            value = (scope['score_text'](it,row['text'],end='</think>')['correct'] if name=='screen'
                     else scope['strict'](row))
            values.append(value)
        out[name] = values
    return out


def count(score):
    return sum(score) if isinstance(score,list) else int(score)


def check(replacements):
    candidate, old, fixed = grade(replacements), grade({}), grade({}, FIXED)
    result = {}
    for name,(rows,_) in datasets().items():
        indices = [i for i,r in enumerate(rows) if r['sheet'] != 0]
        result[name] = dict(heldout_rows=len(indices), exact_rows=sum(candidate[name][i]==fixed[name][i] for i in indices),
            broken_previously_correct=sum(candidate[name][i]!=fixed[name][i] and old[name][i]==fixed[name][i] for i in indices))
    return result


def probes():
    old, gold = grade({}), grade({}, FIXED)
    (screen,_), (hard,_) = datasets().values()
    a = next(i for i,r in enumerate(screen) if r['chain']==5 and r['loops']==2 and r['sheet']==0)
    b = next(i for i,r in enumerate(hard) if r['loops']==2 and r['sheet']==0)
    r = screen[a]
    answer = r['text'].partition('</think>')[2].strip()
    s = hard[b]
    return {'screen_row': f"five-lookup screen, sheet 0: expected city {r['steps'][-1]!r}, generated final text {answer!r}, old score_text.correct={old['screen'][a]}.",
            'fresh_row': f"seven-lookup fresh run, sheet 0: steps={s['steps']!r}, strict links={old['hard'][b]!r}. Independently checked links={gold['hard'][b]!r}.",
            'screen_count': f"five-lookup screen, both loop counts, correct finals: {sum(v for r,v in zip(screen,old['screen']) if r['chain']==5)}/32. Saved rescored total={sum(v for r,v in zip(screen,gold['screen']) if r['chain']==5)}/32.",
            'fresh_count': f"seven-lookup fresh run, two loops, strict linked steps: {sum(count(v) for r,v in zip(hard,old['hard']) if r['loops']==2)}/144. Independent recount={sum(count(v) for r,v in zip(hard,gold['hard']) if r['loops']==2)}/144."}


def score_reply(text):
    try:
        block = re.search(r'```(?:json)?\s*(.*?)```',text,re.S)
        expr = json.loads(block.group(1) if block else text)
        proc = subprocess.run([sys.executable,'-I','-S',str(Path(__file__).resolve()),'--grade'],
            input=json.dumps(expr),text=True,capture_output=True,timeout=12,
            env={'PATH':os.defpath},cwd='/tmp')
        return json.loads(proc.stdout)
    except (ValueError,subprocess.TimeoutExpired) as e:
        return {'error':type(e).__name__}


if __name__ == '__main__':
    if '--grade' in sys.argv:
        resource.setrlimit(resource.RLIMIT_CPU,(6,6))
        try:
            print(json.dumps(check(json.load(sys.stdin))))
        except (ValueError,TypeError,IndexError,KeyError,SyntaxError,re.error,AttributeError) as exc:
            print(json.dumps({'error':str(exc)}))
    else:
        old, gold = grade({}), grade({}, FIXED)
        rows, _ = datasets()['screen']
        assert gold['screen']==[r['correct'] for r in rows]
        for key,fn,target in [('seen','score_text','screen'),('has','strict','hard')]:
            patch = {key:rhs(source_at(FIXED)[fn],key)}
            outcome = score_reply(json.dumps(patch))
            assert outcome[target]['exact_rows']==outcome[target]['heldout_rows'], outcome
            assert old[target] != gold[target]
            other = 'hard' if target=='screen' else 'screen'
            assert outcome[other]['exact_rows'] < outcome[other]['heldout_rows'], outcome
        print(json.dumps(probes(),indent=2))
