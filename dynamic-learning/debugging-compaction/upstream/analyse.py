"""Compare revisions to each frozen first patch. Counts use incidents as the unit."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from panel import CASES,replacement


def syntax(case,text):
    try:
        module=ast.parse(replacement(text,case))
    except (ValueError,SyntaxError):
        return None
    fn=module.body[0]
    if ast.get_docstring(fn):
        fn.body=fn.body[1:]
    return ast.dump(module,include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='qwen2.5-coder:7b')
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent/'results'/args.model.replace(':','-')
    counts={}
    initial=Counter()
    for case in CASES:
        data=json.loads((folder/(case.func+'.json')).read_text())
        first=data['first']; score=first['score']
        initial['passes']+=score['passed']
        initial['valid failures']+=score['valid'] and not score['passed']
        initial['invalid']+=not score['valid']
        baseline=syntax(case,first['text'])
        assert set(data['trials'])=={'full','ordinary','extract','none'}
        for arm,trial in {**data['trials'],**data.get('draft_controls',{})}.items():
            stats=counts.setdefault(arm,Counter())
            result=trial['score']
            stats['solved']+=result['passed']
            stats['recovered valid']+=result['passed'] and score['valid'] and not score['passed']
            stats['lost']+=score['passed'] and not result['passed']
            stats['syntax copied']+=baseline is not None and syntax(case,trial['text'])==baseline
            stats['invalid']+=not result['valid']
            stats['truncated']+=trial['finish']=='length'
            stats['n']+=1
        print(case.func,score['passed'],[k for k,v in data['trials'].items() if v['score']['passed']])
    print('Initial patches',dict(initial))
    for arm,stats in counts.items():
        print(arm,dict(stats))


if __name__=='__main__':
    main()
