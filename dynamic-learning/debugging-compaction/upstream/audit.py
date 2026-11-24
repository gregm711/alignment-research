"""Rescore the public regression separately from all remaining release tests."""
import argparse
import hashlib
import json
from pathlib import Path

from panel import CASES, blob, replacement, run, section

HERE=Path(__file__).resolve().parent


def audit(model):
    root=HERE/'results'/model.replace(':','-')
    output=root.parent/(root.name+'-audit.json')
    data=json.loads(output.read_text()) if output.exists() else {}
    for case in CASES:
        record=json.loads((root/(case.func+'.json')).read_text())
        assert record['revision']==case.rev and len(record['trials'])==4
        entry=data.setdefault(case.func,{'scores':{},'arms':{}})
        attempts={'first': record['first'],**record['trials'],**record.get('draft_controls',{})}
        for name,rev in [('human-parent',case.rev+'^'),('human-fix',case.rev)]:
            attempts[name]={'text':section(blob(rev,'more_itertools/more.py'),case.func)}
        for arm,trial in attempts.items():
            try:
                candidate=replacement(trial['text'],case)
            except (ValueError,SyntaxError):
                candidate=trial['text']
            digest=hashlib.sha256(candidate.encode()).hexdigest()
            if digest not in entry['scores']:
                entry['scores'][digest]=run(case,candidate,audit=True)
            score=entry['scores'][digest]
            if arm in ('human-parent','human-fix'):
                assert score['audit']['named_passed']==(arm=='human-fix')
                assert score['passed']==(arm=='human-fix')
            else:
                assert {k:v for k,v in score.items() if k!='audit'}==trial['score'],(case.func,arm)
            entry['arms'][arm]=digest
        tmp=output.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2)+'\n')
        tmp.replace(output)
    for arm in data[CASES[0].func]['arms']:
        named,other,held,whole,absent=[],[],[],[],[]
        for case in CASES:
            entry=data[case.func]
            if arm not in entry['arms']:continue
            score=entry['scores'][entry['arms'][arm]]
            if 'audit' not in score:
                absent.append(case.func)
                continue
            detail=score['audit']
            if detail['named_passed']:named.append(case.func)
            if detail['other_passed']==detail['other_total']:other.append(case.func)
            if score['withheld_passed']:held.append(case.func)
            if score['passed']:whole.append(case.func)
        print(arm,'named',named,'all other',len(other),'withheld',held,'complete',whole,'unscored',absent,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='qwen2.5-coder:7b')
    audit(parser.parse_args().model)
