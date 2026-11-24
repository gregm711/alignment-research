"""Replay a public-test gate while keeping withheld outcomes outside the decision."""
import argparse
import json
from pathlib import Path

from panel import CASES

ROOT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='qwen2.5-coder:7b')
    model=parser.parse_args().model.replace(':','-')
    audit=json.loads((ROOT/(model+'-audit.json')).read_text())
    outcomes={}
    for c in CASES:
        record=json.loads((ROOT/model/(c.func+'.json')).read_text())
        e=audit[c.func]
        for current,continued in [('first',record['trials']['none']),('human-fix',record['draft_controls']['known-good'])]:
            incumbent=e['scores'][e['arms'][current]]
            visible_pass=incumbent['audit']['named_passed']
            fresh=record['draft_controls']['fresh']
            for gated in (False,True):
                for start,revision in [('revise',continued),('fresh',fresh)]:
                    kept=gated and visible_pass
                    final=incumbent if kept else revision['score']
                    name='/'.join([current,'gate' if gated else 'always',start])
                    outcomes.setdefault(name,{})[c.func]=dict(public_pass=visible_pass,retained=kept,
                        generated=not kept,passed=final['passed'],false_accept=bool(kept and not final['passed']))
    for name,rows in outcomes.items():
        print(name,*(f'{field}={sum(row[field] for row in rows.values())}/{len(rows)}' for field in ('passed','generated','false_accept')))
    (ROOT/(model+'-gate.json')).write_text(json.dumps(outcomes,indent=2)+'\n')


if __name__=='__main__':
    main()
