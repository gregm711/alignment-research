"""Replay only observed actions before a genuine contradiction in an external session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from restore import native_tokenizer
from screen import MODEL, answer, save

CONTEXT=24576
CHOICES={
    'sympy__sympy-12419':('Inspect how the outer summation evaluates a bounded Piecewise term.',
                          'Inspect how individual symbolic identity-matrix entries distinguish equal indices.'),
    'sympy__sympy-13615':('Remove only members known to be in the interval and preserve the exclusion on remaining symbolic members.',
                          'Filter out known interval members and return the remaining finite set as the complete difference.'),
    'sympy__sympy-13757':('Check whether the left expression dispatches multiplication to the polynomial at all.',
                          'Check how the polynomial converts the left expression once its reverse multiplication method is called.'),
}


def content(msg):
    value=msg['content']
    return value if isinstance(value,str) else '\n'.join(part['text'] for part in value if part.get('type')=='text')


def load_case(row):
    folder=Path(os.environ.get('STUDY_TRACES','/tmp/alignment-sympy-traces'))
    folder.mkdir(exist_ok=True,parents=True)
    filename=folder/(row['instance']+'.json')
    raw=filename.read_bytes() if filename.exists() else urllib.request.urlopen(row['url'],timeout=30).read(2_000_000)
    assert hashlib.sha256(raw).hexdigest()==row['sha256']
    if not filename.exists():filename.write_bytes(raw)
    all_msgs=json.loads(raw)['messages']
    index=row['fresh_observation']
    msgs=all_msgs[:index]
    issue=content(msgs[1]).split('<pr_description>',1)[1].split('</pr_description>',1)[0].strip()
    lines=[]
    for m in msgs[2:]:
        if m['role']=='assistant':
            blocks=re.findall(r'```(?:bash|sh)\n(.*?)```',content(m),re.S)
            assert len(blocks)==1
            lines.extend(['ACTION',blocks[0]])
        else:
            assert m['role']=='user'
            lines.extend(['OBSERVATION',content(m)])
    assert msgs[-1]['role']=='assistant' and all_msgs[index]['role']=='user'
    return issue,'\n'.join(lines),content(all_msgs[index])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--recount',action='store_true')
    parser.add_argument('--mask-evidence',action='store_true',help='Replay with only the issue or only the answer choices')
    args=parser.parse_args()
    tok=native_tokenizer(MODEL)
    reader_tok=native_tokenizer(args.model) if args.recount and args.model!=MODEL else tok
    rows=[r for r in json.loads((HERE/'screen.json').read_text()) if r['eligible']]
    assert set(CHOICES)=={r['instance'] for r in rows}
    root=HERE/'results'
    path=root/(('masked-' if args.mask_evidence else '')+args.model.replace(':','-')+'.json')
    out=json.loads(path.read_text()) if path.exists() else {'trials':{}}
    memory_path=root/'handoffs.json'
    notes=json.loads(memory_path.read_text()) if memory_path.exists() else {}
    if not (args.check or args.recount):
        root.mkdir(exist_ok=True)
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as stream:
            tags={m['name']:m['digest'] for m in json.load(stream)['models']}
        assert out.setdefault('model_digest',tags[args.model])==tags[args.model]
    for j,row in enumerate(rows):
        case=row['instance']
        issue,history,fresh=load_case(row)
        prefix=issue+'\n\n'+history
        ids=lambda text:tok.encode(text,add_special_tokens=False).ids
        assert len(ids(prefix))+len(ids(fresh))+600<CONTEXT
        if args.check:
            print(case,len(ids(prefix)),len(ids(fresh)));continue
        if case not in notes:
            assert not args.recount and args.model==MODEL
            entry={'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'writer_digest':tags[MODEL]}
            for name,instruction in [('ordinary','Summarize progress, current diagnosis and next action.'),
                    ('evidence','Preserve concrete observed results, counterevidence and what remains unresolved.')]:
                summary=answer(MODEL,'Write short handoffs for a debugging session.',
                               prefix+'\n\n'+instruction+' Use at most 96 tokens.',96,context=CONTEXT)
                assert len(ids(prefix))<=summary['input_tokens']<CONTEXT
                entry[name]=summary
            budget=min(96,*[len(ids(entry[key]['text'])) for key in ('ordinary','evidence')])
            for key in ('ordinary','evidence'):
                entry[key]['note']=tok.decode(ids(entry[key]['text'])[:budget],skip_special_tokens=False)
            assert len(ids(entry['ordinary']['note']))==len(ids(entry['evidence']['note']))
            notes[case]=entry
            save(memory_path,notes)
        entry=notes[case]
        assert hashlib.sha256(prefix.encode()).hexdigest()==entry['prefix_sha256']
        memories={'full':history,'ordinary':entry['ordinary']['note'],'evidence':entry['evidence']['note'],'none':'No previous investigation.'}
        if args.mask_evidence:
            memories=dict.fromkeys(('issue','choices'),memories['none'])
            fresh='No observation supplied.'
        for arm,note in memories.items():
            shown_issue='No issue supplied.' if arm=='choices' else issue
            for flip in (0,1):
                first=(j+flip)%2
                prompt=(f'Original issue:\n{shown_issue}\nInvestigation before the latest result:\n{note}\n'
                        f'New tool observation:\n{fresh}\nWhich next intervention better addresses the remaining discrepancy?\n'
                        f'A: {CHOICES[case][first]}\nB: {CHOICES[case][1-first]}')
                fingerprint=hashlib.sha256(prompt.encode()).hexdigest()
                key=f'{case}/{arm}/{flip}'
                if key not in out['trials']:
                    assert not args.recount
                    t=answer(args.model,'You review an archived mathematical software debugging session. Select A or B.',
                             prompt,18,reply_format={'type':'object','properties':{'choice':{'enum':['A','B']}},
                                                   'required':['choice'],'additionalProperties':False},context=CONTEXT)
                    assert t['input_tokens']+256<CONTEXT
                    if arm=='full' and args.model==MODEL:
                        assert t['input_tokens']>=len(ids(history))
                    t.update(expected=('A' if first==0 else 'B'),prompt_sha256=fingerprint)
                    out['trials'][key]=t
                    save(path,out)
                t=out['trials'][key]
                assert t['prompt_sha256']==fingerprint
                assert t['expected']==('A' if first==0 else 'B')
                if args.recount:
                    raw_tokens=len(reader_tok.encode(prompt,add_special_tokens=False).ids)
                    assert raw_tokens<=t['input_tokens']<CONTEXT,(key,raw_tokens,t['input_tokens'])
        print(case,flush=True)
    if not args.check:
        for arm in memories:
            solved=[]
            for row in rows:
                case=row['instance']
                trials=[out['trials'][f'{case}/{arm}/{f}'] for f in (0,1)]
                if all(json.loads(t['text'])['choice']==t['expected'] for t in trials):
                    solved.append(case)
            print(arm,solved)
        if args.mask_evidence:return
        deltas=[out['trials'][f"{r['instance']}/evidence/{f}"]['input_tokens']
                -out['trials'][f"{r['instance']}/ordinary/{f}"]['input_tokens'] for r in rows for f in (0,1)]
        print('Evidence minus ordinary native prompt tokens:',min(deltas),'to',max(deltas))


if __name__=='__main__':
    main()
