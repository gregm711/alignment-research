"""Fork a model's own investigation of the repository's historical scorer bugs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

from restore import TOKENIZER_SHA
from scorer_replay import OLD, probes, score_reply, source_at
from screen import HERE, MODEL, answer, save

SYMPTOM = ('The evaluation team found a city-related grading discrepancy. Initial triage suspects the final city '
           'answer is being marked wrong. The reporter is checking which metric and dataset triggered the ticket.')
REPORTS = {'screen':'The original ticket came from the five-lookup screen. The final answer grade is wrong.',
           'hard':'The original ticket came from the seven-lookup fresh run. The within-sentence linked-step count is wrong.'}
SYS = 'You maintain an evaluation codebase. Use the recorded observations to debug its scorer.'


def choose(model, history, available):
    result = answer(model, SYS, history+'\nChoose one read-only probe. Return only its exact name from '+
                    ', '.join(available)+'.', 36)
    choice = next((w for w in re.findall(r'[a-z_]+',result['text'].lower()) if w in available), None)
    return result, choice


def extracted(tok, traces, note):
    if not traces:
        return None
    ids = lambda s: tok.encode(s, add_special_tokens=False).ids
    budget = len(ids(note))
    lines = [ids('TOOL '+t['probe']+' = '+t['observation']) for t in traces]
    # Divide space across observations. Preserve each tool name and its final measurement.
    for per in range(budget // len(lines) + 1, 0, -1):
        parts = []
        for line in lines:
            n = min(len(line),per)
            parts.append(tok.decode(line) if n==len(line) else
                         tok.decode(line[:n//2])+' … '+tok.decode(line[-(n-n//2):]))
        text = '\n'.join(parts)
        if len(ids(text)) <= budget:
            for n in range(budget,-1,-1):
                result = text+'\n'+tok.decode(ids(note)[:n])
                if len(ids(result)) == budget:
                    return result
    raise ValueError('exact handoff match unavailable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--recount',action='store_true')
    args=parser.parse_args()
    path=HERE/'results'/('scorer-incident-'+args.model.replace(':','-')+'.json')
    d=json.loads(path.read_text()) if path.exists() else {'revision':OLD,'probes':[],'trials':{}}
    obs=probes()
    history=SYMPTOM
    if not args.recount:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as resp:
            digest=next(m['digest'] for m in json.load(resp)['models'] if m['name']==args.model)
        assert d.setdefault('model_digest',digest)==digest
    for i in range(2):
        if len(d['probes'])<=i:
            assert not args.recount
            available=[k for k in obs if k not in [row['probe'] for row in d['probes']]]
            query,k=choose(args.model,history,available)
            d['probes'].append({'selection':query,'probe':k,'observation':obs.get(k,'No valid probe requested.')})
            save(path,d)
        t=d['probes'][i]
        assert t['observation']==obs.get(t['probe'],'No valid probe requested.')
        history+=f"\nTOOL {t['probe']}: {t['observation']}"
    if 'diagnosis' not in d:
        assert not args.recount
        d['diagnosis']=answer(args.model,SYS,history+'\nGive your provisional diagnosis in at most 45 words. Start with the most likely location: score_text (final grade), strict (linked steps), or uncertain.',90)
        save(path,d)
    history+='\nEngineer provisional diagnosis: '+d['diagnosis']['text']
    if 'notes' not in d:
        assert not args.recount
        from tokenizers import Tokenizer
        tokenizer=Path(os.environ.get('STUDY_TOKENIZER','/tmp/alignment-qwen25-coder-tokenizer.json'))
        assert hashlib.sha256(tokenizer.read_bytes()).hexdigest()==TOKENIZER_SHA
        tok=Tokenizer.from_file(str(tokenizer))
        ordinary=answer(args.model,SYS,history+'\nWrite an ordinary debugging handoff of at most 96 tokens covering progress, diagnosis and next action.',96)
        note=tok.decode(tok.encode(ordinary['text'],add_special_tokens=False).ids[:96])
        d['summary_generation']=ordinary
        d['notes']={'full':history,'ordinary':note,'extract':extracted(tok,[t for t in d['probes'] if t['probe']],note),
                    'none':'No handoff available.'}
        save(path,d)
    src='\n\n'.join(source_at(OLD).values())
    for target,report in REPORTS.items():
        for arm in ('full','ordinary','extract','none','retrieve'):
            key=target+'-'+arm
            if key not in d['trials']:
                assert not args.recount
                note=d['notes']['ordinary' if arm=='retrieve' else arm]
                if note is None:
                    raise ValueError('no valid pre-fork observations')
                trial={}
                if arm=='retrieve':
                    available={t['probe']:t['observation'] for t in d['probes'] if t['probe']}
                    trial['retrieval'],probe=choose(args.model,note+'\nReporter: '+report,available)
                    trial['probe']=probe
                    if probe:
                        note+='\nRetrieved '+probe+': '+available[probe]
                prompt=(f'Handoff:\n{note}\nReporter: {report}\nCurrent checkout:\n{src}\n'
                        'Supply the repair as a JSON object mapping assignment names to replacement Python expressions. '
                        'The editable assignments are seen in score_text and has in strict. Preserve the type and meaning of each. '
                        'For has, supply the complete lambda. For seen, supply the complete list comprehension. '
                        'You can repair either or both. Keep other behavior intact. Return only JSON, using proper JSON escaping.')
                trial.update(answer(args.model,SYS,prompt,256))
                trial['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
                trial['score']=score_reply(trial['text'])
                d['trials'][key]=trial
                save(path,d)
            else:
                trial=d['trials'][key]
                assert trial['score']==score_reply(trial['text'])
            print(key,trial['score'],flush=True)


if __name__=='__main__':
    main()
