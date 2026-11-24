"""Bounded local Codex search. Every proposal survives, including ties and failures."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from bench import HERE, MAXIMIZE, MINIMIZE, evaluate, image_id

RUNS = HERE/'runs'/'revision-v2'
ISLANDS = ('rule', 'sparse', 'dense', 'active')
DEV_SEED, DEV_N = 2711, 160
LOCKED = ('bench.py','worker.py','baseline.py','program.md','orchestrator.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def dominates(a,b):
    differences = [a[k]-b[k] for k in MAXIMIZE]+[b[k]-a[k] for k in MINIMIZE]
    return min(differences)>=0 and max(differences)>0


def archive(entries):
    done = [r for r in entries if r.get('result')]
    return [r for r in done if not any(dominates(o['result']['metrics'],r['result']['metrics'])
                                       for o in done)]


def ask(prompt, fields):
    schema = dict(type='object', properties={k:{'type':'string'} for k in fields},
                  required=list(fields),additionalProperties=False)
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        schemafile,out = directory/'schema.json',directory/'response.json'
        save(schemafile,schema)
        command = ['codex','exec','--ephemeral','--ignore-user-config','--ignore-rules',
                   '--skip-git-repo-check','--sandbox','read-only','-C',folder,
                   '--output-schema',str(schemafile),'-o',str(out),'-']
        proc = subprocess.run(command,input=prompt,text=True,stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE,cwd=folder,timeout=240)
        if proc.returncode or not out.exists():
            raise RuntimeError('Codex proposal failed: '+proc.stderr[-1200:])
        answer=load(out)
        if any(not isinstance(answer[k],str) or not answer[k].strip() for k in fields):
            raise ValueError('Empty proposal field')
        return answer


def ledger():
    return [load(p) for p in sorted(RUNS.glob('*/record.json'))]


def run_candidate(directory,record,code,image):
    directory.mkdir(exist_ok=False)
    artifact = directory/'candidate.py'
    artifact.write_text(code)
    record.update(id=directory.name,sha256=digest(artifact),source_bytes=artifact.stat().st_size,
                  status='planned')
    save(directory/'record.json',record)  # File the hypothesis before scoring.
    try:
        record.update(result=evaluate(artifact,DEV_SEED,DEV_N,image),status='scored')
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        record.update(status='failed',error=str(exc)[:2000])
    save(directory/'record.json',record)
    print(record['id'],record['status'],record.get('result',{}).get('metrics',record.get('error')),flush=True)
    return record


def initialize(image):
    manifest = {k:digest(HERE/k) for k in LOCKED}
    manifest['runtime_image']=image
    lockfile=RUNS/'manifest.json'
    if lockfile.exists() and load(lockfile)!=manifest:
        raise ValueError('Evaluator or image changed. Start a fresh --runs directory.')
    save(lockfile,manifest)
    base=(HERE/'baseline.py').read_text()
    for mode in ('keep','random','active'):
        folder=RUNS/('base-'+mode)
        if folder.exists():
            continue
        record=dict(island='baseline',hypothesis='Reference policy '+mode,
                    prediction='Informative selection improves balanced accuracy over random checking.',
                    failure='Equal or lower balanced accuracy on the fixed development stream.')
        run_candidate(folder,record,base.replace("MODE = 'active'",f'MODE = {mode!r}'),image)
    if any(r['status']!='scored' for r in ledger() if r['island']=='baseline'):
        raise RuntimeError('A baseline failed. Inspect its record before continuing.')


def results_context(entries):
    return [{'id':r['id'],'metrics':r.get('result',{}).get('metrics'),'idea':r.get('hypothesis'),
             'error':r.get('error')} for r in entries]


def next_round(image):
    entries=ledger()
    attempts=[r for r in entries if r['island']!='baseline']
    index=len(attempts)
    island=ISLANDS[index%len(ISLANDS)]
    lineage=[r for r in attempts if r['island']==island]
    on_front={r['id'] for r in archive(entries)}
    # Two further descendants may follow a dominated design before reselecting.
    parent='base-active'
    if lineage:
        if len(lineage)%3:
            parent=lineage[-1]['id']
        else:
            surviving=[r for r in lineage if r['id'] in on_front]
            if surviving:
                parent=surviving[-1]['id']
    context=(HERE/'program.md').read_text()+'\nRecent results and frontier:\n'+json.dumps(
        results_context(entries[-12:])+results_context(archive(entries)[:8]))
    recent=next((r for r in reversed(lineage) if r.get('result')),None)
    if recent:
        context+='\nObserved failures:\n'+json.dumps(recent['result']['failures'])
    reflections=sorted(RUNS.glob('reflection-*.json'))
    if reflections:
        context+='\nPrior reflection:\n'+json.dumps(load(reflections[-1]))
    needs_reflection=(index>0 and index%5==0) or (len(lineage)>=3 and all(
        r['id'] not in on_front for r in lineage[-3:]))
    if needs_reflection:
        reflection_file=RUNS/f'reflection-{index:04}.json'
        if not reflection_file.exists():
            save(reflection_file,ask(context+'\nReflect only. Identify a failure and propose one falsifiable ablation.',
                                     ('diagnosis','ablation')))
        context+='\nCurrent reflection:\n'+json.dumps(load(reflection_file))
    if index and index%15==0:
        review_file=RUNS/f'review-{index:04}.json'
        if not review_file.exists():
            review_sources='\n'.join((RUNS/r['id']/'candidate.py').read_text() for r in archive(entries)[:4])
            save(review_file,ask(context+review_sources+'\nReview only. Identify shortcuts and one transfer test.',
                                 ('shortcut','transfer_test')))
        context+='\nIndependent review:\n'+json.dumps(load(review_file))
    code=(RUNS/parent/'candidate.py').read_text()
    proposal=ask(context+'\nParent code:\n'+code+f'\nIsland: {island}. Develop one substantive '
                 'mechanistic variation for this family. Use no tools. Return complete Python code as source '
                 'without fences, plus the hypothesis, measurable prediction and failure criterion.',
                 ('hypothesis','prediction','failure','source'))
    source=proposal.pop('source')
    r=run_candidate(RUNS/f'{index+1:04}-{island}',dict(island=island,parent=parent,**proposal),source,image)
    save(RUNS/'frontier.json',[x['id'] for x in archive(ledger())])
    return r


def main():
    global RUNS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=RUNS)
    op=p.add_mutually_exclusive_group(required=True)
    op.add_argument('--init',action='store_true')
    op.add_argument('--rounds',type=int)
    op.add_argument('--confirm',metavar='CANDIDATE_ID')
    args=p.parse_args()
    if args.rounds is not None and not 1<=args.rounds<=100:
        p.error('--rounds must be 1 through 100')
    RUNS=args.runs.resolve()
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        image=image_id()
        initialize(image)
        if args.rounds and (RUNS/'confirmation.json').exists():
            p.error('This campaign has opened confirmation. Start a new campaign for new development.')
        if args.confirm:
            valid={r['id']:r for r in ledger() if r.get('result')}
            if args.confirm not in valid:
                p.error('Unknown or unscored candidate')
            marker=RUNS/'confirmation.json'
            previous=load(marker) if marker.exists() else {}
            if previous and ('baseline' in previous or args.confirm!=previous['candidate']):
                p.error('This campaign already opened confirmation for a frozen candidate')
            selected=RUNS/args.confirm/'candidate.py'
            if digest(selected)!=valid[args.confirm]['sha256']:
                p.error('The candidate changed after development scoring')
            seed=previous['seed'] if previous else int.from_bytes(os.urandom(8),'little')
            record=dict(candidate=args.confirm,seed=seed,image=image,sha256=digest(selected))
            save(marker,record)
            record['result']=evaluate(selected,seed,640,image,examples=False)
            record['baseline']=evaluate(RUNS/'base-active'/'candidate.py',seed,640,image,examples=False)
            save(marker,record)
            print(json.dumps(record,indent=2))
        for _ in range(args.rounds or 0):
            next_round(image)
        save(RUNS/'frontier.json',[r['id'] for r in archive(ledger())])


if __name__=='__main__':
    main()
