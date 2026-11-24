"""Fast, offline regression checks for search scheduling, audit records and Pareto direction."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import orchestrator as loop
from bench import MAXIMIZE, MINIMIZE, episodes

base = {k:.5 for k in MAXIMIZE+MINIMIZE}
assert loop.dominates(dict(base,bytes=0),base)
assert not loop.dominates(base,base)
assert not loop.dominates(dict(base,bytes=0,affected=.1),base)
assert episodes(23,20)==episodes(23,20) and episodes(24,20)!=episodes(23,20)


def score(artifact,*args):
    pre=loop.load(artifact.parent/'record.json')
    assert pre['status']=='planned' and pre['sha256']==loop.digest(artifact)
    # Every proposal trails every baseline, forcing the stepping-stone path.
    value=.5 if pre['island']=='baseline' else .25
    return {'metrics':dict(base,affected=value),'failures':[]}


def proposal(prompt,fields):
    out={k:'test' for k in fields}
    if 'source' in fields:
        out['source']='def step(obs): return ""\n'
    return out


with tempfile.TemporaryDirectory() as directory, patch.object(loop,'RUNS',Path(directory)), \
        patch.object(loop,'evaluate',score), patch.object(loop,'ask',proposal):
    loop.initialize('test-image')
    for _ in range(16):
        loop.next_round('test-image')
    assert len(loop.ledger())==19
    assert loop.load(loop.RUNS/'0005-rule'/'record.json')['parent']=='0001-rule'
    assert loop.load(loop.RUNS/'0009-rule'/'record.json')['parent']=='0005-rule'
    assert loop.load(loop.RUNS/'0013-rule'/'record.json')['parent']=='base-active'
    assert (loop.RUNS/'reflection-0005.json').exists()
    assert (loop.RUNS/'review-0015.json').exists()
    assert len(loop.archive(loop.ledger()))==3
    try:
        loop.initialize('changed-image')
    except ValueError:
        pass
    else:
        raise AssertionError('Evaluator lock accepted runtime drift')
print('Verified archive, lineage, review, reflection, registration and runtime lock.')
