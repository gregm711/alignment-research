"""Rerun the local architecture pilots concurrently in a new directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
# Keep the sequence within each research line while running independent lines together.
LINES = {
    'teaching-cost': ['teaching-cost/hybrid/compare.py'],
    'test-selection': ['remembering-tests/replay.py','remembering-tests/external/run.py'],
    'programs': ['moonshots/program-memory/study.py','moonshots/program-memory/external/study.py'],
    'hypotheses': ['moonshots/multiworld/study.py','moonshots/multiworld/learned.py',
                   'moonshots/multiworld/bottleneck.py'],
    'equilibrium': ['moonshots/equilibrium/run.py'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='new directory for code, logs and recomputed outputs')
    parser.add_argument('--torch-python',default=sys.executable,
                        help='Python with a compatible PyTorch and NumPy for the learned-state pilots')
    args=parser.parse_args()
    dest=args.directory.resolve()
    if dest==ROOT or ROOT in dest.parents:
        parser.error('use an output directory outside the research tree')
    dest.mkdir(parents=True,exist_ok=False)
    for directory in ('teaching-cost','autoresearch','remembering-tests','moonshots'):
        for src in (ROOT/directory).rglob('*.py'):
            target=dest/src.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,target)
    result=dest/'teaching-cost'/'results'
    result.mkdir()
    for name in ('encoder.npz','frozen.json'):
        shutil.copyfile(ROOT/'teaching-cost'/'results'/name,result/name)
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

    def run(line):
        name, scripts=line
        log=dest/(name+'.log')
        with log.open('w') as stream:
            for path in scripts:
                script=dest/path
                python=args.torch_python if script.name in ('learned.py','bottleneck.py') else sys.executable
                subprocess.run([python,script.name],cwd=script.parent,env=environment,
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'{name} finished: {log}',flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,LINES.items()))


if __name__=='__main__':
    main()
