"""Request two parameter hypotheses from the existing Codex harness, using local Ollama."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'autoresearch'))
from orchestrator import ask, save

DEFAULT = dict(rate=0.3, temperature=0.09, mix=0.75, expand=0.18, stop=0.95)
BOUNDS = dict(rate=(0.02, 1.5), temperature=(0.01, 0.5), mix=(0.2, 1),
              expand=(0.02, 0.45), stop=(0.75, 0.999))


def proposals():
    previous = []
    for n in (1, 2):
        path = HERE / 'results' / f'proposal-{n}.json'
        if path.exists():
            previous.append(json.loads(path.read_text()))
            continue
        prompt = ('You are designing a small experiment. No tools. Give a single numeric parameter '
                  'set and a falsifiable hypothesis. A small frozen masked LM assigns names to 4 '
                  'unknown fields; cosine retrieval or SGD on four decoder rows learns from prior '
                  'verified examples. rate scales a unit-normalized decoder SGD step; temperature '
                  'is cosine kernel softmax temperature; mix interpolates retrieval and base field '
                  'probabilities; expand is tolerated probability mass outside the top two fields '
                  'before paying one reasoning action; stop is the posterior probability of the '
                  'best candidate program sufficient to stop probing. Default '+str(DEFAULT)+
                  '. Ranges '+str(BOUNDS)+'. Return decimal strings for all parameters. Prior '
                  'proposals: '+str(previous))
        answer = ask(prompt, ('hypothesis','prediction', *DEFAULT), local_model='qwen2.5-coder:7b')
        try:
            settings = {k:float(answer[k]) for k in DEFAULT}
            if not all(low <= settings[k] <= high for k,(low,high) in BOUNDS.items()):
                raise ValueError('settings outside registered bounds')
            result = dict(hypothesis=answer['hypothesis'], prediction=answer['prediction'], settings=settings)
        except ValueError as error:
            result = dict(status='rejected', response=answer, reason=str(error))
        save(path, result)
        previous.append(result)


if __name__ == '__main__':
    proposals()
