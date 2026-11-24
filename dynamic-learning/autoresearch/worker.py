"""Container entry point. Only the candidate, this entry point and current observations are mounted."""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location('candidate','/task/candidate.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
stage, items = json.load(sys.stdin)
if stage == 'step':
    out = [candidate.step(obs) for obs in items]
elif stage == 'forecast':
    out = [candidate.forecast(obs,row) for obs,row in items]
else:
    out = [[candidate.answer(memory, row, old) for row,old in queries]
           for memory,queries in items]
json.dump(out, sys.stdout,allow_nan=False)
