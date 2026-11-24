"""Frozen held-out source changes and executable checks for the patch audit."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from panel import Case


CASES = [
    Case('fbd55137fe47b44f55190c3714c75f1b21bf834d','chunked_even','ChunkedEvenTests','test_zero_chunk_size',
         'Even chunking behaves incorrectly for nonpositive chunk sizes and can touch the source unexpectedly.',
         """source=iter([7,8,9])
assert list(mi.chunked_even(source,0))==[] and list(source)==[7,8,9]
source=iter([7,8])
try: list(mi.chunked_even(source,-3))
except ValueError as error: assert 'n must be at least 0' in str(error)
else: raise AssertionError('negative length accepted')
assert list(source)==[7,8]
assert list(mi.chunked_even(range(5),3))==[[0,1,2],[3,4]]"""),
    Case('67fcf640477c78605580f5ee703a54c73c1e7a8a','bucket.__init__','BucketTests','test_falsey_validator_is_honoured',
         'A bucket with a user-supplied callable validator can admit keys that the validator rejects.',
         """class Validator:
    def __len__(self): return 0
    def __call__(self,k): return k%2==0
b=mi.bucket([3,4,6,7],key=lambda x:x,validator=Validator())
assert list(b[3])==[] and list(b[4])==[4] and set(b)=={4,6}
assert list(mi.bucket(['a'],key=str)[ 'a' ])==['a']"""),
    Case('958990e22c4ab6daf434d89cf2b86d7a4a7a9e3c','sliced','SlicedTests','test_negative',
         'Invalid slice lengths can silently produce misleading chunks instead of rejecting the request.',
         """for strict in (False,True):
    try: list(mi.sliced((1,2,3,4),-3,strict=strict))
    except ValueError: pass
    else: raise AssertionError('negative length accepted')
assert list(mi.sliced('QWERT',2))==['QW','ER','T']"""),
    Case('cd0a3a87d231033db06a3c222fe893025e398a86','iterate','IterateTests','test_func_controls_iteration_stop',
         'A stepping callback uses StopIteration to signal that its sequence is finished, but the caller gets an error.',
         """def step(n):
    if n < 2: raise StopIteration
    return n-2
assert list(mi.iterate(step,5))==[5,3,1]
def unrelated(n): raise LookupError('signal')
g=mi.iterate(unrelated,1)
assert next(g)==1
try: next(g)
except LookupError as error: assert str(error)=='signal'
else: raise AssertionError('wrongly swallowed callback error')""")
]
