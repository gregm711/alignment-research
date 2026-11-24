"""Independent checks on archived calendars, selected probes, learned states and ledger deltas."""
from datetime import datetime, timedelta
import struct

import numpy as np

from analyse import decode
from ledger import PATHS, QUERIES
from study import OUT, POLICIES, WRITERS


def main():
    with np.load(OUT/'trials.npz') as arc:
        d = {k:arc[k] for k in arc.files}
    rows = d['rows']
    assert rows.shape == (512,9,8)
    # Each ledger is one shared pre-disclosure object for its four forks.
    for ledger, calendar in zip(rows.tolist(), d['calendar']):
        for row, dates in zip(ledger,calendar):
            _,_,_,a,b,term,year,installed = row
            for reversed_date, pair in enumerate(dates):
                fmt = '%m/%d/%Y' if installed == reversed_date else '%d/%m/%Y'
                issued = datetime.strptime(f'{a}/{b}/{year}',fmt).date()
                assert pair.tolist() == [issued.toordinal(), (issued+timedelta(days=term)).toordinal()]
    # Identical visible prefixes yield identical next decisions across worlds.
    for f in range(8):
        for pi in range(len(POLICIES)):
            for j in range(512):
                paths = [PATHS[i] for i in d['trace'][f,pi,2,j]]
                assert len({p[0][0] for p in paths}) == 1
                response = {}
                for w,p in enumerate(paths):
                    assert p[0][0] != p[1][0]
                    assert response.setdefault(p[0],p[1][0]) == p[1][0]
                    assert all(bit == (w > q) for q,bit in p)
                    for b in (0,1):
                        assert PATHS[d['trace'][f,pi,b,j,w]] == p[:b]
    # Execute both maintenance policies and deserialize every reachable evidence
    # path per fit/writer, checking dates by ID against independent world truth.
    tested = 0
    for f in range(8):
        unique = {}
        for (pi,b,j,w), t in np.ndenumerate(d['trace'][f]):
            unique.setdefault((int(t),w),(pi,b,j))
        for (_,w),(pi,b,j) in unique.items():
            role = d['roles'][j]
            ledger = rows[j]
            dates = d['calendar'][j]
            base = {int(ledger[r,0]):tuple(dates[r,0]) for r in role[:6]}
            for wi in range(len(WRITERS)):
                for eager in (False,True):
                    item = decode(d,f,pi,b,j,w,wi,eager)
                    direct = base.copy()
                    for key,issued,due in struct.iter_unpack('<Hii',bytes.fromhex(item['ledger_delta_hex'])):
                        assert key in direct
                        direct[key] = issued,due
                    actual = {key:tuple(datetime.fromisoformat(x).toordinal() for x in pair)
                              for key,pair in item['direct_ledger'].items()}
                    assert direct == actual
                    for r,answer in zip(role[QUERIES],item['queries']):
                        marked = d['truth'][j,w,r]
                        expected = [datetime.fromisoformat(x).toordinal() for x in answer['expected']]
                        assert expected == dates[r,int(marked)].tolist()
                        if pi == 3 and b == 2:
                            assert answer['answer'] == answer['expected']
                        if eager and r in role[:6]:
                            assert direct[answer['invoice_id']] == tuple(datetime.fromisoformat(x).toordinal()
                                                                         for x in answer['answer'])
                    tested += 1
    print(f'Checked 512 calendars, all 16,384 four-world policy decisions, and {tested} serialized replays.')


if __name__ == '__main__':
    main()
