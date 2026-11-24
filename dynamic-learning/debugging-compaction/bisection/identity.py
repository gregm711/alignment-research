"""Enumerate the two-probe decision trees on the anonymous release histories."""
from collections import Counter
from functools import lru_cache
import json

from experiment import HERE,search


def main():
    measured=json.loads((HERE/'measurements.json').read_text())
    worlds={case['position']:case for case in measured.values() if case['eligible'] and case['position'] in (3,4,5,7,8,9)}
    assert sorted(worlds)==[3,4,5,7,8,9]
    anonymous=set()
    proof={}
    for t,case in worlds.items():
        blue=(2,6) if t<6 else (5,9)
        rows=[]
        witnesses=[]
        for revision in (2,5,6,9):
            worker='blue' if revision in blue else 'red'
            served=revision if worker=='blue' else revision+t-6
            assert 0<=served<=12
            passed=case['measurements'][served]['passed']
            rows.append((worker,revision,passed))
            witnesses.append(dict(worker=worker,requested=revision,served=served,passed=passed))
        stripped=tuple((revision,passed) for _,revision,passed in rows)
        assert stripped==((2,False),(5,False),(6,True),(9,True))
        anonymous.add(stripped)
        result=search(case,rows,'blue',blue)
        assert result['correct'] and result['certified']
        proof[t]=dict(fix=case['fix'],witnesses=witnesses,decision=result)
    assert len(anonymous)==1

    @lru_cache(None)
    def optimal(targets,remaining):
        # At a leaf one nomination can cover at most one distinct first-pass position.
        nomination,count=Counter(targets).most_common(1)[0]
        best=(count,{'nominate':nomination})
        if remaining:
            for rev in range(13):
                groups=[tuple(t for t in targets if worlds[t]['measurements'][rev]['passed']==value) for value in (False,True)]
                if not all(groups):continue
                fail,passed=[optimal(g,remaining-1) for g in groups]
                score=fail[0]+passed[0]
                if score>best[0]:
                    best=(score,{'test':rev,'fail':fail[1],'pass':passed[1]})
        return best

    targets=tuple(sorted(worlds))
    pooled,plan=optimal(targets,2)
    identified=sum(optimal(g,2)[0] for g in (targets[:3],targets[3:]))
    assert (pooled,identified)==(4,6)
    # Independently replay the optimizer's chosen tree against each measured history.
    predictions={}
    for t in targets:
        node=plan
        while 'test' in node:
            node=node['pass' if worlds[t]['measurements'][node['test']]['passed'] else 'fail']
        predictions[t]=node['nominate']
    assert sum(t==g for t,g in predictions.items())==pooled
    data=dict(worlds=proof,anonymous_observations=next(iter(anonymous)),source_blind_optimum=pooled,
              optimal_tree=plan,source_blind_nominations=predictions,attributed_certificates=identified)
    (HERE/'identity-results.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Same anonymous observations in six worlds. Optimal source-blind search: 4/6. One attribution bit: 6/6 certificates.')


if __name__=='__main__':
    main()
