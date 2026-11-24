"""Recompute repair counts, cue sensitivity and native input token deltas."""
import argparse
import json
from collections import Counter
from math import comb

from cases import CASES
from restore import WRITER_PATH
from screen import HERE, MODEL, evaluate, function, materials

ARMS = ('full','ordinary','evidence','none','observed','swapped','common')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    args=parser.parse_args()
    folder=HERE/'results'/('restore-'+args.model.replace(':','-'))
    buckets={arm:Counter() for arm in ARMS}
    outcome={}
    coded=lambda v: json.dumps(v,sort_keys=True,separators=(',',':'))
    for i,c in enumerate(CASES):
        _,expected=materials(c,i)
        buggy=evaluate(function(c.source),c.common+c.hidden)
        data=json.loads((folder/(c.name+'.json')).read_text())['trials'].copy()
        if args.model==MODEL:
            data.update(json.loads((WRITER_PATH/(c.name+'.json')).read_text())['trials'])
        for arm in ARMS:
            cats=[]
            for j,name in enumerate(('alder','birch')):
                trial=data[f'{name}-{arm}']
                score=trial['score']
                outputs=coded(score.get('outputs'))
                correct=outputs==coded(expected[j])
                assert correct==score['passed'], 'type-sensitive recount differs from scorer'
                category=('target' if correct else 'opposite' if outputs==coded(expected[1-j]) else
                          'unchanged' if outputs==coded(buggy) else 'error' if 'error' in score else 'other')
                cats.append(category)
                outcome[(c.name,name,arm)]=category
                buckets[arm][(name,category)]+=1
                delta=trial['input_tokens']-data[f'{name}-ordinary']['input_tokens']
                if arm in ('common','observed','swapped'):
                    buckets[arm][('token delta',delta)]+=1
            buckets[arm]['both routes']+=cats==['target','target']
        for name in ('alder','birch'):
            # Correctly follow each note, including the deliberately misattributed one.
            buckets['observed'][('switch',name)]+=(
                outcome[c.name,name,'observed']=='target' and outcome[c.name,name,'swapped']=='opposite')
    print('arm        confirm revise both   reversal outcomes: opposite unchanged other error')
    for arm in ARMS:
        b=buckets[arm]
        print(f"{arm:<10}",b['alder','target'],b['birch','target'],b['both routes'],'   ',
              *[b['birch',k] for k in ('opposite','unchanged','other','error')])
    print('Reversal differences, observed vs ordinary (case-paired):')
    gains=[c.name for c in CASES if outcome[c.name,'birch','observed']=='target' and outcome[c.name,'birch','ordinary']!='target']
    losses=[c.name for c in CASES if outcome[c.name,'birch','ordinary']=='target' and outcome[c.name,'birch','observed']!='target']
    n=len(gains)+len(losses)
    p=min(1,2*sum(comb(n,j) for j in range(min(len(gains),len(losses))+1))/2**n)
    print('gain',gains,'loss',losses,'exact two-sided sign test',round(p,4))
    print('Correct to opposite on label swap, confirm and revise:',
          buckets['observed']['switch','alder'],buckets['observed']['switch','birch'])
    for arm in ('observed','swapped','common'):
        dt={key[1]:v for key,v in buckets[arm].items() if isinstance(key,tuple) and key[0]=='token delta'}
        print(arm,'native prompt token difference vs ordinary (difference: frequency)',dt)


if __name__=='__main__':
    main()
