"""Report wrong provisional diagnoses, recoveries, and paired handoff effects."""
import argparse
from math import comb
import json

from experiment import HERE,MODEL,bounds,object_answer,scenario


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    args=parser.parse_args()
    data=json.loads((HERE/(args.model.replace(':','-')+'.json')).read_text())['episodes']
    measured=json.loads((HERE/'measurements.json').read_text())
    for correction in (False,True):
        branch='correct' if correction else 'confirm'
        rows=[]
        trusted_intervals={}
        for i,(name,case) in enumerate(measured.items()):
            if not case['eligible']:continue
            e=data[name+'/'+branch]
            events,trusted=scenario(case,i,correction)
            guess=object_answer(e['first']['text']).get('first_pass')
            wrong=type(guess) is int and 0<=guess<=12 and guess!=case['position']
            trusted_intervals[name]=bounds([event for event in events if event[0]==trusted])
            rows.append((name,wrong,guess==bounds(events)[1],e['trials']))
        print(branch,'N',len(rows),'initially wrong',sum(r[1] for r in rows),'initial diagnosis matches pooled measurements',sum(r[2] for r in rows))
        for arm in ('full','ordinary','attributed','swapped','draft'):
            repaired=[name for name,wrong,_,trials in rows if wrong and trials[arm]['score']['certified']]
            correct=sum(r[3][arm]['score']['correct'] for r in rows)
            certified=sum(r[3][arm]['score']['certified'] for r in rows)
            print(arm,'correct',correct,'certified',certified,'certified correction of own wrong diagnosis',repaired)
            exact=[]
            endpoints=[]
            for name,_,_,trials in rows:
                choice=object_answer(trials[arm]['text'])
                interval=(choice.get('last_failure'),choice.get('first_pass'))
                if interval==trusted_intervals[name]:exact.append(name)
                if interval==(0,12):endpoints.append(name)
            pooled_repaired=[name for name,wrong,pooled,trials in rows if wrong and pooled and trials[arm]['score']['certified']]
            print('  exact audited interval',len(exact),'independent endpoints only',len(endpoints),
                  'certified after starting with the wrong pooled diagnosis',pooled_repaired)
        for compare in ('ordinary','swapped','full'):
            gains=[];losses=[]
            for name,_,_,trials in rows:
                a,b=(trials[x]['score']['certified'] for x in ('attributed',compare))
                if a and not b:gains.append(name)
                if b and not a:losses.append(name)
            n=len(gains)+len(losses)
            p=min(1,2*sum(comb(n,j) for j in range(min(len(gains),len(losses))+1))/2**n)
            print('attributed vs',compare,'gain',gains,'loss',losses,'two-sided sign p',p)


if __name__=='__main__':
    main()
