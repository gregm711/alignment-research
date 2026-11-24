"""Separate new repairs from retained working programs on the frozen incident panel."""
import argparse
import json

from experiment import CASES,HERE,MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODEL)
    args = parser.parse_args()
    data = json.loads((HERE/(args.model.replace(':','-')+'.json')).read_text())
    assert set(data['cases'])=={c.func for c in CASES}
    for start in ('own','parent','fixed'):
        arms = ('full','ordinary','stale','verified') if start=='own' else ('stale','verified')
        for arm in arms:
            good,repairs,losses,edits,unapplied = [],[],[],[],[]
            for name,row in data['cases'].items():
                branch = row['branches'][start+'/'+arm]
                before,after = [row['scores'][branch[k]]['passed'] for k in ('before','after')]
                if before:good.append(name)
                if not before and after:repairs.append(name)
                if before and not after:losses.append(name)
                if branch['action']=='edit':edits.append(name)
                if branch['action'] in ('unmatched','invalid'):unapplied.append(name)
            print(start,arm,'initial passes',len(good),'final passes',len(good)+len(repairs)-len(losses),
                  'repaired',repairs,'lost',losses,'applied edits',len(edits),'unapplied',len(unapplied))


if __name__=='__main__':
    main()
