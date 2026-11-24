"""Build a portable, offline view of the original controlled trials."""
import json
from cases import CASES
from screen import HERE, MODEL, NAMES, display, materials
from restore import WRITER_PATH

ARMS=('ordinary','observed','swapped','full','none')


def dataset():
    result={}
    for i,c in enumerate(CASES):
        prior=json.loads((WRITER_PATH/(c.name+'.json')).read_text())
        _,expected=materials(c,i)
        item={'purpose':c.purpose,'inputs':c.common+c.hidden,'smoke':len(c.common),'expected':expected,'models':{}}
        captures={display(x) for x in c.captures}
        item['archived_rows']=[j for j,x in enumerate(c.hidden,len(c.common)) if display(x) in captures]
        for model in (MODEL,'gemma3:12b'):
            tag=model.replace(':','-')
            follow=json.loads((HERE/'results'/('restore-'+tag)/(c.name+'.json')).read_text())
            trials={**(prior['trials'] if model==MODEL else {}),**follow['trials']}
            answers=json.loads((HERE/'results'/('readout-'+tag+'.json')).read_text())['trials']
            view={}
            for name in NAMES:
                for arm in ARMS:
                    key=f'{name}-{arm}'
                    t=trials[key]
                    checks=[]
                    for flip in (0,1):
                        a=answers[f'{c.name}/{name}/{arm}/{flip}']
                        checks.append(json.loads(a['text'])['choice']==a['expected'])
                    view[key]={'note':follow['notes'][arm],'patch':t['text'],'tokens':t['input_tokens'],
                               'score':t['score'],'checks':checks}
            item['models'][model]=view
        result[c.name]=item
    return result


if __name__=='__main__':
    template=(HERE/'replay-template.html').read_text()
    payload=json.dumps(dataset(),ensure_ascii=True,separators=(',',':')).replace('<','\\u003c')
    assert template.count('@@TRIALS@@')==1
    (HERE/'replay.html').write_text(template.replace('@@TRIALS@@',payload))
