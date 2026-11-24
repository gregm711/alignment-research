"""Recheck a stored measurement claim before retrying the model's own verdict."""
import argparse
import hashlib
import json
from urllib.request import urlopen

from inertia import HERE, MODEL, content, decision, decision_prompt, load_messages, native_tokenizer, pad, save, unpack
from fork import answer


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--recount', action='store_true')
    args = parser.parse_args()
    root = HERE/'results'
    tag = args.model.replace(':','-')
    original = json.loads((root/f'inertia-{tag}.json').read_text())
    path = root/f'reconsider-{tag}.json'
    data = json.loads(path.read_text()) if path.exists() else {'counts':{},'decisions':{},'branches':{}}
    if not args.recount:
        with urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as response:
            digest = next(m['digest'] for m in json.load(response)['models'] if m['name']==args.model)
        assert data.setdefault('model_digest',digest)==original['model_digest']==digest
    tok = native_tokenizer(args.model)
    row = next(r for r in json.loads((HERE/'independent.json').read_text()) if r['instance'].endswith('-9288'))
    messages = load_messages(row)
    issue = content(messages[1]).split('<pr_description>',1)[1].split('</pr_description>',1)[0].strip()
    for state in ('reopen','close'):
        for source in ('count','wrong'):
            base = original['trials'][state+'/'+source]
            table,cached = base['note'].partition('\nPadding:')[0].rsplit('\nDistinct measured inertias: ',1)
            prior = json.loads(base['text'])['decision']
            rows = unpack(table)
            measured = len({line.rsplit(' = ',1)[1] for line in rows})
            assert rows==content(messages[row['broken_observation' if state=='reopen' else 'verified_observation']]).split('<output>')[1].split('</output>')[0].strip().splitlines()
            check = table+'\nHow many distinct measured inertia values occur across all worker counts?'
            key = fingerprint(check)
            if key not in data['counts']:
                assert not args.recount
                schema = {'type':'object','properties':{'count':{'type':'integer','minimum':1,'maximum':4}},'required':['count'],'additionalProperties':False}
                data['counts'][key] = answer(args.model,'Read the numerical measurements.',check,20,reply_format=schema,context=8192)
                save(path,data)
            inferred = json.loads(data['counts'][key]['text'])['count']
            sizes = []
            for method,count in [('retry',int(cached.removesuffix('.'))),('model',inferred),('parser',measured)]:
                note = pad(tok,table+f'\nDistinct measured inertias: {count}.')
                if method=='retry':assert note==base['note']
                prompt = decision_prompt(issue,row['command'],note)+f'\nYour previous decision was {prior}. The handoff count may have been stale. Reconsider using the measurements and give your final decision.'
                key = fingerprint(prompt)
                if key not in data['decisions']:
                    assert not args.recount
                    data['decisions'][key] = decision(args.model,prompt,8192)
                    save(path,data)
                trial = data['decisions'][key]
                sizes.append(trial['input_tokens'])
                branch = '/'.join([state,source,method])
                record = dict(previous=prior,count=count,prompt_sha256=key,final=json.loads(trial['text'])['decision'])
                if branch not in data['branches']:
                    assert not args.recount
                    data['branches'][branch] = record
                    save(path,data)
                assert data['branches'][branch]==record
                print(branch,prior,'->',record['final'],flush=True)
            assert len(set(sizes))==1


if __name__=='__main__':
    main()
