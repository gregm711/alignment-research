"""Physical decoder transplant and local single-request timing/power calibration."""
import copy
import json
import random
import subprocess
import threading
import time

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from encode import CACHE
from learner import COUNT, History, deploy, features, teacher
from study import OUT, config, save_json
from world import COLORS, case, phrases


def main():
    torch.set_num_threads(1)
    options=config('frozen.json')['settings']
    model=BertForMaskedLM.from_pretrained(str(CACHE),local_files_only=True).eval()
    tok=BertTokenizer.from_pretrained(str(CACHE),local_files_only=True)
    colors=tok.convert_tokens_to_ids(COLORS)
    archive=np.load(OUT/'encoder.npz')
    vec=features(archive)
    center=archive['practice_h'].mean(0).astype(float)
    scale=np.linalg.norm(archive['practice_h']-center,axis=1).mean()
    rng=np.random.default_rng(910000)
    perm=rng.permutation(4)
    history=[History(k,options[k]) for k in ('neighbor','weights')]
    with torch.inference_mode():
        def encode(text):
            batch=tok(text,return_tensors='pt')
            pos=(batch.input_ids[0]==tok.mask_token_id).nonzero().item()
            hidden=model.bert(**batch).last_hidden_state[0,pos]
            h=model.cls.predictions.transform(hidden)
            decoder=model.cls.predictions.decoder
            logits=decoder.weight[colors] @ h + decoder.bias[colors]
            return np.append((h.numpy()-center)/scale,.25),logits.numpy().astype(float)

        texts={domain:phrases(domain) for domain in ('practice','number')}
        training_cpu=time.process_time()
        for i in range(96):
            idx,target,mat=case(rng,'practice',int(rng.integers(4)),perm)
            _,label,_=teacher(target,mat,practice=True)
            v,logits=encode(texts['practice'][idx])
            if label is not None:
                for store in history:
                    store.save(v,logits,label)
        training_cpu=time.process_time()-training_cpu
        # Untie output embeddings, physically modify four decoder rows, and leave input embeddings intact.
        delta=history[1].delta
        corrected = copy.deepcopy(model)
        decoder = corrected.cls.predictions.decoder
        decoder.weight = torch.nn.Parameter(decoder.weight.detach().clone())
        decoder.weight[colors] += torch.tensor(delta[:,:-1]/scale,dtype=decoder.weight.dtype)
        decoder.bias[colors] += torch.tensor(.25*delta[:,-1]-(delta[:,:-1]/scale)@center,dtype=decoder.bias.dtype)
        error = 0.
        for idx,text in enumerate(texts['number']):
            batch=tok(text,return_tensors='pt')
            pos=(batch.input_ids[0]==tok.mask_token_id).nonzero().item()
            actual=corrected(**batch).logits[0,pos,colors].numpy()
            expected=vec['number'][1][idx]+delta@vec['number'][0][idx]
            error=max(error,float(np.max(abs(actual-expected))))
        assert error<1e-4
        cases=[case(rng,'number',i%4,perm) for i in range(32)]
        def task(i,key):
            idx,target,mat=cases[i%len(cases)]
            x,logits=encode(texts['number'][idx])
            store=history[key]
            prior=store.prior(x,logits)
            return deploy(prior,store.settings,target,mat)[0][-1][0]

        def meter():
            cmd=['macmon','pipe','--interval','200']
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,text=True)
            readings=[]
            def read():
                for row in proc.stdout:
                    obj=json.loads(row)
                    readings.append([time.monotonic(),obj['cpu_power'],obj['sys_power']])
            reader=threading.Thread(target=read,daemon=True)
            reader.start()
            schedule=random.Random(101)
            blocks=[]
            try:
                time.sleep(.6)
                for block in range(4):
                    modes=['idle','neighbor','weights']
                    schedule.shuffle(modes)
                    for mode in modes:
                        start=time.monotonic()
                        for phase in ('warmup','measure'):
                            limit=start+(.65 if phase=='warmup' else 2.65)
                            begin=time.monotonic()
                            cpu=time.process_time()
                            n=checksum=0
                            while time.monotonic()<limit:
                                if mode=='idle':
                                    time.sleep(.004)
                                else:
                                    checksum+=task(n,0 if mode=='neighbor' else 1)
                                    n+=1
                            duration=time.monotonic()-begin
                            elapsed=time.process_time()-cpu
                        samples=[(t-begin,c,s) for t,c,s in readings if begin+.22<t<=begin+duration]
                        assert len(samples)>=4
                        blocks.append(dict(block=block,mode=mode,duration=duration,cpu_seconds=elapsed,
                                           decisions=n,checksum=checksum,samples=samples))
            finally:
                proc.terminate()
                proc.wait(timeout=5)
                reader.join(timeout=3)
            return blocks
        blocks=meter()
    for b in blocks:
        if b['mode']=='idle':
            continue
        idle=next(i for i in blocks if i['mode']=='idle' and i['block']==b['block'])
        active_w=np.mean(np.array(b['samples'])[:,1:],axis=0)
        idle_w=np.mean(np.array(idle['samples'])[:,1:],axis=0)
        b['gross_mj']=(1000*b['duration']/b['decisions']*active_w).tolist()
        b['incremental_mj']=(1000*b['duration']/b['decisions']*(active_w-idle_w)).tolist()
    save_json('machine.json',dict(blocks=blocks,transplant_max_logit_error=error,
                                  joint_96_tool_build_cpu_seconds=training_cpu,
                                  timing_scope='single-thread full BERT, lookup/update prior, six-action evaluation and grading; pregenerated records'))
    print('maximum logit error',error,'build CPU seconds',training_cpu)


if __name__=='__main__':
    main()
