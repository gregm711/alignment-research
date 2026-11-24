"""Measure the deployed scalar decision path with the local macmon CPU/system sensors."""
import json
import random
from statistics import mean, median
import subprocess
from threading import Thread
import time

import numpy as np

from transit import ACTIONS, FEATURES, costs, greedy, line, maps, shortest
from study import OUT, predict, read, save


def examples(start, n):
    result=[]
    for seed in range(start,start+n):
        rng=random.Random(seed)
        graphs=[line(rng,False) for _ in range(2)]
        alert=rng.randrange(2)
        p=rng.choice((.15,.35,.5,.8))
        delay=rng.choice((3,7,12,20))
        result.append((graphs,alert,p,delay,rng.random()<p))
    return result


def deploy(case, beta, router):
    graphs,a,p,d,delayed=case
    public=[greedy(g) for g in graphs]
    times=[min(row[2],max(row[1],sum(c*v for c,v in zip(beta,row)))) for row in public]
    gap=times[a]-times[1-a]
    expected=gap+p*d
    value=min(expected,0)-(1-p)*min(gap,0)-p*min(gap+d,0)
    features=(len(graphs[a]),len(graphs[1-a]),gap,p,d,abs(expected),value,
              (public[a][2]-times[a])-(public[1-a][2]-times[1-a]))
    while isinstance(router,list):
        feature,cut,left,right=router
        router=left if features[FEATURES.index(feature)]<=cut else right
    if router=='plan':
        times=[shortest(g) for g in graphs]
    times[a]+=d*(delayed if router=='ask' else p)
    return ACTIONS.index(router),int(times[1]<times[0])


def main():
    frozen=read(OUT/'frozen.json')
    beta=frozen['beta']
    tree=frozen['trees'][frozen['selected']]
    cases=examples(510000,256)
    d=costs(maps(510000,256),np.array(beta))
    routers={'stop':'stop','plan':'plan','ask':'ask','selected':tree}
    for router in routers.values():
        decisions=predict(router,d['x'])
        for i,case in enumerate(cases):
            action,route=deploy(case,beta,router)
            assert action==decisions[i] and route==d['choices'][i,action,int(case[-1])]
    samples=[]
    proc=subprocess.Popen(['macmon','pipe','--interval','200'],stdout=subprocess.PIPE,text=True)
    def collect():
        for row in proc.stdout:
            obj=json.loads(row)
            samples.append((time.monotonic(),float(obj['cpu_power']),float(obj['sys_power'])))
    reader=Thread(target=collect,daemon=True)
    reader.start()
    blocks=[]
    rng=random.Random(53)
    try:
        time.sleep(1)
        for block in range(4):
            modes=list(routers)+['idle']
            rng.shuffle(modes)
            for mode in modes:
                start=time.monotonic()
                n=0
                measured=0
                checksum=0
                while time.monotonic()-start<2.5:
                    begin=time.monotonic()-start
                    if mode=='idle':
                        time.sleep(.005)
                    else:
                        for _ in range(16):
                            _,route=deploy(cases[n%len(cases)],beta,routers[mode])
                            checksum+=route
                            n+=1
                    end=time.monotonic()-start
                    if begin>=.65 and end<=2.25:
                        if not measured:
                            cpu_start=time.process_time()
                            active_start=begin
                        measured+=1
                        cpu_end=time.process_time()
                        active_end=end
                subset=[(stamp-start,cpu,system) for stamp,cpu,system in samples
                        if start+active_start+.20<=stamp<=start+active_end]
                assert len(subset)>=3 and all(c>=0 and s>0 for _,c,s in subset)
                blocks.append(dict(block=block,mode=mode,seconds=active_end-active_start,
                                   process_seconds=cpu_end-cpu_start,
                                   decisions=0 if mode=='idle' else measured*16,
                                   checksum=checksum, power_samples=subset,
                                   cpu_w=mean(c for _,c,_ in subset),system_w=mean(s for _,_,s in subset)))
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        reader.join(timeout=3)
    stats={}
    for mode in routers:
        by_block=[]
        for block in range(4):
            arm=next(r for r in blocks if r['block']==block and r['mode']==mode)
            idle=next(r for r in blocks if r['block']==block and r['mode']=='idle')
            dt=arm['seconds']/arm['decisions']
            by_block.append(dict(cpu_us=1e6*arm['process_seconds']/arm['decisions'],
                                 cpu_incremental_uj=1e6*(arm['cpu_w']-idle['cpu_w'])*dt,
                                 system_incremental_uj=1e6*(arm['system_w']-idle['system_w'])*dt,
                                 system_gross_uj=1e6*arm['system_w']*dt))
        stats[mode]=dict(by_block=by_block,median={k:median(r[k] for r in by_block) for k in by_block[0]})
    save(OUT/'meter.json',dict(blocks=blocks,stats=stats))
    for mode,result in stats.items():
        print(mode,result['median'])


if __name__=='__main__':
    main()
