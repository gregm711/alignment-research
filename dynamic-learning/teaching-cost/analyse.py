"""Summarize paired curves, check budgets, and render the registered endpoints."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from study import OUT, config, save_json, threshold
from learner import METHODS, CUTS, BUDGETS


def interval(x):
    x = np.asarray(x)
    assert x.ndim==1 and len(x)==96
    sampled=np.random.default_rng(86).choice(x,(4000,len(x)),replace=True)
    return [float(x.mean()),*np.quantile(sampled.mean(1),[.025,.975]).tolist()]


def summarize(domain, winner, control):
    with np.load(OUT/f'{domain}-lifetimes.npz') as d:
        common,native,matched = (d[k] for k in ('common','native','matched'))
        keys = [*METHODS,winner+'_reset',winner+'_shuffle']
        n = native[...,0]
        assert n.max()<=96 and n.min()>=0
        assert np.all(native[...,1]+native[...,2]==native[...,5])
        assert np.all(native[...,5]<=np.array(BUDGETS))
        assert np.all(native[...,1]==np.minimum(native[...,1],COUNT_LIMIT))
        assert np.all((d['traces']>=0).sum(-1)==native[...,-1,1])
        assert np.array_equal(common[:,0,keys.index(winner)],common[:,0,keys.index(winner+'_shuffle')])
        assert np.array_equal(common[:,:,keys.index('frozen')],np.repeat(common[:,:1,keys.index('frozen')],len(CUTS),axis=1))
        assert np.all(matched[...,1]==matched[:,:,0:1,:,:,1])
        calls=threshold(common).mean(3)
        solved=(n>=87).mean(3)
        w,c,win = (keys.index(k) for k in ('weights',control,winner))
        damage=solved[:,-1,: ,0,-1]-solved[:,-1,:,1,-1]
        contrasts = {
            'weights_gain_over_own_start':interval(calls[:,0,w]-calls[:,-1,w]),
            'weights_gain_over_control':interval(calls[:,-1,c]-calls[:,-1,w]),
            'winner_gain_over_own_start':interval(calls[:,0,win]-calls[:,-1,win]),
            'winner_gain_over_reset':interval(calls[:,-1,-2]-calls[:,-1,win]),
            'winner_gain_over_shuffled':interval(calls[:,-1,-1]-calls[:,-1,win]),
            'excess_misleading_damage_weights_over_control':interval(damage[:,w]-damage[:,c]),
            'excess_stale_damage_weights_over_control':interval(
                (solved[:,-1,w,0,-1]-solved[:,-1,w,2,-1])-(solved[:,-1,c,0,-1]-solved[:,-1,c,2,-1])),
            'winner_solved_gain_over_reset_same_winner_receipts':interval(
                ((matched[:,-1,win,:,0,0]>=87).mean(1)-(matched[:,-1,-2,:,0,0]>=87).mean(1))),
            'weights_solved_gain_over_control_same_winner_receipts':interval(
                ((matched[:,-1,w,:,0,0]>=87).mean(1)-(matched[:,-1,c,:,0,0]>=87).mean(1)))
        }
        results={}
        for j,key in enumerate(keys):
            entries=[]
            for i,cut in enumerate(CUTS):
                solve=solved[:,i,j]
                entries.append(dict(prior_tools=cut,examples=interval(calls[:,i,j]),
                    reaches=(threshold(common[:,i,j])<=6).mean().item(),
                    replay_accuracy=(common[:,i,j].mean((0,1))/96).tolist(),
                    replay_fraction_at_target=(common[:,i,j]>=87).mean((0,1)).tolist(),
                    native_solved=solve.mean(0).tolist(),
                    native_solved_counts=(n[:,i,j]>=87).sum((0,1)).tolist(),
                    native_mean_counts=native[:,i,j].mean((0,1)).tolist(),
                    fixed_winner_receipts_solved=(matched[:,i,j,...,0]>=87).mean((0,1)).tolist(),
                    misleading_damage=interval(solve[:,0,-1]-solve[:,1,-1]),
                    stale_damage=interval(solve[:,0,-1]-solve[:,2,-1]),
                    memory_bytes=d['memory'][:,i,j].mean().item(),
                    cpu_seconds=d['cpu'][:,i,j].mean(0).tolist()))
            results[key]=entries
        return dict(contrasts=contrasts,learners=results,
                    practice_cost=d['practice'].mean(0).tolist(),cpu_seconds=d['seconds'].sum().item())


COUNT_LIMIT=6


def main():
    frozen=config('frozen.json')
    s={dom:summarize(dom,frozen['winner'],frozen['control']) for dom in ('number','text')}
    save_json('summary.json',s)
    colors={'frozen':'#777777','kernel':'#279e75','neighbor':'#186ab4','weights':'#c65b19','uniform':'#9d72ab'}
    fig,axes=plt.subplots(1,2,figsize=(9,3.9),sharey=True)
    for ax,(domain,content) in zip(axes,s.items()):
        for key in ['uniform','frozen','kernel','neighbor','weights']:
            vals=np.array([row['examples'] for row in content['learners'][key]])
            ax.plot(CUTS,vals[:,0],marker='o',label=key,color=colors[key],lw=1.8)
            ax.fill_between(CUTS,vals[:,1],vals[:,2],color=colors[key],alpha=.10)
        ax.set_title({'number':'Numeric filters','text':'Text filters'}[domain])
        ax.set_xticks(CUTS, ['0','8','32','96'])
        ax.set_xlabel('Earlier practice tools')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Examples to reach 90% accuracy\n(unreached by 6 counted as 7)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,0,1,.89))
    fig.savefig(OUT/'teaching-curve.png',dpi=180)
    for dom,result in s.items():
        print(dom,result['contrasts'])
        for key in METHODS:
            print(key, [(round(r['examples'][0],2),r['native_solved_counts'][0]) for r in result['learners'][key]])


if __name__=='__main__':
    main()
