"""Plot the paired history intervention and the two consequences of limiting its sign."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    history = json.loads((ROOT/"summary.json").read_text())["checkpoints"]
    means = json.loads((ROOT/"direction-summary.json").read_text())["feedback"]["4"]["means"]
    fig,(left,right) = plt.subplots(1,2,figsize=(10.8,4),layout="constrained")
    ticks = list(map(int,history))
    effect = 100*np.array([v["effects"]["history-swap"][2] for v in history.values()])
    x = np.array(ticks)
    left.axhline(0,color=".65",lw=1)
    left.axvline(48,color=".45",lw=1,ls="--")
    left.errorbar(x,effect[:,0],yerr=np.stack([effect[:,0]-effect[:,1],effect[:,2]-effect[:,0]]),
                  fmt="o",color="#16666d",capsize=4)
    left.plot(x[:2],effect[:2,0],color="#16666d")
    # Switching the reference pairing reverses this paired contrast before another update.
    left.plot(x[1:],np.r_[-effect[1,0],effect[2:,0]],color="#16666d")
    left.plot(x[1],-effect[1,0],marker="o",mfc="white",mec="#16666d")
    left.set(xticks=ticks,xlabel="Batches of eight audited incidents",ylabel="Accuracy gain (percentage points)",
             title="Does the learned habit fit the current rule?",ylim=(-10,12))
    left.text(47,-9,"Pairing changes",ha="right",fontsize=9)
    for j,(kind,c,label) in enumerate((("signed","#495a75","Signed attention inputs"),("positive","#329285","Nonnegative attention inputs"))):
        values = np.array(means[kind])[0,0,0,::-1,2]*100
        rects = right.bar(np.arange(2)+(j-.5)*.34,values,.33,color=c,label=label)
        right.bar_label(rects,fmt="%.1f%%",padding=3,fontsize=9)
    right.set(xticks=[0,1],xticklabels=["False report,\nnegative check","True report,\npositive check"],
              ylabel="Unrelated requests flagged (%)",ylim=(0,67),title="How far does the checked edit spread?")
    right.legend(frameon=False,loc="upper right",fontsize=9)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(ROOT/"correction-effects.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()
