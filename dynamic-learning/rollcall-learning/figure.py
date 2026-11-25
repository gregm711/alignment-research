"""Plot the forward-test feedback trade-off and its dependence on surprising evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    s=json.loads((ROOT/"confirmation.json").read_text())["results"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.75),layout="constrained")
    entries=(("self-replay","Neural trust","#865a4b"),("template-replay","Template trust","#256971"))
    panels=((("all",2,"Truthful repairs"),("all",3,"Harmful flips")),
            (("surprising",2,"Unexpected vote"),("usual",2,"Expected vote")))
    for ax,cases,title in zip(axes,panels,("All checked decisions","Truthful repairs by predictability")):
        ax.axhline(0,c="#878787",lw=1)
        for j,(key,name,color) in enumerate(entries):
            y=100*np.array([s[group][key]["versus_constant"][metric] for group,metric,_ in cases])
            x=np.arange(2)+(j-.5)*.18
            ax.errorbar(x,y[:,0],yerr=np.stack([y[:,0]-y[:,1],y[:,2]-y[:,0]]),
                        fmt="o",color=color,capsize=4,label=name)
        ax.set(title=title,xticks=[0,1],xticklabels=[x[2] for x in cases],xlim=(-.38,1.38))
        ax.spines[["top","right"]].set_visible(False)
        ax.set_ylim(-7,2.3)
    axes[0].set_ylabel("Difference from constant trust (percentage points)")
    axes[0].legend(loc="lower left",frameon=False)
    fig.savefig(ROOT/"forward-feedback.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()
