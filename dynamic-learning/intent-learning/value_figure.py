"""Plot paired differences after changing the pretrained attention weights."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    first = json.loads((ROOT/"value-history.json").read_text())["change_minus"]
    second = json.loads((ROOT/"protected-values.json").read_text())["local_minus"]
    labels = ["Head ranking", "Familiar-output protection"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for ax, index, title in zip(axes, [(0,2,0), (1,1,3)],
                                ["Novel-intent accuracy after four reports", "Harm after one false report"]):
        regime, budget, metric = index
        for y, (data, label) in enumerate(zip((first,second),labels)):
            for offset, ref, color in ((.11,"ordinary","#2c567d"),(-.11,"matched","#ae542f")):
                v, lo, hi = 100*np.array(data[ref][regime][budget][metric])
                ax.errorbar(v, y+offset, xerr=[[v-lo],[hi-v]], fmt="o", markersize=6,
                            color=color, capsize=3, label=f"vs {ref} SGD" if y==0 else None)
        ax.axvline(0, color="0.5", lw=.8)
        ax.set_yticks([0,1],labels)
        ax.set_ylim(-.5,1.5)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel("Paired difference (percentage points)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="x",alpha=.15)
    axes[1].tick_params(labelleft=False)
    handles,_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Ordinary SGD", "SGD with matched update norm"],loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.13,1,1])
    fig.savefig(ROOT/"value-interventions.png",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
