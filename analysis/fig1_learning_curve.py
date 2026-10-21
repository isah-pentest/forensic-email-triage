"""Figure 1: TF-IDF+LR learning curve on the fixed 100k test set.

Source: frozen_release_v2/results/E3_nested_learning_curve_tfidf.csv (five subsample seeds per size).
Points are means, error bars are +/-1 SD; the 80k point is the full training split, so it has no spread.
"""
import numpy as np
from analysis import common as C

def data():
    lc = C.v2_result("E3_nested_learning_curve_tfidf.csv")
    return lc.groupby("n_train")[["macro_f1", "pos_recall", "pr_auc"]].agg(["mean", "std"]).fillna(0)

def make(show=False):
    import matplotlib.pyplot as plt
    C.use_paper_style()
    agg = data(); x = agg.index.values
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for col, label, color, marker, ls in [("pos_recall", "Positive recall", C.COLORS["tfidf"], "^", "-"),
                                          ("macro_f1", "Macro F1", C.COLORS["distil"], "s", "-"),
                                          ("pr_auc", "PR-AUC", C.COLORS["green"], "o", "--")]:
        ax.errorbar(x, agg[(col, "mean")], yerr=agg[(col, "std")], color=color, marker=marker, ms=3.5, ls=ls,
                    capsize=2, elinewidth=0.6, label=label)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{int(v) // 1000}k" for v in x])
    ax.minorticks_off(); ax.set_xlim(x.min() * 0.8, x.max() * 1.25)
    lo = min((agg[(c, "mean")] - agg[(c, "std")]).min() for c in ["macro_f1", "pos_recall", "pr_auc"])
    hi = max((agg[(c, "mean")] + agg[(c, "std")]).max() for c in ["macro_f1", "pos_recall", "pr_auc"])
    ax.set_ylim(np.floor(lo * 20) / 20 - 0.05, np.ceil(hi * 20) / 20 + 0.05)
    ax.set_xlabel("Training messages (nested subsets of the 100k training split)")
    ax.set_ylabel("Score (fixed 100k test set)")
    ax.grid(True); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    path = C.save_figure(fig, "fig1_learning_curve")
    if show: plt.show()
    else: plt.close(fig)
    return path

if __name__ == "__main__":
    C.cli(__doc__); print("wrote", make())
