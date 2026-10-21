"""Figure 2: recall of weak-label positives versus review budget (100k test set, text-only models).

Source: frozen_release_v2/predictions/Enron_100000_{tfidf,distilbert}_text_test_predictions.csv.
Shaded bands: 95% percentile bootstrap intervals (1,000 test resamples, seed 42). At very small budgets the
upper band can exceed the oracle line, because each resample has its own prevalence (its own oracle).
"""
import numpy as np, pandas as pd
from analysis import common as C

def data(kmax=15.0, step=0.25):
    y, sc = C.text_models_100k()
    grid = np.round(np.arange(0, kmax + 1e-9, step), 2)
    idx = C.bootstrap_indices(len(y))
    rows = {}
    for m in sc:
        boot = np.array([C.budget_curve(y[i], sc[m][i], grid) for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows[m] = pd.DataFrame({"k": grid, "wr": C.budget_curve(y, sc[m], grid), "lo": lo, "hi": hi})
    return rows, 100 * y.mean()

def make(show=False, kmax=15.0):
    import matplotlib.pyplot as plt
    C.use_paper_style()
    curves, prev = data(kmax)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot([0, prev, kmax], [0, 1, 1], color="black", ls=":", lw=1.0, label="Oracle")
    for m, label, ls in [("tfidf", "TF–IDF+LR", "-"), ("distil", "DistilBERT", "--")]:
        d = curves[m]
        ax.fill_between(d.k, d.lo, d.hi, color=C.COLORS[m], alpha=0.18, lw=0)
        ax.plot(d.k, d.wr, color=C.COLORS[m], ls=ls, lw=1.2, label=label)
    ax.plot([0, kmax], [0, kmax / 100], color=C.COLORS["grey"], ls="-.", lw=0.8, label="Random")
    ax.set_xlim(0, kmax); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, prev] + list(np.arange(5, kmax + 1e-9, 5)))
    ax.set_xticklabels(["0", f"{prev:.1f}"] + [f"{t:g}" for t in np.arange(5, kmax + 1e-9, 5)])
    ax.set_xlabel("Review budget $k$ (% of ranked test messages)")
    ax.set_ylabel("WR@$k$ (share of weak positives)")
    ax.grid(True)
    # legend in the empty region right of the curves and above the random line
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=True, framealpha=0.9, edgecolor="none")
    path = C.save_figure(fig, "fig2_review_budget")
    if show: plt.show()
    else: plt.close(fig)
    return path

if __name__ == "__main__":
    C.cli(__doc__); print("wrote", make())
