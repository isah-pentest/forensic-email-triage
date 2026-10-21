"""Table IV: budget recall at 100k with 95% bootstrap CIs, effort to reach a recall target, and TREC07.

Sources: frozen_release_v2/predictions/Enron_100000_{tfidf,distilbert}_text_test_predictions.csv,
frozen_release_v2/results/E2_bootstrap_ci_all.csv (WR@10/20, same values as Table II),
results/cross_domain_trec07_{eval,predictions}.csv.
WR@2/5 intervals: 1,000 test resamples, seed 42 (the same resamples as the Fig. 2 bands).
"""
import numpy as np, pandas as pd
from analysis import common as C
from analysis.table1_splits import display_df

def make(show=False):
    y, sc = C.text_models_100k(); idx = C.bootstrap_indices(len(y))
    names = {"tfidf": "tfidf_text", "distil": "distilbert_text"}
    rows, latex = [], []
    for k in [2, 5, 10, 20]:
        cells = []
        for m, cfg in names.items():
            if k in (10, 20):
                r = C.e2_ci(C.S100, cfg, f"wr{k}"); pt, lo, hi = r.point, r.ci_low, r.ci_high
            else:
                pt = C.wr_at_k(y, sc[m], k)
                lo, hi = np.percentile([C.budget_curve(y[i], sc[m][i], [k])[0] for i in idx], [2.5, 97.5])
            rows.append({"quantity": f"WR@{k}", "model": m, "estimate": pt, "ci_low": lo, "ci_high": hi})
            cells.append(f"{C.f3(pt)} [{C.f3(lo)}, {C.f3(hi)}]")
        latex.append(f"{k}\\,\\%  & {cells[0]} & {cells[1]} \\\\")
    eff = {}
    for m in names:
        eff[m] = [100 * C.effort_for_recall(y, sc[m], t) for t in (.8, .9, .95)]
        for t, e in zip((80, 90, 95), eff[m]):
            rows.append({"quantity": f"pct_list_for_{t}pct_recall", "model": m, "estimate": e, "ci_low": None, "ci_high": None})
    latex.append("\\% list for 80/90/95\\,\\% & " + " & ".join(" / ".join(f"{e:.1f}" for e in eff[m]) for m in names) + " \\\\")
    tr = C.v1_result("cross_domain_trec07_eval.csv").iloc[0]
    spam = C.v1_result("cross_domain_trec07_predictions.csv").y_true.mean()
    for q, v, lo, hi in [("trec07_macro_f1", tr.macro_f1, tr.macro_f1_ci_low, tr.macro_f1_ci_high),
                         ("trec07_spam_recall", tr.positive_recall, None, None), ("trec07_pr_auc", tr.pr_auc, None, None),
                         ("trec07_roc_auc", tr.roc_auc, None, None), ("trec07_spam_share", spam, None, None)]:
        rows.append({"quantity": q, "model": "tfidf", "estimate": v, "ci_low": lo, "ci_high": hi})
    latex += ["\\midrule",
              f"\\multicolumn{{3}}{{l}}{{\\textit{{TREC07 ($N$={int(tr.n_test):,}; spam {100 * spam:.1f}\\,\\%), TF--IDF+LR}}}} \\\\",
              f"Macro F1 / spam recall & \\multicolumn{{2}}{{c}}{{{C.f3(tr.macro_f1)} [{C.f3(tr.macro_f1_ci_low)}, {C.f3(tr.macro_f1_ci_high)}] / {C.f3(tr.positive_recall)}}} \\\\",
              f"PR-AUC (chance {C.f3(spam)}) & \\multicolumn{{2}}{{c}}{{{C.f3(tr.pr_auc)}}} \\\\",
              f"ROC-AUC (chance 0.500) & \\multicolumn{{2}}{{c}}{{{C.f3(tr.roc_auc)}}} \\\\"]
    df = pd.DataFrame(rows)
    C.save_table(df, "table4_review_budget", latex, ["frozen_release_v2/predictions/Enron_100000_{tfidf,distilbert}_text_test_predictions.csv",
                 "frozen_release_v2/results/E2_bootstrap_ci_all.csv", "results/cross_domain_trec07_{eval,predictions}.csv"])
    if show: display_df(df.round(3))
    return df

if __name__ == "__main__":
    C.cli(__doc__); print(make().round(3).to_string(index=False))
