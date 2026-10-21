"""Table II: Enron test results by scale and configuration, with 95% bootstrap CIs.

Sources: frozen_release_v2/results/E2_bootstrap_ci_all.csv (point estimates and CIs) and
frozen_release_v2/predictions/*_test_predictions.csv (true-positive counts).
"""
import pandas as pd
from analysis import common as C
from analysis.table1_splits import display_df

CONFIGS = [("tfidf_text", "TF--IDF+LR (text)"), ("distilbert_text", "DistilBERT (text)"),
           ("meta_only", "Metadata-only (25)"), ("early_fusion", "TF--IDF+meta, early")]

def make(show=False):
    rows, latex = [], []
    for s in C.SCALES:
        npos = int(C.pred(s, "tfidf_text").y_true.sum())
        for j, (cfg, label) in enumerate(CONFIGS):
            p = C.pred(s, cfg); tp = int(((p.y_pred == 1) & (p.y_true == 1)).sum())
            r = {met: C.e2_ci(s, cfg, met) for met in ["macro_f1", "pos_recall", "pos_f1", "pr_auc", "wr10"]}
            rows.append({"scale": s, "config": cfg, "tp": tp, "n_pos": npos,
                         **{f"{met}{suf}": getattr(r[met], col) for met in r for suf, col in [("", "point"), ("_lo", "ci_low"), ("_hi", "ci_high")]}})
            lead = f"{s.split('_')[1][:-3]}k ({npos} pos.)" if j == 0 else ""
            latex.append(f"{lead} & {label} & {C.ci_text(r['macro_f1'])} & {C.f3(r['pos_recall'].point)} ({tp}) & "
                         f"{C.f3(r['pos_f1'].point)} & {C.ci_text(r['pr_auc'])} & {C.ci_text(r['wr10'])} \\\\")
        if s != C.SCALES[-1]:
            latex.append("\\midrule")
    df = pd.DataFrame(rows)
    C.save_table(df, "table2_main_results", latex,
                 ["frozen_release_v2/results/E2_bootstrap_ci_all.csv", "frozen_release_v2/predictions/*_test_predictions.csv"])
    if show: display_df(df.round(3))
    return df

if __name__ == "__main__":
    C.cli(__doc__); print(make().round(3).to_string(index=False))
