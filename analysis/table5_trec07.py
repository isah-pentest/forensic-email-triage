"""Table V: TREC07 stress test of the Enron-trained 100k TF-IDF+LR text model.

Sources: results/cross_domain_trec07_eval.csv (metrics, macro-F1 bootstrap CI) and
results/cross_domain_trec07_predictions.csv (spam share, i.e. the PR-AUC of a random ordering).
"""
import pandas as pd
from analysis import common as C
from analysis.table1_splits import display_df

def make(show=False):
    tr = C.v1_result("cross_domain_trec07_eval.csv").iloc[0]
    spam = C.v1_result("cross_domain_trec07_predictions.csv").y_true.mean()
    rows = [("macro_f1", tr.macro_f1, tr.macro_f1_ci_low, tr.macro_f1_ci_high, None),
            ("spam_recall", tr.positive_recall, None, None, None), ("spam_f1", tr.positive_f1, None, None, None),
            ("pr_auc", tr.pr_auc, None, None, spam), ("roc_auc", tr.roc_auc, None, None, 0.5)]
    df = pd.DataFrame(rows, columns=["metric", "estimate", "ci_low", "ci_high", "chance_level"])
    df.attrs["n"], df.attrs["spam_share"] = int(tr.n_test), spam
    latex = [f"Macro F1 & {C.f3(tr.macro_f1)} [{C.f3(tr.macro_f1_ci_low)}, {C.f3(tr.macro_f1_ci_high)}] & -- \\\\",
             f"Spam recall / F1 & {C.f3(tr.positive_recall)} / {C.f3(tr.positive_f1)} & -- \\\\",
             f"PR-AUC & {C.f3(tr.pr_auc)} & {C.f3(spam)} \\\\",
             f"ROC-AUC & {C.f3(tr.roc_auc)} & 0.500 \\\\"]
    C.save_table(df, "table5_trec07", latex, ["results/cross_domain_trec07_eval.csv", "results/cross_domain_trec07_predictions.csv"])
    if show: display_df(df.round(3))
    return df

if __name__ == "__main__":
    C.cli(__doc__); print(make().round(3).to_string(index=False))
