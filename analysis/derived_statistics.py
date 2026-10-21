"""Statistics quoted in the paper's text that are not stored in any saved result file.

    python -m analysis.derived_statistics [--v2 PATH] [--v1 PATH]

All are computed from the released 100k test predictions (and, for the superseded run,
results/tfidf_late_fusion_100k_predictions.csv). Paired bootstrap: the same 1,000 test
resamples (seed 42) for both members of each comparison. Writes outputs/derived_statistics.csv.
"""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from analysis import common as C

def make(show=False):
    tf = C.pred(C.S100, "tfidf_text")
    base = tf[["id", "y_true"]]; y = base.y_true.values
    def aligned(df): return base.merge(df[["id", "y_pred", "score"]], on="id")
    M = {"tfidf_text": aligned(tf), "distilbert_text": aligned(C.pred(C.S100, "distilbert_text")),
         "tfidf_nosw_score_only": aligned(C.pred(C.S100, "tfidf_oof_nosw_score_only")),
         "superseded_late_fusion": aligned(C.v1_result("tfidf_late_fusion_100k_predictions.csv"))}
    rows = []
    def add(q, est, lo=None, hi=None, note=""):
        rows.append({"quantity": q, "estimate": est, "ci_low": lo, "ci_high": hi, "note": note})

    # operating points: messages flagged at the 0.5 threshold
    for k, m in M.items():
        add(f"flagged_at_0.5:{k}", int(m.y_pred.sum()))
    # macro F1 when each ranking flags the same number of messages as the superseded run
    budget = int(M["superseded_late_fusion"].y_pred.sum())
    for k in ["tfidf_text", "distilbert_text", "tfidf_nosw_score_only"]:
        o = np.argsort(-M[k].score.values, kind="stable"); yp = np.zeros(len(y), int); yp[o[:budget]] = 1
        add(f"macro_f1_top{budget}:{k}", f1_score(y, yp, average="macro"), note="equal review volume")

    idx = C.bootstrap_indices(len(y))
    s_v1, s_tx = M["superseded_late_fusion"].score.values, M["tfidf_nosw_score_only"].score.values
    d = [average_precision_score(y[i], s_v1[i]) - average_precision_score(y[i], s_tx[i]) for i in idx]
    add("pr_auc_diff:superseded_minus_text_nosw", average_precision_score(y, s_v1) - average_precision_score(y, s_tx),
        *np.percentile(d, [2.5, 97.5]), "paired bootstrap")
    add("pr_auc:superseded_late_fusion", average_precision_score(y, s_v1))
    add("pr_auc:tfidf_nosw_score_only", average_precision_score(y, s_tx))
    s_d, s_t = M["distilbert_text"].score.values, M["tfidf_text"].score.values
    d = [C.wr_at_k(y[i], s_d[i], 2) - C.wr_at_k(y[i], s_t[i], 2) for i in idx]
    add("wr2_diff:distilbert_minus_tfidf", C.wr_at_k(y, s_d, 2) - C.wr_at_k(y, s_t, 2), *np.percentile(d, [2.5, 97.5]),
        "paired bootstrap")
    add("prevalence_pct_100k_test", 100 * y.mean(), note=f"{int(y.sum())} of {len(y)}")

    # truncation counts (E8 stores recall and n; counts are recall * n)
    e8 = C.v2_result("E8_truncation_analysis.csv"); e8 = e8[e8.scale == C.S100]
    for _, r in e8.iterrows():
        where = "inside" if r.trigger_inside_256 else "beyond"
        add(f"truncation_{where}_256:n_positives", int(r.n))
        add(f"truncation_{where}_256:distilbert_tp", int(round(r.distilbert_recall * r.n)))
        add(f"truncation_{where}_256:tfidf_tp", int(round(r.tfidf_recall * r.n)))
    df = pd.DataFrame(rows)
    df.to_csv(C.out_dir() / "derived_statistics.csv", index=False)
    if show:
        from analysis.table1_splits import display_df; display_df(df.round(4))
    return df

if __name__ == "__main__":
    C.cli(__doc__); print(make().round(4).to_string(index=False))
