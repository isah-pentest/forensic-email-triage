"""Table III: late fusion at 100k (held-out stacking) and the superseded run.

Sources: frozen_release_v2/results/E4_clean_stacking.csv, frozen_release_v2/predictions/Enron_100000_*,
results/tfidf_late_fusion_100k_predictions.csv (superseded run).
"""
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from analysis import common as C
from analysis.table1_splits import display_df

ROWS = [("tfidf_oof_nosw_score_only", "TF--IDF score$^\\ast$"), ("tfidf_oof_nosw_score+safe21", "TF--IDF$^\\ast$ + 21 meta"),
        ("tfidf_oof_nosw_score+meta25", "TF--IDF$^\\ast$ + 25 meta"), ("distilbert_valstack_score_only", "DistilBERT scores"),
        ("distilbert_valstack_score+meta25", "DistilBERT + 25 meta")]

def make(show=False):
    e4 = C.v2_result("E4_clean_stacking.csv"); e4 = e4[e4.scale == C.S100].set_index("config")
    rows, latex = [], []
    for cfg, label in ROWS:
        p = C.pred(C.S100, cfg); r = e4.loc[cfg]
        tp, fl = int(((p.y_pred == 1) & (p.y_true == 1)).sum()), int(p.y_pred.sum())
        rows.append({"config": cfg, "macro_f1": r.macro_f1, "tp": tp, "flagged": fl, "pr_auc": r.pr_auc, "wr10": r.wr10})
        latex.append(f"{label} & {C.f3(r.macro_f1)} & {tp} / {fl} & {C.f3(r.pr_auc)} & {C.f3(r.wr10)} \\\\")
    v1 = C.v1_result("tfidf_late_fusion_100k_predictions.csv"); y, s = v1.y_true.values, v1.score.values
    r = {"config": "superseded_v1_late_fusion_21meta", "macro_f1": f1_score(y, v1.y_pred, average="macro"),
         "tp": int(((v1.y_pred == 1) & (v1.y_true == 1)).sum()), "flagged": int(v1.y_pred.sum()),
         "pr_auc": average_precision_score(y, s), "wr10": C.wr_at_k(y, s, 10)}
    rows.append(r)
    latex += ["\\midrule", f"Superseded run (21 meta) & {C.f3(r['macro_f1'])} & {r['tp']} / {r['flagged']} & "
              f"{C.f3(r['pr_auc'])} & {C.f3(r['wr10'])} \\\\"]
    df = pd.DataFrame(rows)
    C.save_table(df, "table3_late_fusion", latex, ["frozen_release_v2/results/E4_clean_stacking.csv",
                 "frozen_release_v2/predictions/Enron_100000_*", "results/tfidf_late_fusion_100k_predictions.csv"])
    if show: display_df(df.round(3))
    return df

if __name__ == "__main__":
    C.cli(__doc__); print(make().round(3).to_string(index=False))
