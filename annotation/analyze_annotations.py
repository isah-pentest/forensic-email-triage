"""Score two completed annotation sheets against the weak labels.

Usage (from the repo root):
  python annotation/analyze_annotations.py annotation/llm_annotations/llm_sheet1_completed.csv \
      annotation/llm_annotations/llm_sheet2_completed.csv annotation/annotation_key.csv [out.csv]

Stratum sizes are read from the superseded 100k late-fusion predictions that the sample
was drawn from (results/tfidf_late_fusion_100k_predictions.csv: TP 167, FP 193, FN 24,
TN 9,616). The sample drew 50/50/24(all)/50 at random within strata, so estimates are
re-weighted by N_stratum / n_stratum. Intervals: percentile bootstrap, seed 42, 2,000
resamples (kappa: resampling messages; proportions: resampling within strata).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

SEED, NB = 42, 2000
REPO = Path(__file__).resolve().parents[1]
a1, a2, key = (pd.read_csv(p) for p in sys.argv[1:4])
out = sys.argv[4] if len(sys.argv) > 4 else REPO / "annotation" / "llm_annotation_summary.csv"

v1 = pd.read_csv(REPO / "results" / "tfidf_late_fusion_100k_predictions.csv")
N = {c: int(((v1.y_pred == (c in ("TP", "FP"))) & (v1.y_true == (c in ("TP", "FN")))).sum()) for c in ["TP", "FP", "FN", "TN"]}
d = key.merge(a1[["annotation_id", "review_worthy"]], on="annotation_id") \
       .merge(a2[["annotation_id", "review_worthy"]], on="annotation_id", suffixes=("_1", "_2"))
for c in ["review_worthy_1", "review_worthy_2"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
assert d[["review_worthy_1", "review_worthy_2"]].notna().all().all(), "every row needs review_worthy = 0 or 1"
r1, r2 = d.review_worthy_1.values.astype(int), d.review_worthy_2.values.astype(int)

kap = cohen_kappa_score(r1, r2)
rng = np.random.default_rng(SEED); kb = []
for _ in range(NB):
    i = rng.integers(0, len(d), len(d)); kb.append(cohen_kappa_score(r1[i], r2[i]))
klo, khi = np.percentile(kb, [2.5, 97.5])
print(f"Cohen's kappa (review_worthy): {kap:.3f} [{klo:.3f}, {khi:.3f}];  raw agreement {(r1 == r2).sum()}/{len(d)}")
dis = d.loc[r1 != r2, "annotation_id"].tolist()
print("Disagreements (adjudicate with humans):", ", ".join(dis) if dis else "none")

def strat(combine, strata):
    rw = combine(r1, r2); fr = {c: rw[(d.case_type == c).values] for c in strata}
    est = lambda f: sum(N[c] * f[c].mean() for c in strata) / sum(N[c] for c in strata)
    r = np.random.default_rng(SEED)
    bs = [est({c: v[r.integers(0, len(v), len(v))] for c, v in fr.items()}) for _ in range(NB)]
    return (est(fr), *np.percentile(bs, [2.5, 97.5]))

rows = [("cohen_kappa_review_worthy", kap, klo, khi), ("raw_agreement", float((r1 == r2).mean()), None, None)]
for rn, comb in [("either", np.maximum), ("both", np.minimum)]:
    for sn, strata in [("weak", ["TP", "FN"]), ("flag", ["TP", "FP"])]:
        e = strat(comb, strata); rows.append((f"p_review_worthy_given_{sn}_{rn}", *e))
        print(f"P(review-worthy | {'weak label = 1' if sn == 'weak' else 'model flags'}), {rn}: {e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]")
ntn = int((d.case_type == "TN").sum()); tn_pos = int(np.maximum(r1, r2)[(d.case_type == "TN").values].sum())
ub = 1 - 0.025 ** (1 / ntn) if tn_pos == 0 else float("nan")
rows += [("tn_stratum_review_worthy_count", tn_pos, None, None), ("tn_upper_bound_97.5", ub, None, None)]
print(f"TN stratum: {tn_pos}/{ntn} review-worthy (one-sided 97.5% upper bound {ub:.3f}); cannot bound what the rules miss.")
print("\nPer-stratum counts (either / both):")
for c in ["TP", "FP", "FN", "TN"]:
    m = (d.case_type == c).values
    print(f"  {c}: {np.maximum(r1, r2)[m].sum()}/{m.sum()}  /  {np.minimum(r1, r2)[m].sum()}/{m.sum()}")
pd.DataFrame(rows, columns=["quantity", "estimate", "ci_low", "ci_high"]).to_csv(out, index=False)
print("wrote", out)
