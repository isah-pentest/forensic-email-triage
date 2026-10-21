"""Score the two completed annotation sheets against the weak labels.
Usage: python analyze_annotations.py annotator1.csv annotator2.csv KEY.csv
Stratum sizes are those of the 100k TF-IDF late-fusion test set (released predictions):
TP 167, FP 193, FN 24, TN 9616. The sample drew 50/50/24(all)/50 at random within strata,
so estimates are re-weighted by N_stratum / n_stratum."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
a1, a2, key = (pd.read_csv(p) for p in sys.argv[1:4])
N = {"TP": 167, "FP": 193, "FN": 24, "TN": 9616}
d = key.merge(a1[["annotation_id", "review_worthy", "phish_like"]], on="annotation_id") \
       .merge(a2[["annotation_id", "review_worthy", "phish_like"]], on="annotation_id", suffixes=("_1", "_2"))
for c in ["review_worthy_1", "review_worthy_2"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
assert d[["review_worthy_1", "review_worthy_2"]].notna().all().all(), "every row needs review_worthy = 0 or 1"
print("Cohen's kappa (review_worthy):", round(cohen_kappa_score(d.review_worthy_1, d.review_worthy_2), 3))
print("Raw agreement:", round((d.review_worthy_1 == d.review_worthy_2).mean(), 3))
# adjudicated label: agreement, else flag for adjudication (here: conservative = 1 if either says 1)
d["rw"] = np.where(d.review_worthy_1 == d.review_worthy_2, d.review_worthy_1, np.nan)
print("Rows needing adjudication:", int(d.rw.isna().sum()), "-> resolve these and re-run; using max() meanwhile")
if d.rw.isna().any():
    print("  disagreements:", ", ".join(d.loc[d.rw.isna(), "annotation_id"]))
d["rw"] = d.rw.fillna(d[["review_worthy_1", "review_worthy_2"]].max(axis=1))

def weighted(strata, n_boot=2000, seed=42):
    """Re-weighted proportion of review-worthy messages over the given strata, with stratified bootstrap CI."""
    rng = np.random.default_rng(seed)
    def est(frames):
        num = sum(N[s] * f.rw.mean() for s, f in frames.items()); den = sum(N[s] for s in frames)
        return num / den
    frames = {s: d[d.case_type == s] for s in strata}
    pt = est(frames); bs = []
    for _ in range(n_boot):
        bs.append(est({s: f.sample(len(f), replace=True, random_state=int(rng.integers(1e9))) for s, f in frames.items()}))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi

rows = [("P(review-worthy | weak label = 1)  [weak-label precision]", ["TP", "FN"]),
        ("P(review-worthy | model flags)      [model precision vs humans]", ["TP", "FP"]),
        ("P(review-worthy | weak label = 0)  [missed by rules]", ["FP", "TN"]),
        ("P(review-worthy | model not flag)   [missed by model]", ["FN", "TN"])]
for name, s in rows:
    pt, lo, hi = weighted(s); print(f"{name}: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
print("\nPer-stratum review-worthy rate:"); print(d.groupby("case_type").rw.agg(["mean", "count"]).round(3))
print("\nNote: the TN stratum (50 of 9,616) gives a coarse estimate of review-worthy messages the rules miss.")
