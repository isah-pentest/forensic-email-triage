"""Build a BLINDED annotation sheet from the released 174-message review sample.
Annotators see only an annotation_id, subject and text. Model outputs and weak labels go to a
separate key file that annotators must not open."""
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
h = pd.read_csv(src)
rng = np.random.default_rng(2027)
h = h.iloc[rng.permutation(len(h))].reset_index(drop=True)
h["annotation_id"] = [f"A{i+1:03d}" for i in range(len(h))]
sheet = h[["annotation_id", "subject", "text"]].copy()
for c in ["review_worthy", "phish_like", "message_type", "confidence", "notes"]:
    sheet[c] = ""
for who in ["annotator1", "annotator2"]:
    sheet.to_csv(f"{out}/annotation_sheet_{who}.csv", index=False)
    with pd.ExcelWriter(f"{out}/annotation_sheet_{who}.xlsx") as xw:
        sheet.to_excel(xw, index=False, sheet_name="annotate")
h[["annotation_id", "id", "case_type", "y_true", "y_pred", "score"]].to_csv(f"{out}/KEY_do_not_share_with_annotators.csv", index=False)
print(len(sheet), "rows written")
