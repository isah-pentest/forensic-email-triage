# %% [markdown]
# # Limitations upgrade — frozen_release_v2
# Run in Colab after `drive.mount('/content/drive')`.
# Reads the existing project files READ-ONLY and writes every new output to
# `<ROOT>/frozen_release_v2/` (the v1 freeze note forbids overwriting v1).
#
# Each experiment maps to one manuscript limitation:
#   E1 per-message predictions for every configuration ...... "Incomplete uncertainty", WR@k only for one model
#   E2 bootstrap CIs + PAIRED bootstrap differences .......... "CI overlap", "no paired test"
#   E3 nested learning curve (fixed 100k test set) + split seeds  "single seed / independent samples"
#   E4 held-out (validation) and out-of-fold stacking ......... "in-sample stacking", "post-hoc safe list"
#   E5 exact/near-duplicate audit + grouped re-split .......... "message-level random split"
#   E6 word-boundary rule sensitivity ......................... substring artefacts in the rules
#   E7 rule-phrase masking probe .............................. "labels are a function of text" (circularity)
#   E8 DistilBERT truncation analysis ......................... DistilBERT recall gap
# Human validation (the largest limitation) is handled by the separate annotation package.
#
# Cost: E1–E7 are CPU/sklearn only (minutes). E1/E8 need DistilBERT inference on a GPU only
# if `<scale>_distil_scores_test.csv` is missing. No DistilBERT re-training is required, except
# the OPTIONAL multi-seed block at the end (GPU, ~hours).

# %%
import os, re, json, hashlib, yaml, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("FET_ROOT", "/content/drive/MyDrive/forensic_email_project"))
V1 = ROOT / "frozen_release_v1"
OUT = ROOT / "frozen_release_v2"
for sub in ["results", "predictions", "manifests"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
SEED = 42
SCALES = ["Enron_10000", "Enron_50000", "Enron_100000"]
N_BOOT = 1000

def find(rel_candidates):
    """Return the first existing path among candidates (v1 freeze first, then project root)."""
    for base in [V1, ROOT]:
        for rel in rel_candidates:
            p = base / rel
            if p.exists():
                return p
    raise FileNotFoundError(f"None of {rel_candidates} found under {V1} or {ROOT}")

RULES = yaml.safe_load(open(find(["rules/phishing_rules_v2.yaml"])))
METADATA_25 = ["is_internal_sender","is_external_sender","num_to_recipients","num_cc_recipients",
  "num_bcc_recipients","has_cc","has_bcc","is_reply_subject","is_forward_subject","num_recipient_domains",
  "is_internal_only_thread","is_mixed_domain_thread","subject_char_len","body_char_len","subject_token_len",
  "body_token_len","body_to_subject_len_ratio","has_empty_subject","has_short_body","hour_of_day",
  "is_business_hours","day_of_week","is_weekend","has_attachment_mention","has_multiple_recipients"]
SAFE_21 = [f for f in METADATA_25 if f not in
           {"is_internal_sender","is_external_sender","is_internal_only_thread","is_mixed_domain_thread"}]

def load_scale(scale):
    """Text + metadata + split label for one scale, aligned on id."""
    text = pd.read_csv(find([f"data/{scale}_full.csv"]))
    meta = pd.read_csv(find([f"data/{scale}_metadata_v1.csv"]))
    keep = ["id", "text", "y_bin"] + [c for c in ["subject", "body", "message"] if c in text.columns]
    df = text[keep].merge(meta[["id"] + METADATA_25], on="id", how="inner")
    parts = []
    for sp in ["train", "val", "test"]:
        ids = pd.read_csv(find([f"splits/{scale}/{sp}_ids.csv"]))
        parts.append(df.merge(ids, on="id").assign(split=sp))
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].fillna("")
    assert df.id.is_unique, "duplicate ids after merge"
    return df

# %% [markdown]
# ## Metric helpers (identical definitions to the manuscript)

# %%
def wr_at_k(y, s, k):
    n = int(len(y) * k / 100)
    o = np.argsort(-s, kind="stable")
    return y[o][:n].sum() / max(y.sum(), 1)

def metrics(y, pred, score):
    return dict(macro_f1=f1_score(y, pred, average="macro"),
                pos_recall=recall_score(y, pred, zero_division=0),
                pos_precision=precision_score(y, pred, zero_division=0),
                pos_f1=f1_score(y, pred, zero_division=0),
                pr_auc=average_precision_score(y, score), roc_auc=roc_auc_score(y, score),
                wr2=wr_at_k(y, score, 2), wr5=wr_at_k(y, score, 5),
                wr10=wr_at_k(y, score, 10), wr20=wr_at_k(y, score, 20),
                tp=int(((pred == 1) & (y == 1)).sum()), n_pos=int(y.sum()), n=len(y))

def bootstrap(preds, n_boot=N_BOOT, seed=SEED):
    """preds: dict name -> DataFrame(id,y_true,y_pred,score) on the SAME test ids.
    Returns per-model percentile CIs and paired differences vs every other model."""
    names = list(preds)
    base = preds[names[0]][["id", "y_true"]]
    al = {k: base.merge(v[["id", "y_pred", "score"]], on="id") for k, v in preds.items()}
    y = base.y_true.values
    rng = np.random.default_rng(seed)
    keys = ["macro_f1", "pos_recall", "pos_f1", "pr_auc", "wr10", "wr20"]
    draws = {k: {m: [] for m in keys} for k in names}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        for k in names:
            m = metrics(y[i], al[k].y_pred.values[i], al[k].score.values[i])
            for mm in keys:
                draws[k][mm].append(m[mm])
    rows = []
    for k in names:
        pt = metrics(y, al[k].y_pred.values, al[k].score.values)
        for mm in keys:
            lo, hi = np.percentile(draws[k][mm], [2.5, 97.5])
            rows.append(dict(model=k, metric=mm, point=pt[mm], ci_low=lo, ci_high=hi))
    diffs = []
    for a in names:
        for b in names:
            if a < b:
                for mm in keys:
                    d = np.array(draws[a][mm]) - np.array(draws[b][mm])
                    pt = metrics(y, al[a].y_pred.values, al[a].score.values)[mm] - \
                         metrics(y, al[b].y_pred.values, al[b].score.values)[mm]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    diffs.append(dict(a=a, b=b, metric=mm, diff=pt, ci_low=lo, ci_high=hi,
                                      frac_draws_a_gt_b=(d > 0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(diffs)

# %% [markdown]
# ## Model builders (configurations as recorded in the v1 notebook)

# %%
def tfidf_lr(stop_words="english"):
    return (TfidfVectorizer(max_features=20000, stop_words=stop_words),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))

def fit_text(tr, te, stop_words="english"):
    vec, clf = tfidf_lr(stop_words)
    Xtr = vec.fit_transform(tr.text); clf.fit(Xtr, tr.y_bin)
    Xte = vec.transform(te.text)
    return clf.predict(Xte), clf.predict_proba(Xte)[:, 1], (vec, clf)

def meta_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])

def fit_meta(tr, te, feats):
    p = meta_pipe().fit(tr[feats], tr.y_bin)
    return p.predict(te[feats]), p.predict_proba(te[feats])[:, 1]

def fit_early(tr, te, feats):
    vec, clf = tfidf_lr("english")
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xtr = hstack([vec.fit_transform(tr.text), csr_matrix(sc.fit_transform(imp.fit_transform(tr[feats])))])
    Xte = hstack([vec.transform(te.text), csr_matrix(sc.transform(imp.transform(te[feats])))])
    clf.fit(Xtr, tr.y_bin)
    return clf.predict(Xte), clf.predict_proba(Xte)[:, 1]

def pred_frame(te, pred, score):
    return pd.DataFrame({"id": te.id.values, "y_true": te.y_bin.values.astype(int),
                         "y_pred": np.asarray(pred).astype(int), "score": np.asarray(score)})

# %% [markdown]
# ## E1 — Per-message test predictions for every configuration
# Reproduction check: each re-run row is compared with the v1 CSV value. A mismatch > 1e-6
# means the v1 number is not reproducible from the frozen data and must be reported as such.

# %%
V1_RESULTS = {
 ("Enron_10000","tfidf_text"):0.7209986557207958, ("Enron_50000","tfidf_text"):0.7354361156873512,
 ("Enron_100000","tfidf_text"):0.7607165392400292,
 ("Enron_10000","meta_only"):0.506934064108952, ("Enron_50000","meta_only"):0.5319259615609351,
 ("Enron_100000","meta_only"):0.5587576716870052,
 ("Enron_10000","early_fusion"):0.692282289719977, ("Enron_50000","early_fusion"):0.7381513485205551,
 ("Enron_100000","early_fusion"):0.7652383769520429,
 ("Enron_10000","distilbert_text"):0.7631133671742809, ("Enron_50000","distilbert_text"):0.7889577414257487,
 ("Enron_100000","distilbert_text"):0.8472342955743479,
}

def distilbert_scores(scale, split, df_split):
    """Use saved v1 DistilBERT scores if present; otherwise run inference with the saved model."""
    for base in [V1 / "data", ROOT / "data"]:
        p = base / f"{scale}_distil_scores_{split}.csv"
        if p.exists():
            s = pd.read_csv(p)
            return df_split[["id"]].merge(s, on="id", how="left")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mdir = find([f"models/distilbert_{scale}_weighted_best"])
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    logits = []
    for i in range(0, len(df_split), 64):
        enc = tok(list(df_split.text.iloc[i:i+64]), truncation=True, max_length=256,
                  padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits.append(model(**enc).logits.float().cpu().numpy())
    lg = np.vstack(logits); pr = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    return pd.DataFrame({"id": df_split.id.values, "distil_prob_0": pr[:, 0], "distil_prob_1": pr[:, 1],
                         "distil_logit_0": lg[:, 0], "distil_logit_1": lg[:, 1]})

ALL_PREDS, REPRO = {}, []
DATA = {}
for scale in SCALES:
    df = load_scale(scale); DATA[scale] = df
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    P = {}
    p, s, _ = fit_text(tr, te);                 P["tfidf_text"] = pred_frame(te, p, s)
    p, s = fit_meta(tr, te, METADATA_25);       P["meta_only"] = pred_frame(te, p, s)
    p, s = fit_early(tr, te, METADATA_25);      P["early_fusion"] = pred_frame(te, p, s)
    ds = distilbert_scores(scale, "test", te)
    assert ds.distil_prob_1.notna().all(), "DistilBERT test scores missing for some ids"
    P["distilbert_text"] = pred_frame(te, (ds.distil_prob_1.values >= 0.5).astype(int), ds.distil_prob_1.values)
    for k, v in P.items():
        v.to_csv(OUT / "predictions" / f"{scale}_{k}_test_predictions.csv", index=False)
        mf = f1_score(v.y_true, v.y_pred, average="macro")
        ref = V1_RESULTS.get((scale, k))
        REPRO.append(dict(scale=scale, config=k, macro_f1_rerun=mf, macro_f1_v1=ref,
                          abs_diff=None if ref is None else abs(mf - ref)))
    ALL_PREDS[scale] = P
repro = pd.DataFrame(REPRO); repro.to_csv(OUT / "results" / "E1_reproduction_check.csv", index=False)
print(repro.round(6))

# %% [markdown]
# ## E2 — Bootstrap CIs and paired differences (all configurations, every scale)

# %%
ci_all, diff_all = [], []
for scale, P in ALL_PREDS.items():
    ci, dd = bootstrap(P)
    ci_all.append(ci.assign(scale=scale)); diff_all.append(dd.assign(scale=scale))
ci_all = pd.concat(ci_all); diff_all = pd.concat(diff_all)
ci_all.to_csv(OUT / "results" / "E2_bootstrap_ci_all.csv", index=False)
diff_all.to_csv(OUT / "results" / "E2_paired_bootstrap_differences.csv", index=False)
print(diff_all[diff_all.metric.isin(["macro_f1", "pos_recall", "wr10"])].round(3).to_string())

# %% [markdown]
# ## E3 — Separate scale from test-set change: nested learning curve on ONE fixed test set
# Train on nested subsamples of the 100k training split; always test on the 100k test split.
# Repeat over 5 subsample seeds. (TF-IDF only; DistilBERT optional block at the end.)

# %%
df = DATA["Enron_100000"]
tr_full, te100 = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
rows = []
for seed in range(5):
    perm = np.random.default_rng(seed).permutation(len(tr_full))
    for n in [8000, 20000, 40000, 80000]:
        sub = tr_full.iloc[perm[:n]]
        p, s, _ = fit_text(sub, te100)
        m = metrics(te100.y_bin.values, p, s); m.update(seed=seed, n_train=n); rows.append(m)
lc = pd.DataFrame(rows); lc.to_csv(OUT / "results" / "E3_nested_learning_curve_tfidf.csv", index=False)
print(lc.groupby("n_train")[["macro_f1", "pos_recall", "pos_f1", "pr_auc", "wr10"]].agg(["mean", "std"]).round(3))

# Split-seed variability at each original scale (re-split the same messages with 5 seeds)
rows = []
for scale in SCALES:
    d = DATA[scale]
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(len(d)); n = len(d)
        tr, te = d.iloc[idx[:int(.8 * n)]], d.iloc[idx[int(.9 * n):]]
        p, s, _ = fit_text(tr, te)
        m = metrics(te.y_bin.values, p, s); m.update(scale=scale, split_seed=seed); rows.append(m)
ss = pd.DataFrame(rows); ss.to_csv(OUT / "results" / "E3_split_seed_variability_tfidf.csv", index=False)
print(ss.groupby("scale")[["macro_f1", "pos_recall", "pos_f1"]].agg(["mean", "std", "min", "max"]).round(3))

# %% [markdown]
# ## E4 — Clean stacking: (a) out-of-fold TF-IDF scores, (b) DistilBERT stacker fitted on the
# validation split only. Feature lists are fixed here, before looking at any test number.

# %%
def oof_text_scores(tr, te, stop_words):
    vec, clf = tfidf_lr(stop_words)
    Xtr = vec.fit_transform(tr.text); Xte = vec.transform(te.text)
    oof = cross_val_predict(clf, Xtr, tr.y_bin, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    clf.fit(Xtr, tr.y_bin)
    return oof, clf.predict_proba(Xte)[:, 1]

rows, stackP = [], {}
for scale in SCALES:
    d = DATA[scale]; tr, va, te = (d[d.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    for sw in ["english", None]:
        oof, te_s = oof_text_scores(tr, te, sw)
        # text-only stacker (same LR, score only) = the correct reference for "does metadata add anything"
        for name, feats in [("score_only", []), ("score+safe21", SAFE_21), ("score+meta25", METADATA_25)]:
            Xtr = tr[feats].copy(); Xte = te[feats].copy()
            Xtr["text_score"] = oof; Xte["text_score"] = te_s
            pipe = meta_pipe().fit(Xtr, tr.y_bin)
            pr, sc = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
            key = f"tfidf_oof_{'sw' if sw else 'nosw'}_{name}"
            stackP.setdefault(scale, {})[key] = pred_frame(te, pr, sc)
            m = metrics(te.y_bin.values, pr, sc); m.update(scale=scale, config=key); rows.append(m)
    # DistilBERT stacker trained on VALIDATION scores (held out from DistilBERT training)
    dv, dt = distilbert_scores(scale, "val", va), distilbert_scores(scale, "test", te)
    sf = ["distil_prob_0", "distil_prob_1", "distil_logit_0", "distil_logit_1"]
    for name, feats in [("score_only", []), ("score+meta25", METADATA_25)]:
        Xv = pd.concat([va[feats].reset_index(drop=True), dv[sf].reset_index(drop=True)], axis=1)
        Xt = pd.concat([te[feats].reset_index(drop=True), dt[sf].reset_index(drop=True)], axis=1)
        pipe = meta_pipe().fit(Xv, va.y_bin)
        pr, sc = pipe.predict(Xt), pipe.predict_proba(Xt)[:, 1]
        key = f"distilbert_valstack_{name}"
        stackP[scale][key] = pred_frame(te, pr, sc)
        m = metrics(te.y_bin.values, pr, sc); m.update(scale=scale, config=key); rows.append(m)
stack = pd.DataFrame(rows); stack.to_csv(OUT / "results" / "E4_clean_stacking.csv", index=False)
print(stack[["scale", "config", "macro_f1", "pos_recall", "pos_f1", "pr_auc", "wr10"]].round(3).to_string())
for scale, P in stackP.items():
    for k, v in P.items():
        v.to_csv(OUT / "predictions" / f"{scale}_{k}_test_predictions.csv", index=False)
    _, dd = bootstrap(P)
    dd.assign(scale=scale).to_csv(OUT / "results" / f"E4_paired_differences_{scale}.csv", index=False)

# %% [markdown]
# ## E5 — Duplicate audit and grouped re-split
# ids are SHA-1 of the raw file (headers included), so copies of one email in different folders
# get different ids. We hash normalised text (subject+body, lower-case, whitespace-collapsed)
# and a normalised subject (Re:/Fw: stripped) as a thread proxy.

# %%
def norm_text(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()
def norm_subject(s):
    s = re.sub(r"^\s*((re|fw|fwd)\s*:\s*)+", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

dup_rows = []
for scale in SCALES:
    d = DATA[scale].copy()
    d["h"] = d.text.map(lambda t: hashlib.sha1(norm_text(t).encode()).hexdigest())
    trh = set(d[d.split == "train"].h)
    te = d[d.split == "test"].copy(); te["dup_in_train"] = te.h.isin(trh)
    P = ALL_PREDS[scale]
    for k, v in P.items():
        v2 = v.merge(te[["id", "dup_in_train"]], on="id")
        for subset, vv in [("all", v2), ("dedup", v2[~v2.dup_in_train])]:
            m = metrics(vv.y_true.values, vv.y_pred.values, vv.score.values)
            m.update(scale=scale, config=k, subset=subset); dup_rows.append(m)
    print(scale, "test msgs whose normalised text occurs in train:", int(te.dup_in_train.sum()),
          f"({te.dup_in_train.mean():.1%}); among positives:", int(te[te.y_bin == 1].dup_in_train.sum()))
pd.DataFrame(dup_rows).to_csv(OUT / "results" / "E5_duplicate_audit.csv", index=False)

# Grouped re-split at 100k: no normalised text and no normalised subject shared across train/test
d = DATA["Enron_100000"].copy()
if "subject" not in d.columns:
    d["subject"] = d.text.str.split(" ", n=1).str[0]   # fallback; prefer the real subject column
d["h"] = d.text.map(lambda t: hashlib.sha1(norm_text(t).encode()).hexdigest())
d["subj"] = d.subject.map(norm_subject)
# union of the two group keys (connected components) so that neither key crosses the split
parent = {}
def f(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for h, s in zip(d.h, d.subj):
    a, b = f("h:" + h), f("s:" + s if s else "h:" + h)
    parent[a] = b
d["group"] = [f("h:" + h) for h in d.h]
rows = []
for seed in range(5):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tri, tei = next(gss.split(d, d.y_bin, d.group))
    p, s, _ = fit_text(d.iloc[tri], d.iloc[tei])
    m = metrics(d.iloc[tei].y_bin.values, p, s); m.update(seed=seed, n_groups=d.group.nunique()); rows.append(m)
g = pd.DataFrame(rows); g.to_csv(OUT / "results" / "E5_grouped_split_tfidf_100k.csv", index=False)
print(g[["macro_f1", "pos_recall", "pos_f1", "pr_auc", "wr10"]].agg(["mean", "std"]).round(3))

# %% [markdown]
# ## E6 — Rule sensitivity: substring vs word-boundary matching
# v2 uses `phrase in text`, so "now" fires inside "know"/"snow" and "sign in" inside "design in".

# %%
def rule_score(text, wb):
    t = str(text).lower(); W = RULES["weights"]
    def hit(L):
        return any((re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t) if wb else (k in t)) for k in L)
    core, urg, act = hit(RULES["core_indicators"]), hit(RULES["urgency_indicators"]), hit(RULES["action_indicators"])
    url = re.search(RULES["regex_indicators"]["url"], t) is not None
    sh = re.search(RULES["regex_indicators"]["url_shorteners"], t) is not None
    s = W["core"]*core + W["urgency"]*urg + W["action"]*act + W["url"]*url + W["url_shortener"]*sh
    if (not core) and hit(RULES["benign_context_indicators"]) and s <= 2:
        return 0
    return s

rows = []
for scale in SCALES:
    d = DATA[scale].copy()
    d["y_v2"] = (d.text.map(lambda t: rule_score(t, False)) >= 3).astype(int)
    d["y_v3"] = (d.text.map(lambda t: rule_score(t, True)) >= 3).astype(int)
    agree_v1 = (d.y_v2 == d.y_bin).mean()   # should be 1.0: confirms the rule re-implementation
    tr, te = d[d.split == "train"], d[d.split == "test"]
    p, s, _ = fit_text(tr.assign(y_bin=tr.y_v3), te.assign(y_bin=te.y_v3))
    m = metrics(te.y_v3.values, p, s)
    m.update(scale=scale, reimpl_agreement=agree_v1, pos_v2=int(d.y_v2.sum()), pos_v3=int(d.y_v3.sum()),
             lost=int(((d.y_v2 == 1) & (d.y_v3 == 0)).sum()), gained=int(((d.y_v2 == 0) & (d.y_v3 == 1)).sum()))
    rows.append(m)
r6 = pd.DataFrame(rows); r6.to_csv(OUT / "results" / "E6_rule_word_boundary_sensitivity.csv", index=False)
print(r6[["scale", "reimpl_agreement", "pos_v2", "pos_v3", "lost", "gained", "macro_f1", "pos_recall"]])

# %% [markdown]
# ## E7 — Circularity probe: mask every rule phrase / URL in the TEST text and re-score
# If recall collapses, the model has learned the rule phrases, not a broader notion of
# suspiciousness. Report recall among weak positives at the same threshold and WR@10.

# %%
PHRASES = sorted(RULES["core_indicators"] + RULES["urgency_indicators"] + RULES["action_indicators"], key=len, reverse=True)
PAT = re.compile("|".join([re.escape(p) for p in PHRASES] + [RULES["regex_indicators"]["url"] + r"\S*"]), flags=re.I)
def mask(t):
    return PAT.sub(" ", str(t))

rows = []
for scale in SCALES:
    d = DATA[scale]; tr, te = d[d.split == "train"], d[d.split == "test"]
    _, _, (vec, clf) = fit_text(tr, te)
    for cond, txt in [("original", te.text), ("rule_phrases_masked", te.text.map(mask))]:
        sc = clf.predict_proba(vec.transform(txt))[:, 1]; pr = (sc >= .5).astype(int)
        m = metrics(te.y_bin.values, pr, sc); m.update(scale=scale, condition=cond); rows.append(m)
r7 = pd.DataFrame(rows); r7.to_csv(OUT / "results" / "E7_rule_masking_probe_tfidf.csv", index=False)
print(r7[["scale", "condition", "pos_recall", "pr_auc", "wr10"]].round(3))
# (Optional) same probe for DistilBERT: call distilbert inference on te.assign(text=te.text.map(mask)).

# %% [markdown]
# ## E8 — DistilBERT truncation: is the label-completing trigger inside the 256-token window?

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def completing_char(t):
    t = str(t).lower(); W = {"core": 2, "urgency": 1, "action": 1, "url": 1, "url_shorteners": 1}; first = {}
    for g in ["core", "urgency", "action"]:
        pos = [t.find(k) for k in RULES[f"{g}_indicators"] if k in t]
        if pos: first[g] = min(pos)
    for g in ["url", "url_shorteners"]:
        m = re.search(RULES["regex_indicators"][g], t)
        if m: first[g] = m.start()
    s = 0
    for g, i in sorted(first.items(), key=lambda x: x[1]):
        s += W[g]
        if s >= 3: return i
    return None

rows = []
for scale in SCALES:
    d = DATA[scale]; te = d[(d.split == "test") & (d.y_bin == 1)].copy()
    te["cpos"] = te.text.map(completing_char)
    te["tok_before"] = [len(tok(str(t)[:c].lower(), add_special_tokens=False)["input_ids"]) if c is not None else np.nan
                        for t, c in zip(te.text, te.cpos)]
    te["inside_window"] = te.tok_before < 254          # 256 minus [CLS]/[SEP]
    db = ALL_PREDS[scale]["distilbert_text"].set_index("id"); tf = ALL_PREDS[scale]["tfidf_text"].set_index("id")
    te["distil_hit"] = te.id.map(db.y_pred); te["tfidf_hit"] = te.id.map(tf.y_pred)
    for inside, g in te.groupby("inside_window"):
        rows.append(dict(scale=scale, trigger_inside_256=inside, n=len(g),
                         distilbert_recall=g.distil_hit.mean(), tfidf_recall=g.tfidf_hit.mean()))
r8 = pd.DataFrame(rows); r8.to_csv(OUT / "results" / "E8_truncation_analysis.csv", index=False)
print(r8.round(3))

# %% [markdown]
# ## Manifest for frozen_release_v2

# %%
import sklearn, platform
manifest = dict(release="frozen_release_v2", created_from="frozen_release_v1 (read-only)", seed=SEED,
                n_bootstrap=N_BOOT, sklearn=sklearn.__version__, python=platform.python_version(),
                experiments=sorted(p.name for p in (OUT / "results").glob("*.csv")),
                predictions=sorted(p.name for p in (OUT / "predictions").glob("*.csv")),
                notes="Feature lists (SAFE_21, METADATA_25) fixed in code before any v2 test evaluation.")
json.dump(manifest, open(OUT / "manifests" / "release_v2_manifest.json", "w"), indent=2)
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## OPTIONAL (GPU, hours): DistilBERT with 3 training seeds at 100k, and a nested DistilBERT
# learning curve on the fixed 100k test set. Re-use the v1 WeightedTrainer settings
# (2 epochs, lr 2e-5, batch 16, wd 0.01, fp16, max_len 256) and save test predictions for each run
# to OUT/predictions so that E2's paired bootstrap can include them.
