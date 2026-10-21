# Weakly Supervised Email Prioritization for Forensic Review

This repository holds the reproducibility artifacts for the paper *Weakly Supervised Email Prioritization for Forensic Review: Scale, Metadata Fusion, and Evidence-Bounded Evaluation* (Mohammed, Yahaya, Aigboduwa, Pham; Sam Houston State University).

## What the labels are

Positive labels come from a deterministic phishing-style rule set, `rules/phishing_rules_v2.yaml`. The rules are applied to the subject and body of each message. A message is positive when its rule score is 3 or more.

The models read the same text the rules were applied to. The results therefore measure how well a model recovers a documented heuristic. They do **not** measure detection of malicious, criminal or forensically relevant messages.

## Releases

| Folder | Contents | Status |
|---|---|---|
| top-level `results/`, `rules/`, `splits/`, `manifests/` | `frozen_release_v1`, the original experiment outputs | Frozen and unchanged. Some v1 claims are superseded; see below. |
| `frozen_release_v2/` | Re-runs with per-message test predictions for **every** configuration, bootstrap CIs, paired comparisons, and validity checks | Used by the paper |
| `annotation/` | Blinded 174-message relevance sample, annotation guide, key, and the **LLM-assisted** annotations | Human validation still pending |
| `code/` | `limitations_upgrade.py` (Colab, produces v2) plus independent verification scripts | |
| `analysis/` | One script per figure and table (`fig1_learning_curve.py`, `fig2_review_budget.py`, `table1_splits.py` … `table4_review_budget.py`), reading the saved results; outputs go to `outputs/` (not tracked) | Colab: `notebooks/reproduce_figures_tables.ipynb` |
| `notebooks/` | The original notebook with the credential cell removed | See the notebook caveat below |

**Notebook caveat.** Cells are stored roughly newest-first. Two definition cells were never saved: `make_training_args`, and the cell defining `train_text_prob`.

Raw Enron/TREC07 data and trained model checkpoints are not redistributed.

## Main results (v2, 100k test set, 191 weak positives)

| Model | Macro F1 [95% CI] | Positive recall | PR-AUC | WR@10 |
|---|---|---|---|---|
| TF-IDF + LR | 0.752 [0.727, 0.775] | 0.890 | 0.626 | 0.979 |
| DistilBERT | 0.847 [0.817, 0.875] | 0.628 | 0.741 | 0.948 |
| Metadata-only LR | 0.559 [0.545, 0.573] | 0.806 | 0.227 | 0.780 |
| TF-IDF + metadata (early fusion) | 0.766 [0.740, 0.790] | 0.911 | 0.664 | 0.974 |

All values are computed from `frozen_release_v2/predictions/`. Paired bootstrap comparisons are in `frozen_release_v2/results/E2_*`. Re-running reproduces the v1 DistilBERT and metadata-only values exactly. TF-IDF values reproduce to within 0.009 macro F1 (scikit-learn 1.6.1).

## Key findings

- **Scale.** On a fixed test set, TF-IDF improves from 8k to 80k training messages:
  - macro F1 0.731 → 0.752;
  - recall 0.78 → 0.89.
  
  Differences between the independently sampled 10k, 50k and 100k subsets are within split-to-split variation.
- **DistilBERT vs TF-IDF.** DistilBERT has higher macro F1 and PR-AUC but lower recall.
  - For 142 of the 191 positives, the phrase that completes the label lies beyond DistilBERT's 256-token input.
  - Its recall on these is 0.563, against 0.816 when the phrase is inside the window.
- **Metadata.** With held-out stacking (out-of-fold or validation scores), metadata does **not** improve ranking. For TF-IDF, PR-AUC falls from 0.654 to 0.596.
  - The v1 late-fusion result (macro F1 0.798) is an operating-point effect.
  - Flagging the same 360 messages using the text score alone gives 0.799.
- **WR@k is recall at a fixed review budget,** an established TAR measure.
  - It is the share of *weak-label positives* found in the top k% of the ranking. It is **not** the share of irrelevant mail filtered out.
  - An oracle reaches 1.0 at k = 1.9% (the prevalence).
- **Duplicates.** 27.1% of 100k test messages also appear in training once text is normalised (31 of 191 positives).
  - Message IDs hash the raw file, headers included, so copies of one email in different folders get different IDs.
- **TREC07 spam/ham.** Ranking is at chance (ROC-AUC 0.485), and spam is the majority class (66.6%). This is a test under a different label definition, not a domain-generalisation result.
- **Relevance (LLM-assisted only).** Two LLMs (GPT-6, Gemini 3 Flash) judged only 4–8% of weak-label positives review-worthy.
  - Agreement between them: κ = 0.55.
  - See `annotation/LLM_ANNOTATION_DISCLOSURE.md`. This is **not** human validation.

## Corrections to v1 claims

The v1 README and earlier manuscript drafts stated the following. They are superseded:

- *"Reviewing only 10% of emails filters out ~96% of irrelevant content."* Incorrect. WR@10 is the share of weak-label positives found in the top 10%.
- *"Metadata improves prioritization."* Not supported under held-out stacking.
- *Bootstrap CIs for all configurations.* v1 computed a CI for one configuration only. v2 computes them for all.
- *"Human review simulation supports qualitative validation."* The v1 sample had no annotations. v2 adds LLM-assisted annotations only.
- The DistilBERT early-fusion results and the TF-IDF late-fusion results at 10k and 50k that appeared in an earlier draft have no supporting code or output.

## Reproducing the paper's figures and tables

**In Colab.** Open `notebooks/reproduce_figures_tables.ipynb`. It clones the repository, then runs one cell per figure and table. Figures display inline, and tables display as data frames. Set `USE_DRIVE = True` to read the saved results from Google Drive instead of the repository copy.

**From the command line.**

```bash
python -m analysis.run_all                     # everything, into outputs/
python -m analysis.fig2_review_budget          # a single figure
python -m analysis.table2_main_results --v2 /path/to/frozen_release_v2
```

**Scripts and sources.**

| Script | Output | Reads |
|---|---|---|
| `fig1_learning_curve.py` | Figure 1 (PDF + PNG) | `E3_nested_learning_curve_tfidf.csv` |
| `fig2_review_budget.py` | Figure 2 (PDF + PNG) | 100k text-model prediction files (bootstrap, seed 42) |
| `table1_splits.py` | Table I | `splits/*/split_meta.yaml` |
| `table2_main_results.py` | Table II | `E2_bootstrap_ci_all.csv`, prediction files |
| `table3_late_fusion.py` | Table III | `E4_clean_stacking.csv`, 100k predictions, superseded run |
| `table4_review_budget.py` | Table IV | 100k predictions, `E2` |
| `table5_trec07.py` | Table V | TREC07 results |
| `derived_statistics.py` | In-text statistics not stored in a saved file (equal-volume macro F1, two paired differences, truncation counts) | 100k predictions, superseded run, `E8` |

**Table output.** Each table is written as CSV and as LaTeX rows (`*_rows.tex`) for the paper. The first lines of each `*_rows.tex` file name its source files.

**Integrity.** `run_all` writes `outputs/MANIFEST.sha256`: the SHA-256 of every saved-result file read and every figure and table written. Outputs are byte-identical across runs. `python -m analysis.verify_manifest` re-hashes everything and fails on any mismatch.

`REPRODUCE.md` covers rebuilding the Enron subsets and re-running the v2 experiments.

## Contact

Isah Alhaji Mohammed, Sam Houston State University.
