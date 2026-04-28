# Scaling Weakly Supervised Forensic Email Triage

This repository contains the reproducibility artifacts for the paper:

> *Scaling Weakly Supervised Forensic Email Triage: Reproducible Weak-Label Evidence Modeling with Text and Metadata*

---

## Overview

This project investigates how weakly supervised machine learning models can be used to support **forensic email triage**, where large volumes of communication must be prioritized for human review.

The study evaluates:

- Text-only models (TF-IDF + Logistic Regression)
- Metadata-only models
- Early fusion (feature-level)
- Late fusion (score-level)
- Transformer-based models (DistilBERT)

Across multiple dataset scales:

- 10,000
- 50,000
- 100,000 emails

---

## ⚠️ Important Scope Note

This system uses **weak supervision** based on heuristic rules.

> The labels are NOT expert-validated forensic ground truth.

All results should be interpreted as:

- evaluation of **triage prioritization behavior**
- NOT detection of confirmed malicious or criminal activity

---

## Repository Structure
frozen_release_v1/
├── manifests/ # experiment metadata and freeze records
├── results/ # evaluation outputs and analysis tables
├── rules/ # weak labeling rules and schemas
├── splits/ # dataset partition definitions
├── models/ # (excluded from GitHub - large artifacts)
├── data/ # (excluded from GitHub - raw datasets) 


---

## Included Artifacts

### Results
- `full_upgrade_comparison_table.csv`
- `distilbert_metadata_late_fusion_results.csv`
- `tfidf_late_fusion_100k_results.csv`
- `bootstrap_ci_results.csv`
- `workload_reduction_100k.csv`
- `qualitative_error_analysis_summary.csv`

### Human Review Simulation
- `human_review_sample.csv`
- `human_review_annotation_template.csv`

### Rules
- `phishing_rules_v2.yaml`
- `metadata_schema_v1.json`
- `metadata_manifest_v1.json`

### Splits
- Fixed train/val/test partitions (seed = 42)

### Manifests
- `final_manuscript_manifest.json`
- `freeze_note.json`

---

## Key Findings

- Scaling dataset size provides the largest performance gains
- Metadata improves prioritization but introduces leakage risks
- Safe metadata reduces performance but improves validity
- Cross-domain evaluation reveals strong domain dependence
- Workload Reduction (WR@k) is a meaningful triage metric

---

## Reproducibility

This repository represents a **frozen experimental release**.

- Random seed: 42
- Fixed dataset splits
- No modification to artifacts after freeze
- All reported results correspond to saved CSV outputs

---

## Workload Reduction (WR@k)

WR@k measures how much data can be safely ignored while retaining high recall.

Example: WR@10 = 0.963 


Interpretation:

> Reviewing only 10% of emails filters out ~96% of irrelevant content.

---

## Human Review Simulation

A stratified sample was created:

| Group | Count |
|------|------|
| TP | 50 |
| FP | 50 |
| TN | 50 |
| FN | 24 |
| Total | 174 |

This dataset supports qualitative validation of triage behavior.

---

## Cross-Domain Evaluation

The model was evaluated on a preprocessed version of the TREC07 spam corpus.

Key result:  Macro F1 = 0.312
ROC-AUC ≈ 0.485 


Interpretation:

- Performance degradation reflects **domain + task mismatch**
- Weak-label models do not directly transfer across label regimes

---

## Metadata and Leakage

Two configurations were evaluated:

- Full metadata (includes overlap-sensitive features)
- Safe metadata (excludes potential leakage sources)

Result: Macro F1 (safe metadata) ≈ 0.797 


Interpretation:

- Metadata contributes to ranking behavior
- Improper feature construction can inflate performance

---

## Limitations

- Weak labels do not represent forensic ground truth
- No expert validation of individual messages
- Cross-domain results reflect task mismatch
- Metadata features may still contain indirect correlations

---

## Not Included

To preserve privacy and repository size:

- Raw Enron dataset
- TREC07 dataset
- Trained model checkpoints
- Kaggle credentials

---

## Citation

If you use this work, please cite:  ""


---

## Contact

For questions or collaboration:

Isah Mohammed  
PhD, Digital and Cyber Forensic Science  
Sam Houston State University
