# Reproducing the paper's results

There are three levels of reproduction. Level 1 needs nothing outside this repository.

## Level 1 — the paper's figures and tables (seconds, CPU)

**In Colab:** open `notebooks/reproduce_figures_tables.ipynb` and run the cells.

**Locally:**

```bash
pip install -r requirements.txt
python -m analysis.run_all            # or: make figures
```

Each figure and table has its own script in `analysis/`. The scripts:
- read the saved results: `frozen_release_v2/`, the v1 `results/`, and `splits/`;
- write to `outputs/`, which is not tracked in git;
- produce figures as PDF and 300-dpi PNG, and tables as CSV plus LaTeX rows (`*_rows.tex`) for the paper.

**Pointing at another copy of the saved results.** Use `--v2 <folder>`, `make figures V2=<folder>`, or the environment variable `FET_V2`, e.g. to read the copy on Google Drive.

Bootstraps use `numpy.random.default_rng(42)`, so results are identical on every run.

**LLM annotation results.** `make annotations` recomputes them and writes `annotation/llm_annotation_summary.csv`.

**Independent check.** `make verify-v2` recomputes every metric from the 36 v2 prediction files and compares them with `frozen_release_v2/results/E2_bootstrap_ci_all.csv`.

## Level 2 — the v2 experiments from the public Enron corpus (minutes, CPU)

1. Download the CMU Enron email dataset (maildir, May 2015 release; the original listing had 517,401 files).
2. Rebuild the exact 10k/50k/100k subsets from the released split IDs:
   ```bash
   python code/rebuild_subsets_from_maildir.py --maildir /path/to/maildir --out /path/to/project
   ```
   - The script hashes every file the way the notebook did, recovers each released ID, and recomputes labels and the 25 metadata features. The code is ported from the notebook.
   - It stops if any label distribution differs from `splits/*/split_meta.yaml`, or if any 100k rule score differs from `results/rule_triggers_100k.csv`.
3. Run `FET_ROOT=/path/to/project FET_REPO=. python code/limitations_upgrade.py`. This re-creates `frozen_release_v2/` and checks the v1 values in `E1_reproduction_check.csv`.

**DistilBERT scores (E1, E4, E8).** The script needs the per-message DistilBERT scores (`<scale>_distil_scores_{train,val,test}.csv`). Put them in `frozen_release_v2/distilbert_scores/`. If they are absent, the script needs the fine-tuned models.

**TF-IDF numbers.** These reproduce to within 0.009 macro F1 of v1, depending on the scikit-learn version. v2 used 1.6.1.

## Level 3 — retraining DistilBERT (GPU) and the TREC07 test

These are **not exactly reproducible** from the released material:

- **DistilBERT.** The notebook cell defining `make_training_args` was never saved. Retraining with the documented hyper-parameters gives comparable, not identical, models. The released predictions and scores define the paper's DistilBERT results.
- **TREC07.** The notebook reads `trec07_external_eval.csv` but contains no code that builds it from the Kaggle preprocessed TREC07 corpus. The released `results/cross_domain_trec07_predictions.csv` defines the paper's TREC07 results.
