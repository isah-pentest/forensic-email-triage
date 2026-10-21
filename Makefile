# Rebuild the paper's figures and tables from the saved results.
# Point V2 at another copy of the results, e.g. make figures V2=/content/drive/MyDrive/forensic_email_project/frozen_release_v2
PY ?= python3
V2 ?= frozen_release_v2

figures:
	$(PY) -m analysis.run_all --v2 $(V2)

annotations:
	$(PY) annotation/analyze_annotations.py annotation/llm_annotations/llm_sheet1_completed.csv \
	  annotation/llm_annotations/llm_sheet2_completed.csv annotation/annotation_key.csv

verify-v2:
	$(PY) code/verify_v2.py frozen_release_v2 . /tmp/v2_metrics_check.csv

.PHONY: figures annotations verify-v2
