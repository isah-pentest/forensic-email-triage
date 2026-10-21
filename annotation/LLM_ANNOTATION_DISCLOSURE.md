# LLM-assisted relevance annotation: disclosure

The two completed sheets in `llm_annotations/` were produced by **large language models, not by human annotators**:

- OpenAI GPT-6
- Google Gemini 3 Flash

They are a **preliminary relevance signal**. They are **not** human validation and **not** ground truth.

## What the models saw

| Item | Detail |
|---|---|
| Input | The blinded sheet `annotation_sheet_blinded.csv`: 174 messages in random order, showing subject and text only. |
| Withheld | Weak labels, model predictions, scores, and the stratum of each message. These are in `annotation_key.csv`. |
| Output fields | `review_worthy` (0/1), `phish_like` (0/1/u), `message_type`, `confidence`, `notes` |

## Details still to be recorded by the authors

| Item | Value |
|---|---|
| Model → sheet mapping (`llm_sheet1` / `llm_sheet2`) | TODO |
| Prompt(s) used, and whether `ANNOTATION_GUIDE.md` was given verbatim | TODO |
| Access dates | TODO |
| Decoding settings (temperature, etc.) | TODO |

## Results

Produced by `analyze_annotations.py` and summarised in `llm_annotation_summary.csv`.

**Agreement between the two models**
- They agreed on 168 of 174 messages.
- Cohen's κ = 0.554 (bootstrap 95% CI 0.138–0.848).

**Share judged review-worthy**

The sample is stratified, so the estimates below are re-weighted to the 100k test set. The strata come from the superseded late-fusion run: TP 167, FP 193, FN 24, TN 9,616.

A message counts under **either** if at least one model marked it review-worthy, and under **both** only if the two models agreed.

| Group | Either model | Both models |
|---|---|---|
| Weak-label positives | 8.0% (95% CI 2.3–15.6) | 4.0% (0.0–9.8) |
| Messages the model flagged | 8.0% (3.1–13.9) | 2.9% (0.0–6.8) |

**What these annotations cannot tell us.** 0 of the 50 TN messages were marked review-worthy. The one-sided 97.5% upper bound on that rate is 7.1%, so the sample cannot bound what the rules miss.

**Disagreements.** The six messages the models disagreed on are listed in `adjudication_sheet.csv`. They should be resolved by human reviewers, not by another model.
