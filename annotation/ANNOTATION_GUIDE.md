# Annotation guide: forensic email triage review sample

## Purpose

You will judge whether each email deserves an investigator's attention. These judgements will tell us how often the paper's rule-based labels agree with human judgement, which is the paper's largest open limitation.

## Setup

**Annotators.** Two people annotate independently. Do not discuss individual messages until both of you have finished.

**Files.**
- `annotation_sheet_annotator1.xlsx` (or `.csv`) goes to annotator 1.
- `annotation_sheet_annotator2.xlsx` goes to annotator 2.
- **Do not open** `KEY_do_not_share_with_annotators.csv`. It holds the rule labels and model outputs. Seeing them would bias your judgements.

**Sample.** The sheet has 174 Enron messages in random order. The sample was drawn so that rule-positive, rule-negative, model-flagged and model-missed messages are all represented. Because of this mix, the share of "suspicious-looking" messages in the sheet is much higher than in the full corpus.

**Time.** Expect roughly 3–4 hours per annotator. Most messages are under 2,000 characters.

## Fields to fill for each row

| Field | Values | Definition |
|---|---|---|
| `review_worthy` | **1** or **0** (required) | **1** if an investigator triaging the mailbox for account compromise, credential requests, phishing or social engineering, or suspicious instructions to act urgently should read this message. **0** if it is routine business, a newsletter, personal mail, or an automated notice with no such concern. |
| `phish_like` | 1 / 0 / u | Does the message *itself* resemble a phishing or social-engineering attempt? Use **u** for unsure. |
| `message_type` | one code | **business**: routine work communication. **newsletter**: digest, bulk mailing or marketing. **system**: automated account or IT notice. **personal**: personal correspondence. **forward**: mainly forwarded content. **other** |
| `confidence` | 1 / 2 / 3 | 1 = guess, 2 = fairly sure, 3 = certain |
| `notes` | free text | Optional. Especially useful when `review_worthy` = 1, to say why. |

## Rules

1. **Judge only what you can read in the message.** Ignore what you know about Enron's history.
2. **Keywords alone are not enough.** Words such as "urgent", "click here", "password" or "log in" do not make a message review-worthy on their own. Judge the request and its context.
3. **Legitimate IT and account notices** (password resets, access granted) count as review-worthy only if something about them looks irregular. Otherwise mark them 0 and type `system`.
4. **Long forwarded threads:** judge the message as a whole. If any part warrants review, mark it 1.
5. **When torn**, choose your best answer and put `confidence` = 1. Do not leave `review_worthy` blank.

## After annotation

1. Run `python analyze_annotations.py annotation_sheet_annotator1.csv annotation_sheet_annotator2.csv KEY_do_not_share_with_annotators.csv`. Before running, export each completed `.xlsx` to `.csv` if you annotated in Excel.
2. The script reports:
   - Cohen's κ and raw agreement between the two annotators.
   - The list of disagreements. Resolve these together in an adjudication meeting, then enter the agreed value in **both** sheets and re-run.
3. After adjudication, the script also reports four re-weighted estimates, each with a 95% bootstrap interval:
   - weak-label precision;
   - model precision against humans;
   - the share of review-worthy messages the rules miss;
   - the share the model misses.
4. Report κ, the number of adjudicated rows and these four estimates in the paper. **Note:** the estimate of what the rules miss rests on only 50 of the 9,616 true-negative messages, so its interval will be wide. State that in the paper.
