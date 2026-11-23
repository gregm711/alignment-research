# Independent reviews of the grounding project

Four reviews by a separate model, working from the saved data in `grounding/results/` on
CPU. This file records what each found and what we did about it. Where we re-ran a
reviewer's number ourselves it is marked "reproduced"; the rest are the reviewer's.

## Review 1 (after Phases 1 to 3, 2026-11-22): the label leak

- **Finding.** NQ-Swap leaves later mentions of the answer in the substituted passage
  (surnames, second mentions, other gold aliases). Many runs scored as "memory" could have
  been read from the passage. 29 of 100 hidden cases were surname-only.
- **Other issues.** Near-duplicate questions on both sides of the train/test split; prompt
  length as a possible confound (length alone 0.605 on hidden cases; classifier 0.698 after
  removing length); first-token measurement misaligned for models that answer in sentences.
- **What we did.** Wrote `audit.py` (drop leaky substituted runs, re-split by answer),
  committed the rule and predictions before the two Phase 4 models' data existed, reported
  the pre-registered numbers first and the cleaned ones second, and withdrew the claim that
  confidence is at chance on hidden cases.

## Review 2 (after the audit, 2026-11-22): how to compare

- **Reproduced our cleaned tables exactly.**
- **Finding.** Leftover mentions are about three times as common in "memory" runs as in
  passage-following runs (57/62/55% against 18/20/20%), so the strict rule is not mostly
  removing chance overlaps. Reproduced.
- **Finding.** granite has twice the rate of filler-word answer starts among hidden cases
  (7/29 against 32/270). Reproduced; dropping them raised granite's difference.
- **Finding.** Some indirect leaks survive cleaning (an answer inferable from a name or an
  adjective in the question). Not quantified by us.
- **Method.** Compare monitors by paired difference, not by whether separate intervals
  overlap; resample answer clusters once per draw across models, because the models share
  questions. Adopted as `pooled.py`.

## Review 3 (after Phase 5, 2026-11-23): recount and the question-kind confound

- **Recount.** No question or first-gold-answer overlap between NQ-Swap and the Phase 5
  items in any model; every Phase 5 row is known unaided. Classifier AUC 0.800 / 0.826 /
  0.786; 30 of 85 caught with 45 of 1,261 false alarms; pooled lead +0.181 [+0.079, +0.284]
  and +0.138 [+0.060, +0.213]; within kind +0.064 [-0.062, +0.199] and +0.043 [-0.084,
  +0.164]. All match ours.
- **Prompt length** no longer separates hidden from grounded in Phase 5 (0.489 to 0.528).
- **The classifier learned the kind of question.** Among grounded runs only, it separates
  "when" from "who" at 0.715 (Phi-4-mini) and 0.898 (granite); top probability does so at
  0.820 and 0.737. Reproduced; we also get 0.931 for Qwen on its 19 "when" runs. The
  reviewer traced it to training data: "when" questions are 44.1% of visible memory runs
  against 17.3% of passage-following runs in granite, 26.5% against 19.1% in Phi-4-mini.
- **Same split on NQ-Swap.** Within kind on cleaned NQ-Swap the classifier's lead is +0.158
  (Phi-4-mini), +0.163 (granite), -0.063 (Qwen). Not re-run by us.
- **Why Qwen has so few grounded "when" questions.** 161 of its 190 "when" questions were
  set aside, 150 with at least one answer matching neither date. Causes: our substitution
  mismatches date formats (a bare year replaced by a full date), and the model splices a
  month from the passage onto a remembered year. Many labelled "when" cases are plainly
  memory.
- **Exploratory.** The single frozen passage head scores 0.906 / 0.705 / 0.804 on Phase 5,
  and within who / when / where 0.90 / 0.90 / 0.94 in Phi-4-mini. Its within-kind pooled
  lead over top probability is +0.151 [+0.067, +0.233]. Reproduced. On cleaned NQ-Swap the
  same quantity was +0.047 [-0.048, +0.138] (reviewer's number). Not pre-registered; one of
  several monitors looked at.
- **Recommendation.** Stop spending, write the note, keep the passage head labelled
  exploratory.

## Review 4 (after the note draft, 2026-11-23): what explains the results

- **Recount.** Refitted all three classifiers from the saved tensors; 0.800 / 0.826 / 0.786
  and 30 of 85 with 45 of 1,261 match. Recomputed the Inkling head from individual trials
  (0.9993 and 1.000; 54 of 55 caught with 101 of 162 false alarms).
- **Whole dataset.** 1,077 of 4,746 NQ-Swap substituted passages (22.7%) keep a word of an
  original answer under our rule. Reproduced exactly (`grounding/checks.py`).
- **Framing.** Leaky runs are ambiguous as evidence of memory use; the filter cannot say
  the model read the surviving word. Adopted in the note.
- **Prior report.** The reviewer cites the same surname problem in another swap dataset
  (GRAB-RAG, Appendix C). We searched for that paper and did not find it, so the note
  attributes the pointer to the reviewer and does not claim priority.
- **Cleaning and re-splitting, separately.** For Phi-4-mini, confidence on hidden cases:
  0.498 original, 0.518 cleaning only, 0.614 re-split only, 0.629 both. For granite,
  cleaning alone moved 0.571 to 0.702. Reproduced exactly; Qwen is 0.485, 0.520, 0.594,
  0.676.
- **Question-word baseline.** A classifier on first-token confidence and the question word,
  fitted on the same visible runs and frozen: 0.722 / 0.896 / 0.746 on Phase 5, so the
  attention classifier's pooled lead falls from +0.181 to +0.016 [-0.037, +0.070]. Our own
  version (L2 0.01, standardised confidence) gives 0.737 / 0.869 / 0.725 and +0.027
  [-0.029, +0.083]. Same conclusion.
- **Wording.** "Hidden" establishes resistance to changing the passage, not how the model
  reached its answer on the original passage. Adopted.
- **Recommendation.** Lead with the benchmark audit, then the split comparison and the
  question-word baseline; keep Inkling separate; keep the single head as a hypothesis.

## Not yet checked by anyone

- The GRAB-RAG reference above.

- Whether the `pminervini/NQ-Swap` copy matches data regenerated from the original code.
- Whether any knowledge-conflict paper we did not open already counts leftover mentions.
- The passage head on data chosen before looking.
