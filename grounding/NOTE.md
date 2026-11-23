# Right answer, passage ignored: can a monitor trained on visible mistakes catch hidden ones?

*A technical research note. Draft for review, 2026-11-23.*

Give a language model a passage and a question. If the passage and the model's memory
agree, the answer is correct whether or not the model used the passage. An overseer reading
the output cannot tell which happened. We call a correct answer that would have stayed the
same had the passage said something else **hidden ungrounded**, and we asked whether a
measurement inside the model can flag it.

- **A dataset problem.** On NQ-Swap, a standard knowledge-conflict benchmark, 55 to 62% of
  the answers we scored as "from memory, against the passage" had part or all of the
  original answer still in the passage. The earlier version of this project drew wrong
  conclusions from those labels.
- **A monitor that carries over.** A classifier over attention, trained only on mistakes
  visible in the output and then frozen, scored 0.79 to 0.83 AUC on hidden ungrounded
  answers on newly built clean items, in three models.
- **A lead we cannot fully attribute.** It beat output confidence by 0.18 AUC pooled over
  three models, in a comparison specified in advance. A check we had not planned showed that
  part of this is the classifier recognising date questions, which models answer from
  memory more often. Within one kind of question the lead is +0.06 with an interval that
  includes zero.
- **Not an alarm.** At its frozen threshold the classifier caught 30 of 85 hidden cases.

Three small open models, one family of short-answer questions. Code, per-run data, every
prediction (most of the last set failed) and two outside recounts are in this repository.

## Setup

Each question is asked three ways: with no passage, with its original passage, and with up
to three passages in which the answer has been replaced by a different one. No model is
modified.

| Term | Meaning |
|---|---|
| visible ungrounded | given a substituted passage, the model gives the original answer. The output shows the mistake. |
| hidden ungrounded | correct with the original passage, and gives the original answer on at least half of the substituted ones |
| grounded | correct with the original passage, and follows every substituted one |

Monitors are fitted on visible cases from one half of the questions, with thresholds set at
10% false alarms, and then frozen. They are tested on hidden cases from other questions,
using only the run with the original passage, where every answer is correct. Features are
read at the position where the answer begins.

- **Attention classifier:** logistic regression over every head's share of attention on the
  passage against the question, after [Lookback Lens](https://arxiv.org/abs/2407.07071).
- **Single heads:** the one head whose attention to the answer's location (or to the whole
  passage) best separates visible cases.
- **Output confidence:** probability of the top first token, counted as suspicious when low.
- **Output-plus-length baseline:** logistic regression on three confidence numbers and
  prompt length, fitted the same way. It sees no internals.

Models: Phi-4-mini-instruct, Qwen2.5-7B-Instruct, granite-3.3-2b-instruct.

## 1. Half of the "memory" answers on NQ-Swap could be read from the passage

[NQ-Swap](https://arxiv.org/abs/2109.05052) replaces the answer entity in a Natural
Questions passage. The paper states that all spans of the answer are replaced. In the copy
we used (`pminervini/NQ-Swap`), whole spans mostly are. Shorter later mentions are not: the
passage says "Zhu Di" where it said "Selena Gomez", and two sentences on still says "Gomez".

| share of runs with part of the original answer still in the passage | Phi-4-mini | Qwen2.5-7B | granite-3.3-2b |
|---|---|---|---|
| runs where the model gave the original answer | 122/214 (57%) | 106/171 (62%) | 112/205 (55%) |
| runs where the model followed the passage | 410/2218 (18%) | 462/2297 (20%) | 391/1989 (20%) |
| original-answer runs, looser rule (whole answer, or its last word) | 54% | 60% | 52% |
| original-answer runs where every word of the model's answer is in the passage | 39% | 48% | 25% |

Leftover mentions are three times as common where the model gave the original answer as
where it followed the passage, so this is not chance word overlap. In 25 to 48% of those
runs the answer could be read outright. In a further 14 to 30% a fragment was present and
the model completed the name.

How much this inflates a published memory-reliance rate depends on the scorer. Ours accepts
a lone surname. Exact-match scoring would be affected less by the first kind and equally by
the second. A limited search (seven queries, the dataset's discussion page, the original
repository, four papers) found no report that counts this. `audit.py` drops the affected
runs; the rule is twelve lines.

**What the bad labels did to us.** Before the audit we had reported that output confidence
was at chance on hidden cases (0.49 to 0.57) and that the best attention head collapsed
from 0.88 to 0.58. On cleaned labels confidence is 0.63 to 0.68 and the collapse is smaller
and fails on one model. An outside reviewer found the leak, not us.

## 2. On cleaned NQ-Swap, attention leads confidence by about 0.11

Cleaning halved the hidden cases (31, 19 and 29 test cases per model). Each monitor's own
interval overlapped the others'. The comparison that fits the data is paired: both monitors
score the same questions, and the three models answer the same questions, so we resampled
answer clusters once per draw across all models. The plan was committed before running.

| classifier minus… (AUC, 95% interval) | Phi-4-mini | Qwen2.5-7B | granite-3.3-2b | pooled |
|---|---|---|---|---|
| top probability | +0.15 [+0.05, +0.25] | +0.07 [-0.14, +0.29] | +0.13 [-0.00, +0.25] | +0.11 [+0.03, +0.20] |
| output-plus-length baseline | +0.17 [+0.08, +0.27] | +0.08 [-0.09, +0.23] | +0.13 [+0.00, +0.25] | +0.13 [+0.05, +0.20] |
| top probability, questions the model already knows | +0.07 | -0.03 | +0.14 | +0.06 [-0.08, +0.19] |

The last row was the open question. A classifier might detect "the model knows this
answer", which goes with ignoring the passage but is a different thing.

## 3. On new clean items the frozen classifier carries over, and known-ness is ruled out

NQ-Swap has 1,667 distinct questions and we had used 1,462. So we built new items from the
Natural Questions training set: who, when and where questions, with a substitution that
replaces every alias and each word of a name, kept only if no word of any gold answer
survives. We ran only questions each model answers correctly with no passage. Nothing was
refitted.

| | Phi-4-mini | Qwen2.5-7B | granite-3.3-2b |
|---|---|---|---|
| hidden ungrounded vs grounded | 32 vs 403 | 16 vs 465 | 37 vs 393 |
| attention classifier | 0.80 [0.69, 0.90] | 0.83 [0.71, 0.91] | 0.79 [0.71, 0.86] |
| output top probability | 0.73 [0.62, 0.83] | 0.54 [0.37, 0.71] | 0.60 [0.50, 0.70] |
| output-plus-length baseline | 0.67 [0.54, 0.80] | 0.73 [0.61, 0.84] | 0.59 [0.49, 0.69] |
| classifier at frozen threshold: caught, false alarms | 14/32, 24/403 | 5/16, 7/465 | 11/37, 14/393 |

Pooled, the classifier leads top probability by +0.18 [+0.08, +0.28] and the second
baseline by +0.14 [+0.06, +0.21]. By the reading rule written beforehand this counts as
support, and since every question here is known to the model, known-ness does not explain
it. The threshold also held: 3.6% false alarms on grounded runs, against 62% when we moved
a threshold across conditions in [the first project](../jlens-inkling/NOTE.md).

Hidden ungrounded answers were rare: 2 to 5% of answers to known questions.

## 4. Part of the lead is the kind of question

We had not planned for this. Hidden cases are concentrated in "when" questions. In Qwen, 10
of 16 hidden cases are "when" questions against 19 of 465 grounded ones, and "is this a
when question" alone scores 0.79.

The classifier does pick up the kind of question. Among grounded runs only, where every
answer followed the passage, it separates "when" from "who" questions at 0.72 (Phi-4-mini),
0.93 (Qwen, 19 "when" runs) and 0.90 (granite). The reviewer traced this to training: in
NQ-Swap, "when" questions were already over-represented among visible mistakes.

Comparing inside each kind of question and averaging:

| classifier minus…, within kind (post hoc) | Phi-4-mini | Qwen2.5-7B | granite-3.3-2b | pooled |
|---|---|---|---|---|
| top probability | +0.09 [-0.05, +0.23] | -0.03 [-0.32, +0.30] | +0.13 [+0.02, +0.25] | +0.06 [-0.06, +0.20] |
| output-plus-length baseline | +0.16 [+0.03, +0.29] | -0.17 [-0.53, +0.13] | +0.13 [+0.05, +0.23] | +0.04 [-0.08, +0.16] |

The lead holds in two models and is absent in Qwen, which has 16 hidden cases. Pooled, it
is unresolved. The reviewer reports the same two-of-three split within kind on cleaned
NQ-Swap.

Knowing which kinds of question a model tends to answer from memory is useful to an
overseer. But it is a property of the question, available without looking inside the
model, and it is not what we set out to measure.

## 5. Exploratory: one head does better within a kind than the classifier

This was not a planned comparison and we give it as an observation to test, not a result.
The single "passage head", chosen on NQ-Swap visible cases and frozen, scored 0.91, 0.71 and
0.80 overall on the new items. Inside each kind of question:

| frozen passage head, new items | Phi-4-mini | Qwen2.5-7B | granite-3.3-2b | pooled |
|---|---|---|---|---|
| AUC within who / when / where | 0.90 / 0.90 / 0.94 | 0.79 / 0.51 / 0.61 | 0.71 / 0.71 / 0.71 | |
| within kind, minus top probability | +0.24 [+0.12, +0.36] | +0.08 [-0.06, +0.26] | +0.13 [-0.02, +0.28] | +0.15 [+0.07, +0.23] |

One number has less room than a thousand weights to absorb the mix of question kinds in
its training data. Against this: we looked at several monitors and are reporting the one
that looks best, and the reviewer found its within-kind lead on cleaned NQ-Swap was only
+0.05 [-0.05, +0.14]. It needs a test on data chosen before looking.

## What we got wrong

- We built Phases 1 to 4 on labels that were about half wrong, and reported conclusions
  from them. The leak was found by an outside reviewer reading raw examples.
- Our explanation for the first result, that an attention head sees only what the model
  does in this run and so cannot see a counterfactual, did not survive cleaning.
- We estimated that NQ-Swap could supply three times the hidden cases. It could supply 10%
  more.
- All five predictions for the final experiment failed: fewer hidden cases than predicted,
  a higher classifier score, a larger lead, a strong Qwen result, and a threshold that held.
- The question-kind confound was not in the plan. We found it after the planned test passed.

## Limits

- Three small open models, short answers, questions from one source.
- 16 to 37 hidden test cases per model in the final experiment.
- "When" questions: a passage about an event often carries other dates from which the
  original can be inferred, and our leak rule cannot see that. The reviewer's reading of
  raw outputs found many that are plainly memory, but not all. Our substitution also
  mismatches date formats, which sent 85% of Qwen's "when" questions to the set-aside pile.
- Labels come from behaviour under substitution, at most three substitutions per question.
- The monitor needs the location of the passage, and the single evidence head needs the
  location of the answer. Here those are known.
- No intervention on any head. These are signals that track groundedness here, not shown
  to be part of how the model decides.

## Related work

Entity substitution to create knowledge conflicts: [Longpre et al.](https://arxiv.org/abs/2109.05052).
Critiques of word-level substitution as incoherent: [Xie et al.](https://arxiv.org/abs/2305.13300).
Attention ratios for detecting answers not grounded in context:
[Lookback Lens](https://arxiv.org/abs/2407.07071). Heads involved in knowledge conflicts,
studied on conflict cases: [Rewired or Gated?](https://arxiv.org/abs/2609.25602). Ranking
scores against thresholded alarms: [From Probe Scores to Alarm Policies](https://arxiv.org/abs/2610.04575).
What we add: the agreement case, where the output is correct and gives nothing away; a
count of leftover answer mentions in NQ-Swap; and a transfer test with the monitor frozen.

## Reproducing

`build.py` records answers, confidence and attention for one model (Modal, one L4 GPU).
`audit.py`, `monitor.py` and `pooled.py` run on a laptop from the saved runs. Commands are
in each file's docstring and the full log, with every plan and prediction, is in
[README.md](README.md). This project cost about $9.30 of GPU time ([ledger](SPEND.md)).

## How this was made

The experiments were designed, run and written up by an AI assistant working with the
author, with plans and predictions committed before each run. A separate model reviewed the
saved data three times; it found the label leak and the structure of the question-kind
confound. Its last recount is in [`../notes/grounding-review.md`](../notes/grounding-review.md).
