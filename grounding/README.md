# Grounding: can internal access catch an answer that ignores its evidence?

**Status: Phases 0 to 3 done on one model. The frozen single-head monitors failed the pre-registered bar. Awaiting independent review.**

## The question

The [Inkling work](../jlens-inkling/NOTE.md) showed that an attention readout can separate
"right answer by the right route" from "right answer by the wrong route" when we induce the
wrong route ourselves. It could not be tested on mistakes the model makes unprompted,
because on that task the model did not make any.

This project asks the question that work left open, on a task where such mistakes are
known to occur: answering from memory when a document is supplied. If a model is asked a
question it already "knows" the answer to and is handed a passage, its answer can be
correct either because it read the passage or because it would have said that anyway. The
second kind is invisible in the output and is a problem the moment the passage and the
model's memory disagree.

Nothing is done to the model's internals anywhere in this project.

## Definitions

Each question comes with an original passage (which contains the true answer) and a
substituted passage (identical except that the answer has been replaced).

- **Follows the passage:** given the substituted passage, the model gives the substituted answer.
- **Visible ungrounded answer:** given the substituted passage, the model gives the original
  answer anyway. An overseer can see this; it contradicts the passage.
- **Hidden ungrounded answer:** given the *original* passage, the model answers correctly,
  and we know from the substituted passage that it would have said the same thing whatever
  the passage said. Correct, and not based on the evidence.
- **Grounded answer:** correct with the original passage, and follows the substituted one.

## Phases

Each phase has a gate. A phase that fails its gate stops the project or changes the plan;
the result is recorded either way. Predictions are written here before each phase runs.

| Phase | What | Gate to continue | Budget |
|---|---|---|---|
| 0 | Do unprompted hidden ungrounded answers occur often enough to study? | At least 10% of correct answers, and at least 100 obtainable | $1 |
| 1 | Fix the model and prompt; build labelled train and test sets split by question | Labels stable under a second prompt wording | $2 |
| 2 | Build monitors using visible ungrounded answers only; freeze them | A monitor separates visible ungrounded answers on held-out questions | $2 |
| 3 | Test the frozen monitors on hidden ungrounded answers | Pre-registered bar, written before the run | $2 |
| 4 | Change conditions (passage format, another dataset, another model); report ranking and false alarms separately | – | $4 |
| 5 | Independent review, then write-up | – | – |

Assumed cap for the whole project: $11. We ask before going past it. Spend is in
[`SPEND.md`](SPEND.md).

## Lessons carried over from the Inkling work

- Use other people's questions. Our own probes were leading.
- Pilot every scoring rule on a handful of items and read the raw outputs.
- Split by question, never by trial, and select only on training data.
- Report false alarms at a frozen threshold, not just a ranking score.
- Have each phase's results independently recounted before building on them.

## Phase 0: the gate

### Plan and predictions (written before running)

**Data.** [NQ-Swap](https://huggingface.co/datasets/pminervini/NQ-Swap) (MIT licence):
real questions from Natural Questions with the original passage and a passage in which the
answer has been swapped for another. 300 distinct questions, sampled with a fixed seed from
those with short answers and passages under 250 words.

**Models.** Qwen3-4B-Instruct-2507 (Apache 2.0) and Phi-4-mini-instruct (MIT): small enough
to be cheap and, we expect, more inclined to answer from memory than a large model. Both are
ordinary transformers with standard attention.

*Changed before any results:* the plan first named Qwen3.5-2B and Qwen3.5-4B. A pilot was
stopped during its first model because Qwen3.5 uses linear-attention layers, which have no
attention map for a later monitor to read and ran very slowly on fallback code. No outputs
from it were looked at.

**Measurement.** Three prompts per question, short greedy answers: the question alone; with
the original passage; with the substituted passage. Answers are scored by whether they
contain the expected string after normalising case and punctuation. No internals are read.

**Predictions.**

1. With the substituted passage, at least one model gives the original answer on 10% or
   more of questions.
2. Among questions answered correctly with the original passage, at least 10% are hidden
   ungrounded answers on at least one model.

**Gate.** Prediction 2 holding, with enough questions in the dataset to collect at least 100
hidden ungrounded answers. If it fails on both models we try one prompt that puts less
weight on the passage, and if that fails we stop.

**What this will not show.** String matching miscounts paraphrased answers. "Would have said
the same whatever the passage said" is established with one substituted passage, not many.

### Result

Raw data, every question and answer: [`results/gate.json`](results/gate.json). 600 questions.

**A second scoring fix, made after reading the full outputs.** Word overlap credited "from
memory" when the gold answer shared words with the question (the question asks about
Guantanamo Bay, the gold answer is "Guantanamo Bay Naval Base", and any answer repeating
the question matched). Gold words that already appear in the question are now ignored, and
answers saying the passage does not contain the information are counted separately as
abstentions. Both scorings are shown; the stricter one is what we use from here on.

| | Qwen3-4B | Phi-4-mini |
|---|---|---|
| correct unaided | 122 | 109 |
| correct with the original passage | 475 | 521 |
| of those, with the substituted passage: follows the passage | 366 | 383 |
| gives the original answer anyway (**hidden ungrounded**) | 17 (3.6%) | **50 (9.6%)** |
| abstains | 57 | 7 |
| other | 35 | 81 |
| hidden ungrounded among questions it also knows unaided | 9 of 122 (7%) | **25 of 109 (23%)** |
| (first scoring, for comparison: hidden ungrounded) | 30 | 54 |

**Against the predictions as first written:** both narrowly missed. With the substituted
passage Phi-4-mini gives the original answer on 8.5% of all questions (predicted 10%), and
9.6% of its correct answers are hidden ungrounded (predicted 10%).

**Against the gate as revised after the pilot:** passed on Phi-4-mini, failed on Qwen3-4B.
Among questions Phi-4-mini knows unaided, 23% of correct answers do not depend on the
passage, and the dataset's 1,349 eligible questions should yield about 110 hidden
ungrounded answers. That count is close to the 100 we asked for, not comfortably above it.

**What we see.** The two models differ in kind. Faced with a passage whose answer has been
swapped, Qwen3-4B usually follows it or says the passage does not answer the question;
Phi-4-mini more often just gives the answer it remembers. Many substitutions in this
dataset are implausible (a person's name where a year belongs), so "ignoring the passage"
is sometimes a reasonable reaction, and what we are labelling is "the answer did not depend
on the passage", not "the model made an error of judgement".

**Decision.** Continue to Phase 1 with Phi-4-mini only.

## Phase 1: build the labelled set

### Plan and predictions (written before running)

**Model and prompt.** Phi-4-mini-instruct, the prompt used in Phase 0.

**Stronger labels.** The dataset has several substituted passages for most questions. Each
question is run with its original passage and with up to three different substitutions.

- *Hidden ungrounded:* correct with the original passage, and gives the original answer on
  at least half of the substitutions (and on at least one).
- *Grounded:* correct with the original passage, and follows every substitution.
- Everything else (abstentions, mixed, unscorable) is set aside and counted.

**Split.** Questions are assigned to train or test by a hash of the question text, half each.

**Recorded for later phases, in the same runs.** For the original-passage run and each
substituted-passage run: how much of every attention head's attention, at the point where
the answer begins, falls on the answer's location in the passage, on the passage as a
whole, and on the question; and the model's confidence in its first answer token. Nothing
is modified. Phases 2 and 3 can then be done offline at no GPU cost.

**Stability check (the gate).** A second prompt wording on 300 questions. The gate is that
at least 80% of questions labelled grounded or hidden ungrounded under both wordings get
the same label.

**Predictions.**

1. At least 80 hidden ungrounded and 300 grounded questions.
2. The stability gate passes.

**A third scoring fix, from the 20-question pilot, before the full run.** Half a name
matched: "Clarence Williams", a different person named in the passage, was credited as the
remembered answer "Hank Williams". An answer now has to contain every word of the gold
answer beyond those in the question, or be a single word equal to its last word.

**What this will not show.** Labels still rest on string matching. A question labelled
hidden ungrounded is one whose answer did not change under these particular substitutions.

### Result (Phase 1)

[`results/phase1.json`](results/phase1.json) holds every question, answer and label. The
attention features (`results/phase1_features.pt`, 22 MB) are not in git; `build.py`
regenerates them in about 35 GPU minutes.

| | all | train | test |
|---|---|---|---|
| hidden ungrounded | 100 | 54 | 46 |
| grounded | 773 | 400 | 373 |
| set aside | 589 | 288 | 301 |

Prediction 1 held (at least 80 and 300). Prediction 2 held: of 145 questions labelled under
both wordings, 141 (97%) got the same label. 56 of the 100 hidden ungrounded questions are
ones the model answers correctly with no passage; the other 44 it does not recall unaided
yet still prefers over the substituted answer.

## Phases 2 and 3: monitors, calibrated on what an overseer can see

### Plan, predictions and bars (written while Phase 1 was still running, before any of its data existed)

Both phases are offline analyses of what Phase 1 records. `monitor.py` does both.

**What an overseer can label.** With a substituted passage, a model that gives the original
answer has visibly ignored the passage. Those runs, and runs that follow the passage, can be
labelled from the output alone. Everything below is fitted on those, on training questions
only. No hidden ungrounded answer is used to build or tune anything.

**Monitors.** Each gives a score where higher means "less grounded".

| Monitor | Uses | Needs to know where the evidence is? |
|---|---|---|
| evidence head | one attention head's attention on the answer's location in the passage (low attention scores high) | yes |
| passage head | one head's attention on the passage as a whole | no |
| all-heads average | mean over every head of attention on the answer's location, nothing selected | yes |
| attention classifier | logistic regression over every head's share of attention on the passage, in the style of Lookback Lens | no |
| output confidence | top-token probability, margin, entropy | no |

Heads are chosen by ranking score on the training questions' visible cases. Thresholds are
set for 10% false alarms on training runs that follow the passage. Then everything is frozen.

**Phase 2 gate.** On the *test* questions' substituted-passage runs, the frozen evidence
head separates visible ungrounded answers from passage-following ones with AUC above 0.8.

**Phase 3 test.** On the test questions' *original-passage* runs, where every answer is
correct: hidden ungrounded against grounded. Reported for every monitor: AUC with a
bootstrap interval over questions, and detection and false alarms at the frozen threshold.

**Bar for "internal access helps".** The best attention monitor's interval lies entirely
above the best output-confidence baseline's. We name the comparison now: evidence head
against top-token probability, and separately passage head against top-token probability.

**A control we commit to.** Hidden ungrounded answers come from questions the model already
knows, so a monitor might only be detecting "the model knows this". We also report hidden
ungrounded against grounded *among questions the model answers correctly with no passage*,
where both groups know the answer and differ only in whether they used the passage.

**Predictions.**

1. The Phase 2 gate passes.
2. The evidence head beats top-token probability on hidden ungrounded answers.
3. Output confidence is a stronger baseline here than on the lookup tables (well above
   chance), because a remembered answer is likely to be a confident one.
4. The frozen threshold's false-alarm rate on test will not match the 10% it was set for.
   The Inkling work showed thresholds are the part that fails to carry over.

**What this will not show.** One model, one dataset, one prompt. Labels by string matching.
Many substituted passages are implausible, so "did not depend on the passage" is established
against an implausible alternative.

### Result (Phases 2 and 3)

[`results/phase2_3.json`](results/phase2_3.json). Fitted on 109 visible ungrounded and
1,138 passage-following runs from training questions. Evidence head: layer 20, head 15.
Intervals are bootstraps over runs (Phase 2 has up to three runs per question, so its
intervals are somewhat too narrow; Phase 3 has one run per question).

| Monitor | Phase 2: visible, test questions (105 vs 1,080) | Phase 3: hidden, test questions (46 vs 373) | Control: known questions only (26 vs 68) |
|---|---|---|---|
| evidence head | **0.88** [0.85, 0.91] | 0.58 [0.49, 0.66] | 0.53 [0.40, 0.65] |
| passage head | 0.80 [0.75, 0.84] | 0.61 [0.52, 0.69] | 0.61 [0.49, 0.73] |
| all-heads average | 0.85 [0.82, 0.89] | 0.64 [0.55, 0.73] | 0.56 [0.44, 0.69] |
| attention classifier | 0.84 [0.79, 0.88] | **0.72** [0.64, 0.80] | 0.66 [0.53, 0.79] |
| output top probability | 0.77 [0.73, 0.81] | 0.50 [0.41, 0.59] | 0.49 [0.36, 0.60] |

At the frozen thresholds on Phase 3: the evidence head catches 9 of 46 hidden ungrounded
answers with 32 of 373 false alarms; the attention classifier catches 17 of 46 with 42 of 373.

**Against the predictions.**

1. Phase 2 gate: passed (0.88 against a bar of 0.8).
2. Evidence head beats top-token probability on hidden answers: **failed.** 0.58 against
   0.50, intervals overlap, and the head's interval includes chance. The passage head also
   fails the bar. Both named comparisons failed.
3. Confidence is a strong baseline here: **wrong.** It is at chance on hidden answers. It
   does work on visible ones (0.77).
4. The frozen threshold's false-alarm rate will drift: **wrong.** Set for 10%, it gave 9 to
   11% on test.

**What we see.** A head that separates visible ungrounded answers at 0.88 is close to
chance on hidden ones. The only monitor clearly above confidence on hidden answers is the
many-head classifier (0.72), which was not one of the two comparisons we named, and its
advantage is not established on the known-questions control (intervals overlap).

**An exploratory look at why (not pre-registered).** Median attention of the evidence head
on the answer's location, test questions:

| | original passage | substituted passage |
|---|---|---|
| grounded questions | 0.23 | 0.19 (followed the passage) |
| hidden ungrounded questions | 0.20 | 0.02 when it gave the remembered answer; 0.09 when it followed |

On the questions where the model will ignore a conflicting passage, it still looks at the
evidence when the passage agrees with it. The head reports what the model is doing in this
run. "Would have said the same whatever the passage said" is a fact about a different run,
and this head mostly does not carry it.

**What this does not show.** One model, one dataset, one prompt, 46 hidden test cases. It
does not show that no internal signal exists: the classifier's 0.72 suggests a weak one. It
is not comparable in kind to the Inkling result, where the hidden mistake was a different
computation in the same run (reading the other row); here the run itself may be the same
computation, and only the counterfactual differs.

## Phase 4: does the collapse from visible to hidden replicate on other models?

### Plan and predictions (written before running)

**Scope.** Other models on the same dataset. We looked for a second public dataset with
matched original and substituted passages and did not find one that loads; building our own
substitutions is deferred. So this phase tests "another model", not "another dataset".

**Choosing models, by behaviour only.** The Phase 0 gate (`gate.py`, 120 questions, shared
scorer) on six candidates: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3,
Phi-3.5-mini-instruct, SmolLM3-3B, granite-3.3-2b-instruct, Qwen3-8B. We take the two with
the highest rate of hidden ungrounded answers among correct answers, provided the rate is
at least 6%, preferring models from different families. No internals are looked at to choose.

**Then, unchanged.** `build.py` and `monitor.py` exactly as run on Phi-4-mini: same prompt,
labels, split, monitors, thresholds and analysis.

**Predictions, for each new model.**

1. The evidence head separates visible ungrounded answers on test questions (AUC above 0.8).
2. **The collapse replicates:** the evidence head's AUC on hidden ungrounded answers is at
   least 0.15 lower than on visible ones, and its interval on hidden lies entirely below its
   interval on visible.
3. Output confidence is within 0.40 to 0.60 on hidden answers.
4. The attention classifier is the best monitor on hidden answers and stays below 0.80.
5. The exploratory observation becomes a prediction: on hidden ungrounded questions, the
   evidence head's median attention on the answer in the original-passage run is at least
   70% of its median on grounded questions.

**What would change our mind.** If on a new model the frozen evidence head reaches 0.75 or
more on hidden answers, the Phi-4-mini result is specific to that model and the collapse
is not a general claim.

### Model choice (behaviour only; no internals looked at)

[`results/gate-candidates.json`](results/gate-candidates.json), 120 questions each.

| Candidate | hidden ungrounded among correct | answers of three words or fewer |
|---|---|---|
| SmolLM3-3B | 11 of 103 (10.7%) | 0% |
| Qwen2.5-7B-Instruct | 10 of 107 (9.3%) | 72% |
| granite-3.3-2b-instruct | 9 of 98 (9.2%) | 60% |
| Mistral-7B-Instruct-v0.3 | 9 of 100 (9.0%) | 36% |
| Phi-3.5-mini-instruct | 8 of 102 (7.8%) | 69% |
| Qwen3-8B | 6 of 108 (5.6%) | 77% |

Five of six clear the 6% bar, so answering from memory with a passage in hand is not
peculiar to Phi-4-mini.

**A departure from the selection rule, made before any features were examined.** By the rule
the two models are SmolLM3-3B and Qwen2.5-7B. A 12-question pilot showed SmolLM3 answers in
full sentences ("There are 163 episodes of..."). Our monitors read attention at the first
generated token, which for that model is not where the answer is produced, so the unchanged
pipeline would be measuring something else. We replace it with the next model from a
different family that answers tersely: granite-3.3-2b-instruct. Phase 4 models:
**Qwen2.5-7B-Instruct and granite-3.3-2b-instruct.** This also means every result here is
limited to models and prompts that give short answers.

## Label audit: the substituted passage often still contains the original answer

An independent reviewer, working from `results/phase1.json`, found that many answers we
scored as "from memory" were read from the passage. NQ-Swap replaces one mention of the
answer; a later mention ("Gomez" after "Selena Gomez") or another listed gold answer can
survive. That explains the 44 hidden ungrounded questions the model could not answer
unaided. The Phase 1 to 3 results above stand as run, and are superseded by the cleaned
analysis below where the two differ.

### Cleaning rule and predictions (written before the Phase 4 models' data exists)

`audit.py`, offline:

- Drop every substituted-passage run in which any word of any gold answer, beyond the
  question's own words, still appears in the substituted passage. Labels are recomputed
  from the runs that remain.
- Re-split train and test by a hash of the normalised first gold answer, so near-duplicate
  questions land on the same side.
- Report the share of answers that begin with a filler word, since the monitors read
  attention at the first generated token.

Then `monitor.py` unchanged on the cleaned set: re-fitted on cleaned training runs, frozen,
tested on cleaned test questions. For the two Phase 4 models the pre-registered numbers on
the original labels are reported first, verbatim.

The reviewer has already reported a partial cleaning of Phi-4-mini (evidence head 0.62,
attention classifier 0.78 on 22 hidden cases), so the Phi-4-mini cleaned numbers are not
blind. The predictions below are for all three models, and are blind for the two new ones.

1. The evidence head still collapses: at least 0.15 lower on hidden than on visible answers.
2. The attention classifier reaches at least 0.70 on cleaned hidden answers on at least two
   of the three models, with its interval above output confidence's.
3. On the known-questions control the classifier stays above 0.65 on at least two models.
4. Cleaning leaves fewer than 60 hidden ungrounded questions per model, so intervals will be
   wide. If fewer than 15 remain in a model's test split, that model's Phase 3 is reported
   as uninformative.
