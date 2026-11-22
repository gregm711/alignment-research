# Grounding: can internal access catch an answer that ignores its evidence?

**Status: Phase 0 passed on one model. Phase 1 is running.**

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

**What this will not show.** Labels still rest on string matching. A question labelled
hidden ungrounded is one whose answer did not change under these particular substitutions.
