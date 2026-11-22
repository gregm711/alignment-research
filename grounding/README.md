# Grounding: can internal access catch an answer that ignores its evidence?

**Status: Phase 0 (the gate) is running.**

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

**Models.** Qwen3.5-2B and Qwen3.5-4B (Apache 2.0), small enough to be cheap and, we
expect, more inclined to answer from memory than a large model.

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
