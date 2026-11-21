# What we found about Inkling 2 Mini's loop

One page. Details, raw numbers and caveats are in [`README.md`](README.md); data is in
[`results/`](results/). Everything here was produced on 2026-11-21 for about $6 of GPU time
(see [`SPEND.md`](SPEND.md)), and has not yet been independently reviewed.

## The model

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) runs 12 of its 24
layers three times per token. Each pass restarts from the same pre-loop state; the only
thing a later pass receives from an earlier one is a set of learned "messages". A sibling
checkpoint was trained to run anywhere from one to five passes.

## Four questions, four answers

**1. What crosses from one pass to the next?** Only the messages, by construction. They carry
what the previous pass worked out, not what the prompt already states. Transplanting another
question's messages into the last pass moved the answer to that question's answer in 33 of
42 two-hop cases, and in 0 of 120 arithmetic cases. Given a multiplication's messages, an
addition prompt sometimes multiplied its own numbers (13 of 112).

**2. At which pass is the answer decided?** In the released model, stopping after one pass
gives the first step of the reasoning: `France` on the way to `Paris`, or `7` for `7 + 5`.
One-hop facts are right after one pass (16/16); two-hop facts go 3/16, 11/16, 15/16 over
passes one to three. The sibling trained to stop early gets the same questions right in one
pass (14/16), so the spreading-out is a product of training, not a requirement of the task.

**3. Is the loop state robust to noise?** The restart state tolerates noise at one position
fairly well. The messages do not: modest noise in the messages into the last pass removed
nearly every factual answer and every refusal. Refusals fell when everything else fell, not
before.

**4. Must an edit be held across all passes to stick?** No. An edit in the last pass is what
counts (14 of 42 answers moved by a direction edit there, against 1 and 6 in passes one and
two, and 7 when held in all three). Earlier edits are re-derived away from the restart state.

## Why it might matter

Looped models move reasoning out of visible text and into hidden passes, which is bad for
anyone who monitors models by reading their reasoning. In this model the hidden intermediate
step can be read by running fewer passes, with no trained probe. That is a point in favour
of being able to oversee this design. It is untested on hard problems, and untested on a
model that has any reason to hide its reasoning.

## How much to trust this

- **Small, hand-written test sets.** 44 probes, 42 prompt pairs, 24 requests per set. The
  directions are clear; the exact numbers would move.
- **Lenses fitted on 3 prompts** where the library suggests about 100.
- **One prompt format** in the main experiments.
- **Several bugs were made and caught** during the work (two scoring errors with this
  model's tokenizer, three memory errors). The corrected runs are what is reported.
- **Stopping the released model early is outside its training.** We cannot fully separate
  "pass one has only done step one" from "pass one's state decodes as step one".
- **One of our own measures turned out to be uninformative** (the likelihood score for
  replies to harmful requests); see question 3 in the README.

## Follow-up on public data

Running: the early-exit finding re-tested on TwoHopFact (400 items) and Anthropic's lens
evaluation sets (148 items), with failure criteria written in advance. Result to be added.

## For an independent reviewer

Places a mistake would most likely hide, in the order we would check them:

1. **Scoring by first token** (`sweep.py`, `patch.py`, `earlyexit.py`). This tokenizer writes
   some words as a space token followed by the word. We handle that by appending the space
   to the prompt. Is that handling right, and does it bias one condition over another?
2. **The pass counter in the hooks** (`fit.py` `PassView`, `clamp.py`, `noise.py`). Edits and
   readouts are assigned to a pass by counting calls. A miscount would silently mislabel
   every per-pass result.
3. **Changing `loop_num_passes` at run time** (`sweep.py`). Is that all it takes to run
   fewer passes, on both checkpoints?
4. **The claim that "both passes" equals "pass 3" in the message swap** (`patch.py`).
5. **Whether the probes are leading.** The two-hop and one-hop probes share answers by
   design. Are there cues in the wording that make the comparison unfair?
6. **Anything in the README stated more strongly than its table supports.**

Reproducing a result needs a Modal account and one 80GB GPU; each script's docstring gives
the command. The cheapest to re-run is the pass sweep (about 5 minutes).
