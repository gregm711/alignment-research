# Right answer, wrong route: testing an internal monitor on a looped language model

*A technical research note. Draft for review, 2026-11-22.*

An overseer usually sees a model's answer, not how it got there. We built cases where a
language model gives the **same correct answer to the same question** in two runs, but in
one of them the answer depends on the wrong piece of evidence. Nothing in the output
distinguishes the two. Then we asked whether a measurement taken inside the model does.

- On matched pairs, one attention head ranked the wrong-route run above the right-route run
  in **139 of 139** pairs. The model's output confidence did so in 70.
- When we changed how the mistakes were caused, the same head still ranked them well (AUC
  0.86 to 0.89, against 0.40 to 0.47 for confidence), but its alarm threshold went off on
  **62% of correct runs**.
- We could not get the model to make this kind of mistake without intervening on it, so
  whether the monitor catches mistakes that arise unprompted is untested.

Everything here is one invented task on one model. Code, per-trial data, three independent
reviews and the full record of what we got wrong are in this repository.

## Setup

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is an open-weights
model that runs 12 of its 24 layers three times per token. Each pass restarts from the same
state; the only thing a later pass receives from an earlier one is a set of learned
"messages". [Goodfire](https://www.goodfire.com/research/interpreting-inkling-2-mini)
mapped how information flows through these loops on the day the model was released. We
intervene on the messages and ask what follows.

The task is a two-step lookup over small invented tables, so nothing can come from memory:

```
Fact sheet.
Colors of animals: the ram is brown, the cow is black, the dog is gold, ...
Fruits for colors: red goes with the orange, black goes with the apple, gold goes with the date, ...
So the fruit that goes with the dog is the
```

The model answers these correctly almost every time (168 of 168 with rows in random order).

## 1. Transplanted messages make the model read a different place

Run one prompt and save the messages going into its final pass. Run a second prompt and
substitute them. The second prompt's answer changes, in a specific way: the model answers
with whatever *its own* prompt has at the **token position** where the first prompt's answer
sat.

To pin that down we made three explanations predict different fruits before running
anything, by giving donor and recipient unrelated tables of different lengths:

| The model reads… | pairs it predicted, of 115 |
|---|---|
| the same token position | **109** |
| the same row counted from the end | 2 |
| the same row counted from the front | 1 |
| (the donor's answer, copied) | 2 |

In one pair the recipient's own answer was `date` and the donor's was `orange`; the three
rules predicted `fig`, `apple` and `olive`. The model said `fig`.
([plan committed before the run](results/discriminate-plan.json), [results](results/discriminate.json))

We do not know why it is a token position or where in the messages this lives. Three of our
guesses about the mechanism failed.

## 2. That gives a way to build mistakes you cannot see from outside

Put two rows in the second table that hold the same fruit. Transplant messages that point
the final pass at the *other* row. The model gives the correct fruit, having read the wrong
row. To establish which row it read without looking inside, repeat the run on a twin prompt
with one fruit changed: if the answer follows the change, that row was the one in use.

We kept only trials where the twin confirmed the route. That gives hundreds of runs with a
correct answer and a wrong route, paired with runs on the same prompt, with the same
answer, by the right route. Output confidence cannot tell them apart (AUC 0.49 to 0.51).

## 3. One attention head can

We measured, for each attention head in the final pass, how much of its attention to the
second table lands on the row the question actually requires. We chose one head using only
*visible* mistakes (transplants that produce a wrong answer, which an overseer could label),
froze it and its threshold, and applied it to the hidden ones on new tables.

| New tables | hidden mistakes / right-route | attention head | all heads averaged | output confidence |
|---|---|---|---|---|
| same wording | 83 / 86 | **0.999** | 0.94 | 0.49 |
| other wording | 88 / 94 | **1.000** | 0.97 | 0.51 |

At the frozen threshold it flagged every hidden mistake, with false alarms on 12 to 13% of
right-route runs. An independent recount of the saved trials found the head ranked the
hidden mistake higher in all 139 matched pairs.
([results](results/oversight-attention.json), [review](../notes/oversight-readout-review.md))

A first attempt, using a linear probe on the final token's internal state, found nothing
(0.48 to 0.53). Where the model looks carried the signal; that probe did not find it.

## 4. Change how the mistake arises, and the alarm stops being usable

The result above has an obvious weakness: we caused the mistakes by moving the final pass's
attention, and then read its attention. So we caused them differently: a transplant one
pass earlier, from an arbitrary prompt, after which the final pass runs on messages the
model computes itself.

| | hidden mistakes | right-route | attention head AUC | confidence AUC |
|---|---|---|---|---|
| donor draw a | 29 | 83 | **0.89** [0.82, 0.95] | 0.47 [0.36, 0.60] |
| donor draw b | 26 | 79 | **0.86** [0.76, 0.94] | 0.40 [0.28, 0.54] |

The head still ranks hidden mistakes above right-route runs. But the threshold fixed
earlier now fires on 101 of 162 right-route runs, while catching 54 of 55 mistakes. With no
intervention at all it fires on 12 to 14% of correct runs.
([results](results/causes.json))

This is the part we think is most useful to anyone evaluating a monitor. Judged by AUC, the
monitor transferred. Judged by how often it would interrupt a correct run, it did not. A
report that gave only the first number would have described a working monitor.
The general point, that a ranking score and a thresholded alarm measure different things,
is made across several monitoring benchmarks in
[From Probe Scores to Alarm Policies](https://arxiv.org/abs/2610.04575); this is one more
instance of it.

## 5. We could not get the model to make these mistakes unprompted

The case that matters is a mistake nobody induced. We tried three ways to get one without
touching the model's internals:

| Approach | confidently right for the wrong reason |
|---|---|
| mention a second, distracting animal in the question | at most 1 in 72 |
| longer tables and a third lookup step | 0 to 1; the model stays reliable, or fails openly with a blank |
| set an everyday association (red, apple) against the table | 0 in 116 |

On this task the model either does the lookup or fails where anyone can see. So the monitor
is untested on unprompted mistakes, and this task cannot test it.

## What we got wrong along the way

About half of our written predictions were wrong, and they are all still in the
[experiment log](README.md) next to the results. Two early claims were withdrawn after
checking: that stopping the model after one pass reveals its intermediate step (it was an
artefact of how we worded our prompts), and that arithmetic survives noise in the messages
(we had scored the wrong token). Reviewers found a readout frozen at a layer that could not
have seen the intervention, a leaky cross-validation, and a shortcut in our own tables that
explained our first lookup results as well as our explanation did.

## Limits

- One invented task, one model, one kind of mistake.
- Every mistake was induced by an intervention on the model.
- The monitor needs to be told which evidence the model *should* use. Here that is trivial;
  in most real settings it is the hard part.
- The head is a signal that tracks the route in these settings. We did not intervene on the
  head, so we have not shown it is part of the mechanism.
- Thresholds were calibrated using some trials an overseer could not have labelled.

## Related work

Goodfire's study of [information flow in Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini)
and Thinking Machines' [release notes](https://thinkingmachines.ai/news/inkling-2-mini/) on
loop count and monitorability. Wang & Reid on the
[Jacobian lens in looped transformers](https://arxiv.org/abs/2609.01924). Using attention
to detect answers not grounded in the provided context goes back at least to
[Lookback Lens](https://arxiv.org/abs/2407.07071). What we add is a controlled comparison in
which the answer is held fixed and only the route differs, and a test of what carries over
when the cause of the mistake changes.

## Reproducing

Each experiment is one script with the command in its docstring; they need a Modal account
and one 80GB GPU. The whole project cost about $14 of GPU time ([ledger](SPEND.md)). The
cheapest results to re-run are the token-position test (`discriminate.py`, about two
minutes) and the matched comparison (`oversight2.py`, about fifteen).

## How this was made

The experiments were designed, run and written up by an AI assistant working with the
author over two days, with predictions committed to the repository before each run. Three
reviews by a separate model, working from the saved data, corrected the work at several
points; they are in [`../notes/`](../notes/).
