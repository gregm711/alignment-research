# Research directions

Ideas for small, cheap experiments on open-weight models that could produce something worth
publishing. Ordered by how strongly we'd recommend them today.

## 1. Jacobian lens on a looped transformer (in progress: [`../jlens-inkling/`](../jlens-inkling/))

Inkling 2 Mini runs its middle layers three times. Looped models do more reasoning internally
and less in visible text, which works against chain-of-thought monitoring. If a cheap linear
readout still recovers intermediate reasoning from inside the loop, that is good news for
oversight of this design; if it doesn't, that is worth saying too.

The lens has already been applied to other looped models (see the experiment's README for the
prior art). Our angle is this model's restart-and-message design and its variable-loop checkpoint.

## 1b. The next bet after the Inkling work: belief against a document

The lookup tables took the oversight question as far as they can: the model does not make
confidently-right-for-the-wrong-reason mistakes there unprompted. A reviewer's proposal,
which we think is right: a document-grounded question where the model has a strong default
answer that can be measured before it sees the document. Give it a passage that sometimes
agrees and sometimes conflicts. First establish, by editing the evidence, that the
unmodified model gives enough correct-looking answers that do not depend on the passage.
Only then evaluate a monitor. See [`../jlens-inkling/NOTE.md`](../jlens-inkling/NOTE.md).

## 2. A side-effects scorecard for abliterated models

Abliterated and "uncensored" models have millions of downloads on Hugging Face and almost no
public evaluation (see [`huggingface-landscape.md`](huggingface-landscape.md)). Removing the
refusal direction is assumed to change only refusal. Does it also change honesty, sycophancy,
willingness to admit uncertainty, or deception under pressure? The benchmarks exist and the
models are free. Cheap, no heavy fitting, and squarely on the measuring side.

## 3. Looking inside abliterated models

Compare a base model and its abliterated version on the same prompt with a lens. Does the
abliterated model still internally register "this is harmful" and answer anyway, or is the
recognition itself gone? The answer bears on how safety training should be done.

## 4. Fine-tune auditing

A tool that takes a community model and its base and reports what changed, including things
the uploader didn't advertise. Emergent misalignment, subliminal learning and homeopathic
poisoning all show fine-tuning changes more than intended. Introspection adapters and
AuditBench are the lab-side versions.

## 5. Expert routing as a free monitor

**Update, 2026-11-21: this is crowded.** RouterInterp (arXiv 2610.11775) and The Expert Strikes
Back (arXiv 2604.02178) already study routing and expert-level interpretation, and a secondhand
list names several more. Only worth doing tied to something unstudied, such as how routing
drifts across loop passes.

In mixture-of-experts models the router's choices are discrete and cheap to log. Do routing
patterns predict refusals, deception or "I'm being tested" behaviour? Can pinning or removing
experts steer the model? Needs no extra training.

## 6. A public reasoning-faithfulness leaderboard

Small models make it affordable to run thousands of interventions: edit the written reasoning
and see whether the answer moves. Capability and price are ranked publicly; how far you can
trust each model's stated reasoning is not.

## Later: an automated research loop

Autoresearch-style hill-climbing needs a single trustworthy metric, experiments that take
minutes, and a low cost per run. Direction 1 has none of those yet. Once a lens is fitted, the
lens library's evaluation sets give a scalar (how often the lens recovers a hidden intermediate
answer) and applying a lens is forward-only and cheap, so a loop over readout variants becomes
reasonable. The known risk is metric gaming: about 2.4% of runs cheated in Anthropic's
automated-researcher study.

## What we won't do

Write jailbreaks, or build tooling whose main use is stripping safeguards from a model.
Measuring what stripping them does is in scope.

## Scan of 2026-11-22: what changed, and what to do next

Depth: arXiv listings on eleven topics, abstracts only for the twelve closest papers, and
the Looped Lens Challenge page via a summary. No paper below was read in full.

**Things that reduce the novelty of what we have**

- [From Probe Scores to Alarm Policies](https://arxiv.org/abs/2610.04575) (Oct 2026) says
  ranking scores and thresholded alarms are "different estimands" and tests this across four
  monitoring benchmarks. Our "the ranking carried over, the alarm did not" result is one
  more instance of a point already made in general. It is not a contribution on its own.
- [Rewired or Gated?](https://arxiv.org/abs/2609.25602) (Sep 2026) reports that the choice
  between the prompt and memory is made by a handful of identifiable attention heads, in
  three model families. With Lookback Lens, that means finding such a head in the grounding
  project would be a replication. What may still be ours: testing on answers that are
  *correct* and ungrounded, calibrating only on visible cases, and the known-questions
  control. Read this paper in full before claiming any of it.
- [When Small Models Are Right for Wrong Reasons](https://arxiv.org/abs/2601.00513) covers
  the behavioural side from written reasoning traces, not internals.

**The opening: the Looped Lens Challenge**

[NDIF's challenge](https://ndif.us/looped-lens-challenge), with Thinking Machines, Goodfire
and Northeastern, opens 2026-11-23. Entrants submit a lens for Inkling 2 Mini as a Python
script; a judge scores the top tokens it reads at chosen layers on a held-out prompt set.
The page asks whether a Jacobian lens still works on this model. It also asks for general
insights into the model. Prizes, closing date and scoring rule were not on the page.

We already have per-pass Jacobian lenses fitted, an adapter for the looped layout, a
logit-lens comparison in which the Jacobian lens showed no advantage, and one finding about
what the messages carry (the token-position result). Needs an NDIF key and a registration.

**Ranking**

1. ~~Enter the challenge.~~ Declined by the author on 2026-11-22: the aim is independent research.
2. Finish grounding Phases 2 and 3 (offline, free), then decide Phase 4 on the result and
   on a full read of the knowledge-conflict paper.
3. Not worth starting: deception probes, evaluation awareness, cross-model probes. All are
   crowded with lab-scale work, and we have no edge.

### Follow-up read of "Rewired or Gated?" (2026-11-22)

Depth: a summary of the HTML version; about half the text was read verbatim by the reader,
the rest summarised. Figures not read.

It studies only cases where the context contradicts memory, on single-token facts
(ParaConflict), with attribution and ablation of heads. It has no case where context and
memory agree, builds no per-example detector, and reports no confidence baseline, false
alarms or transfer. It does not use attention to the answer's location as a signal. So the
grounding project's question (is a *correct* answer ungrounded, judged by a monitor
calibrated on visible cases, against a confidence baseline) is not answered there. Its
limitations section asks for a second, naturally occurring conflict corpus; NQ-Swap is one.

## Rethink from first principles (2026-11-23)

Asked what would be most useful and interesting to do next, after two projects and a first
round of benchmark audits.

**What oversight needs.** To know when a model's output cannot be trusted. There are three
kinds of evidence: what the model says, how its behaviour changes when the situation is
varied, and what happens inside it. The safety case for interpretability is that the third
adds something the first two do not.

**What we can do.** No frontier weights, tens of dollars. What has worked: cheap, careful
measurement with predictions first, an outside reviewer, and testing a result against the
simplest explanation. Our most useful findings came from asking whether a measurement means
what it claims.

**What our own results say.** They disagree, and the disagreement is the interesting part.

- Inkling: output confidence at chance (0.49 to 0.51), one attention head near-perfect.
  Input and output were identical by construction; only the internal route differed.
- Grounding: a baseline on the question word and output confidence came within 0.03 of the
  internal monitor. The input itself gave away the category.

**Hypothesis.** Internal access pays when nothing visible distinguishes the cases, and
published monitors are seldom tested under that condition.

**Direction chosen.** Measure the "inside advantage" across published internal monitors:
reproduce or take the monitor's score, build the strongest cheap outside baseline (input
text, output confidence), report the paired difference, and sort tasks by how much of the
signal is visible from outside.

**Ranked alternatives.** (2) Do models say truthfully why they answered, and is that
improving across model generations? (3) Sycophancy across 15 models from four families over
2.5 years (line-up worked out, estimated $8 to $15, not run).
