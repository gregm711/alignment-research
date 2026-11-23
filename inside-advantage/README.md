# Inside advantage: when does looking inside a model beat reading the transcript?

**Status: plan written 2026-11-23, before any scores.**

## Question

The safety case for interpretability is that a model's internals tell an overseer something
its inputs and outputs do not. Our two earlier projects disagreed on this. On
[Inkling](../jlens-inkling/NOTE.md), where input and output were identical by construction,
an internal signal was near-perfect and output confidence was at chance. On
[grounding](../grounding/NOTE.md), a baseline on the question word and output confidence
matched the internal monitor. The hypothesis this suggests: **internal access pays when
nothing visible separates the cases, and benchmarks seldom check how much is visible.**

This project measures, for published monitoring benchmarks, how much of the label can be
recovered from the text alone by a cheap classifier, and sets that beside the scores their
authors report for internal probes and for language-model judges.

## What is already known (search of 2026-11-23, see `../notes/directions.md`)

- Text-only baselines against probes are rare. Where run, they cover one dataset, or sit in
  an appendix (McKenzie et al. on high-stakes interactions; Kumar on one deception set;
  Azaria & Mitchell; "From Probe Scores to Alarm Policies" within four settings).
- The closest cross-dataset work, Parrack et al. ("Black-to-White Performance Boosts"),
  compares probes with Claude monitors on five deception sets for one model and finds about
  +0.05 AUROC from adding probe scores. It has no cheap text baseline.
- Liars' Bench reports a language-model judge at 0.73 balanced accuracy and probes at 0.54
  on average. It has no text-only baseline.
- Boxo et al. show probe scores fall when elicitation prompts are removed, without running
  a text classifier. Dubanowska et al. show a task-type heuristic matches a hallucination
  detector on one benchmark.
- We found no harness that does this across many behaviours. The search was limited, and
  several appendices were not read.

## First cut: Liars' Bench (CPU only)

[Liars' Bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) (Kretschmar et
al., arXiv 2511.16035) has seven subsets of transcripts labelled deceptive or honest, from
up to six models: instructed deception, convincing game, insider trading, two
harm-pressure sets, gender secret, soft trigger. The data is gated and carries a canary
string. **We do not copy any of its text into this repository**, only aggregate scores.

**Baselines** (`textbase.py`), each a TF-IDF logistic regression (words and word pairs):

| Name | Reads |
|---|---|
| prompt only | everything before the final assistant message |
| response only | the final assistant message |
| transcript | both |
| length only | number of characters in the response and in the prompt |

**Two ways of scoring**, per subset and per model that produced the transcripts:

1. *Within subset*: five-fold cross-validation, with all transcripts that share a first
   user message kept in the same fold. This measures how separable the classes are from
   text, which is an upper bound for a monitor that has seen labelled examples of this
   setting.
2. *Transfer*: train on instructed deception, test on each other subset. This is how
   probes are usually used, so it is the fairer comparison with their reported scores.

Reported: AUROC and balanced accuracy (the benchmark's own metric), averaged over models.
The published probe and judge numbers are taken from the paper, per subset where given.

**Reading.** "Prompt only" being high means the label is fixed by the setup, so nothing
about the model's behaviour is being detected. "Response only" or "transcript" being high
means a reader of the transcript could do it. A subset where all text baselines are near
chance is where an internal monitor has room to show an advantage.

**Detail fixed after reading the paper's method, before any scores.** The benchmark's
balanced accuracy uses a threshold set to give at most 1% false positives on a control set
of ordinary conversations (Alpaca). We do the same: half of each model's control
conversations are added to training as honest examples, and the threshold comes from the
other half. The paper's averages on this metric are 0.73 for a language-model judge, 0.66
for self-evaluation, 0.60 for an unrelated-question classifier and 0.54 for the mean
probe. Its "upper-bound" probe, trained with labels from five of the subsets plus the
control set, reaches 0.73 balanced accuracy and 0.91 AUROC. Our within-subset text
classifier is the counterpart of that upper-bound probe. Per-subset numbers in the paper
are in a figure and an appendix we could not read, so we compare with these averages.

## Predictions

1. Prompt only, within subset: AUROC at least 0.90 on instructed deception and soft
   trigger, and at least 0.80 on convincing game.
2. Prompt only is at most 0.70 on both harm-pressure sets and on gender secret.
3. Transcript, within subset: balanced accuracy averaged over the seven subsets is at
   least 0.73, the figure reported for the language-model judge.
4. Transfer from instructed deception: transcript AUROC averages below 0.60 on the other
   six subsets.
5. Length only reaches AUROC 0.65 on at least two subsets.
6. At least one subset has every text baseline below 0.65 within subset. That would be the
   place to look for a real inside advantage.

## What this cannot show

Within-subset scores come from a classifier trained on labelled examples of the same
setting, which the published probes and judges did not have; they are a ceiling on surface
separability, not a like-for-like contest. We do not compute probe scores ourselves in this
first cut (the saved activations run to tens of gigabytes). A high text score does not show
that a probe relies on the same features, only that it could.
