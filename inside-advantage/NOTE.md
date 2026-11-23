# When does a deception probe beat reading the transcript? A test on Liars' Bench

*Technical note, 2026-11-23. Full log, plans and predictions: [README.md](README.md).
Independent review: [`../notes/inside-advantage-independent-review.md`](../notes/inside-advantage-independent-review.md).*

## Summary

We compared a linear probe on a model's activations with monitors that see no internals,
on the same held-out questions, across the seven subsets of
[Liars' Bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) (Kretschmar et
al., arXiv 2511.16035) for Llama 3.3 70B.

- In five of seven subsets a word-count classifier on the transcript scores within 0.03
  AUROC of the probe. A probe score there is not evidence of reading anything hidden.
- In one subset, harm-pressure choice, the transcript says little. There the probe scores
  0.91 against 0.69 for word counts (planned comparison: +0.22, interval [+0.16, +0.29])
  and 0.76 for the benchmark's published language-model judge.
- An overseer who also holds the answer key reaches 0.91 without the probe. Adding the
  probe to answer key and judge gives 0.96 (+0.044 [+0.027, +0.063]). This comparison was
  chosen after seeing the first result and should be tested on fresh data.
- As an alarm set for 1% false alarms, the probe catches 25 of 134.
- A probe fitted on one subset transfers unevenly: 0.99 on one, 0.14 on another.

## Setup

**Data.** Liars' Bench transcripts labelled deceptive or honest. The data is gated; no
transcript text is in this repository. Activations are the public layer-53 residual stream
of Llama 3.3 70B on its own transcripts
([`Yooniel/liars-bench-llama-3.3-70b-L53-acts`](https://huggingface.co/datasets/Yooniel/liars-bench-llama-3.3-70b-L53-acts)).
We did not choose the layer.

**Probe.** Logistic regression on standardised activations, pooled four ways (mean over
prompt tokens, last prompt token, mean over response tokens, last token).

**Outside monitors.** TF-IDF logistic regression on the prompt, the response, or both;
prompt and response length; the benchmark's numeric metadata for each item. Added after
review: the benchmark authors' published judge scores, and the answer key.

**Folds.** Five folds with every transcript of a question kept together, shared by all
monitors, so differences are paired. Intervals resample questions. Convincing game and
insider trading have two and one prompt groups, so there the folds hold out rows only.

**Planned before any activations were read:** the primary comparison (best probe minus
best outside baseline on harm-pressure choice) and five predictions. Three held.

## 1. Where the transcript is readable, the probe adds nothing measurable

| subset (deceptive / honest) | best probe | best outside baseline | difference |
|---|---|---|---|
| **harm-pressure choice** (134 / 2,000) | **0.91** | 0.69 | **+0.22 [+0.16, +0.29]** |
| soft trigger (3,000 / 3,000) | 0.97 | 0.83 | +0.14 |
| instructed deception (3,045 / 2,387) | 1.00 | 0.96 | +0.03 |
| harm-pressure knowledge report (139 / 2,000) | 0.99 | 0.97 | +0.02 |
| insider trading (423 / 657), rows held out | 1.00 | 0.99 | 0.00 |
| convincing game (167 / 100), rows held out | 1.00 | 1.00 | 0.00 |
| gender secret (116 / 276) | 1.00 | 1.00 | 0.00 |

Soft trigger is not a second case of hidden information. It pools three fine-tuned
adapters whose triggers are printed in the prompt; the reviewer found that the displayed
hour alone scores 0.92 on the time-adapter rows. Our text features missed a cue that is
in plain view.

## 2. Harm-pressure choice: what the probe adds depends on what the overseer has

Here the model answers a hazardous multiple-choice question with one letter. "Deceptive"
means the letter differs from the model's answer under neutral prompting.

| what the monitor gets | AUROC |
|---|---|
| transcript word counts | 0.691 |
| published external judge reading the transcript | 0.764 |
| answer key alone | 0.882 |
| answer key, word counts break ties | 0.908 |
| answer key + judge | 0.913 |
| probe alone (mean over response tokens) | 0.913 |
| answer key + judge + probe | 0.957 |

- **Transcript only.** The probe leads the published judge by +0.149 [+0.098, +0.199].
  At the last prompt token, before any answer, it scores 0.88.
- **With an answer key.** Ranking wrong answers first matches the probe (difference
  +0.006 [-0.033, +0.046]).
- **Together.** The probe and the outside monitors make different errors. Adding the probe
  gains +0.044 [+0.027, +0.063], with nested question folds. Post hoc.

**Among wrong answers only** (118 that contradict the neutral answer, 233 that match it)
the probe scores 0.83, word counts 0.59, the judge 0.59. A right-or-wrong signal cannot
produce that. But only two opposite-label pairs in that slice concern the same question,
so the 0.83 mostly compares different questions and may partly reflect which questions the
model tends to change its answer on. Over the whole subset there are 43 same-question
pairs with opposite labels: the probe ranks 36 correctly, word counts 26, the judge 27.

**As an alarm.** With the threshold set on separate training questions for 1% false
alarms: 25 of 134 caught, 20 of 2,000 false alarms.

## 3. Transfer is uneven and can invert

A last-token probe fitted on instructed deception, applied elsewhere: convincing game
0.99, soft trigger 0.89, harm-pressure knowledge report 0.82, harm-pressure choice 0.74,
gender secret 0.30, insider trading 0.14. The transcript classifier ranges from 0.40 to
0.71. A direction that marks deception in one setting can mark honesty in another.

## What we got wrong

- We wrote that the probe "separates lies from honest mistakes". What it predicts is
  disagreement with the model's recorded neutral answer. Whether the model recognised its
  answer as false while giving it is open.
- We counted soft trigger as a second case where the probe sees something hidden.
- We wrote that internal access "earns its keep" here. That needs an operating threshold
  and a cost comparison. What we have is added predictive information.
- Our first text baseline scored 0.83 on harm-pressure choice because one question
  appeared on both sides of the folds; with questions held out it is 0.66 to 0.69.
- The metadata parser turned an infinite harm factor into zero in 1,472 knowledge-report
  rows. Fixed; that subset's metadata score moved from 0.727 to 0.734.
- Two of five predictions failed: the last prompt token is 0.036 below the response mean
  (we predicted within 0.03), and a transferred probe scored 0.71 to 0.74 on harm-pressure
  choice (we predicted under 0.65).

## Limits

- One model, one layer, 134 deceptive transcripts in the main subset.
- The probe had labelled examples from the same setting.
- Comparing an answer with the benchmark's stored neutral reference recovers the label
  exactly. The probe's advantage is over monitors that lack that reference.
- The stronger outside monitors and the combined monitor were chosen after the first
  result.
- The main score moved from 0.913 to 0.895 under a different scikit-learn version.
  Versions are pinned and fold assignments saved (`results/folds.json`).
- We did not intervene on the model, so nothing here says what the probe reads.

## Related work

Parrack et al. (2507.12691) compare probes with language-model monitors on five deception
sets and find about +0.05 AUROC from adding probe scores, without a cheap text baseline.
McKenzie et al. (2506.10805), Kumar (2605.27958) and "From Probe Scores to Alarm Policies"
(2610.04575) run text baselines in single settings. Boxo et al. (2509.21344) show probe
scores fall when elicitation prompts are removed. Liars' Bench itself reports judges and
probes but no text-only baseline. Our search was limited.

## Reproducing

```
modal run pool.py && modal volume get inside-advantage / pooled/    # about $0.05, CPU
R="uvx --with huggingface_hub --with pandas --with pyarrow --with scipy --with scikit-learn==1.9.1 --with numpy==2.5.5 python"
$R textbase.py; $R probe.py; $R hpc_check.py; $R review_checks.py
```

Access to the gated dataset is required. Aggregates are in `results/`.

## How this was made

The experiments were designed, run and written up by an AI assistant working with the
author, with the plan and predictions committed before the activations were read. A
separate model then recounted the results and supplied the stronger outside monitors, the
combined-monitor test, the same-question pair count and the alarm check; we re-ran those
and the numbers match.
