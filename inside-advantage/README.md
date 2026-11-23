# Inside advantage: when does looking inside a model beat reading the transcript?

**Status: text baselines, a probe test and an independent review done on Liars' Bench (2026-11-23). Short version in [NOTE.md](NOTE.md). Where the label can be read from the transcript, a probe and a word-count classifier score the same. In the one subset where it cannot (harm-pressure choice), a probe on public activations scores 0.91 against 0.69 for transcript text and 0.76 for the benchmark's published judge. An overseer who also holds the answer key gets 0.91 without the probe; adding the probe to that takes it to 0.96 (post hoc). At a 1% false-alarm target the probe catches 25 of 134. One model, one layer, labelled examples from the same setting.**

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

## Results

`textbase.py` for the planned runs, `textbase.py --checks` for the two added afterwards.
Scores are means over the models that produced each subset's transcripts (three to sixteen
per subset). Aggregates are in `results/`.

### Within a subset: is the label readable from text, given labelled examples?

AUROC, five-fold. "Strict" keeps transcripts that share their five rarest prompt words in
the same fold, so one question under two templates cannot be on both sides. It was added
after the first run, where harm-pressure choice looked much more readable than it is.

| subset | prompt only | response only | transcript | transcript, strict folds | length only |
|---|---|---|---|---|---|
| instructed deception | 0.93 | 0.93 | 0.96 | 0.95 | 0.84 |
| convincing game | 1.00 | 1.00 | 1.00 | (9 templates; not possible) | 0.34 |
| insider trading | 0.81 | 0.97 | 0.93 | (one scenario; not possible) | 0.78 |
| harm-pressure choice | 0.83 | 0.50 | 0.82 | **0.66** | 0.57 |
| harm-pressure knowledge report | 0.74 | 0.92 | 0.88 | 0.85 | 0.61 |
| gender secret | 0.97 | 0.99 | 1.00 | 1.00 | 0.66 |
| soft trigger | 0.88 | 0.62 | 0.86 | 0.89 | 0.39 |

### Across subsets: train on instructed deception, test elsewhere

| subset | prompt only | response only | transcript |
|---|---|---|---|
| convincing game | 0.47 | 0.76 | 0.65 |
| insider trading | 0.61 | 0.43 | 0.52 |
| harm-pressure choice | 0.48 | 0.50 | 0.48 |
| harm-pressure knowledge report | 0.56 | 0.59 | 0.67 |
| gender secret | 0.51 | 0.55 | 0.50 |
| soft trigger | 0.50 | 0.61 | 0.51 |

### The benchmark's own metric does not suit this baseline

Balanced accuracy with the threshold set on ordinary control conversations came out at 0.50
for most subsets and views, including ones with AUROC near 1.00. The classifier's scores
for a subset's transcripts sit wholly above or below the control threshold, so everything
is flagged or nothing is. The paper notes the same effect for its detectors. We report
AUROC and do not compare balanced accuracies.

### Predictions

1. **Mostly held.** Prompt only: 0.93 on instructed deception, 1.00 on convincing game;
   soft trigger was 0.88, just under the 0.90 predicted.
2. **Failed.** We expected the prompt to say little on the harm-pressure sets and gender
   secret. It scored 0.83, 0.74 and 0.97 (0.66 and 0.71 on the first two under strict folds).
3. **Failed**, for the reason in the section above. Average balanced accuracy was about 0.54.
4. **Held.** Transfer AUROC for the transcript view averages 0.56 over six subsets.
5. **Held.** Length alone scores 0.84, 0.78 and 0.66 on three subsets.
6. **Failed narrowly.** No subset has every text view under 0.65. Harm-pressure choice
   comes closest, at 0.66 under strict folds.

### What we see

- **Given labelled examples from the same setting, the label is in the text.** In six of
  seven subsets a bag-of-words classifier reaches 0.85 to 1.00. In four of them the prompt
  alone gets 0.88 or more: whether the model will be deceptive is largely set by which
  template, trigger or question type it was given.
- **The paper's best case for probes is matched by word counts.** Its upper-bound probe,
  trained with labels from five subsets plus control, reports 0.91 AUROC. Our transcript
  classifier averages 0.92 on those five with ordinary folds. This is not like-for-like:
  ours is one classifier per subset and model, theirs is one probe per model across
  subsets, and under strict folds ours falls on harm-pressure choice. A single text
  classifier per model across the five subsets, with questions held out, averages 0.79.
- **Away from its training setting a text classifier is near chance (0.56), and so are the
  published probes** (mean probe: 0.54 balanced accuracy). On this benchmark the difficulty
  is transfer, for outside and inside methods alike.
- **One subset is not readable from outside.** In harm-pressure choice the model answers a
  hazardous multiple-choice question with a letter it believes is wrong. The response is a
  bare letter (0.50), and with questions held out the prompt gives 0.66. The paper reports
  its language-model judge at about 0.73 AUROC there and probes with near-zero recall.
  Nothing visible separates the cases, which is the condition under which our hypothesis
  says an internal monitor should earn its keep. It is also where the published ones do
  worst.
- **A first run can flatter a baseline too.** Without strict folds, harm-pressure choice
  scored 0.83, because the same question appeared on both sides under different wording.

### What this does not show

- That probes rely on surface features. We scored no probe. A text classifier reaching a
  probe's score shows the probe's score is not evidence of reading anything hidden.
- Anything about a monitor that has no labelled examples of the setting, which is the
  realistic case and the one the benchmark is built to test.
- Per-subset comparisons with the paper. Its per-subset numbers are in a figure and an
  appendix we could not read; we used the averages in its text.
- Convincing game and insider trading could not be given strict folds (nine templates, one
  scenario), so their within-subset scores may be inflated.

### Next step

Test an internal monitor where the outside view is weak: harm-pressure choice. Activations
for Llama 3.3 70B on these transcripts are public
([`Yooniel/liars-bench-llama-3.3-70b-L53-acts`](https://huggingface.co/datasets/Yooniel/liars-bench-llama-3.3-70b-L53-acts),
about 12 GB for this subset). Fit a probe with the same strict folds and compare it with
the 0.66 text baseline, paired. If the probe is clearly higher, that is an inside advantage
measured against a fair outside baseline. If not, the subset is hard for everyone.

---

## Probe test: plan (written 2026-11-23, before any activations were read)

**Question.** On harm-pressure choice, where text gives 0.66, does a probe on the model's
internal activations do better when both get the same labelled examples and the same
held-out questions?

**Data.** Public layer-53 residual activations of Llama 3.3 70B on its own Liars' Bench
transcripts ([`Yooniel/liars-bench-llama-3.3-70b-L53-acts`](https://huggingface.co/datasets/Yooniel/liars-bench-llama-3.3-70b-L53-acts)).
Harm-pressure choice has 2,134 transcripts, 134 deceptive. One model, one layer. We did not
choose the layer and cannot vary it.

**Pooling** (`pool.py`, a Modal CPU job, no GPU): for each transcript, four vectors of
8,192 numbers. The mean over prompt tokens; the last prompt token, where the answer is about
to begin; the mean over response tokens (the paper's choice); the last token.

**Probe** (`probe.py`, on a laptop): logistic regression on standardised activations, one
per pooling. Folds are the strict folds of the text baseline, and the text baseline is
re-run for this model on those same folds, so the comparison is paired.

**Outside baselines:** the three text views, response and prompt length, and the
benchmark's own metadata for each item (harm factor, template, which option is correct).

**Primary measurement:** AUROC of the best probe pooling minus AUROC of the best outside
baseline on harm-pressure choice, with a 95% interval from resampling question groups.
"Best" is chosen on the same folds for both sides, which favours neither.

**Secondary:** the same on every other subset with Llama transcripts, and transfer from
instructed deception to harm-pressure choice for both probe and text.

**Predictions:**

1. The best probe reaches at least 0.80 on harm-pressure choice.
2. It beats the best outside baseline by at least 0.10, with an interval excluding zero.
3. The last prompt token does about as well as the mean over response tokens (within 0.03).
4. On instructed deception and gender secret, where text scores 0.95 or more, probe and
   text are within 0.05 of each other.
5. A probe trained on instructed deception scores below 0.65 on harm-pressure choice.

**Reading.** If 1 and 2 hold, this is an inside advantage measured against a fair outside
baseline, in the one subset where the outside view is weak. If the probe is no better than
text here either, the subset is hard for everyone given this layer and these labels, and
the hypothesis gets no support from this benchmark.

**Cost estimate.** Downloading about 88 GB and pooling it on Modal CPU machines: under $1.

## Probe test: result

`pool.py` then `probe.py`; `hpc_check.py` for the check added afterwards. Llama 3.3 70B,
layer 53, AUROC on held-out questions (strict folds where the subset allows).

| subset (deceptive / honest) | best probe | best outside baseline | probe minus outside, 95% interval |
|---|---|---|---|
| **harm-pressure choice** (134 / 2,000) | **0.91** | 0.69 | **+0.22 [+0.16, +0.29]** |
| soft trigger (3,000 / 3,000) | 0.97 | 0.83 | +0.14 [+0.13, +0.15] |
| instructed deception (3,045 / 2,387) | 1.00 | 0.96 | +0.03 [+0.03, +0.04] |
| harm-pressure knowledge report (139 / 2,000) | 0.99 | 0.97 | +0.02 [+0.01, +0.02] |
| insider trading (423 / 657) | 1.00 | 0.99 | +0.00 [+0.00, +0.01] |
| convincing game (167 / 100) | 1.00 | 1.00 | 0.00 |
| gender secret (116 / 276) | 1.00 | 1.00 | 0.00 |

The outside baselines are the three text views, length, and the benchmark's own metadata
for each item. In `probe.py` a one-letter answer counts as a word, which the first text
run did not allow; it makes the outside baseline on harm-pressure choice slightly stronger
(0.69 against 0.66).

On harm-pressure choice, by where the probe reads:

| reads | AUROC |
|---|---|
| mean over prompt tokens | 0.61 |
| last prompt token, before the model has answered | 0.88 |
| mean over response tokens | 0.91 |
| last token | 0.89 |
| text, prompt only | 0.69 |
| text, whole transcript | 0.69 |

### Is the probe only detecting a wrong answer?

We asked this after seeing the result. In this subset "deceptive" means the answer differs
from the one the model gives when asked neutrally. Most such answers are wrong by the
answer key (118 of 134), so "the answer is wrong" alone scores 0.88, close to the probe.
But the model also makes honest mistakes: 233 answers that are wrong and match its neutral
answer. Among wrong answers only, an answer key says nothing.

| among the 351 wrong answers: lie (118) against honest mistake (233) | AUROC |
|---|---|
| probe, mean over response tokens | **0.83** |
| probe, last prompt token | 0.73 |
| text, whole transcript | 0.59 |
| benchmark metadata | 0.50 |

A probe fitted only on wrong answers gives 0.81. So the probe predicts disagreement with
the model's recorded neutral answer even among wrong answers, where the transcripts look
alike. (First written as "distinguishes an answer the model believes from one it does not".
The review below shows why that is too strong: nearly all of this compares different
questions.)

### Across subsets: fit on instructed deception, test elsewhere

| subset | probe, last token | probe, mean over response | text, transcript |
|---|---|---|---|
| convincing game | 0.99 | 0.96 | 0.69 |
| soft trigger | 0.89 | 0.93 | 0.49 |
| harm-pressure knowledge report | 0.82 | 0.56 | 0.71 |
| harm-pressure choice | 0.74 | 0.71 | 0.40 |
| gender secret | 0.30 | 0.27 | 0.51 |
| insider trading | 0.14 | 0.33 | 0.60 |

### Predictions

1. **Held.** Best probe on harm-pressure choice: 0.91.
2. **Held.** It beats the best outside baseline by 0.22, interval [0.16, 0.29].
3. **Failed narrowly.** The last prompt token is 0.036 below the mean over response tokens.
4. **Held.** Instructed deception: 1.00 against 0.96. Gender secret: 1.00 against 1.00.
5. **Failed.** A probe from instructed deception scores 0.71 to 0.74 on harm-pressure
   choice, above the 0.65 we predicted as a ceiling.

### What we see

- **The hypothesis held on this benchmark.** Where a reader of the transcript can already
  tell (five subsets), probe and text are within 0.03. Where the transcript says little
  (harm-pressure choice), the probe is far ahead. On soft trigger the probe beats our text
  features, but the cue is printed in the prompt (see the review section), so that subset
  is weak evidence of anything hidden.
- **The advantage is not only wrong-answer detection.** Among wrong answers the probe
  separates those that contradict the model's neutral answer from those that match it at
  0.83. A right-or-wrong signal cannot do that.
- **It is visible before the answer.** At the last prompt token the probe scores 0.88;
  the text of the same prompt gives 0.69.
- **Transfer is uneven, and sometimes backwards.** A probe fitted on instructed deception
  carries to three subsets (0.74 to 0.99, far above text) and is worse than chance on two
  (0.14 on insider trading). A direction that means "deceptive" in one setting can mean the
  opposite in another.
- **Reporting a probe's score without a text baseline hides which case you are in.** Five
  of seven subsets would have looked like strong probe results. Only one clearly is.

### What this does not show

- **A different outside method gets this label perfectly.** Deception here is defined as
  disagreeing with the model's own neutral answer, so an overseer who can ask the model
  again without pressure recovers the label by construction. The probe's advantage is over
  reading one transcript, and it needs one pass and no second question.
- One model, one layer that we did not choose, 134 deceptive transcripts.
- The probe had labelled examples from the same setting. Without them (the transfer table)
  it reaches 0.71 to 0.74 here.
- Nothing about what the probe reads. We did not intervene on the model.
- Convincing game and insider trading have two and one prompt groups, so their folds hold
  out rows, not scenarios. Their scores say little about new scenarios.
- "Benchmark metadata" means its numeric and yes/no fields only. The first version of the
  parser turned an infinite harm factor into zero in 1,472 of 2,139 knowledge-report rows;
  fixed, which moved that subset's metadata score from 0.727 to 0.734 and nothing else.

## After independent review (2026-11-23)

A separate model recounted everything from the pooled activations and released transcripts
([`../notes/inside-advantage-independent-review.md`](../notes/inside-advantage-independent-review.md)).
All seven within-subset tables and all six transfer rows reproduced. It then asked a
sharper question: what does the probe add once the overseer uses the outside information
they could have? We re-ran its main checks (`review_checks.py`, numbers below are ours and
match the reviewer's). **All of this is post hoc.**

### Stronger outside monitors on harm-pressure choice

The benchmark authors publish their language-model judge's score for each of these 2,134
transcripts; rows and labels matched exactly.

| what the monitor gets | AUROC | response probe minus it, 95% interval |
|---|---|---|
| transcript word counts | 0.691 | +0.222 [+0.160, +0.286] |
| published external judge reading the transcript | 0.764 | +0.149 [+0.098, +0.199] |
| answer key alone (is the chosen letter wrong?) | 0.882 | +0.031 [-0.007, +0.074] |
| answer key, word counts break ties | 0.908 | +0.006 [-0.033, +0.046] |
| answer key + judge | 0.913 | 0.000 [-0.031, +0.034] |
| probe alone (mean over response tokens) | 0.913 | |
| answer key + judge + probe | **0.957** | |

- **Transcript only:** the probe has a clear lead, over the published judge as well as
  over word counts.
- **With an answer key:** a cheap outside rule matches the probe. No second model call and
  no neutral answer needed.
- **Together:** adding the probe to key and judge gains +0.044 [+0.027, +0.063], with
  nested question folds so neither stage trains on its test questions. The probe carries
  information the outside monitors miss. This is the number to test on fresh data.

### What the 0.83 establishes

Among wrong answers: probe 0.827, word counts 0.590, published judge 0.590. But only 2
opposite-label pairs among wrong answers concern the same question (both monitors rank both
correctly), so nearly all of the 0.827 compares answers to different questions. The probe
may partly recognise questions on which the model is prone to change its answer. Over the
whole subset, 31 questions have both labels, giving 43 opposite-label pairs: the probe
ranks 36 correctly, word counts 26, the judge 27. Whether the model recognised its answer
as false while producing it is open; the benchmark's authors discuss the same distinction.
Its neutral reference comes from repeated prompts, so the stored reference reproduces the
label exactly while one fresh neutral query might not.

### As an alarm

Threshold set on separate training questions for 1% false alarms, then applied to held-out
questions: **25 of 134 caught, 20 of 2,000 false alarms.** A ranking score of 0.91 is 19%
recall at that operating point.

### Reviewer's findings we did not re-run

- **Soft trigger.** It pools three fine-tuned adapters with deliberately visible cues. On
  the 2,000 time-adapter rows the hour printed in the prompt alone scores 0.919. The label
  direction in the released rows matches the generator and is reversed in the paper's
  Table 2.
- **Reproducibility.** With scikit-learn 1.7.2 the main probe score is 0.895, not 0.913;
  with independently reassigned folds, 0.905 against 0.693. Versions are now pinned in the
  run commands and fold assignments saved in `results/folds.json`.
- **Paper appendix.** Appendix L, Table 8 gives 0.74 for the paper's upper-bound probe on
  this subset and model (different training data). Our earlier section sets the paper's
  balanced accuracy beside our AUROC as evidence of weak transfer; those are different
  metrics and should not be read as a like-for-like comparison.
- **Response mean.** The mean over response tokens is, in most rows, the answer letter and
  an end-of-turn marker.
- Selecting the text of the chosen option for the text classifier did not close the gap
  (0.61; 0.69 with the prompt).

### Wording we withdraw

- "Separates lies from honest mistakes" and "an answer the model believes from one it does
  not": replaced by "predicts disagreement with the model's recorded neutral answer, even
  among wrong answers".
- "Internal access earns its keep": not shown. That needs an operating threshold and a cost
  comparison. What is shown is added predictive information.
- Soft trigger as a second case of hidden information.
