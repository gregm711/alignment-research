# Loop legibility: when a model loops, how much of its reasoning leaves the transcript?

**Status: Inkling part A done and independently recounted; Ouro done, not yet reviewed (2026-11-23). Short version: on these models and this task we did not see reasoning leave the transcript when looping increased. Ouro at 4 loops writes as much as at 2 (14% more on two lookups, 5% less on three, the second within noise) and names every step, but the task was at ceiling for it (40 of 40 everywhere), so this is a weak test. Separately, when Inkling is forced to answer in one token, the unwritten step is readable from inside on two lookups (0.85 against about 0.10 for controls) and partly on three (39 of 73 successful runs).**

## Question

Reading a model's written reasoning is the main monitoring tool in use. Looped models run
a block of layers several times per token, so they can do more work without writing
anything. Z.ai's GLM-5.4 post ([blog](https://z.ai/blog/glm-5.4), 2026-11-23) reports, for
a 30B looped ablation with thinking on, that "average output length drops by 10–14%,
mostly from shorter chains of thought. In effect, part of the reasoning shifts from
explicit output tokens into the implicit depth of the loop." No weights are released at
any size, and the post does not measure what a monitor loses.

We ask, on looped models we can run:

1. As the number of passes goes up, does the model write fewer of its intermediate steps?
2. Can a probe on what one pass hands to the next recover the steps that are no longer
   written?
3. Does transcript plus probe at full looping see as much as the transcript alone did
   with no looping?

## What is already known (search of 2026-11-23, abstracts and model cards only)

- Olszowiec et al. ([2610.18127](https://arxiv.org/abs/2610.18127)) read eight looped
  models at each iteration through the output head. No transcript monitor, no trained
  probe, no reasoning length against iteration count.
- Wang & Reid ([2609.01924](https://arxiv.org/abs/2609.01924)) apply the Jacobian lens to
  Ouro and Huginn. No monitor comparison.
- Ferreira, Aziz & Titov ([2608.04928](https://arxiv.org/abs/2608.04928)) compare text,
  probe and verbalised-latent monitors across explicit and latent reasoning. It appears
  not to use looped models. This is the closest work and we have read only its abstract.
- Our own Inkling log has a hint: on a public multi-hop set, the one-pass output named an
  intermediate in 5 of 80 items and the three-pass output in 1 of 80.
- We found no paper that measures written reasoning against loop count, or a transcript
  monitor against a between-pass probe, on a looped model. Two searches; not exhaustive.

**Added 2026-11-23, after the pilot and after the predictions below were written. The
predictions are left unchanged.** We missed that the Inkling release post
([link](https://thinkingmachines.ai/news/inkling-2-mini/)) already measures reasoning
length against loop count on the variable-loop checkpoint. That checkpoint was trained
with **1 to 5 passes**, not up to 3 as stated in the table above. Their results:

- GPQA Diamond items correct at every pass count: mean reasoning tokens 3,703 / 1,731 /
  1,661 at 1 / 3 / 5 passes. That is already far outside our prediction 2 of "5 to 20%
  shorter".
- Legibility is measured only by surface statistics: words per sentence, share of
  dictionary words, and the rate of "wait / actually / hmm". These are flat, except the
  hedging words, which fall from 9.7 to 3.6. They do not check whether specific steps are
  still written.
- LinuxArena stealth rate (monitor sees tool calls only) is 1.0 / 2.5 / 4.0 / 2.8 / 5.0% at
  1 to 5 passes over 184 pairings. They call this "no clear trend". With the full
  transcript visible, every rate is at most 0.6%.
- RL was done at 3 passes only. They say RL at higher pass counts might compress reasoning
  more, and leave that to future work.

What is still open, and is our question: **which content disappears** when the reasoning
gets shorter, and whether a probe recovers it. Surface legibility can stay flat while the
intermediate steps go unwritten.

## Models

| Model | Loop design | How we vary looping | Written reasoning |
|---|---|---|---|
| `thinkingmachines/TML-d24-varloop-research` (Inkling's variable-loop sibling) | 12 of 24 layers run up to 3 times; each pass restarts and receives learned messages from the last | `loop_num_passes` = 1, 2, 3 on the same weights, which were trained for variable passes | thinking mode with an effort dial |
| `thinkingmachines/Inkling-2-Mini` | same, trained at 3 passes only | 3 passes, as a check | same |
| `ByteDance/Ouro-2.6B-Thinking` | all layers run 4 times; state carried forward | `exit_at_step` = 1 to 4 (trained at 4, so fewer steps is off-distribution) | `<think>` block |

One pass on the same weights is our "non-looped" comparison. It holds knowledge and
training data fixed, which a separate ordinary model would not. Ouro is the second pick
because its loop is a different design (carried state, not restart plus messages), it is
small enough for a cheap GPU, and it writes explicit reasoning.

## Task

Chained lookups over invented tables, so nothing comes from memory and every intermediate
step is known:

```
Fact sheet.
Color for each animal. ram: brown; cow: black; ...
Fruit for each color. red: orange; black: apple; ...
Tool for each fruit. apple: saw; ...
Start from the cow and follow the tables in order. What is the final tool?
```

Chains of 2, 3 and 4 lookups (1, 2 and 3 intermediate steps). Asked through each model's
chat template, with thinking on and with thinking off.

## Measurements

For each model, pass count, thinking setting and item:

- **Correct:** the final answer is the right word.
- **Reasoning length:** generated tokens before the final answer.
- **Named in transcript:** for each intermediate step, whether its word appears in the
  generated text. This is what a transcript monitor can see. All candidate words are in
  the prompt, so only the generated text counts.
- **Decoded by probe:** a linear classifier over the candidate words, reading the state the
  final pass receives from the previous one (Inkling: the messages; Ouro: the carried
  state) at the position just before the answer. Fitted on one set of tables, tested on
  new tables. Chance is one in seven.
- **Controls for the probe:** the same probe on the state before the loop, and a probe
  for the intermediate of a *different* question on the same fact sheet. If those score as
  well, the probe is reading the prompt, not the reasoning.

**Legibility** of a condition = share of intermediate steps, among correctly answered
items, that are named in the transcript. **Inside recovery** = share decoded by the probe.
**Combined** = named or decoded.

## Primary comparison (fixed now)

On the variable-loop Inkling checkpoint, thinking on, items answered correctly at both
pass counts:

- (a) legibility at 3 passes minus legibility at 1 pass, and the same for reasoning length;
- (b) combined at 3 passes minus legibility at 1 pass.

Intervals by resampling fact sheets. Secondary: the same on Ouro between 1 and 4 steps;
accuracy with thinking off at each pass count (how much the loop does with nothing
written); the probe's controls.

## Predictions

1. With thinking off, accuracy on 3- and 4-lookup chains rises by at least 15 points from
   1 pass to 3 on the variable-loop checkpoint.
2. With thinking on, reasoning length at 3 passes is 5 to 20% shorter than at 1 pass, on
   items correct at both.
3. Legibility with thinking on falls by less than 10 points from 1 pass to 3. The model
   writes the steps out when asked to think, loop or no loop. (If so, the loss the GLM
   post implies is small on this task, which is itself worth knowing.)
4. With thinking off, legibility is under 10% at every pass count, and the probe decodes
   at least 60% of intermediate steps at 3 passes, against under 30% for both controls.
5. Combined at 3 passes is at least as high as legibility at 1 pass.
6. On Ouro, reasoning length changes by less than 10% across steps 2 to 4.

## What this cannot show

- Anything about GLM-5.4 or models of that size. These are 2.6B and Inkling-Mini-scale
  models on a toy task with known steps.
- That a probe would work without labelled intermediate steps, which real monitoring
  seldom has.
- That a named step was used, or that an unnamed one was not. Naming is what a monitor
  sees, not what the model did.
- Running Ouro at fewer steps than it was trained for mixes "less looping" with "off
  distribution". The variable-loop Inkling checkpoint does not have that problem.

## Plan and cost

1. **Pilot** (`pilot.py`): variable-loop Inkling, 8 items per chain length, passes 1 to 3,
   thinking on and off. No probe. It tells us whether the model can do the task, what its
   thinking looks like, and whether accuracy is off the ceiling anywhere. About 10 to 25
   GPU minutes on an A100-80GB: **$0.50 to $1.30**.
2. **Main run on Inkling**, with states saved for the probe: estimated **$2 to $3**,
   revised after the pilot.
3. **Ouro**, on a small GPU: under **$1**.

Estimated total $3.50 to $5.50. Spend across all projects so far is about $23.45.

## Pilot result (2026-11-23) and what it changed

`pilot.py`, 26.2 GPU minutes (about $1.35, at the top of the estimate). It was stopped by its
own time guard after the two-lookup chains only: 55 generations at about 30 seconds each,
several times slower than assumed. Counts below are from 8 fact sheets per cell.

| thinking effort | passes | correct | hit the 200-token limit |
|---|---|---|---|
| 0 | 1 / 2 / 3 | 6 / 8 / 8 of 8 | 8 / 7 / 8 of 8 |
| 0.9 | 1 / 2 / 3 | 4 / 6 / 8 of 8 | 8 / 8 / 8 of 8 |

- **The model thinks at length whatever the effort setting and pass count.** It first
  copies the fact sheet into its thinking, then works the chain out in words. Nearly every
  output hit the token limit, so the pilot says nothing about reasoning length.
- **At one pass it is visibly degraded.** In some outputs it reasons about an image that
  is not there.
- **The scorer was wrong.** Because the model copies the prompt, "the step's word appears
  in the output" is true for trivial reasons. The pilot's "steps named" counts are not
  reported.
- **"Thinking off" does not exist on this model** through the effort dial.

### Revised design (written before the main run)

**Part A: one-token answers.** The prompt ends "Answer: the" outside the chat format, so
the model must give the final word at once and no step can be written. Legibility is zero
by construction. 250 fact sheets per chain length (2, 3, 4), variable-loop checkpoint at
1, 2 and 3 passes, released checkpoint at 3. We save the last-token state before the
loop, at the end of each pass and after the last layer, and the messages into each pass at
the last position. Probes are fitted afterwards on a laptop (`analyse.py`): a linear
classifier for each intermediate step, trained on half the fact sheets and tested on the
other half. Controls as planned: the state before the loop, and the steps of a different
start animal on the same sheet. The city table became a metal table so that "the" reads
naturally before every answer.

**Part B: thinking on.** Two-lookup chains, 16 fact sheets, 1 to 3 passes, effort 0, up to
320 new tokens so that most outputs finish. The scorer deletes every copied table row from
the thinking before looking for the step's word. With 16 sheets this can show a large
change in reasoning length and cannot resolve one of 10%.

**What happens to the predictions.** 1 and 4 are tested by part A (4's "legibility under
10%" is now true by construction, so only its probe half is a test). 2 and 3 are tested by
part B, weakly. 5 needs transcript and probe on the same runs and is **not tested** by
this design. 6 waits for Ouro. Added now:

7. In part A, a probe at the end of the final pass decodes the last intermediate step
   better than the first one, on 3- and 4-lookup chains.
8. On chains the variable-loop checkpoint gets right only at 3 passes, the probe at the
   end of pass 1 decodes the first intermediate step above 50%: the early pass does the
   early step.

**Cost.** Part A about 10 to 15 GPU minutes, part B capped at 28: at most $2.80 together.
With the pilot that is $4.15 of the $3.50 to $5.50 agreed, leaving Ouro under $1.

### Wording checks before part A (2026-11-23)

The first launch of part A was stopped after 6 minutes: the model got 0 to 2 of 250 right
because "Answer: the" was continued as "the final tool is ...". Two short checks followed
(12 fact sheets per cell, variable-loop checkpoint; `main.py --peek-only`).

| wording | chain | top word is the answer, 1 pass | 3 passes |
|---|---|---|---|
| "X: Y" rows, "follow the tables in order" | 2 / 3 / 4 | 3 to 6 / 0 to 1 / 0 of 12 | 2 to 4 / 8 to 10 / 0 of 12 |
| sentence rows, "the tool that goes with the fruit that goes with the color of the ant" | 2 / 3 / 4 | 12 / 1 to 4 / 0 of 12 | 11 / 0 to 2 / 0 of 12 |

- The first format was unreliable even on two lookups, so it is dropped. Its apparent gain
  on three lookups at 3 passes is from 12 items in a format the model otherwise fails.
- In the sentence format (the one the earlier Inkling work used), two lookups are at
  ceiling at one pass already. On three and four lookups the model's top word is usually
  not a candidate at all (":", "tool", "metal"): asked for the answer in one token, it does
  not give one, at any pass count.

**Consequences, fixed before the run.** Part A uses the sentence format with the
question-and-answer ending. Accuracy is reported two ways: the top word overall, and the
highest-scoring of the seven candidates (chance 1 in 7). Prediction 1 will be scored on
the second. Part A now asks mainly what the probes can answer: how far along the chain
the internal state gets at each pass count, whether or not the output shows it. Part B is
held back until part A is read; with generation this slow it may be better spent on Ouro.

Spend so far on this project: about $1.95 (pilot $1.35, stopped launch $0.30, two wording
checks $0.15 each).

## Part A: result

`main.py --part-b-minutes 0` (17.4 GPU minutes, about $0.90), then `analyse.py`. 250 fact
sheets per cell. The answer is one token, so no intermediate step is ever written. Probes
are fitted on 125 sheets and tested on the other 125; chance is about 0.14 and the decoy
control (steps of another animal on the same sheet) scores 0.04 to 0.22 everywhere.

### Accuracy: best of the seven candidates is the answer

| lookups | variable-loop, 1 pass | 2 passes | 3 passes | 3 minus 1, 95% interval | released model, 3 passes |
|---|---|---|---|---|---|
| 2 | 0.93 | 0.96 | 0.94 | +0.01 | 0.98 |
| 3 | 0.35 | 0.50 | 0.58 | **+0.23** | 0.34 |
| 4 | 0.19 | 0.18 | 0.24 | +0.05 [-0.01, +0.11] | 0.22 |

By the stricter measure (the top word over the whole vocabulary) three lookups are 0.35,
0.30, 0.38 and four lookups 0.04, 0.00, 0.02: the model often continues with a word that
is not a candidate.

### Probes on the variable-loop checkpoint at 3 passes: which steps can be read, and where

| lookups | site | step 1 | step 2 | step 3 | step 4 |
|---|---|---|---|---|---|
| 2 | before the loop | 0.09 | 0.16 (answer) | | |
| 2 | end of pass 1 | 0.69 | 0.87 (answer) | | |
| 2 | end of pass 3 | 0.79 | 0.97 (answer) | | |
| 2 | messages into the final pass | **0.85** | 0.94 (answer) | | |
| 3 | end of pass 1 | 0.44 | 0.15 | 0.34 (answer) | |
| 3 | end of pass 2 | 0.40 | 0.23 | 0.31 (answer) | |
| 3 | end of pass 3 | 0.42 | 0.34 | 0.48 (answer) | |
| 3 | last layer | 0.21 | **0.42** | 0.58 (answer) | |
| 4 | end of pass 1 | 0.58 | 0.16 | 0.07 | 0.16 (answer) |
| 4 | end of pass 3 | 0.19 | 0.18 | 0.12 | 0.18 (answer) |

Full table for every model, pass count and site: `results/partA-summary.json`.

### What we see

- **A step that is never written can be read from inside.** On two lookups the model
  answers in one token and the intermediate colour is decodable from the last position at
  0.69 after one pass and 0.85 from the messages into the final pass, against about 0.10
  for both controls. This repeats, with a trained probe and controls, what lens work on
  looped models has shown qualitatively.
- **On two lookups the loop is not needed.** One pass already scores 0.93.
- **On three lookups extra passes do real unwritten work, and it shows inside.** Accuracy
  goes from 0.35 to 0.58. The second step is at chance at the end of pass 1 (0.15), 0.23
  after pass 2, 0.34 after pass 3 and 0.42 at the last layer. The later pass is where the
  later step appears. This is the case the GLM post describes: reasoning done in loop
  depth with nothing in the output.
- **The probe recovers only part of it.** Where the model is right 58% of the time, the
  probe reads the second step 34 to 42% of the time. With 125 training sheets, one token
  position and a linear readout these are lower bounds, but as measured a monitor reading
  this state would miss most of the hidden steps.
- **Four lookups fail silently at every pass count,** and the first step, readable at 0.58
  after pass 1, fades to 0.19 by pass 3. More passes did not carry the chain further.
- **The released model computes the first step later:** on two lookups it is 0.24 at the
  end of pass 1 and 0.66 at the end of pass 2.

### Predictions

1. **Held for three lookups (+23 points), failed for four (+5, interval includes zero).**
   Scored on the best candidate, as fixed before the run.
2. **Not tested.** Part B was not run.
3. **Not tested.** In the pilot the model wrote its steps at every pass count, but that
   scorer was faulty.
4. **Held for two lookups, failed for three and four.** Probe 0.79 to 0.85 on two lookups;
   at most 0.42 on three; chance on four. Controls stayed under 0.30 everywhere.
5. **Not tested** by this design.
6. **Not tested.** Ouro not run.
7. **Failed at the site named.** At the end of the final pass on three lookups the first
   step (0.42) is read better than the second (0.34). At the last layer it is the other
   way round (0.21 and 0.42).
8. **Failed.** Among held-out three-lookup items right only at 3 passes, the first step
   is decoded at the end of pass 1 in 13 of 38.

### What this does not show

- **Whether looping shortens written reasoning.** That was the opening question and it is
  unanswered. Generation on this model ran at about 30 seconds per answer, and in the
  pilot it wrote about 200 tokens whatever the pass count.
- That transcript plus probe makes up for anything. Transcript and probe were never
  measured on the same runs.
- That the probe's limit is the state's limit. Small training set, one position, linear.
- Anything beyond one invented task, in a prompt format chosen after two others failed,
  on one model family.
- The model's one-token answers are off its usual path: it is a model that always thinks
  first. Its failures on longer chains may be failures of this format.

### What we got wrong

- The pilot's scorer counted copied prompt text as written steps.
- The first prompt format ("X: Y" rows, "follow the tables in order") was unreliable even
  on two lookups. $0.30 went on a full launch before a 12-item check would have shown it.
- The cost estimate assumed generation several times faster than it is.
- We assumed thinking could be switched off with the effort dial. It cannot.

## After independent review of part A (2026-11-23)

A separate model recounted the saved predictions and refitted the key probes on CPU. Its
counts match ours. It rated the result 6 of 10 on its own and asked for these changes,
which we re-ran (`analyse.py`):

- **The three-lookup gain depends on what counts as correct.** Best of seven candidates:
  87 of 250 at one pass, 144 of 250 at three. The model's actual top word: 87, 76 and 94
  of 250 at one, two and three passes, a gain of 2.8 points with an interval of about -4
  to +10. Extra passes improved the *ranking* of the right candidate. Evidence that they
  improved what the model would actually say is weak. Our first summary led with the
  first number only.
- **How much a monitor can read should be counted on successful runs.** On the 73
  held-out three-lookup sheets where the right answer ranked first among candidates at 3
  passes, the last-layer probe recovered the second step on 39 (53%). Setting overall
  probe accuracy beside overall answer accuracy, as we did, does not give that.
- **Four lookups: the chain starts and is then lost.** The first step is read on 72 of
  125 held-out sheets after pass one. "Chance" described only the later steps.
- **Two lookups stand:** 106 of 125 from the messages into the final pass. Since one pass
  already answers 233 of 250, this shows the step is readable, not that looping was
  needed to compute it.

## Ouro: plan

Written after a 12-sheet pilot. **Committed a few minutes after the full run was launched**
(an edit to this file failed just before launch); no full-run output had been read.

**Question.** The one the Inkling runs left open: with thinking on, does more looping let
the model write less while still solving the problem?

**Pilot** (`ouro.py --pilot`, 12 two-lookup sheets, `ByteDance/Ouro-2.6B-Thinking`, greedy
decoding). Every answer finished and was correct at both 2 and 4 loops. Mean thinking
length was 183 tokens at 2 loops and about 171 at 4, with single sheets moving either way
by up to 100 tokens. The model reasons in plain sentences and states each step. It
sometimes lists a whole table, so "the step's word appears" can be true through that
list; samples were read, and a stricter count is added below. The model card's
recommended library version crashes in the model's own cache code; we use a later one.

**Full run.** The same fact sheets at 2 and 4 loops: 40 sheets of two lookups and 40 of
three, up to 512 new tokens.

**Reported for every cell:** accuracy, share finished, thinking tokens, intermediate
steps named, and steps named in a sentence that also holds the previous step's word
(stricter). All sheets first; sheets solved at both loop counts second.

**Primary comparison:** thinking tokens at 4 loops minus at 2 loops, paired by sheet,
with an interval from resampling sheets.

**Predictions.** Prediction 6 above stands, restated for this design: thinking length
differs by less than 10% between 2 and 4 loops. Added:

9. Accuracy on three lookups is at least 10 points higher at 4 loops than at 2.
10. Steps named (stricter count) differ by less than 10 points between loop counts.

**What would count.** Shorter thinking and fewer named steps at 4 loops, with accuracy
and completion kept, is the effect the GLM post describes. Shorter output because the
model fails or stops is a different thing and will be reported as such.

**Limits known now.** Ouro was trained at 4 loops, so 2 loops is off its training
setting. 40 sheets per cell resolves a difference of roughly 10% in length, not less.
One model, one invented task.

**Cost.** Pilot and two failed starts about $0.15. Full run about 35 GPU minutes on a
small GPU: about $0.50.

## Ouro: result

`ouro.py`, then `ouro_analyse.py`. `ByteDance/Ouro-2.6B-Thinking`, thinking on, greedy,
the same 40 fact sheets per chain length at 2 and at 4 loops.

| lookups | loops | correct | finished | mean thinking tokens | steps named | steps named, stricter |
|---|---|---|---|---|---|---|
| 2 | 2 | 40/40 | 40/40 | 162 | 40/40 | 28/40 |
| 2 | 4 | 40/40 | 40/40 | 185 | 40/40 | 34/40 |
| 3 | 2 | 40/40 | 40/40 | 212 | 80/80 | 77/80 |
| 3 | 4 | 40/40 | 40/40 | 202 | 80/80 | 80/80 |

**Primary comparison,** thinking tokens at 4 loops minus at 2, paired by sheet:
two lookups **+23 tokens [+8, +38]** (14% longer); three lookups **-11 tokens [-28, +4]**
(5% shorter, interval includes zero). Every sheet was solved at both loop counts, so the
"solved at both" comparison is the same.

### What we see

- **More looping did not make this model write less.** Thinking was a little longer on
  two lookups and about the same on three. Every intermediate step was written at both
  loop counts.
- **This is a weak test of the GLM claim.** The task was at ceiling: 2 loops already
  solves everything, so there was no extra work for the loop to absorb. The GLM figure
  came from benchmarks hard enough for looping to change accuracy.
- **The stricter step count undercounts.** It removes copies of the fact sheet's rows, and
  the model's own statement of the first step ("the yak is silver") is word for word a
  row. Reading samples, the step is stated in those cases too. The 28 against 34 on two
  lookups should not be read as a difference in legibility.
- **Batched and single-prompt outputs differ.** Greedy decoding gave the same text for
  the first sheet at 2 loops on two lookups and different text in the other three cells
  (for example 257 against 229 tokens). Padding changes the numbers slightly and the
  wording drifts. Both loop counts were run the same way, but single-sheet lengths carry
  this noise.

### Predictions

6. **Failed on two lookups (+14%), held on three (-5%).** We predicted under 10% either
   way.
9. **Failed.** Accuracy was 40 of 40 at both loop counts; no gain was possible.
10. **Held on three lookups (+4 points), failed narrowly on two (+15 points)**, though the
    two-lookup figure is the undercount described above.

### What this does not show

- That looping never moves reasoning out of the transcript. It would need a task where
  2 loops fail and 4 succeed, or where the written reasoning is long enough to shorten.
  Neither held here.
- Anything about models trained to loop from a pretrained base at scale, which is what
  GLM-5.4 is.
- Ouro at 2 loops is off its training setting, and still solved everything.

### What went wrong on the way

Three launches were stopped or failed before the run that produced these numbers: no L4
GPU for 15 minutes (not billed), the card's library version crashing in the model's cache
code, and one-at-a-time generation at 3 to 6 tokens a second. A batch of 20 three-lookup
prompts at 4 loops ran out of memory; that cell was rerun in batches of 8.

## Where this leaves the question

- **Silent reasoning in the loop is real on Inkling and partly readable.** That is the
  Inkling half, with the caveats in the review section.
- **We did not observe written reasoning shrinking with loop count.** One model, a task
  it finds easy.
- **The comparison we set out to make (transcript at low looping against transcript plus
  probe at high looping, same runs) was not made.** It needs a model that both writes
  reasoning and benefits from looping on the same task, at a speed we can afford. Ouro on
  a harder task is the candidate.

## Harder task on Ouro: plan (written 2026-11-23 after a second independent review, before any run)

**Review of the Ouro result.** A separate model paired the saved outputs by fact sheet
and confirmed the counts (all 160 outputs correct; +23.1 thinking tokens on two lookups,
-10.5 on three). Two corrections, both accepted:

- **Ceiling accuracy does not stop a test of compression.** The model could have solved
  everything and written much less at 4 loops. It did not; on two lookups it wrote more.
  What the ceiling prevents is testing whether a *gain* in accuracy comes with less
  visible reasoning. "Weak test" above was too dismissive of the two-lookup result.
- **The three-lookup comparison changed batch size as well as loop count** (20 at 2
  loops, 8 at 4, after the out-of-memory failure). Batching changes greedy output, so
  the small shortening there should not be put down to loop count.

**Question now.** Is there a chain length where 4 loops solve more than 2, and if so, do
the steps stay written?

**Screen** (`ouro.py --mode screen`): chains of 4, 5, 6 and 7 lookups, 16 fact sheets
each, 2 and 4 loops, same batch size (8) and token cap (1,000) at both.

**Choosing the length, on accuracy and completion only:** the shortest chain where 4
loops are correct on at least 4 more of the 16 sheets than 2 loops, with at least 13 of
16 finished at both. Reasoning length and step mentions on screen sheets are not looked
at. If no length qualifies, we report that and stop.

**Then** (`ouro.py --mode hard --chain N`): 40 fresh fact sheets at that length, same
settings. Reported: accuracy, completion, thinking tokens, steps named (both counts),
on all sheets and on sheets solved at both loop counts.

**Decision rule, from the review.** If 4 loops improve accuracy and still spell out the
steps, report it and stop this branch. If they improve accuracy and omit a specific step
more often on sheets solved at both, go on to editing the table and probing.

**Predictions.**

11. Accuracy at 2 loops falls below 13 of 16 by six lookups.
12. No chain length meets the selection rule: 4 loops will not beat 2 by 4 sheets
    anywhere. (Olszowiec et al. report that reasoning-tuned Ouro does not improve with
    more iterations.)

**Cost cap: $1** for screen and fresh run together. Time guards of 25 and 18 minutes.

### Screen result and the frozen choice (before the fresh run)

`ouro.py --mode screen`, two launches (the first hit its time guard), 39.5 GPU minutes.
Correct of 16, with all 16 finished unless noted:

| lookups | 2 loops | 4 loops |
|---|---|---|
| 4 | 15 | 16 |
| 5 | 16 | 16 |
| 6 | 16 | 16 |
| 7 | 12 (14 finished) | 16 |

- **Seven lookups meets the rule, exactly at its threshold** (4 more sheets correct, at
  least 13 finished at both). With 16 sheets this could be chance; the fresh run is the
  test.
- **A scorer bug was found and fixed here.** Five-lookup answers are city names, which
  the scorer lower-cased on one side only, so every answer scored wrong. The saved text
  was rescored. The earlier runs had no capitalised answers and are unaffected.
- **What we saw of the screen's reasoning.** The job's progress lines print mean thinking
  tokens next to accuracy, and we saw them for four lookups before filtering them out. They
  played no part in the choice, which follows the rule above.
- Predictions 11 and 12 both **failed**: 2 loops is still 16 of 16 at six lookups, and
  one length met the rule.

**Frozen:** seven lookups, fresh fact sheets (seed 8), 2 and 4 loops, batch 8, cap 1,000.
**Changed from the plan:** 24 fresh sheets, not 40. The screen cost about $0.72 of the
$1 cap, and 24 sheets is what the rest buys.
