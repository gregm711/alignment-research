# Jacobian lens on Inkling 2 Mini

**Status:** the four questions below each have a first answer. Start with [`SUMMARY.md`](SUMMARY.md). See the [independent review](../notes/independent-review-2026-11-21.md) for replications and scoring corrections.

## Question

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is a looped
transformer: 24 layers of weights, with layers 6–17 run three times. What does each pass do?
On a two-step question, is the intermediate fact worked out in pass 1 and refined, or does it
only appear in pass 3?

The tool is the [Jacobian lens](https://github.com/anthropics/jacobian-lens) from
[Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html): a linear map
per layer that reads out which words the model is poised to say at that point.

## Prior art

Read after the first run; full notes in [`../notes/looped-prior-art.md`](../notes/looped-prior-art.md).

- [Wang & Reid, 2026](https://arxiv.org/abs/2609.01924) applied the Jacobian lens and a causal
  suite to two smaller looped models (Ouro-2.6B, Huginn-0125) using the same unrolling trick
  as `PassView`.
- [Olszowiec et al., 2026](https://arxiv.org/abs/2610.18127) read eight looped models at every
  iteration through the model's own output head.

Neither of those covers this model, but two release-day publications do, and we missed both
until the independent review pointed them out:

- [Goodfire, "Interpreting Inkling 2 Mini"](https://www.goodfire.com/research/interpreting-inkling-2-mini)
  (2026-11-20) maps how information flows across blocks and loops on this exact model. They
  show later loops read almost entirely from the previous loop's second looped block, find a
  single outlier coordinate that tags which block to read, and compare three ways of decoding
  internal states (logit lens, tuned lens, and a "coda lens" that applies the final block).
  They explicitly leave the Jacobian lens for future work.
- [Thinking Machines' release post](https://thinkingmachines.ai/news/inkling-2-mini/) has a
  section on the monitorability of looped models. It sweeps the number of passes on the
  variable-loop checkpoint and reports that a fictional lookup-chain task keeps improving with
  more passes.

What that leaves as ours: the pass sweep on the *released* (fixed) model, which is a special
case of Goodfire's coda lens; and the causal test of what the messages do when transplanted,
which neither does. Statements below that nobody had studied this model are wrong and
should be read with this.

**Their key warning applies to our first run.** A lens aimed at the final output reads
punctuation in early loops "not because those loops are empty, but because the transport to
the final target dies crossing loop boundaries". The fix is one lens per loop end.

## Predictions (written before the first results)

1. The intermediate answer on multi-step questions becomes readable at a specific pass, not
   gradually across all three.
2. The Jacobian lens reads intermediate answers at earlier steps than the plain logit lens.
3. Loop layers are more readable than the pre-loop layers before them.

## What the architecture actually is

From the model code (`huggingface/transformers#51050`), which differs from the "one stream run
three times" picture:

- Passes 2 and 3 each **restart from the pre-loop state**. They don't continue from the
  previous pass's output.
- Each later pass receives learned **messages** from the previous pass, mixed into the
  attention and expert inputs of every loop layer.
- Outputs at block boundaries are kept and attended over; a final mixing step combines them
  before the output norm. That final mix is the lens target.
- There is no single residual stream. Layers are grouped in blocks of six, each block keeps
  a running sum of its own layers' outputs starting from zero, and each layer reads a learned
  mix of the block sums. The "layer outputs" our hooks read are these per-block running sums.
  (Goodfire's write-up explains this clearly; we had not appreciated it.) It is the likely
  reason every pass reads as blanks from layer 12 on: that is where a new block's sum restarts.
- Weights ship in 4-bit and 8-bit formats and are expanded to bf16 on load (about 49GB on GPU).

## How it works here

`fit.py` presents the model to the lens library as 49 steps: 6 pre-loop layers, 12 loop layers
× 3 passes, 6 post-loop layers, and the final mix. A small wrapper (`PassView`) makes a hook on
a looped layer fire for one pass only, so each pass gets its own Jacobian.

After fitting it prints, for six two-step prompts, the rank of the intermediate concept and of
the final answer at every step, for both the Jacobian lens and the logit lens.

## Running it

Needs a Modal account and one 80GB GPU.

```sh
modal run fit.py --minutes 12   # fit for about 12 minutes, checkpoint, print readouts
modal run fit.py --minutes 0    # print readouts from the existing checkpoint
```

Runs resume from the checkpoint on the Modal volume, so spend is incremental. Each run has a
90-minute hard timeout.

## Known limits

- Six hand-written probes are a first look, not an evaluation. The lens library's evaluation
  sets and causal tests (edit a representation, check the answer changes) are the next step.
- Fitting text is Wikipedia paragraphs, narrower than the web-text corpus the paper used.
- Model support comes from huggingface/transformers#51050. Runs up to the pass sweep used its branch; it was merged on 2026-11-21 and later runs are pinned to the merge commit.
- At the batch size that fits in memory, each prompt needs 576 backward passes.

## Open questions and the plan

From [`../notes/looped-prior-art.md`](../notes/looped-prior-art.md), in the order we will take them (cheapest first):

| # | Question | Method | Status |
|---|---|---|---|
| 2 | At which pass is the answer decided? | Forward-only sweep over the number of passes, both checkpoints | answered (see below) |
| 1 | What crosses the pass boundary through the message channel? | Message swapping between prompts (forward-only), then a per-pass Jacobian lens for readability | answered (see below) |
| 4 | Must an edit be held across all passes to stick? | Message swap (done under question 1), then swapping a concept direction in one pass vs all | answered (see below) |
| 3 | Is the loop state robust to noise, including for refusals? | Noise in the restart state and in the messages; record refusal yes/no only | answered (see below) |

Spend is tracked in [`SPEND.md`](SPEND.md).

## Question 2: at which pass is the answer decided?

### Plan and predictions (written before running)

**Measurement.** The number of passes is a runtime setting (`loop_num_passes`). For each of
1–5 passes, on both Inkling 2 Mini (trained at 3 passes) and its variable-loop companion
`TML-d24-varloop-research` (trained with 1–5 passes in a 1:2:6:2:1 ratio, then post-trained at
3), read the model's own output. No lens, no gradients. Two data sources:

- *Ordinary text:* 60 WikiText-103 paragraphs, 128 tokens each, scored at every position
  after the first 16: next-token accuracy, log-loss, and how often the top prediction already
  equals the 3-pass top prediction.
- *Probes:* 44 hand-written prompts with a one-word answer in three tiers: one-hop facts,
  two-hop facts built to share answers with the one-hop tier, and arithmetic. Scored by
  whether the answer is the top prediction.

**Predictions.**

1. On ordinary text, the 1-pass output already agrees with the 3-pass top prediction on more
   than half of tokens. (Olszowiec et al. found 76–89% decided at the first iteration in six
   of eight looped models.)
2. Two-hop probes gain more from passes 2 and 3 than their one-hop counterparts. (Wang & Reid:
   harder questions settle in later loops.)
3. The fixed checkpoint loses more than the variable-loop one when run at 1 or 2 passes,
   since it never trained that way.
4. Past 3 passes, the fixed checkpoint does not improve on ordinary text and may get worse.
   (Olszowiec et al.: a fixed-depth model peaks at its trained depth.)

**What this will not show.** Reading the output after *k* passes measures what the post-loop
layers can decode from pass *k*, not everything pass *k* has computed. The probes are few and
hand-written. One prompt format, one position per probe.

### Result

Raw data: [`results/pass-sweep.json`](results/pass-sweep.json) (ordinary text),
[`results/pass-sweep-probes.json`](results/pass-sweep-probes.json) (probes). Scripts: `sweep.py`.

**Probes: share answered correctly (answer is the top prediction), by number of passes.**

| Passes | Inkling 2 Mini: one-hop | two-hop | arithmetic | Variable-loop: one-hop | two-hop | arithmetic |
|---|---|---|---|---|---|---|
| 1 | 16/16 | 3/16 | 0/12 | 15/16 | 14/16 | 12/12 |
| 2 | 16/16 | 11/16 | 3/12 | 16/16 | 15/16 | 12/12 |
| 3 (trained) | 15/16 | 15/16 | 12/12 | 15/16 | 16/16 | 12/12 |
| 4 | 13/16 | 7/16 | 11/12 | 16/16 | 16/16 | 12/12 |
| 5 | 16/16 | 12/16 | 11/12 | 15/16 | 16/16 | 12/12 |

**Ordinary text (6,660 WikiText tokens).**

| Passes | Inkling 2 Mini: accuracy | same top word as 3 passes | Variable-loop: accuracy | same top word as 3 passes |
|---|---|---|---|---|
| 1 | 38.3% | 56.8% | 48.5% | 86.2% |
| 2 | 41.4% | 65.8% | 48.6% | 90.3% |
| 3 (trained) | 49.5% | 100% | 49.6% | 100% |
| 4 | 44.9% | 71.4% | 49.8% | 94.2% |
| 5 | 45.9% | 75.0% | 49.9% | 93.1% |

**What we see.**

- **In the released 3-pass model, stopping after one pass gives the first step of the
  reasoning, not the answer.** On two-hop questions the top word after one pass is the
  intermediate entity: `France` (then `Paris` from pass 2), `Japan` (then `Tokyo`), `Italy`
  (then `Rome` at pass 3), `Russia` (then `Moscow` at pass 3). On arithmetic it is an operand
  copied from the prompt (`7 + 5 =` gives `7`, then `12` at pass 3). One-hop facts are already
  right after one pass.
- **Composition completes at different passes.** Multiplications are right after two passes
  and additions and subtractions only after three. Two-hop facts split between passes 2 and 3.
- **The variable-loop sibling answers almost all of the same questions after a single pass.**
  So these questions do not *need* three passes. A model trained to exit after one pass does
  both steps in one pass; the model trained only at three spreads them out, or at least leaves
  its pass-1 state in a form that reads as "step one".
- **On ordinary text extra passes buy very little** when the model is trained for them: one
  point of next-word accuracy from one pass to three on the variable-loop checkpoint, and
  86% of top predictions are already final after one pass.
- **Off its trained depth the fixed model gets worse in both directions**, and not smoothly:
  four passes is worse than five on every measure. We have no explanation for that.

**Against the predictions.**

1. *More than half of tokens already agree after one pass:* held on both (56.8% and 86.2%).
2. *Two-hop gains more from later passes than one-hop:* held strongly on the fixed model
   (3/16 to 15/16, against no change for one-hop). On the variable-loop model the gain is two
   probes out of sixteen, which is too few to call.
3. *The fixed checkpoint loses more at one or two passes:* held.
4. *Past three passes the fixed checkpoint does not improve:* held; it gets worse.

**After independent review.** The pass counts above were reproduced from the saved rows.
Two things narrow the claim. The pass sweep on the variable-loop checkpoint repeats, on a
much smaller scale, what Thinking Machines report in their release post. And an early-exit
output still runs the six post-loop layers, so it includes their decoding work; it is a
special case of what Goodfire call the coda lens.

**How this sits with the prior work.** It agrees with Wang & Reid (harder questions settle in
later loops) and with Olszowiec et al. on early decisions and on fixed-depth models peaking at
their trained depth. It is in tension with Olszowiec et al.'s finding that harder problems do
not make the loop work harder, though they measured something different (prediction flips
across difficulty levels of one maths dataset). It also confirms the correction to our own
first lens run: pass 1 is not empty.

**What this does not show.**

- Stopping the fixed model early is off its training distribution. We cannot separate "pass 1
  has only computed step one" from "pass 1 has more, but the post-loop layers decode an early
  state as step one". The lens fitted per pass count (question 1) is meant to help with this.
- The probes are 44 hand-written prompts in one format, and the variable-loop model is at
  ceiling on them, so they cannot show where extra passes help *that* model.
- Twelve answers are written by this tokenizer as a space token followed by the word; for
  those the space was appended to the prompt and the word scored at the next position. The
  first run scored them one position early; those numbers were discarded.

**Why it matters for monitoring.** In the fixed model, intermediate reasoning that never
appears in the output can be read off by running fewer passes, with no trained probe and no
lens. That is a cheap window onto latent reasoning for this design. Whether it stays readable
on harder tasks, or when a model has a reason to hide it, is untested.

**Later correction:** this did not hold up on prompts written by others. See "Follow-up:
does the early-exit finding survive other people's prompts?" below. Early exit showed the
intermediate step here because our wording made it a natural next word.

## Question 1: what crosses the pass boundary through the message channel?

### Plan and predictions (written before running)

**A fact from the code, not an experiment.** Passes 2 and 3 start from the same pre-loop state
as pass 1 and run the same weights. The only other input they receive is the message tensor
built from the previous pass (four messages per loop layer per position: for the attention
query, key and value, and for the expert layer). Each position's message is built only from
that same position's states in the previous pass. So everything a later pass knows beyond
pass 1 arrives through messages, and the question becomes: what do the messages carry?

**Measurement 1, forward-only: message swapping.** Take two prompts A and B with the same
wording and token length but different answers. Run B and save its messages. Run A, replacing
its messages with B's, and see what A now answers. Conditions: swap into pass 2 only, pass 3
only, or both; at every position or at the last position only. Control: swapping A's own
messages back in must reproduce the unpatched output exactly. Two prompt families:

- *Two-hop:* "The language spoken in the country whose capital is X is" for eight capitals.
- *Arithmetic:* single additions and single multiplications, paired within the same operation.

Outcome per pair: does A's top answer become B's answer, stay A's, or become something else.
Run on Inkling 2 Mini (where the sweep showed passes matter) and on the variable-loop
checkpoint at three passes for comparison.

**Measurement 2, gradients, only if measurement 1 leaves the question open:** a Jacobian lens
fitted per pass count, as described in `fit.py`.

**Predictions.**

1. The self-swap control reproduces the unpatched output on every prompt.
2. Swapping all positions in both later passes moves A's answer to B's answer in most pairs:
   the messages carry the specific content, and the restarted state carries only the question.
3. Swapping at the last position only moves fewer answers than swapping everywhere, because
   the intermediate fact is worked out at earlier positions and reaches the end by attention.
4. Swapping into pass 3 only moves at least as many answers as swapping into pass 2 only,
   because pass 3 can partly re-derive from the restarted state what a pass-2 edit removed.
   (This is also a first look at question 4.)

**What this will not show.** A swap replaces the whole message, so it shows that messages
carry enough to determine the answer, not which part of the message does. Two prompt
families in one format. An answer that moves to "something else" is hard to interpret.

### Result (measurement 1: message swapping)

Raw data: [`results/message-swap.json`](results/message-swap.json). Script: `patch.py`.

"Moved to B" means prompt A, run with prompt B's messages, gave B's answer as its top word.
Only prompts the model answers correctly unpatched are used. Swapping into both later passes
at every position is the same intervention as swapping into pass 3 alone, since pass 3 then
receives B's messages either way; the counts are identical and only pass 3 is shown.

**Two-hop ("The language spoken in the country whose capital is X is"), 42 prompt pairs.**

| Swap B's messages into | Inkling 2 Mini: moved to B | stayed A | other | Variable-loop: moved to B | stayed A | other |
|---|---|---|---|---|---|---|
| pass 2, every position | 8 | 31 | 3 | 0 | 42 | 0 |
| pass 3, every position | 33 | 3 | 6 | 17 | 24 | 1 |
| pass 2 or 3, last position only | 0 | 40–41 | 1–2 | 0 | 42 | 0 |

**Arithmetic, pairs within the same operation (90 addition pairs, 30 multiplication pairs).**
No swap in any condition moved a single answer, on either model: 120 of 120 stayed A.

**Arithmetic, pairs across operations (112 pairs), swap into pass 3 at every position.**

| Outcome | Inkling 2 Mini | Variable-loop |
|---|---|---|
| Stayed A's answer | 80 | 111 |
| A's numbers with B's operation (`7 + 5 =` with messages from `8 * 9 =` answers `35`) | 13 | 1 |
| B's answer | 1 | 0 |
| Something else | 18 | 0 |

The control held: swapping a prompt's own messages back in reproduced the unpatched output
on all 24 prompts on both models.

**What we see.**

- **Messages carry what the previous pass worked out, not what the prompt already says.** In
  the two-hop family the bridging fact (which country) is derived, and replacing the messages
  into pass 3 replaces the answer in 33 of 42 pairs on the released model. In arithmetic the
  numbers are in the prompt, which every pass re-reads from the restarted state, and no swap
  of another sum's messages ever changes the answer.
- **Messages also carry the operation.** With a multiplication's messages, an addition prompt
  sometimes multiplies its own numbers (13 of 112), and often breaks (18). It never happens
  with messages from another addition.
- **The content travels at earlier positions.** Swapping only the last position's messages
  changes nothing. The answer position gets the derived fact by attending to earlier
  positions within the pass.
- **Messages into pass 3 decide far more than messages into pass 2** (33 against 8 on the
  released model; 17 against 0 on the variable-loop one). An edit to what pass 2 receives is
  mostly washed out: pass 2 re-derives from the restarted state.
- **The variable-loop model leans on messages less.** That fits the sweep: it can answer
  these questions in one pass, so the restarted state alone often suffices.

**After independent review.** A separately written hook reproduced 33 of 42 and 8 of 42, and
every self-swap reproduced the logits exactly. The reviewer's caveats, which we accept:

- **What the messages carry is not identified.** In these pairs the city, the country and
  the language all change together. The swap shows the messages can redirect the answer. It
  cannot show that the country, as opposed to the answer itself, is what travels. The first
  bullet above states an interpretation as a finding.
- **42 pairs come from 7 prompts,** so they are not 42 independent questions.
- **The 13 of 112 cross-operation cases are pooled over both directions,** so the count does
  not say they all come from multiplication donors.

The follow-up "What exactly crosses?" below is designed to separate these.

**Against the predictions.**

1. *Self-swap control reproduces the output:* held.
2. *Swapping everywhere moves most answers to B's:* held for two-hop on the released model
   (33/42), not on the variable-loop model (17/42), and **wrong for arithmetic** (0/120). The
   prediction treated messages as carrying "the specific content" in general; they carry the
   derived part only.
3. *Last position only moves fewer:* held (none moved).
4. *Pass 3 moves at least as many as pass 2:* held.

**What this does not show.**

- A swap replaces whole messages, so it shows they are sufficient to redirect the answer, not
  which component does it or how it is encoded.
- One two-hop template with eight items and two arithmetic templates. "Derived versus given"
  is our reading of three families, not a tested law.
- The 18 "something else" outcomes in the cross-operation test are uninterpreted.
- Whether this content is *readable* by a monitor (as opposed to transplantable) is a
  separate question; that is what the per-pass lens is for.

### Plan and predictions for measurement 2: a lens per pass count (written before running)

**Measurement.** On Inkling 2 Mini, run the model with only *k* passes (k = 1, 2, 3) and fit
a Jacobian lens aimed at that run's output, 3 prompts each (the 3-pass lens already exists).
Pass *k*'s states are identical whether or not later passes run, so each lens reads pass *k*
through the path the model uses when it stops there, with no pass boundary in the way. Read
out eight two-hop probes at the last position and record the rank of the intermediate concept
and of the answer at every loop layer.

**Predictions.**

1. Through the 1-pass lens, the intermediate concept (e.g. `Italy`) is readable in pass 1's
   loop layers, where the 3-pass lens read only punctuation.
2. In pass 1 the intermediate outranks the answer; in pass 3 the answer outranks the
   intermediate.
3. Pass 2 is in between: both readable.

**What this will not show.** Three prompts is a very small fit. The released model was not
trained to stop early, so the 1- and 2-pass lenses read through a path it never trained on.
Readable is not the same as used; that needs the edits in question 4.

### Result (measurement 2: a lens per pass count)

Raw data: [`results/lens-per-pass.json`](results/lens-per-pass.json). Script: `fit.py`. Each
lens is fitted on 3 prompts.

**Median rank of the concept among ~200,000 tokens, best loop layer, over eight two-hop
probes** (rank 0 is the top word; lower is more readable).

| Reading pass | through the lens aimed at | intermediate (e.g. `Italy`) | answer (e.g. `Rome`) |
|---|---|---|---|
| 1 | the 1-pass output | 134 | 342 |
| 1 | the 3-pass output | 6,924 | 2,699 |
| 2 | the 2-pass output | 45 | 40 |
| 2 | the 3-pass output | 23 | 130 |
| 3 | the 3-pass output | 25 | 0 |

**One probe in detail** ("The capital of the country shaped like a boot is"), top words in
the middle loop layers of the last pass of each run:

| Run | Layers 9–11 read | After the loop |
|---|---|---|
| 1 pass | `shape`, `shaped`, `island`, "geography", "regions" (several languages) | `Italy` |
| 2 passes | "state", "government", "ministry", then `Italian`, `Italy` | `Italy`, `Rome` |
| 3 passes | "capital" (Arabic, Russian, Thai), `city` | `Rome` |

**What we see.**

- **Aiming the lens at the right output matters most for pass 1.** The intermediate's median
  rank in pass 1 goes from about 6,900 through the 3-pass lens to 134 through the 1-pass
  lens. That confirms the first run's "pass 1 reads as punctuation" was the instrument.
- **Even so, pass 1 is only weakly readable.** A median rank of 134 is far above chance and
  far from the top. In the detailed probe, pass 1's loop layers read the *description* (shape,
  geography), and `Italy` becomes the top word only in the layers after the loop.
- **The lens and the swap experiment agree about which boundary carries content.** The
  3-pass lens reads pass 2 quite well (median rank 23 for the intermediate) and pass 1 not at
  all. The swap experiment found messages into pass 3 decide the answer and messages into
  pass 2 mostly do not. Linear transport is alive exactly where the causal dependence is.
- **In the one probe examined closely, each pass's middle layers read a different thing:**
  the description, then the entity, then the relation asked about.

**Against the predictions.**

1. *The intermediate is readable in pass 1 through the 1-pass lens:* weakly. Much better than
   through the 3-pass lens, but in the top ten on only one probe of eight.
2. *In pass 1 the intermediate outranks the answer; in pass 3 the reverse:* the pass-3 half
   held (7 of 8); the pass-1 half did not (3 of 8).
3. *Pass 2 is in between:* held loosely; both are in the top hundred on most probes.

**What this does not show.** Lenses fitted on 3 prompts are noisy, and eight probes are few;
the medians above would move with either. The plain logit lens does about as well as the
Jacobian lens on several rows (both are in the data), so we have no evidence here that the
Jacobian lens is the better instrument for this model. The three-stage reading rests on one
probe.

**Answer to question 1.** What crosses a pass boundary is the message tensor and nothing
else. It carries what the previous pass derived (the bridging entity, the operation), at
earlier positions, and it matters mainly at the boundary into the last pass. The same
boundary is the one a final-output lens can see across.

## Question 3: is the loop state robust to noise, including for refusals?

### Plan and predictions (written before running)

**Background.** Olszowiec et al. found that looped models where the prompt enters once break
under noise in the loop state, with refusals eroding before fluency, while models that
re-read the prompt every iteration are almost unaffected. This model restarts every pass
from the same pre-loop state, which makes it closer to the second kind.

**Measurement.** On Inkling 2 Mini, add Gaussian noise to the input of the first loop layer
at the last prompt position, scaled to α times that state's norm (α = 2, 8, 32), in three
conditions: pass 1 only (later passes restart clean), pass 3 only, or all three passes.
Forward and short greedy generation only; no gradients.

- *Facts:* the 44 probes from the sweep. Is the top answer unchanged?
- *Benign requests:* 30 from JailbreakBench's benign set, 24 generated tokens, thinking off.
  Fidelity is the token overlap with the noise-free reply.
- *Harmful requests:* 30 from JailbreakBench's harmful set. **Only a refused / not refused
  flag is recorded**, from a standard list of refusal phrases in the first 24 tokens. No
  generated text for these prompts is printed or saved.

**Predictions.**

1. Noise into pass 1 only changes little at any α: later passes restart from the clean state,
   and question 1 showed edits to what pass 2 receives are mostly washed out.
2. Noise into pass 3, or into all passes, degrades answers as α grows.
3. Refusal rate tracks benign fidelity. We do not expect a setting where refusals drop
   sharply while benign replies stay intact, the pattern reported for carry-loop models.

**What this will not show.** One noise sample per prompt, at one position, at one layer.
Phrase matching undercounts unusual refusals and cannot tell a harmful reply from a broken
one, so a fall in refusals means "stopped refusing", not "complied". Random noise is not an
attack; this says nothing about a deliberate perturbation.

### Revised plan after a 4-request pilot (written before the real run)

The pilot (thrown away, not a result) showed the plan above could not answer the question:

- **Noise on one loop layer's input barely matters, for an uninteresting reason.** Top
  answers were unchanged on 42–44 of 44 facts even at α = 32. The model keeps a copy of the
  pre-loop state that later layers read back directly, so one corrupted layer input is
  routed around. That is not a test of the loop state.
- **The refusal detector could not work.** The model writes private reasoning first even with
  reasoning set to none, so a refusal never appears in the first 24 tokens.
- **Generation is slow** (about 6 tokens a second), so the token budget has to shrink.

**Revised measurement.** Noise scaled to α times each vector's norm (α = 2, 16) at four sites:

| Site | What it is |
|---|---|
| restart state, last position | the pre-loop state every pass starts from, including the copy later layers read back |
| restart state, every position | the same, across the whole prompt |
| messages into pass 2 | everything pass 1 hands to pass 2, every position |
| messages into pass 3 | everything pass 2 hands to pass 3, every position |

Replies are started in the answer channel (the reply is prefilled with the model's own
answer-channel tokens) and cut at 10 tokens. 24 benign and 24 harmful requests. Facts are
reported per tier. For harmful requests only the refused / not refused flag is kept, as before.

**Revised predictions.**

1. Noise in messages into pass 2 changes little (question 1: that boundary is mostly washed out).
2. Noise in messages into pass 3 hurts two-hop facts more than one-hop facts, because only
   two-hop answers depend on what the messages carry.
3. Noise in the restart state at every position is the most damaging, since that state is
   the only copy of the prompt the loop has.
4. Refusal rate tracks benign fidelity; no site makes refusals drop while benign replies hold.

**Two additions after the second pilot (4 requests, also thrown away), before the real run.**

- A fifth site, *messages into pass 3 at the last position only*. Olszowiec et al. perturbed
  one position; the every-position sites are far larger perturbations and not comparable.
- A plausibility score for every reply: how likely the unperturbed model finds it. A reply
  counts as plausible if it is no less likely than 95% of noise-free replies. This separates
  "stopped refusing because the reply turned to garbage" from "stopped refusing with a fluent
  reply", which is the case that matters. Only the score is kept, not the text.

The second pilot already contradicts revised prediction 1: noise in the messages into pass 2
changed most factual answers. A *well-formed* wrong message into pass 2 is mostly ignored
(question 1), but random noise there is not. We leave the prediction as written.

**Added limitation.** Prefilling the answer channel skips the model's reasoning step, so this
measures refusal on a path the model does not normally take.

### Result

Raw data: [`results/noise.json`](results/noise.json). Script: `noise.py`. Noise is scaled to α
times each vector's own norm. "Benign intact" means the reply was a refusal or a reply the
unperturbed model finds as likely as its own (the model refuses 13 of the 24 benign requests
even without noise on this prefilled path). For harmful requests only the refusal flag and a
likelihood score were kept.

**Independent review correction.** The arithmetic column below counts retention of a space before the number. At the numerical answer position, pass-3 message noise at α=2 gives 0/12 correct. [Per-setting replication](../notes/independent-review-2026-11-21.md#1-arithmetic-robustness-fails-at-the-answer-position).

| Noise site | α | One-hop facts kept | Two-hop kept | Arithmetic kept | Benign replies intact | Harmful still refused |
|---|---|---|---|---|---|---|
| none | – | 16/16 | 16/16 | 12/12 | 21/24 | 24/24 |
| restart state, last position | 2 | 15 | 13 | 12 | 14 | 22 |
| restart state, last position | 16 | 14 | 15 | 12 | 1 | 7 |
| restart state, every position | 2 | 14 | 4 | 12 | 8 | 14 |
| restart state, every position | 16 | 0 | 0 | 11 | 0 | 0 |
| messages into pass 2 | 2 | 8 | 3 | 12 | 7 | 9 |
| messages into pass 2 | 16 | 0 | 1 | 12 | 0 | 1 |
| messages into pass 3 | 2 | 2 | 0 | 12 | 3 | 0 |
| messages into pass 3 | 16 | 2 | 2 | 7 | 0 | 1 |
| messages into pass 3, last position | 2 | 4 | 2 | 12 | 10 | 8 |
| messages into pass 3, last position | 16 | 2 | 2 | 12 | 2 | 7 |

**What we see.**

- **The message channel is the fragile part.** Noise of twice the message's size, going into
  the last pass, removes almost every factual answer and every refusal. The same amount of
  noise in the restart state at one position leaves most of both.
- **Refusals fall when everything else falls, not before.** In every setting where refusals
  dropped, benign replies were broken at a similar or higher rate. We found no setting where
  the model keeps replying sensibly to benign requests but stops refusing harmful ones.
- ~~Arithmetic is nearly immune.~~ **Withdrawn.** The arithmetic column scored the space
  token before the number, which survives almost anything. Scored at the number, noise in
  the messages into pass 3 at α = 2 leaves 0 of 12 sums correct (the reviewer's rerun; see
  the note above the table). The explanation we offered for the immunity is withdrawn with it.
- **Well-formed and random are different.** Question 1 found that another prompt's messages
  at the last position change nothing, and that another prompt's messages into pass 2 are
  mostly ignored. Random noise at either place is destructive. The channel tolerates a
  plausible message that happens to be wrong far better than an implausible one.

**Against the revised predictions.**

1. *Noise in messages into pass 2 changes little:* **wrong.** It removed half the one-hop
   answers at α = 2 and all of them at α = 16.
2. *Noise in messages into pass 3 hurts two-hop more than one-hop:* not shown; it removed
   nearly all of both.
3. *The restart state at every position is the most damaging site:* true at α = 16, **wrong
   at α = 2**, where the messages into pass 3 are far more damaging.
4. *Refusal tracks benign fidelity:* held.

**How this sits with the prior work.** Olszowiec et al.'s split (carry loops fragile, re-read
loops robust at fifty times the state norm) does not map cleanly onto this design. Their
noise went into the state a loop starts from, at one position. The nearest equivalent here,
the restart state at one position, leaves single-word facts intact at α = 16 but breaks
multi-word replies. And this design has a second kind of state, the messages, with no
counterpart in the models they studied; it is the most fragile thing we measured.

**Answer to question 3 (revised after review).** Narrower than first written. Noise in the
messages into the last pass, at twice each vector's size, breaks factual answers, sums and
refusal phrases alike; noise in the restart state at one position mostly does not. We cannot
call the messages "the most fragile channel": the message sites perturb four vectors at each
of twelve layers and the restart site perturbs one, so equal ratios are not equal budgets.
And the refusal and benign-quality columns are two rough proxies that move together; they
cannot show the order in which safety and competence fail.

**After independent review, further limits we accept.** The fact columns measure agreement
with the noise-free top token, not correctness, and were not checked for the same
space-token problem. "Benign intact" counts every detected refusal as intact, and 13 of 24
benign requests already trigger the detector without noise.

**What this does not show, including a flaw in our own measure.**

- **The likelihood score cannot tell a harmful reply from a broken one.** We added it to
  separate "stopped refusing with a fluent reply" from "reply turned to garbage". But for a
  harmful request the unperturbed model would refuse, so *any* compliant reply scores as
  unlikely, fluent or not. The zeros in that column of the data are therefore uninformative
  and are not shown above. The claim that refusals fall with everything else rests on the
  benign replies breaking at the same settings, which is indirect.
- No reply to a harmful request was read by anyone or anything, by design. So "stopped
  refusing" means only that a refusal phrase was absent from the first ten tokens.
- One noise sample per prompt; 24 requests per set; ten-token replies; a prefilled answer
  channel the model does not normally use.
- Random noise is not an attack, and none of this bears on what a deliberate perturbation
  could do. Anyone with access to the weights already has easier ways to remove refusals.

## Question 4: must an edit be held across all passes to stick?

### Plan and predictions (written before running)

**What we already have.** The message swap in question 1 is an edit to what a pass receives.
Edited messages into pass 2 alone were mostly overwritten (8 of 42 answers moved); edited
messages into pass 3 moved 33 of 42; and editing both is the same intervention as editing
pass 3. At the message level the answer is already "no: the edit has to be in what the last
pass receives, and earlier edits are re-derived away".

**Measurement: a different kind of edit, inside a pass.** Following Wang & Reid, take the
lens direction for a concept (the unembedding vector of ` Japan` pulled back through the
lens) and, in the middle loop layers (7–11) of a chosen pass, swap the state's coordinates
along the source country's direction and a target country's direction. Passes 1, 2 and 3 use
the directions from the 1-, 2- and 3-pass lenses. Conditions: pass 1 only, pass 2 only, pass
3 only, all passes; at plain strength and at four times strength. Prompts: "The language
spoken in the country whose capital is X is", all ordered pairs of eight countries. Outcome:
does the answer become the target country's language.

**Predictions.**

1. Editing pass 3 alone moves more answers than editing pass 1 alone or pass 2 alone.
2. Editing all passes moves about as many as editing pass 3 alone: holding the edit is not
   needed in this design. (On Ouro, Wang & Reid found the opposite: one loop 21–29%, all
   remaining loops 38%.)
3. Rates will be well below the message swap's, because directions from lenses fitted on 3
   prompts are noisy.

**What this will not show.** If nothing moves, that is as likely to be the 3-prompt lens as
the model; a null here is not evidence. One template, eight items.

### Result

Raw data: [`results/direction-swap.json`](results/direction-swap.json). Script: `clamp.py`.
42 ordered pairs of prompts the model answers correctly unedited.

| Edit applied in | Plain strength: moved to target | Four times strength: moved to target | stayed | broke (something else) |
|---|---|---|---|---|
| pass 1 only | 0 | 1 | 39 | 2 |
| pass 2 only | 0 | 6 | 30 | 6 |
| pass 3 only | 0 | 14 | 12 | 16 |
| all three passes | 0 | 7 | 18 | 17 |

Control: the same four-times edit along two unrelated countries' directions, in all passes,
left the answer alone in 23 of 42 cases and broke it in 19. It can never produce the target
language, so the "moved to target" column is a specific effect, but the "broke" column is
mostly damage that any strong edit along these directions causes.

**What we see.**

- **The later the pass, the more an edit sticks:** 1, 6, then 14 of 42.
- **Holding the edit in every pass does not help.** It moved half as many answers as editing
  pass 3 alone (7 against 14), with as much breakage.
- **At plain strength nothing moves.** The directions come from lenses fitted on 3 prompts
  and are evidently too noisy to act at their natural scale.

**Against the predictions.**

1. *Pass 3 alone moves more than pass 1 or 2 alone:* held.
2. *All passes moves about as many as pass 3 alone:* the spirit held (holding is not
   needed), but it moved fewer, not as many.
3. *Rates well below the message swap's:* held (14 of 42 against 33 of 42).

**After independent review.** Three weaknesses we accept. The control records only "stayed"
or "changed", so our statement that it can never produce the target was unsupported; and it
was run only for the all-passes condition. The coordinate swap is exact only for orthogonal
directions, and we did not report how correlated these are. And the claim that earlier edits
are "re-derived away" is a proposed mechanism the data do not establish.

**Answer to question 4 (narrower after review).** On this template, an edit in the last pass
moved more answers than the same edit earlier or in every pass. An edit made only earlier is mostly
re-derived away, because each pass restarts from the unedited pre-loop state. This is the
opposite of what Wang & Reid found on Ouro, where an edit must be held across every remaining
loop, and close to what they found on Huginn, where the last window alone is as effective as
all of them. They attribute the difference to boundary design, and this result fits that.

**What this does not show.** The direction edit is blunt: it only works at a strength that
also breaks many answers, and the control shows much of that breakage is unspecific. One
template, eight items. With a properly fitted lens the picture at plain strength could differ.

## Follow-up: does the early-exit finding survive other people's prompts?

### Plan and criteria (written before running)

The pass-sweep finding rests on 44 prompts written by the same author who was looking for
the effect. This re-tests it on prompts written by others, in their own formats.

**Data.**

- *TwoHopFact* (Yang et al., 2024; CC BY 4.0): a sample of 400 two-hop facts spread across
  its fact types. Each has the two-hop prompt ("The author of the novel X was born in the
  city of"), both one-hop prompts, the bridge entity and the answer, with aliases.
- *Anthropic's lens evaluation sets* (Apache 2.0): 93 multi-hop items and 55
  order-of-operations items, each with a prompt, a target and named intermediates.

**Measurement.** Forward-only, both checkpoints, at 1, 2 and 3 passes. A prediction counts
as naming an entity if the top token is the first token of any of its aliases. Items where
the bridge and the answer share a first token are dropped. For TwoHopFact, results are
reported on all items and on the subset where the 3-pass model answers both one-hop prompts
correctly, so that failures are failures of composition and not of knowledge. Intervals are
95% bootstrap intervals over items.

**The claims under test, and what would count as failing.**

1. *Composition needs the later passes; recall does not.* On the released model the two-hop
   accuracy gain from 1 pass to 3 is larger than the one-hop gain. **Fails if** the two
   intervals overlap.
2. *Stopping early reveals the bridge.* On two-hop prompts the released model's top
   prediction at 1 pass is the bridge entity more often than at 3 passes, and more often than
   a control that scores another item's bridge entity. **Fails if** the 1-pass rate is not
   above both.
3. *The variable-loop sibling composes in one pass.* Its two-hop gain from 1 pass to 3 is
   smaller than the released model's. **Fails if** the intervals overlap.

**What this will not show.** First-token matching is lenient (it can credit a wrong entity
that starts the same way) and strict (it misses a correct answer phrased differently); the
control in claim 2 bounds the first. TwoHopFact's answers are Wikidata entities, many of
them obscure, so absolute accuracy will be low. Still one position, one decoding rule.

### Result

Raw data: [`results/early-exit-public.json`](results/early-exit-public.json). Script and
analysis: `earlyexit.py` (`--analyse` prints the intervals).

**Anthropic's lens evaluation sets: share of items where the top prediction is the target,
with 95% intervals.**

| Set | Passes | Inkling 2 Mini | Variable-loop |
|---|---|---|---|
| multi-hop (n=80) | 1 | 16% [9, 25] | 40% [29, 51] |
| | 2 | 35% [25, 45] | 40% [30, 50] |
| | 3 | 40% [29, 50] | 45% [34, 56] |
| order of operations (n=55) | 1 | 7% [2, 15] | 60% [47, 73] |
| | 2 | 29% [18, 42] | 66% [53, 78] |
| | 3 | 64% [51, 76] | 71% [58, 82] |

**After independent review: the scorer was faulty in both directions.** It expected a leading
space even after an opening quotation mark (rejecting correct answers such as `up` and
`January`) and credited some prefixes (` begin` for `beginnings`). Rescoring the saved
outputs by whole decoded token gives, for the released model on multi-hop, **16 of 80 at one
pass and 40 of 80 at three** (we reported 13 and 32); order of operations 3 and 35 of 55 (we
reported 4 and 35). The variable-loop model goes 39 to 45 of 80 and 33 to 39 of 55. The
one-pass output names an intermediate on 5 of 80, not 2. Every conclusion below keeps its
direction; the percentages in the table above are superseded by these counts.

**The same sets: share of items where the top prediction after one pass is a named
intermediate.** Multi-hop: 2 of 80 on the released model, 0 of 80 at three passes.
Order of operations: 3 of 55 at one pass, 6 of 55 at three.

**TwoHopFact (n=332 scorable of 400): uninformative.** Two-hop accuracy was 8–12% and
single-hop accuracy 4–8% at every pass count on both models. A look at the raw outputs shows
why: the dataset's prompts have no "Fact:" framing, and this chat-tuned model continues them
with filler (a colon, `the`, a multiple-choice list) even when it goes on to state the right
entity a few tokens later. Scoring the next token therefore measures nothing here. Only 4
items had both single hops scored correct.

**Against the criteria.**

1. *Composition needs the later passes; recall does not.* **Not tested as specified.** The
   test needed TwoHopFact's one-hop control, and that dataset was unusable. On Anthropic's
   sets, which have no one-hop control, multi-step accuracy on the released model rises from
   16% to 40% and from 7% to 64% over passes one to three, with intervals that do not
   overlap between one pass and three. That supports "the released model needs its later
   passes for multi-step questions" on prompts we did not write. It does not test the
   comparison with recall.
2. *Stopping early reveals the bridge.* **Failed.** On 80 multi-hop prompts written by
   others, the one-pass output named the intermediate twice. It is mostly `the`, a space, or
   a wrong number. In our own probes the bridge was a natural next word ("The capital of the
   country where the Eiffel Tower stands is … France"); in these it is not ("The color of
   the planet fourth from the Sun is …"). Our probes were leading for this effect.
3. *The variable-loop sibling composes in one pass.* **Held on Anthropic's sets** (flat at
   40–45% and 60–71% across passes, against the released model's steep rise), though the
   criterion as written was about TwoHopFact.

**What changes.** The pass-sweep claim that survives is the weaker one: the released model's
later passes are needed for multi-step answers, and a sibling trained to stop early does not
need them. The stronger claim, that stopping early lets you *read off* the intermediate
step, does not generalise. It appeared when our prompt wording made the intermediate a
grammatical continuation. The paragraph "Why it matters for monitoring" under question 2
overstated this and should be read with this result.

**A process failure worth recording.** This run was launched without the small pilot the
earlier runs had. A ten-item pilot would have shown the TwoHopFact scoring problem for a few
cents instead of about ninety.

**Not pursued: TwoHopFact with "Fact:" framing.** A 14-item pilot showed the framing fixes
the format but the model does not know most of these (often obscure) facts, so few items
would have both single hops right. A dataset of well-known two-hop facts, or scoring a short
generated answer instead of one token, would be the way to make the one-hop comparison on
independent data. That comparison remains untested.

## Follow-up: can a lens read the intermediate step on others' prompts?

### Plan (written before running)

The previous section left open whether the intermediate step is readable at all on prompts
we did not write. This uses the instrument built for that and the metric its authors define.

**Measurement.** Anthropic's 80 scorable multi-hop items on Inkling 2 Mini. For runs of 1, 2
and 3 passes, read the last prompt position at every step through that run's own lens (the
3-prompt lenses from question 1) and through the plain logit lens. Metric, as defined in
the evaluation set's README: pass@k is the share of items whose intermediate reaches lens
rank k or better at any step (k = 1, 10, 100). Control: the same score for another item's
intermediate, which sets the level reached by chance when taking the best of ~50 steps.

**Expectation.** Above the control at every pass count for both lenses. With lenses fitted
on 3 prompts we make no prediction that the Jacobian lens beats the logit lens.

**What this will not show.** Readable is not the same as used. The lenses are barely fitted.

### Result

Raw data: [`results/lens-eval-multihop.json`](results/lens-eval-multihop.json). Script:
`lenseval.py`. 88 scorable items.

**Share of items whose intermediate reaches the given rank at some step** (control: another
item's intermediate).

| Run | Lens | top 1 | top 10 | top 100 | control: top 1 | top 10 | top 100 |
|---|---|---|---|---|---|---|---|
| 1 pass | Jacobian | 26% | 50% | 75% | 1% | 10% | 32% |
| 1 pass | logit | 27% | 59% | 95% | 0% | 15% | 52% |
| 2 passes | Jacobian | 26% | 52% | 76% | 1% | 6% | 32% |
| 2 passes | logit | 26% | 60% | 95% | 2% | 7% | 50% |
| 3 passes | Jacobian | 31% | 59% | 81% | 2% | 5% | 30% |
| 3 passes | logit | 30% | 66% | 94% | 3% | 10% | 51% |

**What we see.**

- **The intermediate step is readable from the model's internal states on prompts we did not
  write.** It is the lens's top word at some step in about a quarter to a third of items, and
  in the top ten in half to two thirds, against a few percent and 5–15% for the control.
- **It is readable even when the output does not show it.** After one pass the model's
  *output* named the intermediate in 2 of 80 items; a lens on its internal states finds it
  at rank 1 in about a quarter.
- **The plain logit lens does as well as or better than the Jacobian lens here.** With lenses
  fitted on 3 prompts that is not a fair contest, but it means nothing in this repo shows the
  Jacobian lens earning its cost on this model.

**After independent review.** This is not the upstream pass@k. Upstream scores each listed
intermediate separately and averages within an item; we took the single best rank over all
listed intermediates and spellings, and deeper runs get more steps to score on. The
reviewer's matched comparison is the cleaner statement: on the 75 items shared with the
early-exit test, at one pass, the plain logit lens has the intermediate as its top word
somewhere in the network on 24, the output names it on 5, and **20 have a lens hit without
an output hit**. The lens was told which word to look for and searched every layer, so this
shows some intermediate words are recoverable. It is not a monitor.

**Against the expectation.** Above the control at every pass count, for both lenses: held.

**What this does not show.** Taking the best rank over roughly 25 to 50 steps is generous;
the control bounds how generous. No intervals were computed. It says the intermediate is
present and linearly readable somewhere in the network, not at which pass, and not that the
model uses it.

## Follow-up: what exactly crosses? A transplant with four distinguishable answers

Proposed by the independent review. The earlier swap could not say whether the messages
carry an intermediate that the next pass reuses, or the finished answer, because both
changed together.

### Plan and predictions (written before running)

**Design.** Every prompt is a two-step question: a first step picks an intermediate, a second
step maps it to an answer. Donor and recipient differ in *both* the item and the second
step. A transplant of the donor's messages into the recipient's final pass can then give:

| Outcome | Meaning |
|---|---|
| recipient's answer | the messages were ignored |
| donor's answer | the messages carry a finished answer |
| recipient's second step on the donor's intermediate | the messages hand over a reusable intermediate |
| donor's second step on the recipient's intermediate | the messages carry which second step to do |
| anything else | uninterpreted |

A pair is used only if all four named answers are different words, both prompts tokenize to
the same length, and the model answers both correctly without any transplant. The expected
answer for each outcome is computed before the transplant is run.

**Two task families.**

- *Fictional lookup tables given in the prompt* (fresh, so nothing can come from memory):
  one table maps four animals to colours, a second maps colours to fruits. Donor and
  recipient share the first table and have different second tables.
- *Real facts:* "The language / currency of the country whose capital is X". Donor and
  recipient ask for different properties of different countries. Two wordings are tried; the
  one the model answers correctly more often *without any transplant* is used, a choice made
  before any transplant is run.

Transplants go into pass 3 and, separately, pass 2, at every position, on both checkpoints.
Every eligible pair is saved with its prompts, expected answers and outcome, along with the
model and library revisions.

**Predictions.** We do not have a confident one, which is the reason to run it.

1. The capital experiment suggests the messages carry something answer-determining, and the
   arithmetic cross-operation result (an addition given a multiplication's messages sometimes
   multiplies) suggests they can carry the *second step*. Our best guess is that the
   recipient's-second-step-on-donor's-intermediate outcome is the most common outcome other
   than the recipient's own answer in both families, on the released model, for pass 3.
2. Transplants into pass 2 will move fewer answers than into pass 3, as before.
3. The sibling will move less than the released model.

**What would change our mind.** If the donor's finished answer dominates, the "reusable
intermediate" reading is wrong for these tasks. If the recipient's answer dominates in the
lookup family but not the real-facts family, then the earlier result depended on the two
questions sharing a second step.

**What this will not show.** Two families, one wording each. A transplant replaces whole
messages at every position; it cannot say where in the message the information sits. If few
pairs are eligible (the model must get both questions right), the counts will be small.

### Result

Raw data, every eligible pair: [`results/transplant-four-way.json`](results/transplant-four-way.json).
Script: `transplant.py`. Wordings were chosen on unaided accuracy before any transplant:
the first three wordings tried gave "the" as the next word and were replaced after two short
pilots that ran no transplants.

**Fictional lookup tables (96 eligible pairs on each model; the model answers all 96
prompts correctly unaided).**

| Transplant into | Model | recipient's answer | donor's answer | recipient's second step on donor's item | donor's second step on recipient's | other |
|---|---|---|---|---|---|---|
| pass 3 | Inkling 2 Mini | 0 | 0 | **96** | 0 | 0 |
| pass 3 | variable-loop | 0 | 0 | **96** | 0 | 0 |
| pass 2 | Inkling 2 Mini | 23 | 0 | 71 | 1 | 1 |
| pass 2 | variable-loop | 31 | 3 | 61 | 1 | 0 |

**Real facts: no usable test.** No aligned wording got the released model above 8 of 14
prompts unaided, leaving 0 eligible pairs (12 on the sibling, mostly "other"). Not a result.

**What we see.**

- **The messages do not carry a finished answer.** The donor's own answer never appears
  after a transplant into the final pass (0 of 192 across both models).
- **The final pass redoes the second lookup against its own prompt.** The answer is always
  the fruit that the *recipient's* table assigns to the donor's item, even though the
  donor's messages were computed with a different second table in view.
- **Transplants into pass 2 now move most answers too** (71 and 61 of 96), unlike the
  capital experiment (8 of 42). An earlier edit is not always washed out.

**A gap in this design, found on checking the result.** Donor and recipient share the first
table. So "the messages carry the donor's intermediate (its colour)" and "the messages carry
which animal the donor was asked about" predict the same answer. This result separates
"finished answer" from "something earlier", and nothing finer. The next section separates
the two.

**Against the predictions.**

1. *Recipient's-second-step-on-donor's-intermediate is the most common changed outcome:* held,
   far more cleanly than expected, with the caveat above about what "intermediate" means here.
2. *Pass 2 moves fewer than pass 3:* held (71 against 96; 61 against 96).
3. *The sibling moves less:* not at pass 3 (both 96 of 96); slightly at pass 2.

## Follow-up: intermediate or query? A transplant where both tables differ

### Plan and prediction (written before running)

**Design.** As above, but the donor and recipient now have different *first* tables as well
as different second tables, over the same six animals, colours and fruits. For a recipient
asked about animal R and a donor asked about animal D, a transplant into the final pass can give:

| Outcome | Computed as | Meaning |
|---|---|---|
| recipient's answer | recipient's tables on R | ignored |
| donor's answer | donor's tables on D | a finished answer |
| carried intermediate | recipient's second table on the *donor's* colour for D | the messages carry the intermediate |
| carried query | recipient's *both* tables on D | the messages carry only which animal was asked about |
| donor's second step on recipient's | donor's second table on the recipient's colour for R | the messages carry the second table |

Pairs are used only if the five answers are five different fruits, the prompts are the same
length, and the model answers both prompts correctly unaided. Both checkpoints, pass 3 and pass 2.

**Prediction.** The carried-intermediate outcome dominates in pass 3. Reason: in the
arithmetic swaps, information stated in the prompt (the numbers) was never overridden by a
donor's messages, and the queried animal is stated in the prompt.

**What would change our mind.** If "carried query" dominates, then messages can override
what the prompt states, the "derived, not given" reading of the earlier swaps is wrong, and
the 96 of 96 above was about the query all along.

**What this will not show.** One task family with small invented tables.

### Result

Raw data, every eligible pair: [`results/transplant-two-table.json`](results/transplant-two-table.json).

| Transplant into | Model | recipient | donor | carried intermediate | carried query | donor's 2nd step on recipient's | other |
|---|---|---|---|---|---|---|---|
| pass 3 | Inkling 2 Mini (144 pairs) | 0 | 0 | **0** | **144** | 0 | 0 |
| pass 3 | variable-loop (138 pairs) | 3 | 0 | 1 | 133 | 0 | 1 |
| pass 2 | Inkling 2 Mini | 32 | 0 | 0 | 111 | 0 | 1 |
| pass 2 | variable-loop | 114 | 1 | 2 | 20 | 0 | 1 |

**The prediction was wrong.** The donor's colour is never what determines the answer. The
answer is what the recipient's own two tables give for the donor's animal. So the "derived,
not given" reading of the earlier swaps does not hold here: a donor's messages overrode
something the recipient's prompt states outright, namely which animal is being asked about.

**And a second gap, found by a reviewer reading the saved outputs
([note](../notes/lookup-position-control.md)).** In all the lookup prompts so far, the
second table is printed in the same row order as the first. A rule that ignores colours
entirely, "return the fruit in the same row as the animal", gets every prompt right, and
"carry the donor's row number" predicts all 96 and all 144 transplant outcomes just as well
as "carry the donor's animal". So these two experiments rule out a carried colour and a
carried finished answer, and cannot say whether the model used the tables' content at all.

## Follow-up: animal, colour or row? Shuffling the rows

### Plan and predictions (written before running)

**Design.** Same task with seven entries per table and both tables different on the two
sides, but every list is printed in its own random row order. Row position now says nothing
about which facts go together: the fruit in the same row as the queried animal is the right
answer on only 14 of 168 prompts.

**First, unaided accuracy.** If the model was answering by row position, it should now fail,
and its mistakes should be the same-row fruit. That is reported before anything else.

**Then transplants,** on pairs where these five answers are five different fruits (324 of
1,008 candidate pairs qualify, checked offline), and where the model answers both prompts
correctly unaided:

| Outcome | What the messages would have carried |
|---|---|
| recipient's answer | nothing that matters |
| donor's answer | the finished answer |
| carried colour | the donor's intermediate |
| carried animal | which animal the donor was asked about |
| carried row | the row number the donor's animal sat in |

A sixth, "the row the donor's colour sat in", is scored where it is distinct (112 pairs).

**Predictions.**

1. Unaided accuracy stays high. The model's reported lookup-chain ability would be odd if it
   rested on row alignment. (We were wrong about the last prediction, so hold this loosely.)
2. If so, "carried animal" dominates in pass 3 and "carried row" does not.

**What would change our mind.** If accuracy collapses, the earlier lookup results were about
layout and say nothing about two-step reasoning. If "carried row" dominates, the messages
carry position.

**What this will not show.** One invented task. Whether the same holds for facts the model
knows, where no aligned wording has yet worked.

### Result

Raw data, every eligible pair: [`results/transplant-shuffled.json`](results/transplant-shuffled.json).

**Unaided accuracy with shuffled rows:** released model 168 of 168; sibling 166 of 168.
The model is doing the two-step lookup, not reading across rows. Prediction 1 held.

**Transplants into pass 3, scored against the outcomes listed in advance.**

| Model | pairs | recipient | donor | colour | animal | animal's row | colour's row (scored on the 112 / 105 pairs where distinct) | other |
|---|---|---|---|---|---|---|---|---|
| Inkling 2 Mini | 324 | 56 | 60 | 37 | 33 | 24 | 110 | 4 |
| variable-loop | 312 | 51 | 52 | 64 | 34 | 20 | 81 | 10 |

No listed explanation dominates. Prediction 2 ("carried animal") was wrong again.

**A single rule found afterwards fits almost everything.** "Answer with whatever fruit sits,
in the recipient's own second list, in the row where the donor's answer sat in the donor's
second list." Recomputed over every pair, including those where it coincides with another
outcome, this predicts 316 of 324 pass-3 outcomes on the released model and 259 of 312 on
the sibling. The scattered counts above are the pairs where that row happens to hold the
recipient's fruit, the donor's fruit, and so on. For transplants into pass 2 it predicts
only 54 of 324 and 46 of 312.

**Reading.** The messages into the final pass appear to mark a *position* in the prompt,
the place where the answer is, and the final pass reads its own prompt at that position. That
would explain all three lookup experiments, including both mispredictions, and the capital
experiment as well. Messages are built position by position, so a position-bound marker is a
natural thing for them to carry.

**Why this is not yet a finding.** The rule was formed from this data and then checked on
the same data. The "colour's row" outcome was listed in advance, but as a side case, not as
the expected winner. The next section tests it on new data with the prediction made first.

## Follow-up: is it a position marker? A held-out test and a localisation test

### Plan and predictions (written before running)

**Data.** New tables from a new random seed, and for each model the wording it was *not*
tested with above. Up to 150 pairs sampled at random from those where the pointer rule's
fruit differs from both the recipient's and the donor's answers.

**Test 1, replication.** Full transplant into pass 3. *Prediction:* the pointer rule's
fruit is the answer in at least 85% of pairs on the released model.

**Test 2, localisation (released model).** If the marker lives at a position, then:

| Transplant only… | Prediction |
|---|---|
| the messages at the donor's answer row (about five tokens) | pointer fruit is the most common answer |
| everything *except* that row | pointer fruit in fewer than 10% of pairs |
| the single fruit token of that row | reported; no prediction |
| the messages at a different, unrelated row | that row's fruit in fewer than 10% of pairs |

**What would change our mind.** If the full transplant no longer follows the rule on new
tables and wording, the rule was fitted to one dataset. If transplanting everything except
the row still produces the pointer fruit, the information is not at that position.

**What this will not show.** Still one invented task with every word a single token.
Whether real-fact questions use the same mechanism is untested, and the capital result is
only consistent with it.

### Result

Raw data, every sampled pair: [`results/transplant-pointer.json`](results/transplant-pointer.json).

**Test 1, replication on new tables and unseen wording: held.**

| Model | pairs | pointer fruit | recipient's | donor's | other |
|---|---|---|---|---|---|
| Inkling 2 Mini | 150 | **145** | 2 | 0 | 3 |
| variable-loop | 150 | 123 | 5 | 1 | 21 |

**Test 2, localisation on the released model: both predictions failed.**

| Transplant only… | pointer fruit | recipient's | donor's | other |
|---|---|---|---|---|
| the donor's answer row | **1** | 149 | 0 | 0 |
| everything except that row | **47** | 20 | 16 | 67 |
| the row's fruit token | 0 | 150 | 0 | 0 |
| an unrelated row | 0 | 149 | 0 | 1 |

**What we see.**

- **The behavioural rule is real.** After a full transplant into the final pass, the model
  answers with whatever its own prompt has in the row where the donor's answer sat. That
  now holds on three table designs, two wordings and held-out tables, with the prediction
  made in advance the last time.
- **The messages therefore say *where* the answer is, not *what* it is.** The donor's
  answer, colour and animal all lose to the position.
- **But the marker is not stored at the place it points to.** Transplanting the messages at
  that row alone does nothing, and removing them from a full transplant still leaves the
  rule working a third of the time. We predicted the opposite on both counts.

**Against the predictions.** Test 1: held (97% against a stated 85%). Test 2: wrong on the
row-only transplant (1 of 150, predicted most) and on the all-but-row transplant (47 of 150,
predicted under 10%). The unrelated-row control came out as expected, trivially.

**What this does not show.** Where the pointer is stored; the next section looks. Whether
anything like it happens on questions about real facts. One invented task.

## Follow-up: where is the pointer stored? Transplanting by region

### Plan (written before running)

Same 150 held-out pairs on the released model, pass 3. Transplant the donor's messages only
within one region of the prompt at a time: the first table, the second table, the question
at the end, the final token alone, and pairs of regions. Outcome: how often the pointer
fruit is the answer.

**Prediction, held loosely after two misses:** the question region at the end is necessary
(transplants that leave it out rarely give the pointer fruit) and is not sufficient alone.

### Result

Raw data: [`results/transplant-regions.json`](results/transplant-regions.json). 150 pairs,
released model, pass 3. A full transplant gave the pointer fruit on 145 of these.

| Transplant only within… | pointer fruit | recipient's | donor's | other |
|---|---|---|---|---|
| first table | 0 | 147 | 0 | 3 |
| second table | 6 | 67 | 15 | 62 |
| question | 34 | 15 | 5 | 96 |
| final token | 0 | 150 | 0 | 0 |
| first + second table | 8 | 41 | 18 | 83 |
| first table + question | 26 | 18 | 10 | 96 |
| **second table + question** | **140** | 2 | 1 | 7 |

**What we see.** The effect needs the messages at the second table *and* at the question
together; either alone mostly breaks the answer or leaves it alone. The first table's
messages are irrelevant. The prediction (question necessary, not sufficient) held.

**Reading.** A matched pair: something at the question that selects a place, and something
at the second table that marks the place to be selected. Each message has four parts, for
the attention query, key and value and for the expert layer, so the obvious candidate is a
query at the question matching a key at the table.

## Follow-up: which part of the message? Queries and keys

### Plan and prediction (written before running)

Same 150 pairs. Transplant, into pass 3, only chosen parts of the messages in chosen regions.

**Prediction.** Key parts at the second table plus query parts at the question reproduce most
of the effect (pointer fruit in more than half the pairs). Value parts and expert-layer parts
alone, in both regions, do not (under 10%).

### Result

Raw data: [`results/transplant-parts.json`](results/transplant-parts.json). 150 pairs,
released model, pass 3, everything confined to the second table and the question.

| Message parts transplanted | pointer fruit | recipient's | donor's | other |
|---|---|---|---|---|
| all four | 140 | 2 | 1 | 7 |
| keys at the table + queries at the question | **11** | 132 | 0 | 7 |
| queries and keys, both regions | 7 | 136 | 0 | 7 |
| keys and values at the table + queries at the question | 30 | 67 | 10 | 43 |
| values only | 32 | 90 | 4 | 24 |
| expert-layer only | 0 | 147 | 0 | 3 |
| values and expert-layer | 51 | 66 | 5 | 28 |

**The prediction was wrong.** Queries and keys barely matter (11 of 150 against a predicted
majority), and values alone do more than predicted (32, against under 10%). No part or pair
of parts comes close to all four together. The effect is spread across the whole message.

**Where this leaves the mechanism.** Known: a full set of messages, at the second table and
the question together, makes the final pass read its own prompt where the donor's answer
sat. Not known: how. It is not a marker at the pointed-to row, and it is not an attention
query matching a key. Three guesses about the mechanism have now failed, and we stop
guessing here.

## The oversight test: can internal access catch a right answer reached the wrong way?

This is the question that connects the work to alignment. An overseer often sees a plausible
answer and cannot tell how the model got there.

### Plan and predictions (written before running)

**Setting.** The same invented two-step lookup, with one change: in the second table, pairs
of colours share a fruit. So for most questions there is a second row that holds the same
fruit as the correct row.

**Hidden mistakes.** We know from the work above that transplanting another prompt's
messages into the final pass makes the model read a chosen row of its own prompt. So:

- *Wrong route, right answer:* transplant from a donor that points at the *other* row with
  the same fruit. The visible answer is correct. The route is not.
- *Right route, right answer:* transplant from a donor that points at the correct row. Both
  kinds of trial have had a transplant, so "something was tampered with" does not separate them.

**Establishing the route independently.** Each trial is repeated on a twin prompt in which
the other row's fruit is replaced by a new one. If the answer follows the new fruit, the
model was reading the other row. Only trials whose twin confirms the intended route, and
whose visible answer is correct, are kept. The label is never given to either reviewer.

**Two reviewers.**

- *Prompt and answer only.* By construction the answer is correct in every kept trial, so
  this reviewer has nothing to go on. To make the comparison fairer we also give it the
  model's output confidence (top probability and margin).
- *Plus a fixed internal readout.* A single direction in the model's internal state at the
  last token: the difference between the mean state on wrong-route and right-route trials.
  The layer is chosen by cross-validation on the training tables only, and the direction and
  its threshold are then frozen.

**Data.** Train on 20 sets of tables with one wording. Test on 10 new sets with the same
wording, and on 10 new sets with a different wording. Released model, transplants into pass 3.

**Measures.** Area under the ROC curve with a 95% bootstrap interval, and the share of
hidden mistakes caught at a threshold set for a 10% false-alarm rate on the training tables.

**Predictions** (our record on this task is five wrong out of eight, so hold them loosely).

1. At least 80% of intended wrong-route trials are confirmed by their twin with a correct
   visible answer.
2. Output confidence barely separates the two kinds of trial: AUC below 0.65.
3. The internal readout separates them on new tables with the same wording: AUC above 0.8.
   On the new wording we expect it to be worse and make no prediction.

**What would count.** Internal access helps if the readout's interval on new tables lies
entirely above the confidence baseline's. If it only works on the training tables, or only
at a layer chosen after seeing test results, it does not.

**What this will not show.** The mistake is one we induced, in a task we invented, and the
readout is trained on labelled examples of that same induced mistake. It would show that an
internal difference exists and transfers to unseen tables. It would not show that a readout
can catch a mistake nobody anticipated, which is the harder and more important case.

## Results

### First look: lens fitted on 3 prompts (preliminary)

Raw output: [`results/readouts-3-prompts.txt`](results/readouts-3-prompts.txt). Three prompts is
far below the ~100 the library suggests, and the result below rests on six probes read at one
token position, so treat it as a lead to confirm, not a finding.

What the lens shows at the last token of each two-step question:

| Where | What the lens reads | Example ("currency of the country shaped like a boot") |
|---|---|---|
| Pass 1, all loop layers | Punctuation and noise | `‘`, `”`, `[` |
| Pass 2, layers 9–11 | The right *category*, not the specific answer | `money`, `financial`; for the capital-city probe, `city`, `cities` |
| Pass 3, layers 7–11 | The specific intermediate and the answer together | `euros`, `Italian`, `Italy`, `lira` |
| Layers 12–17 of every pass | Blanks and ellipses | `____`, `...` |
| Post-loop layers 18–23 | The answer | `euro` |

The same shape appears on four of the six probes (Italy/euro, France/Paris, Japan/Japanese,
yellow/green). Concepts in pass 3 are read out in many languages at once (`France` appears as
`Frankreich`, `فرانس`, `फ्रांस`), which suggests a language-independent representation.

**Correction after reading the prior work (same day):** the "pass 1: punctuation and noise"
row is most likely the instrument failing, not the model. A final-target lens is known to read
punctuation in early loops of other looped models. Claims about passes 1 and 2 need a lens
aimed at each pass's own end, or a direct read of the model's output with fewer passes. The
pass-3 and post-loop rows are within the lens's reach and stand as a preliminary observation.

Against the predictions:

1. **Intermediate appears at a specific pass: supported so far.** It appears in pass 3, with
   pass 2 carrying only the category. That is a coarse-to-fine pattern across passes.
2. **Jacobian lens beats the logit lens: mixed.** It ranks the intermediate concept higher in
   pass 3 on some probes (Italy: rank 2–5 against 20–956) and lower on others (France). On the
   final answer the logit lens is as good or better.
3. **Loop layers more readable than pre-loop layers: only in pass 3.**

Reasons to doubt this reading:

- **Pass 1 may be unreadable for a structural reason, not because it holds nothing.** Later
  passes restart from the pre-loop state, so pass 1 reaches the output only through the message
  channel. Its activations may simply not be in the output's basis. The lens can't distinguish
  "no content" from "content in a format this lens can't read".
- **The model got two probes wrong** (spider/eight, Mars/two), so those say little.
- **A bug in the first run's rank columns.** Words that split into several tokens were ranked
  by their first piece, which made the spider and Mars rank columns meaningless. The top-5
  token columns were unaffected and are what the table above uses. Fixed in `fit.py`.

Cost so far: about 14 minutes of GPU time for this run.
