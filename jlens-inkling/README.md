# Jacobian lens on Inkling 2 Mini

**Status: in progress, being redesigned.** A 3-prompt lens gave a first look (see Results). Prior work shows our lens target was wrong for the early passes, so the longer fit was stopped and the next step is a forward-only sweep over the number of passes.

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

Neither covers this model. Its design (each pass restarts from the pre-loop state; passes are
linked by gated messages) is different from both designs they study, and its variable-loop
companion checkpoint allows the recurrence sweep Wang & Reid list as future work.

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
| 1 | What crosses the pass boundary through the message channel? | Message swapping between prompts (forward-only), then a per-pass Jacobian lens for readability | swap answered (see below); lens pending |
| 4 | Must an edit be held across all passes to stick? | Clamp a concept direction in one pass vs all | not started |
| 3 | Is the loop state robust to noise, including for refusals? | Noise at the loop entry state; record refusal yes/no only | not started |

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
