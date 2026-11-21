# Prior work on interpreting looped language models

Notes from reading the main text of both papers (not the appendices) on 2026-11-21.

## Wang & Reid, "Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?" ([arXiv 2609.01924](https://arxiv.org/abs/2609.01924), September 2026)

**What they did.** Applied the Jacobian lens and eleven families of causal experiments to two
looped models, with a standard model as baseline:

| Model | Design | Intermediate supervision |
|---|---|---|
| Ouro-2.6B | all 48 layers looped 4 times | every loop end is trained to be decodable |
| Huginn-0125 (3.5B) | 4-layer core repeated 16 times between a 2-layer prelude and coda; input re-injected every time | none |
| Qwen3.6-27B | 64 ordinary layers | none |

They used the lens library unmodified, with 1000 WikiText-103 prompts of 128 tokens per lens.
Their "virtual-unrolling adapter" is the same idea as our `PassView`: present each application
of a looped layer as its own layer, and make hooks fire only on the assigned pass.

**Findings that matter for us.**

- **A lens aimed at the final output fails on early loops, and reads punctuation there.** They
  say plainly this is "not because those loops are empty, but because the transport to the
  final target dies crossing loop boundaries". This is very likely what our first run showed in
  pass 1.
- **The fix is a family of lenses, one per loop end.** Only the target changes. Every readout
  is then taken through the lens whose target the source can reach, and comparisons across
  loops are made at matched distance from each lens's own target.
- **Aggregate faithfulness numbers are not a usability test.** At long range the bulk of the
  transported state washes out even in a standard model, yet a specific concept can still
  arrive. They count a readout as valid when it moves a probe token's rank well above chance
  *and* its direction supports a causal intervention.
- **What crosses a loop boundary depends on the boundary's design.** On Ouro each loop decodes
  into a token-like checkpoint and the next loop re-encodes, so the concept does not survive
  linear transport across the boundary even though most of the state does. On Huginn content
  computed by recurrence 2–3 stays readable through all sixteen, but only because the input is
  re-injected every time: cut the injection and the content is gone within one recurrence.
- **Harder questions settle in later loops.** On Ouro, "Paris" (easy) is top-1 at the end of
  loop 1; "Italy" for the boot riddle appears in loop 2; an arithmetic answer settles in loop 4.
- **Interventions follow the loop structure.** On Ouro, editing one loop is mostly undone by
  later loops (21% success for loop 2 alone, 38% when held across all remaining loops). On
  Huginn only the last ~2 recurrences are writable.
- **Weight tying makes fitted lenses repeat across iterations** at matched offsets (cosine
  0.9991 on Huginn), so one iteration's fit can cover all of them.
- **The lens is blind to idle content.** It is built from "how does this state influence the
  output", so anything held but not currently driving the next token reads as absent.
- **Stated future work:** a sweep over test-time recurrence count, and controlled training runs
  that vary only the boundary design.

## Olszowiec et al., "Monitoring latent reasoning in looped language models" ([arXiv 2610.18127](https://arxiv.org/abs/2610.18127), October 2026)

**What they did.** Eight looped models from four families (1–3.5B), read at every iteration
through the model's own output head, with no fitted lens. They split designs into **carry
loops** (prompt enters once; Ouro, LoopUS) and **re-read loops** (prompt re-supplied every
iteration; Huginn, Rec-OLMo, Parcae).

**Findings.**

- **The answer is decided early.** In six of eight models, 76–89% of generated tokens have
  their final top-1 token after the first iteration. Later iterations mostly sharpen it.
- **More iterations are not always better.** Base Ouro is best at its trained depth and worse
  beyond it. Reasoning-tuned Ouro is best at iteration 1 and degrades with every iteration.
  Only Huginn improves steadily.
- **Harder problems do not make the loop work harder** by their measures (MATH levels 1–5).
  This sits in tension with the per-task depth result above; the two papers measure differently.
- **Attention heads keep their roles** across iterations.
- **Noise at the loop's entry state breaks carry loops and not re-read loops.** At twice the
  state norm, carry models still write fluently but refusals erode first (one model's refusal
  rate fell from 73% to 25%). Re-read models are almost unaffected at fifty times the norm.
  That is a direct safety-relevant difference between loop designs.

## Where Inkling 2 Mini sits, and what is still open

Neither paper covers it (it was released on 2026-11-20). Its design is a third kind:

- Passes 2 and 3 **restart from the pre-loop state**, which makes it re-read-like.
- Passes are linked by **gated messages** built from the previous pass's states, which is
  neither Ouro's decode/re-encode checkpoint nor Huginn's plain carried state.
- It is ten times larger than anything in either paper, and uses mixture-of-experts layers.
- A companion checkpoint (`TML-d24-varloop-research`) supports a variable number of passes.

Open questions this leaves for us:

1. **What crosses the pass boundary through the message channel?** The analogue of their
   "does Italy survive the transport" test, for a boundary design nobody has measured.
2. **When is the answer decided, per pass?** Changing the number of passes at inference and
   reading the model's own output needs no fitted lens and no gradients. On the variable-loop
   checkpoint this is the recurrence sweep Wang & Reid list as future work.
3. **Does the noise-robustness split hold?** The carry/re-read result predicts this design
   should be robust to noise in the loop state, including for refusals. Testable cheaply.
4. **Do interventions need to be held across all passes**, as on Ouro, or only the last?

## What this changes in our experiment

- Our first reading ("pass 1 holds nothing readable") should be treated as an instrument
  failure until shown otherwise.
- The lens needs per-pass targets before any claim about passes 1 and 2.
- The forward-only pass sweep (question 2) is cheaper and more direct, and should come first.
- Any readout claim needs a causal check behind it.
