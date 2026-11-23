# Looped transformers: what they are, why labs are adopting them, what it means for monitoring

Written 2026-11-23. Sources and how deeply each was read are listed at the end. This note is
the background; `looped-prior-art.md` covers the interpretability papers in more detail.

## What a looped transformer is

A normal transformer runs each layer once per token. A looped transformer runs a block of
layers with **shared weights** several times per token, and each pass refines the hidden
state left by the one before. You get more depth, meaning more serial computation per token,
without adding parameters.

Most designs have three parts: a **prelude** of ordinary layers, a **looped core**, and a
**coda** of ordinary layers that produces the output. They differ in what links one pass to
the next:

| Design | What crosses the pass boundary | Examples |
|---|---|---|
| Carry | Hidden state passed straight on; prompt enters once | Ouro, LoopUS |
| Re-read | State carried, and the input embedding is injected again every pass | Huginn, Rec-OLMo, Parcae |
| Restart + messages | Each pass restarts from the pre-loop state and receives gated messages from the last pass | Inkling 2 Mini |
| Gated residual update | Per-token, per-pass gates control how much each pass writes to the residual stream | GLM-5.4 |

The idea goes back to the Universal Transformer (2018). What changed in 2025–26 is that it now
works at scale.

## Who uses it (checked against sources, 2026-11-23)

- **Z.ai GLM-5.4: confirmed by the company.** It is a roughly 750B model that loops **28
  layers of the GLM-5 base for two passes**. It was converted from a pretrained base by
  continued training, not trained from scratch. Z.ai checked the design at 1B, 4B, 30B and
  750B, and says it plans to release the weights "after completing necessary safety
  hardening".
- **OpenAI Astra: reported, not confirmed.** The Information, then Fortune (2026-09-03),
  reported that Astra uses recurrent depth "for a portion of the model's internal
  architecture". Jakub Pachocki's reply on X, as Fortune quotes it, does not deny this and
  says CoT monitoring could get harder "for reasons not contingent on architecture changes".
  Fortune reports that OpenAI "limited the extent" of looping so that reasoning stays
  legible. OpenAI's own post of 2026-11-10 ("What affects chain-of-thought monitorability
  across the model lifecycle?") confirms that GPT-6 Astra has "degradation in CoT
  monitorability for non-adversarial settings" and "significantly greater no-CoT
  capability". It also reports that an anonymised "Architecture 2" does better on multi-hop
  and serial computation than its perplexity predicts, but on most no-CoT evals perplexity
  still predicts performance across architectures. **The post never names looping**, so
  the link between looping and Astra's monitorability drop is still inference.
- **Thinking Machines Inkling 2 Mini** (2026-11-20): 12 of 24 layers run 3 times, plus a
  variable-pass research checkpoint. This is the model in `jlens-inkling/`.
- Open research models: Ouro (1.4B and 2.6B, ByteDance), Huginn-0125 (3.5B), Parcae,
  Rec-OLMo.

**On "the default":** that overstates it. One frontier lab has confirmed using it (Z.ai),
and only lightly: two passes over part of the network. One is reported to use it with a cap
(OpenAI). One mid-size open model is built around it (Inkling). It is clearly the direction
things are moving, but not yet the standard.

## Why it seems to be useful

### 1. Depth matters more than parameters for reasoning

Saunshi et al. (ICLR 2025) found that a k-layer model looped L times nearly matches a
kL-layer model on synthetic reasoning tasks (addition, p-hop induction). On those tasks,
accuracy scales with **effective depth**, not parameter count. Z.ai makes the same split:
"memory-intensive tasks benefit from more parameters, while reasoning-intensive tasks
benefit from more effective compute." Knowledge lives in parameters. Step-by-step
composition needs serial depth, and looping supplies depth without storing more knowledge.

### 2. Parameters are the expensive resource when serving

Generating tokens is mostly limited by memory bandwidth: every step reads all the weights.
Looping reuses weights that are already loaded, so extra passes cost FLOPs but very little
extra weight traffic. This is why "matches a model 2–3× larger" matters commercially:

- Ouro 1.4B and 2.6B (7.7T tokens) match 4B and 8B dense models on most benchmarks (their
  own claim).
- Z.ai claims GLM-5.4 (750B) is "comparable to 2T-3T parameter models" (marketing wording,
  with no matched comparison published).
- **Catch: more FLOPs per token, and the KV cache grows with the number of passes.**
  GLM-5.4's main engineering work is avoiding that. Pass 2 reuses pass 1's sparse-attention
  indices and its KV cache, and adds only a sliding window of new KV, so "an extra loop only
  costs one sliding window of KV cache."

### 3. Reasoning in the loop is cheaper than reasoning in tokens

Saunshi et al. show that T loops can simulate T steps of chain of thought, with each loop
acting as a latent "thought". A written CoT token costs a full serial decoding step and grows
the context. A loop pass costs part of a forward pass. In Z.ai's matched-parameter 30B
ablation (about 10% more training compute), looping gave the largest gains on math,
science and long context, and **average output length fell 10–14%, "mostly from shorter
chains of thought. In effect, part of the reasoning shifts from explicit output tokens into
the implicit depth of the loop."** This is the property that most concerns safety.

### 4. Cheap retrofit

Z.ai states plainly that **"looped models trained from scratch still lag behind unshared
models under the same pre-training compute budget."** So looping does not win per unit of
training compute. It wins per parameter, and it can be added to an existing base with modest
continued training. Retrofitting needed fixes: without gates, activation norms "blow up
across iterations". In the trained model, pass 1 writes conservatively and pass 2 writes
fully, and pass 1 writes more on harder tokens.

### 5. A dial for test-time compute (mostly unrealised so far)

In principle you can run more passes on harder inputs. In practice the evidence is mixed.
Olszowiec et al. find the answer is usually settled after pass 1 (76–89% of tokens). Running
Ouro past its trained depth makes it worse, and only Huginn keeps improving. Z.ai lists
adaptive per-token depth as future work. Inkling's release post reports gains out to ten
passes on one lookup-chain task.

### 6. Possibly: harder to distill

This is Fortune's speculation, not a stated reason from any lab. If part of the reasoning
never appears as text, competitors cannot copy it from transcripts.

## Why this matters for alignment

The main monitoring tool in use is reading the written chain of thought. Looping is the
first mainstream architecture change that gives capability for *less* written reasoning, and
it is now built into training. What we already know:

- Z.ai's 10–14% shorter CoT is an aggregate. Nobody has measured *which* steps disappear,
  or whether a monitor misses things it used to catch. That is `loop-legibility/`.
- Loop design affects safety in a measurable way. In carry loops, noise added to the loop
  state erodes refusals (73% → 25% in one model). Re-read loops are robust at 50× the noise
  (Olszowiec et al.).
- Internal readouts do partly work. Lenses fitted per pass decode intermediate content
  (Wang & Reid). Goodfire read Inkling's cross-pass channel. Inkling's release post claims
  looping did not reduce monitorability at its scale. These are possible replacements for
  what the transcript stops showing, but none has been checked against a transcript monitor
  on the same task.
- With only two passes (GLM-5.4) and a reported cap (Astra), the shift so far is small. The
  question to keep tracking is whether labs raise the pass count once adaptive depth works.

## What this note does not establish

- Benchmark claims are the labs' own. We have reproduced none of them.
- OpenAI's architecture is known only from press reports.
- Whether 10–14% shorter CoT is a monitoring loss at all is still open. The removed tokens
  may have been padding.

## Sources and reading depth

- Z.ai, "GLM-5.4: Toward Generalized Coding with Our First Loop Transformer",
  https://z.ai/blog/glm-5.4. Full text read, taken from the page's JS bundle because the page
  is client-rendered. The ablation chart is an image and was not read.
- Fortune, 2026-09-03, https://fortune.com/2026/09/03/reports-openais-astra-model-uses-a-new-more-efficient-ai-architecture-alarms-ai-safety-experts-who-worry-the-method-makes-models-harder-to-control/,
  read in full. aiweekly.co summary, https://aiweekly.co/alerts/openais-astra-uses-recurrent-depth-alarming-safety-experts,
  read in full and treated as weak. The Information, the Astra system card and Pachocki's
  post itself were not read.
- Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers",
  https://arxiv.org/abs/2502.17416. Abstract and summary only.
- Zhu et al., "Scaling Latent Reasoning via Looped Language Models" (Ouro),
  https://arxiv.org/abs/2510.25741. Abstract and summary only.
- Wang & Reid 2609.01924, Olszowiec et al. 2610.18127, Goodfire on Inkling, Inkling release
  post: see `looped-prior-art.md` for what was read.
