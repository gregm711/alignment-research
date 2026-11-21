# Jacobian lens on Inkling 2 Mini

**Status: in progress.** A 3-prompt lens gives a first look (see Results); a ~20-prompt fit is running.

## Question

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is a looped
transformer: 24 layers of weights, with layers 6–17 run three times. What does each pass do?
On a two-step question, is the intermediate fact worked out in pass 1 and refined, or does it
only appear in pass 3?

The tool is the [Jacobian lens](https://github.com/anthropics/jacobian-lens) from
[Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html): a linear map
per layer that reads out which words the model is poised to say at that point.

## Prior art

Found after the first run (titles and dates checked on arXiv; papers not yet read):

- [Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?](https://arxiv.org/abs/2609.01924) (September 2026)
- [Monitoring latent reasoning in looped language models](https://arxiv.org/abs/2610.18127) (October 2026)

So the Jacobian lens has already been applied to looped models. A secondhand summary says they
used Ouro-2.6B and Huginn and found the lens degrades across loop boundaries; that needs
checking against the papers. What we have found no prior work on is this model's design, where
each pass restarts from the pre-loop state and passes exchange gated messages, or on its
variable-loop research checkpoint. Reading both papers is the next step before any claim of novelty.

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
- The model support is an unmerged pull request; results may shift if it changes.
- At the batch size that fits in memory, each prompt needs 576 backward passes.

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
