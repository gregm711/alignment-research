# Research directions

Ideas for small, cheap experiments on open-weight models that could produce something worth
publishing. Ordered by how strongly we'd recommend them today.

## 1. Jacobian lens on a looped transformer (in progress: [`../jlens-inkling/`](../jlens-inkling/))

Inkling 2 Mini runs its middle layers three times. Looped models do more reasoning internally
and less in visible text, which works against chain-of-thought monitoring. If a cheap linear
readout still recovers intermediate reasoning from inside the loop, that is good news for
oversight of this design; if it doesn't, that is worth saying too.

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
