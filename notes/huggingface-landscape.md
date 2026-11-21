# Alignment-adjacent work on Hugging Face

Snapshot from keyword searches of the Hugging Face API on 2026-11-21.

| Area | Results | Most-downloaded example |
|---|---|---|
| "abliterated" models | 1,000+ | 5.2M downloads |
| "uncensored" models | 1,000+ | 2.2M downloads |
| "heretic" models | 1,000+ | 2.2M downloads |
| Sparse autoencoders ("sae") | 1,000+ | 70k downloads |
| Transcoders | 190 | 3k downloads |
| Gemma Scope | 89 | 3k downloads |
| Jacobian lens ("jlens") | 62 | 52 downloads |
| Sleeper-agent test models | 37 | 805 downloads |
| Emergent-misalignment test models | 17 | 101 downloads |
| Abliteration datasets | 28 | 3k downloads |
| Jailbreak datasets | 375 | 170k downloads |
| Refusal datasets | 477 | 3k downloads |
| Sycophancy datasets | 157 | 2.5k downloads |
| Deception datasets | 322 | 4k downloads |

Caveats: counts cap at 1,000; keyword search is noisy ("sae" and "alignment" both match
unrelated things); download figures are Hugging Face's own.

## What stands out

- **Abliteration is a mass-market activity.** It comes from "Refusal in Language Models Is
  Mediated by a Single Direction" (Arditi et al., 2024): find the direction and remove it from
  the weights. The popular results are nearly all Qwen variants packaged to run locally.
- **Interpretability tooling is about 100 times smaller by downloads**, and research test
  models (sleeper agents, emergent misalignment) are smaller again.
- **Evaluation of abliterated models barely exists.** The one dataset found for evaluating
  abliteration had 272 downloads, against millions for the models.
- **Probe datasets are a healthy niche**: saved activations for refusal, sycophancy and
  deception probes exist and are used.
- **The Jacobian lens is already in community hands**, including one repo that appears to use
  it to modify an abliterated model. Not inspected.
- **Commercial "unrestricted" APIs exist** (e.g. abliteration.ai), sold for security research,
  red-teaming, trust-and-safety and training-data generation, with a customer-configured policy layer.

## What it teaches

Safety training in open models is shallow enough to remove in minutes, and millions of people
run the result. That is direct evidence about the robustness of current alignment methods, and
it means the models people run locally are increasingly ones nobody has evaluated.
