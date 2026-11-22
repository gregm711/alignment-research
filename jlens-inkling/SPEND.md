# Spend log

Estimates only. Cost is GPU minutes times an assumed all-in rate of about $3.10 an hour
(one A100-80GB plus CPU and memory on Modal). Modal's billing page has the real figures.

| Date | Run | GPU minutes | Est. cost | Running total |
|---|---|---|---|---|
| 2026-11-21 | Smoke test (failed fast: library did not know the model yet) | ~2 | $0.10 | $0.10 |
| 2026-11-21 | Lens fit, 3 prompts, final target, with readouts | 13.4 | $0.70 | $0.80 |
| 2026-11-21 | Lens fit continuation (stopped early after reading the prior work) | ~7.5 | $0.40 | $1.20 |
| 2026-11-21 | Pass sweep, first attempt (out of memory: my bug) | ~1 | $0.05 | $1.25 |
| 2026-11-21 | Pass sweep, both checkpoints, text and probes | 4.8 | $0.25 | $1.50 |
| 2026-11-21 | Pass sweep, probes only, after fixing the scorer | 2.0 | $0.10 | $1.60 |
| 2026-11-21 | Message swap, first attempt (released model only; crashed loading the second: my bug) | ~4 | $0.20 | $1.80 |
| 2026-11-21 | Message swap, both checkpoints, with cross-operation control | 12.7 | $0.65 | $2.45 |
| 2026-11-21 | Lenses at 1, 2 and 3 passes, 3 prompts each, with readouts | 11.1 | $0.57 | $3.02 |
| 2026-11-21 | Noise pilot, 4 requests (showed the design could not answer the question) | 10.2 | $0.53 | $3.55 |
| 2026-11-21 | Direction swap inside a pass (question 4) | 2.3 | $0.12 | $3.67 |
| 2026-11-21 | Noise pilot 2, 4 requests | 3.6 | $0.19 | $3.86 |
| 2026-11-21 | Noise, full run (took twice as long as estimated: generation is slow) | 29.0 | $1.50 | $5.36 |
| 2026-11-21 | Early-exit follow-up, first attempt (failed preparing the data) | ~1 | $0.05 | $5.41 |
| 2026-11-21 | Early-exit follow-up on public datasets, both checkpoints | 16.9 | $0.87 | $6.28 |
| 2026-11-21 | Diagnostic look at raw TwoHopFact outputs | ~2 | $0.10 | $6.38 |
| 2026-11-21 | Pilot of TwoHopFact with "Fact:" framing (led to not running the full set) | ~2 | $0.10 | $6.48 |
| 2026-11-21 | Lens readout of intermediates on Anthropic's multi-hop set | 2.6 | $0.13 | $6.61 |
| 2026-11-21 | Independent review, arithmetic and capital swaps including one failed import (elapsed job time used as a conservative proxy) | ~5.1 | $0.26 | $6.87 |
| 2026-11-21 | Transplant pilots: unaided accuracy only, three short runs to fix prompt wording | 5.0 | $0.26 | $7.13 |
| 2026-11-21 | Four-outcome message transplant, both checkpoints | 6.0 | $0.31 | $7.44 |
| 2026-11-21 | Two-table message transplant, both checkpoints | 7.3 | $0.38 | $7.82 |
| 2026-11-21 | Shuffled-rows transplant, first attempt (out of GPU memory) | ~1 | $0.05 | $7.87 |
| 2026-11-21 | Shuffled-rows transplant, both checkpoints | 11.2 | $0.58 | $8.45 |
| 2026-11-21 | Held-out pointer and localisation test, both checkpoints | 7.3 | $0.38 | $8.83 |
| 2026-11-21 | Transplant by prompt region, released model | 6.5 | $0.34 | $9.17 |
| 2026-11-21 | Transplant by message part, released model | 11.9 | $0.62 | $9.79 |
| 2026-11-21 | Oversight test pilot, two table sets | ~1.5 | $0.08 | $9.87 |
| 2026-11-21 | Oversight test, three splits, released model | 25.0 | $1.29 | $11.16 |
| 2026-11-21 | Oversight test, second attempt: pilot | ~1.5 | $0.08 | $11.24 |
| 2026-11-21 | Oversight test, second attempt: attention readout with controls | 15.3 | $0.79 | $12.03 |
| 2026-11-21 | Front / end / token-position transplant, released model | 1.3 | $0.07 | $12.10 |
| 2026-11-21 | Spontaneous wrong routes and readout false alarms on unmodified runs | 3.4 | $0.18 | $12.28 |
| 2026-11-22 | Pilot of distracting wordings (no wrong routes produced) | 7.9 | $0.41 | $12.69 |

Budget: stop and ask at $20 total, or before any single run projected above $5.
