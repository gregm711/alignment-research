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

Budget: stop and ask at $20 total, or before any single run projected above $5.
