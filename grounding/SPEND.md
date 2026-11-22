# Spend log

Estimates only: GPU minutes times an assumed all-in rate for the GPU used on Modal
(about $1.00 an hour for an L4 with CPU and memory; about $3.10 for an A100-80GB).

| Date | Run | GPU | Minutes | Est. cost | Running total |
|---|---|---|---|---|---|
| 2026-11-22 | Gate pilot on Qwen3.5 (stopped: unsuitable architecture, no outputs used) | L4 | ~10 | $0.17 | $0.17 |
| 2026-11-22 | Gate pilot, 20 questions, two models | L4 | 4.2 | $0.07 | $0.24 |
| 2026-11-22 | Gate, 600 questions, two models | L4 | 21.7 | $0.36 | $0.60 |
| 2026-11-22 | Phase 1 pilot, 20 questions | L4 | 1.2 | $0.02 | $0.62 |
| 2026-11-22 | Phase 1, 1,462 questions with attention features | L4 | 34.1 | $0.57 | $1.19 |
| 2026-11-22 | Phases 2 and 3 (offline) | none | 0 | $0 | $1.19 |

Assumed cap for this project: $11. The Inkling project's ledger is separate
([`../jlens-inkling/SPEND.md`](../jlens-inkling/SPEND.md), $14.10).
