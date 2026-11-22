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
| 2026-11-22 | Phase 4 screen, six candidate models, 120 questions | L4 | 31.6 | $0.53 | $1.72 |
| 2026-11-22 | Phase 4 build pilots, three models, 12 questions | L4 | ~4 | $0.07 | $1.79 |
| 2026-11-22 | Phase 4 build, Qwen2.5-7B-Instruct | L4 | 69.7 | $1.16 | $2.95 |
| 2026-11-22 | Phase 4 build, granite-3.3-2b-instruct | L4 | 73.4 | $1.22 | $4.17 |
| 2026-11-22 | Label audit and all cleaned analyses (offline) | none | 0 | $0 | $4.17 |
| 2026-11-22 | Pooled analysis and literature check (offline) | none | 0 | $0 | $4.17 |
| 2026-11-22 | Phase 5 pilot, Phi-4-mini | L4 | 1.1 | $0.02 | $4.19 |
| 2026-11-22 | Phase 5 first attempt, three models, cut off at 300 known questions when the local connection dropped (progress kept and resumed) | L4 | ~128 | $2.13 | $6.32 |
| 2026-11-22 | Phase 5 resumed: Phi-4-mini 29.5, Qwen 58.1, granite 90.2 min | L4 | 177.8 | $2.96 | $9.28 |

Phase 5 was estimated at $2.50, then $3.50, and cost $5.11; the author had approved about $3 to $5.

Assumed cap for this project: $11. The Inkling project's ledger is separate
([`../jlens-inkling/SPEND.md`](../jlens-inkling/SPEND.md), $14.10).
