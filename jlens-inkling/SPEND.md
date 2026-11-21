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

Budget: stop and ask at $20 total, or before any single run projected above $5.
