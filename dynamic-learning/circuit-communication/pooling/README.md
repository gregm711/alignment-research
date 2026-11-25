# Pooling many weak signals with shared sources

## Registered protocol, before scoring

A central receiver predicts a balanced hidden binary state. In a local episode one randomly located relevant cell receives a 90%-reliable reading and the others receive independent irrelevant fair bits. In a weak episode every cell receives an independent 57%-reliable reading. In a copied episode exactly 20 randomly located cells relay the same original 57%-reliable reading, and all remaining cells receive independent 57%-reliable readings. Membership and opaque source IDs are freshly shuffled. The incorrect-upstream slice conditions on the shared original disagreeing with truth and uses the naturally occurring frequency. This slice is available only to the grader. A single optional independent external check has 82% reliability. The receiver sees the quality class and source identity only in a transmitted packet. Cells know their own quality class. No policy receives the episode type or latent label. Source identity denotes a perfect common origin in the generator.

In public fixed order the receiver polls a block of four cells at a time. A cell can keep its reading private or send its sign, quality class and source ID. The receiver can stop, poll another block, or request the independent check once and then stop. Polls and empty responses each cost one bit per contacted cell. A nonempty response adds one sign bit, two quality bits and ceil(log2 N) identity bits. The request and returned independent sign cost two bits. Every contacted cell counts as active, including silent cells. This is a serial block schedule with prearranged addressing; physical hop distances, headers, network synchronization and hardware parallelism are outside its cost model. Public scheduling suppresses a potential timing side channel. Uncontacted cells incur no network traffic in this model.

Train on fresh balanced mixtures at 24 and 48 cells with seeds 101 and 102. Estimate class log-likelihood weights from independent first-origin readings during supervised training, with additive smoothing. Fit the repeat contribution (0, quarter, or full repeated weight) by full-message training log loss. Learn a finite routing policy by empirical risk minimization. Send all, informative, or only strong readings; stop above a confidence threshold; optionally request when posterior confidence is low at the terminal budget. Search quotas of one quarter, one half, or all cells, stopping thresholds 1.3/2.4/infinity in log odds, and check thresholds 0/0.8/1.8 in absolute log odds. Objective is natural log loss + 0.00015 times link bits + 0.12 times independent checks. These prices define a single illustrative operating point.

Compare to an equally tuned analytic filter with the true likelihoods and exact source deduplication. Include all-cell, no-check broadcasts with and without deduplication, plus a strong-readings-only filter. Replay the learned policy's exact packets and independent answer to a receiver that treats every packet as independent. This isolates how identical evidence is interpreted. At each width and regime, report accuracy, log loss, Brier score, bits, polled cells, independent checks, timing, and independent-seed paired intervals. Record real execution time and explicitly mark unavailable electrical measurements.

Test 20 fresh seeds at widths 24, 48 and previously unseen 96, with 128 episodes of each kind per seed. Primary mechanism prediction at 48 cells: replay ignoring provenance raises log loss by at least 0.10 nats on the copied panel with a paired 95% interval above zero. Independent-panel replay log loss must be identical. Predict improved independent-panel all-broadcast accuracy from 24 to 96 cells. A larger architectural bet requires the learned policy to match all-message exact-filter accuracy within two points and save 20% of bits at all widths (equal weight per regime), plus lower cost-adjusted loss than the tuned analytic policy by at least 0.01 with a positive paired lower bound at width 48. Fail either criterion and report the boundary. These comparisons evaluate authentic supplied identifiers and a tiny policy class. Learning latent provenance and parallel scaling need separate tests.

## Outcome

On 48-cell episodes containing 20 copies, the learned receiver achieved **82.0%** accuracy. Its source-blind replay achieved **59.7%** with the identical messages, check answers and stopping points. Replay increased log loss by **0.878 nats** (paired seed bootstrap 95% interval 0.814–0.943). When the upstream reading was wrong, the two accuracies were 75.1% and 6.6%. Replay increased log loss by exactly zero on independent episodes. The learner selected a repeat weight of zero from the training grid. This establishes the value of recognizing a supplied shared origin. Source identifiers are trusted inputs.

Both the learned and analytic selective policies retained the informative senders, stopped at an absolute log odds of 1.3, and bought a check when still undecided at the budget. They made the same decisions throughout the test. The 48-cell analytic-minus-learned cost-adjusted loss difference was −0.00009 (−0.00042 to +0.00024). The registered 0.01 advantage failed. The analytic broadcast's accuracy on independent weak signals rose from 74.7% at 24 cells to 92.6% at 96. At the held-out width, the selective policy stopped early, leaving information worth another 4.82 points of accuracy averaged across regimes (paired interval 4.26–5.40). Its 20%-bit saving and two-point accuracy requirement across all widths therefore failed.

| Cells | Selective accuracy | All-message exact accuracy | Selective bits | Full-format broadcast bits | Active cells, selective | Checks, selective |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 84.54% | 76.97% | 157.9 | 240 | 19.1 | 0.469 |
| 48 | 85.46% | 83.48% | 242.2 | 528 | 28.9 | 0.160 |
| 96 | 85.74% | 90.56% | 297.7 | 1,152 | 38.3 | 0.019 |

The full-broadcast controls consume zero external checks. The selective policy's additional measurement helps explain its accuracy at the smaller widths. Control formats also matter. At 48 cells a source-blind broadcast can omit the six identity bits it discards, giving 240 logical bits with otherwise identical fields. That is already below the selective policy's 242.2 bits; the cheaper source-blind broadcast achieved 57.5% on the copied panel. Coding every packet in the registered common format keeps evidence matched but does not establish a universal bandwidth advantage.

### Host measurements

At 48 cells, one prebuilt batch of 1,536 episodes ran 192 times per policy after a warm-up (294,912 episode equivalents). The Apple arm64 process CPU energy counter and wall clock surround only inference. The following single-run numbers are amortized across the batch.

| Policy | CPU energy (µJ/episode) | Wall time (µs/episode) |
|---|---:|---:|
| Learned selective | 3.60 | 1.02 |
| Analytic selective | 3.58 | 0.97 |
| Exact broadcast | 6.18 | 1.68 |
| Source-blind broadcast | 4.00 | 1.08 |
| Strong readings only | 1.81 | 0.48 |

Memory for source membership in this implementation is one boolean per candidate source and batch member. Endpoint IDs, quality classes, a fixed batch schedule and the binary likelihood family are given. Inference includes in-process indexing and decoding. Training, generating observations, network communication, the physical external sensor and whole-device electricity are outside the counter. Independent episodes are simulated in parallel on one CPU. There is no measurement of distributed speedup.

Run `OPENBLAS_NUM_THREADS=1 python3 dynamic-learning/circuit-communication/pooling/run.py` from the repository root. The frozen protocol and first evaluator are in commit `2f41fdc`. An accounting-only rerun removed unused dedup work for the source-blind control and extended timings to 192 repetitions. Every per-seed score reproduced byte for byte (SHA-256 `cf340e32f2bea79de7bb611e093c7f4903d2c33557e611de47ae6fb9898149e3`). Append `--timing` to remeasure the existing policies. [All means, paired contrasts and process costs](results.json) · [Raw per-seed results](seed-results.json.gz)
