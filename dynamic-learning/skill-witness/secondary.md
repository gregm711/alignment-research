# Conditional inference over a fixed program bank

## Registered before opening release outcomes

Initialize each producer's frozen eight-program bank using the ten paid first-release examples. Both readers start uniform and condition sequentially on those same ten labels before deployment. Fit no weights on either panel. Run three arms chronologically per producer. The trained sparse reader selects checks and updates its own state. An analytical likelihood reader with the synthetic generator's 0.22 uniform redraw and 0.08 random-response parameters selects maximum-entropy checks. A second analytical reader receives the trained reader's exact returned checks across the whole lifetime. If a check's output is outside every candidate, neither reader updates. Break equal output-mass ties by bank order. The fixed anchor serves as a simple baseline. All active arms pay two checks on every later version, recording performance before purchasing a check, after one and after two. Earlier revisions always purchase both. These figures measure prediction quality at matched cost. Fixed banks add per-producer resources.

Report whole-revision and row accuracy, equal producer weight in each panel, plus count of candidate-bank misses. Compute ceilings from the best single program in hindsight and independent best choice for every withheld row. Sum the UTF-8 JSON bytes of the banks. Bootstrap paired whole producer histories for native minus each analytical reader at two checks; report uncertainty descriptively for these selected producers. Predict positive performance against frozen anchor across revisions with changes, and expect the native reader to tie the analytic reader on identical evidence. Disclose selection imbalance, curves at ceiling, and synthetic-to-real mismatch.

## Outcomes

At two checks, every producer had identical average row accuracy and whole-revision success across the trained reader, the analytical reader and the analytical reader replaying the trained reader's observations. On dates, the readers reached the bank's ceiling with their first check. On phones, the original anchor was more accurate overall than any adaptive reader. Trained and maximum-entropy policies selected different check IDs on 32 of 600 phone purchases and 0 of 458 date purchases.

| Panel | Revisions | Changed revisions | Frozen anchor rows | Trained reader rows, 0 / 1 / 2 checks | Best single program in hindsight | Best program per row | Entire revision correct at 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dates | 229 | 7 | 86.25% | 89.58 / 90.83 / 90.83% | 90.83% | 90.83% | 90.83% |
| Phones | 300 | 26 | 61.56% | 60.72 / 60.82 / 60.82% | 63.10% | 75.54% | 13.33% |

The phone bank contained a program capable of solving the complete revision on only 13.33% of revisions, averaged across producers. On changed revisions, the active readers solved 41.67% of date revisions across six affected producers and zero phone revisions across nineteen affected producers. Phone row accuracy on those changed revisions was 54.86%, versus the anchor's 53.46%. This available-program bottleneck limits what the transfer test can resolve about the learned controller.

There were 758 verified calls per active policy on dates and 900 on phones, including 300 introductory calls per collection. No available candidate matched the returned label on 44 of 458 subsequent date purchases and 94 of 600 subsequent phone purchases for the trained reader. Frozen program banks occupied 10,841 and 39,146 UTF-8 JSON bytes across the respective collections. Persistent reader states add 16 bytes per producer for the trained reader, in addition to 268 bytes of shared parameters. Inputs, candidates' predicted outputs, and run-time arrays are additional resources. The ordinary analytical reader retains eight float64 values per producer in this implementation.

Run `python3 dynamic-learning/skill-witness/secondary.py ~/.cache/alignment-research/skill-witness/{dates,phones}.json` after building the panels. The output includes all producer-level averages, conditional breakdowns and the input, script and checkpoint digests. [Results](secondary-results.json)
