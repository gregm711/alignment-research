# Archived LLM judgements under hidden duplication

## Protocol before scoring

Use the authors' LLMBar release at commit `900616bff90b6c6c8e1681f7d079250637c55992`. Include all six model names with the Vanilla, Vanilla_NoRules and CoT prompts, giving 18 actual judge configurations. Read the first original-order verdict from each result file. An invalid verdict abstains. Match judge files on instruction, both candidates and benchmark label. Use the 100 Natural items only for setup, selecting a best single judge and competence weights there. Evaluate all four Adversarial subsets. Hash-sort inputs with a fixed public salt. Where an instruction or a pair of options repeats across setup and evaluation, drop the overlapping evaluation items; check repeat instructions within evaluation when computing uncertainty. LLMBar provides an ordering task and independently assigned preference labels, with no deployment timestamps or annotated copying relationships. No downloaded text or model responses will be committed. [Authors' data and protocol](https://github.com/princeton-nlp/LLMBar)

On the native panel compare majority, best calibration judge (fall back to majority if it abstains), calibrated positive log-odds weights, and inverse-neighbor weights using eight unlabeled prior items. Neighbors agree on at least seven of eight binary judgements, treating any abstention as a disagreement. Every judge counts itself once. Resolve ties by choosing the first answer. Compare refresh every 1, 2, 4, 8, 16, 32 or 64 items, freezing at setup, and the previously fixed disagreement trigger. Start each subset with the same last eight Natural ballots in public hash order. Archive item order has no temporal meaning; clock results are an algorithmic sensitivity only. The chronological tests live in the sensor and forecast experiments.

In a separate controlled stress test copy a single observed Vanilla ballot sequence 20 more times, rotating through all six base models so the choice of a weak origin cannot determine the result. Duplicate on setup and evaluation, hide the origin and keep the original labels. Compare the same learners to a cheap control that collapses exact prior eight-ballot histories into one unit of influence. Predict that continuous redundancy weighting improves on majority by at least five points averaged over the six injected origins, while achieving a gain of at least two points on the untouched panel is uncertain. A practical claim requires the untouched-panel gain and matching the stronger calibrated controls. Bootstrap instruction clusters paired by method; repeated artificial copies and alternative origins create no new test questions. Report the real unaltered panel first.

## Results

All 319 Adversarial items qualified; there were 318 distinct instruction strings and 15 abstentions among 5,742 recorded original-order ballots. On the unmodified panel, updating redundancy on every item improved raw majority by 6.90 points (paired instruction bootstrap 95% interval 3.14 to 10.62). The same estimator met only the first half of the practical criterion: choosing a single judge on the separate Natural calibration panel did substantially better. That selection was GPT-4 with the vanilla prompt. It answered 252/319 correctly, including 78/92, 35/47, 35/46 and 104/134 across the four subsets.

| Receiver | Untouched correct / 319 | Untouched accuracy | With 20 injected copies, averaged across six origins | Map builds, untouched |
|---|---:|---:|---:|---:|
| Majority | 139 | 43.57% | 49.06% | 0 |
| Best calibration judge | 252 | 79.00% | 79.00% | 0 |
| Calibration-weighted vote | 161 | 50.47% | 49.11% | 0 |
| Every item | 161 | 50.47% | 50.31% | 319 |
| Every eighth item | 159 | 49.84% | 49.53% | 41 |
| Disagreement trigger | 150 | 47.02% | 41.54% | 25 |

The predicted five-point lift under duplication failed. Refreshing every item gained 1.25 points over majority (interval −1.94 to 4.49). With a copy majority, naive voting largely follows whichever model we copied. Those origins vary in competence; copying one particularly good judge can improve this pool. Counts across injected origins use the same 319 labels. No natural common data source was identified.

For the unmodified panel, median aggregator CPU energy per item was 0.18 µJ for majority, 0.94 for the calibration-selected judge with fallback, 12.29 for the eight-item clock, 17.29 for the gate and 51.11 for a rebuild every item. These implementation-specific timings include loading calibration ballots from memory; the cost of the original model generations is unavailable. All conditions replay archived ballots. In an exploratory order sensitivity over 16 permutations within subsets, the gate solved between 139 and 179 items. No deployment change timing can be inferred from these item orders. Benchmark labels measure instruction-following preference on these selected pairs. Reuse the cached pinned data with `python3 run.py`. [Raw counts, paired intervals and timings](results.json)
