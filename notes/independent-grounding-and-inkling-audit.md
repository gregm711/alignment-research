# Independent recount of the grounding and Inkling claims

23 November 2026. Reviewed against commit 884db50. CPU-only review using the cached pminervini/NQ-Swap dev Arrow file, per-run answers, saved attention tensors and Inkling trial records. No new generations.

## NQ-Swap

Rejoining the original runs to substituted passages by question and replacement answer reproduced every headline count. In Phi, Qwen and Granite order, the word-overlap rule flagged 122/214, 106/171 and 112/205 original-answer runs, versus 410/2218, 462/2297 and 391/1989 passage-following runs. Applied without model or eligibility selection, the rule flags 1,077/4,746 dev rows (22.7%). These counts measure lexical ambiguity. They cannot establish how a model produced an individual answer. The rule can flag unrelated senses of a word and miss indirect evidence. GRAB-RAG independently documents surviving surnames in its own swap construction ([Appendix C](https://arxiv.org/html/2608.22228v1)).

The audit both filtered passages and changed the test split from question hash to answer hash. Crossing these decisions gives the following AUC for low first-token confidence on hidden versus grounded originals.

| Test split | Labels | Phi | Qwen | Granite |
|---|---|---:|---:|---:|
| Original | Original | .498 | .485 | .571 |
| Original | Filtered | .518 | .520 | .702 |
| Answer-grouped | Original | .614 | .594 | .580 |
| Answer-grouped | Filtered | .629 | .676 | .657 |

With the original split held fixed, selecting the evidence head again using only the corresponding visible training data, its visible/hidden AUC moved from .882/.575 to .904/.623 in Phi, .863/.599 to .786/.711 in Qwen, and .899/.641 to .902/.755 in Granite.

## Frozen transfer

Rebuilt the regularized logistic classifier on the clean visible training runs from saved attention tensors, using the ratio, scaling and penalty in monitor.py. The three refits reproduced .800/.826/.786 on new questions, catching 14/32, 5/16 and 11/37 with 24/403, 7/465 and 14/393 false alarms. All 2,277 new model-question runs passed the study's closed-book scorer. Of those, 931 were set aside. Labels were checked from substitution outcomes and original-answer correctness. They describe behavior across prompts rather than directly identifying the source of an answer.

Recomputed pairwise AUC differences with half-credit for ties. Across 5,000 independent bootstrap draws, each draw resampled normalized first-gold-answer clusters jointly across models. The attention lead over low top-token probability was .181 [.079, .285]. Restricting pairs to the same question kind gave .064 [-.059, .198]. The exploratory passage-head lead within kind was .151 [.065, .236].

### Additional exploratory baseline

Fit a separate scikit-learn LogisticRegression with default L2 regularization (C=1), intercept and no scaling per model. Use all cleaned substituted training runs labeled memory or passage, with memory as the positive class. Four inputs in order are negative first-token top probability and indicators that the first lowercase question word is who, when or where. Fit once on visible runs and score the new original-passage rows. No search over penalties or feature sets. The feature choice was made after the question-kind confound was known.

| New-item AUC | Phi | Qwen | Granite |
|---|---:|---:|---:|
| Frozen attention classifier | .800 | .826 | .786 |
| Question word and output confidence | .722 | .896 | .746 |

Equal-model mean attention advantage .016, paired cluster bootstrap 95% interval [-.037, .070]. This directly compares two deployable scoring rules using the saved measurements. The interval conditions on the fitted models. It does not capture uncertainty from selecting this comparison post hoc.

## Inkling

Recounted the original hidden per-trial head scores in oversight-attention.json, obtaining .9993 and 1.000 AUC across wordings. Applying its exact threshold .45085132122039795 to causes.json yielded AUC .889/.862 in the two upstream transplant draws, with 29/29 and 25/26 wrong-route runs caught and 50/83 and 51/79 right-route runs flagged. The untouched condition flagged 50/370 right-route runs. The 62.3% refers specifically to the upstream intervention and adjudicated right routes. Evidence covers artificial table lookups and a specified intervention.

## Publication judgment

Publish the benchmark-specific counts, examples, scoring rule and downstream consequences as a technical note. The transferable ranking signal and its comparison to a cheap question-aware baseline belong alongside the audit. Keep the passage-head result exploratory. Verification here did not regenerate the upstream benchmark or model inference, adjudicate every flagged passage, or establish literature priority.
