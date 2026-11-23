# Independent review of the Liars’ Bench probe comparison

Reviewed commit `f1a1783` on 23 November 2026. I used local pooled activations and the released transcripts. All new calculations ran on CPU. The five predictions appear in the earlier commit `fe1f846`. The three reported successes and two misses follow its stated cutoffs.

## What reproduces

Refitting the probes and outside baselines with scikit-learn 1.9.1 and NumPy 2.5.5 reproduces all seven within-subset AUROC tables to the reported precision. All six transfer tables also reproduce, including .739 on harm-pressure choice and .139 on insider trading for the last-token probe. The harm-pressure choice counts and wrong-answer-only refits reproduce. The mean response probe scored .913 overall and .827 among wrong answers. The last prompt token scored .877 and .729 respectively. The saved model identifiers agreed with the transcript model identifiers in all seven subsets. I did not regenerate the activation files.

I independently parsed all 2,134 choice responses as a single letter with optional terminal punctuation. All five numeric metadata fields matched a separate parser on every choice row. Every label equalled disagreement with the recorded neutral choice. Independently extracting the underlying question stems produced the same partition of rows into 1,128 groups as the repository’s five-word rule. No repeated stem straddled the primary folds. Reassigning these intact groups to folds gave .905 versus .693 overall and .795 versus .593 among wrong answers. With the original groups and scikit-learn 1.7.2 the overall probe score was .895. Pin versions and save fold assignments for an exact recount.

## What an outside monitor can see

I joined the benchmark authors’ published lie-judge scores to the 2,134 activation rows and verified exact agreement of row IDs and labels. No new judge calls were made. The previously reported interval against transcript text also reproduced in an independent paired bootstrap. All intervals below resample whole question groups and condition on the fitted scores. These additional comparisons were selected after seeing the original result. [Published judge scores](https://github.com/Cadenza-Labs/liars-bench/blob/main/results/blackbox/ext_judge_scoring_lie/harm-pressure-choice/llama-v3.3-70b-instruct.csv)

| Method | Overall AUROC | Response probe minus comparator, 95% interval |
|---|---:|---:|
| Mean response activation | .913 | |
| Transcript word and word-pair classifier | .691 | +.222 [.162, .287] |
| Published external transcript judge | .764 | +.149 [.098, .199] |
| Compare the chosen option to the answer key | .882 | +.031 [−.006, .074] |
| Rank wrong answers first, break ties with the transcript classifier | .908 | +.006 [−.031, .047] |
| Answer key and published judge fitted together | .913 | +.000 [−.031, .033] |

The tie-break uses the existing out-of-fold transcript score. It costs no additional model call. For the fitted comparison I used a standardized logistic regression with balanced class weights and C=1 on two inputs, the wrong-answer indicator and the published judge score. It used the same outer question folds and the repository’s fold normalization. Neither keyed comparator used the model’s neutral answer. Using a key assumes a different deployment setting from seeing only a transcript.

## Does the probe add information once the outside monitor is available?

The two standalone monitors reached the same AUROC but made different errors. Adding the response-mean probe to the answer key and judge raised AUROC from **.913 to .957**, a paired gain of **+.044 [.027, .063]**. This was an exploratory comparison on the same sample.

For each outer training fold, generate inner out-of-fold probe scores using four group folds. Fit a second logistic regression to the three training inputs (wrong-answer indicator, published judge score, inner probe score). Standardize its inputs on the training fold and use balanced class weights and C=1. Refit the first-stage probe on all outer training groups, apply both stages to untouched outer test groups, and concatenate the predictions. No outer test labels train either stage. Resample the outer out-of-fold scores by question for the interval. With the repository’s test-fold score normalization, the values are .913 and .957. Using raw decision scores gives .902 and .955, a gain of +.053 [.030, .077]. Both intervals condition on the fitted models and the exploratory feature choice.

This is evidence of complementary predictive information. Whether that information pays for access to activations depends on the available reference information, operating threshold and deployment costs.

## What the wrong-answer slice establishes

The response probe beats both the transcript classifier (.590) and the published judge (.590) among 118 wrong answers that contradict the neutral answer and 233 that agree with it. The paired lead over the external judge was +.236 [.141, .335]. A probe trained and tested exclusively on wrong answers scored .811. A binary correct/incorrect indicator cannot explain these results.

Of the 1,128 question stems, 31 had both labels. Across all answers they supplied 43 opposite-label pairs for the same stem. The response probe ranked 36 of 43 correctly and transcript TF-IDF ranked 26 of 43 correctly. After requiring both answers to be wrong, **only two** such pairs remained. Both scorers got both right. Most of the .827 AUROC therefore compares different questions. It can include differences in the kinds of questions on which the model changes its answer. This distinction matters for interpreting the result as access to a belief during a particular response.

The benchmark establishes its neutral reference using repeated prompts. Comparing a response with that stored reference recovers the label exactly. A fresh stochastic neutral query need not recover the reference. The original authors explicitly discuss belief awareness and possible false labels. [Benchmark construction and limitations](https://arxiv.org/html/2511.16035v1#S4)

## Other findings

- I tried selecting the text of the option named by the response before applying TF-IDF. This gave .611 overall. Adding selected-option words to the prompt gave .688, so this straightforward way of making the letter meaningful to a text classifier did not close the gap.
- For an exploratory alarm check, within each of the five outer folds I trained on four fifths of the training groups and used the remaining fifth to set the threshold at the 99th percentile of honest scores. This caught 25/134 contradictions (18.7%) and flagged 20/2,000 neutral-consistent answers (1.0%) across the untouched outer folds. This estimates one in-setting policy. There is no field cost or downstream utility measurement.
- Soft trigger pools three Llama adapters, each trained with a visible trigger. A transcript-only rule using the displayed hour scored .919 on the 2,000 time-adapter rows. There were 913/1,000 labeled contradictions before noon and 76/1,000 neutral-consistent answers before noon. The direction in the released rows agrees with the generation code and reverses the time labels in the paper’s Table 2. A learned classifier using only the first three words after the passage or a time-of-day feature reached .865 on all 6,000 rows. This subset tests whether a detector can exploit deliberately introduced prompt cues. [Benchmark Table 2](https://arxiv.org/html/2511.16035v1#S4) · [Trigger definitions](https://github.com/maiush/cadenza-sleeperer-agents/blob/main/liars/utils.py) · [Generator](https://github.com/maiush/cadenza-sleeperer-agents/blob/main/liars/preprocess.py)
- Convincing game and insider trading use row-level folds with two and one prompt groups respectively. Their reported intervals resample individual rows. They do not estimate transfer to independent scenarios. Gender-secret also uses an adapter. Small positive differences in the table should be described by magnitude. The best-of-several interval fixes the winners after selection. On the primary subset the fixed response-mean versus transcript comparison gives the same difference.
- The metadata parser silently converts an infinite harm factor to zero in 1,472 of 2,139 knowledge-report rows because the regex drops infinity and the table fills missing values with zero. The choice subset has no such rows. Report the metadata view as numeric fields only, handle infinity explicitly, and preserve a missingness indicator if a value can be absent.
- Appendix L of the benchmark paper is available. Its Table 8 gives .74 for its Llama 3.3 70B upper-bound probe on harm-pressure choice. Its training distribution differs, so this is contextual evidence. The README's earlier comparison also puts AUROC and balanced accuracy side by side as evidence of near-chance transfer. Those metrics answer different questions. [Benchmark Appendix L](https://arxiv.org/html/2511.16035v1#A12)
- The activation provider identifies the final token as an end-of-turn marker. The repository’s response mean includes this marker. In most choice rows this is a mean over the letter and the marker. The pre-answer position follows the provider’s documented assistant-content boundary. [Activation dataset card](https://huggingface.co/datasets/Yooniel/liars-bench-llama-3.3-70b-L53-acts)

The estimated $0.05 is consistent with a short Modal CPU pooling job in the project log. I did not verify a paid invoice. No gated transcript text or per-question predictions are included in this review.
