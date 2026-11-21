# Review of the two oversight experiments

**Decision after the full run.** The attention monitor passed its preregistered test on induced mistakes. It ranked the hidden mistake higher in all 139 matched comparisons, where the prompt and visible answer were identical. Output confidence did so in 70. This is evidence that an internal measurement adds information about the computation in this task. The monitor requires the correct evidence row as input. Generalization to errors the unmodified model makes on its own remains open.

I reviewed plan commit `b958503`, the first attempt, the retry implementation and pilot before the full run produced held-out results. Sections 1–5 and the interpretation criteria below record that review. The final section uses the completed run.

## 1. The first experiment selected a layer upstream of the mistake

The frozen direction reads `L6.pass1`. The donor intervention begins at pass 3. For a fixed recipient prompt, its pass-1 state cannot depend on which donor will be chosen later. On the same-wording test, 66 recipients have both labels. Their frozen scores tie exactly in 62 pairs. On the changed wording, 71 of 73 pairs tie exactly. The largest absolute within-recipient score differences are 0.000008 and 0.000016. The frozen threshold flags every trial in the new wording, giving both 100% detection and 100% false alarms.

The held-out AUCs of 0.477 and 0.486 reproduce. There is also substantive exploratory evidence for a narrow null. Across the eligible downstream layers and final mix, the best saved AUC from the same difference-of-means recipe is only 0.526 and 0.510, even allowing hindsight separately in each split. Restrict future causal readouts to states downstream of the intervention. Fit normalization within training folds and choose layers using entire held-out table groups. The original training folds split individual trials and can put different donor versions of one recipient on opposite sides. [State-probe training](../jlens-inkling/oversight.py#L187) · [Scores and operating points](../jlens-inkling/results/oversight.json)

## 2. The visible-error control has an absolute-position shortcut

`third = next(...)` chooses the first row holding a different fruit. At most the first two rows can hold the correct fruit. The visible-error destination therefore always falls in rows 1–3. I regenerated all planned candidates before the model-dependent filters.

| Split | Visible destination row 1 / 2 / 3 | AUC using earlier donor destination alone, visible | Same rule, hidden |
|---|---|---:|---:|
| Train | 100 / 32 / 12 | 0.868 | 0.500 |
| New tables | 86 / 26 / 8 | 0.881 | 0.500 |
| New tables and wording | 86 / 32 / 2 | 0.886 | 0.500 |

This diagnostic uses the designed donor destination. It quantifies the positional difference in the trial populations. It does not assert that the selected head implements that rule. A head sensitive to early locations can look good in visible calibration and transfer poorly to hidden cases. The control additionally changes answer identity relative to the correct answer; hidden trials deliberately hold it fixed. For the current run, stratify scores by intended destination and correct-row position and compare paired donors for the same recipient. For a confirmation, balance visible-error destinations across all seven rows independently of labels and use fresh table seeds. [Donor selection](../jlens-inkling/oversight2.py#L103)

## 3. Calibration uses counterfactual information

No hidden-positive score is passed to head selection. The right-route training examples nevertheless have to pass a patched twin test. An overseer with only a visibly correct answer could not select those verified negatives. Describe this as training on visible errors **and twin-verified correct routes**. To demonstrate calibration solely from observable labels, use clean, unaided correct trials as an additional negative class, then test the frozen rule on the patched matched pairs. Report any resulting sensitivity to whether a trial was patched.

The alternate fruit appearing on the twin is evidence of causal dependence on the altered row. An unchanged fruit is weaker evidence for the exact route. It is consistent with reading the intended row and with ignoring both rows. An additional twin that changes the intended row would strengthen the negative-class label. The current selection also discards unconfirmed trials. Accuracy estimates apply to the retained population. The two test seeds and wordings were used to evaluate the first attempt, so they are held out from the second fit but already familiar to the research process. [Twin test and retention](../jlens-inkling/oversight2.py#L114)

## 4. Check the statistical decision on whole tables and matched recipients

Each test split has ten independent generator groups, with shared facts, donors and multiple trials per recipient. The implementation bootstraps individual trials and reports separate intervals for each AUC. Keep the registered decision as written. Additionally, resample the ten generator groups with replacement, keep paired scores together and compute the interval for AUC(head) minus AUC(confidence). Report group-by-group sensitivity and the win/tie/loss count among recipients retained in both classes. Those comparisons hold the visible question and answer exactly fixed. Show observed false alarms and detections at the training threshold. AUC measures ranking across all thresholds.

As a check on feasibility, I regrouped the first attempt’s saved test rows by their original generator group. In 3,000 paired cluster-bootstrap resamples (NumPy seed 73), the differences against the reported top-probability baseline were −0.012 [−0.070, 0.030] on new tables and −0.023 [−0.053, 0.008] on changed wording. These intervals are approximate with ten groups. The retry saves held-out head scores, top probability and donor/recipient prompts, enough to repeat that analysis. It omits per-trial margin, head-average and fruit-control scores, so equivalent independent analyses of those endpoints require additional outputs. [Bootstrap](../jlens-inkling/oversight2.py#L135) · [Saved fields](../jlens-inkling/oversight2.py#L187)

## 5. The attention hooks are credible. The control and readout measure less than the write-up suggests

The pinned eager backend returns token attention weights as the second output of each sequence-attention module. Taking the final query row is appropriate. The indices 30–41 select the third pass and 42–47 select the post-loop layers. The pilot returned 18 × 24 finite scores on its sampled trial. I independently tokenized all 448 retry prompts. All 3,136 second-table row spans were disjoint within prompt, decoded to the intended entry and had five tokens each. All 384 planned original/twin prompt pairs differ at exactly one token, replacing the duplicate row’s fruit with the spare. The Pyright import and Modal decorator diagnostics did not block remote execution. [Attention hooks](../jlens-inkling/oversight2.py#L59) · [Pinned eager implementation](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L111-L142)

The head score normalizes away total attention to the table. A head can give its correct row a large share of almost zero mass. Save absolute attention mass and all row shares for the frozen head. Even a successful head is a predictive signal; establishing that it implements the route requires an intervention on the head. The state control predicts membership in a four-of-seven fruit partition, with the layer selected by in-sample AUC. It is a binary fruit-information check. It does not validate seven-way fruit identification, reproduce the first attempt’s layer selection, or demonstrate sensitivity to a donor-induced change. Log its selected layer. [Head score](../jlens-inkling/oversight2.py#L123) · [Fruit control](../jlens-inkling/oversight2.py#L171)

## Interpreting the retry

A frozen head that distinguishes different donors for the same visible prompt and answer, improves on confidence across table groups and retains a useful false-alarm rate would demonstrate internal-monitor utility for these induced, twin-confirmed mistakes. A successful visible control with chance hidden detection would show a failure of transfer from this visible training distribution. A failed visible control would leave the proposed calibration unvalidated. In all cases, report the predeclared predictions and the actual operating points. A fresh, position-balanced confirmation would address the main unresolved design issue.

## Completed retry and independent score checks

Head 23 at layer 11 in pass 3 was chosen on the training split and then frozen. Both controls cleared the predicted 0.90 AUC on both held-out splits. Visible-error AUC was 0.998 and 1.000. The binary fruit-control AUC was 0.998 and 0.985. The hidden-error head intervals cleared the confidence intervals in both splits, meeting the registered decision rule.

| Hidden mistakes | AUC, head / confidence | AUC gain, paired 95% interval | Detected at frozen threshold | False alarms | Same-recipient wins, head / confidence |
|---|---|---|---:|---:|---|
| New tables | 0.999 / 0.489 | +0.511 [0.488, 0.536] | 83/83 | 10/86 | 66/66 vs 34/66 |
| New tables and wording | 1.000 / 0.509 | +0.491 [0.465, 0.515] | 88/88 | 12/94 | 73/73 vs 36/73 |

I reproduced all four head and confidence AUCs for the hidden trials from the saved scores. Each interval above comes from 3,000 paired resamples of whole table groups, with ten groups in each split. Within individual groups the head AUC ranged from 0.975 to 1.000. Even averaging all heads without selection gave hidden-error AUCs of 0.936 and 0.965. The output-margin baseline was 0.489 and 0.515. The matched comparisons hold fixed the recipient prompt, correct evidence row and visible answer. They directly address the concern that the head learned which questions were likely to fail.

The early-destination shortcut in section 2 survived model-dependent filtering on visible trials (AUC 0.893 and 0.915). On hidden trials the same rule scored 0.511 and 0.520, and all seven destination rows occurred in each split. Early destination alone cannot account for the hidden result. Splitting the hidden trials by whether the donor fruit matched the recipient fruit also left the head AUC at or above 0.999 in all four subsets. These are retrospective checks. [Raw results](../jlens-inkling/results/oversight-attention.json)

**Two claims in the completed write-up need narrowing.** The fruit control establishes that a difference-of-means probe can decode a binary fruit partition. The claim that the final-token state “really does not carry the route” exceeds the test. The supported wording is “our state probe did not detect the route.” The median share on the intended row reproduces (81% and 73% on right-route trials, 2.6% and 3.5% on hidden errors). The saved scores show attention moving away from the intended row. Establishing that head 23 attended specifically to the duplicate row needs scores for that row. Establishing that it caused the answer needs a head intervention. [Result interpretation](../jlens-inkling/README.md#L1502)

**Next experiment.** Keep this head, direction and threshold fixed. Use fresh tables with duplicates and distractors, and collect answers from the unmodified model. On each originally correct answer, independently edit the duplicate row and the intended row to establish which edit changes the answer. Use those twin outputs only for evaluation and measure every monitor on the same original forward pass. Record the fraction of cases exhibiting each response pattern and report the unclassified cases. The decisive comparison is whether the head still beats output confidence on spontaneous wrong-route answers at a tolerable false-alarm rate. A scarcity of confirmed errors would establish a limit of the task for evaluating natural failures.
