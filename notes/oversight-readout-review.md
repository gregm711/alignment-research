# Review of the two oversight experiments

**Decision.** The twin-prompt construction yields a useful benchmark for induced mistakes. The first frozen monitor fails on it. The retry is a single-forward-pass detector for a model answering a seven-row lookup task whose correct row the reviewer already knows. Its visible-error calibration, statistical intervals and choice of threshold need separate checks before interpreting either a success or a failure.

Reviewed commit `b958503`, the working copy of `jlens-inkling/oversight2.py`, the completed first attempt and the second pilot. The full attention run had launched and had produced no held-out results at review time. The recommendations below precede seeing those results.

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

The pinned eager backend returns token attention weights as the second output of each sequence-attention module. Taking the final query row is appropriate. The indices 30–41 select the third pass and 42–47 select the post-loop layers. The pilot returned 18 × 24 finite scores on its sampled trial. I independently tokenized all 448 retry prompts. All 3,136 second-table row spans were disjoint within prompt, decoded to the intended entry and had five tokens each. The Pyright import and Modal decorator diagnostics did not block remote execution. [Attention hooks](../jlens-inkling/oversight2.py#L59) · [Pinned eager implementation](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L111-L142)

The head score normalizes away total attention to the table. A head can give its correct row a large share of almost zero mass. Save absolute attention mass and all row shares for the frozen head. Even a successful head is a predictive signal; establishing that it implements the route requires an intervention on the head. The state control predicts membership in a four-of-seven fruit partition, with the layer selected by in-sample AUC. It is a binary fruit-information check. It does not validate seven-way fruit identification, reproduce the first attempt’s layer selection, or demonstrate sensitivity to a donor-induced change. Log its selected layer. [Head score](../jlens-inkling/oversight2.py#L123) · [Fruit control](../jlens-inkling/oversight2.py#L171)

## Interpreting the retry

A frozen head that distinguishes different donors for the same visible prompt and answer, improves on confidence across table groups and retains a useful false-alarm rate would demonstrate internal-monitor utility for these induced, twin-confirmed mistakes. A successful visible control with chance hidden detection would show a failure of transfer from this visible training distribution. A failed visible control would leave the proposed calibration unvalidated. In all cases, report the predeclared predictions and the actual operating points. A fresh, position-balanced confirmation would address the main unresolved design issue.
