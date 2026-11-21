**Verdict.** The donor’s answer row predicts the saved full pass-3 top token on 145/150 new-table pairs for Inkling 2 Mini and 123/150 for its sibling. The released model clears its advance criterion. The experiment establishes a behavioural regularity under intervention. Several distinct message contents could produce it. The wording had already been tested, the fixed layout makes several position rules identical, and repeated donors predict systematic exceptions. The headline counts survive. The shuffled sibling denominator, the pass-2 interpretation and several localisation claims need correction.

## 1. What was checked

At revision `594bfa4`, I executed the deterministic prompt generators with an inert Modal module. I matched both prompts by exact text, parsed the two printed lists and the question, and computed the predictions from those strings. For the central rule, find the donor’s queried animal, look up its colour, locate that colour in the donor’s printed second list, and take the fruit at the same index in the recipient’s printed second list. Each fruit occurs once per list, so locating the donor fruit gives the same index. I compared the entire saved decoded top token, including its leading space, before consulting outcome labels or aggregate counts.

The six files contain 1,710 lookup pair records and 5,370 saved lookup condition outputs. Every saved expected lookup word and every stored outcome label agrees with the independent reconstruction **under the script’s priority convention**. I separately checked the real-facts rows. All 142 numeric body cells compared across the seven README result tables agree after folding the control-fruit category into “other” where the README does so. The sibling denominator error is in a column heading. The prior review reproduced a different capital-swap experiment. This review recounts saved lookup predictions and tests the hook on CPU tensors. It does not independently reproduce model inference. [Audit data, input hashes and failure details](../jlens-inkling/results/independent-review-lookup.json) · [Original experiment](../jlens-inkling/transplant.py) · [Earlier review](independent-review-2026-11-21.md) · [Earlier row-order control](lookup-position-control.md)

## 2. The central rule on every saved pair

“Full” means every message part at every input position into the indicated pass. The pass-2 denominators are the same as for pass 3 in their respective files. The two released-model localisation files contain exactly the same ordered prompt pairs as the pointer file. Joining by both prompt strings supplies their full-control result. Their own most inclusive condition covers only the second table and question and scores 140/150 in both files, with identical top tokens pair by pair.

| Result file | Mini, full pass 3 | Sibling, full pass 3 | Mini / sibling, full pass 2 |
| --- | --- | --- | --- |
| four-way | 96/96 | 96/96 | 71 / 61 hits |
| two-table | 144/144 | 133/138 | 111 / 20 hits |
| shuffled | 316/324 | 259/312 | 54 / 46 hits |
| pointer | 145/150 | 123/150 | No run |
| regions | 145/150 (joined) | No run | No run |
| parts | 145/150 (joined) | No run | No run |

All full **pass-3** misses are listed below. There were zero in either four-way file block or the released two-table block. The regions and parts joins inherit pointer/Mini’s misses. The companion JSON gives each recipient and donor side and animal, group, donor answer-row index, predicted fruit and exact observed token. The real-facts rows have no fruit lists, so the rule has no defined prediction there.

| File and model | Failures | Rows in original JSON (first row = 1) |
| --- | --- | --- |
| two-table/sibling | 5 | 77, 86, 87, 96, 104 |
| shuffled/mini | 8 | 72, 77, 81, 200, 210, 213, 222, 322 |
| shuffled/sibling | 53 | 9, 11, 13, 18, 30, 33, 46, 86, 88, 97, 104, 109, 118, 121, 124, 125, 136, 157, 158, 159, 161, 162, 163, 165, 166, 167, 168, 170, 171, 172, 174, 180, 183, 184, 195, 200, 202, 204, 207, 211, 216, 236, 241, 242, 248, 249, 251, 259, 269, 270, 273, 276, 282 |
| pointer/mini | 5 | 7, 82, 111, 137, 143 |
| pointer/sibling | 27 | 1, 11, 12, 14, 19, 22, 24, 31, 34, 41, 51, 65, 70, 79, 84, 92, 100, 102, 109, 112, 114, 119, 126, 129, 130, 144, 145 |

## 3. Recount of the original outcomes

**Four-way.** R and D denote recipient and donor. On these prompts the recipient second step on the donor item, colour and row all predict the same fruit.

| Pass | Model | Pairs | Recipient | Donor | R step on D | D step on R | Other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Mini | 96 | 0 | 0 | 96 | 0 | 0 |
| 3 | Sibling | 96 | 0 | 0 | 96 | 0 | 0 |
| 2 | Mini | 96 | 23 | 0 | 71 | 1 | 1 |
| 2 | Sibling | 96 | 31 | 3 | 61 | 1 | 0 |

The README calls 71 and 61 the numbers of answers that moved at pass 2. Those are hits for one named outcome. **Any change from the clean recipient answer occurred on 73/96 and 65/96**, respectively. Both still move less than at pass 3. The analogous capital comparison mixes statistics too. The earlier independent per-pair file gives 8/42 donor-language hits and 11/42 changed answers at pass 2. [Four-way rows](../jlens-inkling/results/transplant-four-way.json) · [Capital rows](../jlens-inkling/results/independent-review.json)

The real-facts block contains zero eligible Mini pairs and 12 sibling pairs. Sibling pass 3 yields one recipient, one donor, one donor step on the recipient, and nine “other.” All 12 retain the recipient at pass 2. The best recorded Mini clean score is 8/14. No per-prompt clean tops accompany that maximum. The selected real-facts template asks **language or continent**. The final code tries three fact wordings, although the design describes two and illustrates language/currency. The README’s two preliminary wording pilots and the spend ledger’s three pilot runs may count different stages. The original pilot top tokens and the division into runs cannot be reconstructed from these pair files. [Spend ledger](../jlens-inkling/SPEND.md)

**Two-table.**

| Pass | Model | Pairs | Recipient | Donor | Colour | Animal | D step on R | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Mini | 144 | 0 | 0 | 0 | 144 | 0 | 0 |
| 3 | Sibling | 138 | 3 | 0 | 1 | 133 | 0 | 1 |
| 2 | Mini | 144 | 32 | 0 | 0 | 111 | 0 | 1 |
| 2 | Sibling | 138 | 114 | 1 | 2 | 20 | 0 | 1 |

The carried-colour prediction lost decisively, but the sibling produced it once at pass 3 and retained its own answer three times. The statement “the donor’s colour is never what determines the answer” overgeneralises even as a description of token matches. The four-way design itself gives the colour prediction 96/96 on each model. [Two-table rows](../jlens-inkling/results/transplant-two-table.json)

**Shuffled.** The original classification scores the donor’s second-list answer row only on pairs where its fruit differs from all five core alternatives.

| Pass | Model | Pairs | Recipient | Donor | Colour | Animal | Animal row | Distinct answer row | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Mini | 324 | 56 | 60 | 37 | 33 | 24 | 110 | 4 |
| 3 | Sibling | 312 | 51 | 52 | 64 | 34 | 20 | 81 | 10 |
| 2 | Mini | 324 | 124 | 41 | 32 | 45 | 10 | 4 | 68 |
| 2 | Sibling | 312 | 187 | 27 | 22 | 33 | 10 | 1 | 32 |

The saved sibling rows have **106** distinct sixth outcomes, yielding **81/106**. The README prints 105. On Mini the figure is 110/112. The following split counts the rule on every pair and shows precisely how the original categories hide its hits. A core category sometimes wins even when the pointer predicted another fruit, so the original category totals are not an exact partition of pointer hits. [Shuffled rows](../jlens-inkling/results/transplant-shuffled.json)

| When the pointer fruit is… | Mini, hits / cases | Sibling, hits / cases |
| --- | --- | --- |
| distinct | 110/112 | 81/106 |
| recipient | 56/56 | 45/53 |
| donor | 57/57 | 45/55 |
| carried_colour | 36/39 | 39/39 |
| carried_animal | 33/36 | 30/35 |
| carried_row | 24/24 | 19/24 |

**Pointer and localisation conditions.** “Recipient first” includes donor/recipient ties. The last column counts every top token equal to the donor fruit and overlaps the recipient column. The first five numeric columns form a partition. The published “other” is the sum of the control-fruit and remaining-other columns.

| Model and condition | Pointer | Recipient first | Donor only | Control fruit | Remaining other | Donor hits incl. ties |
| --- | --- | --- | --- | --- | --- | --- |
| Mini, full | 145 | 2 | 0 | 0 | 3 | 0 |
| Mini, only_row | 1 | 149 | 0 | 0 | 0 | 28 |
| Mini, all_but_row | 47 | 20 | 16 | 14 | 53 | 20 |
| Mini, only_fruit_token | 0 | 150 | 0 | 0 | 0 | 29 |
| Mini, other_row | 0 | 149 | 0 | 1 | 0 | 29 |
| Sibling, full | 123 | 5 | 1 | 6 | 15 | 1 |
| Sibling, only_row | 0 | 148 | 0 | 1 | 1 | 30 |

| Model and condition | Pointer | Recipient first | Donor only | Control fruit | Remaining other | Donor hits incl. ties |
| --- | --- | --- | --- | --- | --- | --- |
| Mini, table1 | 0 | 147 | 0 | 1 | 2 | 28 |
| Mini, table2 | 6 | 67 | 15 | 11 | 51 | 32 |
| Mini, question | 34 | 15 | 5 | 22 | 74 | 7 |
| Mini, last_token | 0 | 150 | 0 | 0 | 0 | 29 |
| Mini, table1+table2 | 8 | 41 | 18 | 15 | 68 | 28 |
| Mini, table1+question | 26 | 18 | 10 | 22 | 74 | 12 |
| Mini, table2+question | 140 | 2 | 1 | 3 | 4 | 1 |

| Model and condition | Pointer | Recipient first | Donor only | Control fruit | Remaining other | Donor hits incl. ties |
| --- | --- | --- | --- | --- | --- | --- |
| Mini, all four parts | 140 | 2 | 1 | 3 | 4 | 1 |
| Mini, keys at table 2 + queries at question | 11 | 132 | 0 | 1 | 6 | 27 |
| Mini, queries and keys | 7 | 136 | 0 | 1 | 6 | 26 |
| Mini, keys and values at table 2 + queries at question | 30 | 67 | 10 | 10 | 33 | 25 |
| Mini, values only | 32 | 90 | 4 | 4 | 20 | 22 |
| Mini, expert-layer only | 0 | 147 | 0 | 0 | 3 | 29 |
| Mini, values and expert-layer | 51 | 66 | 5 | 6 | 22 | 19 |

The unrelated-row control produced its **own control fruit on 1/150** and the pointer fruit on 0/150. The predeclared under-10% criterion referred to the former. The row-only condition on the sibling also exists in the JSON. [Pointer rows](../jlens-inkling/results/transplant-pointer.json) · [Region rows](../jlens-inkling/results/transplant-regions.json) · [Part rows](../jlens-inkling/results/transplant-parts.json)

## 4. Which explanations the prompts distinguish

Each cell gives full pass-3 hits for Mini / sibling. Counts here allow overlapping predictions. For the baseline prompt-derived rules, the audit JSON gives every pairwise disagreement count independently of which fruit the model produced. Character-offset and repeated-donor checks are recorded separately.

| Prediction | Four-way | Two-table | Shuffled | Held-out |
| --- | --- | --- | --- | --- |
| Donor answer row and all equivalent rules below | 96 / 96 | 144 / 133 | 316 / 259 | 145 / 123 |
| Unchanged recipient fruit | 0 / 0 | 0 / 3 | 56 / 51 | 2 / 5 |
| Copy donor fruit | 0 / 0 | 0 / 0 | 60 / 52 | 0 / 1 |
| Recipient second table on donor colour | 96 / 96 | 0 / 1 | 37 / 64 | 28 / 37 |
| Both recipient tables on donor animal | 96 / 96 | 144 / 133 | 33 / 34 | 40 / 29 |
| Donor animal’s first-list row → recipient second list | 96 / 96 | 144 / 133 | 24 / 20 | 26 / 28 |
| Donor animal’s first-list row → recipient first-list colour → recipient fruit | 96 / 96 | 144 / 133 | 36 / 52 | 12 / 26 |
| Donor second table on recipient colour | 0 / 0 | 0 / 0 | 32 / 39 | 30 / 27 |
| Mirror the donor answer row in recipient list | 0 / 0 | 0 / 1 | 44 / 42 | 28 / 32 |
| First recipient fruit | 24 / 24 | 24 / 21 | 41 / 32 | 20 / 12 |
| Last recipient fruit | 24 / 24 | 24 / 23 | 39 / 53 | 26 / 29 |

- **Row versus physical position.** Across every full pair, identical predictions come from counting the donor answer row from the front or back, taking the fruit at the donor fruit token’s absolute index or distance from the prompt end, or looking up the recipient colour at the donor colour token’s index. On the seven-row prompts, colour-token indices are 49, 55, 61, 67, 73, 79, 85 and fruit indices are exactly four tokens later. Even choosing the recipient fruit whose *character start* is nearest the donor fruit’s character start gives the same prediction on every pair, measured from either prompt start or second-list start. Mirroring the row is distinguishable. Counting from the same end on both sides is not.
- **Answer content versus address.** Another exact predictor builds the dictionary pairing donor fruit words with recipient fruit words in display order and applies it to the donor’s finished answer. A second equivalent computation takes the recipient colour at the aligned donor colour position, then looks up its fruit. These rules make different demands on information that could cross the boundary. No observed output distinguishes them. The failures of *verbatim* donor copying leave open a donor answer or colour signal used together with other information.
- **Eligibility creates a recipient-only solution in the ordered runs.** Four-way side B rotates the fruit assignment by one place and keeps the animal and colour order. Requiring four distinct outcomes leaves exactly one donor per recipient. Take the recipient’s own answer row and move forward two modulo four. This matches 96/96 on both models without reading the donor. In the two-table design B rotates the animal-to-colour assignment by one and the colour-to-fruit assignment by two. Five distinct outcomes again leave at most one donor per recipient. Advance the recipient answer row by four modulo six for recipient side A or by two for side B. This gives 144/144 and 133/138. Side is a generator label, so the two-table shortcut uses that design information. Neither calculation identifies what the model actually does. Both ordered generators print the same animal order on both sides and print the second list in matching animal order.
- **Shuffling separates the original semantic rivals.** With independent printed row orders, the first-list same-row shortcut is the correct *clean answer* on just 14/168 generated prompts. On shuffled pairs where its fruit differs from all five core predictions, the answer-row rule wins 110/112 on Mini and 81/106 on the sibling. Using the same separation criterion on the held-out sample gives 67/70 Mini and 67/78 sibling. Where the donor-animal and pointer predictions disagree on held-out pairs, the score is 105 to 0 on Mini and 94 to 0 on the sibling. This is direct evidence against unconditional animal transfer. A claim that the model used a particular sequence of internal lookup operations goes beyond it.
- **The paired tables still move together.** Shuffled and held-out sides retain the same assignment rotations as the two-table generator. Each donor prompt occurs with only one opposite-side recipient *table layout*. Varying the recipient question does not change that layout. A codebook specific to the donor and its paired layout can therefore reproduce the pointer prediction without reading the current recipient. The dataset never holds one donor fixed while changing the recipient’s second-list permutation.

A post-hoc diagnostic quantifies this last issue. Predict a row’s observed top token using the most frequent top on the *other* saved rows with exactly the same donor prompt. Break ties lexicographically. The comparison below uses identical coverage. The last column uses the answer-row prediction for singleton donors.

| Sample | Repeated-donor rows | Answer-row hits there | Same-donor vote hits there | Vote with row fallback, all rows |
| --- | --- | --- | --- | --- |
| shuffled/mini | 324 | 316 | 318 | 318/324 |
| shuffled/sibling | 312 | 259 | 296 | 296/312 |
| pointer/mini | 117 | 112 | 117 | 150/150 |
| pointer/sibling | 121 | 101 | 114 | 136/150 |

On Mini held-out rows 7, 111 and 143, the same donor about the pig predicts olive by the row rule, yet all three recipients output grape. This helps explain why a same-donor vote can do better. It uses other saved intervention outcomes and cannot predict a genuinely new donor on its own. It is evidence of clustering and a warning against random pair-level validation. It is not a proposed circuit or an independent out-of-table replication.

## 5. Eligibility, holdout and token scoring

| Generator | Structurally allowed ordered pairs | After distinct-outcome filter, before clean checks | Pairs scored, Mini / sibling | Distinct recipient prompts, Mini / sibling |
| --- | --- | --- | --- | --- |
| Four-way | 288 | 96 | 96 / 96 | 96 / 96 |
| Two-table | 720 | 144 | 144 / 138 | 144 / 138 |
| Shuffled | 1,008 | 324 | 324 / 312 | 156 / 152 |
| Held-out | 672 | 526 | 150 / 150 sampled from 504 / 440 | 88 / 86 |

Structural eligibility requires the same group, opposite sides and different query animals. All generated pairs passing these structural checks were token-length matched within a wording. `prepare` tests the whole next-token ID before pair selection. The four-way branch requires **all four** prompts, including both crossed questions, to be clean-correct. The later branches require the two endpoints. `five_outcomes` and `shuffled_outcomes` agree with the independent prompt parser. In the shuffled branch the first five words must differ. The optional sixth is deleted only when its word duplicates a core prediction. Deletion loses an explanatory label, and leaves the scored top token intact.

**Which clean counts can be recovered?** The following are saved clean aggregates in wording order. The pair files omit the individual clean predictions, especially for unused wordings.

| Family and wordings, in order | Mini correct | Sibling correct | Wording used, Mini / sibling |
| --- | --- | --- | --- |
| Four-way lookup, is_the / so / qa | 76, 96, 96 of 96 | 17, 94, 96 of 96 | so / qa |
| Real facts, speak_pay / language_continent / language_currency_the | 6, 8, 4 of 14 | 9, 11, 5 of 14 | language_continent / language_continent |
| Two-table, so / qa | 144, 144 of 144 | 141, 141 of 144 | so / so |
| Shuffled, so / qa | 168, 168 of 168 | 158, 166 of 168 | so / qa |
| Held-out, selected wording only | 110 of 112 | 104 of 112 | qa / so |

Using only the stored pair lists and reconstructed eligibility, I enumerated the possible clean flags for prompts absent from the rows. Four-way forces all 96 correct on both checkpoints. Two-table forces 144 on Mini and permits 138–141 on the sibling. Shuffled permits 162–168 and 159–166, respectively. The README’s high shuffled clean scores are consistent with the rows, but the reported exact counts and the sibling’s 0/2 same-row mistakes depend on the clean aggregates. Even the lower bounds exclude the simple across-row shortcut as a complete account of clean performance.

For held-out sampling I replayed `random.Random(0).sample` for every clean mask consistent with the observed endpoints, without supplying the saved clean or qualifying counts. Only one mask reproduced the exact ordered sample per model. It implies the reported 110/112 clean and 504 qualifying on Mini, 104/112 and 440 on the sibling. This verifies selection conditional on the generator and sampler. It does not recover the text of an incorrect clean prediction. The held-out samples share just 62 ordered structural pairings across checkpoints, after removing wording differences. The rates are conditional on clean-correct, aligned pairs where the pointer fruit differs from both endpoint answers. They are pair-weighted results from eight groups.

**New tables, familiar wording.** Comparing seed 3 with seed 2 gives zero shared full side mappings or rendered table layouts among the 16 new and 24 earlier sides. Neither individual full mapping overlaps either. Earlier four- and six-entry tables also differ in size. A different seed alone gives no such guarantee. All seven fruits recur. Of the 78 distinct new animal–colour facts, 58 already occurred, and of 65 new colour–fruit facts, 53 already occurred. Both models were run clean on *both* shuffled wordings. The pointer wording was unused for that model’s earlier shuffled **transplants**, but not unseen. The pointer criterion was in commit `23e5ff5` before the corresponding results commit `696eea5`. Mini achieved 96.7% against its 85% target. Sibling achieved 82.0%, with no advance sibling threshold.

**Single tokens and ties.** I encoded the prompts locally using the cached tokenizer. All 31 space-prefixed lookup vocabulary words have distinct single-token IDs. The expected gold fruit extends every regenerated prompt by that same single token, and the printed colour and fruit tokens decode to the expected words. The real-fact answer candidates also pass the boundary check. The tokenizer hash is in the companion JSON. I did not independently fetch a sibling-specific tokenizer. The saved sibling words, top tokens and row offsets are consistent with the local one.

`token(word)` returns `None` for a multi-token word. The four-way check rejects unscorable answers for all four prompts. Later distinctness tests use strings and do not assert all secondary outcomes have distinct, non-null token IDs. An unscorable secondary candidate would silently be unable to win. This is a latent problem for changed vocabularies. For these files, the material `next(...)` ambiguity comes from *identical words*. In the held-out data, the donor equals the recipient on 29 Mini and 30 sibling pairs. The pointer differs from both in every pair. The scorer assigns a tied top to the recipient. For example, on Mini’s row-only transplant the published donor column is zero, but the output equals the donor fruit on 28 rows, all ties. The pointer count is unaffected.

**Control-row assignment.** The code chooses an eligible control row using the running qualifying-pair index modulo the number of available rows, before sampling pairs. It excludes the target row and recipient rows containing either endpoint answer. There are four available choices on 121 Mini pairs and five on 29. The sibling split is 120 and 30. I recovered every saved control fruit from the qualifying-pair order. Control rows 1 through 7 occur 21, 23, 29, 19, 23, 20, 15 times on Mini. The control fruit even equals the recipient lookup on the donor animal on 31 Mini pairs. Calling it a different, outcome-filtered row is accurate. Calling it independently random or unrelated to all candidate explanations is not.

## 6. Hook, region boundaries and what the interventions show

In the pinned `Inkling2LoopTextModel.forward`, the first traversal of the loop has no message builder. Each later pass retains the preceding pass’s selected block/final states for the builder, restarts the hidden and block states from the recipient’s pre-loop states, then calls `message_input` once per loop layer before its decoder. With three passes and twelve loop layers, zero-based calls 0–11 feed pass 2 and 12–23 feed pass 3. Patching pass 2 also changes the inputs used to build the recipient’s later pass-3 messages. A pass-3-only patch leaves the earlier recipient computation in place. [Pinned forward](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1378-L1427)

`Inkling2LoopMessageAttentionResidual` mixes preceding states **at each token position**. Its within-builder attention is over the state/block axis. The four outputs per position have shape batch × position × part × hidden. Both decoder variants unbind the part axis in Q, K, V, MLP order and use *gates* to mix each message with a recipient hidden state. A Q/K-message patch therefore changes inputs to attention computations. Recipient queries and keys still exist with recipient messages on every unpatched cell. [Builder](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L145-L175) · [Decoder gates](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1110-L1146)

`on_message` records detached whole builder outputs on CPU, restores the donor tensor to the output device without a dtype conversion, and uses the corresponding call index. A full swap returns the whole donor output. A positional swap clones the recipient output and overwrites the selected token positions. The parts branch assigns the Cartesian product of positions and part indices in batch member zero. Every experiment forwards one prompt at a time. `forward` resets call state, disables caching, checks for 24 calls and takes the last-position vocabulary argmax. I ran the actual nested hook and forward wrapper on labelled CPU tensors. Six masks, including both full-pass choices, position-only and split part selections, agreed cell by cell with a separate mask implementation. I found no wrong layer, position or part replacement for these inputs.

Token-offset checks were independent on both prompts. All donor/recipient row masks, fruit masks, control-row masks and region token indices match their intended recipient counterparts on the sampled pairs. All saved `row_token_positions` match. Each answer-row mask covers its five lexical tokens from colour through fruit, excluding its trailing comma or period. A marker on the separator is therefore outside “only row.” Region masks include headers and punctuation. Table 1 includes the fact-sheet preamble, and table 2 and question each start at the preceding newline. The held-out Mini region lengths are 43, 48 and 17 tokens, versus 43, 48 and 11 for the sibling wording. No token straddles the region boundaries here. Equal *total* length alone would not protect this indexing with more varied templates.

The causal ordering matters. Within each donor side, every query has exactly the same token prefix through the second list. In a causal decoder those prefix positions cannot know which animal the later question will name. The reviewer’s patch mask selects the answer row using that later question. A row could still supply a query-independent key or value that participates in downstream selection. Row-only swapping gave 1/150, while putting the recipient’s target-row messages back into an otherwise full donor swap lost **98 of the 145** full-condition hits and gained none. The surviving 47 show that swapping this row is unnecessary on some pairs. Its large conditional contribution contradicts a claim that the row does nothing. These results do not locate a separately stored marker.

## 7. Sentence-level publication decisions

| Source and claim | Assessment | What the evidence supports |
| --- | --- | --- |
| SUMMARY item 2, “a place to read, not an answer”; “beat every alternative we could separate” | Behaviour supported, representation overstated | The row prediction beats each listed direct semantic rule on the shuffled data. A translated donor answer, several positional rules and a donor-specific paired-layout rule remain observationally compatible. |
| SUMMARY, “new tables and a new wording”; “168 of 168 unaided” | New tables supported, unseen wording incorrect, clean score aggregate only | Name the switched wording and selected checkpoint. Shuffled accuracy excludes simple same-row clean answering. It does not reveal the algorithm. |
| SUMMARY, “not stored at the row”; “needs…both”; “no subset…reproduces it” | Overstated localisation and unsupported exhaustiveness | The row alone rarely suffices. Table 2 plus question gives 140/150. Several incomplete masks also yield hits. Only selected subsets were tested. |
| SUMMARY, failed guesses, one invented task and no real-facts version | Scope supported with qualification | Specific predictions about animal transfer, row-only sufficiency and the selective Q/K swap failed. A real-facts version was built. The released model supplied zero eligible pairs and the sibling supplied too few for a comparable conclusion. |
| Four-way README, “messages do not carry a finished answer”; “final pass redoes the second lookup” | Output association supported, content and algorithm unsupported | The verbatim donor fruit occurs on 0/192 full pass-3 pairs. The recipient-only half-turn predicts the same changed fruits. A recipient top token also cannot establish that messages were ignored. The four-way named-outcome prediction holds for lookup only. The facts-family prediction is unresolved. |
| Two-table README, animal override and “rule out a carried colour” | Direct colour rule loses; inferred animal transfer unsupported | Ordered display makes animal and row predictions identical. The sibling gives a colour match. The earlier shared-first-table design could never exclude a colour explanation. |
| Shuffled README, “doing the two-step lookup”; “no listed explanation dominates”; rule found afterwards | Qualify the algorithm and the counting convention | No core category has a majority under exclusive scoring. The sixth rule was specified in advance and already wins 110/112 and 81/106 on distinguishable pairs. Its promotion to the main explanation and its all-pair rescoring were post hoc. Its extension to real-fact capital questions is a hypothesis. |
| Pointer README, prediction held; “therefore say where”; “marker is not stored at the place it points to” | Holdout threshold supported; mechanism unsupported | The two directional localisation predictions failed as written. Full transfer generalises within this generator. The target-row patch has low standalone sufficiency and a large effect in the context of the other donor messages. |
| Regions README, both regions needed; “first table’s messages are irrelevant”; matched selector and mark | Strong joint effect supported; absolute necessity and storage unsupported | The question alone gives 34/150 and the tables without the question give 8/150. Adding the first-table patch to table 2 plus question restores five full-control hits. A selector matching a table feature is a hypothesis. The “rare without the question” prediction is consistent with the data. |
| Parts README, “queries and keys barely matter”; “spread across the whole message”; rules out attention matching | Predicted Q/K sufficiency failed; component necessity and circuit exclusion unsupported | Q/K in the specified separate regions gives 11/150. Values alone exceed the under-10% prediction at 32/150. Expert-only meets it at 0/150. Adding Q/K to the same values-plus-expert mask increases hits from 51 to 140, gaining 89 and losing none. Only four of fourteen nonempty proper uniform subsets were tested in both regions. No uniform three-part subset was tested. |

The later oversight **plan** describes a future twin-prompt test. Its assertion that the present experiments establish a chosen internal route inherits the same limitation. The six scoped files contain no twin-route evidence. The earlier capital result also lacks a displayed second table to which this row rule could be applied literally. [README claims and plans](../jlens-inkling/README.md) · [Summary item 2](../jlens-inkling/SUMMARY.md)

## 8. Concrete corrections

1. Change the sibling shuffled denominator from **105 to 106**. Report the distinct-sixth and all-pair scores side by side. Relabel “moves” as **73/96 and 65/96** for any pass-2 change, or retain 71/61 and name the specific outcome.
2. Describe the held-out test as **new table mappings with a previously screened wording**. Report the clean and distinctness filters, qualifying populations, repeated donors and group count with the hit rates. Preserve clean top tokens and token IDs for all wordings in future outputs.
3. Give tied recipient/donor matches their own category or report nonexclusive word hits. Keep the control fruit visible. Check every expected token for eligibility and assert equal recipient/donor masks at both ends before applying future partial transplants.
4. Replace the pointer-storage, first-table irrelevance and Q/K-exclusion claims with the measured standalone and paired effects. Use “none of the tested masks” for the component result. Describe this review as an independent recount with CPU validation of the indexing.
5. To distinguish explanations, cross a fixed donor with multiple independent recipient layouts and fruit assignments. Vary token distance while holding row index fixed, and vary list length or alignment so front and end indexing disagree. Give the same donor answer different donor addresses. Predeclare conflicting token predictions and evaluate on entire held-out table groups. These tests would separate hypotheses that more pairs from the current generator leave equivalent.
