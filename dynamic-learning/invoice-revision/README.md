# Which invoice should a learner check before changing a rule?

Phase 1 preregistration draft, 24 November 2026. The four-world decision tree was enumerated while designing the task. Its ceilings below are mathematical properties of the design. Learner predictions are prospective. Freeze this version in Git before fitting a learner or inspecting its test results, and record any later departures alongside it.

## Question

After a report of an incorrect import, can a learner choose the evidence needed to identify the scope of a correction, preserve that evidence through a restart, and produce correct dates on both new and previously booked invoices? Separate the benefit of selecting evidence from the benefit of the method used to store and apply it.

## Four indistinguishable starts

Create one ledger with an importer, three suppliers using it, and a second importer with different suppliers. All raw dates admit two valid interpretations (for instance, 04/05 versus 05/04), and the two interpretations give different dates. Randomly assign which convention the installed parser uses. Each invoice includes its own payment term in days. Compute due dates with standard calendar arithmetic. Give every learner the same read-only raw records, old booked values, and complaint that one named invoice's **date** was reversed.

Fork this exact starting state into four equally frequent worlds. The correct interpretation was already in force at the time of booking. The report uncovers a previous error. This temporal rule is what makes correction of a historical booking appropriate. In the one-off world the reversed interpretation applies only to the named invoice. Other invoices with the same raw date retain their own interpretation.

Each probe compares one booking with an independent, always-correct reference and returns a single bit. One means the original date should be reversed. No world label, free-text explanation or other record comes back.

| Hidden world | Verify reported invoice | Check a different invoice, same supplier | Check a different supplier, same importer |
|---|---:|---:|---:|
| Reporter mistaken | 0 | 0 | 0 |
| One invoice entered under the other convention | 1 | 0 | 0 |
| Supplier uses the other convention | 1 | 1 | 0 |
| Importer uses the other convention | 1 | 1 | 1 |

The best fixed pair distinguishes three worlds out of four on average under the uniform prior. With two adaptive probes, check the same supplier first. If that booking is correct, verify the reported invoice. If it is wrong, check the other supplier. The outcome identifies all four worlds. Exact best fixed scope accuracy at budgets zero, one and two is 25%, 50% and 75%. Adaptive two-probe accuracy can reach 100%. This small task tests whether a learner can use branching and preserve its result. Success alone would establish little about discovering experiments in richer environments.

## Comparisons

**Evidence selection.** At budgets zero, one and two, evaluate an observation-blind fixed schedule selected on development data, uniform random probes without replacement, a learned selector, and the hand-coded adaptive tree. The learned selector sees the current ledger, complaint and any probe receipts. It never sees the hidden cause. Randomize identifiers, row order, date convention and probe menu order independently of cause. Require a distribution over the four scopes after each receipt and score its probability on the true cause by the Brier score (sum of squared errors). Exact scope accuracy under the uniform prior is the primary selection endpoint. Balanced worlds with equally likely causes must remain equally likely before informative evidence arrives.

**Storage and application.** Feed every writer the exact same ordered probe identities and receipts from each selector, on the same fork. Start with an explicit store of observations and a rule interpreter, a dense writable adapter, and a sparse writable table. Use an exhaustive decoder over the four stated hypotheses as the ceiling. Give the ordinary methods the same task description, training examples and access to read-only records. Keep any cause labels used in training identical across learned arms. Freeze writers and selectors before opening test worlds. On restart discard the complaint transcript, probe receipts, optimizer state and working cache. Only declared persistent state and the original read-only ledger are available. Count stored evidence, keys, indexes and invalidations against the writable-state allowance.

**Budget.** The first implementation is a local CPU feasibility test. Before fitting, freeze the state serialization, learned model sizes, selector inputs, update counts and development tuning grid in this protocol. Use a common maximum of 256 writable bytes per incident in the primary comparison, including addressing metadata. Measure fixed model size, update/query work and record reads separately. Keep the bit-packed explicit store at its actual size. Compare deployable cost as well as accuracy. A method requiring extra per-incident state must report it. The hand-coded oracle is an upper bound and its perfect score is expected.

## Score new work and old bookings separately

After restart, ask for both the interpreted invoice date and payment due date. Balance query groups so a learner cannot win by always preserving the common case.

| Query group | What it tests |
|---|---|
| A new invoice from the reported supplier, another supplier on the importer, and a supplier on the other importer | Prospective application, propagation to a derived date, and preservation of unrelated rules. |
| The reported booking plus three old, unprobed bookings in those same supplier/importer groups | Whether the learner can correct previously materialized values and preserve the records that were already right. The reported booking is scored separately because it can have appeared in a probe. |

Give all methods the original raw inputs and cached values under the same interface. Allow recomputation on read. Log whether the answer used the old value, recalculated from source, or used a stored correction. Also score what the materialized ledger would return **without** invoking the learner, applying only its explicitly issued record updates or invalidations. Report invalidated-but-unmaterialized rows separately from corrected rows. Correct answers after recomputation establish recoverability. A repaired ledger requires the stored dates themselves to be correct. Report reads, recomputations and writes so eager repair and lazy recomputation can be compared fairly.

Primary writer endpoint is the share of complete four-world families for which every required date and due date is correct after restart under identical oracle-selected evidence. Also report each world's success, prospective and historical accuracy, affected rows missed, unaffected rows changed, and exact raw-ledger repairs. With fixed-schedule receipts some worlds are indistinguishable. Report performance against the corresponding evidence ceiling.

## Predictions and decisions

1. A learned selector will exceed the best fixed schedule on test scope accuracy at two probes. Require a paired 95% interval above zero. Report its gap to the known optimal tree and whether it failed on the first decision or on the conditional second decision.
2. With identical decisive receipts, I predict the explicit rule store will solve all four worlds, including old bookings when it can recompute from raw records. Failure triggers an implementation audit before interpreting neural results. A larger prospective-to-historical loss for the sparse writer than the dense writer is a secondary directional prediction. If both can recompute successfully, record the tie and compare its cost.
3. Generate 256 development and 512 test ledgers from disjoint seeds (20261125 and 20261126). Fork each ledger across all four causes and all methods. Use independent generated training ledgers with seeds 4101–4108 for eight paired learned fits. Report per-fit results and paired intervals resampling fits and ledger families, keeping cause forks together. Preserve each receipt, query answer, state size and operation count.
4. Before collecting learner results, check that all four pre-probe inputs are byte-identical within a family, the dates change exactly on their specified scope, all probe patterns match the table, the oracle reaches its ceilings, and query rows other than the reported invoice were never probed. If the explicit store matches neural methods at lower cost, the practical result favors the explicit store. A language-model extension requires a second task family where a fixed three-question script cannot solve evidence selection.

The mistaken-reporter world tests the truth of **one report**. Learning that a source's reliability has changed needs a stream of independently verifiable reports and calibration before and after the change. That is a separate follow-up. The [source-reliability experiment](../plasticity/recognizing-change.md) motivates giving that learner a limited opportunity to verify current evidence, and the [forked-memory study](../forked-memory/NOTE.md) motivates scoring all possible futures of the same starting state.

## Phase 1 freeze before fitting

Use NumPy on one local CPU thread with no pretrained models. This is a structured-input test of learning under a known hypothesis family. All methods get identical equality features (reported invoice, same supplier, same importer, other importer), the installed calendar parser, and access to the original ledger. All other predictions must be learned or derived from receipts. Identifiers are unsigned 16-bit values. Use six booked records (including the three probe candidates) and three fresh ones. World membership is defined by identifier equality. Random probe order is shared across forks.

Each training seed generates 256 ledgers and all four forks, with an independently shuffled probe schedule per ledger. Use the first two observations for the training prefixes of lengths zero, one and two. The selector estimates the probability of each binary probe outcome within each labeled training world with add-one smoothing, uses a uniform prior, and chooses the remaining probe with maximum expected reduction in posterior entropy. Scope estimates use this same learned likelihood for the learned selector and the stated exact likelihood for the other selectors. Resolve equal scope probabilities in the table's order and probe ties by displayed menu order. Choose the fixed order maximizing development accuracy summed across budgets, breaking ties by summed lower Brier score, then by canonical order. Random selection uses a uniformly shuffled displayed menu shared across fits. No selector or writer gets a hidden cause at test time.

The two learned writers add one three-coordinate float32 update per observed (probe type, bit) from a 6-by-3 learned table. The sparse writer can update only the coordinate addressed by that probe type. Each writer learns a 4-by-3 linear readout and four biases for the four query relations, trained on mean binary cross-entropy predicting reversal for each relation with equal weight. Initialize weights at normal standard deviation 0.1, biases zero. Use full-batch Adam, 400 steps, rates 0.01 and 0.03. Choose on development by lowest loss over all four worlds and all random-order prefixes, separately by seed and writer. Prediction threshold is 0.5, with ties keeping the installed parser. This intentionally tests sparse writes with a compositional reader that can consult every slot. Both writers train on the same ordered receipts and target behaviors. Store the reported ID (2 bytes) plus three float32 values for dense, or one byte of occupied-slot bits and one float32 per occupied slot for sparse. The explicit writer uses the reported ID and one byte encoding three ternary receipts. After restart it uses the first most probable scope consistent with these observations. The oracle enumerates consistent worlds separately.

Cross every writer with lazy and eager ledger maintenance. Lazy emits no record updates. Eager scans all six booked rows and writes both corrected dates for every predicted reversal. Charge 10 bytes per changed row (ID and two 32-bit ordinals), in addition to writer state. Neither policy issues invalidations. Record both the six-row direct ledger and seven query answers. Record reads by logical row access, writes by changed row, calendar recomputations by parsed row, and neural arithmetic by update entries touched and readout multiply-adds. The packed archive may share identical policy transcripts, state payloads, outputs and operation counts across trials. Release the decoder and row-generation data alongside it.

Use 2,000 paired bootstrap draws with seed 7001, resampling eight fits and 512 ledger families independently with replacement. The learned-vs-fixed selector accuracy difference and the sparse-minus-dense prospective-to-historical loss (unprobed old rows versus matched new relations) are the interval contrasts. Report the reported row separately. Equal predictions on matched old/new features imply zero prospective-to-historical loss on reads for any deterministic relation-only reader; this structural constraint narrows the secondary prediction before fitting. The direct-ledger result can differ by maintenance policy. Stop each CPU invocation after 10 minutes. The separate source-reliability stream and language-model extension are outside Phase 1.

## Phase 1 results

Protocol freeze: `0722f48`. All eight CPU fits completed across 512 held-out ledgers, each forked into the four causes. With two probes, every fitted selector checked another invoice from the same supplier first and chose its second probe correctly on both branches. It estimated the reliability of the three probe types from labeled training worlds. The entropy criterion and relation features were provided. The selector learned this small decision tree with those advantages.

| Scope identification | No probe | One probe | Two probes | Two-probe Brier score (lower is better) |
|---|---:|---:|---:|---:|
| Best development-fixed order | 25% | 50% | 75% | 0.2500 |
| Uniform random order | 25% | 50% | 75% | 0.2500 |
| Learned likelihood, adaptive choice | 25% | 50% | 100% | 0.00023 |
| Hand-coded adaptive tree | 25% | 50% | 100% | 0 |

The learned-minus-fixed difference was +25 percentage points in every fit and four-world family. Its paired bootstrap interval is [+25, +25]. This is a consequence of the four deterministic patterns and the recovered tree. The 512 layouts exercise ID remapping, menu order and calendars while supplying the same causal structure.

| Writer after the same two decisive receipts | Families with all answers right in all four worlds | Persistent bytes with lazy reads | Mean bytes including eager record writes |
|---|---:|---:|---:|
| Packed receipts and explicit rule interpreter | 512/512 in each fit | 3 | 25.5 |
| Dense adapter | 512/512 in each fit | 14 | 36.5 |
| Sparse table | 512/512 in each fit | 11 | 33.5 |

All three writers also reached the 75% per-world ceiling under either nonadaptive two-probe schedule. Each achieved zero errors on new rows, matched unprobed historical rows, and the named booking under oracle evidence. The secondary forecast that sparse writes would lose more on historical reads failed. All readers were supplied the same relation features and could reparse dates. Matched new and old queries therefore had identical reversal decisions by construction. A broader sparse-versus-dense test needs a task where learning how records relate is part of the problem.

On the *direct* six-row ledger, lazy maintenance left every row correct in only the false-alarm world (25% of worlds). Eager maintenance made the ledger fully correct in 100%, scanning six rows and writing a mean 2.25 corrected records per world (22.5 bytes of delta). No policy issued invalidations. This difference follows from the maintenance policy and was identical across writers. The smaller explicit store wins on incident storage in this family. Wall-plug energy was unmeasured.

The dense writer uses 34 fixed float32 parameters (136 bytes). The sparse writer has 22 active fixed parameters (88 packed bytes), represented by the same 136-byte dense arrays with masked zeros in this NumPy implementation. The explicit writer uses a fixed hand-coded hypothesis interpreter. The learned selector uses twelve fixed float64 probe probabilities (96 bytes). These fixed sizes exclude executable code, interpreter overhead, and the common ledger. With two probes the neural writers touch six dense or two sparse update entries, then use twelve readout multiply-adds to decode all four relations. The explicit reader decodes at most three receipt digits against each of four possible scopes. Operation counts describe the logical policies, rather than CPU instructions in the vectorized simulation.

### Reproduction and trial lookup

Run from this directory with Python and NumPy. A second fit from initialization reproduced every array in the saved archive exactly. The independent audit checked all held-out calendar alternatives, all 16,384 fit/policy/family decisions across four forks, and 1,920 replayed combinations of serialized state, world, and maintenance policy. No paid compute was used.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 study.py
python3 analyse.py
python3 audit.py
python3 analyse.py --trial 0 2 2 0 2 0 1
```

The last command decodes the first fit and ledger, learned selector, two probes, supplier change, explicit writer, eager maintenance. Arguments are zero-based in the order fit, selector (fixed/random/learned/oracle), budget, ledger, world (table order), writer (explicit/dense/sparse), eager flag. It prints the invoice ID and outcome for every receipt, the scope distribution, the restored bytes, all seven date and due-date pairs, their sources, the six-row ledger and operation counts.

[Trial archive](results/trials.npz) · [All policy and writer scores, intervals, operation tables and development choices](results/summary.json) · [Training and selection](study.py) · [Ledger and state serialization](ledger.py) · [Scorer and decoder](analyse.py) · [Independent audit](audit.py)
