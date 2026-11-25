# Dynamic learning and model architectures

How can a model learn something useful from a few experiences, apply it to a new case, and revise it when the world changes? Progress here could mean needing fewer examples to learn a rule, retaining it across interruptions, or spending extra computation only when it helps. Each is measurable on a small model. We also want to understand what changed inside the model when it learned.

The [research assessment](where-to-push.md) proposes a test of how weight updates over a model's lifetime change its ability to learn the next task and its susceptibility to misleading feedback. It also evaluates three related publication bets.

The [plasticity experiment](plasticity/README.md) ran that test in 16 small-model lifetimes. Practice changed where later feedback propagated. The learned update rule missed its capability target and learned more slowly when previously irrelevant features became useful.

The [string-learning study](string-learning/README.md) changes the transformer's internal representation. Identical practice examples in different orders changed few-shot accuracy by up to 10.7 points. Saving the earlier projection recovered much of the loss on the first family. A joint capability and feedback confirmation narrowly missed its retention criterion.

Without a supplied practice boundary, [online selection of past weights](string-learning/unannounced.md) passed the joint comparison on the strings, and [offline consolidation](string-learning/consolidation.md) reproduced most of the two-projection teacher's performance in one projection. On [human-written CLINC150 requests](intent-learning/README.md) the timing benefit was small and the original pretrained representation was stronger.

In the [changing queue policy study](intent-learning/policy-change.md), practice taught a learner which other weights to update after a correction. Choosing a request to verify helped when the verifier was accurate. A 10% verifier error rate exposed a measurable trade-off between repairs and harmful flips. A [MovieLens transfer test](preference-learning/README.md) used real preferences and found no accuracy advantage over a genre-based updater, even when both started from identical personalized weights.

The [attention weight-update study](attention-updates/README.md) traces how the same correction propagates differently after different practice histories, measures the cost when the old relationship breaks, and tests the direction of that propagation on real Jester ratings.

A [conditional weight-write experiment](attention-updates/conditional-writes.md) asks whether the direction of the next edit can depend on earlier corrections. A [historical roll-call study](rollcall-learning/README.md) tests recent weight learning and the revision of a disputed report on naturally occurring group decisions.

A [matched-practice test](rollcall-learning/active-verification.md) measured what training on targeted checks adds. A [longitudinal test](rollcall-learning/persistence.md) found that one independent check improved confidence estimates while every learner still predicted the old habit.

The [handwriting switch experiment](handwriting-switch/README.md) transferred a learned willingness to change tasks by swapping four gate coefficients. A follow-up on different writers tested whether choosing two independent checks let a provisional weight rewrite improve accuracy while keeping false changes rare.

On [written requests](intent-learning/provisional.md), retrieving a similar example matched the benefit of choosing one with the provisional classifier. The [pretrained attention experiment](intent-learning/value-history.md) changes how ordinary gradient steps propagate inside MiniLM while preserving its starting predictions.

The [recurrence, replay and plasticity reading note](recurrence-replay-plasticity.md) connects looped transformers to human experiments on feedback and memory, and proposes a test of selective weight updates.

The [forked-memory study](forked-memory/NOTE.md) tests this question in two learned memories. It includes exact losses, an intervention, a matched-current-performance check, and a random-encoder control.

The [debugging handoff study](debugging-compaction/NOTE.md) asks whether a coding agent can revise a mistaken diagnosis after it compresses its experience. It measures recognition of the evidence and tests the resulting patch separately.

## Starting point — learning during sleep

In *Sample-Efficient In-Context Adaptation via Sleep* (Sangyun Lee and colleagues, November 24, 2026), a transformer periodically revisits a window of past tokens. Extra forward passes refine the keys and values that later tokens can attend to. The model trains with this operation. At inference the updated knowledge lives in the sequence’s cache, so resetting that cache also resets this form of adaptation.

Figure 2 asks a good question about intelligence: how much experience does a model need to make an equally good prediction? On held-out code, the two-pass and four-pass models matched the no-sleep model’s loss at a 16k-token prefix with 2.10× and 2.39× fewer prefix tokens, respectively. The target was the same final 256 tokens in each comparison. The sleep models started worse in the first window and improved as context accumulated. The authors matched training tokens, and longer sleep consumed more training compute. The token savings measure sample efficiency on this evaluation. The compute trade-off is a separate question.

The [weight-space programming review](weight-space-programming.md) follows a second route: a foundation model writes the small model or its update rule directly. It includes a local check of released weights and a proposal for testing whether the resulting model can learn after the writer is gone.

## Three directions worth exploring

| Direction | What changes as experience arrives? | Question for a small experiment |
|---|---|---|
| Sleep and consolidation | A trained model revisits recent evidence and rewrites the cache used for later predictions. An earlier sleep design writes context into the fast weights of a state-space model before clearing its cache. [Earlier sleep paper](https://arxiv.org/abs/2605.26099) | Which experiences repay a second pass? Does revisiting an early rule help the model generalize, or make a later correction harder to learn? |
| Looped transformers | A shared block can run more times before the model answers. Our Inkling work already measures what crosses a pass boundary, and our Ouro work measures what remains visible in text. [Message swaps](../jlens-inkling/NOTE.md) · [Loop legibility](../loop-legibility/README.md) · [Recurrent depth](https://arxiv.org/abs/2502.05171) | With several questions about the same examples, is an extra pass more valuable while reading the examples or while answering each question? Does the gain carry over to the second query? |
| Test-time learning | In a test-time training layer, the sequence-specific state is itself a small model. A self-supervised gradient update changes that state as tokens arrive. [Sun et al.](https://proceedings.mlr.press/v267/sun25h.html) | How many observations does it take to learn a changed rule? When only one of two rules changes, how well does the other survive? |

## First proposed experiment — when is sleep worth spending?

**Original proposal.** Two simplified local pilots and their results are recorded below. Start with a small transformer trained on generated episodes. This gives us control over the evidence, the correct answer, and the timing of every extra pass.

Each episode teaches two machines’ rules through worked examples. For example, one machine rotates a sequence of symbols left, and another reverses it. Later examples either preserve both rules or change one. Query both machines on fresh strings chosen so that the old and new rules predict different answers. Randomize machine names, rule assignments, change position, and symbol strings across episodes. A test query can only be answered by applying a rule to a new input.

Train one weight-shared model to accept an extra pass at different window boundaries by randomizing its placement during training. At evaluation, give each policy the same held-out episodes and the same allowance of one extra pass over an equal-length window. Compare a fixed boundary, a random boundary, and the first window whose example outputs exceed a prediction-loss threshold. Choose the fixed boundary and threshold on development episodes. If the threshold never fires, spend the pass at the last eligible boundary. The trigger sees only examples observed so far. Also score sleep disabled on the same weights to measure the absolute effect of an extra pass.

**Prediction:** after two examples of a changed rule, choosing the window by prediction loss will improve exact-match accuracy on fresh inputs for that rule over the best fixed boundary. Score the unchanged machine separately to catch interference. Include episodes with a surprising but valid example under the old rule to test whether the trigger mistakes difficulty for change.

**Decision:** freeze the generator, policy settings, and scoring before the held-out run. Compare paired accuracy by episode and report a 95% interval, accuracy after one, two, and four new examples, and compute including the trigger. Advance the adaptive policy if its primary gain over fixed placement is positive with an interval excluding zero and its accuracy on the unchanged rule falls by at most two percentage points. If the model cannot apply a stable rule on fresh strings above chance, the task has yet to establish learning and the scheduling comparison is inconclusive.

If timing matters, fork the same example prefix immediately before and after the extra pass. Feed both caches the identical suffix and compare the probability of the old and new answers on several fresh inputs. That intervention asks whether the cache update itself changed which rule the model applies. A benefit on generated rules would justify a second test on unfamiliar code conventions.

### Where autoresearch fits

Karpathy’s autoresearch gives an agent one editable training file, a fixed evaluator, and a short budget per trial. Its default score is validation bits per byte after five minutes of training. A faster implementation can process more tokens and win that comparison. Our question instead fixes the examples the model sees and asks how quickly it learns a changed rule. [Upstream experiment instructions](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md) · [Evaluator](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/prepare.py#L344-L359)

Use the agent loop once the fixed, random, and no-sleep baselines work. Train the sleep-capable model once, freeze it, and let the agent change only the decision of when to spend the extra pass. For example, compare a trigger that reacts to one surprising output with one that waits for two examples supporting a change. Before each trial, record which episode types should benefit and what would disprove the explanation. Give the agent development episodes and errors to inspect. Keep the generator, scoring code, and per-episode pass budget fixed.

Log every candidate and its cost, including failed ideas, within a total search budget. Select on development performance, then freeze one policy and compare it with the fixed schedule on untouched episodes and independently trained checkpoints. Use the changed-rule accuracy and unchanged-rule retention criteria above. This lets us judge both the policy and the explanation that led to it.

### Local feasibility pilot — plan recorded before training

Use two machines choosing the left or right symbol of a fresh pair. Each window contains one worked example per machine. The two symbols always differ, giving chance accuracy of 50%. Balance the changed and stable episodes, vary the change after two to five windows, and query after one, two, and four subsequent windows. A held-out query pair has never appeared in that episode. This small task tests revision within two known rules.

Use a tiny attention model with shared weights for encoding a window and revisiting it. Keep the resulting window representations available for later queries. This is a simplified writable memory. Train for 800 updates on CPU with one uniformly random revisit per episode. Select the best fixed timing (first, last, or a fixed ordinal capped at the final window) and surprise threshold on development episodes at two examples. Surprise is the mean negative log probability of the two observed answers under the preceding context. The first window has no preceding evidence and is ineligible for the trigger. At evaluation, report trigger scoring costs separately. Preserve per-episode outcomes.

Prediction: stable no-sleep accuracy will exceed 60%, and surprise timing will improve two-example changed-rule accuracy over fixed timing. If the stable gate fails, stop. If a held-out comparison has no positive paired lower bound, stop policy search on this pilot. One model seed and a synthetic task make this a feasibility result. Measure on fresh stable episodes how often the trigger fires before fallback.

A symbolic baseline can infer each machine’s current rule from its most recent worked example. It should score 100%. A neural success would establish that this pilot measures learned revision in this architecture; a practical advantage over a simple algorithm would require a richer task.

**Pilot 1 outcome.** On all 1,152 fresh episodes, the no-sleep model and every selected policy answered both queries correctly. The changed-rule comparison at two examples was 192/192 for each policy. Every timing tied on development data. The gain prediction failed and search on this task stops. [Per-episode predictions](results/sleep-timing-pilot.json)

### Conditional rule pilot — plan recorded before the second run

Give each machine two independently sampled rules, one for pairs whose left symbol is 0–3 and one for 4–7. On a change episode flip one entry for one machine. Sampling still gives exactly one demonstration per machine per window. A demonstration can now leave the answer for the query’s category unresolved. Query pairs are drawn independently of which entry changed. Report the primary comparison across all queried categories and separately score queries that exercise the flipped entry. Use a simple reference learner that recalls the latest observed choice for that machine and category, defaulting to left if the category is unseen. Its errors help distinguish ambiguous evidence from a neural model’s errors.

Keep the schedule candidates, update budget, stable gate, and decision rule from pilot 1. Use fresh seeds for training, development, and evaluation. Prediction: the no-sleep model will exceed 60% on stable episodes but fall below 95% on changed-machine queries after two examples. The surprise policy will beat fixed timing on this primary measure. A tied or inconclusive paired comparison ends this pilot.

**Pilot 2 outcome.** The no-sleep model answered 98.1% of stable queries correctly. Development selected the second window for fixed sleep and a loss threshold of 0.25 for adaptive sleep. On the 192 held-out changed episodes with two new examples, accuracy on the affected machine was:

| Policy | Correct |
|---|---:|
| Recall latest observed choice in the query category | 170/192 |
| Sleep disabled | 166/192 |
| Fixed sleep | 166/192 |
| Random sleep | 166/192 |
| Surprise-triggered sleep | 165/192 |

Adaptive and fixed sleep differed on one primary answer, a loss for adaptive sleep. The paired difference was −0.52 percentage points (episode bootstrap 95% interval −1.56 to 0.00). Both preserved the other machine’s answer on 188/192 queries. Of the 88 primary queries that actually exercised the flipped entry, adaptive sleep got 62 correct, fixed sleep 63, and the simple recall rule 68. The stable-task and headroom predictions held. The predicted timing advantage failed. [Per-episode predictions](results/conditional-rule-pilot.json)

The trigger spent its pass before the final window on 398/576 stable episodes. An online implementation of this threshold would require 2.27 additional two-machine prediction calls per episode on average. The evaluator precomputes all pre-sleep scores to compare thresholds. Revisit budgets are equal, total inference costs differ. The original surprising-but-valid control has only been approximated with ordinary stable episodes. This is one training seed per task, with two independent examples encoded together in each window and no cross-window computation until query time.

**Decision.** Stop schedule search on both tasks. The evidence for developing a scheduler would be stronger on a task where an extra pass at a known informative window reliably changes answers for the better. Test that premise on development data before optimizing another trigger.

**Reproduce locally** with PyTorch on CPU from the repository root:

```sh
python3 dynamic-learning/pilot.py
python3 dynamic-learning/pilot.py --conditional
```

## Research direction — what does a model keep so it can change its mind?

A model can answer correctly today while throwing away evidence it will need tomorrow. That gives us an architectural objective beyond immediate accuracy: spend limited memory on information that lets a future correction generalize. The sleep pilots above had little need for this ability. A latest-example lookup already performed well.

**A concrete test.** A sorter sends objects to bins. On early examples, each color always occurs with one shape. Either feature explains the assignments. Draw new arbitrary names and bin mappings for each episode, with at least four bins. Then make the early examples inaccessible. Fork the same learner state into two futures. A single crossed example reveals that color determines the bin in one future and shape determines it in the other. Ask about new color–shape combinations involving bins absent from that clue. Early accuracy and the entire early history are identical across the forks. The single clue identifies which earlier relationships matter, and cannot supply all the bin assignments by itself.

Start with three explicit learners: store one candidate mapping, store both, or retain as many raw examples as the memory budget permits. Count the storage of keys, labels, and selectors. If raw examples already retain everything at the chosen budget, report that and measure the effect of repeated examples and irrelevant attributes. A Bayesian learner tracking the possible rules gives a reference for what the evidence warrants. Once the task distinguishes strategies, compare a learned recurrent memory with a weight-written learner at the same state size, using the same stream and evaluation budget. A sleep pass earns a place only if it helps after charging for the extra work. For the alignment test, fork a third future where a single oddball should have little lasting influence under the stated noise process.

The causal question is whether the learner retained the alternative *before* the correction. Evaluate both possible futures from that saved state. If a probe finds a candidate representation, perturb it and test whether only the corresponding future loses accuracy. Compare its predictions with confidence and with an external monitor that sees the same stream. Replicate on a learned model before claiming the diagnostic generalizes beyond a circuit built for the test.

Latent-task inference in transformers, different responses to oddballs and rule changes, and prediction of future trainability from present state all have prior art. The proposed contribution is to test what bounded memory preserves through consolidation, under paired later corrections, with a causal check on the retained evidence. Novelty remains open. [Transformer task inference](https://arxiv.org/abs/2605.03780) · [Change points and oddballs](https://www.jneurosci.org/content/42/12/2524) · [Forecasting future learning](https://arxiv.org/abs/2610.09621)
