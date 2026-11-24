# Which explanations survive a correction?

A controlled study of ambiguity, memory, and correction in small neural learners. November 24, 2026.

A sorter sees five objects go to five bins. Each color always occurs with one shape, so two rules explain everything it has seen. We remove the examples and save the sorter's memory. In one possible future, a crossed example reveals that color matters. In the other, the same object receives a different label and reveals that shape matters. We ask both versions to sort the same new object.

We can account for every bit available to the learner when the correction arrives. This lets us ask two different questions about the less frequently rewarded explanation. Can the trained model act on it? Could *any* reader of the same memory still recover it?

**Main result.** In a learner with one 20-bit memory, balanced training produced 96.5% success on both possible futures. Giving color-based corrections seven times as often dropped joint success to 59.8%. Reversing the bias dropped it to 51.6%. We reproduced the direction with a second memory architecture, traced several failures to identical bit patterns, and showed that an exhaustive decoder can also make an untrained representation appear highly informative. Each finding below specifies its decoder and evaluation distribution.

## What training pressure did to the stored evidence

First we trained a learner that writes one binary word per bin. It can learn how to encode each color–shape pairing. During training, either explanation is correct half the time, or one is correct seven times as often. At evaluation both explanations get equal weight. A future changes the feature that determines the answer. The episode's two underlying feature-to-bin mappings stay fixed.

With four bits per bin, favoring one explanation reduced the recoverability of the other. To measure this exactly, freeze the encoder and enumerate all 14,400 possible worlds. Group the worlds by their identical five-word memories. For a queried feature value, the best guess in each group is the most frequent correct bin. The resulting percentage is the exact ceiling for a uniform single-feature query under this world prior. We measure it *before the correction*, so differences here are attributable to what was stored.

| Training frequency (color : shape) | Recoverable color mapping | Recoverable shape mapping | Model correct after color clue | Model correct after shape clue |
|---|---:|---:|---:|---:|
| 1 : 7 | 80.0% | 100.0% | 67.4% | 99.3% |
| 1 : 1 | 94.3% | 93.1% | 78.4% | 81.6% |
| 7 : 1 | 99.3% | 84.4% | 98.6% | 70.6% |

Means over fresh training seeds 44, 55, and 66. The first two columns use the whole uniform world prior. The last two use 1,024 held-out episodes with clue and query involving four distinct bins. These are different tasks, so subtracting across columns does not decompose errors. At six bits per bin, all nine confirmation fits recovered both maps and answered both futures correctly. In the original cohort, four-bit color-heavy training reduced exact shape recoverability from 97.0% to 86.7%, with losses of 6, 10, and 15 points across the three matched seeds. [Original results](results/audit.json) · [Fresh seeds and reversed training odds](results/confirmation/audit.json)

**A case where the answer was irretrievably lost.** In two held-out worlds, the seed-44 learner stored the same five words, numerically (2, 5, 4, 14, 9). Both futures presented the same correction: color 3 paired with shape 2 goes to bin 1. Both then asked for color 4 paired with shape 3. Shape was the true rule in both. In one world the required answer was bin 4 and in the other it was bin 2. The worlds differed in how shapes 3 and 4 had been assigned. Even telling the model which feature to use could not recover that distinction from these bits. [Worlds 8770 and 8768 in the saved audit](results/confirmation/audit.json)

**A case where more could be read from the same memory.** On those 1,024 confirmation episodes, the four-bit color-heavy models averaged 70.6% on the shape future. Giving the trained reader the correct selector raised that to 71.1%. A decoder using the known world distribution, the stored bits and the observed correction reached 95.8% on the same queries. Telling the original model which feature matters recovers little here. Its method of reading that feature from its memory accounts for much of the gap. This reference decoder knows the generator, the encoding function, and the sampling constraints. It never receives the original prefix or the identity of the test world.

**Matched present-day performance.** We also required the balanced, color-heavy, and shape-heavy learners to give the correct bin for *every* original object before the clue. Across the three confirmation seeds, 762, 686, and 677 of 1,024 episodes qualified. On those same episodes, color-heavy learners answered every color future correctly and only 66.5–73.6% of shape futures correctly. Preserving current task accuracy can therefore mask different abilities to revise an answer. This post hoc filter matches only the five top answers. [Matched episodes and scores](results/confirmation/matched-current-behavior.json)

**Intervening on storage.** In 198 held-out prefix/donor pairs, swapping two shapes preserved the color solution and changed the required shape answer. We replaced only the two affected binary words with their donor values while holding the clue and query fixed. Across color-heavy fits, 8.6% of these interventions left the bits identical. In the shape-heavy fits none were invisible. Requiring the original shape prediction to be correct *and* the patched one to follow the donor gave 54.9% of all trials for color-heavy and 99.2% for shape-heavy learners. Color predictions also changed, on 1.2% and 19.9% of the respective trials. The latter side effect limits claims of selective editing. In 185 control pairs that preserved both correct answers, the color-heavy model still changed 20.9% of shape predictions. [Intervention outcomes and control](results/confirmation/intervention.json)

## Strong baselines and a failed prediction

Five paired records always solve the task if retained. Two permutation ranks cost only 14 bits and solve both futures perfectly. In the 20-bit budget, a FIFO buffer fits two uncompressed triples of nine bits, and in 30 bits it fits three. In the confirmation cohort those buffers got both futures correct on 16.2% and 60.4% of trials, respectively, even with Bayesian decoding of the retained evidence. The small neural model outperforms some simple storage choices. This experiment gives no efficiency advantage over an explicit compressed learner. Our local shared encoder has its own constraints: a single bin can have 25 different color–shape pairs, so five bits per bin already suffice for a lossless local code by construction.

We then removed the requirement that the query involve bins absent from the clue. On fresh held-out samples, for a shape-rare model, changing its readout to enforce that each bin has exactly one shape took rare-future accuracy from 74.9% to 78.8%. In the color-rare direction the gain was 65.9% to 70.7%. **Both missed the specified five-point gain.** The full prior-aware decoder on these independently drawn queries reached 91.6% and 88.0%, respectively, without using the disjoint-query constraint. The cheaper readout enumerates 120 permutations per axis. No inference-speed advantage was measured. [Every paired prediction](results/decoder-control.json)

## A second architecture and an untrained control

A global encoder adds a learned vector for each arriving triple, then writes a single binary state. It has no dedicated destination slots. We retained the same two candidate features and selector. This architecture and the first use different training budgets, so the comparison tests the direction of the effect. It cannot rank the two architectures by training efficiency.

| 20-bit global memory, color : shape | Model after color clue | Model after shape clue | Both futures correct | Recoverable color mapping | Recoverable shape mapping |
|---|---:|---:|---:|---:|---:|
| 1 : 7 | 51.6% | 99.9% | 51.6% | 73.0% | 100.0% |
| 1 : 1 | 98.0% | 98.5% | 96.5% | 99.8% | 100.0% |
| 7 : 1 | 99.9% | 59.9% | 59.8% | 100.0% | 82.3% |

Means over training seeds 77, 88 and 99. Each received 6,000 updates. The 30-bit positive control passed for all three seeds, averaging 99.3% joint success under balanced training. Biased training at 30 bits still cost about eight points of joint success, even though exact single-feature recovery exceeded 99% in both directions. In contrast, the dedicated-slot 30-bit confirmation cohort performed perfectly. Memory size alone does not explain whether an alternative stays usable. [Global models and controls](results/global/)

The **untrained** 30-bit global encoders allowed 99.9% exact single-feature recovery in both directions. A regularized linear reader given 2,048 fully labeled training worlds scored just 48.2% on color and 46.5% on shape, evaluated on every held-out world. An encoding can act as a fingerprint for a finite world. An observer with the complete generator and encoder can look it up. That measurement is an upper bound on what remains physically distinguishable. Evidence for useful learning comes from behavior, constrained readers and tests on independent worlds.

The equally supervised independent linear reader also corroborates the training effect. At 20 bits the balanced global model allowed 90.5% color and 92.5% shape accuracy. Those numbers fell to 53.4% for rare color and 61.1% for rare shape. Both still exceeded the corresponding untrained reader (43.0% color and 41.5% shape). Results of the same readout for the dedicated-slot encodings are included in the global controls directory.

## Can an overseer see the problem before the correction?

For each frozen memory, we measured its failure rate over every one of the 120 possible rare-future clue/query combinations. A monitor had to forecast that rate *before the actual correction*. We calibrated each monitor on 2,048 training worlds and evaluated it on all 2,880 held-out worlds for each of the six biased four-bit confirmation models. The external observer knows the candidate rules, original demonstrations and the task generator.

| Evidence available before the actual correction | Mean absolute error in predicted failure rate |
|---|---:|
| Average training-world risk | 15.12 percentage points |
| The five demonstrations, including their attribute pairings | 8.32 points |
| Demonstrations and five current-answer confidence distributions summarized by correctness, maximum and correct-bin probability | 7.06 points |
| Also rehearse one hypothetical correction and query on a copy | 6.53 points |
| Instead run ten diagnostic queries over two corrected copies, covering every feature value | 0.68 points |
| Read the five stored associations through the rare-feature head, along with current behavior | 0.42 points |

Internal access beat the original one-rehearsal comparison by more than the predicted two points. A stronger black-box check, designed after that result, closed nearly all the gap. Its ten queries test all five possible feature values twice. This is feasible because we know the task family and can fork the same model state. Exhaustively rehearsing all 120 would give the exact target by definition. The result shows how to forecast vulnerability in this particular closed world. It leaves the relative cost and coverage of these monitors open for larger tasks. [Per-world forecasts and risk counts](results/confirmation/monitor.json)

## Relation to earlier work

Underspecified demonstrations have been used to measure which feature a model favors. Human working-memory experiments have cued a feature after the original stimulus disappears, and neural models have studied how such cues change access to stored features. Distinguishing information that exists from information a constrained observer can use is an established principle. Our contribution is a small, reproducible correction task where one can exhibit irrecoverable losses as pairs of indistinguishable worlds, test competing readers on the same saved state and reverse the training preference to check which feature suffers. We do not establish field-wide priority. [Feature bias in in-context learning](https://aclanthology.org/2023.acl-long.632/) · [Retrospective feature cues](https://pmc.ncbi.nlm.nih.gov/articles/PMC5725244/) · [Neural model of retrospective attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC5788052/) · [Usable information](https://arxiv.org/abs/2002.10689)

## Reproduction and scope

The plans, predictions and failure criteria were written in the working tree before the corresponding runs. The first fit, subsequent controls and their timing are labeled in the protocol. These plans have local provenance only. All runs used local CPU. All 48 fits were repeated from initialization. The 30 destination-slot fits reproduced their saved predictions, selectors and codebooks. The 18 shared-memory fits reproduced every saved endpoint prediction and diagnostic score. The standard-library audit independently recomputes the uniform-query ceilings and checks every trained slot model's endpoint counts. We independently replayed the slot readout with one canonical numerical output per distinct binary word. Small floating-point kernel differences between nominally identical words changed zero archived model decisions on the original or fresh-query cohorts. [Protocol](README.md) · [Original episodes and references](results/episodes.json) · [Confirmation episodes and references](results/confirmation/episodes.json)

Run in this order from this directory, with Python, PyTorch and NumPy. The recorded environment was Python 3.10.10, PyTorch 2.1.0.dev20230517 and NumPy 1.26.4. No network or GPU is needed for these commands once dependencies are present. A stable-release check on Python 3.12 and PyTorch 2.5.1 repeated the three seed-44 four-bit conditions. Training produced different codebooks but the same directional result: exact rare-feature recovery was 75.8% for rare color and 84.0% for rare shape, against 97.0% and 95.0% respectively in the balanced fit. [Runtime check](results/confirmation/runtime-check.json)

```sh
python3 study.py
python3 study.py --confirm
python3 audit.py
python3 decode.py
python3 global_memory.py
python3 intervene.py
python3 monitor.py
```

The two feature types, answer space and allowed selectors are supplied to the model. The split holds out complete pairs of mappings within a known finite family. Episodes can repeat and paired queries reuse one prefix, so individual predictions are not independent replicates. Biased training also supplies fewer examples of the rare endpoint. These measurements describe the resulting encoder at a fixed training budget, and cannot distinguish a long-run optimum from undertraining. The ceilings are exact under the uniform whole-family prior, while measured behavioral accuracies are sample estimates and training-seed averages. This is evidence about correction in small controlled learners. It leaves open how these mechanisms behave in natural language, with noisy corrections, or under open-ended rule discovery.
