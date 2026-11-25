# Can disagreement beat a clock on bursty hidden-source changes?

## Protocol recorded before the first score

Each of 96 independent worlds contains eight calibration events and 1,200 scored events. Use the existing hidden-source generator's rates. Twenty of 32 stable relay addresses read a single 62%-reliable ancestor, then independently flip their reading 4% of the time. Each remaining relay has its own 62%-reliable reading of the balanced binary target. The hidden set of 20 is freshly sampled at each change; the target and readings are fresh at every event. Quiet intervals have discrete lengths uniform from 100 through 500 events. After each quiet interval there are three changes separated by independently sampled 4–16 event gaps. Resume the quiet clock after the third change. Each world gets an independent schedule. This is a clustered renewal model with bounded quiet intervals. First score seed 764802, then confirm independently on seed 764803 without retuning.

Repeat the existing disagreement gate exactly (radius one, group degree at least eight, agreement threshold .75, two flags out of four, six-event refractory interval). All policies receive exactly the same 32 bits per scored event. Primary endpoint is prediction of the hidden binary state with zero external checks. Additional readouts use the earlier frozen purchase threshold of two absolute votes and the same independent 80%-reliable answer when purchased. Replay the gate's exact check receipts to every reader as an evidence-matched control. Score the first eight events after an actual change, other events, and events where the shared ancestor is wrong. Report logical cost (including setup), CPU energy, wall time and refresh counts separately. Each map rebuild computes 32 by 32 Hamming distances. A never-refreshed map still costs one initialization. Naive majority spends no setup.

Compete against majority, a frozen map and fixed refresh periods 1–16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512. Clocks can batch known contiguous intervals and skip historical state updates until a rebuild. Compare to the upper accuracy envelope of the tested clocks and all ex-ante randomized mixtures of two clocks, at the gate's measured CPU energy per event. Repeat at matched expected map rebuild count to separate efficient scheduling from the resulting implementation cost. Clock mixtures can also use less than the budget. Select the best affordable mixture again within each paired world bootstrap replicate; measured policy energies are held fixed. Repeat energy measurements in randomized order on a common fixed batch. Per-process CPU energy omits sensors, wires and host overhead. A positive result requires over 0.5 percentage points of accuracy gain on the energy envelope with a paired 95% lower bound above zero on both seeds. A stronger logical result requires the same against the rebuild envelope. Run a matched control on the original 64-event fixture to check that the receiver reproduces its predecessor.

## Timing sensitivity registered after opening both accuracy panels

The primary timings allow a clock to batch multiple scheduled windows from a preloaded trace. An application that needs an answer before the next arrival gives up that optimization. Repeat CPU metering with every clock processing one arrival at a time, vectorized only over the same simultaneous worlds as the gate. Include all the same clocks and same-world bootstrap. Check margins exactly match the primary clock implementation on a small prefix at all periods. The gate's logical outputs and the accuracy panels remain fixed. The gap might shrink or reverse under the stricter latency constraint. Treat this as an implementation sensitivity rather than a fresh accuracy replication.

## Results

Each seed contributed 115,200 scored decisions. The first panel contained 975 group changes and 7,325 decisions in the first eight events after a change; the independent confirmation contained 967 changes and 7,249 such decisions. Fits and thresholds remained frozen across panels.

| Gate minus best affordable mixture of fixed clocks, accuracy in percentage points | Initial panel | Independent confirmation |
|---|---:|---:|
| At equal expected map rebuilds | +0.85 [0.61, 1.03] | +0.77 [0.53, 0.98] |
| At equal process CPU energy, clocks may batch scheduled arrivals | −0.36 [−0.58, −0.20] | −0.40 [−0.64, −0.21] |

The confirmation gate rebuilt the map 25.46 times per 1,200 decisions including calibration. A mixture of 32- and 48-event clocks reached 76.35% at that budget. The gate reached 77.12%. At the gate's 2.39 µJ budget the strongest clock mixture reached 77.52%. A plain eight-event clock already reached 77.33% at 1.69 µJ. The logical prediction passed on both panels and the energy prediction failed on both. The intervals resample entire worlds and reselect the best affordable mixture in each bootstrap draw. CPU energy is conditional on the reported median measurements. Each randomized mixture picks one clock before a world starts, with mixing probability independent of its observations.

### Confirmation at zero external checks

| Policy | Accuracy | First eight after a shift | Rest of stream | Shared ancestor wrong | Map rebuilds per world | CPU µJ/decision | Wall µs/decision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 62.03% | 61.66% | 62.06% | 0.00% | 0 | 0.06 | 0.013 |
| Freeze after calibration | 66.65% | 62.48% | 66.93% | 16.73% | 1 | 0.26 | 0.09 |
| Refresh every 32nd event | 76.67% | 62.63% | 77.61% | 54.03% | 38 | 0.60 | 0.18 |
| Disagreement gate | 77.12% | 63.95% | 78.01% | 53.36% | 25.46 | 2.39 | 0.66 |
| Refresh every eighth event | 77.33% | 63.93% | 78.23% | 56.60% | 150 | 1.69 | 0.42 |
| Refresh every event | 77.94% | 68.29% | 78.59% | 58.50% | 1,200 | 11.66 | 2.93 |

Each zero-check dependence-aware reader receives 32.213 relay bits per scored event including calibration; majority needs 32. The same counts apply to relay visits. The gate saves internal map recomputation but still inspects each arrival for disagreement. All methods receive current readings from every relay. A wrong hidden ancestor controls most of the relays by construction, making majority especially weak on that conditional slice.

With optional verification on the confirmation panel, the gate reached 83.84% with 0.344 checks per event. The eight-event clock reached 84.03% using 0.358 checks. On the exact checks bought by the gate, it reached 83.29%. Check allocation and interpretation of the same receipts answer different questions. These check-enabled figures are secondary; primary energy measurements cover inference with checks disabled.

Run `python3 dynamic-learning/circuit-communication/bursty/study.py` from the repository root with NumPy 2. [Initial policy, per-world and timing data](panel-764802.json) · [Confirmation](panel-764803.json)

### Arrival-by-arrival energy sensitivity

Requiring an answer on every arrival preserved the outcome. At observed gate energy, accuracy trailed the best clock mixture by 0.33 points [0.15, 0.54] initially and 0.37 points [0.19, 0.62] on confirmation. Confirmation gate and eight-event clock consumed 2.58 and 1.92 µJ per decision respectively. The strong clock result survives removal of cross-time batching. Both implementations still batch 48 independent worlds on one laptop. Real network power, sequential latency for one decision and clock periods outside the tested grid remain unmeasured. Run `python3 dynamic-learning/circuit-communication/bursty/arrival.py` after the primary experiment. [Raw arrival-by-arrival timing and envelope differences](arrival-results.json)
