# When does a bounded workspace pay for itself?

Five sequential CPU experiments, 25 November 2026. The best use of the workspace depended on what made the problem hard. With the observations fixed, sharing partial computations extended the depth of programs we could recover. When observations were available for purchase, splitting a contradiction could identify a real change cheaply. On externally specified feedback rules, ordinary linear elimination still solved more cases with less energy than the fitted controller.

## Extra compute on the same evidence

Each episode required inducing a composition of supplied operations, predicting 12 new outputs, revising after one instruction changed, and predicting 12 more. Every method saw the same four demonstrations in each phase. A solve required all 24 outputs correct. At least one revision demonstration contradicted the old program. The following are the deepest random-table programs with at least 90% observed exact solves at each compute allowance, with 24 worlds per depth. A dash means no evaluated depth passed.

| Primitive table reads allowed per phase | Independent sampling | Depth-first search | Bidirectional search |
|---:|---:|---:|---:|
| 128 | 2 | 2 | — |
| 512 | 2 | 4 | 4 |
| 2,048 | 2 | 4 | 8 |
| 8,192 | 4 | 6 | 8 |

At depth six, depth-first search and bidirectional search both solved 24/24 at their first passing allowances. They used 9.07 and 0.58 mJ per solve respectively. At depth eight, bidirectional search solved 24/24 at 2,048 reads per phase, using 2.33 mJ per solve versus 0/24 for depth-first search at that allowance. Even 24/24 has a two-sided exact 95% binomial lower bound around 86%. For affine operations, composing eight steps still yields one affine function. The two-coefficient solver and bidirectional search both achieved 24/24 there, using 0.023 and 2.77 mJ per solve. These are implementation comparisons between known algorithms. The join index stored up to 256 prefixes in addition to inverse tables, recursion and its current program. [Raw cases, counts and energy](results.json)

## Choosing what to check

The next task used up to 64 unknown binary feedback coefficients. A learner predicted eight new starting states after both three and seven applications, then repeated after a rule change. It received one common witness each phase and could buy further one-bit observations. The workspace held finite equations and a candidate rule. A width of 64 permits 2^64 masks. The linear representation and probe interface were supplied. Changing the probe budget here changes **available evidence** as well as compute.

- A fitted check-count policy took 1.24 CPU joules to fit. On the three-tap transfer panels it solved 603 of 768 episode-budget evaluations, compared with 637 for eight fixed checks. Worlds recur at different budgets. The small unbatched timing samples for this iteration are noisy. [Fitted policy and training cost](feedback-policy.json) · [Paired results](feedback-results.json)
- On fresh 64-bit sparse worlds with 72 probes per phase, checking the suspected input directly raised exact solves from 91/96 to 96/96 versus one random check. Five episodes improved, zero regressed. On a constructed five-tap rule that passes this particular test, the same change dropped 86/96 to 75/96. Zero improved and eleven regressed. Four ordinary random checks also attained 96/96 on the sparse panel. [Comparison and energy trials](targeted-results.json)
- Splitting a contradictory observation can isolate one genuinely wrong coefficient even when several are wrong. With 32 probes per phase it raised sparse-world solves from 51/96 to 91/96, with 42 repairs and two regressions relative to four-check verification followed by basis completion. Paid probes per episode fell from 37.9 to 25.8. At 72 probes the four-check method reached 96/96, while splitting stayed at 91/96. On the first external 64-bit panel, unrestricted splitting got 0/6 versus 6/6 for basis completion. [Raw results](correction-results.json) · [Source coefficient sets](polynomials.json) · [pycrc catalogue](https://pycrc.org/models.html)

## Learning when to split or switch

The final recurrent solver could answer a probe from its equation bank, split a contradiction, test a compact rule, or switch to measuring the full basis. Only the switching policy was fitted. It occupied 32 lookup entries keyed by width, budget and the previous mask's density. The logical actions were supplied. Training used 0.85 process CPU joules. A fixed competitor switched after two corrections. A replay decoder received the fitted solver's exact questions and answers for free. Its timing excludes the decision cost and serves only to locate the source of the accuracy difference. [Frozen policy](deduction-policy.json)

| Fresh 64-bit panel | Basis | Always split | Split with deductions | Fixed switch after two | Fitted switch | Replay |
|---|---:|---:|---:|---:|---:|---:|
| Sparse, 32 probes per phase | 21/96 | 93/96 | 93/96 | 69/96 | 93/96 | 93/96 |
| Dense mixture, 72 probes per phase | 96/96 | 56/96 | 86/96 | 95/96 | 95/96 | 95/96 |

On the dense mixture, deductions reduced paid probes per episode from 51.9 to 40.2 while improving accuracy. At matched 95/96 accuracy, fixed switching used 42.3 probes and 0.519 mJ per solve. The fitted switch used 117.1 probes and 0.641 mJ. On sparse panels the fitted solver's hardest width with at least 90% observed solves rose from 8 to 32 to 64 as the probe budget rose from 8 to 16 to 32. Exact basis reached widths 8, 16 and 32. Its fit-free competitor using deductions also reached 8, 32 and 64. [Paired trials, communication counts and timing](deduction-three-dense.json)

We then applied the frozen algorithms to 23 distinct RevEng generator polynomials absent from pycrc. Every ordered transition within a width was tested with a common random permutation of wires, for 136 episodes. Solvers could query the simulated device. They could not see the polynomial identifiers. At 72 probes per phase, ordinary basis measurement solved **136/136 at 0.279 mJ per solve**, compared with **123/136 at 0.379 mJ** for the fitted switch. The equation decoder replayed the fitted observations and also solved 123/136. Widths 8, 16, 32 and 64 contributed 42, 72, 20 and two pairs respectively. These are reused external coefficients in the same simulated linear task, providing no independent replication across different mathematics or physical devices. [External results](deduction-reveng.json) · [Frozen coefficient values and source digest](reveng-polynomials.json) · [RevEng catalogue](https://reveng.sourceforge.io/crc-catalogue/all.htm)

## What to carry forward

Keep the ability to share intermediate work, and keep the exact solver in every comparison. The fitted gate earned no general energy advantage. In the external panel, retaining its questions and exchanging its decoder preserved the whole solve count. The open question is whether a recurrent learner can **discover which representation makes a new family cheap**, given observable behavior. Another tuned threshold on these same feedback registers would provide little evidence about that.

Energy is XNU-attributed process CPU energy for the Python implementations. Reported per-solve costs divide measured panel energy by completed episodes, charging unsuccessful cases. Batching and three measurements reduce counter noise; small differences need independent hardware replication. Device traffic uses an n-bit stimulus and one response bit. The simulated oracle is inside the meter; a physical device and a distributed network would have their own energy costs. The first study bounded logical prefix entries, and later studies bounded state by width and probe allowance. Equal physical workspace bytes, wall-plug power, GPU throughput, a neural core, arbitrary nonlinear rule discovery and cluster scaling remain unmeasured.

The [protocol](README.md) records the predictions before each run. To reproduce locally, run each script separately from this directory with Python 3.10 or newer. Use the checked-in catalogue snapshots. The XNU counter requires an Apple silicon Mac with clang. Other platforms produce timing and logical counts.

```sh
python3 study.py
python3 feedback.py
python3 targeted.py
python3 correction.py
python3 deduction.py fit
python3 deduction.py reveng
python3 deduction.py three dense
```
