# Online revision on historical formatter releases

Setup calls count toward total cost. Accuracy, solved revisions, conditional false accepts and regressions use subsequent releases. The primary cap is two checks. Each interval resamples entire paired producer histories 10,000 times. Shared model state and shared upstream changes can correlate different producers, so these intervals describe the observed streams conditionally.

## dates

| Arm at cap 2 | Calls incl. setup | Solved revisions | Exact rows | Newly broken / eligible | False accepts / accepted | Solved after change |
|---|---:|---:|---:|---:|---:|---:|
| native | 716 | 22/229 | 35.5% | 2267/10151 | 83/103 | 0/7 |
| ordered | 705 | 20/229 | 47.1% | 2019/12187 | 54/74 | 0/7 |
| ordinary | 536 | 229/229 | 100.0% | 0/22644 | 0/226 | 7/7 |
| codegen | 536 | 229/229 | 100.0% | 0/22644 | 0/226 | 7/7 |
| swapped | 713 | 77/229 | 65.4% | 2933/15871 | 66/104 | 1/7 |
| frozen | 678 | 80/229 | 35.2% | 1836/9960 | 0/80 | 1/7 |
| wide | 632 | 117/229 | 77.7% | 751/18287 | 45/162 | 0/7 |

Relative call savings, solved-revision and regression differences compare native with the named arm. The last column counts from calibration as release one and requires strictly lower cumulative calls with cumulative solved-revision rate within two percentage points at every remaining prefix.

| At cap 2 | Call savings (95% CI) | Solved difference (95% CI) | Regression difference (95% CI) | First durable break-even |
|---|---:|---:|---:|---:|
| ordered | -1.6% [-4.6%, 0.6%] | 0.9% [0.0%, 2.7%] | 5.8% [0.8%, 13.0%] | none observed |
| ordinary | -33.6% [-38.4%, -27.9%] | -90.4% [-99.6%, -79.1%] | 22.3% [15.3%, 32.7%] | none observed |
| codegen | -33.6% [-38.4%, -27.9%] | -90.4% [-99.6%, -79.1%] | 22.3% [15.3%, 32.7%] | none observed |
| swapped | -0.4% [-2.8%, 1.1%] | -24.0% [-32.0%, -16.4%] | 3.9% [-1.5%, 12.0%] | none observed |
| frozen | -5.6% [-10.5%, -1.6%] | -25.3% [-41.2%, -11.3%] | 3.9% [-8.2%, 13.5%] | none observed |
| wide | -13.3% [-21.0%, -6.4%] | -41.5% [-58.0%, -25.7%] | 18.2% [11.1%, 28.5%] | none observed |

With 300 setup calls and one compulsory check on each of 229 revisions, every policy spends at least **529 checks**. Against ordinary (536 calls), the most any policy could save is **1.31%** even with perfect accuracy. Against codegen (536 calls), the most any policy could save is **1.31%** even with perfect accuracy.

At one check per revision the native and fixed-order selectors pay identical costs. The paired difference in exact-row accuracy is -11.0% [-18.4%, -4.2%], and the difference in solved revisions is -6.1% [-9.9%, -3.0%].

Persistent state at cap two uses 2,482,176 bytes for native, 9,928,704 bytes for wide, 4,082 bytes for ordinary, 4,082 bytes for codegen. This counts the neural fast-weight tensor or the serialized explicit memory. The query weights, generator weights, inference buffers and source data are separate.

The local generator was invoked on 40 prompts at cap two (21,040 prompt tokens and 1,082 generated tokens, including exact cache replays). These tokens are separate from verified formatter calls.

The joint cap-two gate was **missed** for dates.

| Arm | Cap 1 calls / solved | Cap 2 calls / solved | Cap 3 calls / solved |
|---|---:|---:|---:|
| native | 529 / 11.4% | 716 / 9.6% | 836 / 12.2% |
| ordered | 529 / 17.5% | 705 / 8.7% | 861 / 14.8% |
| ordinary | 529 / 93.9% | 536 / 100.0% | 539 / 100.0% |
| codegen | 529 / 93.9% | 536 / 100.0% | 539 / 100.0% |
| swapped | unavailable | 713 / 33.6% | unavailable |
| frozen | unavailable | 678 / 34.9% | unavailable |
| wide | unavailable | 632 / 51.1% | unavailable |

## phones

| Arm at cap 2 | Calls incl. setup | Solved revisions | Exact rows | Newly broken / eligible | False accepts / accepted | Solved after change |
|---|---:|---:|---:|---:|---:|---:|
| native | 842 | 0/300 | 22.1% | 699/1974 | 79/79 | 0/26 |
| ordered | 849 | 0/300 | 10.8% | 352/1100 | 97/97 | 0/26 |
| ordinary | 611 | 40/300 | 59.1% | 30/5254 | 251/291 | 0/26 |
| codegen | 609 | 40/300 | 59.1% | 40/5294 | 251/291 | 0/26 |
| swapped | 843 | 0/300 | 11.9% | 590/1196 | 90/90 | 0/26 |
| frozen | 880 | 0/300 | 8.8% | 168/947 | 40/40 | 0/26 |
| wide | 732 | 0/300 | 27.0% | 410/2400 | 187/187 | 0/26 |

Relative call savings, solved-revision and regression differences compare native with the named arm. The last column counts from calibration as release one and requires strictly lower cumulative calls with cumulative solved-revision rate within two percentage points at every remaining prefix.

| At cap 2 | Call savings (95% CI) | Solved difference (95% CI) | Regression difference (95% CI) | First durable break-even |
|---|---:|---:|---:|---:|
| ordered | 0.8% [-3.0%, 4.8%] | 0.0% [0.0%, 0.0%] | 3.4% [-5.9%, 14.1%] | 5 |
| ordinary | -37.8% [-43.1%, -32.0%] | -13.3% [-26.7%, -3.3%] | 34.8% [26.0%, 45.5%] | none observed |
| codegen | -38.3% [-43.6%, -32.4%] | -13.3% [-26.7%, -3.3%] | 34.7% [25.8%, 45.3%] | none observed |
| swapped | 0.1% [-2.2%, 2.3%] | 0.0% [0.0%, 0.0%] | -13.9% [-24.7%, -3.6%] | 11 |
| frozen | 4.3% [-1.2%, 9.3%] | 0.0% [0.0%, 0.0%] | 17.7% [-5.4%, 32.2%] | 4 |
| wide | -15.0% [-20.0%, -10.4%] | 0.0% [0.0%, 0.0%] | 18.3% [7.5%, 30.1%] | none observed |

With 300 setup calls and one compulsory check on each of 300 revisions, every policy spends at least **600 checks**. Against ordinary (611 calls), the most any policy could save is **1.80%** even with perfect accuracy. Against codegen (609 calls), the most any policy could save is **1.48%** even with perfect accuracy.

At one check per revision the native and fixed-order selectors pay identical costs. The paired difference in exact-row accuracy is 5.9% [2.2%, 10.0%], and the difference in solved revisions is 0.0% [0.0%, 0.0%].

Persistent state at cap two uses 2,482,176 bytes for native, 9,928,704 bytes for wide, 8,024 bytes for ordinary, 7,903 bytes for codegen. This counts the neural fast-weight tensor or the serialized explicit memory. The query weights, generator weights, inference buffers and source data are separate.

The local generator was invoked on 48 prompts at cap two (21,933 prompt tokens and 1,588 generated tokens, including exact cache replays). These tokens are separate from verified formatter calls.

The joint cap-two gate was **missed** for phones.

| Arm | Cap 1 calls / solved | Cap 2 calls / solved | Cap 3 calls / solved |
|---|---:|---:|---:|
| native | 600 / 0.0% | 842 / 0.0% | 1081 / 0.0% |
| ordered | 600 / 0.0% | 849 / 0.0% | 1077 / 0.0% |
| ordinary | 600 / 13.3% | 611 / 13.3% | 618 / 13.3% |
| codegen | 600 / 13.3% | 609 / 13.3% | 616 / 13.3% |
| swapped | unavailable | 843 / 0.0% | unavailable |
| frozen | unavailable | 880 / 0.0% | unavailable |
| wide | unavailable | 732 / 0.0% | unavailable |

The joint gate requires at least 20% call savings at the point estimate, a positive lower bound on savings, a lower bound of at least −2 percentage points on solved revisions, and an upper bound below +2 points on excess regressions against each practical control. A missing comparison cannot establish a pass. Additional paired intervals, discordant revision counts and full cumulative curves are in [summary.json](results/summary.json).
