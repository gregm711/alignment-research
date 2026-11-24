# When does local relaxation pay for correcting a belief?

## Protocol fixed before scoring

One stream observes revisions to a chain of 128 real-valued sensors. Adjacent sensors share fixed random positive coupling; each sensor also has its own observation. All methods minimize the same quadratic energy, half the sum of squared observation errors plus half the sum of coupling-weighted squared neighbor differences. Give each method the previous state and the newly revised observations. Start all states at zero. Across 16 independent chain seeds (2026112500–2026112515), run 24 successive updates in four conditions. Change either one sensor or 32 distinct sensors per update. Couplings are uniform random between 0.5 and 1.5 times either 0.25 or 6. Increments are signed, independent and of unit magnitude. Use identical edits for all methods within a condition. Match maximum absolute state error to 0.001 after **every** update. Record per-update error versus a double-precision exact solve, neighbor visits, queue pushes/pops, state writes and single-process CPU time for the update. Report cached precomputation separately. Timings exclude benchmark scoring and represent this Python implementation only.

1. Reuse a tridiagonal factorization and solve the full chain once per edit.
2. Relax the energy by warm-started synchronous Jacobi steps until every absolute residual is at most 0.001. With unit observation anchors the inverse has infinity norm one, which certifies this tolerance.
3. Maintain the signed residual of every node across edits. Only transmit a correction to its neighbors after its residual exceeds the same tolerance. Account for checking arrivals and queue bookkeeping. This is an event-driven residual solver for a quadratic predictive-error model.
4. Use a stronger conventional incremental algorithm. Cache the diagonal of the inverse and left/right attenuation factors in linear space. For each revised sensor propagate its exact inverse column in both directions. For an edit with k changed sensors, truncate each tail once its magnitude drops to 0.001 / (24k), a per-stream error budget even if truncation errors accumulate.

Prediction: on single-sensor weak-coupling edits, event-driven relaxation will cut neighbor visits by at least 5x against synchronous relaxation. An advantage that warrants scaling requires at least 2x fewer neighbor visits than the incremental linear solver, faster median CPU updates against that solver, and all state-error bounds satisfied. Report performance under strong coupling and many simultaneous changes irrespective of the primary result. These tests measure a known solver on a chain. Equilibrium prediction under changing inputs and learning a coupling are separate.

Check the latter with the actual equilibrium-propagation difference of local edge energies at free and nudged equilibria. In independent 12-node networks with 16 seeds, nudge the last node toward a target by adding β times half its squared error to the energy. Compare the estimated gradient for every edge strength against direct symmetric finite differences of the *unperturbed* supervised loss. Record normalized L2 error for forward and centered nudging at β=0.1, 0.01 and 0.001. Require the median centered discrepancy below 0.0001 at β=0.01 and decreasing as β shrinks. Solving both equilibria exactly isolates the mathematical update from the expense of relaxation. No model training benchmark or energy measurement is implied.

## Results

All four solvers stayed within the 0.001 state-error limit on all 1,536 edits (16 chains × 24 edits × four conditions). Event-driven relaxation used 59 times fewer neighbor visits than full synchronous sweeps in the single-change, weak-coupling condition. The stronger incremental baseline needed still fewer visits and CPU time. The threshold for further scaling failed.

| Coupling and edited inputs | Full exact solve | Synchronous sweeps | Event-driven residuals | Incremental inverse tail |
|---|---:|---:|---:|---:|
| Weak, one | 254 | 1,778 | 30 | **12** |
| Weak, 32 | **254** | 2,032 | 560 | 486 |
| Strong, one | 254 | 16,256 | 1,075 | **41** |
| Strong, 32 | **254** | 20,320 | 7,205 | 1,637 |

Cells are median neighbor visits per update, including final settling scans and attempted propagation across a cut-off. The residual solver also needed a median 30 queue operations in the first condition and 7,313 in the last. In the first condition medians were 25 µs for full exact, 348 µs for sweeps, 9 µs for residuals and 2 µs for inverse tails. In the last condition an exact solve took 26 µs and residual relaxation took 2,064 µs. CPU measurements include per-update checks and routing. The counter for state writes covers writes to estimated state, including direct-solver substitutions; queue counts are separate and auxiliary residual-array writes are included in timing. Setup expenses and per-update measurements appear in the raw results. Every method uses linear persistent space. The inverse-tail solver precomputes the exact response to changing one source implicitly in three vectors. Fixed, known, one-dimensional coupling makes this possible. Dense simultaneous corrections can make an ordinary full solve the cheaper option.

For edge learning we perturbed each of 11 couplings separately and compared the finite-difference loss gradient with estimates computed from the local edge energies of nudged equilibria. Median normalized errors were:

| Nudge magnitude | One-sided | Centered |
|---|---:|---:|
| 0.1 | 0.1118 | 0.01235 |
| 0.01 | 0.01222 | 0.0001228 |
| 0.001 | 0.001233 | 0.000001228 |

The centered rule's error decreased roughly one hundredfold per tenfold decrease in nudge size. It missed the registered threshold of 0.0001 at 0.01. This check identifies an accurate local learning signal at sufficiently small nudges in an exactly solved convex system. Settling a nonlinear state, dealing with multiple equilibria and updating weights online remain untested. [Equilibrium propagation](https://arxiv.org/abs/1602.05179)

**Decision.** The activity-sparsity claim survived comparison with synchronous sweeps. The experiment gives no reason to scale this linear residual implementation as a general learner. A future test worth doing would change the couplings between corrections, forcing incremental solvers to pay to rebuild their cache, then compare cost and error with learned, nonlinear dependencies. This experiment measures numerical propagation of revisions to observations. It does not measure semantic inconsistency detection, general intelligence or joules.

**Reproduce** with `python3 dynamic-learning/moonshots/equilibrium/run.py`. The script uses the standard library. Independent dense inversions on 2-, 12- and 128-node chains agreed to within 1e-12 with every one of 142 reconstructed inverse columns and direct solves. Numerical results reproduce with fixed seeds; CPU times vary. Registration SHA-256 before scoring was `8fe3225c4b79ab5c54643084f9c40e01c38a8c7dad2a290b1c4dc797f0c42edd` (the text before the results header, ending with a single newline). [Runner](run.py) · [Per-edit measurements](updates.csv.gz) · [Gradient pairs](gradients.json) · [Summary](summary.json)
