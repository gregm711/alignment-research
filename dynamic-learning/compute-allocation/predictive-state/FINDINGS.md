# Which representations generalize beyond the examples?

A classical recurrent learner recovered a new coordinate system from binary input/output observations. Applied to previously unseen rules, it sometimes made long computations cheaper and sometimes lost to a small state table. All reported solves below require success both before and after the change.

## Same evidence, changing computation

The learner represents a history by the observable outcomes of possible continuations. It keeps independent response patterns and infers what appending either symbol does in those coordinates. We supplied the finite-field linear-algebra prior. The dimensions, coordinates and operators were fitted per machine from observations. This is an application of classical Hankel realization. [Learning background](https://aclanthology.org/D14-2002/)

All arms received 255 short outcomes and 12 guard examples in each phase. With latent-rank caps 2, 4, 8 and 15, the highest generated dimension reaching at least 90% observed all-string success was **2, 4, 8 and 8**, respectively. The 8-dimensional result was 22/24. With 24 cases, even a clean 22/24 is a noisy estimate of reliability. Scrambling the continuation alignment eliminated every certified solve on the generated panels.

A separate auditor used the hidden generator to decide whether the fitted realization agreed on **every finite string**. It checks an invariant of the paired machines. Both start inside a finite-dimensional subspace, each possible next input stays inside, and the outputs agree throughout that subspace. The learner received neither generator nor certificate. An independent verifier checks a saved proof for one eight-dimensional case and its revision, without importing the simulator. Run `python3 verify.py` from this directory. [Example proof](example-proof.json) · [Verifier](verify.py) · [Generated outcomes, costs and audits](generated.json)

On the seven independently specified Tomita languages, all 42 ordered changes were certified. The original sampler struggled with rare accepted strings and used its documented periodic sampling repair on 12 of 84 phases. The all-string result is independent of that sampling. The seven grammars are the distinct external units. A rerun allowed a conventional state learner seven different allocations of the same data and interleaved all arms for seven timing rounds.

| Panel | Coordinate discovery | Expanded state learner | Manual cascade with discovery fallback |
|---|---:|---:|---:|
| Tomita, 42 ordered revisions | 42 at 2.61 mJ/solve | 30 at 2.96 mJ/solve | 42 at 3.25 mJ/solve |
| Six-dimensional generated machines, 24 revisions | 22 at 4.22 mJ/solve | 0 | 22 at 9.22 mJ/solve |
| Word-count polynomial controls, 24 revisions | 24 at 3.35 mJ/solve | 24 at 1.95 mJ/solve | 24 at 5.25 mJ/solve |

The external comparison gives about a 20% measured saving against the cascade at equal accuracy. The expanded state learner was cheaper at equal accuracy on the polynomial controls. These results cover the implemented solvers. The state learner is a deterministic observation-table baseline. More elaborate state-merging algorithms remain possible. [Original external outcomes](tomita-results.json) · [Interleaved energy trials](layout-results.json) · [Pinned source machines](https://github.com/LC-John/RNN2DFA/blob/ffa40c0a3c295e6e991dc983aa8c89c9fac928db/tomita/tomita.py)

## A single problem with extreme compositional depth

A separate 12-world-per-width panel used fresh reversible operators. The visible program specified four shared blocks at each level, each formed by concatenating two blocks from the previous level. Level 96 describes a block of length 2^96. The forms of the transitions and the composed answer were hidden. Three executors shared the same fitted representation, data, hardware and program. One memoized reached states. Another composed the learned operators on a basis. A third compiled their action on every latent state.

At eight dimensions, basis composition solved **11/12 at level 96** within 4,096 work units per phase and about **21.7 mJ per solve**. The other two recovered 0/12 at that allowance, and 0/12 even at 32,768 units at level 96. Eleven was the short-evidence identification ceiling. Dropping the block or substituting level six failed all twelve episodes. On level 24 all three executed 11/12 at a common 32,768-unit cap. They used **8.16 mJ** for basis composition, **16.88 mJ** for dense composition and **89.16 mJ** for memoization. At four dimensions and level 96, dense and basis compilation both solved 12/12 and used similar energy.

The work allowance is an implementation count, independent of the process-energy measurement. Long-block reference outputs came from a separate composition implementation checked against literal expansion at level six. The input explicitly supplied a shared program. This experiment tests using newly learned structure to evaluate that program. It cannot establish superhuman mathematical invention, training scalability or performance on arbitrary uncompressed computations. [Raw budget and depth curves](depth-results.json)

## A change invisible throughout training

Six possible changes agreed on every observation of length at most twelve. They also agreed on a single later contrary observation containing a new 15- or 31-bit pattern. Each learner could then query the changed machine, with every transmitted bit and probe charged. The explicit library covered exact match, prefix, suffix and contiguous occurrence. The alternative used prefixes and suffixes of the witnessed failure as candidate coordinates, and learned the remaining effect from device responses. The unchanged base machine was composed with that newly recovered effect.

On all **24 off-menu revisions** (odd occurrence parity or ordered subsequence), the four named shortcuts abstained. Coordinate discovery and the cascade with discovery as fallback both recovered the exact revised rule once the budget reached 2,048. Discovery needed **294 paid probes at length 15 and 1,094 at length 31**. On the four familiar scopes the explicit learner achieved the same accuracy as discovery with **18 probes**, including verification. Increasing the detection ability by adding both off-menu functions to the menu would change the task. More importantly, the explicit library already retains its advantage across this particular six-scope panel by trying its cheap guesses first and falling back when necessary. We have no evidence for a universal discovery-first policy.

The learning code was frozen before implementing the two off-menu simulators. The positive observation was the bare motif. The learner was given both its old fitted machine and access to the new device. Real changes can have irrelevant context, noise, or no trusted way to query the world. This experiment isolates how to construct a new representation after genuine contrary evidence arrives. [Frozen learner](revision.py) · [Revision outcomes, query traffic and CPU costs](changed-results.json)

## What to optimize across architectures

Use **joules per verified solve at a fixed observation allowance and target reliability**. Charge every acquisition, verification attempt, failed branch and revision inside the deployment boundary. Include failed episodes in the numerator. Mark a reliability target as unreached whenever the solver misses it. When there is irreducible noise, track held-out predictive log loss against a calibrated attainable target. Keep the physical cost of acquiring a new observation explicit.

Track two distinct slopes. At fixed training, measure the hardest *individual problem* solved reliably as inference energy grows. Then vary training data, training joules and model size independently enough to compare frontiers against both standard and recurrent Transformers. Containing a Transformer as a special case establishes expressivity; the optimizer, hardware and overhead decide the realized scaling curve. Alternative architectures can compete directly.

These experiments ran sequential lightweight CPU jobs. Energy is XNU-attributed process CPU energy, including deployment-side fit and prediction. The strongest comparisons above used batched interleaved timing or long-running panels. Some brief revision timing samples were missing or noisy, so the revision conclusion uses metered communication counts. Exact theorem auditing is tracked separately from prediction in the initial study. There was zero offline neural training. General neural representation discovery, few-shot adaptation beyond these rule classes, measured wall-plug power, and model/data scaling against a Transformer remain open.

[Preregistered protocols](README.md) · [Primary study](study.py) · [Extreme-depth execution](depth.py) · [Change simulator](changed-worlds.py)
