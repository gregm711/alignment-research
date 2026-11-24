# Local Codex experiments on dynamic learning

One Codex proposal produces one Python candidate. The evaluator buys it at most two verified observations per incident and restarts it with at most 32 bytes of saved state. A second evaluation measures whether it could predict the next observation on a common evidence stream. See the [task, function signatures and pre-run predictions](program.md).

This adapts the editable-candidate/fixed-evaluator pattern in [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). The CPU task has incident-level outcomes and several costs. The search keeps every candidate and rotates the prompts for four families: explicit rules, sparse writes, dense writes and active querying. The family labels steer proposals; inspect the saved code to identify the mechanism actually implemented. All four families begin with the same working active reference. A recently dominated candidate can receive two more attempts. A new architecture still has to earn its place on held-out work.

## Run

Run from this directory with Python 3, Docker, the local `python:3.12-alpine` image, and an authenticated `codex` CLI for proposal generation. The generator uses the signed-in Codex account. Model usage counts against its quota; local candidate evaluation uses no GPU. The default run directory is `runs/revision-v2/` and is excluded from Git.

```sh
python3 orchestrator.py --init
python3 orchestrator.py --rounds 4
python3 orchestrator.py --confirm 0001-rule
```

The last command is an example: use a scored candidate ID from your run. It selects a new random seed, compares the frozen candidate and active reference on the same 640 incidents, and closes that campaign to further development. A partial confirmation can resume for the same ID and seed. Start an independent campaign with `--runs /path/to/new/run`. A new seed from this generator tests sampling variation. A strong scientific claim also needs an independently authored domain held outside this workspace until selection is complete.

Each attempt has a directory containing `candidate.py` and `record.json`. The record captures parent, source hash, hypothesis, prediction and failure criterion before execution, followed by development counts and example failures or an error. A manifest pins the evaluator, instructions, baseline, runner and Docker image. `frontier.json` lists all successful candidates that no other candidate matches on every metric and beats on at least one. Always keeping old values can remain on this list by using zero probes; frontier membership alone is a weak achievement.

A fresh Codex call reflects after five completed proposals or three consecutive off-frontier attempts within a family. Every fifteenth completed proposal triggers a separate review before the next candidate. These calls produce diagnosis and transfer-test records and have no role in scoring. Neither the runner nor candidates commit, reset or clean the working repository. Bounded calls and archived candidate files make the loop resumable. A crashed evaluator trial retains its source and error. If a process dies after filing its proposal, its record remains marked planned.

## Interpret the measurements

Report affected-row repairs, preservation of unaffected rows, false-alarm retention, fresh and previously booked accuracy, mean receipts purchased and serialized bytes. The predictive score is the mean negative log base two of the probability assigned to the actual verified outcome. Every method predicts the same three receipts before seeing them, one depth at a time. Choosing easy probes cannot improve this score. Errors in the candidate-controlled action path still affect task performance. Fresh and booked queries have different row mixes, so compare their absolute levels with care.

The runtime starts a read-only, network-disabled, single-CPU Docker container for each stage, with a 30-second deadline, 128 MB of memory and no mounted repository. Code receives batches of public observations and must solve episodes independently. The container and file hash restrict accidental leakage. This is an honest-research setup. A generator published in the repository and an author who can read all candidate code do not provide an adversarially sealed benchmark. Model calls, interpreter startup and idle hardware matter for real energy use; these measurements cover accuracy, incident memory and probe counts.

## First campaign selection rule

For the bounded two-proposal smoke run, confirm the successful proposal with the highest mean of affected and unaffected accuracy. Break ties by lower bits per receipt and then by fewer probes. Compare it to the active reference on the fresh seed even if neither proposal improves on development. This tests the full loop and provides one same-generator replication. It does not test across domains.

## First end-to-end run, 25 November 2026

Two real Codex proposal calls produced executable candidates. On the 160 development incidents, random-order checking and the active reference both scored 0.538 bits per common receipt because they share a forecast model. Active checking repaired 86.4% of affected rows, compared with 66.9% for random-order checking. Both generated candidates reached the same 0.529 bits per receipt yet chose different trade-offs. The one-byte rule candidate preserved 98.9% of unaffected rows and repaired 89.7% of affected rows. The candidate from the sparse family preserved 93.6% and repaired 98.4%. Both used the published generator probabilities to compute Bayesian forecasts; both ultimately wrote decisions as bits. This was algorithm design with a known simulator, rather than a comparison of trained neural memory architectures.

The predeclared selection rule chose the second candidate. On the same 640 fresh incidents:

| Metric | Active reference | Codex candidate |
|---|---:|---:|
| Bits per common receipt | 0.5724 | 0.5645 |
| Affected rows correctly repaired | 85.0% | 98.9% |
| Unaffected rows preserved | 97.3% | 94.0% |
| Rows preserved on false reports | 97.8% | 94.5% |
| Fresh / previously booked query accuracy | 92.5% / 93.3% | 96.1% / 95.3% |
| Mean verified probes | 1.88 | 1.95 |
| Mean incident bytes after restart | 1.00 | 1.75 |

The gain in repairs survived on fresh data while unintended changes increased. Neither policy dominated. This measures one generator and one fresh seed. The reported improvements cannot identify how much came from the reporter prior, lookahead or decision encoding. Mechanism attribution would require ablations. The two-proposal pilot completed before the first scheduled reflection, and the scheduling paths were exercised separately by the offline check.

[Candidate and hypothesis](pilot/0002-sparse/record.json) · [Candidate source](pilot/0002-sparse/candidate.py) · [First candidate](pilot/0001-rule/candidate.py) · [Fresh-seed counts and settings](pilot/confirmation.json) · [Development frontier](pilot/frontier.json) · [Evaluator manifest](pilot/manifest.json)

The completed campaign is preserved under `pilot/`. The default run command starts a fresh working campaign. `python3 check.py` checks Pareto ordering, registration, lineage, reflection/review scheduling and manifest enforcement offline.

[When to ask for evidence or plan longer](ask-or-plan/README.md) follows the invoice pilot with two preregistered transit test sets, a simple check-or-stop baseline, and local CPU and power measurements.

[Can practice make an unfamiliar tool cheaper to learn?](../teaching-cost/README.md) compares actual BERT weight updates, retrieval and active evidence gathering across lifetimes. It includes a second task family and an equal-byte comparison of predictive loss and teaching cost.
