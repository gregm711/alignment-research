# Choosing a frozen reader for a bounded memory writer

## Stage 0 decision fixed before model scores

The development screen uses constructed forks over familiar datasets. Give the same pretrained reader an entire history, then ask one present question and each of two future questions separately. The model sees only one question at a time. Score its choice between A and B from next-token logits using the model's native chat template. The hidden history, later clue, allowed answers and order match across reader candidates. Nothing about the confirmation corpus is used to choose a reader.

Screen the 350M and 1.2B Liquid instruction models and a conventional Qwen dense instruction checkpoint on the same 24 development episodes (seed 4701). Report present, each future and the fraction getting both futures right. Require at least 90% present and 80% joint future accuracy within **each** development domain. Check that withdrawing history breaks joint future performance by at least 30 percentage points overall. Models must also support finite nonzero gradients to continuous input embeddings through frozen weights. Prefer the smallest passing model. If conventional non-hybrid attention wins at similar size and measured cost, it supplies a useful replication backbone. If none passes, extend to a stronger open checkpoint without lowering the gate. Replicate the selected reader on 48 new development episodes (seed 4702) against the same thresholds before recommending it. This screens the base reader, not either trained memory writer. Sampling uncertainty at these sizes is appreciable.

On development only, pick one fixed persistent byte allowance in a range that forces ordinary text truncation and still leaves enough information for a useful encoding. Run the selected reader with the UTF-8 suffix of the history under that allowance. Count how many episodes fit in full, whether both branches are solvable after truncation, and whether a simple pre-clue compact representation can preserve both. Reject a budget as an architectural test if an obvious lossless task-specific encoder fits and saturates both future scores. That is an informative control and can motivate a richer task. Changing the task after a screen requires a newly frozen evaluation.

## Training experiment once the setup qualifies

Keep one frozen base reader, one architecture for writing memory, identical fixed capacity, common training histories, optimizer, update and compute budgets. Randomize seeds in paired runs. Train the present-only writer using present questions; train the second writer with both present and future supervision. Match the number of targets and optimizer steps by sampling equal target counts. Neither writer sees which branch actually occurs before it stores its state. Reset all activations, KV caches and instance-dependent adapters; count all surviving episode-specific bytes including lengths, addressing and scales. Keep global trained parameters separate and match them between arms. The evaluator supplies the same future clue at no differential cost. Compare full context, empty context, byte-matched raw evidence, summary, retrieval and a compact explicit representation.

Report present accuracy, both-futures-correct accuracy, each fork, paired intervals by independent history and three separate matched slices: present answers agree, both present answers are correct, and both present answers are wrong. Give each denominator. Identical present predictions on the agreement slice are guaranteed by selection; the unconditional estimates remain primary. Transplant stored states between histories matched on present truth and query/clue, with opposite future truth; compare to identity and same-future donors. Cross check selectors with writers only after the fixed-clue comparison, and replay exact purchased evidence. Freeze architecture, byte allowance and scoring before choosing outcomes from new confirmation data. This stage has not authorized or incurred a paid training run.

Add SmolLM2-135M-Instruct as a cheap lower-size control before any model outcomes. Give it the same full-history and empty-history screen; a pass would be eligible for the same confirmation. The 56-byte development cap and evidence checks are reported in [dev.md](dev.md).

If all four initial checkpoints miss the per-domain screen, test Qwen3-4B-Instruct-2507 using its pinned MLX 4-bit conversion. Keep the same questions, byte accounting, fresh-seed replication and A/B readout. Different numerical precision and execution runtime limit direct timing and size comparisons. A stronger frozen reader can still qualify as a development reference while the task-specific compact controls preclude using these fixtures alone for an architectural claim.

## Reader selection

All five open checkpoints ran locally on the same 24 episodes. The unit is a history: both futures must be answered correctly to count as one joint success. All accepted gradients to an early continuous input with frozen decoder parameters. The full-history screen still rejected every model. The best joint count only exceeds the next by one history; this panel cannot establish a meaningful family ranking. Timings and precision differ by backend. [Artificial Analysis and Hugging Face catalog](catalog.md) · [PyTorch runtime checks](results/runtime-checks.json) · [MLX reader and gradient check](reader_mlx.py)

| Reader | Present with full history /24 | Both futures with full history /24 | Both with empty history /24 | Both with 56-byte suffix /24 |
|---|---:|---:|---:|---:|
| SmolLM2-135M-Instruct | 10 | 6 | 6 | 6 |
| LFM2.5-350M | 11 | 5 | 4 | 6 |
| LFM2.5-1.2B-Instruct | 11 | 5 | 1 | 5 |
| Qwen3-1.7B | 18 | 11 | 4 | 2 |
| Qwen3-4B-Instruct-2507, MLX 4-bit | 19 | 12 | 1 | 3 |

We retained the 4B checkpoint as a diagnostic reference and ran it on 48 new development histories. With full context it solved both futures on 10/24 date and 12/24 intent histories (22/48 combined) and got 33/48 present decisions correct. Its weak performance leaves the chief source of any later failure ambiguous. A 56-byte writer using pretrained MiniLM signatures on intent and fitted date grammar on dates solved 44/48 forecasting pairs with no per-episode intent labels. The baseline gets a task-specific encoder and a pretrained representation; its 56-byte limit applies to persistent episode state. The registered spending gate stays closed. [Development controls](dev.md) · [Qwen replication](results/qwen4-replication.json)

The matched-present slice and state-swap interventions belong to a comparison of trained writers. No writer was trained in this stage, no confirmation corpus was opened, and all compute was local.
