# Does a brief scratchpad repair fluid-reasoning mistakes?

## Before seeing diagnostic outcomes

Use the local Qwen3-4B-Instruct-2507 4-bit checkpoint and the fluid study's original balanced development seed. Take the first six entire worlds in fixture order in each of its two domains. If there are multiple representations of one world, use one canonical full-history representation. Score both futures per world. Use the same question, clue, history and answer options across conditions. The diagnostic never receives outcomes before inference.

1. Direct: force the original next-token A/B choice.
2. Blank notes: keep an empty scratch-notes field, then force the same choice.
3. Own notes: ask for brief intermediate deductions with no task-family hints, generate greedily with a hard cap of 48 tokens, append the resulting notes to the original evidence using the blank-notes framing and force A/B.

The two passes have separate model state. The model can write an answer in its notes despite the instruction. Disclose the frequency of capped notes; count actual generated scratch tokens, input tokens and elapsed time separately. For each named domain and overall report question accuracy, both-futures accuracy and paired repairs/overturns. Give the first repair, first overturn and first unchanged world by ID if present. Preserve per-question records, notes, direct answers and gold for reproducibility. Passing the diagnostic requires a net gain of at least two correct questions against direct and at least one against blank notes. With 12 worlds, this is a development diagnostic of explicit reasoning at this particular budget. A negative result leaves longer or trained latent reasoning unresolved.

Fixture clarification before opening any fluid outcomes: the generator has one task family at two depths. Use worlds 0–5 at both depths and both forks (seed 776031), giving six independent worlds, twelve world-depth pairs and 24 answers. Summaries labeled by domain above therefore report depth. A matched blank-notes control separates effects of the second prompt from information in the notes. Other concurrent work on this laptop can affect wall time.
