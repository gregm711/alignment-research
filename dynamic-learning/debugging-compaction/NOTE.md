# When a debugging handoff preserves the observation, can the agent use it?

Two local models could recover a correction from a compact handoff when it retained an attributed input/output example. Giving either model that handoff still produced zero passing reversal repairs in eight controlled cases. The full transcript also produced few repairs. A model’s ability to tell the user what happened and its ability to change the implementation are different measurements.

## The controlled comparison

An engineer initially suspects one consumer build. After handoff a request trace identifies the actual build. Two archived builds behaved identically on smoke tests but differently at a boundary. The same pre-fork memory had to support both possible routes. The model either chose the archived output in a separate recognition trial or wrote a replacement function tested on additional inputs. A recognition success required selecting the right answer with the two choices in both possible orders. Each cell below counts cases out of eight on the route that overturned the initial diagnosis.

| Reader and outcome | Full history | Ordinary summary | Attributed example | Source labels swapped | No history |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder 7B, recognizes the right output | 8 | 1 | 6 | 0 | 1 |
| Qwen2.5-Coder 7B, writes a passing patch | 1 | 0 | 0 | 0 | 0 |
| Gemma 3 12B, recognizes the right output | 8 | 1 | 7 | 0 | 3 |
| Gemma 3 12B, writes a passing patch | 0 | 0 | 0 | 0 | 0 |

The extractor records the first observed input on which the two builds disagree and the output each build produced. It runs before the request trace identifies the affected build, without reading hidden tests. The record replaces part of the ordinary summary. Both readers received identical text written by Qwen or extracted from the archive. Qwen's observed and ordinary prompts matched exactly in native input tokens on every pair. Gemma's corresponding prompts differed by −2 to +7 tokens. The true and swapped observations used identical remaining ordinary prose and matched each other in native tokens in both models.

Five Qwen cases and seven Gemma cases changed from consistently selecting the right output to consistently selecting the opposite output when the two source labels were swapped. This intervention gives direct evidence that source attribution influenced recognition on those cases. Recognition improved on six cases and fell on one for Qwen when comparing the attributed example with the ordinary summary. Gemma improved on six and fell on zero. The readers saw the same eight cases, so those counts cannot be pooled as sixteen independent incidents. [Qwen recognition trials](results/readout-qwen2.5-coder-7b.json) · [Gemma recognition trials](results/readout-gemma3-12b.json)

In the batching case, one build kept an incomplete batch and the other raised ValueError. Qwen correctly identified which behavior belonged to the affected build in both answer orders. Its replacement function still returned the final short batch. Both the ordinary and attributed repair prompts used 196 input tokens. This example can be inspected from the original handoff through the complete hidden test outputs. [Ordinary note and patches](results/qwen2.5-coder-7b/batch.json) · [True and swapped observations and patches](results/restore-qwen2.5-coder-7b/batch.json)

## What the patch failures establish

The registered feasibility threshold required the full history to solve both routes in at least four cases. Both readers scored zero of eight on that measure. Qwen's full-history reversal patches reproduced the original buggy behavior in four cases, implemented the opposite build in three, and matched the target in one. Gemma reproduced the original buggy behavior in seven and produced another incorrect behavior in one. For either reader, explicitly instructing the summarizer to preserve competing evidence yielded zero passing reversal patches. The experiment cannot estimate how much ordinary compaction degrades an otherwise capable debugging agent. Old source code remained visible in every arm and gave the reader an easy candidate to repeat. The suspected route also always had the same name, so a general tendency to cling to a prior belief is unmeasured here. [Predictions and protocol](README.md) · [Qwen raw patches](results/restore-qwen2.5-coder-7b/) · [Gemma raw patches](results/restore-gemma3-12b/)

In a separate replay of this repository's real scoring bugs, Qwen inspected two tool outputs, including evidence that the linked-step scorer missed two relations on a real row. It nevertheless wrote that the linked-step scorer appeared correct and blamed the final-answer scorer. This is a documented instance of a mistaken self-generated diagnosis. The model's subsequent attempts to edit two assignment expressions all failed the repair interface, including a retry with constrained JSON. Both attempted archive lookups also used invalid tool names. Those failures leave the effect of compaction on that incident unresolved. [Complete trace and edits](results/scorer-incident-qwen2.5-coder-7b.json)
