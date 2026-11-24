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

## Where this fits

MemoryArena already showed that good recall scores can coexist with poor performance on tasks requiring an agent to apply memory. CompAct-Bench found that summaries can distort the status of unresolved issues. Yao and colleagues evaluated whether a summary made for one question retains evidence needed for a later one. Salle and colleagues found that an agent can often recover missing information from the worktree. The narrower contribution tested here is a paired intervention: exchange *who produced an observed output* while keeping the observation and the available space essentially identical, then independently grade recognition and executable repair. The controlled recognition result warrants the source-swap check. A claim about real debugging outcomes needs evidence from the upstream panel and broader replications. [MemoryArena](https://arxiv.org/abs/2602.16313) · [CompAct-Bench](https://arxiv.org/abs/2611.25673) · [Later-question evaluation](https://arxiv.org/abs/2610.27301) · [Compaction-friendly benchmarks](https://arxiv.org/abs/2610.31778)

## What happened when Qwen got a second repair attempt

Every arm started from the same ordinary-summary patch for the given case and route. Qwen then saw either the recorded expectation alone or the result of running a saved check on its draft. Total model input tokens matched exactly within each case and route. The retained observation came from the archive and the hidden evaluation inputs stayed hidden.

| Second-turn input | Passing reversal patches | Passing confirmation patches |
|---|---:|---:|
| Executed check with the correct source attribution | 2/8 | 2/8 |
| Same input and expectation, unexecuted | 1/8 | 2/8 |
| Executed check on a shared smoke input | 0/8 | 2/8 |
| Executed check with the source labels swapped | 1/8 | 2/8 |

The batching repair passed with the correct expectation whether or not the check ran. The deduplication repair passed with an executed check even when its source labels were swapped. The registered count thresholds were met, but neither individual rescue distinguishes a benefit of executing the **correctly attributed** evidence from both controls. Two wins against the common check in eight cases are weak evidence of a transferable repair advantage. [All 64 second-turn patches and scores](results/feedback-qwen2.5-coder-7b.json)

Selecting among four independently proposed patches offered little additional headroom. Qwen generated the candidate pool using only the buggy source, task purpose and shared smoke examples. On reversal tasks, a passing patch existed in only two of eight pools. Screening the candidates using the attributed example found both in forward order, versus one with the smoke example. In reverse order both methods found one. In five cases no candidate even matched the attributed example. One selected candidate matched that example but failed the held-out suite. This missed the registered prediction of two additional repairs. [All 32 candidates, checks and order comparisons](results/select-patch-qwen2.5-coder-7b.json)

[Open the offline case replay](replay.html) to inspect either reader's initial one-turn recognition and repair trials.
