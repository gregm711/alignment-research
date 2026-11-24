# Repair after a real upstream regression report

## Frozen panel and predictions

The controlled experiment distinguishes recognition of an observation from writing a fix. This panel instead uses actual source code, version history, and upstream tests from more-itertools. Eight previously untested revisions are fixed before model generation. They share one library, so these are eight incidents within one project. Some concern similar edge cases. Inspect the original report for the scope of each upstream change.

| Upstream fix | Function tested |
|---|---|
| [71e070b](https://github.com/more-itertools/more-itertools/commit/71e070b35d8c136484e91189719c2a4eccd36fc3) | all_unique |
| [ce676d2](https://github.com/more-itertools/more-itertools/commit/ce676d24136e6efa932df8a4f3bf4801b89e1dd0) | split_before |
| [e6ce1b0](https://github.com/more-itertools/more-itertools/commit/e6ce1b0f65a57c459fc5004ad84158447b65b0ac) | peekable._get_slice |
| [6b1907d](https://github.com/more-itertools/more-itertools/commit/6b1907de293a655fb77881bf81948f95fd15bd7c) | seekable.peek |
| [edb3346](https://github.com/more-itertools/more-itertools/commit/edb3346f835ca917efbfda5e2d6664ab952da369) | numeric_range.__reversed__ |
| [f51a53b](https://github.com/more-itertools/more-itertools/commit/f51a53bfd2fe9504063a33ef5f4a73e30d82d0e2) | interleave_evenly |
| [0e6acdf](https://github.com/more-itertools/more-itertools/commit/0e6acdf9b60765ecf9634d6f5c132ac1bebc616b) | chunked |
| [7847d43](https://github.com/more-itertools/more-itertools/commit/7847d4324a581da8d0919734a8e50b2a6573b780) | value_chain |

Begin with the parent revision, the affected function, and a short researcher-written symptom. The agent chooses a read-only inspection of existing tests or nearby implementation, states its diagnosis and freely writes a candidate function. Exclude the new regression report from this first attempt. Next, reveal the same regression test from the upstream repair in each arm together with the current source and first candidate. Treatments are the full investigation, a short pre-report ordinary handoff, a size-matched extract of tool output, and no earlier narrative. The original source and the candidate remain available in all arms. Each arm gets one attempt at revision. Freeze the first attempt and handoffs before revealing the new test. Count an initially wrong valid patch that later passes as a repair revision. Review what the original diagnosis actually says before counting a change of hypothesis.

For each incident, execute the complete affected upstream test class plus independent withheld variants using a disposable local Python container. The named upstream regression is visible at the revision step. Report passes on the rest of the class and withheld examples too. Validate that the original function fails and the upstream function passes before model trials. Run local Qwen2.5-Coder 7B first. Predict at least three of eight initial candidates will fail despite being valid replacements, and fresh report plus current source alone will recover at least two. Compare all arms on all cases including cases already solved. A penalty specific to compression requires full-history successes lost under ordinary compaction. If the fresh report alone suffices, record that the old history buys little here.

The public report shown to the model is reconstructed. The code and tests are historical. No original human or production-agent conversation is claimed. [Upstream license](LICENSE).

## Reproduction

Use Docker and a local Git clone containing the pinned revisions. The runner reads package source from the parent commit and tests from the fix commit. It replaces only the affected function. For a multi-function upstream fix, the panel tests the function named in the table. Each trial gets a read-only temporary checkout inside a network-disabled container. Raw model-generated Python runs only there. The model receives neither withheld examples nor their pass counts.

```sh
git clone --filter=blob:none --no-checkout https://github.com/more-itertools/more-itertools.git /tmp/alignment-more-itertools
docker pull python@sha256:c5ba3deef00c7758561a0f03c26f44f10589263d40e732b8748e0d1f38f6dd89
python3 panel.py
python3 run.py --recount
```

Run these commands from this folder. Set `MORE_ITERTOOLS_REPO` to reuse a different local clone. After the controls pass, launching `python3 run.py` generates or resumes model trials through local Ollama with the same tokenizer as the parent study. The parent/fixed controls independently failed/passed both the affected test class and withheld variants on all eight incidents. The eight classes contain 92 test methods in total. Revisions in the table link to the upstream code and reports.

Run the same panel with Gemma 3 12B after Qwen. This replication generates each reader's own investigation, first patch and summary. Compare arms within model, conditional on the same starting patch, and count initially invalid outputs separately. Because the two models can begin from different patches, their raw recovery denominators may differ. The pre-fork extraction budget continues to use Qwen tokenization. Report actual per-reader prompt lengths. This addition was recorded after the first Qwen incident was complete and before any Gemma upstream trial.

## Does the patch left in the worktree carry the mistake?

The no-narrative arm still shows the agent its own current candidate. To distinguish that persistent code from conversational memory, run three additional no-narrative controls per incident. Use the same reporter test and parent function. Remove the candidate block entirely for a fresh attempt from the original checkout. Replace the candidate with the unmodified parent function to control for a second copy of existing code. Replace it with the actual upstream fix as an answer-assisted positive control to measure the interface's ability to retain a working patch. Compare each against the already registered no-narrative arm with the model's own draft. Run all eight cases, including initially correct cases. Preserve output limits and the executable grader. These prompt lengths differ. A fresh-start win is an effect of removing the draft in this particular interface. It does not prove a psychological commitment mechanism. Predict that at least one reader will solve more cases after removing its draft, and that each reader will keep a known-good candidate on a majority of cases. This was added after observing some Qwen primary failures, before any of the new control calls or Gemma panel outputs.

## Did the model fix the reported bug but break something else?

A zero on the complete test class and withheld inputs can hide partial recovery. Re-execute each saved first draft and revision. Record whether the exact test supplied by the reporter passed, how many other upstream test methods passed, and whether the withheld variants passed. Compare revisions with the first draft on the same incident. Report an improvement on the named test separately from a complete repair. A passing named test with failing withheld variants could reflect overfitting or incomplete implementation. Count identical candidate functions once for execution and reuse their result. The parent must fail the named test and the fixed implementation must pass it on every incident.
