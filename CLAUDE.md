# alignment-research

## Goal

Help with AI alignment and contribute what we learn. Alignment here means AI that serves
**human flourishing and truth**. The work is about understanding: learning how current models
work and behave, monitoring where they stand, and publishing what is useful to others.

When choosing what to do next, prefer the option that teaches us or other people something
true about models over the one that only looks impressive.

## How to work in this repo

- **This repo is published.** Keep third-party papers, logs, model files, keys and personal
  details out of git. `.gitignore` already covers the downloads, `*.log` and `*.pt`.
- **Save what we learn.** Findings, reading notes and reasoning go in `notes/` or the
  experiment's README, not only in chat.
- **Predictions first.** Write them in the experiment's README before running it, and leave
  them there afterwards whether or not they held.
- **Be truthful about results.** Report negative results, state how deep any reading or
  evaluation went, and say what a result does not establish.
- **Measure and explain.** No jailbreaks and no tooling whose main use is removing safeguards.
  Measuring what removing them does is in scope.
- **Spend carefully.** GPU jobs run on Modal. Give a cost estimate before a large run, keep hard
  timeouts on every job, checkpoint so runs resume, and ask before exceeding the agreed budget.

## Layout

- `jlens-inkling/`: Jacobian lens on Inkling 2 Mini (looped transformer).
- `grounding/`: monitors for correct answers that ignore the supplied passage; NQ-Swap audit.
- `benchmark-audits/`: the same leak check applied to other datasets (CPU only).
- `inside-advantage/`: text-only baselines against published internal monitors (CPU only).
- `loop-legibility/`: how much reasoning leaves the transcript when a model loops, and whether a probe recovers it.
- `papers/`: index and download scripts for the reading library.
- `notes/`: state of the art, research directions, Hugging Face landscape.
