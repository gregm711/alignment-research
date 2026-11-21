# alignment-research

Small, cheap experiments in AI alignment and interpretability on open-weight models, plus the
reading list and notes behind them.

## Goal

To help with AI alignment and contribute what we learn. By alignment we mean AI that serves
**human flourishing and truth**.

We are here to understand: to learn how current models work and behave, to monitor where they
stand, and to publish whatever we find that is useful to others, including the results that
don't go the way we hoped.

## What's here

| Path | What it is |
|---|---|
| [`jlens-inkling/`](jlens-inkling/) | Experiments on Inkling 2 Mini, a looped transformer: what each loop pass computes, what passes hand to each other, and how robust that is. Start with its [`SUMMARY.md`](jlens-inkling/SUMMARY.md). |
| [`papers/`](papers/) | Index of ~160 alignment and interpretability papers and posts, with the scripts that download them. The files themselves are not committed. |
| [`notes/sota-2026.md`](notes/sota-2026.md) | Where the field stands as of November 2026, from reading the newest work. |
| [`notes/looped-prior-art.md`](notes/looped-prior-art.md) | What two 2026 papers already found about interpreting looped models, and what is still open. |
| [`notes/directions.md`](notes/directions.md) | Research directions we think are worth trying, and why. |
| [`notes/huggingface-landscape.md`](notes/huggingface-landscape.md) | What the open-model community is doing with alignment-adjacent techniques. |

## Rebuilding the paper library

```sh
python3 -I papers/fetch.py papers        # canonical papers (arXiv + a few web-only posts)
python3 -I papers/fetch_labs.py papers   # more from Anthropic, OpenAI, Google DeepMind and others
```

Both are standard-library only. arXiv items are verified against their real titles before saving.

## Ground rules

- Measure and explain; don't build tools whose main use is removing safeguards.
- Write predictions down before running an experiment.
- Report negative results and say what a result does not establish.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
