# Repository Guidelines

## Project Structure

- `jlens-inkling/` contains the Python experiments on the looped Inkling 2 Mini model. Start with `SUMMARY.md` for findings and `README.md` for protocols and limitations. Tracked JSON and text outputs live in `results/`, and GPU costs are recorded in `SPEND.md`.
- `papers/` holds the reading index and two standard-library download scripts. `notes/` contains literature reviews, prior art, and research directions.

## Development Commands

Run these commands from the repository root.

- `python3 -m compileall -q jlens-inkling papers` checks Python syntax without starting a GPU job.
- `python3 jlens-inkling/earlyexit.py --analyse` recomputes the early-exit statistics from checked-in results on the CPU.
- `(cd jlens-inkling && modal run fit.py --minutes 12)` fits a Jacobian lens and saves a checkpoint. Use `--minutes 0` to read an existing checkpoint. Both modes require a Modal account and an 80 GB GPU.
- `python3 -I papers/fetch.py papers` downloads the main collection. Run `python3 -I papers/fetch_labs.py papers` for the lab collection.

## Python Style

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Follow adjacent scripts for argument and output conventions. Keep result filenames descriptive and hyphenated, as in `pass-sweep.json`. The repository has no configured formatter or linter.

## Experiment Validation

Use local checks and experiment-specific controls. The repository has no automated test suite or coverage target. Record predictions and failure criteria in the experiment README before running. Preserve those predictions, report negative results and limitations, and link the raw outputs. Estimate costs before large Modal runs, set hard timeouts, checkpoint long jobs, and ask before exceeding the agreed budget. Keep downloaded papers, weights, logs, credentials, and personal details out of Git.

## Commits and Pull Requests

Use concise, descriptive commit subjects. Follow the history’s experiment labels where helpful, such as `Question 1: message-swap results on both checkpoints`. In pull requests, explain the change and why it matters, list the commands and controls used to verify it, and link relevant issues or result files when available.
