# Reproducing the local code-generation comparator

Install Ollama and serve the locally available `qwen2.5-coder:7b` on its default loopback port. The runner verifies the model digest against the constant in `llm.py` before generating. Use Python with NumPy and the two panel files produced by the source collectors. From the experiment directory run the following commands in this order to reproduce the cache-sharing pattern within each project.

```sh
for cap in 2 1 3; do
    python3 study.py dates codegen "$cap"
    python3 study.py phones codegen "$cap"
done
```

Each introduction and each failed paid check triggers one generation attempt. The prompt includes the producer, current paid examples, incumbent program, one stored introduction example and the previous failed example when available. The comparator separately fits an ordinary formatter. It selects the candidate matching the most purchased current outputs, breaking ties in favor of the incumbent, then ordinary synthesis, then the model proposal. The next revision checks the last failing input first. A short Python expression returned by Qwen runs in a restricted interpreter. A malformed proposal is counted as a generation attempt.

Outputs are stored under `results/`. The top-level `model_calls` and `model_tokens` count logical generation attempts and generated tokens over a full stream. The nested `codegen_cost` object also records prompt tokens. Fields beginning with `actual_` count generations and tokens that were physically served during this run. Identical requests reuse the saved response under `~/.cache/alignment-research/skill-witness/codegen/`; the cache key covers the model digest and exact request bytes. On a warm cache, the historical token counts are replayed and physical counts fall. Ollama reports input tokens actually evaluated during the recorded request, which can depend on prefix caching. Neither accounting method measures joules.

The explicit comparator's `state_bytes` counts the UTF-8 JSON encoding of its live per-producer operation and examples. Model weights, runtime memory and the reproducibility cache are separate resources. The verifier-call count in the study concerns purchased formatter outputs and excludes model generation.
