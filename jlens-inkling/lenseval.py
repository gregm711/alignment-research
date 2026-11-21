"""Follow-up: can a lens read the intermediate step on Anthropic's multi-hop prompts?

    modal run lenseval.py        # writes results/lens-eval-multihop.json

Needs the per-pass lenses from `fit.py --passes 1,2,3` on the Modal volume. Metric from the
evaluation set's own README: pass@k = share of items whose intermediate reaches lens rank <= k at any step.
"""
import json
import pathlib

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
ITEMS = "https://raw.githubusercontent.com/anthropics/jacobian-lens/main/data/evaluations/lens-eval-multihop.json"
KS = [1, 10, 100]

app = modal.App("jlens-inkling-lenseval")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer", "datasets")
    .pip_install("git+https://github.com/huggingface/transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3")
    .pip_install("git+https://github.com/anthropics/jacobian-lens")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("fit")  # the 49-step adapter
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=20 * 60)
def run():
    import time, urllib.request
    import torch, transformers, jlens
    from fit import InklingLens

    t0 = time.time()
    items = json.loads(urllib.request.urlopen(ITEMS, timeout=60).read())["items"]
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    cfg = hf.model.language_model.config

    def token_ids(words):  # single-token spellings only
        variants = {f"{sp}{v}" for w in words for sp in ("", " ") for v in (w, w.lower(), w.capitalize())}
        return [i[0] for v in variants if len(i := tok.encode(v, add_special_tokens=False)) == 1]

    out = {"ks": KS, "by_passes": {}}
    for k in (1, 2, 3):
        cfg.loop_num_passes = k
        model = InklingLens(hf, tok)
        lens = jlens.JacobianLens.load(f"/cache/Inkling-2-Mini-p{k}_jacobian_lens.pt")
        rows = []
        for j, it in enumerate(items):
            own = token_ids(it["intermediates"])
            other = [t for t in token_ids(items[(j + 1) % len(items)]["intermediates"]) if t not in own]
            if not own or not other:
                continue
            prompt = it["prompt"].rstrip()
            row = {"name": it["name"]}
            for label, use_j in (("jacobian", True), ("logit", False)):
                logits, _, _ = lens.apply(model, prompt, positions=[-1], use_jacobian=use_j)
                stack = torch.stack([logits[l][0] for l in lens.source_layers])  # [steps, vocab]
                best = lambda ids: int(min((stack > stack[:, [c]]).sum(-1).min() for c in ids))  # best rank over steps and spellings
                row[label] = {"intermediate": best(own), "control": best(other)}
            rows.append(row)
        table = {label: {who: {f"pass@{n}": sum(r[label][who] < n for r in rows) / len(rows) for n in KS}
                         for who in ("intermediate", "control")} for label in ("jacobian", "logit")}
        out["by_passes"][k] = {"n": len(rows), "table": table, "rows": rows}
        print(f"\n=== {k} pass(es), n={len(rows)}")
        for label in table:
            for who in table[label]:
                print(f"  {label:<9} {who:<13} " + "  ".join(f"{m} {v:.2f}" for m, v in table[label][who].items()), flush=True)
        model.close()
    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main():
    path = pathlib.Path(__file__).parent / "results" / "lens-eval-multihop.json"
    path.write_text(json.dumps(run.remote(), indent=1))
    print("wrote", path)
