"""Pool public Llama 3.3 70B layer-53 activations on Liars' Bench into four vectors per transcript (Modal, CPU only).

    modal run pool.py        # writes pooled-{subset}.npz to the Modal volume "inside-advantage"
    modal volume get inside-advantage / pooled/     # then fetch them (about 0.3 GB)
"""
import modal

REPO = "Yooniel/liars-bench-llama-3.3-70b-L53-acts"
SUBSETS = ["harm-pressure-choice", "harm-pressure-knowledge-report", "instructed-deception", "convincing-game",
           "gender-secret", "insider-trading", "soft-trigger", "alpaca"]
app = modal.App("inside-advantage-pool")
vol = modal.Volume.from_name("inside-advantage", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "safetensors", "huggingface_hub", "hf_transfer", "numpy").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})


@app.function(image=image, cpu=4, memory=24576, volumes={"/out": vol}, timeout=60 * 60, ephemeral_disk=60 * 1024)
def pool(subset: str):
    import json, os, time
    import numpy as np, torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    t0 = time.time()
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, f"{subset}/metadata.jsonl", repo_type="dataset"))]
    out = np.zeros((len(rows), 4, 8192), dtype=np.float16)
    for chunk in sorted({r["chunk"] for r in rows}):
        path = hf_hub_download(REPO, f"{subset}/{chunk}", repo_type="dataset")
        acts = load_file(path)["acts"]
        for i, r in enumerate(rows):
            if r["chunk"] != chunk:
                continue
            a, s = acts[r["start"]:r["end"]].float(), r["assistant_start"]
            assert len(a) == r["n_tokens"] and 0 < s < len(a), r
            out[i] = torch.stack([a[:s].mean(0), a[s - 1], a[s:].mean(0), a[-1]]).numpy()
        del acts
        os.remove(os.path.realpath(path))
        print(subset, chunk, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez(f"/out/pooled-{subset}.npz", x=out, row_idx=np.array([r["row_idx"] for r in rows]), deceptive=np.array([r["deceptive"] for r in rows]),
             model=np.array([r["model"] for r in rows]))
    vol.commit()
    return subset, len(rows), (time.time() - t0) / 60


@app.local_entrypoint()
def main():
    for subset, n, minutes in pool.map(SUBSETS):
        print(f"{subset}: {n} transcripts, {minutes:.1f} CPU-machine minutes")
