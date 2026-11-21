"""Question 4: must an edit be held across all passes to stick? Swap a concept direction inside a pass.

    modal run clamp.py        # writes results/direction-swap.json

Needs the per-pass lenses from `fit.py --passes 1,2,3` on the Modal volume.
"""
import itertools
import json
import pathlib

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
BAND = range(7, 12)  # the loop layers where the lens read concepts
CONDITIONS = {"pass1": {1}, "pass2": {2}, "pass3": {3}, "all": {1, 2, 3}}
STRENGTHS = [1, 4]

app = modal.App("jlens-inkling-clamp")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install("git+https://github.com/huggingface/transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("patch")  # reuse the capital / country / language items
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=30 * 60)
def run():
    import collections, time
    import torch, transformers
    from patch import CAPITALS

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    text = hf.model.language_model
    text.config.loop_num_passes = 3
    unembed = hf.lm_head.weight.float()

    def one_token(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    items = [{"city": c, "country": k, "lang": l, "country_id": one_token(k), "lang_id": one_token(l),
              "ids": tok(f"Fact: The language spoken in the country whose capital is {c} is",
                         return_tensors="pt", add_special_tokens=False).input_ids.cuda()} for c, k, l in CAPITALS]
    items = [it for it in items if it["country_id"] is not None and it["lang_id"] is not None]

    # Lens direction for each country at each (pass, layer): the country token's unembedding vector pulled
    # back through that pass's own lens. In a k-pass run, loop layer i of pass k is step 12 * (k - 1) + i.
    direction = {}
    for p in (1, 2, 3):
        J = torch.load(f"/cache/Inkling-2-Mini-p{p}_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"]
        for i in BAND:
            Jt = J[12 * (p - 1) + i].float().cuda().T  # [d_in, d_out]
            for it in items:
                d = Jt @ unembed[it["country_id"]]
                direction[p, i, it["country"]] = d / d.norm()

    # Hooks on the band layers: on the chosen passes, swap the state's coordinates along src and tgt.
    edit = {"passes": set(), "src": None, "tgt": None, "strength": 1.0}
    calls = collections.Counter()
    text.register_forward_pre_hook(lambda mod, args: calls.clear())

    def make_hook(i):
        def hook(mod, args, output):
            calls[i] += 1
            p = calls[i]
            if p not in edit["passes"]:
                return None
            h = output.float()
            ds, dt = direction[p, i, edit["src"]], direction[p, i, edit["tgt"]]
            a, b = h @ ds, h @ dt
            return (h + edit["strength"] * (a - b)[..., None] * (dt - ds)).to(output.dtype)
        return hook

    for i in BAND:
        text.layers[i].register_forward_hook(make_hook(i))

    def top(ids, passes=(), src=None, tgt=None, strength=1.0):
        edit.update(passes=set(passes), src=src, tgt=tgt, strength=float(strength))
        with torch.no_grad():
            return int(hf(input_ids=ids, use_cache=False).logits[0, -1].argmax())

    usable = [it for it in items if top(it["ids"]) == it["lang_id"]]
    counts = {f"{c}/x{s}": {"to_target": 0, "stays": 0, "other": 0} for c in CONDITIONS for s in STRENGTHS}
    control = {"stays": 0, "other": 0}
    examples, n_pairs = [], 0
    for A, B in itertools.permutations(usable, 2):
        n_pairs += 1
        row = {"A": A["city"], "target": B["lang"]}
        for c, passes in CONDITIONS.items():
            for s in STRENGTHS:
                t = top(A["ids"], passes, A["country"], B["country"], s)
                kind = "to_target" if t == B["lang_id"] else "stays" if t == A["lang_id"] else "other"
                counts[f"{c}/x{s}"][kind] += 1
                row[f"{c}/x{s}"] = tok.decode([t])
        # Control: the same edit along two countries that are neither A's nor the target's.
        others = [it for it in usable if it is not A and it is not B][:2]
        t = top(A["ids"], {1, 2, 3}, others[0]["country"], others[1]["country"], max(STRENGTHS))
        control["stays" if t == A["lang_id"] else "other"] += 1
        if len(examples) < 14:
            examples.append(row)

    print(f"{len(usable)}/{len(items)} prompts right unedited, {n_pairs} pairs")
    for k, c in counts.items():
        print(f"  {k:<10} to_target {c['to_target']:>3}  stays {c['stays']:>3}  other {c['other']:>3}", flush=True)
    print("  control (unrelated directions, all passes, strongest):", control)
    out = {"n_prompts": len(items), "n_correct_unedited": len(usable), "n_pairs": n_pairs, "counts": counts,
           "control": control, "examples": examples, "gpu_minutes": (time.time() - t0) / 60}
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main():
    out = run.remote()
    path = pathlib.Path(__file__).parent / "results" / "direction-swap.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
