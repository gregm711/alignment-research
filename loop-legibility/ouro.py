"""Ouro-2.6B-Thinking: does the model write less when it loops more? Same fact sheets at each loop count.

    modal run ouro.py --pilot      # 12 two-lookup sheets at 2 and 4 loops; prints samples to check the scorer
    modal run --detach ouro.py     # full run; results on the volume as it goes
    modal volume get loop-legibility ouro.json results/ouro.json

Loop count is set the way the model card documents (config.total_ut_steps). Greedy decoding, thinking on, 20 prompts per batch (left-padded).
"""
import json

import modal

from tasks import items, score_text

MODEL = "ByteDance/Ouro-2.6B-Thinking"
LOOPS, CHAINS, SHEETS, BATCH = [2, 4], [2, 3], 40, 20

app = modal.App("loop-legibility-ouro")
vol = modal.Volume.from_name("loop-legibility", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.57.1", "accelerate", "safetensors", "hf_transfer")  # the card says 4.54.1; the repo's cache fix needs 4.56 or later
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("tasks")
)


@app.function(image=image, gpu=["A10G", "L4"], cpu=4, memory=32768, volumes={"/cache": vol}, timeout=70 * 60)
def run(pilot: bool = False, minutes: float = 50.0):
    import time
    import torch, transformers

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    think_end = tok.convert_tokens_to_ids("</think>")
    chat = lambda it: tok.apply_chat_template([{"role": "user", "content": it["chat"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def generate(batch, k, max_new):
        model.config.total_ut_steps = model.model.total_ut_steps = k
        enc = tok([chat(it) for it in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)[:, enc.input_ids.shape[1]:].tolist()
        res = []
        for new in out:
            cut = next((i for i, t in enumerate(new) if t in stop), None)
            res.append((new if cut is None else new[:cut], cut is None))
        return res

    import os
    path = "/cache/ouro-pilot.json" if pilot else "/cache/ouro.json"
    rows = json.load(open(path))["rows"] if os.path.exists(path) and not pilot else []  # resume: finished cells are kept
    for chain in ([2] if pilot else CHAINS):
        todo, max_new = items(5, 12 if pilot else SHEETS, chain), 400 if chain == 2 else 800
        for k in LOOPS:
            if sum((x["chain"], x["loops"]) == (chain, k) for x in rows) >= len(todo):
                continue
            rows = [x for x in rows if (x["chain"], x["loops"]) != (chain, k)]
            size = 8 if (chain, k) == (3, 4) else BATCH  # 20 prompts of 800 tokens at 4 loops do not fit in 24 GB
            torch.cuda.empty_cache()
            solo = generate(todo[:1], k, max_new)[0][0]  # one prompt alone, to compare with the same prompt inside a batch
            for start in range(0, len(todo), size):
                if time.time() - t0 > minutes * 60:
                    print("time guard: stopping", flush=True)
                    break
                batch = todo[start:start + size]
                for it, (new, hit) in zip(batch, generate(batch, k, max_new)):
                    text = tok.decode(new, skip_special_tokens=False)
                    rows.append({"chain": chain, "sheet": it["sheet"], "loops": k, "new_tokens": len(new), "hit_limit": hit,
                                 "thinking_tokens": new.index(think_end) if think_end in new else len(new), "steps": it["steps"], "text": text,
                                 **score_text(it, text, end="</think>")})
                    if it["sheet"] == 0:
                        print(f"chain {chain} loops {k}: batched output identical to the single-prompt output: {new == solo} "
                              f"({len(new)} and {len(solo)} tokens)\n{text}\n", flush=True)
                r = [x for x in rows if (x["chain"], x["loops"]) == (chain, k)]
                print(f"chain {chain} loops {k}: {len(r)} sheets, correct {sum(x['correct'] for x in r)}, finished {sum(x['finished'] for x in r)}, hit limit {sum(x['hit_limit'] for x in r)}, "
                      f"mean thinking tokens {sum(x['thinking_tokens'] for x in r) / len(r):.0f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                json.dump({"rows": rows, "gpu_minutes": (time.time() - t0) / 60}, open(path, "w"))
                vol.commit()
    print(f"total {(time.time() - t0) / 60:.1f} GPU minutes", flush=True)
    return {"rows": rows, "gpu_minutes": (time.time() - t0) / 60}


@app.local_entrypoint()
def main(pilot: bool = False):
    out = run.remote(pilot)
    print(f"{out['gpu_minutes']:.1f} GPU minutes")
