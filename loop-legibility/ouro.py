"""Ouro-2.6B-Thinking: does the model write less when it loops more? Same fact sheets at each loop count.

    modal run ouro.py --pilot      # 12 two-lookup sheets at 2 and 4 loops; prints samples to check the scorer
    modal run --detach ouro.py     # full run; results on the volume as it goes
    modal volume get loop-legibility ouro.json results/ouro.json

Loop count is set the way the model card documents (config.total_ut_steps). Greedy decoding, thinking on.
"""
import json

import modal

from tasks import items, score_text

MODEL = "ByteDance/Ouro-2.6B-Thinking"
LOOPS, CHAINS, SHEETS, MAX_NEW = [2, 4], [2, 3], 40, 512

app = modal.App("loop-legibility-ouro")
vol = modal.Volume.from_name("loop-legibility", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==4.57.1", "accelerate", "safetensors", "hf_transfer")  # the card says 4.54.1; the repo's cache fix needs 4.56 or later
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("tasks")
)


@app.function(image=image, gpu=["L4", "A10G", "T4"], cpu=4, memory=32768, volumes={"/cache": vol}, timeout=70 * 60)
def run(pilot: bool = False, minutes: float = 55.0):
    import time
    import torch, transformers

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vol.commit()
    rows, n_tok, gen_s = [], 0, 0.0
    for chain in ([2] if pilot else CHAINS):
        for it in items(5, 12 if pilot else SHEETS, chain):
            if time.time() - t0 > minutes * 60:
                break  # whole sheets only, so every sheet has every loop count
            ids = tok.apply_chat_template([{"role": "user", "content": it["chat"]}], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt").cuda()
            for k in LOOPS:
                model.config.total_ut_steps = model.model.total_ut_steps = k
                t1 = time.time()
                with torch.no_grad():
                    new = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)[0][ids.shape[1]:]
                gen_s, n_tok = gen_s + time.time() - t1, n_tok + len(new)
                text = tok.decode(new, skip_special_tokens=False)
                end = (new == tok.convert_tokens_to_ids("</think>")).nonzero()
                rows.append({"chain": chain, "sheet": it["sheet"], "loops": k, "new_tokens": len(new), "hit_limit": len(new) >= MAX_NEW,
                             "thinking_tokens": int(end[0]) if len(end) else len(new), "steps": it["steps"], "text": text, **score_text(it, text, end="</think>")})
                r = rows[-1]
                print(f"chain {chain} sheet {it['sheet']} loops {k}: {r['thinking_tokens']} thinking tokens, finished {r['finished']}, correct {r['correct']}, named {r['named']}", flush=True)
                if pilot and it["sheet"] < 3:
                    print(f"steps {it['steps']}\n{text}\n", flush=True)
            json.dump({"rows": rows, "gpu_minutes": (time.time() - t0) / 60}, open("/cache/ouro-pilot.json" if pilot else "/cache/ouro.json", "w"))
            vol.commit()
    print(f"{n_tok / max(gen_s, 1e-9):.1f} tokens per second; total {(time.time() - t0) / 60:.1f} GPU minutes", flush=True)
    return {"rows": rows, "gpu_minutes": (time.time() - t0) / 60}


@app.local_entrypoint()
def main(pilot: bool = False):
    out = run.remote(pilot)
    print(f"{out['gpu_minutes']:.1f} GPU minutes")
