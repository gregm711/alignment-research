"""Pilot: can the variable-loop Inkling checkpoint do chained lookups, and what does it write at each pass count?

    python3 pilot.py --show          # print two prompts, no GPU
    modal run pilot.py               # writes results/pilot.json (our own invented tables; raw outputs kept)

No probe here. See README.md, "Plan and cost".
"""
import json
import pathlib
import random
import sys

MODEL = "thinkingmachines/TML-d24-varloop-research"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
PASSES, EFFORTS, CHAINS, PER_CHAIN, MAX_NEW = [1, 2, 3], [0.0, 0.9], [2, 3, 4], 8, 200
KINDS = [("animal", ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]),
         ("color", ["red", "blue", "green", "gold", "pink", "gray", "black", "white", "brown", "purple", "silver", "yellow"]),
         ("fruit", ["fig", "lime", "grape", "apple", "orange", "date", "olive", "pear", "plum", "peach"]),
         ("tool", ["hammer", "saw", "drill", "rake", "spade", "wrench", "chisel", "ladder", "broom"]),
         ("city", ["Paris", "Lima", "Oslo", "Cairo", "Rome", "Delhi", "Tokyo", "Quito", "Seoul"])]
ROWS = 7


def items(seed, n_sheets, chain):
    """One fact sheet per item. `steps` are the values after each lookup; the last is the answer."""
    rng, out = random.Random(seed * 100 + chain), []
    for s in range(n_sheets):
        vals = [rng.sample(words, ROWS) for _, words in KINDS[: chain + 1]]
        tables = [dict(zip(vals[i], rng.sample(vals[i + 1], ROWS))) for i in range(chain)]  # one-to-one, so no step is ambiguous
        lines = []
        for i, t in enumerate(tables):
            a, b = KINDS[i][0], KINDS[i + 1][0]
            rows = ", ".join(f"the {k} goes with the {t[k]}" for k in rng.sample(list(t), ROWS))
            lines.append(f"{b.capitalize()}s for {a}s: {rows}.")
        start = rng.choice(vals[0])
        steps = [start]
        for t in tables:
            steps.append(t[steps[-1]])
        prompt = "Fact sheet.\n" + "\n".join(lines) + f"\nStart from the {start} and follow the tables in order. What is the final {KINDS[chain][0]}?"
        out.append({"sheet": s, "chain": chain, "prompt": prompt, "intermediates": steps[1:-1], "answer": steps[-1],
                    "answer_candidates": vals[chain]})
    return out


def score(it, text):
    """Final answer = the last candidate of the answer kind that the output mentions."""
    low = text.lower()
    seen = [(low.rfind(w.lower()), w) for w in it["answer_candidates"] if w.lower() in low]
    return {"correct": bool(seen) and max(seen)[1] == it["answer"], "named": [w.lower() in low for w in it["intermediates"]]}


if "--show" in sys.argv:
    for c in (2, 4):
        it = items(0, 1, c)[0]
        print(it["prompt"], "\n ->", it["intermediates"], it["answer"], "\n")
    it = items(0, 1, 3)[0]
    assert score(it, f"first {it['intermediates'][0]}, so the answer is {it['answer']}.") == {"correct": True, "named": [True, False]}
    assert not score(it, "I do not know")["correct"]
    sys.exit()

import modal

app = modal.App("loop-legibility-pilot")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)  # the weights are already cached here
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=30 * 60)
def run():
    import time
    import torch, transformers

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = hf.model.language_model.config
    rows = []
    for chain in CHAINS:
        for it in items(0, PER_CHAIN, chain):
            for effort in EFFORTS:
                chat = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
                ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                for k in PASSES:
                    if time.time() - t0 > 26 * 60:  # stop cleanly inside the timeout and keep what we have
                        return {"rows": rows, "gpu_minutes": (time.time() - t0) / 60, "stopped_early": True}
                    cfg.loop_num_passes = k
                    with torch.no_grad():
                        new = hf.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False)[0][ids.shape[1]:]
                    text = tok.decode(new, skip_special_tokens=False)
                    rows.append({"sheet": it["sheet"], "chain": chain, "effort": effort, "passes": k, "new_tokens": len(new),
                                 "hit_limit": len(new) >= MAX_NEW, "text": text, **score(it, tok.decode(new, skip_special_tokens=True))})
                    if it["sheet"] == 0:
                        print(f"--- chain {chain} effort {effort} passes {k} | {rows[-1]['correct']} {rows[-1]['named']} | {len(new)} tokens\n{text}", flush=True)
        for effort in EFFORTS:
            for k in PASSES:
                r = [x for x in rows if (x["chain"], x["effort"], x["passes"]) == (chain, effort, k)]
                print(f"chain {chain} effort {effort} passes {k}: correct {sum(x['correct'] for x in r)}/{len(r)}, mean tokens {sum(x['new_tokens'] for x in r) / len(r):.0f}, "
                      f"steps named {sum(sum(x['named']) for x in r)}/{sum(len(x['named']) for x in r)}, hit limit {sum(x['hit_limit'] for x in r)}", flush=True)
    return {"rows": rows, "gpu_minutes": (time.time() - t0) / 60}


@app.local_entrypoint()
def main():
    out = run.remote()
    (pathlib.Path(__file__).parent / "results" / "pilot.json").write_text(json.dumps(out, indent=1))
    print(f"{out['gpu_minutes']:.1f} GPU minutes")
