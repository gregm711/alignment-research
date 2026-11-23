"""Main run on Inkling: chained lookups at 1, 2 and 3 passes.

    python3 main.py --show      # print a prompt, check the scorer; no GPU
    modal run --detach main.py  # part A then part B; results on the volume as it goes
    modal volume get jlens-inkling-cache loop-legibility/ results/vol/   # then: analyse.py

Part A (forward only): the model must answer in one token, so no step is written. Saves the answer and the
last-token state at the end of each pass, for probes fitted later on a laptop.
Part B (generation): thinking on, two-lookup chains, enough tokens to finish. Reasoning length and written steps.
"""
import json
import sys

VARLOOP, FIXED = "thinkingmachines/TML-d24-varloop-research", "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
CHAINS, SHEETS_A, SHEETS_B, MAX_NEW = [2, 3, 4], 250, 16, 320
from tasks import KINDS, ROWS, WORDINGS, items, score_text

if "--show" in sys.argv:
    it = items(0, 1, 3)[0]
    print(it["direct"], "\n ->", it["steps"], "| decoy", it["decoy_steps"])
    a, b, c = it["steps"][1:]
    copied = " ".join(it["pairs"])
    assert score_text(it, f"{copied} so {it['steps'][0]} -> {a}, then next.<|end_message|><|content_text|>The final tool is **{c}**.") == {"finished": True, "correct": True, "named": [True, False]}
    assert score_text(it, f"{copied} thinking")["named"] == [False, False] and not score_text(it, f"{copied} thinking")["finished"]
    sys.exit()

import modal

app = modal.App("loop-legibility-main")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)  # the weights are already cached here
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("tasks")
)
OUT = "/cache/loop-legibility"


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=58 * 60)
def run(part_b_minutes: float = 28.0):
    import gc, os, time
    import torch, transformers

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(VARLOOP)
    first = lambda w: tok.encode(" " + w, add_special_tokens=False)[0]
    kinds = []
    for kind, words in KINDS:  # keep words whose first token is theirs alone, so one token identifies the answer
        firsts = [first(w) for w in words]
        kinds.append((kind, [w for w, f in zip(words, firsts) if firsts.count(f) == 1]))
        assert len(kinds[-1][1]) >= ROWS, kind
    data = {c: items(1, SHEETS_A, c, kinds) for c in CHAINS}

    def load(name):
        hf = transformers.AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        text, states, msgs = hf.model.language_model, [], []
        grab = lambda mod, args, output: states.append((output if torch.is_tensor(output) else output[0])[0, -1].float().cpu())
        for layer in text.layers:
            layer.register_forward_hook(grab)
        text.message_input.register_forward_hook(lambda mod, args, output: msgs.append((output if torch.is_tensor(output) else output[0])[0, -1].float().cpu()))
        return hf, text, states, msgs

    # ---- Part A: one-token answers, states saved ----
    rows, saved = [], {}
    for name, passes in ((FIXED, [3]), (VARLOOP, [1, 2, 3])):  # the variable-loop model last: part B reuses it
        hf, text, states, msgs = load(name)
        vol.commit()
        for k in passes:
            text.config.loop_num_passes = k
            S, M = [], []
            for c in CHAINS:
                for it in data[c]:
                    ids = tok(it["direct"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                    states.clear(); msgs.clear()
                    with torch.no_grad():
                        logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
                    assert len(states) == 12 + 12 * k, len(states)
                    # sites: before the loop, end of each pass, last layer
                    S.append(torch.stack([states[5]] + [states[5 + 12 * j] for j in range(1, k + 1)] + [states[-1]]).half())
                    if msgs:
                        M.append(torch.stack(msgs).half())
                    cand = [first(w) for w in it["candidates"]]
                    rows.append({"model": name.split("/")[-1], "passes": k, "chain": c, "sheet": it["sheet"], "steps": it["steps"], "decoy_steps": it["decoy_steps"],
                                 "top1_right": int(logits.argmax()) == first(it["steps"][-1]), "pick": it["candidates"][int(logits[cand].argmax())]})
                r = [x for x in rows if (x["model"], x["passes"], x["chain"]) == (name.split("/")[-1], k, c)]
                print(f"A {name.split('/')[-1]} passes {k} chain {c}: top-1 right {sum(x['top1_right'] for x in r)}/{len(r)} | {(time.time() - t0) / 60:.1f} min", flush=True)
            saved[f"{name.split('/')[-1]}|{k}"] = {"states": torch.stack(S), "messages": torch.stack(M) if M else None}
            if M:
                print("messages per run:", tuple(M[0].shape), flush=True)
        if name == FIXED:
            del hf, text, states, msgs
            gc.collect(); torch.cuda.empty_cache()
    torch.save(saved, f"{OUT}/states.pt")
    json.dump({"rows": rows, "gpu_minutes": (time.time() - t0) / 60}, open(f"{OUT}/partA.json", "w"))
    vol.commit()
    print(f"part A saved at {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- Part B: thinking on, two-lookup chains ----
    tb, gens = time.time(), []
    for it in items(2, SHEETS_B, 2, kinds):
        chat = tok.apply_chat_template([{"role": "user", "content": it["chat"]}], tokenize=False, add_generation_prompt=True, reasoning_effort=0.0)
        ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        if time.time() - tb > part_b_minutes * 60:
            break  # whole items only, so every item has all three pass counts
        for k in (1, 2, 3):
            text.config.loop_num_passes = k
            with torch.no_grad():
                new = hf.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False)[0][ids.shape[1]:]
            out = tok.decode(new, skip_special_tokens=False)
            end = (new == tok.convert_tokens_to_ids("<|end_message|>")).nonzero()
            gens.append({"sheet": it["sheet"], "passes": k, "new_tokens": len(new), "thinking_tokens": int(end[0]) if len(end) else len(new),
                         "steps": it["steps"], "text": out, **score_text(it, out)})
            print(f"B sheet {it['sheet']} passes {k}: {gens[-1]['thinking_tokens']} thinking tokens, finished {gens[-1]['finished']}, correct {gens[-1]['correct']}, named {gens[-1]['named']}", flush=True)
        json.dump({"rows": gens, "gpu_minutes": (time.time() - tb) / 60}, open(f"{OUT}/partB.json", "w"))
        vol.commit()
    print(f"total {(time.time() - t0) / 60:.1f} GPU minutes", flush=True)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=8 * 60)
def peek():
    """Which wording makes the one-token answer the answer? 12 sheets per chain, both wordings, 1 and 3 passes."""
    import torch, transformers
    tok = transformers.AutoTokenizer.from_pretrained(VARLOOP)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(VARLOOP, dtype=torch.bfloat16, device_map="cuda").eval()
    for w in WORDINGS:
        for k in (1, 3):
            hf.model.language_model.config.loop_num_passes = k
            for c in CHAINS:
                right, tops = 0, []
                for it in items(1, 12, c, wording=w):
                    ids = tok(it["direct"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                    with torch.no_grad():
                        top = int(hf(input_ids=ids, use_cache=False).logits[0, -1].argmax())
                    right += top == tok.encode(" " + it["steps"][-1], add_special_tokens=False)[0]
                    tops.append(tok.decode([top]))
                print(f"{w} passes {k} chain {c}: {right}/12 | {tops[:6]}", flush=True)


@app.local_entrypoint()
def main(part_b_minutes: float = 28.0, peek_only: bool = False):
    if peek_only:
        return peek.remote()
    run.remote(part_b_minutes)
