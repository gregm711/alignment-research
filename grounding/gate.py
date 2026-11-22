"""Phase 0, the gate: how often does a model answer from memory when a passage is supplied?

    modal run gate.py --n 20      # pilot: read the raw outputs
    modal run gate.py             # 300 questions, two models; writes results/gate.json

No internals are read or modified. See README.md for definitions.
"""
import json
import pathlib
import re
import string

import modal

MODELS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]
WITH_PASSAGE = "Passage: {passage}\n\nQuestion: {question}?\nAnswer with just the answer, using the passage."
CLOSED_BOOK = "Question: {question}?\nAnswer with just the answer."

app = modal.App("grounding-gate")
vol = modal.Volume.from_name("grounding-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "safetensors", "datasets", "pillow", "hf_transfer", "sentencepiece")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def norm(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def has(answer_text, golds):
    a = f" {norm(answer_text)} "
    return any(f" {norm(g)} " in a for g in golds if norm(g))


def clean(passage):
    return re.sub(r"\s+", " ", re.sub(r"</?[A-Za-z]+>", " ", passage)).strip()


def sample(n, seed=0):
    import random
    from datasets import load_dataset

    seen, items = set(), []
    for r in load_dataset("pminervini/NQ-Swap", split="dev"):
        q = r["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        if max(len(a.split()) for a in r["org_answer"]) > 5 or len(r["org_context"].split()) > 250:
            continue
        if has(" ".join(r["sub_answer"]), r["org_answer"]) or has(" ".join(r["org_answer"]), r["sub_answer"]):
            continue  # the two answers must be distinguishable by string
        items.append({"question": q, "original": clean(r["org_context"]), "substituted": clean(r["sub_context"]),
                      "original_answer": r["org_answer"], "substituted_answer": r["sub_answer"]})
    random.Random(seed).shuffle(items)
    return items[:n], len(items)


@app.function(image=image, gpu="L4", cpu=4, memory=24576, volumes={"/cache": vol}, timeout=40 * 60)
def run(n: int):
    import gc, time
    import torch, transformers

    t0 = time.time()
    items, pool = sample(n)
    out = {"n": len(items), "pool_of_eligible_questions": pool, "transformers": transformers.__version__, "models": {}}

    def one_model(name):
        tok = transformers.AutoTokenizer.from_pretrained(name)
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        except Exception:
            model = transformers.AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()

        def ask(text):
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False,
                                          return_tensors="pt", return_dict=True)["input_ids"].cuda()
            with torch.no_grad():
                gen = model.generate(input_ids=ids, max_new_tokens=24, do_sample=False)
            return tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

        rows = []
        for it in items:
            closed = ask(CLOSED_BOOK.format(question=it["question"]))
            orig = ask(WITH_PASSAGE.format(passage=it["original"], question=it["question"]))
            sub = ask(WITH_PASSAGE.format(passage=it["substituted"], question=it["question"]))
            mem, ctx = has(sub, it["original_answer"]), has(sub, it["substituted_answer"])
            rows.append({"question": it["question"], "original_answer": it["original_answer"], "substituted_answer": it["substituted_answer"],
                         "closed_book": closed, "with_original": orig, "with_substituted": sub,
                         "knows_closed_book": has(closed, it["original_answer"]),
                         "correct_with_original": has(orig, it["original_answer"]),
                         "substituted": "both" if mem and ctx else "memory" if mem else "passage" if ctx else "other"})
        correct = [r for r in rows if r["correct_with_original"]]
        s = {"knows_closed_book": sum(r["knows_closed_book"] for r in rows),
             "correct_with_original": len(correct),
             "with_substituted": {k: sum(r["substituted"] == k for r in rows) for k in ("passage", "memory", "both", "other")},
             "hidden_ungrounded": sum(r["substituted"] == "memory" for r in correct),
             "grounded": sum(r["substituted"] == "passage" for r in correct),
             "hidden_ungrounded_among_known": f"{sum(r['substituted'] == 'memory' for r in correct if r['knows_closed_book'])}/{sum(r['knows_closed_book'] for r in correct)}"}
        print(f"\n=== {name} | n={len(rows)} | {s}", flush=True)
        for r in rows[:8]:
            print(f"   Q: {r['question'][:60]} | gold {r['original_answer'][0]!r} sub {r['substituted_answer'][0]!r} | closed {r['closed_book'][:30]!r} "
                  f"| orig {r['with_original'][:30]!r} | sub {r['with_substituted'][:30]!r} -> {r['substituted']}")
        return {"summary": s, "rows": rows}

    for name in MODELS:
        out["models"][name] = one_model(name)
        gc.collect(); torch.cuda.empty_cache()
    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"eligible questions in the dataset: {pool} | total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(n: int = 300):
    out = run.remote(n)
    name = "gate.json" if n >= 300 else "gate-pilot.json"
    (pathlib.Path(__file__).parent / "results" / name).write_text(json.dumps(out, indent=1))
    print("wrote", name)
