"""Question 2: at which pass is the answer decided? Forward-only sweep over the number of loop passes.

    modal run sweep.py        # writes results/pass-sweep.json
"""
import json
import pathlib

import modal

MODELS = ["thinkingmachines/Inkling-2-Mini", "thinkingmachines/TML-d24-varloop-research"]
PASSES = [1, 2, 3, 4, 5]
REF = 3  # the pass count both checkpoints were post-trained at

app = modal.App("jlens-inkling-sweep")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer", "datasets")
    .pip_install("git+https://github.com/huggingface/transformers@add-inkling2-mini")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

ONE_HOP = [
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"), ("The capital of Egypt is", "Cairo"),
    ("The capital of England is", "London"), ("The capital of Russia is", "Moscow"),
    ("The capital of Canada is", "Ottawa"), ("The capital of China is", "Beijing"),
    ("The currency used in Italy is the", "euro"), ("The currency used in Japan is the", "yen"),
    ("The language spoken in Japan is", "Japanese"), ("The language spoken in Portugal is", "Portuguese"),
    ("The language spoken in Russia is", "Russian"), ("The language spoken in France is", "French"),
    ("The language spoken in Germany is", "German"), ("The colour you get by mixing blue with yellow is", "green"),
]
# Same answers as ONE_HOP, in the same order, reached through one extra step.
TWO_HOP = [
    ("The capital of the country where the Eiffel Tower stands is", "Paris"),
    ("The capital of the country famous for sushi is", "Tokyo"),
    ("The capital of the country shaped like a boot is", "Rome"),
    ("The capital of the country where the pyramids of Giza stand is", "Cairo"),
    ("The capital of the country where Big Ben stands is", "London"),
    ("The capital of the country where the Kremlin stands is", "Moscow"),
    ("The capital of the country whose flag is a red maple leaf is", "Ottawa"),
    ("The capital of the country that hosted the 2008 Summer Olympics is", "Beijing"),
    ("The currency used in the country shaped like a boot is the", "euro"),
    ("The currency used in the country famous for sushi is the", "yen"),
    ("The language spoken in the country whose capital is Tokyo is", "Japanese"),
    ("The language spoken in the country whose capital is Lisbon is", "Portuguese"),
    ("The language spoken in the country whose capital is Moscow is", "Russian"),
    ("The language spoken in the country where the Eiffel Tower stands is", "French"),
    ("The language spoken in the country whose capital is Berlin is", "German"),
    ("The colour you get by mixing the colour of the sky with the colour of a lemon is", "green"),
]
ARITHMETIC = [("7 + 5 =", "12"), ("9 + 8 =", "17"), ("6 * 7 =", "42"), ("8 * 9 =", "72"), ("15 + 27 =", "42"),
              ("47 + 38 =", "85"), ("63 - 28 =", "35"), ("12 * 12 =", "144"), ("100 - 37 =", "63"),
              ("25 * 4 =", "100"), ("81 / 9 =", "9"), ("144 / 12 =", "12")]
TIERS = {"one_hop": ONE_HOP, "two_hop": TWO_HOP, "arithmetic": ARITHMETIC}


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=40 * 60)
def run(with_text: bool = True):
    import gc, time
    import torch, transformers
    from datasets import load_dataset

    t0 = time.time()
    text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    paragraphs = [t.strip() for t in text if len(t) > 900]
    out = {"passes": PASSES, "ref": REF, "models": {}}

    for name in MODELS:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        hf = transformers.AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()
        cfg = hf.model.language_model.config

        def logits_at(ids, k):
            cfg.loop_num_passes = k
            with torch.no_grad():
                return hf(input_ids=ids, use_cache=False).logits.float()

        # Ordinary text: 60 paragraphs x 128 tokens, positions 16.. scored against the true next token.
        rows = [tok(p, add_special_tokens=False).input_ids[:128] for p in paragraphs]
        ids = torch.tensor([r for r in rows if len(r) == 128][:60], device="cuda")
        text_stats, settled, n_tok = {}, {}, 0
        if with_text:
            # One batch at a time: full-vocabulary scores for every pass count do not fit on the GPU together.
            sums = {k: {"accuracy": 0.0, "log_loss": 0.0, "agrees_with_ref_top1": 0.0, "kl_from_ref": 0.0} for k in PASSES}
            settled_sum, n_tok = {k: 0.0 for k in PASSES if k <= REF}, 0
            for i in range(0, len(ids), 12):
                batch = ids[i:i + 12]
                target = batch[:, 17:]
                lp = {k: torch.log_softmax(logits_at(batch, k)[:, 16:-1], -1) for k in PASSES}
                top = {k: v.argmax(-1) for k, v in lp.items()}
                for k in PASSES:
                    sums[k]["accuracy"] += (top[k] == target).sum().item()
                    sums[k]["log_loss"] += -lp[k].gather(-1, target[..., None]).sum().item()
                    sums[k]["agrees_with_ref_top1"] += (top[k] == top[REF]).sum().item()
                    sums[k]["kl_from_ref"] += (lp[REF].exp() * (lp[REF] - lp[k])).sum().item()
                for k in settled_sum:  # top prediction already final: equal to the ref from pass k through the ref
                    settled_sum[k] += torch.stack([top[j] == top[REF] for j in PASSES if k <= j <= REF]).all(0).sum().item()
                n_tok += target.numel()
                del lp, top
            text_stats = {k: {m: v / n_tok for m, v in d.items()} for k, d in sums.items()}
            settled = {k: v / n_tok for k, v in settled_sum.items()}

        # Probes: is the answer the top prediction? This tokenizer writes some words and all numbers as a
        # bare space token followed by the word (" Cairo" -> "Ġ", "Cairo"), so for those the space is
        # appended to the prompt and the bare word is scored at the next position.
        def single(variants):
            return [i[0] for v in variants if len(i := tok.encode(v, add_special_tokens=False)) == 1]

        probe_stats, probe_rows = {}, []
        for tier, probes in TIERS.items():
            hits = {k: [] for k in PASSES}
            for prompt, answer in probes:
                forms = (answer, answer.lower(), answer.capitalize())
                cands, text_in = single({" " + w for w in forms}), "Fact: " + prompt
                if not cands:
                    cands, text_in = single(set(forms)), text_in + " "
                if not cands:
                    continue
                pid = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                row = {"tier": tier, "prompt": prompt, "answer": answer, "by_pass": {}}
                for k in PASSES:
                    last = logits_at(pid, k)[0, -1]
                    order = last.argsort(descending=True)
                    rank = min(int((order == c).nonzero()[0, 0]) for c in cands)
                    hits[k].append(rank == 0)
                    row["by_pass"][k] = {"rank": rank, "top3": [tok.decode([t]) for t in order[:3].tolist()]}
                probe_rows.append(row)
            probe_stats[tier] = {"n": len(hits[PASSES[0]]), "top1_rate": {k: sum(v) / max(len(v), 1) for k, v in hits.items()}}

        out["models"][name] = {"n_text_tokens": n_tok, "text": text_stats, "settled_by_pass": settled,
                               "probes": probe_stats, "probe_rows": probe_rows}
        print(f"\n=== {name} | {n_tok} text tokens")
        print("passes   acc    logloss  agree@ref  KL(ref||k) | " + "  ".join(f"{t}(n={probe_stats[t]['n']})" for t in TIERS))
        for k in PASSES:
            s = text_stats.get(k) or dict.fromkeys(("accuracy", "log_loss", "agrees_with_ref_top1", "kl_from_ref"), float("nan"))
            print(f"  {k}    {s['accuracy']:.3f}   {s['log_loss']:.3f}    {s['agrees_with_ref_top1']:.3f}     {s['kl_from_ref']:.3f}    | "
                  + "      ".join(f"{probe_stats[t]['top1_rate'][k]:.2f}" for t in TIERS), flush=True)
        print("settled by pass:", {k: round(v, 3) for k, v in settled.items()}, flush=True)
        del hf
        gc.collect(); torch.cuda.empty_cache()

    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"\ntotal {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(text: bool = True, out_name: str = "pass-sweep.json"):
    out = run.remote(text)
    path = pathlib.Path(__file__).parent / "results" / out_name
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
