"""Mistakes the model makes unprompted, third approach: answering from association instead of the table.

    modal run prior.py --pilot           # step 1: how often does association override or replace the evidence?
    modal run prior.py                   # step 2: fresh tables, frozen readout vs output confidence

Nothing is done to the model's internals. The readout (head 23 at L11 of the final pass, threshold 0.45)
is frozen from oversight2.py and never refitted.
"""
import json
import pathlib
import random

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
HEAD_CALL, HEAD, THRESHOLD = 35, 23, 0.45085132122039795
ANIMALS = ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]
ASSOCIATED = {"red": "apple", "green": "lime", "purple": "grape", "brown": "date", "black": "olive"}
NEUTRAL = ["blue", "gold", "pink", "gray", "white", "silver", "yellow"]
FRUITS = ["fig", "lime", "grape", "apple", "orange", "date", "olive"]

app = modal.App("jlens-inkling-prior")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def prompts(seed, n_tables):
    """Seven-row, two-step tables. Each table has three colours with an everyday fruit association; one
    question per such colour. `agree`: the table gives that colour its associated fruit. `conflict`: the
    table gives it another fruit and puts the associated fruit on a different row."""
    rng, out = random.Random(seed), []
    for t in range(n_tables):
        loaded, neutral = rng.sample(sorted(ASSOCIATED), 3), rng.sample(NEUTRAL, 4)
        colors, animals = loaded + neutral, rng.sample(ANIMALS, 7)
        t1 = dict(zip(animals, rng.sample(colors, 7)))
        order1, order2 = rng.sample(animals, 7), rng.sample(colors, 7)
        first = "Fact sheet.\nColors of animals: " + ", ".join(f"the {a} is {t1[a]}" for a in order1) + ".\n"
        for c in loaded:
            want = ASSOCIATED[c]
            rest = [f for f in FRUITS if f != want]
            rng.shuffle(rest)
            q = next(a for a in animals if t1[a] == c)
            tail = f"So the fruit that goes with the {q} is the"

            def render(table):
                return first + "Fruits for colors: " + ", ".join(f"{k} goes with the {table[k]}" for k in order2) + ".\n" + tail

            # agreement: this colour gets its associated fruit; the other six rows get five other fruits (one repeated), none equal to it
            agree = {c: want, **dict(zip([k for k in colors if k != c], (rest[:5] + rest[:1])))}
            spare = rest[5]
            out.append({"kind": "agree", "table": t, "color": c, "answer": want, "associated": want, "spare": spare,
                        "true_row": order2.index(c), "row_entries": [f"{k} goes with the {agree[k]}" for k in order2],
                        "prompt": render(agree), "edit_intended": render({**agree, c: spare})})
            # conflict: this colour gets a different fruit, and the associated fruit sits on a neutral colour's row
            lure_color = neutral[0]
            others = [k for k in colors if k not in (c, lure_color)]
            conflict = {c: rest[0], lure_color: want, **dict(zip(others, rest[1:6]))}
            out.append({"kind": "conflict", "table": t, "color": c, "answer": rest[0], "associated": want,
                        "true_row": order2.index(c), "row_entries": [f"{k} goes with the {conflict[k]}" for k in order2],
                        "prompt": render(conflict)})
    return out


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=45 * 60)
def run(seed: int, n_tables: int):
    import time
    import torch, transformers
    from huggingface_hub import model_info

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="eager").eval()
    text = hf.model.language_model
    text.config.loop_num_passes = 3
    attns = []
    for layer in text.layers:
        layer.self_attn.register_forward_hook(lambda mod, args, output: attns.append(output[1][0, :, -1].float().cpu()))

    def forward(prompt):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        attns.clear()
        with torch.no_grad():
            probs = torch.softmax(hf(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
        assert len(attns) == 48
        top2 = probs.topk(2)
        return {"top": tok.decode([int(top2.indices[0])]).strip(), "p_top": float(top2.values[0]),
                "margin": float(top2.values[0] - top2.values[1]), "attn": attns[HEAD_CALL][HEAD]}

    rows = []
    for it in prompts(seed, n_tables):
        out = forward(it["prompt"])
        offs = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        start, per_row = it["prompt"].index("Fruits for colors:"), []
        for entry in it["row_entries"]:
            a = it["prompt"].index(entry, start)
            per_row.append(float(sum(out["attn"][i] for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a)))
        total = max(sum(per_row), 1e-9)
        row = {"kind": it["kind"], "color": it["color"], "prompt": it["prompt"], "answer": it["answer"], "associated": it["associated"],
               "top": out["top"], "p_top": out["p_top"], "margin": out["margin"],
               "off_correct_row": 1 - per_row[it["true_row"]] / total, "fruit_table_attention": total}
        if it["kind"] == "conflict":
            row["outcome"] = "table" if out["top"] == it["answer"] else "association" if out["top"] == it["associated"] else "other"
        elif out["top"] != it["answer"]:
            row["outcome"] = "answer_wrong"
        else:
            edited = forward(it["edit_intended"])["top"]
            row["edit_intended"] = edited
            row["outcome"] = ("reads_the_row" if edited == it["spare"] else "answers_from_association" if edited == it["answer"] else "unclear")
        rows.append(row)

    def auc(pos, neg):
        return None if not pos or not neg else sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    count = lambda kind: {k: sum(r["outcome"] == k for r in rows if r["kind"] == kind) for k in sorted({r["outcome"] for r in rows if r["kind"] == kind})}
    hidden = [r for r in rows if r["outcome"] == "answers_from_association"]
    reads = [r for r in rows if r["outcome"] == "reads_the_row"]
    summary = {"conflict": count("conflict"), "agree": count("agree"),
               "frozen_head_auc": auc([r["off_correct_row"] for r in hidden], [r["off_correct_row"] for r in reads]),
               "confidence_auc": auc([-r["p_top"] for r in hidden], [-r["p_top"] for r in reads]),
               "flagged_hidden": f"{sum(r['off_correct_row'] > THRESHOLD for r in hidden)}/{len(hidden)}",
               "flagged_reads_the_row": f"{sum(r['off_correct_row'] > THRESHOLD for r in reads)}/{len(reads)}"}
    print("conflict (visible):", summary["conflict"])
    print("agreement (hidden):", summary["agree"])
    print(f"frozen head AUC {summary['frozen_head_auc']} | confidence AUC {summary['confidence_auc']} | flagged hidden {summary['flagged_hidden']}, "
          f"flagged reads-the-row {summary['flagged_reads_the_row']}")
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "seed": seed, "summary": summary, "rows": rows, "gpu_minutes": (time.time() - t0) / 60}
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(pilot: bool = False):
    here = pathlib.Path(__file__).parent / "results"
    if pilot:
        (here / "prior-pilot.json").write_text(json.dumps(run.remote(60, 40), indent=1))
    else:
        (here / "prior-test.json").write_text(json.dumps(run.remote(61, 150), indent=1))
    print("wrote results")
