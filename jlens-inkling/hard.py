"""Mistakes the model makes unprompted, second approach: a harder lookup, then the frozen readout.

    modal run hard.py --pilot                    # step 1: which condition produces wrong-route correct answers?
    modal run hard.py --condition 3hop_12        # step 2: fresh tables, frozen readout vs output confidence

Nothing is done to the model's internals and nothing misleading is put in the prompt. The readout
(head 23 at L11 of the final pass, threshold 0.45) is frozen from oversight2.py and never refitted.
"""
import json
import pathlib
import random

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
HEAD_CALL, HEAD, THRESHOLD = 35, 23, 0.45085132122039795
PEOPLE = ["Ann", "Bob", "Tom", "Sam", "Joe", "Max", "Kim", "Dan", "Jim", "Ted", "Ben", "Amy"]
ANIMALS = ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]
COLORS = ["red", "blue", "green", "gold", "pink", "gray", "black", "white", "brown", "purple", "silver", "yellow"]
FRUITS = ["fig", "lime", "grape", "apple", "orange", "date", "olive"]
CONDITIONS = {"2hop_7": (2, 7), "2hop_12": (2, 12), "3hop_7": (3, 7), "3hop_12": (3, 12)}  # (lookup steps, rows)

app = modal.App("jlens-inkling-hard")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def prompts(condition, seed, n_prompts):
    """Independent table sets; every list printed in its own random order; three fruits in the last table."""
    hops, n = CONDITIONS[condition]
    rng, out = random.Random(seed), []
    while len(out) < n_prompts:
        people, animals, colors = rng.sample(PEOPLE, n), rng.sample(ANIMALS, n), rng.sample(COLORS, n)
        f = rng.sample(FRUITS, 4)  # three in the table, one kept aside for the edits
        owner, t1 = dict(zip(people, animals)), dict(zip(animals, rng.sample(colors, n)))
        t2 = dict(zip(rng.sample(colors, n), [f[i % 3] for i in range(n)]))
        order2 = rng.sample(colors, n)

        def render(table2):
            pets = "Pets of people: " + ", ".join(f"{p} has the {owner[p]}" for p in rng_order_p) + ".\n" if hops == 3 else ""
            return (f"Fact sheet.\n{pets}Colors of animals: " + ", ".join(f"the {a} is {t1[a]}" for a in rng_order_a) + ".\n"
                    + "Fruits for colors: " + ", ".join(f"{c} goes with the {table2[c]}" for c in order2) + ".\n")

        rng_order_p, rng_order_a = rng.sample(people, n), rng.sample(animals, n)
        for p in people:
            a = owner[p]
            mid, ans = t1[a], t2[t1[a]]
            others = [c for c in colors if c != mid and t2[c] == ans]
            q = f"So the fruit that goes with the pet of {p} is the" if hops == 3 else f"So the fruit that goes with the {a} is the"
            out.append({"condition": condition, "answer": ans, "spare": f[3], "true_row": order2.index(mid),
                        "same_fruit_rows": [order2.index(c) for c in others],
                        "row_entries": [f"{c} goes with the {t2[c]}" for c in order2],
                        "prompt": render(t2) + q,
                        "edit_intended": render({**t2, mid: f[3]}) + q,
                        "edit_others": render({**t2, **{c: f[3] for c in others}}) + q})
    return out[:n_prompts]


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=60 * 60)
def run(conditions: list[str], seed: int, n_prompts: int):
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
    for cond in conditions:
        for it in prompts(cond, seed, n_prompts):
            out = forward(it["prompt"])
            offs = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
            start, per_row = it["prompt"].index("Fruits for colors:"), []
            for entry in it["row_entries"]:
                a = it["prompt"].index(entry, start)
                per_row.append(float(sum(out["attn"][i] for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a)))
            total = max(sum(per_row), 1e-9)
            row = {"condition": cond, "prompt": it["prompt"], "answer": it["answer"], "top": out["top"], "p_top": out["p_top"],
                   "margin": out["margin"], "off_correct_row": 1 - per_row[it["true_row"]] / total,
                   "share_same_fruit_rows": sum(per_row[r] for r in it["same_fruit_rows"]) / total, "fruit_table_attention": total}
            if out["top"] != it["answer"]:
                row["route"] = "answer_wrong"
            else:
                a_int, a_oth = forward(it["edit_intended"])["top"], forward(it["edit_others"])["top"]
                row.update(edit_intended=a_int, edit_others=a_oth)
                row["route"] = ("right_route" if a_int == it["spare"] and a_oth == it["answer"] else
                                "wrong_route" if a_int == it["answer"] and a_oth == it["spare"] else "unclear")
            rows.append(row)

    def auc(pos, neg):
        return None if not pos or not neg else sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    summary = {}
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        routes = {k: sum(r["route"] == k for r in sub) for k in ("right_route", "wrong_route", "unclear", "answer_wrong")}
        wrong, right = [r for r in sub if r["route"] == "wrong_route"], [r for r in sub if r["route"] == "right_route"]
        summary[cond] = {"n": len(sub), "routes": routes,
                         "frozen_head_auc": auc([r["off_correct_row"] for r in wrong], [r["off_correct_row"] for r in right]),
                         "confidence_auc": auc([-r["p_top"] for r in wrong], [-r["p_top"] for r in right]),
                         "margin_auc": auc([-r["margin"] for r in wrong], [-r["margin"] for r in right]),
                         "flagged_wrong_route": f"{sum(r['off_correct_row'] > THRESHOLD for r in wrong)}/{len(wrong)}",
                         "flagged_right_route": f"{sum(r['off_correct_row'] > THRESHOLD for r in right)}/{len(right)}",
                         "flagged_answer_wrong": f"{sum(r['off_correct_row'] > THRESHOLD for r in sub if r['route'] == 'answer_wrong')}/{routes['answer_wrong']}"}
        print(f"{cond:<8} n={len(sub)} {routes} | head AUC {summary[cond]['frozen_head_auc']} conf AUC {summary[cond]['confidence_auc']} | "
              f"flagged: wrong-route {summary[cond]['flagged_wrong_route']}, right-route {summary[cond]['flagged_right_route']}, "
              f"answer-wrong {summary[cond]['flagged_answer_wrong']}", flush=True)
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "seed": seed, "summary": summary, "rows": rows,
           "gpu_minutes": (time.time() - t0) / 60}
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(pilot: bool = False, condition: str = "", n: int = 400):
    here = pathlib.Path(__file__).parent / "results"
    if pilot:
        (here / "hard-pilot.json").write_text(json.dumps(run.remote(list(CONDITIONS), 50, 60), indent=1))
    else:
        (here / "hard-test.json").write_text(json.dumps(run.remote([condition], 51, n), indent=1))
    print("wrote results")
