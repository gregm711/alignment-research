"""Which position rule? Separating "row counted from the front", "row counted from the end" and "same token position".

    uvx --with tokenizers python discriminate.py --plan TOKENIZER_JSON   # offline: writes results/discriminate-plan.json
    modal run discriminate.py                                            # runs the committed plan, writes results/discriminate.json

Donor and recipient tables are generated independently. One has a 5-row second table padded with two
6-token filler lines, the other a 7-row second table, so the prompts are the same length in tokens but
the three rules point at three different rows. Every explanation's predicted fruit is computed offline
and committed in the plan before any model is run.
"""
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).parent
PLAN = HERE / "results" / "discriminate-plan.json"
ANIMALS = ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]
COLORS = ["red", "blue", "green", "gold", "pink", "gray", "black", "white", "brown", "purple", "silver", "yellow"]
FRUITS = ["fig", "lime", "grape", "apple", "orange", "date", "olive"]
FILLER = "This line means nothing.\n"   # must be exactly 6 tokens; checked when the plan is built
QUESTION = "So the fruit that goes with the {q} is the"


def table_set(rng, n_rows):
    """One independently generated pair of tables, printed in random row orders. With n_rows=5 the second
    table lists only five of the seven colours and the prompt carries two filler lines."""
    animals, colors = rng.sample(ANIMALS, 7), rng.sample(COLORS, 7)
    t1 = dict(zip(animals, colors))
    listed = rng.sample(colors, n_rows)
    t2 = dict(zip(listed, rng.sample(FRUITS, n_rows)))
    first = ", ".join(f"the {a} is {t1[a]}" for a in rng.sample(animals, 7))
    second = ", ".join(f"{c} goes with the {t2[c]}" for c in listed)
    pad = FILLER if n_rows == 5 else ""
    body = f"Fact sheet.\nColors of animals: {first}.\n{pad}Fruits for colors: {second}.\n{pad}"
    return [{"n_rows": n_rows, "item": a, "mid": t1[a], "answer": t2[t1[a]], "row": listed.index(t1[a]),
             "t1": t1, "t2": t2, "rows": [t2[c] for c in listed], "prompt": body + QUESTION.format(q=a)}
            for a in animals if t1[a] in t2]


def build_plan(tokenizer_json):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_json)
    assert len(tok.encode(FILLER, add_special_tokens=False).ids) == 6, "filler must be 6 tokens"
    rng = random.Random(7)
    short = [table_set(rng, 5) for _ in range(8)]
    long_ = [table_set(rng, 7) for _ in range(8)]
    for it in (x for s in short + long_ for x in s):
        enc = tok.encode(it["prompt"], add_special_tokens=False)
        start, it["n_tokens"], it["fruit_token"] = it["prompt"].index("Fruits for colors:"), len(enc.ids), []
        for fruit in it["rows"]:  # token index of each row's fruit word
            a = it["prompt"].index(f"the {fruit}", start) + 4
            hits = [i for i, (x, y) in enumerate(enc.offsets) if x < a + len(fruit) and y > a]
            assert len(hits) == 1, (fruit, hits)
            it["fruit_token"].append(hits[0])

    def predictions(R, D):
        j, nD, nR = D["row"], D["n_rows"], R["n_rows"]
        p = {"recipient": R["answer"], "donor_copied": D["answer"]}
        if j < nR:
            p["row_from_front"] = R["rows"][j]
        if 0 <= nR - (nD - j) < nR:
            p["row_from_end"] = R["rows"][nR - (nD - j)]
        if D["fruit_token"][j] in R["fruit_token"]:
            p["same_token_position"] = R["rows"][R["fruit_token"].index(D["fruit_token"][j])]
        if D["mid"] in R["t2"]:
            p["donor_colour"] = R["t2"][D["mid"]]
        if D["item"] in R["t1"] and R["t1"][D["item"]] in R["t2"]:
            p["donor_animal"] = R["t2"][R["t1"][D["item"]]]
        return p

    core = ("recipient", "donor_copied", "row_from_front", "row_from_end", "same_token_position")
    pairs = []
    for direction, donors, recipients in (("donor_short_recipient_long", short, long_), ("donor_long_recipient_short", long_, short)):
        for D in (x for s in donors for x in s):
            used = 0
            for rset in rng.sample(recipients, len(recipients)):  # each donor meets several unrelated recipient layouts
                R = rng.choice(rset)
                p = predictions(R, D)
                if R["n_tokens"] != D["n_tokens"] or not all(k in p for k in core) or len({p[k] for k in core}) < 5:
                    continue
                pairs.append({"direction": direction, "donor": D["prompt"], "recipient": R["prompt"],
                              "donor_answer": D["answer"], "recipient_answer": R["answer"], "donor_row": D["row"], "predicted": p})
                used += 1
                if used == 4:
                    break
    PLAN.write_text(json.dumps({"filler": FILLER, "n_pairs": len(pairs), "pairs": pairs}, indent=1))
    by = {d: sum(x["direction"] == d for x in pairs) for d in ("donor_short_recipient_long", "donor_long_recipient_short")}
    print(f"wrote {PLAN.name}: {len(pairs)} pairs {by}; distinct donors {len({x['donor'] for x in pairs})}, recipients {len({x['recipient'] for x in pairs})}")
    print(pairs[0]["donor"], "\n---\n", pairs[0]["recipient"], "\n", pairs[0]["predicted"], "donor row", pairs[0]["donor_row"])


if "--plan" in sys.argv:
    build_plan(sys.argv[sys.argv.index("--plan") + 1])
    sys.exit()

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
app = modal.App("jlens-inkling-discriminate")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=30 * 60)
def run(plan: dict):
    import time
    import torch, transformers
    from huggingface_hub import model_info

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    text = hf.model.language_model
    text.config.loop_num_passes = 3
    msg = {"mode": None, "i": 0, "saved": [], "src": None}

    def on_message(mod, args, output):
        i, msg["i"] = msg["i"], msg["i"] + 1
        if msg["mode"] == "record":
            msg["saved"].append(output.detach().cpu())
        elif msg["mode"] == "patch" and i >= 12:  # calls 12-23 feed the final pass
            return msg["src"][i].to(output.device)

    text.message_input.register_forward_hook(on_message)

    def forward(prompt, mode=None, src=None):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        msg.update(mode=mode, i=0, saved=[], src=src)
        with torch.no_grad():
            top = tok.decode([int(hf(input_ids=ids, use_cache=False).logits[0, -1].argmax())]).strip()
        assert msg["i"] == 24
        return top, msg["saved"], ids.shape[1]

    clean = {}
    for pr in sorted({x[k] for x in plan["pairs"] for k in ("donor", "recipient")}):
        clean[pr] = forward(pr, "record")
    rows = []
    for x in plan["pairs"]:
        (d_top, d_msgs, d_n), (r_top, _, r_n) = clean[x["donor"]], clean[x["recipient"]]
        if d_top != x["donor_answer"] or r_top != x["recipient_answer"]:
            continue  # both prompts must be answered correctly unaided
        assert d_n == r_n, (d_n, r_n)
        top, _, _ = forward(x["recipient"], "patch", d_msgs)
        rows.append({**x, "top": top, "matches": [k for k, v in x["predicted"].items() if v == top]})
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "transformers": TRANSFORMERS_PIN,
           "unaided_right": f"{sum(v[0] in {x['donor_answer'] for x in plan['pairs'] if x['donor'] == k} | {x['recipient_answer'] for x in plan['pairs'] if x['recipient'] == k} for k, v in clean.items())}/{len(clean)}",
           "rows": rows, "gpu_minutes": (time.time() - t0) / 60}
    for d in ("donor_short_recipient_long", "donor_long_recipient_short"):
        sub = [r for r in rows if r["direction"] == d]
        keys = ("recipient", "donor_copied", "row_from_front", "row_from_end", "same_token_position")
        print(f"{d}: {len(sub)} eligible pairs | " + "  ".join(f"{k} {sum(k in r['matches'] for r in sub)}" for k in keys)
              + f"  none_of_these {sum(not set(r['matches']) & set(keys) for r in sub)}", flush=True)
    print(f"unaided right {out['unaided_right']} | total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main():
    out = run.remote(json.loads(PLAN.read_text()))
    (HERE / "results" / "discriminate.json").write_text(json.dumps(out, indent=1))
    print("wrote results/discriminate.json")
