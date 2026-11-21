"""Follow-up: what exactly crosses between passes? A message transplant with four distinguishable answers.

    modal run transplant.py --clean-only     # pilot: how many prompts does each model answer correctly?
    modal run transplant.py                  # writes results/transplant-four-way.json (every eligible pair)

Donor and recipient are two-step questions that differ in both the item and the second step, so the
recipient's answer after receiving the donor's messages says what the messages carried:
  recipient      -> ignored
  donor          -> a finished answer
  r_step_on_d    -> a reusable intermediate (recipient's second step applied to the donor's intermediate)
  d_step_on_r    -> which second step to do (donor's second step applied to the recipient's intermediate)
"""
import itertools
import json
import pathlib
import random

import modal

MODELS = ["thinkingmachines/Inkling-2-Mini", "thinkingmachines/TML-d24-varloop-research"]
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
PASS_SETS = {"pass3": range(12, 24), "pass2": range(0, 12)}  # message-builder calls feeding each later pass

app = modal.App("jlens-inkling-transplant")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---- Family 1: fictional lookup tables given in the prompt -------------------------------------------
# Every word below is a single token for this tokenizer when it follows a space (checked offline).
ANIMALS = ["cat", "dog", "bat", "pig", "cow", "hen", "rat", "bee", "ant", "ram", "elk", "yak"]
COLORS = ["red", "blue", "green", "gold", "pink", "gray", "black", "white", "brown", "purple", "silver", "yellow"]
FRUITS = ["fig", "lime", "grape", "apple", "orange", "date", "olive"]
N_TABLES = 12


LOOKUP_WORDINGS = {"is_the": "Fact: The fruit for the color of the {q} is the",
                   "so": "So the fruit that goes with the {q} is the",
                   "qa": "Question: Which fruit goes with the color of the {q}?\nAnswer: the"}


def lookup_prompt(animals, t1, t2, query, wording):
    first = ", ".join(f"the {a} is {t1[a]}" for a in animals)
    second = ", ".join(f"{c} goes with the {t2[c]}" for c in (t1[a] for a in animals))
    return f"Fact sheet.\nColors of animals: {first}.\nFruits for colors: {second}.\n" + LOOKUP_WORDINGS[wording].format(q=query)


def lookup_items(wording):
    """Each item is one prompt: (table id, which second table, query animal) with its intermediate and answer."""
    rng, items = random.Random(0), []
    for t in range(N_TABLES):
        animals, colors, fruits = (rng.sample(pool, 4) for pool in (ANIMALS, COLORS, FRUITS))
        t1 = dict(zip(animals, colors))
        t2a = dict(zip(colors, fruits))
        t2b = dict(zip(colors, fruits[1:] + fruits[:1]))  # every colour gets a different fruit than in t2a
        for step, t2 in (("a", t2a), ("b", t2b)):
            for q in animals:
                items.append({"group": t, "step": step, "item": q, "mid": t1[q], "answer": t2[t1[q]],
                              "prompt": lookup_prompt(animals, t1, t2, q, wording)})
    return items


# ---- Family 2: real facts ---------------------------------------------------------------------------
# (capital, country, language, currency, continent). Capitals, languages and currencies are single tokens.
COUNTRIES = [("Tokyo", "Japan", "Japanese", "yen", "Asia"), ("London", "England", "English", "pound", "Europe"),
             ("Beijing", "China", "Chinese", "yuan", "Asia"), ("Paris", "France", "French", "euro", "Europe"),
             ("Berlin", "Germany", "German", "euro", "Europe"), ("Rome", "Italy", "Italian", "euro", "Europe"),
             ("Madrid", "Spain", "Spanish", "euro", "Europe")]
# wording -> (template, the two properties asked about). Chosen by unaided accuracy, before any transplant.
FACT_WORDINGS = {
    "speak_pay": ("Fact: In the country whose capital is {city}, people {verb} in", ("language", "currency")),
    "language_continent": ("Fact: The {prop} of the country whose capital is {city} is", ("language", "continent")),
    "language_currency_the": ("Fact: The {prop} of the country whose capital is {city} is the", ("language", "currency")),
}
FIELD = {"language": 2, "currency": 3, "continent": 4}
VERB = {"language": "speak", "currency": "pay", "continent": ""}


def fact_items(wording):
    template, props = FACT_WORDINGS[wording]
    return [{"group": 0, "step": prop, "item": c[0], "mid": c[1], "answer": c[FIELD[prop]],
             "prompt": template.format(prop=prop, city=c[0], verb=VERB[prop])} for c in COUNTRIES for prop in props]


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=40 * 60)
def run(clean_only: bool = False):
    import gc, time
    import torch, transformers
    from huggingface_hub import model_info

    t0 = time.time()
    out = {"transformers": TRANSFORMERS_PIN, "models": {}}

    def one_model(name):
        tok = transformers.AutoTokenizer.from_pretrained(name)
        hf = transformers.AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()
        text = hf.model.language_model
        text.config.loop_num_passes = 3
        state = {"mode": None, "i": 0, "saved": [], "src": None, "calls": ()}

        def on_message(mod, args, output):
            i, state["i"] = state["i"], state["i"] + 1
            if state["mode"] == "record":
                state["saved"].append(output.detach().clone())
            elif state["mode"] == "patch" and i in state["calls"]:
                return state["src"][i]

        text.message_input.register_forward_hook(on_message)

        def forward(ids, mode=None, src=None, calls=()):
            state.update(mode=mode, i=0, saved=[], src=src, calls=set(calls))
            with torch.no_grad():
                top = int(hf(input_ids=ids, use_cache=False).logits[0, -1].argmax())
            assert state["i"] == 24, state["i"]
            return top, state["saved"]

        def token(word):  # the word as it would follow the prompt; None unless it is exactly one token
            ids = tok.encode(" " + word, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None

        def prepare(items):
            """Run each prompt clean; keep its messages and whether the model gets it right."""
            for it in items:
                it["ans_id"] = token(it["answer"])
                it["ids"] = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                it["clean_top"], it["msgs"] = forward(it["ids"], "record")
                it["right"] = it["ans_id"] is not None and it["clean_top"] == it["ans_id"]
            return items

        res = {"revision": model_info(name).sha, "clean": {}}
        fams = {}
        for fam, wordings, build in (("lookup", LOOKUP_WORDINGS, lookup_items), ("facts", FACT_WORDINGS, fact_items)):
            tried = {w: prepare(build(w)) for w in wordings}
            best = max(tried, key=lambda w: sum(it["right"] for it in tried[w]))  # chosen before any transplant
            fams[fam] = tried[best]
            res["clean"][fam] = {**{w: f"{sum(it['right'] for it in v)}/{len(v)}" for w, v in tried.items()}, "used": best}
            if clean_only:
                for w, v in tried.items():
                    for it in v[:3] + v[-2:]:
                        print(f"   [{w}] want {it['answer']!r:<11} got {tok.decode([it['clean_top']])!r:<11} | ...{it['prompt'][-62:]!r}")
        print(f"\n=== {name.split('/')[-1]} | unaided accuracy: {res['clean']}", flush=True)
        if clean_only:
            return res

        for fam, items in fams.items():
            index = {(it["group"], it["step"], it["item"]): it for it in items}
            rows = []
            for R, D in itertools.permutations(items, 2):
                if R["group"] != D["group"] or R["step"] == D["step"] or R["item"] == D["item"]:
                    continue  # same tables / fact set, different second step, different item
                r_on_d, d_on_r = index[R["group"], R["step"], D["item"]], index[D["group"], D["step"], R["item"]]
                four = [R, D, r_on_d, d_on_r]
                if not all(x["right"] for x in four) or len({x["ans_id"] for x in four}) < 4 or R["ids"].shape != D["ids"].shape:
                    continue  # the model must know all four facts, and the four answers must be different words
                expected = {"recipient": R["ans_id"], "donor": D["ans_id"], "r_step_on_d": r_on_d["ans_id"], "d_step_on_r": d_on_r["ans_id"]}
                row = {"recipient": R["prompt"], "donor": D["prompt"],
                       "expected": {k: tok.decode([v]) for k, v in expected.items()}}
                for ps, calls in PASS_SETS.items():
                    top, _ = forward(R["ids"], "patch", D["msgs"], calls)
                    row[ps] = {"top": tok.decode([top]), "outcome": next((k for k, v in expected.items() if v == top), "other")}
                rows.append(row)
            counts = {ps: {k: sum(r[ps]["outcome"] == k for r in rows) for k in ("recipient", "donor", "r_step_on_d", "d_step_on_r", "other")}
                      for ps in PASS_SETS}
            res[fam] = {"n_pairs": len(rows), "n_recipient_prompts": len({r["recipient"] for r in rows}), "counts": counts, "rows": rows}
            print(f"--- {fam}: {len(rows)} eligible pairs over {res[fam]['n_recipient_prompts']} recipient prompts")
            for ps, c in counts.items():
                print(f"    into {ps}: " + "  ".join(f"{k} {v}" for k, v in c.items()), flush=True)
        return res

    for name in MODELS[:1] if clean_only else MODELS:
        out["models"][name] = one_model(name)  # locals die on return, freeing the model
        gc.collect(); torch.cuda.empty_cache()
    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(clean_only: bool = False):
    out = run.remote(clean_only)
    if not clean_only:
        path = pathlib.Path(__file__).parent / "results" / "transplant-four-way.json"
        path.write_text(json.dumps(out, indent=1))
        print("wrote", path)
