"""Question 1: what do the messages between loop passes carry? Forward-only message swapping.

    modal run patch.py        # writes results/message-swap.json

Run prompt B and save its messages; run prompt A with B's messages swapped in; see what A answers.
"""
import itertools
import json
import pathlib

import modal

MODELS = ["thinkingmachines/Inkling-2-Mini", "thinkingmachines/TML-d24-varloop-research"]

app = modal.App("jlens-inkling-patch")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install("git+https://github.com/huggingface/transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# (prompt, answer, intermediate or None). Pairs are formed within a family only.
CAPITALS = [("Tokyo", "Japan", "Japanese"), ("Moscow", "Russia", "Russian"), ("Berlin", "Germany", "German"),
            ("Paris", "France", "French"), ("Rome", "Italy", "Italian"), ("London", "England", "English"),
            ("Madrid", "Spain", "Spanish"), ("Beijing", "China", "Chinese")]
FAMILIES = {
    "two_hop": [(f"The language spoken in the country whose capital is {city} is", lang, country)
                for city, country, lang in CAPITALS],
    "addition": [(f"{a} + {b} =", str(a + b), None)
                 for a, b in [(7, 5), (9, 8), (6, 3), (4, 9), (8, 6), (15, 27), (47, 38), (23, 41), (52, 19), (31, 46)]],
    "multiplication": [(f"{a} * {b} =", str(a * b), None) for a, b in [(6, 7), (8, 9), (3, 4), (7, 8), (5, 6), (9, 3)]],
}
# which message_input calls to patch: 12 loop layers per later pass, pass 2 first then pass 3
PASS_SETS = {"pass2": range(0, 12), "pass3": range(12, 24), "both": range(0, 24)}


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=40 * 60)
def run():
    import gc, time
    import torch, transformers

    t0 = time.time()
    out = {"models": {}}
    def one_model(name):
        tok = transformers.AutoTokenizer.from_pretrained(name)
        hf = transformers.AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()
        text = hf.model.language_model
        text.config.loop_num_passes = 3

        # One hook on the message builder. mode: None (pass through), "record", or a patch spec.
        state = {"mode": None, "i": 0, "saved": [], "src": None, "calls": (), "last_only": False}

        def on_message(mod, args, output):
            i, state["i"] = state["i"], state["i"] + 1
            if state["mode"] == "record":
                state["saved"].append(output.detach().clone())
            elif state["mode"] == "patch" and i in state["calls"]:
                if not state["last_only"]:
                    return state["src"][i]
                patched = output.clone()
                patched[:, -1] = state["src"][i][:, -1]
                return patched

        text.message_input.register_forward_hook(on_message)

        def forward(ids, mode=None, src=None, calls=(), last_only=False):
            state.update(mode=mode, i=0, saved=[], src=src, calls=set(calls), last_only=last_only)
            with torch.no_grad():
                logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
            assert state["i"] == 24, state["i"]  # 12 loop layers x 2 later passes
            return logits, state["saved"]

        def single(words):
            return {i[0] for w in words for v in (w, w.lower(), w.capitalize())
                    if len(i := tok.encode(v, add_special_tokens=False)) == 1}

        def prepare(prompt, answer, mid):
            """Token ids plus the token sets that count as the answer / the intermediate. Words this tokenizer
            writes as space + word are scored after appending the space to the prompt."""
            ans, text_in = single([" " + answer]), "Fact: " + prompt
            if not ans:
                ans, text_in = single([answer]), text_in + " "
            ids = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            mids = single([" " + mid, mid]) if mid else set()
            return {"prompt": prompt, "answer": answer, "ids": ids, "ans": ans, "mid": mids}

        res, usable_by_family = {}, {}
        for fam, items in FAMILIES.items():
            preps = [prepare(*it) for it in items]
            for p in preps:
                p["clean"], p["msgs"] = forward(p["ids"], "record")
                p["clean_top"] = int(p["clean"].argmax())
                selfswap, _ = forward(p["ids"], "patch", p["msgs"], PASS_SETS["both"])
                p["self_ok"] = bool(torch.equal(selfswap.argmax(), p["clean"].argmax())) and \
                    float((selfswap - p["clean"]).abs().max()) < 1e-2
            usable = [p for p in preps if p["ans"] and p["clean_top"] in p["ans"]]  # only prompts the model gets right
            usable_by_family[fam] = usable
            counts = {f"{ps}/{'last' if lo else 'all'}": {"to_B": 0, "stays_A": 0, "B_intermediate": 0, "other": 0}
                      for ps in PASS_SETS for lo in (False, True)}
            examples, n_pairs = [], 0
            for A, B in itertools.permutations(usable, 2):
                if A["ids"].shape != B["ids"].shape or A["ans"] & B["ans"]:
                    continue  # positions must line up, and the two answers must differ
                n_pairs += 1
                row = {"A": A["prompt"], "B": B["prompt"]}
                for ps, calls in PASS_SETS.items():
                    for lo in (False, True):
                        lg, _ = forward(A["ids"], "patch", B["msgs"], calls, lo)
                        top = int(lg.argmax())
                        kind = ("to_B" if top in B["ans"] else "stays_A" if top in A["ans"]
                                else "B_intermediate" if top in B["mid"] else "other")
                        key = f"{ps}/{'last' if lo else 'all'}"
                        counts[key][kind] += 1
                        row[key] = tok.decode([top])
                if len(examples) < 12:
                    examples.append(row)
            res[fam] = {"n_prompts": len(preps), "n_correct_clean": len(usable), "n_pairs": n_pairs,
                        "self_swap_reproduces_clean": sum(p["self_ok"] for p in preps), "counts": counts, "examples": examples}
            print(f"\n=== {name.split('/')[-1]} | {fam}: {len(usable)}/{len(preps)} prompts right unpatched, {n_pairs} pairs, "
                  f"self-swap ok {res[fam]['self_swap_reproduces_clean']}/{len(preps)}")
            for key, c in counts.items():
                print(f"  {key:<11} to_B {c['to_B']:>3}  stays_A {c['stays_A']:>3}  B_intermediate {c['B_intermediate']:>3}  other {c['other']:>3}", flush=True)
        # Control: messages from a different operation. "7 + 5 =" with messages from "6 * 7 =":
        # does it answer 12 (its own), 42 (B's), or 35 (its own numbers with B's operation)?
        ops = {"addition": lambda a, b: a + b, "multiplication": lambda a, b: a * b}
        cross = {f"{ps}/all": {"stays_A": 0, "to_B": 0, "A_numbers_B_operation": 0, "other": 0} for ps in PASS_SETS}
        n_cross, cross_examples = 0, []
        for fa, fb in (("addition", "multiplication"), ("multiplication", "addition")):
            for A, B in itertools.product(usable_by_family[fa], usable_by_family[fb]):
                if A["ids"].shape != B["ids"].shape:
                    continue
                x, _, y, _ = A["prompt"].split()
                mixed = single([str(ops[fb](int(x), int(y)))])
                if mixed & A["ans"] or mixed & B["ans"] or A["ans"] & B["ans"]:
                    continue  # the three outcomes must be distinguishable
                n_cross += 1
                row = {"A": A["prompt"], "B": B["prompt"]}
                for ps, calls in PASS_SETS.items():
                    top = int(forward(A["ids"], "patch", B["msgs"], calls)[0].argmax())
                    kind = ("stays_A" if top in A["ans"] else "to_B" if top in B["ans"]
                            else "A_numbers_B_operation" if top in mixed else "other")
                    cross[f"{ps}/all"][kind] += 1
                    row[ps] = tok.decode([top])
                if len(cross_examples) < 12:
                    cross_examples.append(row)
        res["cross_operation"] = {"n_pairs": n_cross, "counts": cross, "examples": cross_examples}
        print(f"\n=== {name.split('/')[-1]} | cross_operation: {n_cross} pairs")
        for key, c in cross.items():
            print(f"  {key:<11} " + "  ".join(f"{k} {v:>3}" for k, v in c.items()), flush=True)
        return res

    for name in MODELS:
        out["models"][name] = one_model(name)  # locals (model, hooks, saved messages) die on return
        gc.collect(); torch.cuda.empty_cache()

    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"\ntotal {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main():
    out = run.remote()
    path = pathlib.Path(__file__).parent / "results" / "message-swap.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
