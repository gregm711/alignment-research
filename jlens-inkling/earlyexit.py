"""Follow-up: does the early-exit finding hold on prompts other people wrote?

    modal run earlyexit.py                 # writes results/early-exit-public.json (per-item rows)
    python3 earlyexit.py --analyse         # prints rates with bootstrap intervals from that file

Data: TwoHopFact (Yang et al., 2024, CC BY 4.0) and Anthropic's lens evaluation sets (Apache 2.0).
"""
import ast
import json
import pathlib
import random
import sys

RESULTS = pathlib.Path(__file__).parent / "results" / "early-exit-public.json"
MODELS = ["thinkingmachines/Inkling-2-Mini", "thinkingmachines/TML-d24-varloop-research"]
PASSES = [1, 2, 3]
N_TWOHOP = 400
LENS_EVALS = "https://raw.githubusercontent.com/anthropics/jacobian-lens/main/data/evaluations/"


def analyse():
    data = json.loads(RESULTS.read_text())
    rng = random.Random(0)

    def rate(rows, f):
        vals = [f(r) for r in rows]
        boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
        return sum(vals) / len(vals), boots[49], boots[1949]

    def diff(rows, f, g):  # paired difference f - g with a bootstrap interval
        vals = [f(r) - g(r) for r in rows]
        boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
        return sum(vals) / len(vals), boots[49], boots[1949]

    fmt = lambda t: f"{100 * t[0]:5.1f}% [{100 * t[1]:5.1f}, {100 * t[2]:5.1f}]"
    for name, m in data["models"].items():
        print(f"\n######## {name}")
        two = m["twohopfact"]
        known = [r for r in two if r["hop1_right_at_3"] and r["hop2_right_at_3"]]
        for label, rows in (("all items", two), ("both single hops known", known)):
            print(f"\nTwoHopFact, {label}: n={len(rows)}")
            for k in PASSES:
                print(f"  {k} pass(es): answer {fmt(rate(rows, lambda r: r['two_hop'][str(k)]['answer']))}"
                      f" | bridge {fmt(rate(rows, lambda r: r['two_hop'][str(k)]['bridge']))}"
                      f" | another item's bridge {fmt(rate(rows, lambda r: r['two_hop'][str(k)]['other_bridge']))}")
            print(f"  one-hop (first hop) answer: 1 pass {fmt(rate(rows, lambda r: r['hop1_right_at_1']))}"
                  f" | 3 passes {fmt(rate(rows, lambda r: r['hop1_right_at_3']))}")
            print(f"  gain 1->3 passes: two-hop {fmt(diff(rows, lambda r: r['two_hop']['3']['answer'], lambda r: r['two_hop']['1']['answer']))}"
                  f" | one-hop {fmt(diff(rows, lambda r: r['hop1_right_at_3'], lambda r: r['hop1_right_at_1']))}")
            print(f"  bridge named at 1 pass minus at 3 passes: {fmt(diff(rows, lambda r: r['two_hop']['1']['bridge'], lambda r: r['two_hop']['3']['bridge']))}"
                  f" | minus control: {fmt(diff(rows, lambda r: r['two_hop']['1']['bridge'], lambda r: r['two_hop']['1']['other_bridge']))}")
        for setname in ("lens-eval-multihop", "lens-eval-order-ops"):
            rows = m[setname]
            print(f"\n{setname}: n={len(rows)}")
            for k in PASSES:
                print(f"  {k} pass(es): target {fmt(rate(rows, lambda r: r['by_pass'][str(k)]['target']))}"
                      f" | an intermediate {fmt(rate(rows, lambda r: r['by_pass'][str(k)]['intermediate']))}")


if "--analyse" in sys.argv:
    analyse()
    sys.exit()

import modal

app = modal.App("jlens-inkling-earlyexit")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer", "pandas")
    .pip_install("git+https://github.com/huggingface/transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=60 * 60)
def run():
    import gc, time, urllib.request
    import pandas as pd
    import torch, transformers
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    df = pd.read_csv(hf_hub_download("soheeyang/TwoHopFact", "TwoHopFact.csv", repo_type="dataset"))
    per_type = -(-N_TWOHOP // df["fact_comp_type"].nunique())  # ceil: spread the sample across fact types
    sample = (df.sample(frac=1, random_state=0).groupby("fact_comp_type").head(per_type)
              .sample(frac=1, random_state=0).head(N_TWOHOP))
    aliases = lambda cell: [a for group in ast.literal_eval(cell) for a in group]
    twohop = [{"type": r["fact_comp_type"], "two": r["r2(r1(e1)).prompt"], "hop1": r["r1(e1).prompt"], "hop2": r["r2(e2).prompt"],
               "bridge": aliases(r["e2.aliases"]), "answer": aliases(r["e3.aliases"])} for _, r in sample.iterrows()]
    lens_sets = {s: json.loads(urllib.request.urlopen(LENS_EVALS + s + ".json", timeout=60).read())["items"]
                 for s in ("lens-eval-multihop", "lens-eval-order-ops")}
    print(f"TwoHopFact: {len(twohop)} items over {sample['fact_comp_type'].nunique()} fact types", flush=True)
    out = {"passes": PASSES, "models": {}}

    def one_model(name):
        tok = transformers.AutoTokenizer.from_pretrained(name)
        hf = transformers.AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()
        cfg = hf.model.language_model.config
        space = tok.encode(" ", add_special_tokens=False)

        def first_tokens(words, bare):
            """First token of each spelling, written the way it would follow the prompt. `bare`: the prompt
            already ends in the tokenizer's separate space token, so the word comes without a leading space."""
            out_ids = set()
            for w in words:
                for v in {w, w[:1].upper() + w[1:], w[:1].lower() + w[1:]}:
                    ids = tok.encode(v if bare else " " + v, add_special_tokens=False)
                    if not bare and ids[: len(space)] == space:
                        continue  # written as space + word: not reachable from this prompt variant
                    if ids:
                        out_ids.add(ids[0])
            return out_ids

        def variant(prompt, answer_words):
            """(input ids, bare?) Words this tokenizer writes as space + word are scored after appending the space."""
            prompt = prompt.rstrip()
            bare = not first_tokens(answer_words[:1], bare=False)
            ids = tok(prompt + (" " if bare else ""), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            return ids, bare

        def top_at(ids, k):
            cfg.loop_num_passes = k
            with torch.no_grad():
                return int(hf(input_ids=ids, use_cache=False).logits[0, -1].argmax())

        res = {"twohopfact": [], "lens-eval-multihop": [], "lens-eval-order-ops": []}
        for j, it in enumerate(twohop):
            ids, bare = variant(it["two"], it["answer"])
            ans, bridge = first_tokens(it["answer"], bare), first_tokens(it["bridge"], bare)
            other = first_tokens(twohop[(j + 1) % len(twohop)]["bridge"], bare)  # control: another item's bridge
            if not ans or not bridge or ans & bridge:
                continue
            row = {"type": it["type"], "two_hop": {}}
            for k in PASSES:
                t = top_at(ids, k)
                row["two_hop"][k] = {"answer": t in ans, "bridge": t in bridge, "other_bridge": t in (other - ans - bridge)}
            h1, b1 = variant(it["hop1"], it["bridge"])
            h2, b2 = variant(it["hop2"], it["answer"])
            row["hop1_right_at_1"] = top_at(h1, 1) in first_tokens(it["bridge"], b1)
            row["hop1_right_at_3"] = top_at(h1, 3) in first_tokens(it["bridge"], b1)
            row["hop2_right_at_3"] = top_at(h2, 3) in first_tokens(it["answer"], b2)
            res["twohopfact"].append(row)
        for setname, items in lens_sets.items():
            for it in items:
                ids, bare = variant(it["prompt"], [it["target"]])
                tgt, mids = first_tokens([it["target"]], bare), first_tokens(it["intermediates"], bare)
                if not tgt or not (mids - tgt):
                    continue
                res[setname].append({"name": it["name"], "by_pass": {
                    k: (lambda t: {"target": t in tgt, "intermediate": t in (mids - tgt), "top": tok.decode([t])})(top_at(ids, k))
                    for k in PASSES}})
        short = name.split("/")[-1]
        for k in PASSES:
            two = res["twohopfact"]
            print(f"{short} | {k} pass(es) | TwoHopFact n={len(two)}: answer {sum(r['two_hop'][k]['answer'] for r in two)}, "
                  f"bridge {sum(r['two_hop'][k]['bridge'] for r in two)}, other bridge {sum(r['two_hop'][k]['other_bridge'] for r in two)} | "
                  + " | ".join(f"{s} n={len(res[s])}: target {sum(r['by_pass'][k]['target'] for r in res[s])}, "
                               f"intermediate {sum(r['by_pass'][k]['intermediate'] for r in res[s])}" for s in lens_sets), flush=True)
        return res

    for name in MODELS:
        out["models"][name] = one_model(name)
        gc.collect(); torch.cuda.empty_cache()
    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=15 * 60)
def peek(n: int, prefix: str = ""):
    """Diagnostic: print what the released model actually says on the first n TwoHopFact one-hop and two-hop prompts."""
    import pandas as pd
    import torch, transformers
    from huggingface_hub import hf_hub_download

    df = pd.read_csv(hf_hub_download("soheeyang/TwoHopFact", "TwoHopFact.csv", repo_type="dataset"))
    per_type = -(-N_TWOHOP // df["fact_comp_type"].nunique())
    sample = (df.sample(frac=1, random_state=0).groupby("fact_comp_type").head(per_type)
              .sample(frac=1, random_state=0).head(n))
    tok = transformers.AutoTokenizer.from_pretrained(MODELS[0])
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODELS[0], dtype=torch.bfloat16, device_map="cuda").eval()
    for _, r in sample.iterrows():
        for label, col, gold in (("hop1", "r1(e1).prompt", "e2.value"), ("two ", "r2(r1(e1)).prompt", "e3.value")):
            ids = tok(prefix + r[col], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                top = hf(input_ids=ids, use_cache=False).logits[0, -1].float().topk(3).indices.tolist()
                cont = tok.decode(hf.generate(input_ids=ids, max_new_tokens=8, do_sample=False)[0][ids.shape[1]:])
            print(f"{label} | gold {r[gold]!r:<28} | top3 {[tok.decode([t]) for t in top]} | continues {cont!r} | {r[col][-70:]}", flush=True)


@app.local_entrypoint()
def main(peek_n: int = 0, prefix: str = ""):
    if peek_n:
        return peek.remote(peek_n, prefix)
    RESULTS.write_text(json.dumps(run.remote(), indent=1))
    print("wrote", RESULTS)
