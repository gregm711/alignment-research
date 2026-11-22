"""Does the frozen attention readout catch wrong routes that have a different, untargeted cause?

    modal run causes.py --pilot      # two table sets: how many hidden wrong routes does each disturbance give?
    modal run causes.py              # writes results/causes.json

Disturbances are upstream of the final pass and not aimed at any row: random noise in the messages into
pass 2, random noise in the restart state, or an arbitrary other prompt's messages into pass 2. The
readout (head 23 at L11 of the final pass, threshold 0.45) is frozen from oversight2.py.
"""
import json
import pathlib
import random

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
HEAD_CALL, HEAD, THRESHOLD = 35, 23, 0.45085132122039795
SETS = [("so", 70, 8), ("qa", 71, 8)]
CAUSES = ["none", "noise_messages_pass2_x1", "noise_messages_pass2_x2", "noise_restart_state_x2", "transplant_pass2_a", "transplant_pass2_b"]

app = modal.App("jlens-inkling-causes")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("oversight")
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=70 * 60)
def run(pilot: bool = False):
    import time
    import torch, transformers
    from huggingface_hub import model_info
    from oversight import groups

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="eager").eval()
    text = hf.model.language_model
    text.config.loop_num_passes = 3
    st = {"mode": None, "i": 0, "saved": [], "src": None, "alpha": 0.0, "seed": 0}
    attns = []

    def bump(x, salt):  # unit-Gaussian noise scaled to alpha times each vector's own norm; same seed -> same noise
        g = torch.Generator(device="cpu").manual_seed(st["seed"] * 1000 + salt)
        d = torch.randn(x.shape, generator=g).to(x.device, torch.float32)
        xf = x.float()
        return (xf + st["alpha"] * xf.norm(dim=-1, keepdim=True) * d / d.norm(dim=-1, keepdim=True)).to(x.dtype)

    def on_message(mod, args, output):  # calls 0-11 feed pass 2; the final pass (calls 12-23) is never touched
        i, st["i"] = st["i"], st["i"] + 1
        if st["mode"] == "record":
            st["saved"].append(output.detach().cpu())
        elif i < 12 and st["mode"] == "transplant":
            return st["src"][i].to(output.device)
        elif i < 12 and st["mode"] == "noise_messages":
            return bump(output, 1 + i)

    def on_restart(mod, args, output):  # output of the last pre-loop layer: what every pass restarts from
        if st["mode"] == "noise_restart":
            return bump(output, 0)

    text.message_input.register_forward_hook(on_message)
    text.layers[text.config.loop_layer_index_start - 1].register_forward_hook(on_restart)
    for layer in text.layers:
        layer.self_attn.register_forward_hook(lambda mod, args, output: attns.append(output[1][0, :, -1].float().cpu()))

    def forward(prompt, mode=None, src=None, alpha=0.0, seed=0):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        st.update(mode=mode, i=0, saved=[], src=src, alpha=alpha, seed=seed)
        attns.clear()
        with torch.no_grad():
            probs = torch.softmax(hf(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
        assert st["i"] == 24 and len(attns) == 48
        return {"top": tok.decode([int(probs.argmax())]).strip(), "p_top": float(probs.max()), "n": ids.shape[1],
                "msgs": st["saved"], "attn": attns[HEAD_CALL][HEAD]}

    rows, rng, trial = [], random.Random(0), 0
    for wording, seed, n in SETS:
        for items in groups(wording, seed, 2 if pilot else n):
            clean = {id(it): forward(it["prompt"], "record") for it in items}
            for it in items:
                if it["other_row"] is None or clean[id(it)]["top"] != it["answer"]:
                    continue
                true_entry = it["row_entries"][it["true_row"]]
                edit_intended = it["prompt"].replace(true_entry, true_entry.rsplit(" ", 1)[0] + " " + it["spare"])
                assert it["prompt"].count(true_entry) == 1
                offs = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
                start, spans = it["prompt"].index("Fruits for colors:"), []
                for entry in it["row_entries"]:
                    a = it["prompt"].index(entry, start)
                    spans.append([i for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a])
                donors = [d for d in items if d["side"] != it["side"] and clean[id(d)]["n"] == clean[id(it)]["n"]]
                picks = rng.sample(donors, 2) if len(donors) >= 2 else donors * 2
                for cause in CAUSES:
                    trial += 1
                    kw = ({"mode": None} if cause == "none" else
                          {"mode": "noise_messages", "alpha": float(cause[-1]), "seed": trial} if cause.startswith("noise_messages") else
                          {"mode": "noise_restart", "alpha": 2.0, "seed": trial} if cause.startswith("noise_restart") else
                          {"mode": "transplant", "src": clean[id(picks[cause.endswith("_b")])]["msgs"]})
                    out = forward(it["prompt"], **kw)
                    per_row = [float(out["attn"][sp].sum()) for sp in spans]
                    total = max(sum(per_row), 1e-9)
                    row = {"cause": cause, "wording": wording, "prompt": it["prompt"], "answer": it["answer"], "top": out["top"],
                           "p_top": out["p_top"], "off_correct_row": 1 - per_row[it["true_row"]] / total,
                           "share_other_row": per_row[it["other_row"]] / total}
                    if out["top"] != it["answer"]:
                        row["route"] = "answer_wrong"
                    else:  # the same disturbance (same donor, same noise) on the two edited prompts
                        a_int, a_oth = forward(edit_intended, **kw)["top"], forward(it["twin_prompt"], **kw)["top"]
                        row.update(edit_intended=a_int, edit_other=a_oth)
                        fi, fo = a_int == it["spare"], a_oth == it["spare"]
                        row["route"] = "right_route" if fi and not fo else "wrong_route" if fo and not fi else "unclear"
                    rows.append(row)

    def auc(pos, neg):
        return None if not pos or not neg else sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    def boot(pos, neg, n=1000):
        if len(pos) < 5 or len(neg) < 5:
            return None
        r, vals = random.Random(0), []
        for _ in range(n):
            vals.append(auc(r.choices(pos, k=len(pos)), r.choices(neg, k=len(neg))))
        vals.sort()
        return [auc(pos, neg), vals[25], vals[975]]

    summary = {}
    for cause in CAUSES:
        sub = [r for r in rows if r["cause"] == cause]
        routes = {k: sum(r["route"] == k for r in sub) for k in ("right_route", "wrong_route", "unclear", "answer_wrong")}
        wrong, right = [r for r in sub if r["route"] == "wrong_route"], [r for r in sub if r["route"] == "right_route"]
        summary[cause] = {"n": len(sub), "routes": routes,
                          "frozen_head_auc": boot([r["off_correct_row"] for r in wrong], [r["off_correct_row"] for r in right]),
                          "confidence_auc": boot([-r["p_top"] for r in wrong], [-r["p_top"] for r in right]),
                          "flagged": {k: f"{sum(r['off_correct_row'] > THRESHOLD for r in sub if r['route'] == k)}/{v}" for k, v in routes.items()}}
        fmt = lambda v: "n/a" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
        print(f"{cause:<26} {routes} | head {fmt(summary[cause]['frozen_head_auc'])} | confidence {fmt(summary[cause]['confidence_auc'])} | "
              f"flagged {summary[cause]['flagged']}", flush=True)
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "summary": summary, "rows": rows, "gpu_minutes": (time.time() - t0) / 60}
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(pilot: bool = False):
    out = run.remote(pilot)
    name = "causes-pilot.json" if pilot else "causes.json"
    (pathlib.Path(__file__).parent / "results" / name).write_text(json.dumps(out, indent=1))
    print("wrote", name)
