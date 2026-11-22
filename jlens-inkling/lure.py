"""Mistakes the model makes unprompted: distract it through the prompt, then test the frozen readout.

    modal run lure.py --pilot                 # step 1: which distracting wording produces hidden wrong routes?
    modal run lure.py --wording NAME          # step 2: fresh tables, frozen readout vs output confidence

No internals are modified. The readout (head 23 at L11 of the final pass, threshold 0.45) is frozen
from oversight2.py and never refitted here.
"""
import json
import pathlib

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
HEAD_CALL, HEAD, THRESHOLD = 35, 23, 0.45085132122039795
PLAIN = "So the fruit that goes with the {q} is the"
WORDINGS = {"plain": PLAIN,
            "not_the": "So the fruit that goes with the {q}, not the {d}, is the",
            "mentioned": "Someone just asked about the {d}. So the fruit that goes with the {q} is the",
            "correction": "So the fruit that goes with the {d}, sorry, the {q}, is the",
            "next": "So the fruit that goes with the {q} (the {d} comes next) is the"}

app = modal.App("jlens-inkling-lure")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("oversight")
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=45 * 60)
def run(wordings: list[str], seed: int, n_groups: int):
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

    def readout(prompt, it, attn):
        """1 - share of the head's second-table attention on the correct row; also the share on the other row."""
        offs = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        start, per_row = prompt.index("Fruits for colors:"), []
        for entry in it["row_entries"]:
            a = prompt.index(entry, start)
            per_row.append(float(sum(attn[i] for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a)))
        total = max(sum(per_row), 1e-9)
        return 1 - per_row[it["true_row"]] / total, per_row[it["other_row"]] / total

    rows = []
    for items in groups("so", seed, n_groups):
        for it in items:
            if it["other_row"] is None:
                continue
            same_side = [x for x in items if x["side"] == it["side"]]
            by_row = {x["true_row"]: x["item"] for x in same_side}  # which animal's colour sits in each row
            third = next(r for r in range(7) if r not in (it["true_row"], it["other_row"]) and it["row_fruits"][r] != it["answer"])
            body = it["prompt"][: -len(PLAIN.format(q=it["item"]))]
            assert it["prompt"].endswith(PLAIN.format(q=it["item"]))
            true_entry = it["row_entries"][it["true_row"]]
            twin_true_body = body.replace(true_entry, true_entry.rsplit(" ", 1)[0] + " " + it["spare"])
            twin_other_body = it["twin_prompt"][: -len(PLAIN.format(q=it["item"]))]
            assert twin_true_body != body and twin_other_body != body
            for w in wordings:
                tail = WORDINGS[w].format(q=it["item"], d=by_row[it["other_row"]])
                out = forward(body + tail)
                off, other_share = readout(body + tail, it, out["attn"])
                row = {"wording": w, "prompt": body + tail, "answer": it["answer"], "top": out["top"], "p_top": out["p_top"],
                       "margin": out["margin"], "off_correct_row": off, "share_other_row": other_share}
                if out["top"] == it["answer"]:
                    a_true, a_other = forward(twin_true_body + tail)["top"], forward(twin_other_body + tail)["top"]
                    row.update(edit_intended_row=a_true, edit_other_row=a_other)
                    ft, fo = a_true == it["spare"], a_other == it["spare"]
                    row["route"] = ("right_route" if ft and not fo else "wrong_route" if fo and not ft
                                    else "both_edits_followed" if ft and fo else "neither_edit_followed")
                else:
                    row["route"] = "answer_wrong"
                # visible case: distract toward a row with a different fruit; does the answer become that fruit?
                if w != "plain":
                    vis = forward(body + WORDINGS[w].format(q=it["item"], d=by_row[third]))["top"]
                    row["visible_distraction"] = "followed" if vis == it["row_fruits"][third] else "correct" if vis == it["answer"] else "other"
                rows.append(row)

    def auc(pos, neg):
        if not pos or not neg:
            return None
        return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    summary = {}
    for w in wordings:
        sub = [r for r in rows if r["wording"] == w]
        routes = {k: sum(r["route"] == k for r in sub) for k in ("right_route", "wrong_route", "both_edits_followed", "neither_edit_followed", "answer_wrong")}
        wrong, right = [r for r in sub if r["route"] == "wrong_route"], [r for r in sub if r["route"] == "right_route"]
        summary[w] = {"n": len(sub), "routes": routes,
                      "visible_distraction": {k: sum(r.get("visible_distraction") == k for r in sub) for k in ("followed", "correct", "other")},
                      "frozen_head_auc": auc([r["off_correct_row"] for r in wrong], [r["off_correct_row"] for r in right]),
                      "confidence_auc": auc([-r["p_top"] for r in wrong], [-r["p_top"] for r in right]),
                      "flagged_wrong_route": f"{sum(r['off_correct_row'] > THRESHOLD for r in wrong)}/{len(wrong)}",
                      "flagged_right_route": f"{sum(r['off_correct_row'] > THRESHOLD for r in right)}/{len(right)}"}
        print(f"{w:<11} n={len(sub)} routes {routes} | visible {summary[w]['visible_distraction']} | head AUC {summary[w]['frozen_head_auc']} "
              f"conf AUC {summary[w]['confidence_auc']} | flagged wrong {summary[w]['flagged_wrong_route']} right {summary[w]['flagged_right_route']}", flush=True)
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "seed": seed, "n_groups": n_groups, "summary": summary,
           "rows": rows, "gpu_minutes": (time.time() - t0) / 60}
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(pilot: bool = False, wording: str = ""):
    here = pathlib.Path(__file__).parent / "results"
    if pilot:
        out = run.remote(list(WORDINGS), 40, 6)
        (here / "lure-pilot.json").write_text(json.dumps(out, indent=1))
    else:
        out = run.remote([wording], 41, 30)
        (here / "lure-test.json").write_text(json.dumps(out, indent=1))
    print("wrote results")
