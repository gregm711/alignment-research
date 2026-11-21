"""Does the unmodified model ever take the wrong route by itself, and what does the frozen readout do on unmodified runs?

    modal run natural.py        # writes results/natural-routes.json

No transplants. The route is established by editing the intended row and, separately, the other row
holding the same fruit. The readout (head 23 at L11 of the final pass, threshold 0.45) is frozen from
oversight2.py and is not refitted here.
"""
import json
import pathlib

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
HEAD_CALL, HEAD, THRESHOLD = 35, 23, 0.45085132122039795  # L11.pass3 is the 36th layer application
SETS = [("so", 30, 10), ("qa", 31, 10)]

app = modal.App("jlens-inkling-natural")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("oversight")
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=30 * 60)
def run():
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
        return tok.decode([int(probs.argmax())]).strip(), float(probs.max()), attns[HEAD_CALL][HEAD]

    rows = []
    for wording, seed, n in SETS:
        for items in groups(wording, seed, n):
            for it in items:
                if it["other_row"] is None:
                    continue
                top, p_top, att = forward(it["prompt"])
                offs = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
                start, per_row = it["prompt"].index("Fruits for colors:"), []
                for entry in it["row_entries"]:
                    a = it["prompt"].index(entry, start)
                    per_row.append(float(sum(att[i] for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a)))
                total = max(sum(per_row), 1e-9)
                row = {"wording": wording, "prompt": it["prompt"], "answer": it["answer"], "top": top, "p_top": p_top,
                       "off_correct_row": 1 - per_row[it["true_row"]] / total, "share_other_row": per_row[it["other_row"]] / total,
                       "table2_attention": total}
                if top == it["answer"]:
                    true_entry = it["row_entries"][it["true_row"]]
                    twin_true = it["prompt"].replace(true_entry, true_entry.rsplit(" ", 1)[0] + " " + it["spare"])
                    assert twin_true != it["prompt"] and it["prompt"].count(true_entry) == 1
                    a_true, a_other = forward(twin_true)[0], forward(it["twin_prompt"])[0]
                    follows_true, follows_other = a_true == it["spare"], a_other == it["spare"]
                    row.update(edit_intended_row=a_true, edit_other_row=a_other,
                               route=("right_route" if follows_true and not follows_other else
                                      "wrong_route" if follows_other and not follows_true else
                                      "both_edits_change_it" if follows_true and follows_other else "neither_edit_changes_it"))
                else:
                    row["route"] = "answer_wrong"
                rows.append(row)
    counts = {}
    for r in rows:
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    right = [r for r in rows if r["route"] == "right_route"]
    out = {"model": MODEL, "revision": model_info(MODEL).sha, "counts": counts, "n_prompts": len(rows),
           "false_alarms_on_unmodified_right_route": f"{sum(r['off_correct_row'] > THRESHOLD for r in right)}/{len(right)}",
           "flagged_by_route": {k: f"{sum(r['off_correct_row'] > THRESHOLD for r in rows if r['route'] == k)}/{v}" for k, v in counts.items()},
           "rows": rows, "gpu_minutes": (time.time() - t0) / 60}
    print("routes:", counts)
    print("frozen readout flags, by route:", out["flagged_by_route"])
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main():
    out = run.remote()
    (pathlib.Path(__file__).parent / "results" / "natural-routes.json").write_text(json.dumps(out, indent=1))
    print("wrote results/natural-routes.json")
