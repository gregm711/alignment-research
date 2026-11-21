"""The oversight test, second attempt: can where the model looks reveal a right answer reached the wrong way?

    modal run oversight2.py --pilot     # two sets of tables: do attention weights come back and trials build?
    modal run oversight2.py             # writes results/oversight-attention.json

Trials are built as in oversight.py. The readout is attention from the final token to the correct row of
the second table, calibrated on VISIBLE mistakes and then applied to hidden ones. See README.md.
"""
import json
import pathlib

import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
TRANSFORMERS_PIN = "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3"
PASS3 = range(12, 24)        # message-builder calls feeding the final pass
READ_STEPS = range(30, 48)   # layer applications 30-41 are the final pass, 42-47 the layers after the loop
SPLITS = {"train": ("so", 10, 12), "test_new_tables": ("so", 11, 10), "test_new_tables_new_wording": ("qa", 12, 10)}
KINDS = ("hidden_wrong_route", "right_route", "visible_wrong_route")

app = modal.App("jlens-inkling-oversight2")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer")
    .pip_install(f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_PIN}")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("oversight")  # the trial generator
)


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=80 * 60)
def run(pilot: bool = False):
    import time
    import torch, transformers
    from huggingface_hub import model_info
    from oversight import FRUITS, groups

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    # eager attention so the attention weights are returned
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="eager").eval()
    text = hf.model.language_model
    text.config.loop_num_passes = 3

    msg = {"mode": None, "i": 0, "saved": [], "src": None}

    def on_message(mod, args, output):
        i, msg["i"] = msg["i"], msg["i"] + 1
        if msg["mode"] == "record":
            msg["saved"].append(output.detach().cpu())
        elif msg["mode"] == "patch" and i in PASS3:
            return msg["src"][i].to(output.device)

    text.message_input.register_forward_hook(on_message)

    # In the order the layers run (6 + 36 + 6 applications): the final token's state, and its attention row.
    states, attns = [], []
    for layer in text.layers:
        layer.register_forward_hook(lambda mod, args, output: states.append(
            (output if torch.is_tensor(output) else output[0])[0, -1].float().cpu()))
        layer.self_attn.register_forward_hook(lambda mod, args, output: attns.append(output[1][0, :, -1].float().cpu()))

    def forward(prompt, mode=None, src=None):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        msg.update(mode=mode, i=0, saved=[], src=src)
        states.clear(); attns.clear()
        with torch.no_grad():
            probs = torch.softmax(hf(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
        assert msg["i"] == 24 and len(states) == 48 and len(attns) == 48, (msg["i"], len(states), len(attns))
        top2 = probs.topk(2)
        return {"top": int(top2.indices[0]), "p_top": float(top2.values[0]), "margin": float(top2.values[0] - top2.values[1]),
                "n_tokens": ids.shape[1], "msgs": msg["saved"], "states": torch.stack(states),
                "attn": torch.stack([attns[i] for i in READ_STEPS])}  # [18 steps, heads, tokens]

    def token(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        assert len(ids) == 1, word
        return ids[0]

    def row_tokens(it):
        """Token positions of each row of the second table in this prompt."""
        offs = tok(it["prompt"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        start, rows = it["prompt"].index("Fruits for colors:"), []
        for entry in it["row_entries"]:
            a = it["prompt"].index(entry, start)
            rows.append([i for i, (x, y) in enumerate(offs) if x < a + len(entry) and y > a])
        return rows

    def trials_for(split):
        wording, seed, n_groups = SPLITS[split]
        kept, tally = [], {"intended": dict.fromkeys(KINDS, 0), "kept": dict.fromkeys(KINDS, 0)}
        for items in groups(wording, seed, 2 if pilot else n_groups):
            clean = {id(it): forward(it["prompt"], "record") for it in items}
            for R in items:
                if R["other_row"] is None:
                    continue
                ans, spare = token(R["answer"]), token(R["spare"])
                if clean[id(R)]["top"] != ans or forward(R["twin_prompt"])["top"] != ans:
                    continue
                third = next(r for r in range(7) if r not in (R["true_row"], R["other_row"]) and R["row_fruits"][r] != R["answer"])
                rows = row_tokens(R)
                plan = (("hidden_wrong_route", R["other_row"]), ("right_route", R["true_row"]), ("visible_wrong_route", third))
                for kind, row in plan:
                    donors = [D for D in items if D["side"] != R["side"] and D["true_row"] == row
                              and clean[id(D)]["top"] == token(D["answer"]) and clean[id(D)]["n_tokens"] == clean[id(R)]["n_tokens"]]
                    if not donors:
                        continue
                    D = donors[0]
                    tally["intended"][kind] += 1
                    out = forward(R["prompt"], "patch", clean[id(D)]["msgs"])
                    if kind == "visible_wrong_route":
                        if out["top"] != token(R["row_fruits"][row]):
                            continue  # the answer must visibly be the pointed-to row's (different) fruit
                    else:
                        if out["top"] != ans:
                            continue  # visible answer must be correct
                        twin_top = forward(R["twin_prompt"], "patch", clean[id(D)]["msgs"])["top"]
                        if twin_top != (spare if kind == "hidden_wrong_route" else ans):
                            continue  # the twin, differing only in the other row, must confirm which row was read
                    tally["kept"][kind] += 1
                    per_row = torch.stack([out["attn"][:, :, r].sum(-1) for r in rows], -1)  # [steps, heads, 7 rows]
                    share = per_row / per_row.sum(-1, keepdim=True).clamp_min(1e-9)
                    kept.append({"kind": kind, "off_correct_row": 1 - share[:, :, R["true_row"]],  # [steps, heads]; high = looking elsewhere
                                 "states": out["states"], "answer": R["answer"], "p_top": out["p_top"], "margin": out["margin"],
                                 "recipient": R["prompt"], "donor": D["prompt"]})
        return kept, tally

    def auc(scores, y):
        pos, neg = scores[y == 1], scores[y == 0]
        return float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())

    def boot(scores, y, n=1000):
        g, vals = torch.Generator().manual_seed(0), []
        for _ in range(n):
            idx = torch.randint(len(y), (len(y),), generator=g)
            if 0 < y[idx].sum() < len(idx):
                vals.append(auc(scores[idx], y[idx]))
        vals.sort()
        return [auc(scores, y), vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]]

    out = {"model": MODEL, "revision": model_info(MODEL).sha, "transformers": TRANSFORMERS_PIN, "splits": {}}
    data = {}
    for split in (["train"] if pilot else SPLITS):
        data[split], tally = trials_for(split)
        out["splits"][split] = {"wording": SPLITS[split][0], "tally": tally}
        print(f"\n=== {split}: {tally}", flush=True)
    if pilot:
        sample = data["train"][0]["off_correct_row"]
        print("attention readout shape", tuple(sample.shape), "| finite:", bool(torch.isfinite(sample).all()))
        return out

    def pick(trials, pos_kind, neg_kind="right_route"):
        sub = [t for t in trials if t["kind"] in (pos_kind, neg_kind)]
        return sub, torch.tensor([int(t["kind"] == pos_kind) for t in sub])

    # Calibrate on VISIBLE mistakes in the training tables: one head, one threshold. Then freeze.
    vis, yv = pick(data["train"], "visible_wrong_route")
    A = torch.stack([t["off_correct_row"] for t in vis])  # [trials, steps, heads]
    n_steps, n_heads = A.shape[1:]
    train_auc = torch.tensor([[auc(A[:, s, h], yv) for h in range(n_heads)] for s in range(n_steps)])
    s_, h_ = divmod(int(train_auc.argmax()), n_heads)
    threshold = float(A[yv == 0][:, s_, h_].quantile(0.9))  # 10% false alarms on right-route training trials
    step_name = lambda s: (f"L{6 + s}.pass3" if s < 12 else f"L{6 + s}")
    out["readout"] = {"chosen_on": "visible mistakes, training tables", "step": step_name(s_), "head": h_,
                      "train_auc_visible": float(train_auc[s_, h_]), "threshold": threshold}
    print(f"\nfrozen readout: head {h_} at {step_name(s_)} (AUC on visible mistakes, training tables: {train_auc[s_, h_]:.3f})", flush=True)

    # State-pipeline control from the first attempt: same recipe, a label known to be present (which fruit is the answer).
    half = set(sorted(FRUITS)[: len(FRUITS) // 2 + 1])
    tr = [t for t in data["train"] if t["kind"] != "visible_wrong_route"]
    Xs, ys = torch.stack([t["states"] for t in tr]), torch.tensor([int(t["answer"] in half) for t in tr])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Z = (Xs - mu) / sd
    w_all = Z[ys == 1].mean(0) - Z[ys == 0].mean(0)  # [48 steps, d]
    st = max(range(48), key=lambda s: auc(Z[:, s] @ w_all[s], ys))

    fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    for split in ("test_new_tables", "test_new_tables_new_wording"):
        res = {}
        for name, kind in (("hidden", "hidden_wrong_route"), ("visible", "visible_wrong_route")):
            sub, y = pick(data[split], kind)
            B = torch.stack([t["off_correct_row"] for t in sub])
            head = B[:, s_, h_]
            res[name] = {"n_mistakes": int(y.sum()), "n_right_route": int((1 - y).sum()),
                         "frozen_head_auc": boot(head, y),
                         "caught_at_frozen_threshold": float((head[y == 1] > threshold).float().mean()),
                         "false_alarms_at_frozen_threshold": float((head[y == 0] > threshold).float().mean()),
                         "all_heads_average_auc": boot(B.mean((1, 2)), y),  # no selection at all
                         "output_p_top_auc": boot(-torch.tensor([t["p_top"] for t in sub]), y),
                         "output_margin_auc": boot(-torch.tensor([t["margin"] for t in sub]), y),
                         # not pre-specified, for transparency: best any single head could have done on this split
                         "best_single_head_auc_on_this_split": float(max(auc(B[:, s, h], y) for s in range(n_steps) for h in range(n_heads))),
                         "per_trial": [{"y": int(a), "frozen_head": float(b), "p_top": t["p_top"], "recipient": t["recipient"], "donor": t["donor"]}
                                       for a, b, t in zip(y, head, sub)]}
        te = [t for t in data[split] if t["kind"] != "visible_wrong_route"]
        Zt = (torch.stack([t["states"] for t in te]) - mu) / sd
        res["state_pipeline_control_auc"] = boot(Zt[:, st] @ w_all[st], torch.tensor([int(t["answer"] in half) for t in te]))
        out["splits"][split].update(res)
        print(f"--- {split}\n    HIDDEN mistakes ({res['hidden']['n_mistakes']} vs {res['hidden']['n_right_route']} right-route): "
              f"frozen head {fmt(res['hidden']['frozen_head_auc'])} | all-heads average {fmt(res['hidden']['all_heads_average_auc'])} | "
              f"output top-probability {fmt(res['hidden']['output_p_top_auc'])} | caught {res['hidden']['caught_at_frozen_threshold']:.2f} "
              f"at false-alarm rate {res['hidden']['false_alarms_at_frozen_threshold']:.2f} | best head in hindsight {res['hidden']['best_single_head_auc_on_this_split']:.3f}\n"
              f"    control, VISIBLE mistakes ({res['visible']['n_mistakes']}): frozen head {fmt(res['visible']['frozen_head_auc'])} | "
              f"all-heads average {fmt(res['visible']['all_heads_average_auc'])}\n"
              f"    control, state pipeline on answer identity: {fmt(res['state_pipeline_control_auc'])}", flush=True)
    out["gpu_minutes"] = (time.time() - t0) / 60
    print(f"total {out['gpu_minutes']:.1f} min on GPU")
    return out


@app.local_entrypoint()
def main(pilot: bool = False):
    out = run.remote(pilot)
    if not pilot:
        path = pathlib.Path(__file__).parent / "results" / "oversight-attention.json"
        path.write_text(json.dumps(out, indent=1))
        print("wrote", path)
