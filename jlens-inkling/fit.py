"""Fit Jacobian lenses on Inkling 2 Mini (looped transformer), one readout per loop pass.

    modal run fit.py --minutes 12                       # 3-pass lens on Inkling 2 Mini
    modal run fit.py --minutes 0                        # just print readouts from the checkpoint
    modal run fit.py --model thinkingmachines/TML-d24-varloop-research --passes 1,2,3 --prompts 4

`--passes k` runs the model with only k loop passes and aims the lens at that run's output, so
pass k is read through the path the model itself uses when it stops there. A lens aimed at the
3-pass output cannot read passes 1 and 2 (transport dies at the pass boundary; Wang & Reid 2026).

Re-running resumes from the checkpoint on the Modal volume, so spend is incremental.
"""
import modal

DEFAULT_MODEL = "thinkingmachines/Inkling-2-Mini"

app = modal.App("jlens-inkling")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer", "datasets")
    # Inkling 2 Mini support: huggingface/transformers#51050, pinned to its merge commit.
    .pip_install("git+https://github.com/huggingface/transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3")
    .pip_install("git+https://github.com/anthropics/jacobian-lens")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# (prompt, intermediate concept the model must pass through, final answer)
PROBES = [
    ("Fact: The currency used in the country shaped like a boot is the", "Italy", "euro"),
    ("Fact: The number of legs on the animal that spins webs is", "spider", "eight"),
    ("Fact: The capital of the country where the Eiffel Tower stands is", "France", "Paris"),
    ("Fact: The language spoken in the country whose capital is Tokyo is", "Japan", "Japanese"),
    ("Fact: The colour you get by mixing the colour of the sky with the colour of a lemon is", "yellow", "green"),
    ("Fact: The planet named after the Roman god of war has this many moons:", "Mars", "two"),
]


class PassView:
    """One loop pass of a looped layer: forwards hook registration but fires only on call number `k`."""

    def __init__(self, module, k, counts):
        self.module, self.k, self.counts = module, k, counts

    def register_forward_hook(self, fn):
        def only_this_pass(mod, inputs, output):
            if self.counts[mod] == self.k + 1:  # the counting hook below has already run
                return fn(mod, inputs, output)

        return self.module.register_forward_hook(only_this_pass)


class InklingLens:
    """jlens LensModel over Inkling 2 Mini. `layers` is the 49 steps a token goes through:
    6 pre-loop layers, 12 loop layers x 3 passes, 6 post-loop layers, then the final mix (the lens target)."""

    def __init__(self, hf, tokenizer):
        import collections

        self.hf, self.tokenizer = hf, tokenizer
        self.text = text = hf.model.language_model
        cfg = text.config
        a, b, passes = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
        hf.eval()
        for p in hf.parameters():
            p.requires_grad_(False)

        self._counts = collections.Counter()
        self._handles = [text.register_forward_pre_hook(lambda mod, args: self._counts.clear())]
        for layer in text.layers[a:b]:
            self._handles.append(layer.register_forward_hook(lambda mod, i, o: self._counts.update([mod])))

        self.layers, self.names = [], []
        for i in range(a):
            self.layers.append(text.layers[i]); self.names.append(f"L{i}")
        for k in range(passes):
            for i in range(a, b):
                self.layers.append(PassView(text.layers[i], k, self._counts)); self.names.append(f"L{i}.pass{k + 1}")
        for i in range(b, cfg.num_hidden_layers):
            self.layers.append(text.layers[i]); self.names.append(f"L{i}")
        self.layers.append(text.final_input); self.names.append("final")
        self.n_layers, self.d_model = len(self.layers), cfg.hidden_size

    def close(self):
        for h in self._handles:
            h.remove()

    def encode(self, text, *, max_length=512):
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        return ids.to(self.text.embed_tokens.weight.device)

    def forward(self, input_ids):
        return self.text(input_ids=input_ids, use_cache=False)

    def unembed(self, residual):
        w = self.hf.lm_head.weight
        return self.hf.logit_softcap(self.hf.lm_head(self.text.norm(residual.to(w.dtype).to(w.device))).float())


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=90 * 60)  # timeout is the spend cap per run
def run(model_name: str, passes: list[int], minutes: float, max_prompts: int):
    import gc, json, logging, os, time
    import torch, transformers, jlens
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    cfg = hf.model.language_model.config
    vol.commit()
    print(f"loaded {model_name} in {time.time() - t0:.0f}s | VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB | "
          f"transformers {transformers.__version__}", flush=True)
    text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    prompts = [t.strip() for t in text if len(t) > 900][:max_prompts]  # long paragraphs -> full 128 tokens
    all_results = {}

    for k in passes:
        cfg.loop_num_passes = k
        model = InklingLens(hf, tok)
        tag = f"{model_name.split('/')[-1]}-p{k}"
        CKPT, LENS = f"/cache/{tag}_fit_ckpt.pt", f"/cache/{tag}_jacobian_lens.pt"
        if tag == "Inkling-2-Mini-p3" and os.path.exists("/cache/fit_ckpt.pt") and not os.path.exists(CKPT):
            os.rename("/cache/fit_ckpt.pt", CKPT)  # the first run's checkpoint, saved before files were tagged
        print(f"\n######## {tag}: {model.n_layers} steps", flush=True)

        # Sanity: the model answers, every step is seen once per forward, and passes differ.
        ids = model.encode(PROBES[0][0])
        with torch.no_grad():
            print("greedy:", repr(tok.decode(hf.generate(input_ids=ids, max_new_tokens=8, do_sample=False)[0][ids.shape[1]:])))
            with jlens.hooks.ActivationRecorder(model.layers, at=range(model.n_layers)) as rec:
                model.forward(ids)
            assert len(rec.activations) == model.n_layers, len(rec.activations)
            if k > 1:  # L6 in pass 1 vs L6 in the last pass
                assert not torch.equal(rec.activations[6], rec.activations[6 + 12 * (k - 1)]), "PassView is broken"

        # Fit one prompt at a time until the time budget for this pass count is spent; each prompt is committed.
        done = torch.load(CKPT, map_location="cpu", weights_only=True)["next_idx"] if os.path.exists(CKPT) else 0
        dim_batch, deadline = 16, time.time() + minutes * 60 / len(passes)
        while minutes > 0 and done < len(prompts):
            t1 = time.time()
            try:
                jlens.fit(model, prompts[: done + 1], dim_batch=dim_batch, checkpoint_path=CKPT, checkpoint_every=None)
            except torch.OutOfMemoryError:
                if dim_batch == 4:
                    raise
                dim_batch //= 2
                gc.collect(); torch.cuda.empty_cache()
                print(f"out of memory, retrying with dim_batch={dim_batch}", flush=True)
                continue
            done += 1
            vol.commit()
            per_prompt = time.time() - t1
            print(f"== {tag}: {done}/{len(prompts)} prompts | {per_prompt:.0f}s per prompt | dim_batch {dim_batch} | "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if time.time() + per_prompt > deadline:
                break

        state = torch.load(CKPT, map_location="cpu", weights_only=True)
        lens = jlens.JacobianLens({l: J / state["n_done"] for l, J in state["jacobian_sum"].items()},
                                  n_prompts=state["n_done"], d_model=model.d_model)
        lens.save(LENS)

        # Readouts at the last token: where do the intermediate concept and the answer become readable?
        def rank(logits, word):
            # Single-token spellings only: a word that splits shares its first piece (e.g. a bare
            # space) with unrelated words, which made the first run's spider/eight ranks meaningless.
            variants = {f"{sp}{w}" for sp in ("", " ") for w in (word, word.lower(), word.capitalize())}
            cands = [ids[0] for v in variants if len(ids := tok.encode(v, add_special_tokens=False)) == 1]
            if not cands:
                return None
            order = logits.argsort(descending=True)
            return min(int((order == c).nonzero()[0, 0]) for c in cands)

        results = []
        for prompt, mid, ans in PROBES:
            jl, final, _ = lens.apply(model, prompt, positions=[-1])
            ll, _, _ = lens.apply(model, prompt, positions=[-1], use_jacobian=False)
            rows = [{"step": model.names[l], "top": [tok.decode([t]) for t in jl[l][0].topk(5).indices.tolist()],
                     "mid_rank": rank(jl[l][0], mid), "ans_rank": rank(jl[l][0], ans),
                     "logitlens_mid_rank": rank(ll[l][0], mid), "logitlens_ans_rank": rank(ll[l][0], ans)}
                    for l in lens.source_layers]
            results.append({"prompt": prompt, "mid": mid, "ans": ans, "rows": rows,
                            "model_top": [tok.decode([t]) for t in final[0].topk(5).indices.tolist()]})
            print(f"\n### {prompt}\n    intermediate={mid!r} answer={ans!r} | model says {results[-1]['model_top']}")
            print(f"    {'step':<11}{'mid':>7}{'ans':>7}  {'(logit lens: mid, ans)':<24} top-5 lens tokens")
            for r in rows:
                print(f"    {r['step']:<11}{str(r['mid_rank']):>7}{str(r['ans_rank']):>7}  "
                      f"({str(r['logitlens_mid_rank']):>6},{str(r['logitlens_ans_rank']):>6}){'':<8} {r['top']}")
        all_results[tag] = {"n_prompts": lens.n_prompts, "results": results}
        print(f"\n{tag}: lens fitted on {lens.n_prompts} prompts", flush=True)
        model.close()
        del lens, state
        gc.collect(); torch.cuda.empty_cache()

    vol.commit()
    all_results["gpu_minutes"] = (time.time() - t0) / 60
    print(f"\ntotal {all_results['gpu_minutes']:.1f} min on GPU", flush=True)
    return all_results


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL, passes: str = "3", minutes: float = 12, prompts: int = 100, out: str = ""):
    import json, pathlib

    res = run.remote(model, [int(k) for k in passes.split(",")], minutes, prompts)
    if out:
        path = pathlib.Path(__file__).parent / "results" / out
        path.write_text(json.dumps(res, indent=1))
        print("wrote", path)
