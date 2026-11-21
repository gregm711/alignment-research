"""Fit a Jacobian lens on Inkling 2 Mini (looped transformer), one readout per loop pass.

    modal run fit.py --minutes 12      # fit for ~12 min, checkpoint, print readouts
    modal run fit.py --minutes 0       # no fitting: just print readouts from the checkpoint

Re-running resumes from the checkpoint on the Modal volume, so spend is incremental.
"""
import modal

MODEL = "thinkingmachines/Inkling-2-Mini"
CKPT, LENS, RESULTS = "/cache/fit_ckpt.pt", "/cache/jacobian_lens.pt", "/cache/readouts.json"

app = modal.App("jlens-inkling")
vol = modal.Volume.from_name("jlens-inkling-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "accelerate", "safetensors", "tiktoken", "sentencepiece", "pillow", "hf_transfer", "datasets")
    # Inkling 2 Mini support is an open PR on Hugging Face's own branch (huggingface/transformers#51050).
    .pip_install("git+https://github.com/huggingface/transformers@add-inkling2-mini")
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
        text.register_forward_pre_hook(lambda mod, args: self._counts.clear())
        for layer in text.layers[a:b]:
            layer.register_forward_hook(lambda mod, i, o: self._counts.update([mod]))

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

    def encode(self, text, *, max_length=512):
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        return ids.to(self.text.embed_tokens.weight.device)

    def forward(self, input_ids):
        return self.text(input_ids=input_ids, use_cache=False)

    def unembed(self, residual):
        w = self.hf.lm_head.weight
        return self.hf.logit_softcap(self.hf.lm_head(self.text.norm(residual.to(w.dtype).to(w.device))).float())


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=65536, volumes={"/cache": vol}, timeout=90 * 60)  # timeout is the spend cap per run
def run(minutes: float, max_prompts: int, chunk: int):
    import gc, json, logging, os, time
    import torch, transformers, jlens
    from datasets import load_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = InklingLens(hf, tok)
    vol.commit()
    print(f"loaded in {time.time() - t0:.0f}s | VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB | "
          f"{model.n_layers} steps | transformers {transformers.__version__}", flush=True)

    # Sanity: the model answers, and every one of the 49 steps is seen exactly once per forward.
    ids = model.encode(PROBES[0][0])
    print("tokens:", [tok.decode([t]) for t in ids[0].tolist()][:8], "...")
    with torch.no_grad():
        print("greedy:", repr(tok.decode(hf.generate(input_ids=ids, max_new_tokens=8, do_sample=False)[0][ids.shape[1]:])))
        with jlens.hooks.ActivationRecorder(model.layers, at=range(model.n_layers)) as rec:
            model.forward(ids)
        assert len(rec.activations) == model.n_layers, len(rec.activations)
        p1, p3 = rec.activations[6], rec.activations[6 + 24]  # L6 pass 1 vs pass 3
        assert not torch.equal(p1, p3), "loop passes returned identical activations: PassView is broken"

    # Fit in chunks until the time budget is spent; every chunk is committed, so a kill loses little.
    if minutes > 0:
        text = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
        prompts = [t.strip() for t in text if len(t) > 900][:max_prompts]  # long paragraphs -> full 128 tokens
        dim_batch, deadline, done = 16, time.time() + minutes * 60, 0
        if os.path.exists(CKPT):
            done = torch.load(CKPT, map_location="cpu", weights_only=True)["next_idx"]
        while done < len(prompts):
            t1 = time.time()
            try:
                jlens.fit(model, prompts[: done + chunk], dim_batch=dim_batch, checkpoint_path=CKPT, checkpoint_every=None)
            except torch.OutOfMemoryError:
                if dim_batch == 4:
                    raise
                dim_batch //= 2
                gc.collect(); torch.cuda.empty_cache()
                print(f"out of memory, retrying with dim_batch={dim_batch}", flush=True)
                continue
            done = min(done + chunk, len(prompts))
            vol.commit()
            per_chunk = time.time() - t1
            print(f"== {done}/{len(prompts)} prompts | {per_chunk / chunk:.0f}s per prompt | dim_batch {dim_batch} | "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if time.time() + per_chunk > deadline:
                break

    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    lens = jlens.JacobianLens({l: J / state["n_done"] for l, J in state["jacobian_sum"].items()},
                              n_prompts=state["n_done"], d_model=model.d_model)
    lens.save(LENS)

    # Readouts at the last token: where do the intermediate concept and the answer become readable?
    def rank(logits, word):
        cands = {tok.encode(v, add_special_tokens=False)[0] for v in (word, " " + word, word.lower(), " " + word.lower(), word.capitalize(), " " + word.capitalize())}
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
            print(f"    {r['step']:<11}{r['mid_rank']:>7}{r['ans_rank']:>7}  ({r['logitlens_mid_rank']:>6},{r['logitlens_ans_rank']:>6})"
                  f"{'':<8} {r['top']}")
    json.dump({"n_prompts": lens.n_prompts, "results": results}, open(RESULTS, "w"))
    vol.commit()
    print(f"\nlens fitted on {lens.n_prompts} prompts | total {(time.time() - t0) / 60:.1f} min on GPU", flush=True)


@app.local_entrypoint()
def main(minutes: float = 12, max_prompts: int = 100, chunk: int = 1):
    run.remote(minutes, max_prompts, chunk)
