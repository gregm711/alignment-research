"""Frozen HF decoder for next-token A/B ranking on native chat prompts."""
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINTS = Path.home() / ".cache/alignment-research/fork-writer-checkpoints"


class Reader:
    def __init__(self, model_id, revision, batch_size=4, device=None):
        if not re.fullmatch("[a-f0-9]{40}", revision):
            raise ValueError("Supply an immutable Hugging Face revision")
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        self.batch_size = batch_size
        local = CHECKPOINTS / model_id / revision
        source = str(local) if (local / "config.json").is_file() else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.model, loading = AutoModelForCausalLM.from_pretrained(
            source, revision=revision, dtype=self.dtype, output_loading_info=True,
        )
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incompatible checkpoint {model_id}: {loading}")
        self.model.requires_grad_(False).eval().to(self.device)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        answers = self.tokenizer(["A", "B"], add_special_tokens=False).input_ids
        if any(len(ids) != 1 for ids in answers) or answers[0] == answers[1]:
            raise ValueError(f"A and B must each be distinct single tokens: {answers}")
        self.answer_ids = [ids[0] for ids in answers]
        self.elapsed = 0.

    def _tokens(self, prompts):
        text = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts]
        return self.tokenizer(text, padding=True, return_tensors="pt",
                              add_special_tokens=False).to(self.device)

    def _scores(self, tokens, embeddings=None):
        # Right padding avoids changing the causal/convolutional prefix of each example.
        last = tokens.attention_mask.sum(dim=1) - 1
        positions, selection = torch.unique(last, sorted=True, return_inverse=True)
        kwargs = {"input_ids": tokens.input_ids} if embeddings is None else {"inputs_embeds": embeddings}
        logits = self.model(
            **kwargs, attention_mask=tokens.attention_mask, logits_to_keep=positions, use_cache=False,
        ).logits
        return logits[torch.arange(len(last), device=self.device), selection][:, self.answer_ids].float()

    @torch.inference_mode()
    def predict_many(self, prompts):
        start = time.perf_counter()
        answers = []
        for i in range(0, len(prompts), self.batch_size):
            tokens = self._tokens(prompts[i:i+self.batch_size])
            choices = self._scores(tokens).argmax(dim=1).tolist()
            answers.extend("AB"[j] for j in choices)
        self.elapsed = time.perf_counter() - start
        return answers

    def gradient_check(self):
        """Differentiate the final A/B margin with respect to a single early memory vector."""
        tokens = self._tokens(["The record says A. Which label is in the record? Answer A or B."])
        embeds = self.model.get_input_embeddings()(tokens.input_ids).detach()
        index = embeds.shape[1] // 3
        memory = embeds[:, index:index+1].clone().requires_grad_(True)
        joined = torch.cat((embeds[:, :index], memory, embeds[:, index+1:]), dim=1)
        scores = self._scores(tokens, joined)
        gradient, = torch.autograd.grad(scores[0, 0] - scores[0, 1], memory)
        norm = gradient.float().norm().item()
        assert 0 < norm < float("inf") and not any(p.grad is not None for p in self.model.parameters())
        return norm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify frozen readout and differentiability")
    parser.add_argument("model")
    parser.add_argument("revision")
    parser.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args()
    reader = Reader(args.model, args.revision, device=args.device)
    queries = ["Return exactly this character: A", "The destination is B. Output just its letter, A or B."]
    batch = reader.predict_many(queries)
    assert batch == [reader.predict_many([query])[0] for query in queries]
    print(reader.device, batch, "input gradient norm", reader.gradient_check())
