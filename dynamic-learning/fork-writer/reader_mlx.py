"""Frozen, local 4-bit Qwen3-4B-Instruct-2507 readout for the fork screen."""
from pathlib import Path
import math

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load


MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
CHECKPOINT = Path.home() / ".cache/alignment-research/fork-writer-checkpoints" / REVISION


class Reader:
    def __init__(self, model_id, revision, batch_size=4):
        if (model_id, revision) != (MODEL, REVISION):
            raise ValueError("This readout expects the pinned Qwen3 4-bit checkpoint")
        self.model, self.tokenizer = load(str(CHECKPOINT))
        self.model.freeze()
        self.model.eval()
        assert not tree_flatten(self.model.trainable_parameters())
        answers = [self.tokenizer.encode(s, add_special_tokens=False) for s in "AB"]
        assert len(answers[0]) == len(answers[1]) == 1 and answers[0] != answers[1]
        self.answers = mx.array([a[0] for a in answers])
        self.batch_size = batch_size

    def _tokens(self, prompts):
        seqs = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            assert text.endswith("<|im_start|>assistant\n")
            seqs.append(self.tokenizer.encode(text, add_special_tokens=False))
        width = max(map(len, seqs))
        last = mx.array([len(s) - 1 for s in seqs])
        # Right padding occurs strictly after every selected causal output.
        return mx.array([s + [self.tokenizer.eos_token_id] * (width-len(s)) for s in seqs]), last

    def _scores(self, ids, positions, embeddings=None):
        states = self.model.model(ids, input_embeddings=embeddings)
        selected = states[mx.arange(ids.shape[0]), positions]
        logits = (self.model.model.embed_tokens.as_linear(selected) if self.model.args.tie_word_embeddings
                  else self.model.lm_head(selected))
        return logits[:, self.answers].astype(mx.float32)

    def predict_many(self, prompts):
        answers = []
        for i in range(0, len(prompts), self.batch_size):
            ids, last = self._tokens(prompts[i:i+self.batch_size])
            choices = mx.argmax(self._scores(ids, last), axis=1).tolist()
            answers.extend("AB"[c] for c in choices)
        return answers

    def gradient_check(self):
        ids, last = self._tokens(["Remember CEDAR. Which term was given? A: CEDAR. B: LARK."])
        embeds = self.model.model.embed_tokens(ids)

        def logit_difference(slot):
            x = mx.concatenate([embeds[:, :4], slot, embeds[:, 5:]], axis=1)
            scores = self._scores(ids, last, embeddings=x)
            return scores[0, 0] - scores[0, 1]

        gradient = mx.grad(logit_difference)(embeds[:, 4:5])
        norm = mx.linalg.norm(gradient.astype(mx.float32)).item()
        assert 0 < norm and math.isfinite(norm)
        return norm


if __name__ == "__main__":
    reader = Reader(MODEL, REVISION)
    prompts = ["Return exactly the character A.", "Report B. Reply with just its letter."]
    batch = reader.predict_many(prompts)
    assert batch == ["A", "B"] == [reader.predict_many([p])[0] for p in prompts]
    print("Direct and padded readouts:", batch, "early input gradient norm:", reader.gradient_check())
