"""Explicit per-producer retrieval, ordinary program induction and optional local codegen."""
import json

from llm import propose, run_llm
from ops import execute, infer


def run(op, fields):
    if op and op[0] == "llm":
        return run_llm(op, fields)
    return execute(op, fields)


class Comparator:
    def __init__(self, model=False):
        self.model = model
        self.memory = {}
        self.calls = 0
        self.tokens = 0
        self.prompt_tokens = 0
        self.actual_calls = 0
        self.actual_tokens = 0
        self.actual_prompt_tokens = 0

    def codegen_cost(self):
        return dict(calls=self.calls, actual_calls=self.actual_calls,
                    prompt_tokens=self.prompt_tokens, generated_tokens=self.tokens,
                    actual_prompt_tokens=self.actual_prompt_tokens,
                    actual_generated_tokens=self.actual_tokens)

    def bytes(self):
        return len(json.dumps(self.memory, separators=(",", ":"), ensure_ascii=False).encode())

    def predict(self, producer, fields):
        record = self.memory.get(producer)
        if record is None:
            return None
        return run(record["operation"], fields)

    def choose(self, producer, xs, used):
        last = self.memory.get(producer, {}).get("failed_idx")
        if last is not None and 0 <= last < len(xs) and last not in used:
            return last
        return next((i for i in range(len(xs)) if i not in used), None)

    def learn(self, producer, examples, failed_idx=None, intro=False):
        """On repair, examples are this revision's ordered paid receipts, most recent last."""
        examples = list(examples)
        if not examples:
            raise ValueError("A repair or introduction needs verified evidence")
        record = self.memory.setdefault(producer, {})
        prior = record.get("operation")
        candidates = [prior, infer(examples)]
        if self.model:
            # Historical evidence is identified separately and never validates a new repair.
            retrieved = {"operation": prior, "example": record.get("example")} if record else None
            op, metadata = propose(examples, retrieved, record.get("failure"), producer)
            self.calls += metadata["generation_calls"]
            self.tokens += metadata["generated_tokens"] or 0
            self.prompt_tokens += metadata["prompt_tokens"] or 0
            if metadata["actual_generation_calls"]:
                self.actual_calls += 1
                self.actual_tokens += metadata["generated_tokens"] or 0
                self.actual_prompt_tokens += metadata["prompt_tokens"] or 0
            candidates.append(op)
        available = [op for op in candidates if op is not None]
        # Stable ties favor an incumbent, followed by the short ordinary synthesis.
        if available:
            record["operation"] = max(available, key=lambda op: sum(
                run(op, fields) == label for fields, label in examples))
        else:
            record["operation"] = None
        if intro or "example" not in record:
            record["example"] = examples[0]
        if failed_idx is not None:
            record["failed_idx"] = int(failed_idx)
            record["failure"] = examples[-1]


if __name__ == "__main__":
    model = Comparator()
    dates = [{"y": "2030", "m": "2", "d": "3"}, {"y": "2031", "m": "11", "d": "21"}]
    model.learn("a", list(zip(dates, ["2030/2/3", "2031/11/21"])), intro=True)
    assert model.predict("a", dates[1]) == "2031/11/21"
    model.learn("a", [(dates[1], "21-11-2031")], failed_idx=1)
    assert model.choose("a", dates, []) == 1 and model.choose("a", dates, [1]) == 0
    assert model.predict("a", dates[1]) == "21-11-2031" and model.predict("b", dates[1]) is None
    assert model.bytes() > 0 and model.tokens == model.calls == 0
