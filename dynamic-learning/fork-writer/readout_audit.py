"""Exploratory free-generation audit. Never used to change the primary readout."""
import json
from pathlib import Path

import torch

from dev import episodes
from reader import Reader


def main():
    model_id = "LiquidAI/LFM2.5-1.2B-Instruct"
    rev = "b9d816ab510f9a28dca37cb182e9da8b0226e13c"
    reader = Reader(model_id, rev)
    examples = episodes(4701, 24)
    rows, prompts = [], []
    for domain in sorted({episode["domain"] for episode in examples}):
        chosen = set()
        for which, answer in [(0,"A"), (1,"B"), (2,"A"), (0,"B")]:
            i, episode = next((i,e) for i,e in enumerate(examples)
                              if e["domain"] == domain and i not in chosen and e["answers"][which] == answer)
            chosen.add(i)
            q = [episode["present"], *episode["forks"]][which]
            prompts.append(f"{episode['history']}\n\n{q}\nReply with just A or B.".strip())
            rows.append(dict(domain=domain, id=i, question=which, gold=answer))
    with torch.inference_mode():
        batched = reader.predict_many(prompts)
        for row, prompt, pred in zip(rows, prompts, batched, strict=True):
            tokens = reader._tokens([prompt])
            single = reader._scores(tokens)[0]
            greedy = reader.model.generate(**tokens, do_sample=False, max_new_tokens=24, use_cache=True)
            new = greedy[0, tokens.input_ids.shape[1]:]
            text = reader.tokenizer.decode(new, skip_special_tokens=True).strip()
            row.update(forced_batch=pred, forced_single="AB"[int(single.argmax())],
                       single_a_minus_b=float(single[0]-single[1]),
                       greedy_output=text, new_token_ids=new.tolist())
    for row in rows:
        print(row["domain"], row["id"], row["question"], row["gold"], row["forced_batch"], repr(row["greedy_output"]))
    out=Path(__file__).with_name("results")/"lfm12-readout-audit.json"
    out.write_text(json.dumps(dict(model=model_id,revision=rev,seed=4701,rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
