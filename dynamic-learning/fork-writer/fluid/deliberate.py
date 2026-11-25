"""Paired direct versus bounded scratchpad readout on unseen fluid worlds."""
import json
from pathlib import Path
import sys
import time

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))
from reader_mlx import MODEL, REVISION, Reader


def with_notes(prompt, notes):
    return f"{prompt}\n\nYour scratch notes:\n{notes}\n\nReply with just A or B."


def decision(reader, prompt):
    ids, _ = reader._tokens([prompt])
    start = time.monotonic()
    answer = reader.predict_many([prompt])[0]
    return dict(answer=answer, input_tokens=int(ids.size), seconds=time.monotonic()-start)


def scratch(reader, prompt):
    request = prompt + "\n\nBefore the final answer, write only brief intermediate deductions for this problem."
    ids, _ = reader._tokens([request])
    start = time.monotonic()
    chunks = stream_generate(reader.model, reader.tokenizer, ids[0], max_tokens=48,
                             sampler=make_sampler(temp=0))
    text = []
    for chunk in chunks:
        text.append(chunk.text)
    assert chunk.generation_tokens <= 48 and chunk.finish_reason in ("length", "stop")
    return dict(text="".join(text), input_tokens=int(ids.size), tokens=chunk.generation_tokens,
                finish=chunk.finish_reason, seconds=time.monotonic()-start)


def summarize(rows):
    out = {"questions": len(rows), "worlds": len(set(r["world"] for r in rows))}
    units = sorted(set((r["world"], r["depth"]) for r in rows))
    for arm in ("direct", "blank", "own"):
        hit = [r[arm]["answer"] == r["gold"] for r in rows]
        joint = sum(all(good for good, row in zip(hit, rows)
                        if (row["world"], row["depth"]) == unit) for unit in units)
        out[arm] = dict(correct=sum(hit), joint=joint, joint_total=len(units))
    for base in ("direct","blank"):
        def cases(repair):
            return [[r["world"],r["depth"],r["fork"]] for r in rows
                    if (r[base]["answer"] != r["gold"]) == repair and
                    (r["own"]["answer"] == r["gold"]) == repair]
        out["own_vs_"+base] = dict(repairs=cases(True), overturns=cases(False))
    out["scratch_tokens"] = sum(r["notes"]["tokens"] for r in rows)
    out["capped"] = sum(r["notes"]["finish"] == "length" for r in rows)
    out["time_seconds"] = {arm:round(sum(r[arm]["seconds"] for r in rows),3)
                           for arm in ("direct","blank","own","notes")}
    out["input_tokens"] = {arm:sum(r[arm]["input_tokens"] for r in rows)
                           for arm in ("direct","blank","own","notes")}
    return out


def main():
    from run import tasks
    cases = [r for r in tasks(seed=776031, count=24) if r["world"] < 6]
    assert {(r["world"], r["depth"], r["fork"]) for r in cases} == {
        (w, d, f) for w in range(6) for d in (1,2) for f in (0,1)}
    assert len(cases) == 24
    reader = Reader(MODEL, REVISION)
    reader.predict_many(["Reply with just A or B."])
    rows = []
    for case in cases:
        p = case["prompt"]
        direct = decision(reader, p)
        blank = decision(reader, with_notes(p,""))
        notes = scratch(reader,p)
        own = decision(reader,with_notes(p,notes["text"]))
        rows.append(dict(world=case["world"],depth=case["depth"],fork=case["fork"],
                         gold=case["answer"], direct=direct, blank=blank, notes=notes, own=own))
    summary = {str(d):summarize([r for r in rows if r["depth"] == d]) for d in (1,2)}
    summary["overall"] = summarize(rows)
    record = dict(model=MODEL,revision=REVISION,seed=776031,summary=summary,rows=rows)
    (HERE / "deliberation-results.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
