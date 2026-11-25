"""Run three separate forced choices per history and retain paired predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from dev import episodes

HERE = Path(__file__).resolve().parent


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    grouped["overall"] = rows
    scores = {}
    for name, group in grouped.items():
        hits = [[p == y for p, y in zip(r["pred"], r["gold"], strict=True)] for r in group]
        scores[name] = dict(n=len(hits), present=sum(h[0] for h in hits),
                            fork0=sum(h[1] for h in hits), fork1=sum(h[2] for h in hits),
                            both=sum(h[1] and h[2] for h in hits))
    return scores


def run(args):
    fixtures = episodes(args.seed, args.count)
    if args.mlx:
        from reader_mlx import Reader
    else:
        from reader import Reader
    model = Reader(args.model, args.revision)
    result = dict(model=args.model, revision=args.revision, seed=args.seed, modes={})
    for mode in args.modes:
        rows, prompts = [], []
        for i, episode in enumerate(fixtures):
            history = episode["history"]
            if mode == "empty":
                history = ""
            elif mode == "suffix":
                if args.bytes is None or args.bytes <= 0:
                    raise ValueError("Set the calibrated fixed byte budget")
                history = history.encode()[-args.bytes:].decode(errors="ignore")
            for question in [episode["present"], *episode["forks"]]:
                prompts.append(f"{history}\n\n{question}\nReply with just A or B.".strip())
            rows.append(dict(id=i, domain=episode["domain"], gold=list(episode["answers"])))
        start = time.monotonic()
        answers = model.predict_many(prompts)
        elapsed = time.monotonic() - start
        assert len(answers) == len(prompts) and set(answers) <= {"A", "B"}
        for i, row in enumerate(rows):
            row["pred"] = answers[3*i:3*i+3]
        result["modes"][mode] = dict(scores=summarize(rows), rows=rows, seconds=elapsed,
                                    byte_budget=args.bytes if mode=="suffix" else None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2)+"\n")
        tmp.replace(args.out)
        print(mode, result["modes"][mode]["scores"], round(elapsed, 2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("revision")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=4701)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--modes", nargs="+", choices=("full","empty","suffix"), default=("full","empty"))
    p.add_argument("--bytes", type=int)
    p.add_argument("--mlx", action="store_true")
    run(p.parse_args())
