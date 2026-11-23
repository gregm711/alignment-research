"""Does the original answer survive the edit? Counts for three datasets (CPU only).

    uvx --with datasets --with modal python leakcheck.py     # writes results/*.json

The rule is grounding/scoring.py's `leaky`; see README.md.
"""
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "grounding"))
from audit import leaky_loose  # noqa: E402  (grounding/audit.py)
from scoring import STOP, leaky, outcome, words  # noqa: E402

OUT = HERE / "results"
CUB_MODELS = ["gpt2-xl", "pythia-6.9b", "qwen-1.5b", "qwen-1.5b-instruct", "qwen-7b", "qwen-7b-instruct", "qwen-32b",
              "qwen-32b-instruct", "cohere-command-a", "gpt-4.1-mini", "gpt-4.1"]
CONFLICTQA_MODELS = ["chatgpt", "gpt4", "llama2-70b", "llama2-7b", "palm2", "qwen7b", "vicuna33b", "vicuna7b"]
untag = lambda s: re.sub(r"\s+", " ", re.sub(r"</?[A-Za-z]+>", " ", s)).strip()
pct = lambda a, b: f"{a}/{b} ({a / b:.1%})" if b else "0/0"


def cub_nq():
    from datasets import load_dataset
    res, clean_ids = {}, None
    for m in CUB_MODELS:
        ds = load_dataset("copenlu/cub-nq", m, split="test")
        rows = [r for r in ds if r["context_type"] == "edited"]
        flag = {r["example_id"]: leaky([r["target_true"].strip()], r["question"], untag(r["context"])) for r in rows}
        if clean_ids is None:
            loose = sum(leaky_loose([r["target_true"].strip()], r["question"], untag(r["context"])) for r in rows)
            res["dataset"] = {"context_types": dict(collections.Counter(r["context_type"] for r in ds)),
                              "edited_contexts_leaky_strict": pct(sum(flag.values()), len(rows)), "edited_contexts_leaky_loose": pct(loose, len(rows))}
            clean_ids = sorted(k for k, v in flag.items() if not v)
        if "generated_w_context" not in ds.column_names or not rows[0]["generated_w_context"]:
            res[m] = "no saved full answers; not scored"
            continue
        c = collections.Counter()
        for r in rows:
            o = outcome(str(r["generated_w_context"]).strip().split("\n")[0], [r["target_true"].strip()], [r["target_new"].strip()], r["question"])
            c[o, flag[r["example_id"]]] += 1
        n = lambda o, f=None: sum(v for (oo, ff), v in c.items() if oo == o and (f is None or ff == f))
        tot, tot_clean = sum(c.values()), sum(v for (_, f), v in c.items() if not f)
        res[m] = {"outcomes": {o: n(o) for o in ("memory", "passage", "both", "abstains", "other")},
                  "leaky_share_where_original_answer_given": pct(n("memory", True), n("memory")),
                  "leaky_share_where_context_followed": pct(n("passage", True), n("passage")),
                  "original_answer_rate_all_items": pct(n("memory"), tot),
                  "original_answer_rate_non_leaky_items": pct(n("memory", False), tot_clean)}
    (OUT / "cub-nq.json").write_text(json.dumps(res, indent=1))
    (OUT / "cub-nq-clean-ids.json").write_text(json.dumps(clean_ids))
    return res


def faithfulness_qa():
    from datasets import load_dataset
    by, clean_ids = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0]), []  # strict, loose, whole answer, new answer missing, n, swapped entity survives
    for split in ("train", "validation", "test"):
        for r in load_dataset("Laurie/faithfulness-qa-dataset", "all", split=split):
            ctx, gold = r["modified_context"], [r["original_answer"]]
            s = leaky(gold, r["question"], ctx)
            flags = [s, leaky_loose(gold, r["question"], ctx), bool(words(gold[0])) and " " + " ".join(words(gold[0])) + " " in " " + " ".join(words(ctx)) + " ",
                     r["faithful_answer"].lower() not in ctx.lower(), 1, r["original_entity"].lower() in ctx.lower(),
                     leaky([r["original_entity"]], r["question"], ctx), leaky_loose([r["original_entity"]], r["question"], ctx)]
            for key in ("all", "source: " + r["source"], "entity type: " + r["entity_type"]):
                for i, f in enumerate(flags):
                    by[key][i] += f
            if not flags[6]:  # for this dataset the swapped entity, not the whole answer, is what must not survive
                clean_ids.append(r["id"])
    res = {k: {"leaky_strict": pct(v[0], v[4]), "leaky_loose": pct(v[1], v[4]), "whole_original_answer_still_present": pct(v[2], v[4]),
               "new_answer_not_in_modified_context": pct(v[3], v[4]),
               "swapped_entity_string_still_present": pct(v[5], v[4]),
               "word_of_swapped_entity_still_present": pct(v[6], v[4]), "swapped_entity_last_word_still_present": pct(v[7], v[4])} for k, v in sorted(by.items(), key=lambda kv: -kv[1][4])}
    (OUT / "faithfulness-qa.json").write_text(json.dumps(res, indent=1))
    (OUT / "faithfulness-qa-clean-ids.json").write_text(json.dumps(clean_ids))
    return res


def conflictqa():
    """Exploratory. Answers here are sentences, so: the content words of the model's own answer (not in the question,
    not in the counter answer, longer than three letters) ALL appear in the evidence written against it."""
    from huggingface_hub import hf_hub_download
    res = {}
    for m in CONFLICTQA_MODELS:
        hit = n = 0
        for line in open(hf_hub_download("osunlp/ConflictQA", f"conflictQA-popQA-{m}.json", repo_type="dataset")):
            r = json.loads(line)
            content = {w for w in words(r["memory_answer"]) if len(w) > 3} - set(words(r["question"])) - set(words(r["counter_answer"])) - STOP
            if content:
                n += 1
                hit += content <= set(words(r["counter_memory"]))
        res[m] = pct(hit, n)
    (OUT / "conflictqa.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    assert leaky(["Selena Gomez"], "who has the most followers", untag("<P> Zhu Di sings . Gomez has many followers . </P>"))
    for name, fn in (("CUB-NQ", cub_nq), ("Faithfulness-QA", faithfulness_qa), ("ConflictQA", conflictqa)):
        print("=== " + name)
        print(json.dumps(fn(), indent=1))
