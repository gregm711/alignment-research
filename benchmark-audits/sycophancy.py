"""Part 2: four checks on two sycophancy datasets (CPU only).

    python3 sycophancy.py        # downloads the files to a temp folder, writes results/sycophancy.json
"""
import collections
import hashlib
import json
import pathlib
import re
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "grounding"))
from scoring import gives, words  # noqa: E402

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
GH = "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/{}"
MWE = ["sycophancy_on_nlp_survey.jsonl", "sycophancy_on_philpapers2020.jsonl", "sycophancy_on_political_typology_quiz.jsonl"]
pct = lambda a, b: f"{a}/{b} ({a / b:.1%})"


def fetch(url, tmp):
    path = pathlib.Path(tmp) / hashlib.md5(url.encode()).hexdigest()
    urllib.request.urlretrieve(url, path)
    return path


def stem(q):
    """The question without the user's biography: its last sentence plus the answer options."""
    head, sep, options = q.partition("\n (A)")
    head = head.split("\n\nChoices:")[0].strip()
    return re.split(r"(?<=[.!?])\s+", head)[-1].strip() + " |" + re.sub(r"\s+", " ", options.replace("Answer:", "")).strip()


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        sha = lambda p: hashlib.sha1(p.read_bytes()).hexdigest()[:12]
        hf = {f: fetch(HF.format("Anthropic/model-written-evals", "sycophancy/" + f), tmp) for f in MWE}
        gh = {f: fetch(GH.format(f), tmp) for f in MWE}
        out["file_hashes"] = {f: {"hugging_face": sha(hf[f]), "github": sha(gh[f])} for f in MWE}
        for f in MWE:
            rows = [json.loads(l) for l in open(gh[f])]
            a = sum(r["answer_matching_behavior"].strip().startswith("(A)") for r in rows)
            stems = collections.Counter(stem(r["question"]) for r in rows)
            out[f] = {"items": len(rows), "agreeing_answer_is_A": pct(a, len(rows)), "distinct_questions": len(stems),
                      "items_in_ten_most_common_questions": pct(sum(n for _, n in stems.most_common(10)), len(rows)),
                      "letters_used": dict(collections.Counter(r["answer_matching_behavior"].strip()[:3] for r in rows).most_common(6))}
        rows = [json.loads(l) for l in open(fetch(HF.format("meg-tong/sycophancy-eval", "answer.jsonl"), tmp))]
        base = list({r["base"]["question"]: r["base"] for r in rows}.values())  # one entry per distinct question
        for b in base:  # some source datasets list one answer only
            b["answer"] = b.get("answer") or [b["correct_answer"]]
            b["answer"] = b["answer"] if isinstance(b["answer"], list) else [b["answer"]]
        wrong_is_right = [b for b in base if gives(b["incorrect_answer"], b["answer"], "")]
        not_listed = [b for b in base if not gives(b["correct_answer"], b["answer"], "")]
        in_question = [b for b in base if any(words(a) and set(words(a)) <= set(words(b["question"])) for a in b["answer"])]
        out["sycophancy-eval answer.jsonl"] = {
            "prompts": len(rows), "distinct_questions": len(base), "source_datasets": dict(collections.Counter(b["dataset"] for b in base)),
            "templates": dict(collections.Counter(r["metadata"]["prompt_template"][:60] for r in rows)),
            "suggested_wrong_answer_matches_a_listed_correct_answer": pct(len(wrong_is_right), len(base)),
            "correct_answer_not_among_listed_answers": pct(len(not_listed), len(base)),
            "a_listed_answer_is_wholly_in_the_question": pct(len(in_question), len(base)),
            "examples_wrong_is_right": [[b["question"][:90], b["incorrect_answer"], b["answer"][:4]] for b in wrong_is_right[:8]],
            "examples_answer_in_question": [[b["question"][:110], [a for a in b["answer"] if words(a) and set(words(a)) <= set(words(b["question"]))][:2]] for b in in_question[:8]]}
    (HERE / "results" / "sycophancy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    assert stem("Hello. I like tea. Do you agree? Tea is good.\n\nChoices:\n (A) Agree\n (B) Disagree\n\nAnswer:") == "Tea is good. |Agree (B) Disagree"
    main()
