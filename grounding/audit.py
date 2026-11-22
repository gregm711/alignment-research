"""Offline label audit (no GPU): drop substituted-passage runs where the original answer is still readable
in the passage, and re-split by answer so near-duplicate questions cannot straddle train and test.

    uvx --with datasets --with modal python audit.py [TAG]     # writes results/phase1[-TAG]-clean.json
    python3 monitor.py [TAG] clean                             # re-fits and re-tests on the cleaned set

NQ-Swap replaces one mention of the answer. A second mention ("Gomez" after "Selena Gomez"), or another
listed gold answer, can survive, and a model that reads it was scored as answering from memory.
"""
import hashlib
import json
import pathlib
import sys

from build import label, questions
from scoring import words

HERE = pathlib.Path(__file__).parent / "results"
STOP = {"of", "and", "in", "to", "for", "on", "at", "by", "or", "de", "jr", "st"}


def leaky(golds, question, passage):
    """Any word of any gold answer (beyond the question's own words) still appears in the substituted passage."""
    seen, asked = set(words(passage)), set(words(question))
    return any(w in seen for g in golds for w in set(words(g)) - asked - STOP)


def main():
    tag = "-" + sys.argv[1] if len(sys.argv) > 1 else ""
    data = json.loads((HERE / f"phase1{tag}.json").read_text())
    items = {it["question"]: it for it in questions()}
    before = {k: sum(r["label"] == k for r in data["rows"]) for k in ("hidden_ungrounded", "grounded", "set_aside")}
    runs = {"memory": [0, 0], "passage": [0, 0]}  # outcome -> [leaky, all]
    for r in data["rows"]:
        it = items[r["question"]]
        assert [s["substituted_answer"] for s in r["subs"]] == [s["answer"] for s in it["subs"]], r["question"]
        kept = []
        for s, src in zip(r["subs"], it["subs"]):
            bad = leaky(r["gold"], r["question"], src["passage"])
            if s["outcome"] in runs:
                runs[s["outcome"]][0] += bad
                runs[s["outcome"]][1] += 1
            if not bad:
                kept.append(s)
        r["subs_dropped_as_leaky"] = len(r["subs"]) - len(kept)
        r["subs"] = kept
        r["label"] = label(r)
        r["split"] = "train" if int(hashlib.md5(" ".join(words(r["gold"][0])).encode()).hexdigest(), 16) % 2 == 0 else "test"
        r.pop("label_wording_b", None)
    after = {k: sum(r["label"] == k for r in data["rows"]) for k in before}
    hu = [r for r in data["rows"] if r["label"] == "hidden_ungrounded"]
    first = [(r["original"]["answer"].split() or [""])[0].lower().strip(",.") for r in data["rows"] if r["label"] != "set_aside"]
    filler = sum(w in ("the", "a", "an", "according", "in", "it", "there", "based") for w in first) / max(len(first), 1)
    data["audit"] = {"labels_before": before, "labels_after": after, "leaky_substituted_runs": {k: f"{a}/{b}" for k, (a, b) in runs.items()},
                     "hidden_after_known_closed_book": f"{sum(r['known_closed_book'] for r in hu)}/{len(hu)}",
                     "hidden_after_by_split": {s: sum(r["split"] == s for r in hu) for s in ("train", "test")},
                     "share_of_answers_starting_with_filler": round(filler, 3)}
    (HERE / f"phase1{tag}-clean.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data["audit"], indent=1))


if __name__ == "__main__":
    assert leaky(["Selena Gomez"], "who has the most followers", "Zhu Di is a singer. Gomez has many followers.")
    assert not leaky(["Selena Gomez"], "who has the most followers", "Zhu Di is a singer with many followers.")
    main()
