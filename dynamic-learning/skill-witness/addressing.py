"""Replay the wide arm with distinct public addresses and the same order."""
import json
from itertools import count

from direct import _hash
from study import CACHE, HERE, run


def unique_ids(names, slots):
    assert len(names) <= slots
    out = {}
    for index, name in enumerate(sorted(names)):
        for suffix in count():
            key = f"{index:03d}-{suffix}"
            if _hash(key) % slots == index:
                out[name] = key
                break
    assert len({_hash(v) % slots for v in out.values()}) == len(names)
    assert sorted(out.values()) == list(out.values())
    return out


def main():
    results = {}
    for project in ("dates", "phones"):
        panel = json.loads((CACHE / f"{project}.json").read_text())
        raw = json.loads((HERE / "results" / f"{project}-wide-2.json").read_text())
        ids = unique_ids(panel["producers"], 32)
        panel["producers"] = {ids[k]: v for k, v in panel["producers"].items()}
        fresh = run(panel, project, "wide", 2)
        reverse = {value: key for key, value in ids.items()}
        for r, s in zip(raw["rows"], fresh["rows"], strict=True):
            assert r["producer"] == reverse[s["producer"]]
            assert all(r[k] == s[k] for k in ("setup", "changed", "total", "time"))
        def stats(out):
            rows = [r for r in out["rows"] if not r["setup"]]
            return dict(perfect=sum(r["success"] for r in rows), revisions=len(rows),
                        exact_rows=sum(r["correct"] for r in rows), total_rows=sum(r["total"] for r in rows),
                        checks=sum(r["checks"] for r in out["rows"]))
        results[project] = dict(original=stats(raw), distinct=stats(fresh),
                                original_used_cells=len({_hash(k) % 32 for k in ids}),
                                distinct_used_cells=len(ids), state_bytes=raw["state_bytes"],
                                address_dictionary_bytes=len(json.dumps(ids, separators=(",", ":")).encode()))
    (HERE / "addressing-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
