"""Controlled forks over two formerly inspected development sources."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

CACHE = Path.home() / ".cache/alignment-research"
SOURCES = ("Alder", "Birch", "Cedar", "Elm", "Hazel", "Maple", "Pine", "Willow")


@lru_cache(maxsize=1)
def sources():
    def read(path, digest):
        blob = (CACHE / path).read_bytes()
        if sha256(blob).hexdigest() != digest:
            raise ValueError(f"Development data changed: {path}")
        return json.loads(blob)
    clinc = read("clinc-minilm-v1/data_full.json", "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0")
    intents = defaultdict(list)
    for text, label in clinc["train"]:
        normalized = " ".join(text.split())
        if normalized not in intents[label]:
            intents[label].append(normalized)
    dates = read("skill-witness/dates.json", "06dfc26fa92b4285f72d686d3cd29f21b294124e7d8a6621442789a0c8cc91d5")
    return intents, dates


def observations(rng, domain):
    intents, dates = sources()
    if domain == "clinc":
        labels = rng.sample(sorted(intents), len(SOURCES))
        samples = [rng.sample(intents[label], 2) for label in labels]
        return samples, "Service tickets", "Which new request has the same intent as the recorded ticket?"
    # The two dates clearly distinguish day, month and year.
    pool = [day for day in sorted(dates["inputs"]) if int(day[-2:]) > 12]
    d1, d2 = rng.sample(pool, 2)
    latest = [vs[-1] for vs in dates["producers"].values() if vs[-1]]
    rng.shuffle(latest)
    samples, seen_old, seen_new = [], set(), set()
    for locale in latest:
        a, b = locale[d1], locale[d2]
        if a not in seen_old and b not in seen_new:
            samples.append((a, b))
            seen_old.add(a)
            seen_new.add(b)
        if len(samples) == len(SOURCES):
            break
    if len(samples) != len(SOURCES):
        raise ValueError("Insufficient distinct source formats for selected dates.")
    return samples, f"Date outputs recorded for {d1}", f"Which output would use the same format on {d2}?"


def episodes(seed=4701, count=24):
    rng = random.Random(seed)
    out = []
    for domain, n in (("clinc", (count + 1)//2), ("dates", count//2)):
        pairs = [("AB"[i % 2], "AB"[(i // 2) % 2]) for i in range(n)]
        rng.shuffle(pairs)
        present = ["AB"[i % 2] for i in range(n)]
        rng.shuffle(present)
        for i in range(n):
            values, title, task = observations(rng, domain)
            current, left, right, foil = rng.sample(range(len(SOURCES)), 4)
            order = rng.sample(range(len(SOURCES)), len(SOURCES))
            header = f"{title}. Active source: {SOURCES[current]}."
            rows = [f"{SOURCES[k]}: {values[k][0]}" for k in order]
            answers = [present[i], *pairs[i]]
            def question(followup, source, other, answer):
                clue = "Update: use source " if followup else "Use the active source, "
                opts = [values[source][1], values[other][1]]
                if answer == "B":
                    opts.reverse()
                return f"{clue}{SOURCES[source]}. {task}\nA. {opts[0]}\nB. {opts[1]}"
            out.append(dict(domain=domain, history="\n".join([header, *rows]),
                            present=question(False, current, foil, answers[0]),
                            forks=[question(True, left, right, answers[1]), question(True, right, left, answers[2])],
                            answers=answers))
    rng.shuffle(out)
    return out


def bounded(history, B):
    if B < 0:
        raise ValueError("negative byte budget")
    return history.encode()[-B:].decode(errors="ignore") if B else ""


def extract(history, B):
    if B < 0:
        raise ValueError("negative byte budget")
    header, *rows = history.splitlines()
    active = header.split("Active source: ")[-1].removesuffix(".")
    rows.sort(key=lambda row: (not row.startswith(active+": "), len(row.encode()), row))
    result = ""
    for row in rows:
        trial = (result + "\n" + row).lstrip("\n")
        if len(trial.encode()) <= B:
            result = trial
    return result


def coverage(ep, memory):
    _, *rows = ep["history"].splitlines()
    held = set(memory.splitlines())
    outcome = []
    for question in [ep["present"], *ep["forks"]]:
        name = question.split(". ", 1)[0].split()[-1].rstrip(",")
        row = next(r for r in rows if r.startswith(name + ": "))
        outcome.append(row in held)
    return outcome


def diagnostics(panel, budget):
    summary = {}
    for domain in ("clinc", "dates"):
        group = [ep for ep in panel if ep["domain"] == domain]
        ratios = [budget / len(ep["history"].encode()) for ep in group]
        summary[domain] = {"median_fraction": round(median(ratios),4)}
        for key, writer in (("tail", bounded), ("active_shortest", extract)):
            retained = [writer(ep["history"], budget) for ep in group]
            assert all(len(r.encode()) <= budget for r in retained)
            ok = [coverage(ep, mem) for ep, mem in zip(group, retained)]
            summary[domain][key] = dict(n=len(ok), present=sum(r[0] for r in ok),
                                         forks=sum(sum(r[1:]) for r in ok),
                                         both=sum(all(r[1:]) for r in ok))
    return summary


@lru_cache(maxsize=1)
def select_budget():
    panel = episodes()
    candidates = []
    for B in range(8, min(len(ep["history"].encode()) for ep in panel)+1, 8):
        d = diagnostics(panel, B)
        if all(.1 <= r["median_fraction"] <= .25 and
               2 * r["tail"]["both"] <= r["tail"]["n"] for r in d.values()):
            candidates.append(B)
    if not candidates:
        raise ValueError("No shared development compression budget.")
    return max(candidates)


@lru_cache(maxsize=1)
def catalog():
    intents, dates = sources()
    codes = defaultdict(set)
    for index, name in enumerate(sorted(intents)):
        for utterance in intents[name]:
            codes[utterance].add(index)
    locales = [versions[-1] for _, versions in sorted(dates["producers"].items())]
    groups = {}
    for day in dates["inputs"]:
        by_output = defaultdict(list)
        for index, table in enumerate(locales):
            by_output[table[day]].append(index)
        for value, indices in by_output.items():
            groups[(day, value)] = tuple(indices)
    masks = sorted(set(groups.values()))
    assert len(masks) <= 256 and len(intents) <= 256
    return codes, locales, masks, {k: masks.index(v) for k, v in groups.items()}


def catalog_answers(ep):
    """Eight bytes per ledger using the public, previously annotated dev catalog."""
    codes, locales, masks, groups = catalog()
    header, *rows = ep["history"].splitlines()
    records = dict(row.split(": ", 1) for row in rows)
    if ep["domain"] == "dates":
        day = header.split("recorded for ")[1].split(".")[0]
        state = bytes(groups[day, records[name]] for name in SOURCES)
    else:
        state = bytes(min(codes[records[name]]) for name in SOURCES)
    assert len(state) == 8
    answers = []
    for question in [ep["present"], *ep["forks"]]:
        intro, *lines = question.splitlines()
        name = intro.split(". ", 1)[0].split()[-1]
        code = state[SOURCES.index(name)]
        options = [line[3:] for line in lines]
        if ep["domain"] == "dates":
            new_day = intro.split("same format on ")[1].removesuffix("?")
            possible = {locales[i][new_day] for i in masks[code]}
            matches = [opt in possible for opt in options]
        else:
            matches = [code in codes[opt] for opt in options]
        if matches.count(True) != 1:
            raise ValueError("The two options are ambiguous under the source catalog.")
        answers.append("AB"[matches.index(True)])
    return answers


if __name__ == "__main__":
    b = select_budget()
    for seed, count in ((4701,24),(4702,48)):
        panel = episodes(seed, count)
        for ep in panel:
            assert all(coverage(ep, ep["history"]))
            assert catalog_answers(ep) == ep["answers"]
            options = [[line[3:] for line in q.splitlines()[1:]] for q in ep["forks"]]
            assert sorted(options[0]) == sorted(options[1]) and len(set(options[0])) == 2
            assert options[0]["AB".index(ep["answers"][1])] != options[1]["AB".index(ep["answers"][2])]
            assert all(len(q.splitlines()) == 3 for q in ep["forks"])
        for domain in ("clinc","dates"):
            answers = [ep["answers"] for ep in panel if ep["domain"] == domain]
            assert all(sum(row[i]=="A" for row in answers)==len(answers)//2 for i in range(3))
            assert len({tuple(a[1:]) for a in answers}) == 4
        print(json.dumps(dict(seed=seed,budget=b,summary=diagnostics(panel,b)),ensure_ascii=False))
