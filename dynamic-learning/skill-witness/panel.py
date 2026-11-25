"""Expose one frozen bank of eight operations per real producer and a disjoint query pool."""

from hashlib import blake2b
import json
from pathlib import Path

import numpy as np

from ops import execute, infer


def _input_key(fields):
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _rank_key(key):
    return blake2b(
        key.encode(), key=b"skill-witness-public-v1", digest_size=16
    ).digest(), key


def _date_pool(anchor):
    def template(order, sep, padded=True, zero="0"):
        specs = [f * 2 if padded and f in ("m", "d") else f for f in order]
        return ["date", zero, specs, ["", sep, sep, ""]]

    generic = [
        template(order, sep, pad)
        for order in ("mdy", "dmy", "ymd")
        for sep in ("/", "-", ".", " ")
        for pad in (True, False)
    ]
    base = anchor or generic[0]
    if base[0] != "date":
        raise ValueError("Invalid date calibration")
    _, zero, specs, literals = base
    changes = {
        "separator": [
            ["date", zero, specs, [literals[0], sep, sep, literals[-1]]]
            for sep in ("/", "-", ".", " ")
        ],
        "order": [
            [
                "date",
                zero,
                [next(s for s in specs if s[0] == f) for f in order],
                literals,
            ]
            for order in ("dmy", "mdy", "ymd")
        ],
        "width": [
            [
                "date",
                zero,
                [
                    ((s[0] if len(s) == 2 else s[0] * 2) if s[0] in fields else s)
                    for s in specs
                ],
                literals,
            ]
            for fields in ("md", "m", "d", "y")
        ],
        "digits": [["date", digit, specs, literals] for digit in ("0", "٠", "۰", "०")],
        "public": generic,
    }
    return base, changes


def _phone_pool(anchor):
    def template(prefix, cuts, separators):
        return ["phone", "0", 0, prefix, cuts, separators, ""]

    plain = template("", [], [])
    generic = [plain, template("0", [], []), template("(", [3, 6], [") ", "-"])]
    for cuts in ([3, 6], [-7, -4], [-8, -4], [-6, -4, -2], [-4], [2, -4], [3, 5, 7]):
        for sep in (" ", "-", "."):
            generic.append(template("", cuts, [sep] * len(cuts)))
    base = anchor or plain

    def change(op, kind, value):
        if op[0] == "switch":
            return [
                "switch",
                op[1],
                change(op[2], kind, value),
                [[k, change(o, kind, value)] for k, o in op[3]],
            ]
        _, zero, skip, pre, cuts, seps, end = op
        if kind == "separator":
            seps = [value] * len(seps)
        elif kind == "prefix":
            pre = value
        elif kind == "cuts":
            cuts = value
            seps = [" "] * len(cuts)
        elif kind == "skip":
            skip = value
        return ["phone", zero, skip, pre, cuts, seps, end]

    changes = {
        k: [change(base, k, v) for v in options]
        for k, options in [
            ("separator", (" ", "-", ".", "")),
            ("prefix", ("", "0", "(0) ")),
            ("cuts", ([-7, -4], [-4], [-8, -4], [-6, -4, -2])),
            ("skip", (0, 1)),
        ]
    }
    changes["public"] = generic
    return base, changes


def bank(examples, fields):
    """Fit once, then generate deterministic candidate edits; use only the paid examples for ranking."""
    fit = infer(examples)
    pool = _phone_pool if "n" in examples[0][0] else _date_pool
    anchor, categories = pool(fit)

    def loss(op):
        return sum(execute(op, x) != y for x, y in examples)

    if fit is None:
        anchor = min((op for ops in categories.values() for op in ops), key=loss)
        anchor, categories = pool(anchor)
    ordered = {
        name: iter(sorted(candidates, key=loss))
        for name, candidates in categories.items()
    }
    chosen, signatures = [], set()

    def add(op):
        # Alias equality uses public inputs only. All-undefined programs do not consume a slot.
        signature = tuple(execute(op, f) for f in fields)
        if signature in signatures or all(p is None for p in signature):
            return False
        chosen.append(op)
        signatures.add(signature)
        return True

    add(anchor)
    while len(chosen) < 8:
        n = len(chosen)
        for candidates in ordered.values():
            if len(chosen) == 8:
                break
            for op in candidates:
                if add(op):
                    break
        if len(chosen) == n:
            raise ValueError(
                "Fewer than eight behaviorally distinct candidates on public input grid"
            )
    return chosen


def episodes(path):
    panel = json.loads(Path(path).read_text())
    inputs = panel["inputs"]
    keys = {i: _input_key(f) for i, f in inputs.items()}
    first, query, reserved = {}, {}, set()
    for name, series in sorted(panel["producers"].items()):
        if len(series) != len(panel["releases"]):
            raise ValueError("Invalid source chronology")
        idx = next((i for i, row in enumerate(series) if row is not None), None)
        if idx is None:
            continue
        ids = sorted(series[idx], key=lambda i: _rank_key(keys[i]))
        if len({keys[i] for i in ids}) != len(ids) or len(ids) < 12:
            raise ValueError("Duplicate inputs or insufficient public fixture grid")
        first[name] = idx
        query[name] = ids[:10]
        reserved.update(keys[i] for i in query[name])
    for name, start in first.items():
        series = panel["producers"][name]
        calibrate = series[start]
        qids = query[name]
        tids = sorted(
            (i for i in calibrate if keys[i] not in reserved),
            key=lambda i: _rank_key(keys[i]),
        )
        if not tids:
            raise ValueError("No independent grading inputs")
        ids = qids + tids
        candidates = bank(
            [(inputs[i], calibrate[i]) for i in qids], [inputs[i] for i in ids]
        )
        pred = np.array(
            [[execute(op, inputs[i]) for op in candidates] for i in ids], dtype=object
        )
        pred.setflags(write=False)
        for j, row in enumerate(series[start + 1 :], start + 1):
            if row is None:
                continue
            if row.keys() != calibrate.keys():
                raise ValueError(f"Upstream input grid changed for {name} at {j}")
            gold = np.array([row[i] for i in ids], dtype=object)
            yield (
                name,
                len(qids),
                candidates,
                j,
                panel["releases"][j],
                qids,
                tids,
                pred,
                gold,
            )
