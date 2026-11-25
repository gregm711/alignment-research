"""Summarize complete paired producer histories. Bootstrap clusters within each corpus."""

from collections import defaultdict
from pathlib import Path
import argparse
import json
import re

import numpy as np


ARMS = ("native", "ordered", "ordinary", "codegen", "swapped", "frozen", "wide")
CAPS = (1, 2, 3)
DRAWS = 10000
SEED = 88341
METRICS = (
    "checks",
    "revisions",
    "solved",
    "correct",
    "total",
    "previously_correct",
    "broken",
    "accepted",
    "false_accept",
    "changed",
    "changed_solved",
    "setup_checks",
)


def ratio(n, d):
    n, d = np.broadcast_arrays(np.asarray(n, dtype=float), d)
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def counts(rows):
    a = np.zeros(len(METRICS), dtype=np.int64)
    for r in rows:
        a[0] += r["checks"]
        if r["setup"]:
            a[-1] += r["checks"]
            continue
        a[1:-1] += [
            1,
            r["success"],
            r["correct"],
            r["total"],
            r["previously_correct"],
            r["broken"],
            r["accepted"],
            r["false_accept"],
            bool(r["changed"]),
            bool(r["changed"]) and r["success"],
        ]
    return a


def finite(x):
    return float(x) if np.isfinite(x) else None


def metrics(rows):
    c, n, solved, corr, total, prev, broken, accepted, false, changed, cs, setup = (
        counts(rows)
    )
    rates = {
        "success_rate": ratio(solved, n),
        "accuracy": ratio(corr, total),
        "regression_rate": ratio(broken, prev),
        "false_accept_given_accept": ratio(false, accepted),
        "changed_success_rate": ratio(cs, changed),
    }
    return {
        **dict(
            zip(
                METRICS,
                map(
                    int,
                    [
                        c,
                        n,
                        solved,
                        corr,
                        total,
                        prev,
                        broken,
                        accepted,
                        false,
                        changed,
                        cs,
                        setup,
                    ],
                ),
            )
        ),
        **{k: finite(v) for k, v in rates.items()},
    }


def contrast(x, y):
    """Last dimension contains totals. Ratios pool counts across selected histories."""
    return {
        "savings": 1 - ratio(x[..., 0], y[..., 0]),
        "success_delta": ratio(x[..., 2], x[..., 1]) - ratio(y[..., 2], y[..., 1]),
        "accuracy_delta": ratio(x[..., 3], x[..., 4]) - ratio(y[..., 3], y[..., 4]),
        "regression_delta": ratio(x[..., 6], x[..., 5]) - ratio(y[..., 6], y[..., 5]),
        "false_accept_delta": ratio(x[..., 8], x[..., 7]) - ratio(y[..., 8], y[..., 7]),
    }


def chronology(pairs):
    buckets = defaultdict(list)
    for records in pairs:
        for rank, pair in enumerate(records, 1):
            buckets[rank].append(pair)
    a = np.zeros(len(METRICS), dtype=np.int64)
    b = a.copy()
    curve = []
    for nth, entries in sorted(buckets.items()):
        aa, bb = zip(*entries)
        a += counts(aa)
        b += counts(bb)
        delta = contrast(a, b)
        # A tie on cost is not a saving. At setup there are no scored revisions.
        good = bool(a[1] and a[0] < b[0] and delta["success_delta"] >= -0.02)
        curve.append(
            {
                "nth_observed_release": nth,
                "producers": len(entries),
                "native_cumulative_checks": int(a[0]),
                "other_cumulative_checks": int(b[0]),
                "cumulative_success_delta": finite(delta["success_delta"]),
                "ahead_at_comparable_success": good,
            }
        )
    first = next(
        (
            c["nth_observed_release"]
            for i, c in enumerate(curve)
            if all(t["ahead_at_comparable_success"] for t in curve[i:])
        ),
        None,
    )
    return {"curve": curve, "first_durable_break_even": first}


def pair_rows(left, right):
    def index(rows):
        out = {}
        for r in rows:
            key = (r["producer"], r["time"])
            if key in out:
                raise ValueError(f"Duplicate producer/time {key}")
            out[key] = r
        return out

    a, b = index(left), index(right)
    if a.keys() != b.keys():
        return {
            "status": "incomplete_pairs",
            "native_rows": len(a),
            "comparator_rows": len(b),
            "paired_rows": len(a.keys() & b.keys()),
        }
    records = defaultdict(list)
    discordance = [0, 0]
    for key in sorted(a):
        x, y = a[key], b[key]
        if (
            x["setup"] != y["setup"]
            or x["total"] != y["total"]
            or x["changed"] != y["changed"]
        ):
            raise ValueError(f"Mismatched shared grading grid at {key}")
        records[key[0]].append((x, y))
        if not x["setup"]:
            if x["success"] and not y["success"]:
                discordance[0] += 1
            if y["success"] and not x["success"]:
                discordance[1] += 1
    pairs = list(records.values())
    if not pairs:
        return {"status": "empty_pair"}
    cube = np.asarray([[counts(arm) for arm in zip(*p)] for p in pairs])
    pts = contrast(*cube.sum(axis=0))
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(len(pairs), np.ones(len(pairs)) / len(pairs), size=DRAWS)
    xs, ys = (w @ cube[:, i, :] for i in range(2))
    draws = contrast(xs, ys)
    stats = {}
    for m, p in pts.items():
        sample = draws[m][np.isfinite(draws[m])]
        # Undefined draws occur only when a sampled denominator is zero.
        ci = (
            list(map(float, np.quantile(sample, [0.025, 0.975])))
            if len(sample) and len(pairs) >= 2
            else [None, None]
        )
        stats[m] = {"estimate": finite(p), "ci95": ci, "defined_draws": len(sample)}
    s, u, r = (stats[k] for k in ("savings", "success_delta", "regression_delta"))
    estimable = all(
        t["estimate"] is not None and all(v is not None for v in t["ci95"])
        for t in (s, u, r)
    )
    gate = (
        bool(
            5 * int(cube[:, 0, 0].sum()) <= 4 * int(cube[:, 1, 0].sum())
            and s["ci95"][0] > 0
            and u["ci95"][0] >= -0.02 - 1e-12
            and r["ci95"][1] < 0.02 - 1e-12
        )
        if estimable
        else None
    )
    return {
        "status": "paired",
        "producers": len(pairs),
        "contrast": stats,
        "passes_primary_margins": gate,
        "native_only_solved": discordance[0],
        "other_only_solved": discordance[1],
        **chronology(pairs),
    }


def load(root):
    data = defaultdict(dict)
    files = {}
    resources = {}
    for path in sorted(root.glob("*.json")):
        m = re.fullmatch(r"(.+)-(" + "|".join(ARMS) + r")-([123])\.json", path.name)
        if not m:
            continue
        project, arm, cap = m.groups()
        obj = json.loads(path.read_text())
        rows = obj["rows"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"No results in {path}")
        resources.setdefault(project, {})[f"{arm}-{cap}"] = {
            k: obj[k]
            for k in (
                "state_bytes",
                "query_weight_bytes",
                "model_calls",
                "model_tokens",
                "prompt_tokens",
                "duration_seconds",
                "codegen_cost",
            )
            if k in obj
        }
        data[project][arm, int(cap)] = rows
        files.setdefault(project, {})[f"{arm}-{cap}"] = path.name
    return data, files, resources


def summarize(root):
    data, files, resources = load(root)
    out = {
        "bootstrap": {
            "draws": DRAWS,
            "seed": SEED,
            "cluster": "producer",
            "interval": "paired percentile 95%",
        },
        "projects": {},
    }
    for project, runs in data.items():
        comparisons = {}
        for cap in CAPS:
            a = runs.get(("native", cap))
            for arm in ARMS[1:]:
                b = runs.get((arm, cap))
                comparisons[f"{arm}-{cap}"] = (
                    pair_rows(a, b) if a and b else {"status": "unavailable"}
                )
        gates = [
            comparisons[f"{arm}-2"].get("passes_primary_margins")
            for arm in ("ordinary", "codegen")
        ]
        arms = {f"{a}-{k}": metrics(rows) for (a, k), rows in runs.items()}
        native = arms.get("native-2", {})
        floor = native.get("setup_checks", 0) + native.get("revisions", 0)
        floor_data = {
            "min_checks": floor,
            "setup": native.get("setup_checks"),
            "revisions": native.get("revisions"),
            "max_savings": {
                arm: finite(1 - ratio(floor, arms[f"{arm}-2"]["checks"]))
                for arm in ("ordinary", "codegen")
                if native and f"{arm}-2" in arms
            },
        }
        out["projects"][project] = {
            "arms": arms,
            "resources": resources[project],
            "structural_floor_at_cap2": floor_data,
            "files": files[project],
            "vs_native": comparisons,
            "primary_gate": False
            if False in gates
            else True
            if all(x is True for x in gates)
            else None,
        }
    return out


def human(out):
    lines = [
        "# Online revision on historical formatter releases",
        "",
        "Setup calls count toward total cost. Accuracy, solved revisions, conditional false accepts and regressions use subsequent releases. The primary cap is two checks. Each interval resamples entire paired producer histories 10,000 times. Shared model state and shared upstream changes can correlate different producers, so these intervals describe the observed streams conditionally.",
        "",
    ]

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def interval(m):
        return f"{pct(m['estimate'])} [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]"

    for name, p in out["projects"].items():
        lines += [
            f"## {name}",
            "",
            "| Arm at cap 2 | Calls incl. setup | Solved revisions | Exact rows | Newly broken / eligible | False accepts / accepted | Solved after change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            t = p["arms"].get(f"{arm}-2")
            if t:
                lines.append(
                    f"| {arm} | {t['checks']} | {t['solved']}/{t['revisions']} | {pct(t['accuracy'])} | {t['broken']}/{t['previously_correct']} | {t['false_accept']}/{t['accepted']} | {t['changed_solved']}/{t['changed']} |"
                )
            else:
                lines.append(f"| {arm} | unavailable | | | | | |")
        lines += [
            "",
            "Relative call savings, solved-revision and regression differences compare native with the named arm. The last column counts from calibration as release one and requires strictly lower cumulative calls with cumulative solved-revision rate within two percentage points at every remaining prefix.",
            "",
            "| At cap 2 | Call savings (95% CI) | Solved difference (95% CI) | Regression difference (95% CI) | First durable break-even |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm in ARMS[1:]:
            t = p["vs_native"][f"{arm}-2"]
            if t["status"] == "paired":
                d = t["contrast"]
                lines.append(
                    f"| {arm} | {interval(d['savings'])} | {interval(d['success_delta'])} | {interval(d['regression_delta'])} | {t['first_durable_break_even'] or 'none observed'} |"
                )
            else:
                lines.append(f"| {arm} | {t['status']} | | | |")
        floor = p["structural_floor_at_cap2"]
        if floor["setup"] is not None:
            lines += [
                "",
                f"With {floor['setup']} setup calls and one compulsory check on each of {floor['revisions']} revisions, every policy spends at least **{floor['min_checks']} checks**. "
                + " ".join(
                    f"Against {arm} ({p['arms'][arm + '-2']['checks']} calls), the most any policy could save is **{100 * saving:.2f}%** even with perfect accuracy."
                    for arm, saving in floor["max_savings"].items()
                    if saving is not None
                ),
            ]
        ordered = p["vs_native"]["ordered-1"]
        if ordered["status"] == "paired":
            lines += [
                "",
                f"At one check per revision the native and fixed-order selectors pay identical costs. The paired difference in exact-row accuracy is {interval(ordered['contrast']['accuracy_delta'])}, and the difference in solved revisions is {interval(ordered['contrast']['success_delta'])}.",
            ]
        memory = p["resources"]
        if "state_bytes" in memory.get("native-2", {}):
            lines += [
                "",
                "Persistent state at cap two uses "
                + ", ".join(
                    f"{memory[arm + '-2']['state_bytes']:,} bytes for {arm}"
                    for arm in ("native", "wide", "ordinary", "codegen")
                    if "state_bytes" in memory.get(arm + "-2", {})
                )
                + ". This counts the neural fast-weight tensor or the serialized explicit memory. The query weights, generator weights, inference buffers and source data are separate.",
            ]
        local_cost = memory.get("codegen-2", {}).get("codegen_cost")
        if local_cost:
            lines += [
                "",
                f"The local generator was invoked on {local_cost['calls']} prompts at cap two ({local_cost['prompt_tokens']:,} prompt tokens and {local_cost['generated_tokens']:,} generated tokens, including exact cache replays). These tokens are separate from verified formatter calls.",
            ]
        label = {True: "met", False: "missed", None: "unevaluable"}[p["primary_gate"]]
        lines += [
            "",
            f"The joint cap-two gate was **{label}** for {name}.",
            "",
            "| Arm | Cap 1 calls / solved | Cap 2 calls / solved | Cap 3 calls / solved |",
            "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            cells = []
            for k in CAPS:
                t = p["arms"].get(f"{arm}-{k}")
                cells.append(
                    f"{t['checks']} / {pct(t['success_rate'])}" if t else "unavailable"
                )
            lines.append("| " + " | ".join([arm] + cells) + " |")
        lines.append("")
    lines += [
        "The joint gate requires at least 20% call savings at the point estimate, a positive lower bound on savings, a lower bound of at least −2 percentage points on solved revisions, and an upper bound below +2 points on excess regressions against each practical control. A missing comparison cannot establish a pass. Additional paired intervals, discordant revision counts and full cumulative curves are in [summary.json](results/summary.json).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=here / "results")
    parser.add_argument("--report", type=Path, default=here / "results.md")
    args = parser.parse_args()
    report = summarize(args.results)
    if not report["projects"]:
        parser.error("no results files found")
    (args.results / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    args.report.write_text(human(report))
