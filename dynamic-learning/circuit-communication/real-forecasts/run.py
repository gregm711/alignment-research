"""Chronological transfer of an eight-observation source estimator to CDC FluSight."""
from datetime import date, timedelta
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
import pandas as pd

from fetch import CACHE
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from meter import measure

WARM = 8
ARMS = ("majority", "count_mean", "count_median", "reliability", "frozen",
        "clock_1", "clock_2", "clock_4", "clock_8", "clock_16", "reactive")
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
STATE_CODES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


def snapshot(day):
    return pd.read_csv(CACHE / f"{day}_flu_target_hospital_admissions_data.csv").dropna(subset=["value"])


def load_panel(days, truth):
    records, anchors, eligible_days, omitted = [], [], [], []
    for day in days:
        frame = pd.read_csv(CACHE / f"{day}_flu_forecasts_data.csv")
        frame = frame[(frame.horizon == 1) & frame.abbreviation.isin(STATE_CODES)
                      & (~frame.model.str.startswith("FluSight-"))].copy()
        snapshot_path = CACHE / f"{day}_flu_target_hospital_admissions_data.csv"
        if set(frame.reference_date) != {day} or not snapshot_path.exists():
            omitted.append(day)
            continue
        obs = snapshot(day)
        if obs.week_ending_date.max() >= day:
            omitted.append(day)
            continue
        eligible_days.append(day)
        assert set(frame.target_end_date) == {(date.fromisoformat(day) + timedelta(days=7)).isoformat()}
        assert not frame.duplicated(["model", "location_name"]).any()
        records.append(frame.dropna(subset=["quantile_0.5"]))
        obs = obs[obs.week_ending_date < day].sort_values("week_ending_date")
        current = obs.groupby("location_name").last()
        assert (current.week_ending_date.max() < frame.forecast_due_date.min())
        anchors.append(current.value)
    days = eligible_days
    early = pd.concat(records[:WARM])
    teams = early.groupby("model").forecast_team.first()
    roster, seen = [], set()
    for model, _ in sorted(early.model.value_counts().items(), key=lambda x: (-x[1], x[0])):
        if teams[model] not in seen:
            roster.append(model)
            seen.add(teams[model])
        if len(roster) == 16:
            break
    assert len(roster) >= 8
    states = sorted(early.location_name.unique())
    assert len(states) == 51
    targets = [r.target_end_date.iloc[0] for r in records]
    q = np.stack([r.pivot(index="location_name", columns="model", values="quantile_0.5")
                  .reindex(index=states, columns=roster) for r in records])
    anchor = np.stack([a.reindex(states) for a in anchors])
    outcome = truth.reindex(index=targets, columns=states).to_numpy()
    cutoff = snapshot(days[WARM-1]).pivot(index="week_ending_date", columns="location_name", values="value")
    known = cutoff.reindex(index=targets[:WARM], columns=states).to_numpy()
    finite = np.isfinite(q[:WARM]) & np.isfinite(known[:, :, None]) & np.isfinite(anchor[:WARM, :, None])
    matches = (q[:WARM] > anchor[:WARM, :, None]) == (known[:, :, None] > anchor[:WARM, :, None])
    good = (matches & finite).sum(axis=(0, 1))
    bad = (~matches & finite).sum(axis=(0, 1))
    weights = np.maximum(0, np.log((good + 1)/(bad + 1)))
    eligible = np.isfinite(outcome) & np.isfinite(anchor) & (np.isfinite(q).sum(axis=2) >= 4)
    return dict(q=q, anchor=anchor, weights=weights, gold=(outcome>anchor)[WARM:],
                mask=eligible[WARM:], days=days, omitted=omitted, states=states, roster=roster,
                calibration_labels=int(np.isfinite(known).sum()))


def degrees(bits, valid):
    common = valid[:, :, None] & valid[:, None, :]
    diff = (bits[:, :, None] ^ bits[:, None, :]) & common
    together = (POPCOUNT[common] >= 6) & (POPCOUNT[diff] <= 1)
    axis = np.arange(bits.shape[1])
    together[:, axis, axis] = True
    return together.sum(axis=2)


def predict(q, anchor, weights, arm):
    value = q[WARM:]
    if arm == "count_mean":
        return np.nanmean(value, axis=2)>anchor[WARM:], 0
    if arm == "count_median":
        return np.nanmedian(value, axis=2)>anchor[WARM:], 0
    valid = np.isfinite(q) & np.isfinite(anchor[:, :, None])
    sign = np.where(valid, np.where(q > anchor[:, :, None], 1, -1), 0)
    if arm in ("majority", "reliability"):
        if arm == "majority":
            return sign[WARM:].sum(axis=2)>0, 0
        score = sign[WARM:] @ weights
        fallback = valid[WARM:] @ weights == 0
        return np.where(fallback, sign[WARM:].sum(axis=2), score)>0, 0

    history = np.packbits(np.moveaxis(sign[:WARM] > 0, 0, -1), axis=-1)[:, :, 0]
    available = np.packbits(np.moveaxis(valid[:WARM], 0, -1), axis=-1)[:, :, 0]
    degree = degrees(history, available)
    suspects = degree >= 8
    flags = np.zeros(len(history), dtype=np.uint8)
    age = np.zeros(len(history), dtype=int)
    period = int(arm.split("_")[1]) if arm.startswith("clock_") else None
    outputs, refreshes = [], 0
    for i in range(WARM, len(q)):
        votes = sign[i]
        outputs.append((votes / degree).sum(axis=1) > 0)
        if i+1 == len(q) or arm == "frozen":
            continue
        history = (history << 1) | (votes > 0)
        available = (available << 1) | valid[i]
        if period:
            if (i-WARM+1) % period == 0:
                degree = degrees(history, available)
                refreshes += len(history)
        else:
            size = (valid[i] & suspects).sum(axis=1)
            weak = (size == 0) | (np.abs((votes*suspects).sum(axis=1)) < .75*size)
            flags = ((flags << 1) | weak) & 15
            age += 1
            choose = (age >= 6) & (POPCOUNT[flags] >= 2)
            if choose.any():
                degree[choose] = degrees(history[choose], available[choose])
                suspects[choose] = degree[choose] >= 8
                flags[choose] = age[choose] = 0
                refreshes += int(choose.sum())
    return np.stack(outputs), refreshes


def block_interval(delta, sizes):
    rng = np.random.default_rng(8304)
    draws = 0.
    totals = 0.
    for d, n in zip(delta, sizes):
        start = rng.integers(len(d), size=(5000, (len(d)+3)//4))
        ids = ((start[:, :, None] + np.arange(4)) % len(d)).reshape(5000, -1)[:, :len(d)]
        draws += d[ids].sum(axis=1)
        totals += n[ids].sum(axis=1)
    return np.quantile(draws/totals, [.025,.975]).tolist()


def main():
    days = sorted(p.name[:10] for p in CACHE.glob("*flu_forecasts_data.csv"))
    truth = pd.read_csv(CACHE/"target-hospital-admissions.csv").pivot(index="date",columns="location_name",values="value")
    panels = [load_panel([d for d in days if lo <= d <= hi],truth) for lo,hi in
              (("2024-11","2025-06"),("2025-11","2026-06"))]
    out = dict(seasons=[], arms={}, differences={})
    total = sum(p["mask"].sum() for p in panels)
    counts, denominators = {}, [p["mask"].sum(axis=1) for p in panels]
    for p in panels:
        signs = np.where(np.isfinite(p["q"]),np.where(p["q"]>p["anchor"][:,:,None],1,-1),0)
        active = np.isfinite(p["q"])
        out["seasons"].append(dict(first=p["days"][0],last=p["days"][-1],test_dates=p["days"][WARM:],
            omitted_issues=p["omitted"],roster=p["roster"],eligible=int(p["mask"].sum()),state_weeks=int(p["mask"].size),
            missing_fraction=float(1-active[WARM:].mean()), rises=int((p["gold"]&p["mask"]).sum()),
            calibration_labels=p["calibration_labels"],signs_test=int(active[WARM:].sum()),
            signs_calibration=int(active[:WARM].sum()),calibration_slots=int(active[:WARM].size),
            test_slots=int(active[WARM:].size)))
        p["hard"]=(abs(signs[WARM:].sum(axis=2))<=2)&p["mask"]
    for arm in ARMS:
        def infer():
            return [predict(p["q"],p["anchor"],p["weights"],arm) for p in panels]
        ret, c=measure(infer)
        count=max(1,int(.14/max(c["wall_seconds"],.00001)))
        def repeat():
            for _ in range(count):
                infer()
        timings=[measure(repeat)[1] for _ in range(3)]
        right=[(a==p["gold"])&p["mask"] for p,(a,_) in zip(panels,ret)]
        counts[arm]=[r.sum(axis=1) for r in right]
        hard_n=sum(p["hard"].sum() for p in panels)
        hard_right=sum((r&p["hard"]).sum() for r,p in zip(right,panels))
        out["arms"][arm]=dict(correct=int(sum(r.sum() for r in right)),n=int(total),
            season_correct=[int(r.sum()) for r in right],accuracy=float(sum(r.sum() for r in right)/total),
            hard_n=int(hard_n),hard_correct=int(hard_right),refreshes_after_setup=sum(r for _,r in ret),
            setup_maps=0 if arm in ARMS[:4] else 102,
            raw_timings=timings,repetitions=count,
            cpu_uj=(float(median(t["cpu_joules"] for t in timings)*1e6/count/total)
                    if all(t["cpu_joules"] is not None for t in timings) else None),
            wall_us=float(median(t["wall_seconds"] for t in timings)*1e6/count/total),
            issue_correct=[x.tolist() for x in counts[arm]])
    for arm in ARMS:
        delta=[a-b for a,b in zip(counts["reactive"],counts[arm])]
        out["differences"][arm]=[float(sum(d.sum() for d in delta)/total),*block_interval(delta,denominators)]
    out["issue_denominators"]=[x.tolist() for x in denominators]
    (HERE/"results.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
