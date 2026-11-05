"""Personal records and streaks."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .core import Context


def longest_streak(flags: pd.Series):
    """(length, start, end) of the longest run of True values (NaN breaks a run)."""
    best = (0, None, None)
    run, run_start = 0, None
    for day, ok in flags.items():
        if bool(ok) and not pd.isna(ok):
            if run == 0:
                run_start = day
            run += 1
            if run > best[0]:
                best = (run, run_start, day)
        else:
            run = 0
    return best


def current_streak(flags: pd.Series) -> int:
    n = 0
    for ok in flags.iloc[::-1]:
        if bool(ok) and not pd.isna(ok):
            n += 1
        else:
            break
    return n


def personal_records(ctx: Context) -> List[dict]:
    d, fmt = ctx.daily, ctx.fmt
    out: List[dict] = []

    def add(label, value, when, detail=""):
        out.append({"label": label, "value": value, "when": when, "detail": detail})

    if "steps" in d and d["steps"].notna().any():
        day = d["steps"].idxmax()
        add("Most steps in a day", fmt.num(d["steps"].max()), fmt.date(day))
        wk = ctx.weekly.get("steps")
        if wk is not None and wk.notna().any():
            add("Best week (avg steps/day)", fmt.num(wk.max()), f"week of {fmt.date(wk.idxmax())}")
        length, s, e = longest_streak(d["steps"] >= 10000)
        if length >= 3:
            add("Longest 10,000-step streak", f"{length} days", fmt.span(s, e))
    if "distance_km" in d and d["distance_km"].notna().any():
        add("Farthest day on foot", fmt.distance(d["distance_km"].max()), fmt.date(d["distance_km"].idxmax()))
    if "exercise_ring_pct" in d and d["exercise_ring_pct"].notna().sum() > 30:
        length, s, e = longest_streak(d["exercise_ring_pct"] >= 100)
        if length >= 3:
            add("Longest Exercise-ring streak", f"{length} days", fmt.span(s, e))
    rings = [c for c in ("move_ring_pct", "exercise_ring_pct", "stand_ring_pct") if c in d]
    if len(rings) == 3 and d[rings].notna().all(axis=1).sum() > 30:
        all_closed = (d[rings] >= 100).all(axis=1).where(d[rings].notna().all(axis=1))
        length, s, e = longest_streak(all_closed)
        if length >= 2:
            add("Longest all-rings streak", f"{length} days", fmt.span(s, e))
    if "resting_hr" in ctx.monthly and ctx.monthly["resting_hr"].notna().sum() >= 3:
        mon = ctx.monthly["resting_hr"].dropna()
        add("Lowest monthly resting HR", fmt.value("resting_hr", mon.min()), fmt.month(mon.idxmin()))
    if "vo2max" in d and d["vo2max"].notna().any():
        add("Highest cardio fitness (VO₂ max)", fmt.value("vo2max", d["vo2max"].max()), fmt.date(d["vo2max"].idxmax()))
    if "hrv" in ctx.monthly and ctx.monthly["hrv"].notna().sum() >= 3:
        mon = ctx.monthly["hrv"].dropna()
        add("Highest monthly HRV", fmt.value("hrv", mon.max()), fmt.month(mon.idxmax()))

    w = ctx.m.workouts
    if w is not None and len(w):
        longest = w.loc[w["duration_min"].idxmax()]
        add("Longest workout", fmt.minutes(longest["duration_min"]), fmt.date(longest["date"]), longest["label"])
        for act, label in (("Running", "Longest run"), ("Cycling", "Longest ride"), ("Walking", "Longest walk"),
                           ("Hiking", "Longest hike"), ("Swimming", "Longest swim")):
            sub = w[(w["activity"] == act) & w["distance_km"].notna()]
            if len(sub):
                r = sub.loc[sub["distance_km"].idxmax()]
                add(label, fmt.distance(r["distance_km"], 2), fmt.date(r["date"]),
                    fmt.minutes(r["duration_min"]))
        runs = w[(w["activity"] == "Running") & (w["distance_km"] >= 5) & w["pace_min_per_km"].notna()]
        if len(runs):
            r = runs.loc[runs["pace_min_per_km"].idxmin()]
            add("Fastest run of 5 km+ (avg pace)", fmt.pace(r["pace_min_per_km"]), fmt.date(r["date"]),
                fmt.distance(r["distance_km"], 2))
    return out
