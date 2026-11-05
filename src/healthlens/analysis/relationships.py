"""Which of your metrics move together?

For a fixed list of plausible pairs (e.g. "exercise today -> sleep tonight") we

1. remove slow changes from both series (centred 29-day rolling median), so
   seasons or fitness phases don't create spurious links;
2. compute a Spearman correlation on the day-to-day residuals, with an
   effective sample size that accounts for autocorrelation;
3. correct for testing many pairs (Benjamini-Hochberg);
4. translate surviving links into plain numbers: how the outcome differed
   between your top-third and bottom-third days (block-bootstrap 95 % CI).

These are associations in *your* data, not proof of cause and effect.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .core import Context
from .stats import block_bootstrap_diff, bh_adjust, detrend_rolling, spearman


@dataclass(frozen=True)
class Pair:
    x: str
    y: str
    lag: int               # pair x on day d with y on day d + lag
    x_words: str           # "exercise"
    y_words: str           # "sleep that night"
    question: str


PAIRS = [
    Pair("exercise_min", "sleep_any_h", 1, "exercise", "sleep that night", "Does exercise change how long you sleep?"),
    Pair("steps", "sleep_any_h", 1, "steps", "sleep that night", "Do active days change how long you sleep?"),
    Pair("workout_min", "sleep_any_h", 1, "workout time", "sleep that night", "Do workouts change your sleep?"),
    Pair("sleep_any_h", "resting_hr", 0, "sleep", "resting heart rate the next day",
         "Does sleep affect your resting heart rate?"),
    Pair("sleep_any_h", "hrv", 0, "sleep", "HRV the next day", "Does sleep affect your HRV?"),
    Pair("sleep_any_h", "steps", 0, "sleep", "steps the next day", "Do you move more after a longer night?"),
    Pair("sleep_any_h", "exercise_min", 0, "sleep", "exercise the next day",
         "Do you exercise more after a longer night?"),
    Pair("exercise_min", "resting_hr", 1, "exercise", "resting heart rate the next day",
         "Does a hard day show up in tomorrow's resting heart rate?"),
    Pair("exercise_min", "hrv", 1, "exercise", "HRV the next day", "Does a hard day show up in tomorrow's HRV?"),
    Pair("bedtime_h", "sleep_any_h", 0, "a later bedtime", "sleep that night", "Do late nights cost you sleep?"),
    Pair("steps", "resting_hr", 1, "steps", "resting heart rate the next day",
         "Is a busy day followed by a higher resting heart rate?"),
    Pair("daylight_min", "sleep_any_h", 1, "time in daylight", "sleep that night",
         "Does daylight change your sleep?"),
    Pair("resting_hr", "hrv", 0, "resting heart rate", "HRV the same day", "Do resting HR and HRV move together?"),
]

MONTHLY_PAIRS = [
    Pair("exercise_min", "resting_hr", 0, "exercise", "resting heart rate", "Long-term: exercise and resting HR"),
    Pair("steps", "resting_hr", 0, "steps", "resting heart rate", "Long-term: activity and resting HR"),
    Pair("exercise_min", "vo2max", 0, "exercise", "cardio fitness (VO₂ max)", "Long-term: exercise and fitness"),
    Pair("steps", "weight_kg", 0, "steps", "weight", "Long-term: activity and weight"),
]


@dataclass
class Finding:
    pair: Pair
    level: str            # "daily" | "monthly"
    rho: float
    n: int
    n_eff: float
    p: float
    q: float = 1.0
    low_cut: float = np.nan
    high_cut: float = np.nan
    y_low: float = np.nan
    y_high: float = np.nan
    diff: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    n_low: int = 0
    n_high: int = 0

    @property
    def strength(self) -> str:
        r = abs(self.rho)
        return "strong" if r >= 0.5 else "moderate" if r >= 0.3 else "weak" if r >= 0.1 else "very weak"

    @property
    def significant(self) -> bool:
        return self.q < 0.05 and abs(self.rho) >= 0.08 and (
            self.level == "monthly" or (np.isfinite(self.ci_low) and (self.ci_low > 0 or self.ci_high < 0)))


def _aligned(daily: pd.DataFrame, p: Pair, detrend: bool = True):
    if p.x not in daily or p.y not in daily:
        return None
    x = daily[p.x].astype(float)
    y = daily[p.y].astype(float).shift(-p.lag)
    if detrend:
        xr = detrend_rolling(x)
        yr = detrend_rolling(daily[p.y].astype(float)).shift(-p.lag)
    else:
        xr, yr = x, y
    df = pd.DataFrame({"x": x, "y": y, "xr": xr, "yr": yr}).dropna()
    return df


def daily_findings(ctx: Context, min_n: int = 40) -> List[Finding]:
    out: List[Finding] = []
    for p in PAIRS:
        df = _aligned(ctx.daily, p)
        if df is None or len(df) < min_n:
            continue
        c = spearman(df["xr"].to_numpy(), df["yr"].to_numpy())
        if c is None:
            continue
        f = Finding(pair=p, level="daily", rho=c.rho, n=c.n, n_eff=c.n_eff, p=c.p_value)
        lo, hi = df["x"].quantile([1 / 3, 2 / 3])
        low = df["x"] <= lo
        high = df["x"] >= hi if hi > lo else df["x"] > lo
        if low.sum() >= 10 and high.sum() >= 10:
            sel = df[low | high]
            # outcome adjusted for slow drifts, expressed around your overall mean
            adj = sel["yr"] + df["y"].mean()
            d = block_bootstrap_diff(adj, high[low | high], seed=zlib.crc32((p.x + p.y).encode()))
            if d is not None:
                f.low_cut, f.high_cut = float(lo), float(hi)
                f.y_low, f.y_high = d.mean_b, d.mean_a
                f.diff, f.ci_low, f.ci_high = d.diff, d.low, d.high
                f.n_low, f.n_high = d.n_b, d.n_a
        out.append(f)
    qs = bh_adjust([f.p for f in out])
    for f, q in zip(out, qs):
        f.q = q
    return out


def monthly_findings(ctx: Context, min_n: int = 18) -> List[Finding]:
    out: List[Finding] = []
    mon = ctx.monthly
    for p in MONTHLY_PAIRS:
        if p.x not in mon or p.y not in mon:
            continue
        df = mon[[p.x, p.y]].dropna()
        if len(df) < min_n:
            continue
        c = spearman(df[p.x].to_numpy(), df[p.y].to_numpy())
        if c is None:
            continue
        f = Finding(pair=p, level="monthly", rho=c.rho, n=c.n, n_eff=c.n_eff, p=c.p_value)
        lo, hi = df[p.x].quantile([1 / 3, 2 / 3])
        f.low_cut, f.high_cut = float(lo), float(hi)
        f.y_low = float(df.loc[df[p.x] <= lo, p.y].mean())
        f.y_high = float(df.loc[df[p.x] >= hi, p.y].mean())
        f.diff = f.y_high - f.y_low
        f.n_low = int((df[p.x] <= lo).sum())
        f.n_high = int((df[p.x] >= hi).sum())
        out.append(f)
    qs = bh_adjust([f.p for f in out])
    for f, q in zip(out, qs):
        f.q = q
    return out


def _x_phrases(x: str, words: str, lo: float, hi: float, fmt):
    """('On days with 45 min or more of exercise', 'on days with 5 min or less')."""
    if x == "bedtime_h":
        return f"On nights you went to bed after {fmt.clock(hi)}", f"on nights before {fmt.clock(lo)}"
    if x == "sleep_any_h":
        return (f"After nights with {fmt.hours(hi)} or more of sleep",
                f"after nights with {fmt.hours(lo)} or less")
    if x == "steps":
        return f"On days with {fmt.num(hi)}+ steps", f"on days with {fmt.num(lo)} or fewer"
    if x == "resting_hr":
        return (f"On days your resting heart rate was {fmt.value(x, hi)} or more",
                f"on days at {fmt.value(x, lo)} or less")
    return (f"On days with {fmt.value(x, hi)} or more of {words}", f"on days with {fmt.value(x, lo)} or less")


def describe(f: Finding, ctx: Context) -> str:
    """One plain-language sentence for a finding."""
    fmt = ctx.fmt
    p = f.pair
    if f.level == "monthly":
        return (f"In months with more {p.x_words} (top third: {fmt.value(p.x, f.high_cut)} a day or more), "
                f"your {p.y_words} averaged {fmt.value(p.y, f.y_high)}, versus {fmt.value(p.y, f.y_low)} in "
                f"months with the least {p.x_words} ({fmt.value(p.x, f.low_cut)} or less).")
    when_high, when_low = _x_phrases(p.x, p.x_words, f.low_cut, f.high_cut, fmt)
    if p.y in ("sleep_any_h",):
        direction = "more" if f.diff > 0 else "less"
    else:
        direction = "higher" if f.diff > 0 else "lower"
    amount = fmt.delta(p.y, abs(f.diff)).lstrip("+−")
    return (f"{when_high}, {p.y_words} averaged {fmt.value(p.y, f.y_high)} vs {fmt.value(p.y, f.y_low)} "
            f"{when_low} — {amount} {direction}.")
