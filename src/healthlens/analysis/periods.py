"""Notable periods, long-term "chapters" and recent changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .core import Context
from .stats import change_points, robust_baseline


# ---------------------------------------------------------------------------
# Notable periods: resting heart rate well above your own baseline
# ---------------------------------------------------------------------------
@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    peak: float            # highest resting HR in the period
    baseline: float        # your usual resting HR just before
    excess: float          # peak - baseline
    context: Dict[str, float] = field(default_factory=dict)   # % change in HRV / steps, sleep diff (h)


def elevated_rhr_periods(ctx: Context, min_excess: float = 3.0, z: float = 2.0, max_periods: int = 6) -> List[Period]:
    d = ctx.daily
    if "resting_hr" not in d or d["resting_hr"].notna().sum() < 60:
        return []
    rhr = d["resting_hr"]
    med, mad = robust_baseline(rhr, window=60, lag=7, min_periods=20)
    scale = mad.clip(lower=1.5)
    dev = rhr - med
    flag = ((dev >= min_excess) & (dev / scale >= z)).fillna(False).to_numpy()
    idx = d.index
    periods: List[Period] = []
    i, n = 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (flag[j + 1] or (j + 2 < n and flag[j + 2] and np.isnan(rhr.iloc[j + 1]))):
            j += 1
        flagged = int(flag[i:j + 1].sum())
        seg = rhr.iloc[i:j + 1]
        base = float(med.iloc[i])
        if flagged >= 3 or (flagged >= 2 and float(seg.max()) - base >= 5):
            p = Period(start=idx[i], end=idx[j], days=j - i + 1, peak=float(seg.max()), baseline=base,
                       excess=float(seg.max() - base))
            p.context = _context(d, i, j)
            periods.append(p)
        i = j + 1
    periods.sort(key=lambda p: p.excess * np.sqrt(p.days), reverse=True)
    return sorted(periods[:max_periods], key=lambda p: p.start, reverse=True)


def _context(d: pd.DataFrame, i: int, j: int) -> Dict[str, float]:
    """How other signals behaved during the period vs the 30 days before."""
    out: Dict[str, float] = {}
    lo = max(0, i - 37)
    for col in ("hrv", "steps", "sleep_any_h", "resp_rate"):
        if col not in d:
            continue
        before = d[col].iloc[lo:max(lo, i - 7)].median()
        during = d[col].iloc[i:j + 1].mean()
        if not (np.isfinite(before) and np.isfinite(during)) or before == 0:
            continue
        if col == "sleep_any_h" or col == "resp_rate":
            out[col] = float(during - before)
        else:
            out[col] = float(100 * (during - before) / before)
    return out


# ---------------------------------------------------------------------------
# Long-term chapters (change points in monthly means)
# ---------------------------------------------------------------------------
@dataclass
class Chapter:
    start: pd.Timestamp
    end: pd.Timestamp
    mean: float
    months: int


def chapters(ctx: Context, key: str, max_points: int = 3) -> List[Chapter]:
    if key not in ctx.monthly:
        return []
    s = ctx.monthly[key].dropna()
    if len(s) < 12:
        return []
    cps = change_points(s.to_numpy(), min_size=3, max_points=max_points)
    bounds = [0] + cps + [len(s)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = s.iloc[a:b]
        end = seg.index[-1] + pd.offsets.MonthEnd(0)
        out.append(Chapter(start=seg.index[0], end=end, mean=float(seg.mean()), months=len(seg)))
    return out


# ---------------------------------------------------------------------------
# Recent changes: the last 30 days vs your usual
# ---------------------------------------------------------------------------
@dataclass
class Recent:
    key: str
    recent: float
    usual: float
    z: float
    n_recent: int


RECENT_KEYS = ["steps", "exercise_min", "active_kcal", "resting_hr", "hrv", "walking_hr", "sleep_any_h",
               "bedtime_h", "vo2max", "weight_kg", "resp_rate"]


def recent_changes(ctx: Context, days: int = 30) -> List[Recent]:
    d = ctx.daily
    end = ctx.last_day
    start = end - pd.Timedelta(days=days - 1)
    out = []
    for key in RECENT_KEYS:
        if key not in d:
            continue
        s = d[key]
        if key == "exercise_min":
            s = s.where(ctx.worn)
        recent = s.loc[start:end].dropna()
        base = s.loc[start - pd.Timedelta(days=365):start - pd.Timedelta(days=1)]
        min_recent = 3 if key in ("vo2max", "weight_kg") else 10
        if len(recent) < min_recent or base.notna().sum() < 60:
            continue
        roll = base.rolling(days, min_periods=max(3, min_recent)).mean().dropna()
        sd = float(roll.std()) if len(roll) > 10 else float("nan")
        usual = float(base.mean())
        if not np.isfinite(sd) or sd == 0:
            continue
        val = float(recent.mean())
        out.append(Recent(key=key, recent=val, usual=usual, z=(val - usual) / sd, n_recent=len(recent)))
    return out
