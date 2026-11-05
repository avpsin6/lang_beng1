"""Aggregations shared by the insights, the report and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..catalog import ALL_METRICS
from ..metrics import Metrics
from ..units import DisplayUnits
from .fmt import Formatter

# Metrics where a per-day *average* is the natural monthly/yearly summary
PER_DAY = {"steps", "distance_km", "flights", "active_kcal", "basal_kcal", "exercise_min", "stand_min",
           "stand_hours", "daylight_min", "mindful_min", "cycling_km", "swimming_m", "watch_wear_h"}
# Metrics summarised by the median (robust to odd readings)
MEDIAN = {"resting_hr", "walking_hr", "hrv", "vo2max", "weight_kg", "bmi", "body_fat_pct", "spo2", "resp_rate",
          "sleep_hr", "hr_mean", "walking_speed_kmh", "step_length_cm", "walking_asymmetry_pct",
          "double_support_pct", "env_audio_db", "headphone_audio_db", "wrist_temp_c"}
# Totals per period
TOTALS = {"workouts_n", "workout_min", "workout_kcal", "high_hr_events", "low_hr_events", "irregular_rhythm_events"}
MIN_DAYS_PER_MONTH = 10
MIN_DAYS_PER_YEAR = 60

WORN_DAY_HOURS = 10.0   # a day counts as "Watch worn" with >= 10 h of heart-rate coverage


def month_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx.values.astype("datetime64[M]").astype("datetime64[ns]"))


def numeric_columns(daily: pd.DataFrame) -> List[str]:
    return [c for c in daily.columns if c in ALL_METRICS and pd.api.types.is_numeric_dtype(daily[c])]


def aggregate(daily: pd.DataFrame, keys: pd.DatetimeIndex, min_days: int) -> pd.DataFrame:
    """Summarise numeric daily columns by the given period keys (one per row)."""
    cols = numeric_columns(daily)
    g = daily[cols].groupby(keys)
    out = {}
    counts = g.count()
    for c in cols:
        if c in TOTALS:
            out[c] = g[c].sum(min_count=1)
        elif c in MEDIAN:
            out[c] = g[c].median()
        else:
            out[c] = g[c].mean()
        if c in PER_DAY:
            out[c] = out[c].where(counts[c] >= min_days)
    res = pd.DataFrame(out)
    for c in cols:
        res[f"n_{c}"] = counts[c]
    res.index.name = "period"
    return res


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp       # inclusive

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= self.start) & (df.index <= self.end)]

    @property
    def label(self) -> str:
        return f"{Formatter.date(self.start)} – {Formatter.date(self.end)}"


@dataclass
class Context:
    """Everything an insight generator needs."""
    m: Metrics
    fmt: Formatter
    daily: pd.DataFrame
    monthly: pd.DataFrame
    yearly: pd.DataFrame
    weekly: pd.DataFrame
    recent: Window
    previous: Window
    last_day: pd.Timestamp
    first_day: pd.Timestamp
    worn: pd.Series                     # bool per day
    extras: Dict[str, object] = field(default_factory=dict)

    def has(self, col: str, min_count: int = 1) -> bool:
        return col in self.daily and int(self.daily[col].notna().sum()) >= min_count


def build_context(m: Metrics, units: Optional[str] = None) -> Context:
    daily = m.daily.copy()
    system = units or m.unit_system
    locale = (m.meta.get("locale") or "").lower()
    fmt = Formatter(DisplayUnits(system), clock24=not locale.startswith("en_us"))
    if len(daily) == 0:
        raise ValueError("The export contains no daily data to analyse.")
    # last day with meaningful data (any key metric)
    key_cols = [c for c in ("steps", "resting_hr", "active_kcal", "hr_mean", "sleep_any_h") if c in daily]
    has_data = daily[key_cols].notna().any(axis=1) if key_cols else pd.Series(True, index=daily.index)
    last_day = has_data[has_data].index.max()
    first_day = has_data[has_data].index.min()
    daily = daily.loc[first_day:last_day]

    worn = (daily["watch_wear_h"].fillna(0) >= WORN_DAY_HOURS) if "watch_wear_h" in daily else \
        pd.Series(False, index=daily.index)

    monthly = aggregate(daily, month_index(daily.index), MIN_DAYS_PER_MONTH)
    years = pd.DatetimeIndex(daily.index.values.astype("datetime64[Y]").astype("datetime64[ns]"))
    yearly = aggregate(daily, years, MIN_DAYS_PER_YEAR)
    yearly.index = yearly.index.year

    # weekly exercise (Mon-Sun) for the WHO 150-min guideline
    week = daily.index - pd.to_timedelta(daily.index.day_of_week, unit="D")
    wk = pd.DataFrame({"worn_days": worn.groupby(week).sum()})
    if "exercise_min" in daily:
        wk["exercise_min"] = daily["exercise_min"].fillna(0).where(worn).groupby(week).sum(min_count=1)
    if "workouts_n" in daily:
        wk["workouts"] = daily["workouts_n"].groupby(week).sum(min_count=1)
    if "steps" in daily:
        wk["steps"] = daily["steps"].groupby(week).mean()
    wk.index.name = "week"

    recent = Window(last_day - pd.Timedelta(days=364), last_day)
    previous = Window(last_day - pd.Timedelta(days=729), last_day - pd.Timedelta(days=365))
    return Context(m=m, fmt=fmt, daily=daily, monthly=monthly, yearly=yearly, weekly=wk, recent=recent,
                   previous=previous, last_day=last_day, first_day=first_day, worn=worn)


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
KPI_KEYS = ["steps", "exercise_min", "resting_hr", "hrv", "vo2max", "sleep_any_h", "workouts_n", "weight_kg"]


def window_value(ctx: Context, key: str, window: Window) -> Optional[float]:
    if key not in ctx.daily:
        return None
    s = window.slice(ctx.daily)[key].dropna()
    min_n = 1 if key in ("vo2max", "weight_kg") else (5 if key == "sleep_any_h" else 20)
    if key == "workouts_n":
        sub = window.slice(ctx.daily)
        worn_days = int(ctx.worn.loc[sub.index].sum()) if len(sub) else 0
        if worn_days < 30:
            return None
        return float(sub["workouts_n"].fillna(0).sum() / (len(sub) / 7))
    if len(s) < min_n:
        return None
    if key == "exercise_min":
        worn = ctx.worn.loc[s.index]
        s = s.where(worn).dropna()
        if len(s) < 20:
            return None
        return float(window.slice(ctx.daily)["exercise_min"].fillna(0).where(ctx.worn).mean())
    if key in MEDIAN:
        return float(s.median())
    return float(s.mean())


def kpis(ctx: Context) -> List[dict]:
    tiles = []
    for key in KPI_KEYS:
        now = window_value(ctx, key, ctx.recent)
        if now is None:
            continue
        before = window_value(ctx, key, ctx.previous)
        m = ALL_METRICS[key]
        spark = ctx.monthly[key].dropna() if key in ctx.monthly else pd.Series(dtype=float)
        label = m.label
        if key == "sleep_any_h":
            measured = ctx.recent.slice(ctx.daily).get("sleep_h", pd.Series(dtype=float)).notna().sum()
            label = "Sleep" if measured >= 0.5 * ctx.recent.slice(ctx.daily)["sleep_any_h"].notna().sum() \
                else "Time in bed"
        if key == "workouts_n":
            label = "Workouts / week"
        if key == "exercise_min":
            label = "Exercise / day"
        tiles.append({
            "key": key, "label": label, "value": now, "previous": before,
            "delta": (now - before) if before is not None else None,
            "better": m.better, "spark": spark,
        })
    return tiles


# ---------------------------------------------------------------------------
# Year-in-review table
# ---------------------------------------------------------------------------
YEAR_ROWS = ["steps", "distance_km", "exercise_min", "active_kcal", "stand_hours", "workouts_n", "workout_min",
             "resting_hr", "walking_hr", "hrv", "vo2max", "weight_kg", "sleep_h", "in_bed_h", "bedtime_h",
             "move_ring_pct", "exercise_ring_pct", "stand_ring_pct"]


def yearly_table(ctx: Context) -> pd.DataFrame:
    y = ctx.yearly.copy()
    d = ctx.daily
    years = sorted(set(d.index.year))
    out = pd.DataFrame(index=years)
    out.index.name = "year"
    for key in YEAR_ROWS:
        if key not in y:
            continue
        if key in ("workouts_n", "workout_min"):
            out[key] = d[key].groupby(d.index.year).sum(min_count=1) if key in d else np.nan
        elif key in ("move_ring_pct", "exercise_ring_pct", "stand_ring_pct"):
            closed = (d[key] >= 100).where(d[key].notna())
            out[key] = closed.groupby(d.index.year).mean() * 100
        elif key in ("sleep_h", "in_bed_h", "bedtime_h"):
            g = d[key].groupby(d.index.year)
            out[key] = g.mean().where(g.count() >= 10)
        else:
            out[key] = y[key]
    out["days"] = d.groupby(d.index.year).size()
    out["worn_days"] = ctx.worn.groupby(d.index.year).sum()
    if "sleep_any_h" in d:
        out["nights"] = d["sleep_any_h"].notna().groupby(d.index.year).sum()
    if "sleep_h" in d:
        out["nights_measured"] = d["sleep_h"].notna().groupby(d.index.year).sum()
    return out
