"""The daily table: one row per local calendar day, one column per metric.

Conventions
-----------
* A day is the local calendar day of the sample (using the UTC offset stored
  with each timestamp). Samples crossing midnight are split proportionally.
* Missing is not zero: a day without any step samples has ``NaN`` steps, not 0.
* Cumulative metrics shared by Watch and iPhone are de-duplicated first
  (see :mod:`healthlens.metrics.dedup`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .. import catalog
from ..catalog import EVENT_TYPES, METRICS
from ..store import HealthStore
from .dedup import dedup_values


def _local(df: pd.DataFrame, col: str = "start") -> np.ndarray:
    off = "start_off" if col == "start" else "end_off"
    return df[col].to_numpy(np.int64) + df[off].to_numpy(np.int64) * 60


def daily_sum(df: pd.DataFrame, values: np.ndarray) -> pd.Series:
    """Sum values per local day, splitting samples that cross midnight."""
    if len(df) == 0:
        return pd.Series(dtype=float)
    ls = _local(df, "start")
    le = _local(df, "end")
    d0 = np.floor_divide(ls, 86400)
    d1 = np.maximum(d0, np.floor_divide(np.maximum(le - 1, ls), 86400))
    same = d1 == d0
    cross = (d1 == d0 + 1)
    multi = ~same & ~cross
    days = [d0[same], d0[multi]]
    vals = [values[same], values[multi]]
    if cross.any():
        boundary = (d0[cross] + 1) * 86400
        dur = np.maximum(le[cross] - ls[cross], 1)
        f = (boundary - ls[cross]) / dur
        days += [d0[cross], d1[cross]]
        vals += [values[cross] * f, values[cross] * (1 - f)]
    s = pd.Series(np.concatenate(vals)).groupby(np.concatenate(days)).sum()
    return s


def _by_day(df: pd.DataFrame) -> np.ndarray:
    return np.floor_divide(_local(df, "start"), 86400)


def compute_daily(store: HealthStore, nights: pd.DataFrame, naps: pd.DataFrame,
                  workouts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Return (daily, hourly, dedup_stats)."""
    cols: Dict[str, pd.Series] = {}
    dedup_stats: Dict[str, dict] = {}
    hourly_steps = None

    for m in METRICS:
        if not store.has(m.hk):
            continue
        df = store.values(m.hk, m.unit)
        if len(df) == 0:
            continue
        v = df["value"].to_numpy(np.float64)
        if m.agg == "sum":
            if m.dedup and df["source_id"].nunique() > 1:
                v2 = dedup_values(df)
                dedup_stats[m.key] = {"raw_total": float(v.sum()), "dedup_total": float(v2.sum())}
                v = v2
            cols[m.key] = daily_sum(df, v)
            if m.key == "steps":
                mid = (_local(df, "start") + _local(df, "end")) // 2
                hourly_steps = pd.Series(v).groupby(np.floor_divide(mid, 3600)).sum()
        else:
            day = _by_day(df)
            g = pd.Series(v).groupby(day)
            if m.agg == "mean":
                cols[m.key] = g.mean()
            elif m.agg == "last":
                cols[m.key] = g.last()
            elif m.agg == "max":
                cols[m.key] = g.max()
            elif m.agg == "min":
                cols[m.key] = g.min()
    # ---------------- heart rate ----------------
    hourly_hr = None
    if store.has("HeartRate"):
        hr = store.values("HeartRate", "bpm")
        hr = hr[(hr["value"] > 25) & (hr["value"] < 250)]
        hour = np.floor_divide(_local(hr), 3600)
        hmean = pd.Series(hr["value"].to_numpy()).groupby(hour).mean()
        hourly_hr = hmean
        hday = np.floor_divide(hmean.index.to_numpy(), 24)
        cols["hr_mean"] = pd.Series(hmean.to_numpy()).groupby(hday).mean()
        day = np.floor_divide(hour, 24)
        g = pd.Series(hr["value"].to_numpy()).groupby(day)
        cols["hr_min"] = g.min()
        cols["hr_max"] = g.max()
        watch = (hr["kind"] == "watch").to_numpy()
        if watch.any():
            wh = np.unique(hour[watch])
            cols["watch_wear_h"] = pd.Series(np.ones(len(wh))).groupby(np.floor_divide(wh, 24)).sum()
    # ---------------- category types ----------------
    if store.has("AppleStandHour"):
        st = store.records("AppleStandHour")
        st = st[st["cat"] == "AppleStandHourStood"]
        hours = np.unique(np.floor_divide(_local(st), 3600))
        cols["stand_hours"] = pd.Series(np.ones(len(hours))).groupby(np.floor_divide(hours, 24)).sum()
    if store.has("MindfulSession"):
        ms = store.records("MindfulSession")
        cols["mindful_min"] = daily_sum(ms, (ms["end"] - ms["start"]).to_numpy(np.float64) / 60)
    for key, hk in EVENT_TYPES.items():
        if store.has(hk):
            ev = store.records(hk)
            cols[key] = pd.Series(np.ones(len(ev))).groupby(_by_day(ev)).sum()

    daily = pd.DataFrame(cols)
    if len(daily) == 0:
        return pd.DataFrame(), pd.DataFrame(), dedup_stats
    daily.index = pd.to_datetime(daily.index.astype(np.int64), unit="D")

    # ---------------- activity rings ----------------
    act = store.activity_summary()
    if len(act):
        act = act.set_index("date")
        worn = (act["active_kcal"].fillna(0) > 0) | (act["exercise_min"].fillna(0) > 0) | \
               (act["stand_hours"].fillna(0) > 0)
        ring = pd.DataFrame(index=act.index)
        ring["move_goal_kcal"] = act["active_goal_kcal"]
        ring["exercise_goal_min"] = act["exercise_goal_min"]
        ring["stand_goal_h"] = act["stand_goal_h"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ring["move_ring_pct"] = np.where(act["active_goal_kcal"] > 0,
                                             100 * act["active_kcal"] / act["active_goal_kcal"], np.nan)
            ex_goal = act["exercise_goal_min"].where(act["exercise_goal_min"] > 0, 30.0)
            ring["exercise_ring_pct"] = 100 * act["exercise_min"] / ex_goal
            st_goal = act["stand_goal_h"].where(act["stand_goal_h"] > 0, 12.0)
            ring["stand_ring_pct"] = 100 * act["stand_hours"] / st_goal
        ring.loc[~worn, ["move_ring_pct", "exercise_ring_pct", "stand_ring_pct"]] = np.nan
        daily = daily.join(ring, how="outer")

    # ---------------- sleep (by morning of waking) ----------------
    if nights is not None and len(nights):
        n = nights.set_index("wake_date")
        daily = daily.join(pd.DataFrame({
            "sleep_h": n["sleep_h"], "in_bed_h": n["in_bed_h"],
            "sleep_any_h": n["sleep_h"].fillna(n["in_bed_h"]),
            "sleep_kind": n["kind"],
            "sleep_efficiency_pct": n["efficiency_pct"],
            "deep_h": n["deep_h"], "rem_h": n["rem_h"], "core_h": n["core_h"], "awake_h": n["awake_h"],
            "bedtime_h": n["bedtime_h"], "wake_h": n["wake_h"], "sleep_hr": n["sleep_hr"],
        }), how="outer")
    if naps is not None and len(naps):
        daily = daily.join(naps.set_index("wake_date")[["nap_h"]], how="outer")

    # ---------------- workouts ----------------
    if workouts is not None and len(workouts):
        g = workouts.groupby("date")
        daily = daily.join(pd.DataFrame({
            "workouts_n": g.size().astype(float), "workout_min": g["duration_min"].sum(),
            "workout_kcal": g["energy_kcal"].sum(min_count=1),
        }), how="outer")
        for c in ("workouts_n", "workout_min"):
            if c in daily:
                # days with watch data but no workout -> 0 workouts (not missing)
                has_watch = daily.get("watch_wear_h", pd.Series(index=daily.index, dtype=float)).fillna(0) > 0
                daily.loc[has_watch & daily[c].isna(), c] = 0.0

    # full calendar, trimmed of the partial export day
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    export_day = _export_day(store)
    if export_day is not None and daily.index.max() >= export_day:
        daily = daily[daily.index < export_day]
    daily.index.name = "date"

    hourly = _hourly_table(hourly_steps, hourly_hr, daily.index)
    return daily, hourly, dedup_stats


def _export_day(store: HealthStore) -> Optional[pd.Timestamp]:
    text = store.meta.get("export_date")
    if not text or len(text) < 10:
        return None
    try:
        return pd.Timestamp(text[:10])
    except ValueError:
        return None


def _hourly_table(steps: Optional[pd.Series], hr: Optional[pd.Series], days: pd.DatetimeIndex) -> pd.DataFrame:
    parts = {}
    if steps is not None:
        parts["steps"] = steps
    if hr is not None:
        parts["hr"] = hr
    if not parts:
        return pd.DataFrame(columns=["date", "hour", "steps", "hr"])
    h = pd.DataFrame(parts)
    idx = h.index.to_numpy(np.int64)
    h["date"] = pd.to_datetime(np.floor_divide(idx, 24), unit="D")
    h["hour"] = (idx % 24).astype(np.int8)
    h = h.reset_index(drop=True)
    if len(days):
        h = h[(h["date"] >= days.min()) & (h["date"] <= days.max())]
    for c in ("steps", "hr"):
        if c not in h:
            h[c] = np.nan
    return h[["date", "hour", "steps", "hr"]].reset_index(drop=True)


def metric_columns(daily: pd.DataFrame):
    """Metric objects for the numeric columns present in ``daily``."""
    return [catalog.ALL_METRICS[c] for c in daily.columns if c in catalog.ALL_METRICS]
