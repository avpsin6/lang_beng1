"""Workout clean-up: local dates, duplicates across apps, pace and HR zones."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..catalog import PACE_WORKOUTS, workout_label

OVERLAP_DUPLICATE = 0.5    # >50 % time overlap with the same activity = same workout
ZONE_EDGES = (0.6, 0.7, 0.8, 0.9)   # fractions of estimated max heart rate


def max_hr_estimate(age: Optional[float]) -> Optional[float]:
    """Tanaka et al. (2001): 208 - 0.7 x age. A population estimate only."""
    if age is None or not np.isfinite(age):
        return None
    return 208.0 - 0.7 * float(age)


def clean_workouts(raw: pd.DataFrame, heart_rate: Optional[pd.DataFrame] = None,
                   age: Optional[float] = None) -> pd.DataFrame:
    if raw is None or len(raw) == 0 or "start" not in raw:
        return pd.DataFrame(columns=["date", "activity", "label", "duration_min", "distance_km", "energy_kcal"])
    w = raw.copy()
    w = w[(w["end"] > w["start"]) & (w["duration_min"].fillna(0) > 0.5)]
    w["local_start"] = w["start"] + w["start_off"].astype(np.int64) * 60
    w["date"] = pd.to_datetime(np.floor_divide(w["local_start"].to_numpy(np.int64), 86400), unit="D")
    w["hour"] = (w["local_start"] % 86400) / 3600.0
    w["label"] = w["activity"].map(workout_label)
    w, n_duplicates = _drop_duplicates(w)

    # fill HR stats for older exports from the heart-rate samples
    zone_cols = [f"z{i}_min" for i in range(1, 6)]
    for c in zone_cols:
        w[c] = np.nan
    w["hr_from_samples"] = False
    if heart_rate is not None and len(heart_rate):
        hr = heart_rate[heart_rate["kind"].isin(["watch", "app"])]
        t = hr["start"].to_numpy(np.int64)
        order = np.argsort(t, kind="stable")
        t = t[order]
        v = hr["value"].to_numpy(np.float64)[order]
        hrmax = max_hr_estimate(age)
        avg, mx, zones = [], [], []
        for s, e in zip(w["start"].to_numpy(np.int64), w["end"].to_numpy(np.int64)):
            lo, hi = np.searchsorted(t, s), np.searchsorted(t, e)
            if hi - lo < 3:
                avg.append(np.nan)
                mx.append(np.nan)
                zones.append([np.nan] * 5)
                continue
            seg_t, seg_v = t[lo:hi], v[lo:hi]
            dt = np.minimum(np.diff(np.r_[seg_t, e]), 120)          # cap gaps at 2 min
            avg.append(float(np.average(seg_v, weights=np.maximum(dt, 1))))
            mx.append(float(seg_v.max()))
            if hrmax:
                idx = np.digitize(seg_v / hrmax, ZONE_EDGES)
                zones.append([float(dt[idx == z].sum()) / 60 for z in range(5)])
            else:
                zones.append([np.nan] * 5)
        avg = np.array(avg)
        mx = np.array(mx)
        missing = w["avg_hr"].isna().to_numpy()
        w.loc[missing, "avg_hr"] = avg[missing]
        w["hr_from_samples"] = missing & np.isfinite(avg)
        miss_max = w["max_hr"].isna().to_numpy()
        w.loc[miss_max, "max_hr"] = mx[miss_max]
        zones = np.array(zones, dtype=float).reshape(-1, 5)
        for i, c in enumerate(zone_cols):
            w[c] = zones[:, i]

    w["pace_min_per_km"] = np.where(
        w["activity"].isin(PACE_WORKOUTS) & (w["distance_km"] > 0.3),
        w["duration_min"] / w["distance_km"], np.nan)
    w["speed_kmh"] = np.where(w["distance_km"] > 0.3, w["distance_km"] / (w["duration_min"] / 60), np.nan)
    w = w.sort_values("start").reset_index(drop=True)
    w.attrs["duplicates_removed"] = n_duplicates
    return w


def _drop_duplicates(w: pd.DataFrame):
    """Same session recorded by two apps (e.g. Watch + Strava): keep the best source."""
    if len(w) < 2:
        return w, 0
    prio = {"watch": 1, "phone": 2, "app": 3, "manual": 4}
    w = w.sort_values("start").reset_index(drop=True)
    w["_prio"] = w["kind"].map(prio).fillna(5) if "kind" in w else 5
    start = w["start"].to_numpy(np.int64)
    end = w["end"].to_numpy(np.int64)
    act = w["activity"].to_numpy(dtype=object)
    pr = w["_prio"].to_numpy()
    src = w["source_id"].to_numpy()
    drop = np.zeros(len(w), dtype=bool)
    for i in range(len(w)):
        if drop[i]:
            continue
        j = i + 1
        while j < len(w) and start[j] < end[i]:
            if not drop[j] and src[j] != src[i] and act[j] == act[i]:
                overlap = min(end[i], end[j]) - max(start[i], start[j])
                shorter = max(1, min(end[i] - start[i], end[j] - start[j]))
                if overlap / shorter > OVERLAP_DUPLICATE:
                    if pr[j] < pr[i]:
                        drop[i] = True
                        break
                    drop[j] = True
            j += 1
    kept = w[~drop].drop(columns="_prio").copy()
    return kept, int(drop.sum())
