"""Turning raw sleep samples into one row per night.

Sleep data in Apple Health is messy, especially when the Watch is only worn to
bed occasionally:

* iPhone Bedtime / Sleep Schedule writes **in bed** periods (from schedule and
  phone use) — a rough proxy for time in bed, *not* measured sleep;
* the Watch (watchOS 7+) writes **asleep** periods; from watchOS 9 these are
  split into Core / Deep / REM stages plus Awake;
* third-party apps (AutoSleep, Pillow, ...) write their own versions of both.

Approach
--------
1. Map values to stages (inbed, asleep, core, deep, rem, awake).
2. Cluster all samples into *episodes* (gaps < 90 min join).
3. Per episode, pick ONE source for "asleep" (Watch > apps > iPhone) and one
   for "in bed" (iPhone first) — never add sources together, which would
   double count.
4. Episodes centred between 18:00 and 14:00 belong to a night (dated by the
   morning you woke up) and are combined, so split sleep counts fully; if the
   night totals < 3 h, or an episode is centred 14:00-18:00, it is a nap.
5. Each night records what kind of data it is based on: ``stages``,
   ``asleep`` or ``inbed`` — the report keeps these apart.

Local times use the sample's ``HKTimeZone`` metadata when present (so travel
doesn't make you look like you slept at noon), else the export's offset.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..catalog import ASLEEP_STAGES, SLEEP_STAGE
from .dedup import _union

EPISODE_GAP = 90 * 60
MAIN_SLEEP_MIN_H = 3.0
NIGHT_START_H = 18      # an episode centred between 18:00 and 14:00 belongs to a night ...
NAP_FROM_H = 14         # ... one centred 14:00-18:00 is a daytime nap
SLEEP_PRIORITY = {"watch": 1, "app": 2, "phone": 3, "manual": 4}
INBED_PRIORITY = {"phone": 1, "watch": 2, "app": 3, "manual": 4}

NIGHT_COLUMNS = ["wake_date", "kind", "sleep_h", "in_bed_h", "core_h", "deep_h", "rem_h", "asleep_unspec_h",
                 "awake_h", "efficiency_pct", "bedtime_h", "onset_h", "wake_h", "midsleep_h", "awakenings",
                 "sleep_source", "inbed_source", "start_utc", "end_utc", "onset_utc", "final_wake_utc",
                 "sleep_hr", "segments"]


def _tz_offsets(df: pd.DataFrame) -> np.ndarray:
    """Offset (seconds) to local time for each sample, honouring HKTimeZone."""
    off = df["start_off"].to_numpy(dtype=np.int64) * 60
    if "tz" not in df or df["tz"].isna().all():
        return off
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover
        return off
    from datetime import datetime, timezone

    out = off.copy()
    tzs = df["tz"].to_numpy(dtype=object)
    starts = df["start"].to_numpy(dtype=np.int64)
    cache: Dict[str, Optional[object]] = {}
    for i in np.flatnonzero(pd.notna(df["tz"]).to_numpy()):
        name = tzs[i]
        if name not in cache:
            try:
                cache[name] = ZoneInfo(name)
            except Exception:  # unknown zone / missing tzdata
                cache[name] = None
        zone = cache[name]
        if zone is None:
            continue
        dt = datetime.fromtimestamp(int(starts[i]), tz=timezone.utc).astimezone(zone)
        out[i] = int(dt.utcoffset().total_seconds())
    return out


def _union_len(s: np.ndarray, e: np.ndarray) -> float:
    if len(s) == 0:
        return 0.0
    a, b = _union(np.empty(0), np.empty(0), s.astype(float), e.astype(float))
    return float((b - a).sum())


def build_nights(records: pd.DataFrame, heart_rate: Optional[pd.DataFrame] = None):
    """Return (nights, naps) DataFrames from SleepAnalysis records."""
    empty = pd.DataFrame(columns=NIGHT_COLUMNS)
    if records is None or len(records) == 0:
        return empty, pd.DataFrame(columns=["wake_date", "nap_h"])
    df = records.copy()
    df["stage"] = df["cat"].map(SLEEP_STAGE)
    df = df[df["stage"].notna() & (df["end"] > df["start"])].copy()
    if len(df) == 0:
        return empty, pd.DataFrame(columns=["wake_date", "nap_h"])
    df["local_off"] = _tz_offsets(df)
    df = df.sort_values("start", kind="stable").reset_index(drop=True)

    # 1) episodes
    start = df["start"].to_numpy(dtype=np.int64)
    end = df["end"].to_numpy(dtype=np.int64)
    run_end = np.maximum.accumulate(end)
    new = np.r_[True, start[1:] > run_end[:-1] + EPISODE_GAP]
    df["episode"] = np.cumsum(new) - 1

    hr_t = hr_v = None
    if heart_rate is not None and len(heart_rate):
        hr = heart_rate[heart_rate["kind"] == "watch"]
        hr_t = hr["start"].to_numpy(dtype=np.int64)
        order = np.argsort(hr_t, kind="stable")
        hr_t = hr_t[order]
        hr_v = hr["value"].to_numpy(dtype=np.float64)[order]

    # 2) classify episodes: night-window (midpoint 18:00-14:00) vs daytime nap
    summaries = []
    for ep_id, ep in df.groupby("episode", sort=True):
        dur, mid_local = _episode_duration(ep)
        if dur is None:
            continue
        night_day = int(np.floor((mid_local - NIGHT_START_H * 3600) / 86400)) + 1
        hour = (mid_local % 86400) / 3600
        summaries.append((ep_id, dur, night_day, NAP_FROM_H <= hour < NIGHT_START_H))
    if not summaries:
        return empty, pd.DataFrame(columns=["wake_date", "nap_h"])
    summ = pd.DataFrame(summaries, columns=["episode", "duration_h", "night_day", "daytime"])

    # 3) combine all night-window episodes of the same night; short totals become naps
    rows, nap_rows = [], []
    night_eps = summ[~summ["daytime"]]
    for night_day, g in night_eps.groupby("night_day", sort=True):
        if g["duration_h"].sum() >= MAIN_SLEEP_MIN_H:
            row = _episode(df[df["episode"].isin(g["episode"])], hr_t, hr_v, int(night_day))
            if row is not None:
                rows.append(row)
        else:
            nap_rows.extend((int(night_day), d) for d in g["duration_h"])
    nap_rows.extend((int(r.night_day), r.duration_h) for r in summ[summ["daytime"]].itertuples())
    naps = pd.DataFrame(nap_rows, columns=["night_day", "nap_h"])
    if len(naps):
        naps = naps.groupby("night_day", as_index=False)["nap_h"].sum()
        naps["wake_date"] = pd.to_datetime(naps["night_day"], unit="D")
        naps = naps[["wake_date", "nap_h"]]
    else:
        naps = pd.DataFrame(columns=["wake_date", "nap_h"])
    if not rows:
        return empty, naps
    nights = pd.DataFrame(rows)
    return nights[NIGHT_COLUMNS].sort_values("wake_date").reset_index(drop=True), naps


def _episode_duration(ep: pd.DataFrame):
    """(hours asleep — or in bed if no sleep data —, local midpoint seconds)."""
    off = int(ep["local_off"].iloc[0])
    sleep_sid, asleep = _pick_source(ep, ASLEEP_STAGES, SLEEP_PRIORITY)
    inbed_sid, inbed = _pick_source(ep, ("inbed",), INBED_PRIORITY)
    chosen = asleep if sleep_sid is not None else inbed
    if len(chosen) == 0:
        return None, None
    dur = _union_len(chosen["start"].to_numpy(), chosen["end"].to_numpy()) / 3600
    mid = (int(chosen["start"].min()) + int(chosen["end"].max())) / 2 + off
    return dur, mid


def _pick_source(ep: pd.DataFrame, stages, priority: Dict[str, int]):
    sub = ep[ep["stage"].isin(stages)]
    if len(sub) == 0:
        return None, sub
    best = None
    for sid, g in sub.groupby("source_id"):
        total = _union_len(g["start"].to_numpy(), g["end"].to_numpy())
        key = (priority.get(g["kind"].iloc[0], 9), -total)
        if best is None or key < best[0]:
            best = (key, sid)
    chosen = sub[sub["source_id"] == best[1]]
    return best[1], chosen


def _episode(ep: pd.DataFrame, hr_t, hr_v, night_day: int) -> Optional[dict]:
    off = int(ep["local_off"].iloc[0])
    sleep_sid, asleep = _pick_source(ep, ASLEEP_STAGES, SLEEP_PRIORITY)
    inbed_sid, inbed = _pick_source(ep, ("inbed",), INBED_PRIORITY)
    if sleep_sid is None and inbed_sid is None:
        return None  # e.g. only "awake" samples

    def stage_h(stage: str) -> float:
        g = asleep[asleep["stage"] == stage] if sleep_sid is not None else asleep
        return _union_len(g["start"].to_numpy(), g["end"].to_numpy()) / 3600 if len(g) else 0.0

    sleep_h = _union_len(asleep["start"].to_numpy(), asleep["end"].to_numpy()) / 3600 if sleep_sid is not None \
        else np.nan
    in_bed_h = _union_len(inbed["start"].to_numpy(), inbed["end"].to_numpy()) / 3600 if inbed_sid is not None \
        else np.nan
    has_stages = sleep_sid is not None and asleep["stage"].isin(("core", "deep", "rem")).any()
    kind = "stages" if has_stages else ("asleep" if sleep_sid is not None else "inbed")

    awake_h = np.nan
    awakenings = np.nan
    onset = final_wake = None
    if sleep_sid is not None:
        onset = int(asleep["start"].min())
        final_wake = int(asleep["end"].max())
        aw = ep[(ep["stage"] == "awake") & (ep["source_id"] == sleep_sid)
                & (ep["start"] >= onset) & (ep["end"] <= final_wake)]
        awake_h = _union_len(aw["start"].to_numpy(), aw["end"].to_numpy()) / 3600 if len(aw) else 0.0
        # awakenings = gaps between asleep blocks (or explicit awake segments) of >= 2 min
        a, b = _union(np.empty(0), np.empty(0), asleep["start"].to_numpy(float), asleep["end"].to_numpy(float))
        gaps = a[1:] - b[:-1]
        awakenings = int((gaps >= 120).sum())
        if not has_stages and len(gaps):
            awake_h = float(gaps[gaps > 0].sum()) / 3600

    candidates = [t for t in ((int(inbed["start"].min()) if inbed_sid is not None else None), onset)
                  if t is not None]
    bed = min(candidates)
    up = final_wake if final_wake is not None else int(inbed["end"].max())
    wake_day = night_day
    prev_midnight = (wake_day - 1) * 86400

    def clock(t: Optional[int]) -> float:
        return np.nan if t is None else (t + off - prev_midnight) / 3600.0

    bedtime_h = clock(bed)
    wake_h = clock(up) - 24.0
    onset_h = clock(onset)
    mid_start = onset if onset is not None else bed
    midsleep_h = (clock(mid_start) + clock(up)) / 2.0

    efficiency = np.nan
    if sleep_sid is not None and inbed_sid is not None and in_bed_h > 0:
        efficiency = float(min(100.0, 100.0 * sleep_h / in_bed_h))

    sleep_hr = np.nan
    if hr_t is not None and onset is not None:
        lo, hi = np.searchsorted(hr_t, onset), np.searchsorted(hr_t, final_wake)
        if hi - lo >= 3:
            sleep_hr = float(np.mean(hr_v[lo:hi]))

    return {
        "wake_date": pd.Timestamp(np.datetime64(int(wake_day), "D")),
        "kind": kind,
        "sleep_h": sleep_h, "in_bed_h": in_bed_h,
        "core_h": stage_h("core") if has_stages else np.nan,
        "deep_h": stage_h("deep") if has_stages else np.nan,
        "rem_h": stage_h("rem") if has_stages else np.nan,
        "asleep_unspec_h": stage_h("asleep") if sleep_sid is not None else np.nan,
        "awake_h": awake_h, "awakenings": awakenings,
        "efficiency_pct": efficiency,
        "bedtime_h": bedtime_h, "onset_h": onset_h, "wake_h": wake_h, "midsleep_h": midsleep_h,
        "sleep_source": int(sleep_sid) if sleep_sid is not None else -1,
        "inbed_source": int(inbed_sid) if inbed_sid is not None else -1,
        "start_utc": int(bed), "end_utc": int(up),
        "onset_utc": onset if onset is not None else -1,
        "final_wake_utc": final_wake if final_wake is not None else -1,
        "sleep_hr": sleep_hr,
        "segments": int(len(asleep)) if sleep_sid is not None else 0,
    }
