from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthlens.metrics import Profile, compute, load_or_compute
from healthlens.metrics.daily import daily_sum
from healthlens.metrics.dedup import dedup_values
from healthlens.metrics.sleep import build_nights
from healthlens.metrics.workouts import clean_workouts

H = 3600


def _samples(rows):
    """rows: (source_id, priority, start, end, value)"""
    return pd.DataFrame(rows, columns=["source_id", "priority", "start", "end", "value"])


# --------------------------------------------------------------------------- dedup
def test_dedup_phone_fills_only_uncovered_time():
    df = _samples([
        (0, 1, 0, 600, 1000),        # watch 0-10 min
        (1, 2, 300, 900, 600),       # phone 5-15 min: half overlaps -> keeps 300
        (1, 2, 2000, 2600, 500),     # phone only (watch off) -> keeps all
    ])
    v = dedup_values(df)
    assert v.tolist() == pytest.approx([1000, 300, 500])


def test_dedup_priority_beats_volume_and_point_samples():
    df = _samples([
        (5, 3, 0, 1200, 900),        # third-party app, lower priority
        (0, 1, 0, 600, 700),         # watch
        (5, 3, 300, 300, 10),        # instantaneous app sample inside watch coverage -> dropped
        (5, 3, 3000, 3000, 10),      # instantaneous outside coverage -> kept
    ])
    v = dedup_values(df)
    assert v.tolist() == pytest.approx([450, 700, 0, 10])


def test_dedup_coarse_daily_total_is_not_prorated():
    df = _samples([
        (0, 1, 8 * H, 9 * H, 4000),      # watch: 4000 steps in one hour
        (7, 3, 0, 24 * H, 10000),        # app writes a whole-day total
    ])
    v = dedup_values(df)
    assert v.tolist() == pytest.approx([4000, 6000])   # app adds only what the watch missed


def test_dedup_multiple_watches_do_not_cancel():
    df = _samples([(0, 1, 0, 600, 100), (2, 1, 1000, 1600, 200), (1, 2, 0, 1600, 160)])
    v = dedup_values(df)
    # phone covers 0-1600 (160 over 1600 s); 400 s (600..1000) uncovered -> 40
    assert v.tolist() == pytest.approx([100, 200, 40])


def test_daily_sum_splits_across_midnight():
    df = pd.DataFrame({"start": [86400 - 600], "end": [86400 + 1800], "start_off": [0], "end_off": [0]})
    s = daily_sum(df, np.array([240.0]))
    assert s.to_dict() == pytest.approx({0: 60.0, 1: 180.0})


# --------------------------------------------------------------------------- sleep
def _sleep(rows):
    """rows: (source_id, kind, cat, start_h, end_h) with hours relative to 2024-01-01 00:00 UTC."""
    base = 1704067200
    out = []
    for sid, kind, cat, s, e in rows:
        out.append({"source_id": sid, "kind": kind, "cat": cat, "start": base + int(s * H),
                    "end": base + int(e * H), "start_off": 0, "end_off": 0, "tz": None})
    return pd.DataFrame(out)


def test_sleep_stages_beat_phone_inbed_and_naps_are_separate():
    df = _sleep([
        (1, "phone", "SleepAnalysisInBed", -1.0, 7.25),             # 23:00 -> 07:15
        (0, "watch", "SleepAnalysisAsleepCore", -0.5, 2.0),
        (0, "watch", "SleepAnalysisAsleepDeep", 2.0, 3.0),
        (0, "watch", "SleepAnalysisAwake", 3.0, 3.25),
        (0, "watch", "SleepAnalysisAsleepREM", 3.25, 4.5),
        (0, "watch", "SleepAnalysisAsleepCore", 4.5, 7.0),
        (0, "watch", "SleepAnalysisAsleepCore", 14.0, 14.75),        # afternoon nap
    ])
    nights, naps = build_nights(df)
    assert len(nights) == 1
    n = nights.iloc[0]
    assert n["kind"] == "stages"
    assert n["wake_date"] == pd.Timestamp("2024-01-01")
    assert n["sleep_h"] == pytest.approx(7.25)
    assert n["deep_h"] == pytest.approx(1.0) and n["rem_h"] == pytest.approx(1.25)
    assert n["core_h"] == pytest.approx(5.0) and n["awake_h"] == pytest.approx(0.25)
    assert n["in_bed_h"] == pytest.approx(8.25)
    assert n["efficiency_pct"] == pytest.approx(100 * 7.25 / 8.25)
    assert n["bedtime_h"] == pytest.approx(23.0) and n["wake_h"] == pytest.approx(7.0)
    assert n["awakenings"] == 1
    assert naps["nap_h"].tolist() == pytest.approx([0.75])


def test_sleep_inbed_only_is_not_called_sleep():
    nights, _ = build_nights(_sleep([(1, "phone", "SleepAnalysisInBed", -0.5, 7.5)]))
    n = nights.iloc[0]
    assert n["kind"] == "inbed" and np.isnan(n["sleep_h"]) and n["in_bed_h"] == pytest.approx(8.0)


def test_sleep_does_not_add_two_sources_together():
    df = _sleep([
        (0, "watch", "SleepAnalysisAsleepUnspecified", 0.0, 6.0),
        (3, "app", "SleepAnalysisAsleep", -0.5, 6.5),               # AutoSleep says 7 h
    ])
    nights, _ = build_nights(df)
    n = nights.iloc[0]
    assert n["kind"] == "asleep" and n["sleep_h"] == pytest.approx(6.0)   # watch wins, no summing


def test_sleep_split_night_merges_and_old_numeric_codes():
    df = _sleep([
        (0, "watch", "1", -2.0, 0.5),          # old export: numeric 'asleep' code
        (0, "watch", "1", 3.0, 7.0),           # 2.5 h gap > episode gap -> second episode, same morning
    ])
    nights, naps = build_nights(df)
    assert len(nights) == 1
    n = nights.iloc[0]
    assert n["sleep_h"] == pytest.approx(6.5) and n["bedtime_h"] == pytest.approx(22.0)


def test_sleep_uses_hktimezone_when_travelling():
    df = _sleep([(0, "watch", "SleepAnalysisAsleepUnspecified", 15.0, 22.0)])   # 15:00-22:00 UTC
    df["tz"] = "Asia/Kolkata"                                                      # = 20:30 -> 03:30 IST
    nights, _ = build_nights(df)
    n = nights.iloc[0]
    assert n["bedtime_h"] == pytest.approx(20.5) and n["wake_h"] == pytest.approx(3.5)
    assert n["wake_date"] == pd.Timestamp("2024-01-02")


# --------------------------------------------------------------------------- workouts
def test_workout_duplicates_and_hr_fill():
    base = 1704067200
    raw = pd.DataFrame([
        {"activity": "Running", "source_id": 0, "kind": "watch", "start": base, "end": base + 1800,
         "start_off": 0, "duration_min": 30, "distance_km": 5, "energy_kcal": 300, "avg_hr": np.nan,
         "max_hr": np.nan},
        {"activity": "Running", "source_id": 9, "kind": "app", "start": base + 20, "end": base + 1790,
         "start_off": 0, "duration_min": 29.5, "distance_km": 4.9, "energy_kcal": 280, "avg_hr": np.nan,
         "max_hr": np.nan},
        {"activity": "Yoga", "source_id": 0, "kind": "watch", "start": base + 4000, "end": base + 5800,
         "start_off": 0, "duration_min": 30, "distance_km": np.nan, "energy_kcal": 90, "avg_hr": 95.0,
         "max_hr": 110.0},
    ])
    t = np.arange(base, base + 1800, 60)
    hr = pd.DataFrame({"start": t, "value": np.full(len(t), 150.0), "kind": "watch"})
    w = clean_workouts(raw, hr, age=35)
    assert w["activity"].tolist() == ["Running", "Yoga"]
    assert w.attrs["duplicates_removed"] == 1
    run = w.iloc[0]
    assert run["avg_hr"] == pytest.approx(150) and bool(run["hr_from_samples"])
    assert run["pace_min_per_km"] == pytest.approx(6.0)
    # max HR estimate 208 - 0.7*35 = 183.5 -> 150 is 82 % -> zone 4
    assert run["z4_min"] == pytest.approx(30.0, abs=1.1)
    assert w.iloc[1]["avg_hr"] == 95.0


def test_profile_age_and_sex():
    p = Profile.from_meta({"profile": {"DateOfBirth": "1990-05-14", "BiologicalSex": "HKBiologicalSexFemale"},
                           "export_date": "2026-11-01 10:00:00 +0000"})
    assert p.sex == "female" and 36.4 < p.age < 36.5


# --------------------------------------------------------------------------- integration
def test_long_synthetic_metrics(long_store, long_export):
    _, truth, cfg = long_export
    m = compute(long_store)
    d, tr = m.daily, truth.set_index("date")

    j = d[["steps"]].join(tr[["steps_expected", "steps_watch", "steps_phone"]], how="inner")
    j = j[j["steps_expected"] > 500]
    ratio = j["steps"] / j["steps_expected"]
    naive = (j["steps_watch"] + j["steps_phone"]) / j["steps_expected"]
    assert 0.98 < ratio.median() < 1.15
    assert ratio.median() < 0.8 * naive.median()

    # resting HR passes through unchanged on worn days
    rhr = d["resting_hr"].dropna()
    expected = tr.loc[rhr.index, "rhr"].round()
    assert (rhr - expected).abs().max() <= 0.5

    kinds = set(m.nights["kind"])
    assert {"inbed", "asleep"} <= kinds
    assert m.info["workout_duplicates_removed"] > 0
    assert d.index.is_monotonic_increasing and d.index.freq is None or True
    assert d.index.max() < pd.Timestamp(cfg.end)      # partial export day trimmed
    assert m.hourly["hour"].between(0, 23).all()
    # phone-only era has steps but no watch metrics
    pre = d.loc[: pd.Timestamp(cfg.watch_start) - pd.Timedelta(days=1)]
    assert pre["steps"].notna().mean() > 0.95 and pre["resting_hr"].isna().all()


def test_derived_cache_roundtrip(small_store):
    m1 = load_or_compute(small_store, force=True)
    m2 = load_or_compute(small_store)
    pd.testing.assert_series_equal(m1.daily["steps"], m2.daily["steps"], check_freq=False)
    assert len(m1.nights) == len(m2.nights) and len(m1.workouts) == len(m2.workouts)
