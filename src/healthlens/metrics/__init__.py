"""Derived tables: daily metrics, nights, workouts, hourly profile.

``compute(store)`` builds everything from the Parquet cache; ``load_or_compute``
caches the result under ``cache/derived`` keyed by the import fingerprint and
``METRICS_VERSION`` so re-running the report is fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from ..store import HealthStore
from .daily import compute_daily
from .sleep import build_nights
from .workouts import clean_workouts

METRICS_VERSION = 4


@dataclass
class Profile:
    age: Optional[float] = None
    sex: Optional[str] = None          # "male" | "female" | None
    birth_year: Optional[int] = None

    @classmethod
    def from_meta(cls, meta: dict) -> "Profile":
        prof = meta.get("profile", {}) or {}
        dob = prof.get("DateOfBirth")
        sex_raw = (prof.get("BiologicalSex") or "").lower()
        sex = "male" if sex_raw.endswith("male") and not sex_raw.endswith("female") else \
            ("female" if sex_raw.endswith("female") else None)
        age = birth_year = None
        if dob:
            try:
                born = date.fromisoformat(dob[:10])
                ref_text = (meta.get("export_date") or "")[:10]
                ref = date.fromisoformat(ref_text) if ref_text else date.today()
                age = (ref - born).days / 365.25
                birth_year = born.year
            except ValueError:
                pass
        return cls(age=age, sex=sex, birth_year=birth_year)


@dataclass
class Metrics:
    daily: pd.DataFrame
    nights: pd.DataFrame
    naps: pd.DataFrame
    workouts: pd.DataFrame
    hourly: pd.DataFrame
    sources: pd.DataFrame
    meta: dict
    profile: Profile
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def unit_system(self) -> str:
        return self.meta.get("unit_system", "metric")


def compute(store: HealthStore, progress: Optional[Callable[[str], None]] = None) -> Metrics:
    say = progress or (lambda msg: None)
    profile = Profile.from_meta(store.meta)
    hr = store.values("HeartRate", "bpm") if store.has("HeartRate") else None
    say("sleep")
    nights, naps = build_nights(store.records("SleepAnalysis") if store.has("SleepAnalysis") else None, hr)
    say("workouts")
    workouts = clean_workouts(store.workouts(), hr, profile.age)
    say("daily metrics")
    daily, hourly, dedup_stats = compute_daily(store, nights, naps, workouts)
    store.release()
    info = {
        "dedup": dedup_stats,
        "workout_duplicates_removed": int(workouts.attrs.get("duplicates_removed", 0)) if len(workouts) else 0,
    }
    return Metrics(daily=daily, nights=nights, naps=naps, workouts=workouts, hourly=hourly,
                   sources=store.sources, meta=store.meta, profile=profile, info=info)


def _key(store: HealthStore) -> str:
    return f"{store.meta.get('fingerprint')}-{store.meta.get('parser_version')}-{METRICS_VERSION}"


def load_or_compute(store: HealthStore, progress: Optional[Callable[[str], None]] = None,
                    force: bool = False) -> Metrics:
    folder = store.dir / "derived"
    stamp = folder / "stamp.json"
    if not force and stamp.exists():
        try:
            info = json.loads(stamp.read_text())
            if info.get("key") == _key(store):
                return Metrics(
                    daily=pd.read_parquet(folder / "daily.parquet"),
                    nights=pd.read_parquet(folder / "nights.parquet"),
                    naps=pd.read_parquet(folder / "naps.parquet"),
                    workouts=pd.read_parquet(folder / "workouts.parquet"),
                    hourly=pd.read_parquet(folder / "hourly.parquet"),
                    sources=store.sources, meta=store.meta, profile=Profile.from_meta(store.meta),
                    info=info.get("info", {}),
                )
        except (OSError, ValueError, KeyError):
            pass
    m = compute(store, progress)
    folder.mkdir(parents=True, exist_ok=True)
    m.daily.to_parquet(folder / "daily.parquet")
    for name in ("nights", "naps", "workouts", "hourly"):
        df = getattr(m, name).copy()
        df.attrs = {}
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string") if c not in ("indoor",) else df[c]
        df.to_parquet(folder / f"{name}.parquet", index=False)
    stamp.write_text(json.dumps({"key": _key(store), "info": m.info}, default=float))
    return m
