"""Read access to the Parquet cache written by :mod:`healthlens.parser`."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import catalog, units


class CacheMissingError(FileNotFoundError):
    pass


class HealthStore:
    """Thin, cached accessor around the cache folder."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise CacheMissingError(
                f"No imported data found in {self.dir}. Run `healthlens import <export.zip>` first.")
        self.meta: dict = json.loads(meta_path.read_text())
        self._records: Dict[str, pd.DataFrame] = {}
        self.sources: pd.DataFrame = pd.read_parquet(self.dir / "sources.parquet")
        self.source_kind = dict(zip(self.sources["source_id"], self.sources["kind"]))
        self.source_name = dict(zip(self.sources["source_id"], self.sources["name"]))

    # ------------------------------------------------------------------
    @property
    def types(self) -> Dict[str, dict]:
        return self.meta.get("types", {})

    def has(self, short_type: str) -> bool:
        return short_type in self.types and self.types[short_type]["count"] > 0

    @property
    def profile(self) -> dict:
        return self.meta.get("profile", {})

    @property
    def unit_system(self) -> str:
        return self.meta.get("unit_system", "metric")

    def records(self, short_type: str) -> pd.DataFrame:
        """All samples of one type, sorted by start time.

        Adds ``kind`` (watch/phone/app/manual) and ``priority`` columns from
        the sources table, and drops exact duplicates (same source, start,
        end and value), which occasionally appear after iCloud re-syncs.
        """
        if short_type in self._records:
            return self._records[short_type]
        folder = self.dir / "records" / short_type
        if not folder.exists():
            df = _empty_records()
        else:
            parts = sorted(folder.glob("*.parquet"))
            table = pa.concat_tables([pq.read_table(p) for p in parts]) if parts else None
            df = table.to_pandas() if table is not None else _empty_records()
        if len(df):
            df = df.drop_duplicates(["source_id", "start", "end", "value", "cat"])
            df = df.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
        df["kind"] = df["source_id"].map(self.source_kind).fillna("app")
        prio = dict(zip(self.sources["source_id"], self.sources["priority"]))
        df["priority"] = df["source_id"].map(prio).fillna(5).astype("int16")
        self._records[short_type] = df
        return df

    def values(self, short_type: str, canonical: str) -> pd.DataFrame:
        """Records with ``value`` converted to the canonical unit."""
        df = self.records(short_type).copy()
        if len(df):
            df["value"] = units.convert(df["value"].to_numpy(), df["unit"].to_numpy(), canonical)
            df = df[np.isfinite(df["value"].to_numpy())]
        return df

    def workouts(self) -> pd.DataFrame:
        path = self.dir / "workouts.parquet"
        df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if len(df) and "source_id" in df:
            df["kind"] = df["source_id"].map(self.source_kind).fillna("app")
            df["source"] = df["source_id"].map(self.source_name).fillna("")
        return df

    def activity_summary(self) -> pd.DataFrame:
        path = self.dir / "activity_summary.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def release(self, short_type: Optional[str] = None) -> None:
        """Free cached DataFrames (large exports)."""
        if short_type is None:
            self._records.clear()
        else:
            self._records.pop(short_type, None)


def _empty_records() -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": pd.Series([], dtype="int32"),
        "start": pd.Series([], dtype="int64"),
        "end": pd.Series([], dtype="int64"),
        "start_off": pd.Series([], dtype="int16"),
        "end_off": pd.Series([], dtype="int16"),
        "value": pd.Series([], dtype="float64"),
        "cat": pd.Series([], dtype=object),
        "unit": pd.Series([], dtype=object),
        "hr_ctx": pd.Series([], dtype="int8"),
        "user_entered": pd.Series([], dtype=bool),
        "tz": pd.Series([], dtype=object),
    })


# ---------------------------------------------------------------------------
# Time helpers shared by the metrics layer
# ---------------------------------------------------------------------------
def local_seconds(utc: np.ndarray, offset_min: np.ndarray) -> np.ndarray:
    return np.asarray(utc, dtype=np.int64) + np.asarray(offset_min, dtype=np.int64) * 60


def local_day(utc: np.ndarray, offset_min: np.ndarray) -> np.ndarray:
    """Local calendar day as days since 1970-01-01."""
    return np.floor_divide(local_seconds(utc, offset_min), 86400)


def days_to_dates(days: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.asarray(days, dtype="int64").astype("datetime64[D]").astype("datetime64[ns]"))


def date_to_day(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).values.astype("datetime64[D]").astype(np.int64)
