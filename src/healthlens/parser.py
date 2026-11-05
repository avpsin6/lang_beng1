"""Streaming parser: Apple Health export -> Parquet cache.

Design notes
------------
* The export (``export.zip`` from the iPhone Health app) can be several GB
  after 6-9 years of Watch data. We stream it straight out of the zip with
  ``xml.etree.ElementTree.iterparse`` and clear elements as we go, so memory
  stays bounded no matter how large the file is.
* Records are buffered per type and flushed to ``cache/records/<Type>/part-N``
  Parquet files. Dates are kept as strings while buffering and converted in
  bulk with numpy (much faster than ``datetime.strptime`` per row).
* Exports from some iOS versions contain a DTD or stray control characters
  that strict XML parsers reject. :class:`SanitizingReader` drops the DTD
  (we don't need it) and strips characters that are illegal in XML 1.0.
  If parsing still fails we retry once in "repair" mode, which also replaces
  invalid UTF-8.
* The original timestamps keep their UTC offset, so we store UTC epoch
  seconds plus the offset in minutes; local wall-clock time is recoverable.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import catalog, devices, units
from .catalog import short_category_value, short_type, short_workout, type_kind

PARSER_VERSION = 3
MISSING_TIME = np.iinfo(np.int64).min

RECORD_SCHEMA = pa.schema([
    ("source_id", pa.int32()),
    ("start", pa.int64()),        # UTC epoch seconds
    ("end", pa.int64()),
    ("start_off", pa.int16()),    # UTC offset of the original timestamp, minutes
    ("end_off", pa.int16()),
    ("value", pa.float64()),      # numeric value (quantity types)
    ("cat", pa.string()),         # category value, e.g. "SleepAnalysisAsleepCore"
    ("unit", pa.string()),
    ("hr_ctx", pa.int8()),        # HKMetadataKeyHeartRateMotionContext (-1 = absent)
    ("user_entered", pa.bool_()), # HKWasUserEntered
    ("tz", pa.string()),          # HKTimeZone metadata (sleep, workouts)
])

ProgressFn = Callable[[str, float, Dict[str, int]], None]


class NotAnExportError(ValueError):
    """The input does not look like an Apple Health export."""


# ---------------------------------------------------------------------------
# Locating the XML inside a zip / folder
# ---------------------------------------------------------------------------
@dataclass
class ExportSource:
    description: str
    size: int
    open: Callable[[], "object"]
    fingerprint: str
    container: Optional[Path] = None   # zip or folder holding extras (ECG CSVs, routes)
    xml_name: str = ""


def _fingerprint_file(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    return h.hexdigest()[:20]


def locate_export(path: os.PathLike) -> ExportSource:
    """Find the main export XML in a zip, a folder or a plain XML file.

    The XML is called ``export.xml`` in English but is localised in some
    languages, so we pick the largest .xml that is not the CDA clinical file.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    if p.is_dir():
        candidates = [c for c in p.rglob("*.xml") if "cda" not in c.name.lower()]
        if not candidates:
            raise NotAnExportError(f"No export XML found in folder {p}")
        xml = max(candidates, key=lambda c: c.stat().st_size)
        return ExportSource(str(xml), xml.stat().st_size, lambda: open(xml, "rb"),
                            _fingerprint_file(xml), container=xml.parent, xml_name=xml.name)
    if zipfile.is_zipfile(p):
        zf = zipfile.ZipFile(p)
        infos = [i for i in zf.infolist()
                 if i.filename.lower().endswith(".xml") and "cda" not in Path(i.filename).name.lower()
                 and not i.is_dir()]
        if not infos:
            zf.close()
            raise NotAnExportError(f"No export XML found inside {p.name}")
        info = max(infos, key=lambda i: i.file_size)
        zf.close()

        def _open():
            z = zipfile.ZipFile(p)
            return _ZipMember(z, z.open(info))

        return ExportSource(f"{p.name}:{info.filename}", info.file_size, _open,
                            _fingerprint_file(p), container=p, xml_name=info.filename)
    return ExportSource(str(p), p.stat().st_size, lambda: open(p, "rb"), _fingerprint_file(p),
                        container=p.parent, xml_name=p.name)


class _ZipMember:
    """File-like wrapper that closes the zip archive together with the member."""

    def __init__(self, zf: zipfile.ZipFile, member):
        self._zf, self._m = zf, member

    def read(self, n: int = -1) -> bytes:
        return self._m.read(n)

    def close(self) -> None:
        self._m.close()
        self._zf.close()


# ---------------------------------------------------------------------------
# Sanitising reader
# ---------------------------------------------------------------------------
_BAD_CTRL = bytes(range(0, 9)) + b"\x0b\x0c" + bytes(range(0x0E, 0x20))
_BAD_CHARREF = re.compile(
    rb"&#(?:[xX]0*(?:[0-8bBcCeEfF]|1[0-9a-fA-F])|0*(?:[0-8]|1[124-9]|2[0-9]|3[01]));")
_VERSION_RE = re.compile(rb"HealthKit Export Version:\s*(\d+)")


class SanitizingReader:
    """Byte stream filter in front of the XML parser.

    * removes everything before ``<HealthData`` (XML declaration + DTD) and
      re-adds a clean declaration;
    * deletes control characters that XML 1.0 forbids, raw or as ``&#..;``;
    * in ``repair`` mode, also replaces invalid UTF-8 sequences.
    """

    HEAD_LIMIT = 64 << 20

    def __init__(self, raw, repair: bool = False):
        self._raw = raw
        self._repair = repair
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace") if repair else None
        self._pending = b""
        self._carry = b""
        self._eof = False
        self.bytes_read = 0
        self.export_version: Optional[int] = None
        self._pending = self._read_head()

    def _raw_read(self, n: int) -> bytes:
        data = self._raw.read(n)
        self.bytes_read += len(data)
        return data

    def _read_head(self) -> bytes:
        buf = b""
        while b"<HealthData" not in buf:
            chunk = self._raw_read(1 << 20)
            if not chunk:
                break
            buf += chunk
            if len(buf) > self.HEAD_LIMIT:
                break
        idx = buf.find(b"<HealthData")
        if idx < 0:
            raise NotAnExportError("This file does not look like an Apple Health export "
                                   "(no <HealthData> element found).")
        m = _VERSION_RE.search(buf[:idx])
        self.export_version = int(m.group(1)) if m else None
        return b'<?xml version="1.0" encoding="UTF-8"?>\n' + self._clean(buf[idx:], final=False)

    def _clean(self, data: bytes, final: bool) -> bytes:
        if self._decoder is not None:
            data = self._decoder.decode(data, final=final).encode("utf-8")
        data = self._carry + data
        self._carry = b""
        if not final:
            # Don't split a character reference such as "&#x0B;" across chunks.
            amp = data.rfind(b"&", max(0, len(data) - 12))
            if amp != -1 and data.find(b";", amp) == -1:
                self._carry, data = data[amp:], data[:amp]
        data = data.translate(None, _BAD_CTRL)
        if b"&#" in data:
            data = _BAD_CHARREF.sub(b"", data)
        return data

    def read(self, n: int = 1 << 16) -> bytes:
        if n is None or n < 0:
            n = 1 << 16
        while len(self._pending) < n and not self._eof:
            chunk = self._raw_read(max(n, 1 << 16))
            if not chunk:
                self._eof = True
                self._pending += self._clean(b"", final=True)
                break
            self._pending += self._clean(chunk, final=False)
        out, self._pending = self._pending[:n], self._pending[n:]
        return out


# ---------------------------------------------------------------------------
# Vectorised date parsing:  "2020-09-20 08:05:11 -0700"
# ---------------------------------------------------------------------------
def _days_from_civil(y, m, d):
    """Days since 1970-01-01 (Howard Hinnant's algorithm), vectorised."""
    y = y - (m <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _parse_one_date(text: Optional[str]) -> Tuple[int, int]:
    """Slow fallback for timestamps in an unexpected format."""
    from datetime import datetime, timezone

    if not text:
        return MISSING_TIME, 0
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.strptime(text, fmt)
            off = int(dt.utcoffset().total_seconds() // 60)
            return int(dt.timestamp()), off
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(text.replace(" ", "T", 1))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp()), int(dt.utcoffset().total_seconds() // 60)
    except ValueError:
        return MISSING_TIME, 0


def parse_hk_dates(values: Sequence[Optional[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert HealthKit timestamp strings to (UTC epoch seconds, offset minutes)."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16)
    arr = np.array([v if isinstance(v, str) else "" for v in values], dtype="U26")
    codes = arr.view(np.uint32).reshape(n, 26)
    ok = ((codes[:, 4] == 45) & (codes[:, 7] == 45) & (codes[:, 10] == 32) & (codes[:, 13] == 58)
          & (codes[:, 16] == 58) & (codes[:, 19] == 32) & ((codes[:, 20] == 43) | (codes[:, 20] == 45))
          & (codes[:, 25] == 0))
    d = codes.astype(np.int64) - 48
    digit_cols = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 21, 22, 23, 24]
    digits_ok = np.all((d[:, digit_cols] >= 0) & (d[:, digit_cols] <= 9), axis=1)
    ok &= digits_ok
    year = d[:, 0] * 1000 + d[:, 1] * 100 + d[:, 2] * 10 + d[:, 3]
    month = d[:, 5] * 10 + d[:, 6]
    day = d[:, 8] * 10 + d[:, 9]
    hh = d[:, 11] * 10 + d[:, 12]
    mm = d[:, 14] * 10 + d[:, 15]
    ss = d[:, 17] * 10 + d[:, 18]
    sign = np.where(codes[:, 20] == 45, -1, 1)
    off = sign * ((d[:, 21] * 10 + d[:, 22]) * 60 + d[:, 23] * 10 + d[:, 24])
    ok &= (month >= 1) & (month <= 12) & (day >= 1) & (day <= 31) & (hh <= 23) & (mm <= 59) & (ss <= 60)
    local = _days_from_civil(year, month, day) * 86400 + hh * 3600 + mm * 60 + ss
    utc = local - off * 60
    if not ok.all():
        for i in np.flatnonzero(~ok):
            utc[i], off[i] = _parse_one_date(values[i])
    return utc.astype(np.int64), off.astype(np.int16)


def _parse_quantity(text: Optional[str]) -> Tuple[float, str]:
    """'2000 cm' -> (2000.0, 'cm'); '68 degF' -> (68.0, 'degF')."""
    if not text:
        return float("nan"), ""
    parts = text.strip().split(" ", 1)
    try:
        return float(parts[0]), (parts[1].strip() if len(parts) > 1 else "")
    except ValueError:
        return float("nan"), ""


def _f(value: Optional[str]) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------
@dataclass
class _SourceInfo:
    name: str
    model: str
    hardware: str
    kind: str
    versions: Counter = field(default_factory=Counter)


class _Importer:
    FLUSH_TYPE = 250_000      # rows per type before writing a Parquet part
    FLUSH_TOTAL = 1_200_000   # global cap on buffered rows

    def __init__(self, out: Path, progress: Optional[ProgressFn], total_bytes: int):
        self.out = out
        self.records_dir = out / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self.total_bytes = max(total_bytes, 1)
        self.buffers: Dict[str, list] = {}
        self.buffered = 0
        self.part_no: Counter = Counter()
        self.sources: List[_SourceInfo] = []
        self.source_ids: Dict[Tuple[str, str], int] = {}
        self.type_counts: Counter = Counter()
        self.type_units: Dict[str, Counter] = defaultdict(Counter)
        self.type_first: Dict[str, int] = {}
        self.type_last: Dict[str, int] = {}
        self.type_sources: Dict[str, Counter] = defaultdict(Counter)
        self.workouts: List[dict] = []
        self.activity: List[tuple] = []
        self.corr_rows: Dict[str, list] = defaultdict(list)
        self.correlations = 0
        self.other_elements: Counter = Counter()
        self.me: Dict[str, str] = {}
        self.export_date: Optional[str] = None
        self.locale: Optional[str] = None
        self.n_records = 0
        self._last_progress = 0.0

    # -- sources ---------------------------------------------------------
    def source_id(self, name: str, device: Optional[str], version: Optional[str]) -> int:
        tail = devices.device_tail(device)
        key = (name, tail)
        sid = self.source_ids.get(key)
        if sid is None:
            model, hardware = devices.parse_device(device)
            info = _SourceInfo(name, model, hardware, devices.classify(name, model, hardware))
            sid = len(self.sources)
            self.sources.append(info)
            self.source_ids[key] = sid
        if version:
            self.sources[sid].versions[version] += 1
        return sid

    # -- elements --------------------------------------------------------
    def handle(self, elem) -> None:
        tag = elem.tag
        if tag == "Record":
            self.record(elem, self.buffers)
        elif tag == "Workout":
            self.workout(elem)
        elif tag == "ActivitySummary":
            self.activity_summary(elem)
        elif tag == "Correlation":
            self.correlations += 1
            for child in elem:
                if child.tag == "Record":
                    self.record(child, self.corr_rows, count=False)
        elif tag == "Me":
            self.me = {k.replace("HKCharacteristicTypeIdentifier", ""): v for k, v in elem.attrib.items()}
        elif tag == "ExportDate":
            self.export_date = elem.get("value")
        else:
            self.other_elements[tag] += 1

    def record(self, elem, target: Dict[str, list], count: bool = True) -> None:
        a = elem.attrib
        t = a.get("type", "Unknown")
        sid = self.source_id(a.get("sourceName", ""), a.get("device"), a.get("sourceVersion"))
        hr_ctx, user_entered, tz = -1, False, None
        if len(elem):
            for child in elem:
                if child.tag != "MetadataEntry":
                    continue
                key = child.get("key")
                if key == "HKMetadataKeyHeartRateMotionContext":
                    try:
                        hr_ctx = int(child.get("value") or -1)
                    except ValueError:
                        hr_ctx = -1
                elif key == "HKWasUserEntered":
                    user_entered = child.get("value") in ("1", "true", "YES")
                elif key == "HKTimeZone":
                    tz = child.get("value")
        buf = target.get(t)
        if buf is None:
            buf = target[t] = []
        buf.append((sid, a.get("unit"), a.get("startDate"), a.get("endDate"), a.get("value"),
                    hr_ctx, user_entered, tz))
        if count:
            self.n_records += 1
            self.buffered += 1
            if len(buf) >= self.FLUSH_TYPE:
                self.flush(t)
            elif self.buffered >= self.FLUSH_TOTAL:
                for name in list(self.buffers):
                    self.flush(name)

    def workout(self, elem) -> None:
        a = elem.attrib
        sid = self.source_id(a.get("sourceName", ""), a.get("device"), a.get("sourceVersion"))
        w = {
            "activity": short_workout(a.get("workoutActivityType", "Other")),
            "source_id": sid,
            "start_s": a.get("startDate"),
            "end_s": a.get("endDate"),
            "duration": _f(a.get("duration")),
            "duration_unit": a.get("durationUnit") or "min",
            "distance": _f(a.get("totalDistance")),
            "distance_unit": a.get("totalDistanceUnit") or "",
            "energy": _f(a.get("totalEnergyBurned")),
            "energy_unit": a.get("totalEnergyBurnedUnit") or "",
            "swim_strokes": _f(a.get("totalSwimmingStrokeCount")),
            "flights": _f(a.get("totalFlightsClimbed")),
        }
        meta: Dict[str, str] = {}
        stats: Dict[str, Dict[str, object]] = {}
        route = None
        events = Counter()
        for child in elem:
            tag = child.tag
            if tag == "MetadataEntry":
                meta[child.get("key", "")] = child.get("value", "")
            elif tag == "WorkoutStatistics":
                stats[short_type(child.get("type", ""))] = {
                    "sum": _f(child.get("sum")), "avg": _f(child.get("average")),
                    "min": _f(child.get("minimum")), "max": _f(child.get("maximum")),
                    "unit": child.get("unit") or "",
                }
            elif tag == "WorkoutEvent":
                events[(child.get("type") or "").replace("HKWorkoutEventType", "")] += 1
            elif tag == "WorkoutRoute":
                for sub in child.iter("FileReference"):
                    route = sub.get("path")
            elif tag == "WorkoutActivity":
                for sub in child:
                    if sub.tag == "WorkoutStatistics":
                        key = short_type(sub.get("type", ""))
                        stats.setdefault(key, {
                            "sum": _f(sub.get("sum")), "avg": _f(sub.get("average")),
                            "min": _f(sub.get("minimum")), "max": _f(sub.get("maximum")),
                            "unit": sub.get("unit") or "",
                        })
        w["meta"] = meta
        w["stats"] = stats
        w["route"] = route
        w["events"] = dict(events)
        self.workouts.append(w)

    def activity_summary(self, elem) -> None:
        a = elem.attrib
        unit = a.get("activeEnergyBurnedUnit") or "kcal"
        fn = units.converter(unit, "kcal") or (lambda v: v)
        self.activity.append((
            a.get("dateComponents"),
            float(fn(np.float64(_f(a.get("activeEnergyBurned"))))),
            float(fn(np.float64(_f(a.get("activeEnergyBurnedGoal"))))),
            _f(a.get("appleMoveTime")), _f(a.get("appleMoveTimeGoal")),
            _f(a.get("appleExerciseTime")), _f(a.get("appleExerciseTimeGoal")),
            _f(a.get("appleStandHours")), _f(a.get("appleStandHoursGoal")),
        ))

    # -- flushing ----------------------------------------------------------
    @staticmethod
    def _safe(t: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", short_type(t)) or "Unknown"

    def _to_table(self, t: str, rows: list) -> pa.Table:
        sid, unit, sd, ed, val, ctx, ue, tz = zip(*rows)
        start, start_off = parse_hk_dates(sd)
        end, end_off = parse_hk_dates(ed)
        if type_kind(t) == "category":
            cat = [short_category_value(v) if isinstance(v, str) else None for v in val]
            value = pd.to_numeric(pd.Series(val, dtype=object), errors="coerce").to_numpy(dtype=float)
        else:
            cat = [None] * len(val)
            value = pd.to_numeric(pd.Series(val, dtype=object), errors="coerce").to_numpy(dtype=float)
        keep = (start != MISSING_TIME) & (end != MISSING_TIME)
        end = np.where(keep & (end < start), start, end)
        table = pa.table({
            "source_id": pa.array(sid, type=pa.int32()),
            "start": pa.array(start, type=pa.int64()),
            "end": pa.array(end, type=pa.int64()),
            "start_off": pa.array(start_off, type=pa.int16()),
            "end_off": pa.array(end_off, type=pa.int16()),
            "value": pa.array(value, type=pa.float64()),
            "cat": pa.array(cat, type=pa.string()),
            "unit": pa.array(unit, type=pa.string()),
            "hr_ctx": pa.array(ctx, type=pa.int8()),
            "user_entered": pa.array(ue, type=pa.bool_()),
            "tz": pa.array(tz, type=pa.string()),
        }, schema=RECORD_SCHEMA)
        if not keep.all():
            table = table.filter(pa.array(keep))
        return table

    def _write(self, t: str, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        folder = self.records_dir / self._safe(t)
        folder.mkdir(exist_ok=True)
        pq.write_table(table, folder / f"part-{self.part_no[t]:04d}.parquet", compression="zstd")
        self.part_no[t] += 1
        short = short_type(t)
        self.type_counts[short] += table.num_rows
        starts = table.column("start").to_numpy()
        ends = table.column("end").to_numpy()
        self.type_first[short] = int(min(starts.min(), self.type_first.get(short, starts.min())))
        self.type_last[short] = int(max(ends.max(), self.type_last.get(short, ends.max())))
        for u, c in Counter(table.column("unit").to_pylist()).items():
            self.type_units[short][u or ""] += c
        sids, counts = np.unique(table.column("source_id").to_numpy(), return_counts=True)
        for s, c in zip(sids.tolist(), counts.tolist()):
            self.type_sources[short][s] += c

    def flush(self, t: str) -> None:
        rows = self.buffers.pop(t, None)
        if not rows:
            return
        self.buffered -= len(rows)
        self._write(t, self._to_table(t, rows))

    def maybe_progress(self, bytes_read: int, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        if force or now - self._last_progress > 1.0:
            self._last_progress = now
            self.progress("parse", min(bytes_read / self.total_bytes, 1.0),
                          {"records": self.n_records, "workouts": len(self.workouts)})

    # -- finishing ----------------------------------------------------------
    def finish(self, export_version: Optional[int]) -> dict:
        for t in list(self.buffers):
            self.flush(t)
        self._merge_correlations()
        self._write_workouts()
        self._write_activity()
        self._write_sources()
        types = {}
        for short, n in sorted(self.type_counts.items()):
            types[short] = {
                "count": int(n),
                "first": self.type_first.get(short),
                "last": self.type_last.get(short),
                "units": dict(self.type_units[short]),
                "label": catalog.label_for_type(short),
                "sources": {str(k): int(v) for k, v in self.type_sources[short].items()},
            }
        unit_totals: Counter = Counter()
        for short, u in self.type_units.items():
            unit_totals.update(u)
        return {
            "parser_version": PARSER_VERSION,
            "export_date": self.export_date,
            "export_version": export_version,
            "locale": self.locale,
            "profile": self.me,
            "n_records": int(sum(self.type_counts.values())),
            "n_workouts": len(self.workouts),
            "n_activity_summaries": len(self.activity),
            "n_correlations": self.correlations,
            "other_elements": dict(self.other_elements),
            "types": types,
            "unit_system": units.detect_system(dict(unit_totals)),
        }

    def _merge_correlations(self) -> None:
        """Add correlation members (e.g. blood pressure pairs) not also exported top-level."""
        for t, rows in self.corr_rows.items():
            table = self._to_table(t, rows)
            folder = self.records_dir / self._safe(t)
            if folder.exists():
                existing = pa.concat_tables([pq.read_table(p) for p in sorted(folder.glob("*.parquet"))])
                ex = existing.select(["source_id", "start", "end", "value"]).to_pandas()
                new = table.to_pandas()
                key = ["source_id", "start", "end", "value"]
                merged = new.merge(ex.drop_duplicates(key), on=key, how="left", indicator=True)
                mask = (merged["_merge"] == "left_only").to_numpy()
                table = table.filter(pa.array(mask))
            self._write(t, table)

    def _write_workouts(self) -> None:
        if not self.workouts:
            pq.write_table(pa.table({"activity": pa.array([], pa.string())}), self.out / "workouts.parquet")
            return
        start, start_off = parse_hk_dates([w["start_s"] for w in self.workouts])
        end, end_off = parse_hk_dates([w["end_s"] for w in self.workouts])
        rows = []
        for i, w in enumerate(self.workouts):
            rows.append(_workout_row(w, int(start[i]), int(end[i]), int(start_off[i]), int(end_off[i])))
        df = pd.DataFrame(rows)
        df = df[df["start"] != MISSING_TIME]
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), self.out / "workouts.parquet")

    def _write_activity(self) -> None:
        cols = ["date", "active_kcal", "active_goal_kcal", "move_min", "move_goal_min",
                "exercise_min", "exercise_goal_min", "stand_hours", "stand_goal_h"]
        df = pd.DataFrame(self.activity, columns=cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
        df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), self.out / "activity_summary.parquet")

    def _write_sources(self) -> None:
        rows = []
        for sid, s in enumerate(self.sources):
            versions = [v for v, _ in s.versions.most_common()]
            rows.append({
                "source_id": sid, "name": s.name, "model": s.model, "hardware": s.hardware,
                "kind": s.kind, "friendly": devices.friendly_model(s.model, s.hardware),
                "priority": devices.KIND_PRIORITY.get(s.kind, 5),
                "versions": ", ".join(sorted(versions, key=_version_key)[-6:]),
            })
        df = pd.DataFrame(rows, columns=["source_id", "name", "model", "hardware", "kind", "friendly",
                                         "priority", "versions"])
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), self.out / "sources.parquet")


def _version_key(v: str):
    out = []
    for part in re.split(r"[.\s]", v):
        out.append(int(part) if part.isdigit() else 0)
    return out


def _workout_row(w: dict, start: int, end: int, start_off: int, end_off: int) -> dict:
    stats = w["stats"]
    meta = w["meta"]

    def conv(value: float, unit: str, canonical: str) -> float:
        if value is None or np.isnan(value):
            return float("nan")
        fn = units.converter(unit, canonical)
        return float(fn(np.float64(value))) if fn else float("nan")

    duration_min = conv(w["duration"], w["duration_unit"], "min")
    if np.isnan(duration_min):
        duration_min = (end - start) / 60.0

    def stat(key: str, field_name: str, canonical: str) -> float:
        s = stats.get(key)
        if not s:
            return float("nan")
        return conv(s[field_name], s["unit"], canonical)

    energy = stat("ActiveEnergyBurned", "sum", "kcal")
    if np.isnan(energy):
        energy = conv(w["energy"], w["energy_unit"], "kcal")
    distance = float("nan")
    for key in ("DistanceWalkingRunning", "DistanceCycling", "DistanceSwimming", "DistanceWheelchair",
                "DistanceDownhillSnowSports", "DistancePaddleSports", "DistanceRowing", "DistanceSkatingSports",
                "DistanceCrossCountrySkiing"):
        val = stat(key, "sum", "km")
        if not np.isnan(val):
            distance = val
            break
    if np.isnan(distance):
        distance = conv(w["distance"], w["distance_unit"], "km")

    elev, elev_unit = _parse_quantity(meta.get("HKElevationAscended"))
    temp, temp_unit = _parse_quantity(meta.get("HKWeatherTemperature"))
    hum, _ = _parse_quantity(meta.get("HKWeatherHumidity"))
    if not np.isnan(hum) and hum > 100:
        hum = hum / 100.0
    mets, _ = _parse_quantity(meta.get("HKAverageMETs"))
    indoor = meta.get("HKIndoorWorkout")
    return {
        "activity": w["activity"],
        "source_id": w["source_id"],
        "start": start, "end": end, "start_off": start_off, "end_off": end_off,
        "duration_min": duration_min,
        "distance_km": distance,
        "energy_kcal": energy,
        "avg_hr": stat("HeartRate", "avg", "bpm"),
        "min_hr": stat("HeartRate", "min", "bpm"),
        "max_hr": stat("HeartRate", "max", "bpm"),
        "steps": stat("StepCount", "sum", "count"),
        "elevation_m": conv(elev, elev_unit, "m"),
        "temperature_c": conv(temp, temp_unit, "degC"),
        "humidity_pct": hum,
        "avg_mets": mets,
        "indoor": None if indoor is None else indoor in ("1", "true", "YES"),
        "swim_strokes": w["swim_strokes"],
        "flights": w["flights"],
        "route": w["route"],
        "tz": meta.get("HKTimeZone"),
        "brand": meta.get("HKWorkoutBrandName"),
        "laps": int(w["events"].get("Lap", 0)),
        "pauses": int(w["events"].get("Pause", 0)),
        "has_stats": bool(stats),
    }


def _iter_top_level(reader) -> Iterable[Tuple[str, object]]:
    """Yield ('root', elem) once, then ('elem', elem) for each child of <HealthData>."""
    depth = 0
    root = None
    for event, elem in ET.iterparse(reader, events=("start", "end")):
        if event == "start":
            depth += 1
            if depth == 1:
                root = elem
                yield "root", elem
            continue
        depth -= 1
        if depth == 1:
            yield "elem", elem
            root.clear()


def import_export(path: os.PathLike, cache_dir: os.PathLike, progress: Optional[ProgressFn] = None,
                  force: bool = False) -> dict:
    """Parse an Apple Health export into ``cache_dir``. Returns the meta summary.

    The cache is written to a temporary folder first and swapped in only once
    parsing succeeded, so a failed import never destroys a previous cache.
    """
    src = locate_export(path)
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not force and meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
            if old.get("fingerprint") == src.fingerprint and old.get("parser_version") == PARSER_VERSION:
                old["skipped"] = True
                return old
        except (OSError, ValueError):
            pass

    started = time.monotonic()
    last_error: Optional[Exception] = None
    for repair in (False, True):
        tmp = cache_dir.parent / (cache_dir.name + ".importing")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        raw = src.open()
        try:
            reader = SanitizingReader(raw, repair=repair)
            imp = _Importer(tmp, progress, src.size)
            for kind, elem in _iter_top_level(reader):
                if kind == "root":
                    imp.locale = elem.get("locale")
                    continue
                imp.handle(elem)
                imp.maybe_progress(reader.bytes_read)
            imp.maybe_progress(src.size, force=True)
            meta = imp.finish(reader.export_version)
        except ET.ParseError as exc:
            last_error = exc
            shutil.rmtree(tmp, ignore_errors=True)
            if progress:
                progress("retry", 0.0, {"records": 0})
            continue
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        finally:
            raw.close()
        meta.update({
            "source_file": src.description,
            "source_size": src.size,
            "fingerprint": src.fingerprint,
            "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parse_seconds": round(time.monotonic() - started, 1),
            "repair_mode": repair,
            "ecg_files": _count_extras(src, "electrocardiograms"),
            "route_files": _count_extras(src, "workout-routes"),
        })
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        _swap_in(tmp, cache_dir)
        return meta
    raise NotAnExportError(f"Could not parse the export even in repair mode: {last_error}")


def _count_extras(src: ExportSource, folder: str) -> int:
    try:
        if src.container and src.container.is_file() and zipfile.is_zipfile(src.container):
            with zipfile.ZipFile(src.container) as zf:
                return sum(1 for n in zf.namelist() if f"/{folder}/" in f"/{n}" and not n.endswith("/"))
        if src.container and src.container.is_dir():
            d = src.container / folder
            return sum(1 for _ in d.iterdir()) if d.exists() else 0
    except (OSError, zipfile.BadZipFile):
        return 0
    return 0


def _swap_in(tmp: Path, cache_dir: Path) -> None:
    old = cache_dir.parent / (cache_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if cache_dir.exists():
        cache_dir.rename(old)
    tmp.rename(cache_dir)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
