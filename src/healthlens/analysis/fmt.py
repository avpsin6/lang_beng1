"""Human-friendly number, time and unit formatting for report text."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..catalog import ALL_METRICS
from ..units import DisplayUnits


class Formatter:
    def __init__(self, units: DisplayUnits, clock24: bool = True):
        self.units = units
        self.clock24 = clock24

    # -- basic numbers ---------------------------------------------------------
    @staticmethod
    def num(x: Optional[float], decimals: int = 0) -> str:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "–"
        if decimals == 0:
            return f"{x:,.0f}"
        return f"{x:,.{decimals}f}"

    @staticmethod
    def pct(x: Optional[float], decimals: int = 0, signed: bool = False) -> str:
        if x is None or not math.isfinite(x):
            return "–"
        return f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%"

    # -- metric values -----------------------------------------------------------
    def value(self, key: str, x: Optional[float], with_unit: bool = True, decimals: Optional[int] = None) -> str:
        m = ALL_METRICS.get(key)
        if x is None or not math.isfinite(float(x)):
            return "–"
        if m is None:
            return self.num(x, 1)
        if m.unit == "clock":
            return self.clock(x)
        if m.unit == "h":
            return self.hours(x)
        dec = m.decimals if decimals is None else decimals
        v = self.units.convert(float(x), m.unit)
        label = self.units.label(m.unit)
        text = self.num(v, dec)
        return f"{text} {label}".strip() if with_unit and label else text

    def delta(self, key: str, d: Optional[float], decimals: Optional[int] = None) -> str:
        m = ALL_METRICS.get(key)
        if d is None or not math.isfinite(d):
            return "–"
        if m is not None and m.unit in ("h",):
            return ("+" if d >= 0 else "−") + self.hours(abs(d))
        if m is not None and m.unit == "clock":
            return ("+" if d >= 0 else "−") + self.minutes(abs(d) * 60)
        dec = (m.decimals if m else 1) if decimals is None else decimals
        v = self.units.convert_delta(d, m.unit) if m else d
        label = self.units.label(m.unit) if m else ""
        sign = "+" if v >= 0 else "−"
        return f"{sign}{self.num(abs(v), dec)} {label}".strip()

    def unit_label(self, key: str) -> str:
        m = ALL_METRICS.get(key)
        if not m:
            return ""
        if m.unit in ("clock",):
            return ""
        return self.units.label(m.unit)

    # -- time ------------------------------------------------------------------------
    @staticmethod
    def hours(h: Optional[float]) -> str:
        """7.42 -> '7 h 25 min'."""
        if h is None or not math.isfinite(h):
            return "–"
        total = int(round(h * 60))
        hh, mm = divmod(total, 60)
        if hh == 0:
            return f"{mm} min"
        return f"{hh} h {mm:02d} min"

    @staticmethod
    def minutes(m: Optional[float]) -> str:
        if m is None or not math.isfinite(m):
            return "–"
        m = int(round(m))
        if m < 60:
            return f"{m} min"
        return f"{m // 60} h {m % 60:02d} min"

    def clock(self, h: Optional[float]) -> str:
        """Hours after midnight (may exceed 24) -> '11:24 pm' / '23:24'."""
        if h is None or not math.isfinite(h):
            return "–"
        total = int(round(h * 60)) % (24 * 60)
        hh, mm = divmod(total, 60)
        if self.clock24:
            return f"{hh:02d}:{mm:02d}"
        suffix = "am" if hh < 12 else "pm"
        h12 = hh % 12 or 12
        return f"{h12}:{mm:02d} {suffix}"

    def hour_label(self, hour: int) -> str:
        if self.clock24:
            return f"{hour:02d}:00"
        suffix = "am" if hour < 12 else "pm"
        return f"{hour % 12 or 12} {suffix}"

    # -- distance & pace --------------------------------------------------------------
    def distance(self, km: Optional[float], decimals: int = 1) -> str:
        if km is None or not math.isfinite(km):
            return "–"
        return f"{self.num(self.units.convert(km, 'km'), decimals)} {self.units.label('km')}"

    def pace(self, min_per_km: Optional[float]) -> str:
        if min_per_km is None or not math.isfinite(min_per_km) or min_per_km <= 0:
            return "–"
        if self.units.system == "imperial":
            value, label = min_per_km * 1.609344, "/mi"
        else:
            value, label = min_per_km, "/km"
        total = int(round(value * 60))
        return f"{total // 60}:{total % 60:02d} {label}"

    # -- dates ------------------------------------------------------------------------
    @staticmethod
    def date(d) -> str:
        if d is None or pd.isna(d):
            return "–"
        d = pd.Timestamp(d)
        return f"{d.day} {d.strftime('%b %Y')}"

    @staticmethod
    def month(d) -> str:
        if d is None or pd.isna(d):
            return "–"
        return pd.Timestamp(d).strftime("%b %Y")

    @staticmethod
    def span(start, end) -> str:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s.year == e.year and s.month == e.month:
            return f"{s.day}–{e.day} {e.strftime('%b %Y')}"
        if s.year == e.year:
            return f"{s.day} {s.strftime('%b')} – {e.day} {e.strftime('%b %Y')}"
        return f"{Formatter.date(s)} – {Formatter.date(e)}"
