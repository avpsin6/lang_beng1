"""Unit handling.

Apple Health stores each sample in the unit the user's iPhone preferred at the
time it was written ("km" vs "mi", "kg" vs "lb", "kcal" vs "kJ" ...). We
convert everything to one canonical unit per dimension for analysis and convert
back to the user's preferred system only for display.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Factor tables (value_in_unit * factor = value_in_base)
# ---------------------------------------------------------------------------
LENGTH_TO_M = {
    "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
    "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144,
}
MASS_TO_KG = {
    "kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.45359237,
    "oz": 0.028349523125, "st": 6.35029318,
}
ENERGY_TO_KCAL = {
    "kcal": 1.0, "Cal": 1.0, "cal": 0.001, "kJ": 0.2390057361, "J": 0.0002390057361,
}
TIME_TO_MIN = {
    "min": 1.0, "s": 1 / 60, "ms": 1 / 60000, "hr": 60.0, "h": 60.0, "d": 1440.0,
}
SPEED_TO_KMH = {
    "km/hr": 1.0, "km/h": 1.0, "m/s": 3.6, "mi/hr": 1.609344, "mi/h": 1.609344,
    "m/hr": 0.001, "ft/s": 1.09728,
}

# Canonical units whose raw HealthKit spelling differs but needs no scaling.
_ALIASES = {
    "bpm": ("count/min",),
    "br/min": ("count/min",),
    "mL/kg/min": ("mL/min·kg", "ml/min·kg", "mL/(kg·min)", "mL/kg·min"),
    "METs": ("kcal/hr·kg", "kcal/(kg·hr)", "kcal/kg·hr"),
    "kg/m²": ("count",),  # BodyMassIndex is exported with unit "count"
}

_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "m": LENGTH_TO_M,
    "kg": MASS_TO_KG,
    "kcal": ENERGY_TO_KCAL,
    "min": TIME_TO_MIN,
    "km/h": SPEED_TO_KMH,
}


def _temp_to_c(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "degF":
        return (values - 32.0) * 5.0 / 9.0
    if unit == "K":
        return values - 273.15
    return values


def converter(unit: Optional[str], canonical: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return a vectorised function converting ``unit`` -> ``canonical``.

    ``canonical`` is the unit the metric catalog wants (e.g. "km", "kg", "%").
    Returns None when no sensible conversion exists (caller keeps raw values).
    """
    unit = (unit or "").strip()
    if canonical == "%":
        # HealthKit percentages are fractions (0.97 == 97 %).
        if unit == "%":
            return lambda v: v * 100.0
        return None
    if unit == canonical or unit in _ALIASES.get(canonical, ()):
        return lambda v: v
    if canonical == "degC":
        if unit in ("degC", "degF", "K"):
            return lambda v: _temp_to_c(v, unit)
        return None
    if canonical == "mg/dL":
        if unit.startswith("mmol") and unit.endswith("/L"):
            return lambda v: v * 18.0156
        return None
    for base, table in _DIMENSIONS.items():
        if canonical in table and unit in table:
            factor = table[unit] / table[canonical]
            return lambda v, f=factor: v * f
    return None


def convert(values: np.ndarray, units: np.ndarray, canonical: str) -> np.ndarray:
    """Convert an array of values with per-row units to the canonical unit.

    Rows whose unit cannot be converted are returned as NaN (and should be
    rare: they indicate an app writing data in an unexpected unit).
    """
    values = np.asarray(values, dtype="float64")
    units = np.array([u if isinstance(u, str) else "" for u in np.asarray(units, dtype=object)],
                     dtype=object)
    out = np.full(values.shape, np.nan)
    for unit in set(units.tolist()):
        mask = units == unit
        if unit == "":
            out[mask] = values[mask]  # unit-less: keep as is
            continue
        fn = converter(unit, canonical)
        if fn is not None:
            out[mask] = fn(values[mask])
    return out


# ---------------------------------------------------------------------------
# Display system (metric / imperial)
# ---------------------------------------------------------------------------
# canonical unit -> (imperial unit label, factor)
_IMPERIAL = {
    "km": ("mi", 1 / 1.609344),
    "m": ("ft", 1 / 0.3048),
    "cm": ("in", 1 / 2.54),
    "kg": ("lb", 1 / 0.45359237),
    "km/h": ("mph", 1 / 1.609344),
    "degC": ("°F", None),  # handled specially
}

_LABELS = {
    "count": "", "kcal": "kcal", "min": "min", "bpm": "bpm", "ms": "ms",
    "%": "%", "km": "km", "m": "m", "cm": "cm", "kg": "kg", "km/h": "km/h",
    "degC": "°C", "dBASPL": "dB", "mL/kg/min": "mL/kg/min", "br/min": "breaths/min",
    "W": "W", "m/s": "m/s", "METs": "METs", "mmHg": "mmHg", "mg/dL": "mg/dL",
    "h": "h", "events": "", "kg/m²": "kg/m²",
}


class DisplayUnits:
    """Converts canonical values into the user's preferred display system."""

    def __init__(self, system: str = "metric"):
        if system not in ("metric", "imperial"):
            raise ValueError("units must be 'metric' or 'imperial'")
        self.system = system

    def label(self, canonical: str) -> str:
        if self.system == "imperial" and canonical in _IMPERIAL:
            return _IMPERIAL[canonical][0]
        return _LABELS.get(canonical, canonical)

    def factor(self, canonical: str) -> float:
        if self.system == "imperial" and canonical in _IMPERIAL and canonical != "degC":
            return float(_IMPERIAL[canonical][1])
        return 1.0

    def convert(self, value, canonical: str):
        """Convert a scalar/array from canonical to display units."""
        if value is None:
            return None
        if self.system == "imperial" and canonical == "degC":
            return np.asarray(value) * 9.0 / 5.0 + 32.0 if not np.isscalar(value) else value * 9.0 / 5.0 + 32.0
        return value * self.factor(canonical)

    def convert_delta(self, value, canonical: str):
        """Convert a difference (no offset for temperatures)."""
        if self.system == "imperial" and canonical == "degC":
            return value * 9.0 / 5.0
        return value * self.factor(canonical)


def detect_system(unit_counts: Dict[str, int]) -> str:
    """Guess metric vs imperial from how often each unit appears in the export."""
    imperial = sum(unit_counts.get(u, 0) for u in ("mi", "lb", "ft", "in", "degF", "mi/hr", "yd"))
    metric = sum(unit_counts.get(u, 0) for u in ("km", "kg", "cm", "m", "degC", "km/hr"))
    return "imperial" if imperial > metric else "metric"
