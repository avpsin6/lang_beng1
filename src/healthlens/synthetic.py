"""Synthetic Apple Health export generator.

Used by ``healthlens demo`` and by the test-suite, so the whole pipeline can be
exercised without anyone's real health data. The simulated person has:

* an iPhone for the whole period and an Apple Watch from ``watch_start``,
  with realistic overlap between Watch and iPhone step samples (the main
  de-duplication challenge in real exports);
* "life chapters" (normal -> lockdown dip -> running phase -> steady -> more
  sedentary), seasonality, weekday rhythms and day-to-day noise;
* built-in relationships for the insights engine to rediscover: exercise ->
  slightly longer sleep that night, short sleep -> higher resting heart rate
  and lower HRV next morning, fitness -> lower resting HR / higher VO2 max;
* illness episodes (resting HR up, HRV down, activity down for a few days);
* sleep tracked only occasionally (bursts of Watch sleep tracking, iPhone
  Bedtime "in bed" on most nights, a month of a third-party sleep app);
* the old (iOS <= 15) and new (iOS 16+) export formats for sleep values and
  workouts, DST changes in the timestamps, and Unicode source names.

The numbers are plausible, not physiologically exact.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPOCH = date(1970, 1, 1)


def _day(d: date) -> int:
    return (d - EPOCH).days


def _date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    start: date = date(2017, 3, 1)
    end: date = date(2026, 10, 31)
    watch_start: Optional[date] = date(2018, 10, 6)
    watch_hardware: str = "Watch4,2"            # Series 4, 44 mm
    export_ios: int = 18                        # >= 16: new sleep values + WorkoutStatistics
    seed: int = 42
    owner: str = "Alex"
    units: str = "metric"                       # "metric" | "imperial"
    tz: str = "us-eastern"                      # "us-eastern" | "uk" | "+0530" | "+0000"
    hr_every_min: int = 20                      # background heart-rate sampling interval
    workout_hr_every_s: int = 60
    samples_per_hour: int = 1                   # step/distance samples per active hour per device
    sleep_tracking: str = "rare"                # "rare" | "often" | "never"
    third_party: bool = True                    # AutoSleep month + Strava duplicates
    dob: str = "1990-05-14"
    sex: str = "HKBiologicalSexMale"


TZ_NAMES = {"us-eastern": "America/New_York", "uk": "Europe/London", "+0530": "Asia/Kolkata",
            "+0000": "UTC"}


class _TZ:
    """Tiny DST model for timestamps (avoids depending on system tz data)."""

    def __init__(self, spec: str):
        self.spec = spec
        if spec == "us-eastern":
            self.std, self.dst, self.rule = -300, -240, "us"
        elif spec == "uk":
            self.std, self.dst, self.rule = 0, 60, "eu"
        else:
            sign = -1 if spec.startswith("-") else 1
            hh, mm = int(spec[1:3]), int(spec[3:5])
            self.std = self.dst = sign * (hh * 60 + mm)
            self.rule = None
        self._cache: Dict[int, Tuple[int, int, int]] = {}

    @staticmethod
    def _nth_sunday(year: int, month: int, n: int) -> date:
        d = date(year, month, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    @staticmethod
    def _last_sunday(year: int, month: int) -> date:
        d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() + 1) % 7)

    def _transitions(self, year: int) -> Tuple[int, int]:
        if self.rule == "us":
            return _day(self._nth_sunday(year, 3, 2)), _day(self._nth_sunday(year, 11, 1))
        return _day(self._last_sunday(year, 3)), _day(self._last_sunday(year, 10))

    def day_info(self, day: int) -> Tuple[int, int, int]:
        """(offset at local midnight, local second of change or -1, offset after)."""
        hit = self._cache.get(day)
        if hit:
            return hit
        if self.rule is None:
            info = (self.std, -1, self.std)
        else:
            year = _date(day).year
            start, end = self._transitions(year)
            change_s = 2 * 3600 if self.rule == "us" else 3600
            if day == start:
                info = (self.std, change_s, self.dst)
            elif day == end:
                info = (self.dst, change_s, self.std)
            elif start < day < end:
                info = (self.dst, -1, self.dst)
            else:
                info = (self.std, -1, self.std)
        self._cache[day] = info
        return info

    def offset(self, local_s: int) -> int:
        day, sec = divmod(int(local_s), 86400)
        before, change, after = self.day_info(day)
        if change < 0 or sec < change:
            return before
        return after


# ---------------------------------------------------------------------------
# Software versions over time
# ---------------------------------------------------------------------------
WATCHOS = [(date(2018, 9, 17), "5.0"), (date(2019, 9, 19), "6.0"), (date(2020, 9, 16), "7.0"),
           (date(2021, 9, 20), "8.0"), (date(2022, 9, 12), "9.0"), (date(2023, 9, 18), "10.0")]
IOS = [(date(2016, 9, 13), "10.0"), (date(2017, 9, 19), "11.0"), (date(2018, 9, 17), "12.0"),
       (date(2019, 9, 19), "13.0"), (date(2020, 9, 16), "14.0"), (date(2021, 9, 20), "15.0"),
       (date(2022, 9, 12), "16.0"), (date(2023, 9, 18), "17.0"), (date(2024, 9, 16), "18.0"),
       (date(2025, 9, 15), "26.0")]
PHONES = [(date(2010, 1, 1), "iPhone9,3"), (date(2019, 10, 2), "iPhone12,1"), (date(2023, 10, 6), "iPhone15,4")]


def _version(table, d: date, max_major: Optional[int] = None) -> str:
    """Software version on date ``d``; minor version creeps up through the year."""
    eligible = [(since, ver) for since, ver in table
                if d >= since and (max_major is None or _major(ver) <= max_major)]
    since, ver = eligible[-1] if eligible else table[0]
    minor = min(6, max(0, (d - since).days) // 60)
    return f"{_major(ver)}.{minor}"


def _major(version: str) -> int:
    return int(version.split(".")[0])


# ---------------------------------------------------------------------------
# XML writer
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


DTD = """<!DOCTYPE HealthData [
<!-- HealthKit Export Version: 14 -->
<!ELEMENT HealthData (ExportDate,Me,(Record|Correlation|Workout|ActivitySummary|ClinicalRecord|Audiogram|VisionPrescription)*)>
<!ATTLIST HealthData
  locale CDATA #REQUIRED
>
<!ELEMENT ExportDate EMPTY>
<!ATTLIST ExportDate
  value CDATA #REQUIRED
>
<!ELEMENT Me EMPTY>
<!ATTLIST Me
  HKCharacteristicTypeIdentifierDateOfBirth                   CDATA #REQUIRED
  HKCharacteristicTypeIdentifierBiologicalSex                 CDATA #REQUIRED
  HKCharacteristicTypeIdentifierBloodType                     CDATA #REQUIRED
  HKCharacteristicTypeIdentifierFitzpatrickSkinType           CDATA #REQUIRED
  HKCharacteristicTypeIdentifierCardioFitnessMedicationsUse   CDATA #REQUIRED
>
<!ELEMENT Record ((MetadataEntry|HeartRateVariabilityMetadataList)*)>
<!ATTLIST Record
  type          CDATA #REQUIRED
  unit          CDATA #IMPLIED
  value         CDATA #IMPLIED
  sourceName    CDATA #REQUIRED
  sourceVersion CDATA #IMPLIED
  device        CDATA #IMPLIED
  creationDate  CDATA #IMPLIED
  startDate     CDATA #REQUIRED
  endDate       CDATA #REQUIRED
>
<!ELEMENT MetadataEntry EMPTY>
<!ATTLIST MetadataEntry
  key   CDATA #REQUIRED
  value CDATA #REQUIRED
>
<!ELEMENT Workout ((MetadataEntry|WorkoutEvent|WorkoutRoute|WorkoutStatistics)*)>
<!ATTLIST Workout
  workoutActivityType   CDATA #REQUIRED
  duration              CDATA #IMPLIED
  durationUnit          CDATA #IMPLIED
  totalDistance         CDATA #IMPLIED
  totalDistanceUnit     CDATA #IMPLIED
  totalEnergyBurned     CDATA #IMPLIED
  totalEnergyBurnedUnit CDATA #IMPLIED
  sourceName            CDATA #REQUIRED
  sourceVersion         CDATA #IMPLIED
  device                CDATA #IMPLIED
  creationDate          CDATA #IMPLIED
  startDate             CDATA #REQUIRED
  endDate               CDATA #REQUIRED
>
<!ELEMENT ActivitySummary EMPTY>
<!ATTLIST ActivitySummary
  dateComponents           CDATA #IMPLIED
  activeEnergyBurned       CDATA #IMPLIED
  activeEnergyBurnedGoal   CDATA #IMPLIED
  activeEnergyBurnedUnit   CDATA #IMPLIED
  appleMoveTime            CDATA #IMPLIED
  appleMoveTimeGoal        CDATA #IMPLIED
  appleExerciseTime        CDATA #IMPLIED
  appleExerciseTimeGoal    CDATA #IMPLIED
  appleStandHours          CDATA #IMPLIED
  appleStandHoursGoal      CDATA #IMPLIED
>
]>
"""


class _Out:
    def __init__(self, fh, tz: _TZ, rng: np.random.Generator):
        self.fh = fh
        self.tz = tz
        self.rng = rng
        self.lines: List[str] = []
        self._prefix: Dict[int, str] = {}
        self.count = 0

    def ts(self, local_s: float) -> str:
        local_s = int(local_s)
        day, sec = divmod(local_s, 86400)
        prefix = self._prefix.get(day)
        if prefix is None:
            prefix = self._prefix[day] = _date(day).strftime("%Y-%m-%d ")
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        off = self.tz.offset(local_s)
        sign = "-" if off < 0 else "+"
        off = abs(off)
        return f"{prefix}{h:02d}:{m:02d}:{s:02d} {sign}{off // 60:02d}{off % 60:02d}"

    def record(self, rtype: str, src: str, unit: Optional[str], start: float, end: float, value,
               meta: Optional[Sequence[Tuple[str, str]]] = None) -> None:
        unit_attr = f' unit="{unit}"' if unit else ""
        head = (f'<Record type="{rtype}" {src}{unit_attr} creationDate="{self.ts(end + 90)}" '
                f'startDate="{self.ts(start)}" endDate="{self.ts(end)}" value="{value}"')
        if meta:
            body = "".join(f'\n  <MetadataEntry key="{k}" value="{_esc(v)}"/>' for k, v in meta)
            self.lines.append(f" {head}>{body}\n </Record>\n")
        else:
            self.lines.append(f" {head}/>\n")
        self.count += 1

    def raw(self, text: str) -> None:
        self.lines.append(text)

    def flush(self) -> None:
        if self.lines:
            self.fh.write("".join(self.lines))
            self.lines = []


# ---------------------------------------------------------------------------
# The simulated person
# ---------------------------------------------------------------------------
@dataclass
class _Workout:
    activity: str
    start: float      # local seconds
    minutes: float
    distance_km: float
    kcal: float
    hr_avg: float
    hr_max: float
    steps: float
    indoor: bool
    source: str = "watch"


def _phase(d: date) -> Tuple[str, float]:
    if d < date(2018, 10, 1):
        return "early", 1.0
    if d < date(2020, 3, 15):
        return "normal", 1.05
    if d < date(2021, 6, 1):
        return "lockdown", 0.63
    if d < date(2024, 1, 1):
        return "active", 1.22
    if d < date(2025, 6, 1):
        return "steady", 1.0
    return "desk", 0.88


WORKOUT_PLAN = {
    # weekday (0=Mon) -> list of (activity, probability, summer_only)
    "normal": {0: [("Walking", .6, False)], 2: [("Walking", .6, False)], 3: [("TraditionalStrengthTraining", .4, False)],
               5: [("Cycling", .55, True)]},
    "lockdown": {1: [("HighIntensityIntervalTraining", .6, False)], 3: [("HighIntensityIntervalTraining", .6, False)],
                 6: [("Yoga", .5, False)], 4: [("Walking", .45, False)], 2: [("Walking", .35, False)]},
    "active": {1: [("Running", .75, False)], 3: [("Running", .75, False)], 5: [("Running", .8, False)],
               0: [("TraditionalStrengthTraining", .5, False)], 6: [("Cycling", .4, True)]},
    "steady": {5: [("Running", .6, False)], 2: [("Yoga", .5, False)], 0: [("Walking", .5, False)],
               4: [("Walking", .5, False)]},
    "desk": {5: [("Walking", .5, False)], 6: [("Walking", .45, False)], 2: [("TraditionalStrengthTraining", .3, False)]},
    "early": {},
}

INDOOR = {"HighIntensityIntervalTraining", "Yoga", "TraditionalStrengthTraining"}


class _Person:
    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.days = np.arange(_day(cfg.start), _day(cfg.end) + 1)
        self.n = len(self.days)
        self._latents()

    def _latents(self) -> None:
        cfg, rng, n = self.cfg, self.rng, self.n
        dates = [_date(d) for d in self.days]
        self.dates = dates
        self.weekday = np.array([d.weekday() for d in dates])
        doy = np.array([d.timetuple().tm_yday for d in dates])
        years = np.array([(d - date(2016, 1, 1)).days / 365.25 for d in dates])
        self.years = years
        self.phase = [_phase(d) for d in dates]
        phase_factor = np.array([p[1] for p in self.phase])
        # smooth phase transitions over ~3 weeks
        kernel = np.ones(21) / 21
        phase_factor = np.convolve(np.pad(phase_factor, 10, mode="edge"), kernel, mode="valid")
        season = 1 + 0.13 * np.sin(2 * np.pi * (doy - 100) / 365.25)
        wk = np.array([1.0, 1.0, 1.02, 1.0, 1.05, 1.18, 0.84])[self.weekday]
        z = np.zeros(n)
        e = rng.normal(0, 0.24, n)
        for i in range(1, n):
            z[i] = 0.35 * z[i - 1] + e[i]

        # illness episodes ~1.5 per year, 4-8 days
        ill = np.zeros(n, dtype=bool)
        i = int(rng.integers(30, 200))
        while i < n:
            length = int(rng.integers(4, 9))
            ill[i:i + length] = True
            i += length + int(rng.integers(150, 330))
        self.ill = ill

        # vacations: 2 per year, 7-12 days; some without the watch
        vac = np.zeros(n, dtype=bool)
        no_watch = np.zeros(n, dtype=bool)
        for year in range(cfg.start.year, cfg.end.year + 1):
            for month in (rng.integers(4, 7), rng.integers(8, 12)):
                s = _day(date(year, int(month), int(rng.integers(1, 20)))) - self.days[0]
                length = int(rng.integers(7, 13))
                if 0 <= s < n:
                    vac[s:s + length] = True
                    if rng.random() < 0.35:
                        no_watch[s:s + length] = True
        self.vacation = vac

        # watch wear
        wstart = _day(cfg.watch_start) if cfg.watch_start else 10 ** 9
        self.has_watch = self.days >= wstart
        not_worn = (rng.random(n) < 0.05) | no_watch
        for s, length in ((date(2021, 1, 11), 14), (date(2024, 8, 5), 20)):  # broken strap, long trip
            k = _day(s) - self.days[0]
            if 0 <= k < n:
                not_worn[k:k + length] = True
        self.worn = self.has_watch & ~not_worn

        # workouts
        self.workouts: List[List[_Workout]] = [[] for _ in range(n)]
        exercise = np.zeros(n)
        fitness = 20.0
        self.fitness = np.zeros(n)
        weight = 79.0
        self.weight = np.zeros(n)
        for k in range(n):
            ph = self.phase[k][0]
            d = dates[k]
            summer = d.month in (5, 6, 7, 8, 9)
            self.fitness[k] = fitness
            if self.worn[k] and not ill[k]:
                for act, p, summer_only in WORKOUT_PLAN.get(ph, {}).get(int(self.weekday[k]), []):
                    if summer_only and not summer:
                        continue
                    if vac[k]:
                        p *= 0.4
                    if rng.random() < p:
                        self.workouts[k].append(self._make_workout(act, k, fitness, weight))
            ex = sum(w.minutes * (0.7 if w.activity == "Walking" else 0.25 if w.activity == "Yoga" else 1.0)
                     for w in self.workouts[k])
            exercise[k] = ex
            fitness += (min(ex, 90) + 8 - fitness) / 42.0
            target_w = {"early": 79, "normal": 78.5, "lockdown": 80, "active": 74.5, "steady": 75.5, "desk": 77}[ph]
            weight += (target_w - weight) / 120.0 + rng.normal(0, 0.05)
            self.weight[k] = weight
        self.workout_ex = exercise

        # sleep: night ending on the morning of day k
        weekend_morning = np.isin(self.weekday, (5, 6))
        bedtime = np.where(np.isin(self.weekday, (5, 6)), 24.45, 23.3) + rng.normal(0, 0.45, n)
        bedtime -= np.where(np.roll(exercise, 1) >= 30, 0.18, 0.0)          # exercise yesterday -> earlier
        wake = np.where(weekend_morning, 8.2, 6.85) + np.where(weekend_morning, rng.normal(0, 0.55, n),
                                                               rng.normal(0, 0.22, n))
        wake += np.where(ill, 1.0, 0.0)
        wake += np.where(np.roll(exercise, 1) >= 30, 0.07, 0.0)
        vacation_shift = np.where(vac, 0.5, 0.0)
        bedtime += vacation_shift
        wake += vacation_shift * 1.6
        latency = np.clip(rng.normal(0.25, 0.1, n), 0.05, 0.8)
        awake_mid = np.clip(rng.gamma(2.0, 0.12, n), 0.02, 1.2)
        self.bedtime = bedtime                       # hours after midnight of the previous day (e.g. 23.3)
        self.wake = wake                             # hours after midnight of day k
        self.asleep_h = (wake + 24 - bedtime) - latency - awake_mid
        self.latency = latency
        self.awake_mid = awake_mid

        # physiology
        sleep_short = np.clip(7.0 - self.asleep_h, 0, 3)
        fit = np.minimum(self.fitness, 60)
        self.rhr = (65.5 + 0.12 * years - 0.13 * fit + 0.9 * sleep_short + 6.5 * ill
                    + rng.normal(0, 1.1, n))
        self.hrv = np.exp(np.log(47) - 0.012 * years + 0.009 * (fit - 25) + 0.06 * (self.asleep_h - 7)
                          - 0.28 * ill + rng.normal(0, 0.16, n))
        self.vo2 = 35.5 + 0.17 * fit - 0.15 * years + rng.normal(0, 0.6, n)
        self.walking_hr = 112 - 0.3 * fit + 2.0 * ill + rng.normal(0, 2.2, n)
        steps = (8200 * phase_factor * season * wk * np.exp(z) * np.where(ill, 0.45, 1.0)
                 * (1 + 0.04 * np.clip(self.asleep_h - 7, -3, 2)) * np.where(vac, 1.3, 1.0))
        self.base_steps = steps

        # recorded sleep sources
        tracked = np.zeros(n, dtype="U10")       # "", "watch", "autosleep"
        inbed_phone = np.zeros(n, dtype=bool)
        mode = cfg.sleep_tracking
        watch7 = date(2020, 9, 16)
        bursts = [(date(2020, 10, 5), 21), (date(2021, 3, 10), 14), (date(2022, 11, 1), 30),
                  (date(2023, 6, 12), 14), (date(2024, 1, 8), 10), (date(2025, 2, 3), 21), (date(2026, 3, 1), 12)]
        for k in range(n):
            d = dates[k]
            if not self.has_watch[k] or d < watch7 or mode == "never":
                pass
            elif mode == "often":
                if rng.random() < 0.8:
                    tracked[k] = "watch"
            else:
                if any(s <= d < s + timedelta(days=L) for s, L in bursts) or rng.random() < 0.03:
                    tracked[k] = "watch"
            if cfg.third_party and date(2021, 7, 5) <= d < date(2021, 8, 2) and self.has_watch[k]:
                tracked[k] = "autosleep"
            if date(2018, 1, 15) <= d < date(2019, 9, 1):
                inbed_phone[k] = rng.random() < 0.6
            elif d >= date(2020, 10, 1):
                inbed_phone[k] = rng.random() < 0.75
        tracked[~self.worn] = ""
        self.sleep_tracked = tracked
        self.inbed_phone = inbed_phone

    def _make_workout(self, act: str, k: int, fitness: float, weight: float) -> _Workout:
        rng = self.rng
        wd = int(self.weekday[k])
        hour = 9.5 if wd >= 5 else (7.0 if rng.random() < 0.5 else 18.5)
        start = self.days[k] * 86400 + hour * 3600 + rng.integers(0, 1800)
        wf = weight / 75.0
        if act == "Running":
            minutes = rng.uniform(28, 55)
            pace = max(4.9, 7.3 - 0.035 * fitness + rng.normal(0, 0.15))      # min/km
            dist = minutes / pace
            return _Workout(act, start, minutes, dist, 10.8 * minutes * wf, 160 - 0.35 * fitness + rng.normal(0, 4),
                            178 - 0.2 * fitness, minutes * 165, False)
        if act == "Walking":
            minutes = rng.uniform(30, 62)
            dist = minutes / rng.uniform(10.2, 12.0)
            return _Workout(act, start, minutes, dist, 4.6 * minutes * wf, 108 + rng.normal(0, 4), 128, minutes * 112,
                            False)
        if act == "Cycling":
            minutes = rng.uniform(45, 95)
            dist = minutes / 60 * rng.uniform(17, 25)
            return _Workout(act, start, minutes, dist, 8.0 * minutes * wf, 136 + rng.normal(0, 5), 165, 0, False)
        if act == "HighIntensityIntervalTraining":
            minutes = rng.uniform(20, 32)
            return _Workout(act, start, minutes, 0, 10.0 * minutes * wf, 146 + rng.normal(0, 5), 176, minutes * 40, True)
        if act == "Yoga":
            minutes = rng.uniform(30, 46)
            return _Workout(act, start, minutes, 0, 3.0 * minutes * wf, 92 + rng.normal(0, 4), 112, 0, True)
        minutes = rng.uniform(35, 52)
        return _Workout(act, start, minutes, 0, 5.2 * minutes * wf, 116 + rng.normal(0, 5), 148, minutes * 15, True)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
WEEKDAY_PROFILE = np.array([0, 0, 0, 0, 0, 0, 0.2, 1.6, 1.4, 0.6, 0.5, 0.6, 1.3, 0.9, 0.5, 0.5, 0.7, 1.4, 1.5, 0.9,
                            0.7, 0.4, 0.2, 0.05])
WEEKEND_PROFILE = np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 0.9, 1.3, 1.4, 1.2, 1.1, 1.2, 1.3, 1.2, 1.0, 0.9, 0.8,
                            0.6, 0.4, 0.2, 0.05])

HK_Q = "HKQuantityTypeIdentifier"
HK_C = "HKCategoryTypeIdentifier"


def generate_export(path: Path, cfg: Optional[SyntheticConfig] = None) -> pd.DataFrame:
    """Write a synthetic export (``.zip`` or ``.xml``) and return per-day ground truth."""
    cfg = cfg or SyntheticConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            with zf.open("apple_health_export/export.xml", "w", force_zip64=True) as raw:
                fh = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
                truth = _write(fh, cfg)
                fh.flush()
                fh.detach()
            zf.writestr("apple_health_export/export_cda.xml",
                        '<?xml version="1.0" encoding="UTF-8"?>\n<ClinicalDocument/>\n')
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            truth = _write(fh, cfg)
    return truth


def _write(fh, cfg: SyntheticConfig) -> pd.DataFrame:
    p = _Person(cfg)
    rng = p.rng
    tz = _TZ(cfg.tz)
    out = _Out(fh, tz, rng)
    imperial = cfg.units == "imperial"
    dist_unit, dist_factor = ("mi", 1 / 1.609344) if imperial else ("km", 1.0)
    mass_unit, mass_factor = ("lb", 1 / 0.45359237) if imperial else ("kg", 1.0)
    tzname = TZ_NAMES.get(cfg.tz, "UTC")
    watch_name = f"{cfg.owner}’s Apple Watch"
    phone_name = f"{cfg.owner}’s iPhone"
    asleep_value = "HKCategoryValueSleepAnalysisAsleepUnspecified" if cfg.export_ios >= 16 else \
        "HKCategoryValueSleepAnalysisAsleep"
    new_workouts = cfg.export_ios >= 16

    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    fh.write(DTD)
    fh.write('<HealthData locale="en_US">\n')
    fh.write(f' <ExportDate value="{out.ts(_day(cfg.end) * 86400 + 21 * 3600 + 300)}"/>\n')
    fh.write(f' <Me HKCharacteristicTypeIdentifierDateOfBirth="{cfg.dob}" '
             f'HKCharacteristicTypeIdentifierBiologicalSex="{cfg.sex}" '
             f'HKCharacteristicTypeIdentifierBloodType="HKBloodTypeNotSet" '
             f'HKCharacteristicTypeIdentifierFitzpatrickSkinType="HKFitzpatrickSkinTypeNotSet" '
             f'HKCharacteristicTypeIdentifierCardioFitnessMedicationsUse="None"/>\n')

    truth_rows = []
    activity_rows = []
    workouts_xml: List[str] = []
    high_hr_days = set(rng.choice(np.flatnonzero(p.worn), size=min(3, int(p.worn.sum())), replace=False).tolist()) \
        if p.worn.any() else set()

    height_written = False
    for k in range(p.n):
        day = int(p.days[k])
        d = p.dates[k]
        midnight = day * 86400
        ios = _version(IOS, d)
        phone_hw = [hw for since, hw in PHONES if d >= since][-1]
        addr = f"0x{int(rng.integers(0x280000000, 0x2ffffffff)):x}"
        phone_dev = _esc(f"<<HKDevice: {addr}>, name:iPhone, manufacturer:Apple Inc., model:iPhone, "
                         f"hardware:{phone_hw}, software:{ios}>")
        phone_src = f'sourceName="{phone_name}" sourceVersion="{ios}" device="{phone_dev}"'
        wos = _version(WATCHOS, d, max_major=10 if cfg.watch_hardware.startswith("Watch4") or
                       cfg.watch_hardware.startswith("Watch5") else (8 if cfg.watch_hardware.startswith("Watch3")
                                                                    else None))
        watch_dev = _esc(f"<<HKDevice: {addr}>, name:Apple Watch, manufacturer:Apple Inc., model:Watch, "
                         f"hardware:{cfg.watch_hardware}, software:{wos}>")
        watch_src = f'sourceName="{watch_name}" sourceVersion="{wos}" device="{watch_dev}"'
        clock_src = f'sourceName="Clock" sourceVersion="{ios}" device="{phone_dev}"'
        health_src = f'sourceName="Health" sourceVersion="{ios}"'
        autosleep_src = 'sourceName="AutoSleep" sourceVersion="6.8.2"'
        strava_src = 'sourceName="Strava" sourceVersion="300.1"'
        wmaj = _major(wos)
        imaj = _major(ios)
        worn = bool(p.worn[k])
        tracked = str(p.sleep_tracked[k])
        tomorrow_tracked = str(p.sleep_tracked[k + 1]) if k + 1 < p.n else ""

        # ---------------- sleep for the night ending this morning ----------------
        bed_s = midnight - 86400 + p.bedtime[k] * 3600
        onset_s = bed_s + p.latency[k] * 3600
        wake_s = midnight + p.wake[k] * 3600
        night_truth = {"sleep_tracked": tracked, "inbed_phone": bool(p.inbed_phone[k])}
        if p.inbed_phone[k]:
            src = clock_src if imaj <= 13 else phone_src
            ib_start = bed_s - rng.uniform(0, 900)
            ib_end = wake_s + rng.uniform(120, 1800)
            out.record(f"{HK_C}SleepAnalysis", src, None, ib_start, ib_end, "HKCategoryValueSleepAnalysisInBed",
                       [("HKTimeZone", tzname)])
            night_truth["inbed_h"] = (ib_end - ib_start) / 3600
        recorded_asleep = 0.0
        if tracked:
            segments = _sleep_segments(rng, onset_s, wake_s, p.awake_mid[k], stages=(tracked == "watch" and wmaj >= 9))
            src = watch_src if tracked == "watch" else autosleep_src
            if tracked == "autosleep":
                out.record(f"{HK_C}SleepAnalysis", src, None, bed_s, wake_s + 300, "HKCategoryValueSleepAnalysisInBed")
            for s0, s1, stage in segments:
                value = {"asleep": asleep_value, "core": "HKCategoryValueSleepAnalysisAsleepCore",
                         "deep": "HKCategoryValueSleepAnalysisAsleepDeep", "rem": "HKCategoryValueSleepAnalysisAsleepREM",
                         "awake": "HKCategoryValueSleepAnalysisAwake"}[stage]
                if stage == "awake" and not (tracked == "watch" and wmaj >= 9):
                    continue
                out.record(f"{HK_C}SleepAnalysis", src, None, s0, s1, value, [("HKTimeZone", tzname)])
                if stage != "awake":
                    recorded_asleep += s1 - s0
            night_truth["asleep_recorded_h"] = recorded_asleep / 3600
            # overnight heart rate, HRV, respiratory rate
            t = onset_s + 300
            while t < wake_s:
                hr = p.rhr[k] - 4 + rng.normal(0, 2.0)
                out.record(f"{HK_Q}HeartRate", watch_src, "count/min", t, t, f"{max(38, hr):.0f}",
                           [("HKMetadataKeyHeartRateMotionContext", "1")])
                t += 600
            for _ in range(2):
                t = rng.uniform(onset_s + 1800, wake_s - 1800)
                out.record(f"{HK_Q}HeartRateVariabilitySDNN", watch_src, "ms", t, t + 60,
                           f"{p.hrv[k] * rng.uniform(1.0, 1.35):.4f}")
            if wmaj >= 8 and tracked == "watch":
                t = onset_s + 1200
                while t < wake_s - 600:
                    rr = 14.2 + (0.9 if p.ill[k] else 0) + rng.normal(0, 0.5)
                    out.record(f"{HK_Q}RespiratoryRate", watch_src, "count/min", t, t + 120, f"{rr:.1f}")
                    t += 3600 * rng.uniform(0.6, 1.4)
        night_truth["asleep_true_h"] = float(p.asleep_h[k])

        # ---------------- wear window for the day ----------------
        wake_today = wake_s
        bed_tonight = midnight + p.bedtime[k + 1] * 3600 if k + 1 < p.n else midnight + 23.3 * 3600
        if worn:
            if tomorrow_tracked:
                charge = (midnight + 20.5 * 3600, midnight + 21.5 * 3600)
                wear_end = bed_tonight
            else:
                charge = (midnight + 7.3 * 3600, midnight + 7.9 * 3600) if rng.random() < 0.5 else \
                    (midnight + 19.0 * 3600, midnight + 20.0 * 3600)
                wear_end = bed_tonight - 900
            wear_start = wake_today + 1200
        else:
            charge = (0, 0)
            wear_start = wear_end = 0

        def worn_at(t: float) -> bool:
            return worn and wear_start <= t < wear_end and not (charge[0] <= t < charge[1])

        # ---------------- steps / distance / energy by hour ----------------
        weekend = d.weekday() >= 5
        profile = (WEEKEND_PROFILE if weekend else WEEKDAY_PROFILE) * rng.uniform(0.6, 1.4, 24)
        wake_h = p.wake[k]
        profile[: int(wake_h)] = 0
        profile = profile / profile.sum()
        hourly = p.base_steps[k] * profile
        wk_steps = np.zeros(24)
        for w in p.workouts[k]:
            h = int((w.start - midnight) // 3600)
            if 0 <= h < 24:
                wk_steps[h] += w.steps * 0.6
                if h + 1 < 24:
                    wk_steps[h + 1] += w.steps * 0.4
        hourly = hourly + wk_steps
        stride_km = 0.00074 + 0.00002 * rng.normal()
        watch_steps = phone_steps = expected = 0.0
        active_by_hour = np.zeros(24)
        stood = 0
        exercise_today = 0.0
        for h in range(24):
            s_h = float(hourly[h])
            h0 = midnight + h * 3600
            if s_h < 5:
                continue
            mid = h0 + 1800
            w_on = worn_at(mid)
            carried = rng.random() < (0.75 if not weekend else 0.6) or not p.has_watch[k]
            n_seg = max(1, cfg.samples_per_hour) + (1 if s_h > 2500 else 0)
            seg = 3600 / n_seg
            parts = rng.dirichlet(np.ones(n_seg)) * s_h
            hour_watch = hour_phone = 0.0
            for j in range(n_seg):
                v = parts[j]
                if v < 3:
                    continue
                dur = min(seg - 60, max(60.0, v / 95.0 * 60))
                st = h0 + j * seg + rng.uniform(0, max(1.0, seg - dur - 30))
                en = st + dur
                if w_on:
                    out.record(f"{HK_Q}StepCount", watch_src, "count", st, en, f"{v:.0f}")
                    out.record(f"{HK_Q}DistanceWalkingRunning", watch_src, dist_unit, st, en,
                               f"{v * stride_km * dist_factor:.5f}")
                    hour_watch += round(v)
                if carried:
                    pv = v * rng.uniform(0.78, 0.98)
                    ps = st + rng.uniform(-120, 90)
                    pe = en + rng.uniform(-60, 150)
                    out.record(f"{HK_Q}StepCount", phone_src, "count", ps, pe, f"{pv:.0f}")
                    out.record(f"{HK_Q}DistanceWalkingRunning", phone_src, dist_unit, ps, pe,
                               f"{pv * stride_km * 0.97 * dist_factor:.5f}")
                    hour_phone += round(pv)
            watch_steps += hour_watch
            phone_steps += hour_phone
            expected += hour_watch if w_on else hour_phone
            active_by_hour[h] = s_h
        # flights
        n_flights = rng.poisson(6 * p.phase[k][1])
        for _ in range(int(n_flights)):
            t = midnight + rng.uniform(wake_h + 0.5, 22) * 3600
            src = watch_src if worn_at(t) else phone_src
            out.record(f"{HK_Q}FlightsClimbed", src, "count", t, t + rng.uniform(15, 60), f"{rng.integers(1, 3)}")

        # ---------------- watch-only signals ----------------
        hr_samples = 0
        active_today = 0.0
        if worn:
            # workouts first (they shape HR and energy)
            wk_by_hour = np.zeros(24)
            for w in p.workouts[k]:
                end = w.start + w.minutes * 60
                h = int((w.start - midnight) // 3600)
                if 0 <= h < 24:
                    wk_by_hour[h] += w.kcal
                t = w.start
                while t < end:
                    frac = (t - w.start) / max(1.0, end - w.start)
                    ramp = min(1.0, frac * 6)
                    hr = 95 + (w.hr_avg - 95) * ramp + rng.normal(0, 4)
                    out.record(f"{HK_Q}HeartRate", watch_src, "count/min", t, t, f"{hr:.0f}",
                               [("HKMetadataKeyHeartRateMotionContext", "2")])
                    hr_samples += 1
                    t += cfg.workout_hr_every_s
                workouts_xml.append(_workout_xml(w, watch_src, out, new_workouts, dist_unit, dist_factor, tzname,
                                                 rng, p.vo2[k]))
                if cfg.third_party and w.activity == "Running" and date(2022, 1, 1) <= d < date(2023, 1, 1) \
                        and rng.random() < 0.6:
                    dup = _Workout(w.activity, w.start + 20, w.minutes - 0.6, w.distance_km * 0.99, w.kcal * 0.95,
                                   w.hr_avg, w.hr_max, w.steps, w.indoor, source="strava")
                    workouts_xml.append(_workout_xml(dup, strava_src, out, new_workouts, dist_unit, dist_factor,
                                                     tzname, rng, None))
                if w.activity in ("Running", "Walking", "Hiking") and w.minutes >= 20:
                    out.record(f"{HK_Q}VO2Max", watch_src, "mL/min·kg", end + 60, end + 60, f"{p.vo2[k]:.2f}",
                               [("HKVO2MaxTestType", "2")])
                exercise_today += w.minutes * (0.7 if w.activity == "Walking" else
                                               0.25 if w.activity == "Yoga" else 1.0)
            # background heart rate
            t = wear_start
            while t < wear_end:
                if worn_at(t) and not any(w.start <= t < w.start + w.minutes * 60 for w in p.workouts[k]):
                    h = int((t - midnight) // 3600)
                    act = active_by_hour[h] if 0 <= h < 24 else 0
                    hr = p.rhr[k] + 11 + min(25.0, act / 120.0) + rng.normal(0, 4)
                    ctx = "2" if act > 1500 else "1"
                    out.record(f"{HK_Q}HeartRate", watch_src, "count/min", t, t, f"{hr:.0f}",
                               [("HKMetadataKeyHeartRateMotionContext", ctx)])
                    hr_samples += 1
                t += cfg.hr_every_min * 60 * rng.uniform(0.8, 1.2)
            # resting / walking HR, HRV
            out.record(f"{HK_Q}RestingHeartRate", watch_src, "count/min", midnight + 300, wear_end - 60,
                       f"{p.rhr[k]:.0f}")
            if p.base_steps[k] > 2500:
                out.record(f"{HK_Q}WalkingHeartRateAverage", watch_src, "count/min", midnight + 300, wear_end - 60,
                           f"{p.walking_hr[k]:.0f}")
            for _ in range(3):
                t = rng.uniform(wear_start, max(wear_start + 1, wear_end))
                if worn_at(t):
                    out.record(f"{HK_Q}HeartRateVariabilitySDNN", watch_src, "ms", t, t + 60,
                               f"{p.hrv[k] * rng.uniform(0.7, 1.1):.4f}")
            # active / basal energy, exercise, stand
            for h in range(24):
                h0 = midnight + h * 3600
                basal = (1650 + 12 * (p.weight[k] - 75)) / 24 * rng.uniform(0.97, 1.03)
                out.record(f"{HK_Q}BasalEnergyBurned", watch_src, "kcal", h0, h0 + 3599, f"{basal:.3f}")
                if not worn_at(h0 + 1800):
                    continue
                act_kcal = active_by_hour[h] * 0.042 + wk_by_hour[h] + rng.uniform(2, 8)
                out.record(f"{HK_Q}ActiveEnergyBurned", watch_src, "kcal", h0 + 60, h0 + 3540, f"{act_kcal:.3f}")
                active_today += act_kcal
                # brisk walking outside workouts (workout minutes are credited separately below)
                brisk = min(50.0, max(0.0, (active_by_hour[h] - wk_steps[h] - 1100) / 95.0))
                if brisk >= 1:
                    out.record(f"{HK_Q}AppleExerciseTime", watch_src, "min", h0 + 300, h0 + 300 + brisk * 60,
                               f"{brisk:.0f}")
                    exercise_today += round(brisk)
                is_stood = active_by_hour[h] > 120 or rng.random() < 0.55
                out.record(f"{HK_C}AppleStandHour", watch_src, None, h0, h0 + 3600,
                           "HKCategoryValueAppleStandHourStood" if is_stood else "HKCategoryValueAppleStandHourIdle")
                stood += int(is_stood)
                if wmaj >= 7:
                    out.record(f"{HK_Q}AppleStandTime", watch_src, "min", h0 + 600, h0 + 1200,
                               f"{rng.integers(2, 14) if is_stood else rng.integers(0, 2)}")
                if wmaj >= 6:
                    db = 58 + rng.normal(0, 6) + (10 if h in (8, 17, 18) and not weekend else 0)
                    out.record(f"{HK_Q}EnvironmentalAudioExposure", watch_src, "dBASPL", h0 + 900, h0 + 2700,
                               f"{db:.2f}")
            # workout exercise minutes as explicit samples
            for w in p.workouts[k]:
                credit = w.minutes * (0.7 if w.activity == "Walking" else 0.25 if w.activity == "Yoga" else 1.0)
                if credit >= 1:
                    out.record(f"{HK_Q}AppleExerciseTime", watch_src, "min", w.start, w.start + credit * 60,
                               f"{credit:.0f}")
            # SpO2 on Series 6+
            if cfg.watch_hardware >= "Watch6" and wmaj >= 7:
                for _ in range(3):
                    t = rng.uniform(wear_start, max(wear_start + 1, wear_end))
                    out.record(f"{HK_Q}OxygenSaturation", watch_src, "%", t, t, f"{min(1.0, rng.normal(0.972, 0.012)):.2f}")
            if k in high_hr_days:
                t = midnight + 15 * 3600
                out.record(f"{HK_C}HighHeartRateEvent", watch_src, None, t, t + 600, "HKCategoryValueNotApplicable",
                           [("HKHeartRateEventThreshold", "120 count/min")])
            if rng.random() < 0.1:
                t = rng.uniform(wear_start, max(wear_start + 1, wear_end))
                out.record(f"{HK_C}MindfulSession", watch_src, None, t, t + 60 * int(rng.integers(1, 6)),
                           "HKCategoryValueNotApplicable")
            activity_rows.append((d, active_today, exercise_today, stood))
        elif p.has_watch[k]:
            activity_rows.append((d, 0.0, 0.0, 0))

        # ---------------- iPhone-only signals ----------------
        if imaj >= 13 and rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 3))):
                t = midnight + rng.uniform(8, 21) * 3600
                out.record(f"{HK_Q}HeadphoneAudioExposure", phone_src, "dBASPL", t, t + rng.uniform(900, 3600),
                           f"{rng.normal(71, 5):.2f}")
        if imaj >= 14 and p.base_steps[k] > 2000:
            speed = 4.9 + 0.012 * p.fitness[k] + rng.normal(0, 0.15)
            for _ in range(int(rng.integers(1, 4))):
                t = midnight + rng.uniform(8, 20) * 3600
                sp = speed if not imperial else speed / 1.609344
                out.record(f"{HK_Q}WalkingSpeed", phone_src, "km/hr" if not imperial else "mi/hr", t, t + 30,
                           f"{sp:.3f}")
                out.record(f"{HK_Q}WalkingStepLength", phone_src, "cm" if not imperial else "in", t, t + 30,
                           f"{(72 + rng.normal(0, 2)) / (2.54 if imperial else 1):.2f}")
                out.record(f"{HK_Q}WalkingDoubleSupportPercentage", phone_src, "%", t, t + 30,
                           f"{0.27 + rng.normal(0, 0.01):.3f}")
                out.record(f"{HK_Q}WalkingAsymmetryPercentage", phone_src, "%", t, t + 30,
                           f"{max(0, rng.normal(0.01, 0.01)):.3f}")
        if imaj >= 15 and d.weekday() == 0:
            out.record(f"{HK_Q}AppleWalkingSteadiness", phone_src, "%", midnight - 7 * 86400, midnight,
                       f"{min(1, rng.normal(0.95, 0.02)):.2f}")
        # manual weight entries in some phases
        if (date(2019, 1, 1) <= d < date(2020, 12, 31) or date(2022, 2, 1) <= d < date(2024, 6, 30)) \
                and d.weekday() == 6 and rng.random() < 0.8:
            t = midnight + 8 * 3600 + rng.uniform(0, 1800)
            out.record(f"{HK_Q}BodyMass", health_src, mass_unit, t, t,
                       f"{(p.weight[k] + rng.normal(0, 0.35)) * mass_factor:.1f}", [("HKWasUserEntered", "1")])
        if not height_written:
            out.record(f"{HK_Q}Height", health_src, "cm" if not imperial else "ft", midnight + 43200, midnight + 43200,
                       "178" if not imperial else "5.84", [("HKWasUserEntered", "1")])
            height_written = True

        truth_rows.append({
            "date": pd.Timestamp(d), "steps_true": float(hourly.sum()), "steps_watch": watch_steps,
            "steps_phone": phone_steps, "steps_expected": expected, "worn": worn,
            "workouts": len(p.workouts[k]), "rhr": float(p.rhr[k]), "hrv": float(p.hrv[k]),
            "ill": bool(p.ill[k]), "phase": p.phase[k][0], "hr_samples": hr_samples,
            **night_truth,
        })
        if len(out.lines) > 20000:
            out.flush()
    out.flush()

    # workouts & activity summaries go after records, like Apple's exports
    fh.write("".join(workouts_xml))
    move_goal = 500
    for d, active, ex, stood in activity_rows:
        if d >= date(2022, 1, 1):
            move_goal = 600
        fh.write(f' <ActivitySummary dateComponents="{d.isoformat()}" activeEnergyBurned="{active:.3f}" '
                 f'activeEnergyBurnedGoal="{move_goal}" activeEnergyBurnedUnit="kcal" appleMoveTime="0" '
                 f'appleMoveTimeGoal="0" appleExerciseTime="{ex:.0f}" appleExerciseTimeGoal="30" '
                 f'appleStandHours="{stood}" appleStandHoursGoal="12"/>\n')
    fh.write("</HealthData>\n")
    return pd.DataFrame(truth_rows)


def _sleep_segments(rng, onset: float, wake: float, awake_h: float, stages: bool) -> List[Tuple[float, float, str]]:
    """Split a night into asleep / stage segments with a few awakenings."""
    segs: List[Tuple[float, float, str]] = []
    total = wake - onset
    n_wake = int(rng.integers(1, 4))
    awake_total = awake_h * 3600
    wake_points = sorted(rng.uniform(onset + 0.15 * total, wake - 0.1 * total, n_wake))
    awake_each = awake_total / n_wake
    t = onset
    blocks = []
    for wp in wake_points:
        if wp - t > 600:
            blocks.append((t, wp))
            t = wp + awake_each
            segs.append((wp, wp + awake_each, "awake"))
    if wake - t > 300:
        blocks.append((t, wake))
    if not stages:
        return sorted([(a, b, "asleep") for a, b in blocks] + segs)
    out: List[Tuple[float, float, str]] = list(segs)
    for a, b in blocks:
        t = a
        while t < b - 60:
            frac = (t - onset) / max(1.0, total)
            deep = max(3.0, 24 * (1 - frac) ** 1.5 + rng.normal(0, 3)) * 60
            rem = max(4.0, 8 + 20 * frac + rng.normal(0, 3)) * 60
            core1 = rng.uniform(25, 40) * 60
            core2 = rng.uniform(10, 20) * 60
            for dur, stage in ((core1, "core"), (deep, "deep"), (core2, "core"), (rem, "rem")):
                e = min(b, t + dur)
                if e - t >= 60:
                    out.append((t, e, stage))
                t = e
                if t >= b - 60:
                    break
    return sorted(out)


def _workout_xml(w: _Workout, src: str, out: _Out, new_format: bool, dist_unit: str, dist_factor: float,
                 tzname: str, rng, vo2) -> str:
    end = w.start + w.minutes * 60
    act = f"HKWorkoutActivityType{w.activity}"
    attrs = (f'workoutActivityType="{act}" duration="{w.minutes:.4f}" durationUnit="min"')
    dist = w.distance_km * dist_factor
    if not new_format:
        if w.distance_km > 0:
            attrs += f' totalDistance="{dist:.4f}" totalDistanceUnit="{dist_unit}"'
        attrs += f' totalEnergyBurned="{w.kcal:.3f}" totalEnergyBurnedUnit="kcal"'
    head = (f' <Workout {attrs} {src} creationDate="{out.ts(end + 30)}" startDate="{out.ts(w.start)}" '
            f'endDate="{out.ts(end)}">\n')
    body = [f'  <MetadataEntry key="HKIndoorWorkout" value="{1 if w.indoor else 0}"/>\n',
            f'  <MetadataEntry key="HKTimeZone" value="{tzname}"/>\n']
    if not w.indoor:
        body.append(f'  <MetadataEntry key="HKElevationAscended" value="{rng.uniform(500, 9000):.0f} cm"/>\n')
        body.append(f'  <MetadataEntry key="HKWeatherTemperature" value="{rng.uniform(35, 85):.0f} degF"/>\n')
        body.append(f'  <MetadataEntry key="HKWeatherHumidity" value="{rng.uniform(3000, 9000):.0f} %"/>\n')
    body.append(f'  <MetadataEntry key="HKAverageMETs" value="{w.kcal / max(1, w.minutes) * 60 / 75:.4f} kcal/hr·kg"/>\n')
    n_pauses = int(rng.integers(0, 2))
    for _ in range(n_pauses):
        t = w.start + rng.uniform(0.2, 0.8) * w.minutes * 60
        body.append(f'  <WorkoutEvent type="HKWorkoutEventTypePause" date="{out.ts(t)}"/>\n')
        body.append(f'  <WorkoutEvent type="HKWorkoutEventTypeResume" date="{out.ts(t + 40)}"/>\n')
    if new_format:
        body.append(f'  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" startDate="{out.ts(w.start)}" '
                    f'endDate="{out.ts(end)}" sum="{w.kcal:.3f}" unit="kcal"/>\n')
        body.append(f'  <WorkoutStatistics type="HKQuantityTypeIdentifierHeartRate" startDate="{out.ts(w.start)}" '
                    f'endDate="{out.ts(end)}" average="{w.hr_avg:.2f}" minimum="{w.hr_avg - 30:.0f}" '
                    f'maximum="{w.hr_max:.0f}" unit="count/min"/>\n')
        if w.distance_km > 0:
            dtype = "DistanceCycling" if w.activity == "Cycling" else "DistanceWalkingRunning"
            body.append(f'  <WorkoutStatistics type="HKQuantityTypeIdentifier{dtype}" startDate="{out.ts(w.start)}" '
                        f'endDate="{out.ts(end)}" sum="{dist:.4f}" unit="{dist_unit}"/>\n')
        if w.steps > 0:
            body.append(f'  <WorkoutStatistics type="HKQuantityTypeIdentifierStepCount" startDate="{out.ts(w.start)}" '
                        f'endDate="{out.ts(end)}" sum="{w.steps:.0f}" unit="count"/>\n')
    if not w.indoor and w.source == "watch":
        stamp = out.ts(w.start)[:16].replace(" ", "_").replace(":", ".")
        body.append(f'  <WorkoutRoute {src} creationDate="{out.ts(end + 60)}" startDate="{out.ts(w.start)}" '
                    f'endDate="{out.ts(end)}">\n   <FileReference path="/workout-routes/route_{stamp}.gpx"/>\n'
                    f'  </WorkoutRoute>\n')
    return head + "".join(body) + " </Workout>\n"
