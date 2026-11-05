"""Plain-language findings generated from your data.

Wording rules (deliberate):
* describe patterns in *your* data and compare with *your* own baseline first;
* population references are cited, approximate and framed as context;
* never diagnose; where Apple itself recommends seeing a doctor (e.g.
  irregular rhythm notifications) we say so calmly;
* associations are not called causes.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..catalog import ALL_METRICS
from ..metrics import Metrics
from . import periods as P
from . import relationships as R
from .core import Context, build_context, kpis, window_value, yearly_table
from .records import current_streak, longest_streak, personal_records
from .stats import Trend, block_bootstrap_diff, trend


@dataclass
class Insight:
    id: str
    domain: str
    title: str
    detail: str
    tone: str = "neutral"          # positive | neutral | attention | info
    score: float = 50.0
    metric: Optional[str] = None
    caveat: Optional[str] = None
    numbers: Dict[str, object] = field(default_factory=dict)


@dataclass
class Analysis:
    ctx: Context
    insights: List[Insight]
    kpis: List[dict]
    yearly: pd.DataFrame
    records: List[dict]
    periods: List[P.Period]
    chapters: Dict[str, List[P.Chapter]]
    relationships: List[R.Finding]
    recent: List[P.Recent]
    trends: Dict[str, Trend]
    errors: List[str]
    generated_at: str

    def by_domain(self, domain: str) -> List[Insight]:
        return [i for i in self.insights if i.domain == domain]

    def key_findings(self, n: int = 8) -> List[Insight]:
        pool = [i for i in self.insights if i.domain != "data" and i.tone != "info"]
        return sorted(pool, key=lambda i: -i.score)[:n]


# FRIEND registry treadmill VO2max medians (Kaminsky et al., Mayo Clin Proc 2015), mL/kg/min.
FRIEND_MEDIAN = {
    "male": [(20, 48.0), (30, 42.4), (40, 37.8), (50, 32.6), (60, 28.2), (70, 24.4)],
    "female": [(20, 37.6), (30, 30.2), (40, 26.7), (50, 23.4), (60, 20.0), (70, 18.3)],
}


def friend_median(age: Optional[float], sex: Optional[str]) -> Optional[float]:
    if age is None or sex not in FRIEND_MEDIAN or age < 20 or age >= 80:
        return None
    for lo, val in reversed(FRIEND_MEDIAN[sex]):
        if age >= lo:
            return val
    return None


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _pct_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


# ---------------------------------------------------------------------------
# Data coverage
# ---------------------------------------------------------------------------
def gen_coverage(ctx: Context) -> List[Insight]:
    d, fmt = ctx.daily, ctx.fmt
    years = (ctx.last_day - ctx.first_day).days / 365.25
    worn_days = int(ctx.worn.sum())
    wear = d.loc[ctx.worn, "watch_wear_h"].mean() if "watch_wear_h" in d and worn_days else float("nan")
    src = ctx.m.sources
    devices = src[src["kind"].isin(["watch", "phone"])].copy()
    names = []
    for _, r in devices.sort_values("first").iterrows():
        if pd.isna(r.get("first")) or int(r.get("records", 0) or 0) < 50:
            continue
        a = pd.Timestamp(int(r["first"]), unit="s").year
        b = pd.Timestamp(int(r["last"]), unit="s").year
        names.append(f"{r['friendly'] or r['name']} ({a}–{b})" if a != b else f"{r['friendly'] or r['name']} ({a})")
    nights = int(d["sleep_any_h"].notna().sum()) if "sleep_any_h" in d else 0
    measured = int(d["sleep_h"].notna().sum()) if "sleep_h" in d else 0
    detail = (f"Your export covers {fmt.date(ctx.first_day)} to {fmt.date(ctx.last_day)} ({years:.1f} years). "
              f"The Watch was worn for most of the day on {fmt.num(worn_days)} days"
              + (f", about {wear:.0f} hours on a typical worn day" if math.isfinite(wear) else "") + ". "
              f"Sleep was recorded on {fmt.num(nights)} nights, {fmt.num(measured)} of them measured by a Watch or "
              f"sleep app (the rest are time in bed from the iPhone).")
    if names:
        detail += " Devices: " + "; ".join(names) + "."
    return [Insight("coverage", "data", f"{years:.1f} years of data", detail, tone="info", score=0,
                    numbers={"years": years, "worn_days": worn_days, "nights": nights, "measured": measured,
                             "devices": names})]


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------
def gen_steps(ctx: Context) -> List[Insight]:
    now = window_value(ctx, "steps", ctx.recent)
    if now is None:
        return []
    before = window_value(ctx, "steps", ctx.previous)
    fmt = ctx.fmt
    title = f"You average {fmt.num(now)} steps a day"
    detail = f"Over the last 12 months you took {fmt.num(now)} steps on an average day"
    score = 55.0
    if before:
        ch = _pct_change(now, before)
        title += f", {'up' if ch >= 0 else 'down'} {abs(ch):.0f}% on the year before"
        detail += f", compared with {fmt.num(before)} in the 12 months before that"
        score += min(25, abs(ch))
    detail += (". Large studies link more daily steps with lower risk of early death, with the benefit levelling "
               "off at roughly 8,000–10,000 steps for adults under 60 and 6,000–8,000 for older adults "
               "(Paluch et al., Lancet Public Health 2022).")
    tone = "positive" if now >= 8000 else ("attention" if now < 5000 else "neutral")
    return [Insight("steps", "activity", title, detail, tone=tone, score=score, metric="steps",
                    numbers={"now": now, "before": before})]


def gen_exercise(ctx: Context) -> List[Insight]:
    wk = ctx.weekly
    if "exercise_min" not in wk:
        return []
    full = wk[(wk["worn_days"] >= 6) & (wk.index >= ctx.recent.start)]
    if len(full) < 8:
        full = wk[wk["worn_days"] >= 6]
        if len(full) < 8:
            return []
        when = "in weeks you wore your Watch"
    else:
        when = "in the last 12 months"
    share = float((full["exercise_min"] >= 150).mean() * 100)
    median = float(full["exercise_min"].median())
    fmt = ctx.fmt
    title = f"You met the 150-minute weekly activity guideline in {share:.0f}% of weeks"
    detail = (f"{_cap(when)}, your Exercise ring minutes added up to {fmt.minutes(median)} in a typical week "
              f"({len(full)} fully-tracked weeks). The WHO recommends at least 150–300 minutes of moderate "
              f"(or 75–150 of vigorous) activity a week; Apple counts brisk walking and above as exercise.")
    tone = "positive" if share >= 75 else ("attention" if share < 35 else "neutral")
    return [Insight("exercise", "activity", title, detail, tone=tone, score=60 + abs(share - 50) / 3,
                    metric="exercise_min", numbers={"share": share, "median_week": median, "weeks": len(full)})]


def gen_rings(ctx: Context) -> List[Insight]:
    d = ctx.daily
    cols = [c for c in ("move_ring_pct", "exercise_ring_pct", "stand_ring_pct") if c in d]
    if len(cols) < 3:
        return []
    rec = ctx.recent.slice(d)[cols].dropna()
    if len(rec) < 30:
        rec = d[cols].dropna()
        if len(rec) < 30:
            return []
    rates = {c: float((rec[c] >= 100).mean() * 100) for c in cols}
    all3 = float((rec[cols] >= 100).all(axis=1).mean() * 100)
    closed_all = (d[cols] >= 100).all(axis=1).where(d[cols].notna().all(axis=1))
    best, s, e = longest_streak(closed_all)
    cur = current_streak(closed_all)
    fmt = ctx.fmt
    title = f"You closed all three rings on {all3:.0f}% of days"
    detail = (f"On days you wore the Watch in the last year: Move {rates['move_ring_pct']:.0f}%, "
              f"Exercise {rates['exercise_ring_pct']:.0f}%, Stand {rates['stand_ring_pct']:.0f}%.")
    if best >= 2:
        detail += f" Your longest all-rings streak was {best} days ({fmt.span(s, e)})."
    if cur >= 2:
        detail += f" You're currently on a {cur}-day streak."
    return [Insight("rings", "activity", title, detail, tone="positive" if all3 >= 50 else "neutral", score=45,
                    numbers={"rates": rates, "all3": all3, "best_streak": best})]


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def gen_weekday(ctx: Context) -> List[Insight]:
    d = ctx.daily
    if "steps" not in d:
        return []
    s = d["steps"].loc[d.index >= ctx.last_day - pd.Timedelta(days=730)].dropna()
    if len(s) < 60:
        return []
    by = s.groupby(s.index.day_of_week).mean()
    hi, lo = int(by.idxmax()), int(by.idxmin())
    weekend = pd.Series(s.index.day_of_week >= 5, index=s.index)
    diff = block_bootstrap_diff(s, weekend)
    fmt = ctx.fmt
    title = f"{WEEKDAYS[hi]}s are your most active day, {WEEKDAYS[lo]}s the quietest"
    detail = f"Over the last two years you averaged {fmt.num(by[hi])} steps on {WEEKDAYS[hi]}s and {fmt.num(by[lo])} on {WEEKDAYS[lo]}s."
    if diff is not None and diff.clear:
        more = "more" if diff.diff > 0 else "fewer"
        detail += f" Weekends average {fmt.num(abs(diff.diff))} {more} steps than weekdays."
    return [Insight("weekday", "activity", title, detail, score=35 + min(20, 100 * (by[hi] - by[lo]) / by.mean() / 3),
                    metric="steps", numbers={"by_weekday": by.to_dict()})]


def gen_time_of_day(ctx: Context) -> List[Insight]:
    h = ctx.m.hourly
    if h is None or len(h) == 0 or h["steps"].notna().sum() < 500:
        return []
    h = h[h["date"] >= ctx.last_day - pd.Timedelta(days=730)]
    days = h["date"].nunique()
    by = h.groupby("hour")["steps"].sum() / max(days, 1)
    peak = int(by.idxmax())
    total = by.sum()
    morning = by.loc[by.index < 12].sum() / total * 100
    evening = by.loc[by.index >= 18].sum() / total * 100
    fmt = ctx.fmt
    title = f"Your most active hour is {fmt.hour_label(peak)}–{fmt.hour_label((peak + 1) % 24)}"
    detail = (f"About {morning:.0f}% of your steps happen before noon and {evening:.0f}% after 6 pm "
              f"(last two years).")
    return [Insight("time_of_day", "activity", title, detail, score=30, metric="steps",
                    numbers={"peak_hour": peak, "morning_pct": morning, "evening_pct": evening})]


def gen_seasonality(ctx: Context) -> List[Insight]:
    mon = ctx.monthly
    if "steps" not in mon:
        return []
    s = mon["steps"].dropna()
    if len(s) < 24:
        return []
    rel = s / s.groupby(s.index.year).transform("mean")
    counts = s.groupby(s.index.year).transform("count")
    rel = rel[counts >= 9]
    if rel.index.year.nunique() < 2:
        return []
    prof = rel.groupby(rel.index.month).mean()
    hi, lo = int(prof.idxmax()), int(prof.idxmin())
    ratio = (prof[hi] / prof[lo] - 1) * 100
    if ratio < 8:
        return []
    names = pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B")
    title = f"You move about {ratio:.0f}% more in {names[hi - 1]} than in {names[lo - 1]}"
    detail = ("Comparing each month with the average of its own year (so long-term changes don't distort it), "
              f"{names[hi - 1]} is consistently your most active month and {names[lo - 1]} your least active.")
    return [Insight("seasonality", "trends", title, detail, score=40 + min(25, ratio / 2), metric="steps",
                    numbers={"profile": prof.to_dict(), "ratio": ratio})]


def gen_chapters(ctx: Context) -> List[Insight]:
    ch = P.chapters(ctx, "steps")
    if len(ch) < 2:
        return []
    fmt = ctx.fmt
    big = []
    for a, b in zip(ch[:-1], ch[1:]):
        change = _pct_change(b.mean, a.mean)
        if abs(change) >= 12:
            big.append((b, change))
    if not big:
        return []
    parts = [f"{fmt.month(c.start)} – {fmt.month(c.end)}: {fmt.num(c.mean)} steps/day" for c in ch]
    b, change = max(big, key=lambda t: abs(t[1]))
    title = (f"Your activity has had {len(ch)} distinct chapters — the biggest shift was "
             f"{'a rise' if change > 0 else 'a drop'} of {abs(change):.0f}% from {fmt.month(b.start)}")
    detail = "Periods where your typical daily steps stayed at a different level: " + "; ".join(parts) + "."
    return [Insight("chapters", "trends", title, detail, score=55 + min(25, abs(change) / 2), metric="steps",
                    numbers={"chapters": [(str(c.start.date()), str(c.end.date()), c.mean) for c in ch]})]


# ---------------------------------------------------------------------------
# Heart & fitness
# ---------------------------------------------------------------------------
def _trend_insight(ctx: Context, key: str, threshold: float, insight_id: str, domain: str = "heart"):
    mon = ctx.monthly
    if key not in mon:
        return None, None
    s = mon[key].dropna()
    if len(s) < 12:
        return None, None
    tr = trend(s)
    return tr, s


def gen_resting_hr(ctx: Context) -> List[Insight]:
    tr, s = _trend_insight(ctx, "resting_hr", 2.0, "rhr")
    now = window_value(ctx, "resting_hr", P_window(ctx, 90))
    if now is None:
        return []
    fmt = ctx.fmt
    first_year = s.loc[s.index < s.index[0] + pd.DateOffset(years=1)].median() if s is not None else float("nan")
    tone, score = "neutral", 50.0
    if tr is not None and tr.significant and abs(tr.change) >= 2:
        direction = "down" if tr.change < 0 else "up"
        title = (f"Resting heart rate is {direction} {abs(tr.change):.0f} bpm since {s.index[0].year} "
                 f"(now {fmt.value('resting_hr', now)})")
        tone = "positive" if tr.change < 0 else "attention"
        score = 60 + min(30, abs(tr.change) * 4)
    elif tr is not None:
        title = f"Resting heart rate has been steady around {fmt.value('resting_hr', now)}"
        score = 40
    else:
        title = f"Your resting heart rate is about {fmt.value('resting_hr', now)}"
    detail = f"Your typical resting heart rate over the last 90 days was {fmt.value('resting_hr', now)}"
    if math.isfinite(first_year) and s is not None:
        detail += f"; in your first year of Watch data it was {fmt.value('resting_hr', first_year)}"
    detail += (". For adults, 60–100 bpm is the usual range and regular aerobic exercise tends to lower it. "
               "Temporary rises of several bpm commonly accompany illness, short sleep, stress, alcohol or heat.")
    if now < 50:
        detail += (" Values below 50 bpm are common in fit people; if you also feel dizzy, faint or unusually "
                   "tired, mention it to a doctor.")
    return [Insight("resting_hr", "heart", title, detail, tone=tone, score=score, metric="resting_hr",
                    numbers={"now": now, "change": tr.change if tr else None, "p": tr.p_value if tr else None})]


def P_window(ctx: Context, days: int):
    from .core import Window
    return Window(ctx.last_day - pd.Timedelta(days=days - 1), ctx.last_day)


def gen_hrv(ctx: Context) -> List[Insight]:
    tr, s = _trend_insight(ctx, "hrv", 0, "hrv")
    now = window_value(ctx, "hrv", P_window(ctx, 90))
    if now is None or s is None:
        return []
    fmt = ctx.fmt
    change_pct = 100 * tr.change / tr.start_value if tr and tr.start_value else 0
    if tr is not None and tr.significant and abs(change_pct) >= 10:
        title = (f"Heart-rate variability is {'up' if change_pct > 0 else 'down'} about {abs(change_pct):.0f}% "
                 f"since {s.index[0].year} (now {fmt.value('hrv', now)})")
        tone = "positive" if change_pct > 0 else "neutral"
        score = 50 + min(25, abs(change_pct) / 2)
    else:
        title = f"Heart-rate variability has been fairly steady around {fmt.value('hrv', now)}"
        tone, score = "neutral", 35
    detail = (f"Your typical HRV (SDNN) over the last 90 days was {fmt.value('hrv', now)}. HRV differs a lot "
              "between people and naturally declines with age, so your own trend matters more than the number. "
              "Higher-than-usual values generally mean you're well recovered; drops often follow hard training, "
              "poor sleep, illness or stress.")
    return [Insight("hrv", "heart", title, detail, tone=tone, score=score, metric="hrv",
                    numbers={"now": now, "change_pct": change_pct})]


def gen_vo2(ctx: Context) -> List[Insight]:
    d = ctx.daily
    if "vo2max" not in d or d["vo2max"].notna().sum() < 3:
        return []
    s = d["vo2max"].dropna()
    recent = s.loc[s.index >= s.index[-1] - pd.Timedelta(days=180)]
    now = float(recent.median())
    fmt = ctx.fmt
    prof = ctx.m.profile
    ref = friend_median(prof.age, prof.sex)
    tr = trend(ctx.monthly["vo2max"].dropna(), min_points=8) if "vo2max" in ctx.monthly else None
    title = f"Cardio fitness (VO₂ max) is about {fmt.value('vo2max', now)}"
    tone, score = "neutral", 55.0
    detail = f"Based on {len(recent)} Watch estimates from your last six months of outdoor walks and runs"
    if ref is not None:
        rel = now - ref
        detail += (f". For context, the typical (median) value for {prof.sex} adults in your age group in a large "
                   f"US treadmill-test registry is roughly {ref:.0f} (FRIEND registry, 2015)")
        if rel >= 2:
            title += " — above typical for your age"
            tone, score = "positive", 65
        elif rel <= -2:
            title += " — below typical for your age"
            score = 65
    if tr is not None and tr.significant and abs(tr.change) >= 1.5:
        detail += (f". It has {'risen' if tr.change > 0 else 'fallen'} by about {abs(tr.change):.1f} since "
                   f"{tr_start_year(ctx, 'vo2max')}")
        score += 10
    detail += (". Cardio fitness is one of the strongest everyday predictors of long-term health, and it responds "
               "to regular brisk walking, running or cycling.")
    return [Insight("vo2max", "heart", title, detail, tone=tone, score=score, metric="vo2max",
                    caveat="Watch estimates can differ from a lab test by several points, especially for very fit "
                           "or less active people.",
                    numbers={"now": now, "reference": ref})]


def tr_start_year(ctx: Context, key: str) -> int:
    return int(ctx.monthly[key].dropna().index[0].year)


def gen_walking_hr(ctx: Context) -> List[Insight]:
    tr, s = _trend_insight(ctx, "walking_hr", 2.0, "walking_hr")
    if tr is None or not tr.significant or abs(tr.change) < 3:
        return []
    fmt = ctx.fmt
    title = f"Walking heart rate is {'down' if tr.change < 0 else 'up'} {abs(tr.change):.0f} bpm since {s.index[0].year}"
    detail = (f"Your average heart rate during everyday walking went from about {fmt.value('walking_hr', tr.start_value)} "
              f"to {fmt.value('walking_hr', tr.end_value)}. A falling walking heart rate usually reflects improving "
              f"fitness.")
    return [Insight("walking_hr", "heart", title, detail, tone="positive" if tr.change < 0 else "neutral",
                    score=45 + min(20, abs(tr.change) * 2), metric="walking_hr")]


def gen_hr_notifications(ctx: Context) -> List[Insight]:
    d = ctx.daily
    out = []
    if "irregular_rhythm_events" in d and d["irregular_rhythm_events"].sum() > 0:
        n = int(d["irregular_rhythm_events"].sum())
        days = d.index[d["irregular_rhythm_events"] > 0]
        out.append(Insight(
            "irregular_rhythm", "heart", f"Your Watch logged {n} irregular rhythm notification{'s' if n > 1 else ''}",
            f"Most recently on {ctx.fmt.date(days.max())}. Apple designed these notifications to flag a rhythm that "
            "may be atrial fibrillation (AFib). They are not a diagnosis, but Apple recommends discussing them with a "
            "doctor — worth doing if you haven't already.", tone="attention", score=95,
            numbers={"count": n}))
    if "high_hr_events" in d and d["high_hr_events"].sum() > 0:
        n = int(d["high_hr_events"].sum())
        out.append(Insight(
            "high_hr", "heart", f"{n} high heart-rate notification{'s' if n > 1 else ''}",
            "These fire when your heart rate stays above your chosen threshold while you seem inactive for about 10 "
            "minutes. An occasional one can come from stress, caffeine, illness or a loose band; frequent or "
            "unexplained ones are worth mentioning to a doctor.", tone="neutral", score=50 + min(30, n),
            numbers={"count": n}))
    if "low_hr_events" in d and d["low_hr_events"].sum() > 0:
        n = int(d["low_hr_events"].sum())
        out.append(Insight(
            "low_hr", "heart", f"{n} low heart-rate notification{'s' if n > 1 else ''}",
            "These fire when your heart rate stays below your chosen threshold for about 10 minutes. Common in very "
            "fit people and during sleep; if you also feel dizzy or faint, talk to a doctor.", tone="neutral",
            score=45 + min(30, n), numbers={"count": n}))
    return out


def gen_spo2(ctx: Context) -> List[Insight]:
    d = ctx.daily
    if "spo2" not in d or d["spo2"].notna().sum() < 14:
        return []
    s = d["spo2"].dropna()
    med = float(s.median())
    low_days = float((s < 92).mean() * 100)
    detail = (f"Your typical daily blood-oxygen average is {med:.0f}%. Readings of 95–100% are typical at sea level. "
              "Apple states the Blood Oxygen app is for wellness only; wrist readings are easily thrown off by fit, "
              "movement and cold hands.")
    tone = "neutral"
    if low_days > 10:
        detail += (f" On {low_days:.0f}% of days your average was below 92% — if that comes with breathlessness or "
                   "you snore heavily, it is worth raising with a doctor.")
        tone = "attention"
    return [Insight("spo2", "heart", f"Blood oxygen typically around {med:.0f}%", detail, tone=tone, score=30,
                    metric="spo2")]


def gen_weight(ctx: Context) -> List[Insight]:
    d = ctx.daily
    if "weight_kg" not in d:
        return []
    s = d["weight_kg"].dropna()
    if len(s) < 8 or (s.index[-1] - s.index[0]).days < 180:
        return []
    fmt = ctx.fmt
    first = s.iloc[: max(3, len(s) // 10)].median()
    last = s.iloc[-max(3, len(s) // 10):].median()
    title = f"Weight: {fmt.value('weight_kg', first)} → {fmt.value('weight_kg', last)}"
    detail = (f"From {len(s)} weigh-ins between {fmt.date(s.index[0])} and {fmt.date(s.index[-1])}. "
              f"Weight entries are often irregular, so short-term wiggles mean little.")
    return [Insight("weight", "body", title, detail, score=25, metric="weight_kg")]


# ---------------------------------------------------------------------------
# Sleep
# ---------------------------------------------------------------------------
def gen_sleep_coverage(ctx: Context) -> List[Insight]:
    n = ctx.m.nights
    if n is None or len(n) == 0:
        return [Insight("sleep_none", "sleep", "No sleep data recorded",
                        "Your export has no sleep records. Turning on Sleep Schedule on the iPhone records time in "
                        "bed; wearing the Watch to bed with the Sleep app measures actual sleep and stages.",
                        tone="info", score=0)]
    counts = n["kind"].value_counts()
    total = int(counts.sum())
    measured = int(counts.get("stages", 0) + counts.get("asleep", 0))
    fmt = ctx.fmt
    title = f"Sleep was recorded on {fmt.num(total)} nights"
    detail = (f"{fmt.num(measured)} nights were measured by the Watch or a sleep app "
              f"({int(counts.get('stages', 0))} with sleep stages); the other {fmt.num(int(counts.get('inbed', 0)))} "
              "only have time in bed from the iPhone's Bedtime / Sleep Schedule, which is based on your schedule and "
              "phone use rather than measured sleep. HealthLens keeps the two apart.")
    if measured < 30:
        detail += " Wearing the Watch to bed more often would make the sleep analysis much more informative."
    return [Insight("sleep_coverage", "sleep", title, detail, tone="info", score=0,
                    numbers={"total": total, "measured": measured, "stages": int(counts.get("stages", 0))})]


def gen_sleep_duration(ctx: Context) -> List[Insight]:
    d = ctx.daily
    fmt = ctx.fmt
    out = []
    if "sleep_h" in d and d["sleep_h"].notna().sum() >= 14:
        s = d["sleep_h"].dropna()
        share7 = float((s >= 7).mean() * 100)
        avg = float(s.mean())
        tone = "attention" if avg < 6.5 else ("positive" if avg >= 7 else "neutral")
        out.append(Insight(
            "sleep_measured", "sleep", f"On measured nights you slept {fmt.hours(avg)} on average",
            f"Across {len(s)} nights measured by the Watch or a sleep app, you got 7 hours or more on {share7:.0f}% of "
            "them. Sleep experts recommend at least 7 hours a night for adults (AASM / Sleep Research Society).",
            tone=tone, score=60 + (10 if avg < 6.5 else 0), metric="sleep_h",
            numbers={"avg": avg, "share7": share7, "n": len(s)}))
    if "in_bed_h" in d and d["in_bed_h"].notna().sum() >= 14:
        s = d["in_bed_h"].dropna()
        rec = s.loc[s.index >= ctx.recent.start]
        base = rec if len(rec) >= 14 else s
        avg = float(base.mean())
        when = "in the last 12 months" if len(rec) >= 14 else "overall"
        detail = (f"Your iPhone recorded an average of {fmt.hours(avg)} in bed per night {when} ({len(base)} nights). "
                  "Actual sleep is usually somewhat shorter than time in bed — healthy sleepers are asleep for "
                  "roughly 85–90% of it.")
        if "sleep_efficiency_pct" in d and d["sleep_efficiency_pct"].notna().sum() >= 7:
            eff = float(d["sleep_efficiency_pct"].median())
            detail += f" On nights with both measures, you were asleep for about {eff:.0f}% of your time in bed."
        tone = "attention" if avg < 7 else "neutral"
        out.append(Insight("time_in_bed", "sleep", f"You spend about {fmt.hours(avg)} in bed", detail, tone=tone,
                           score=50, metric="in_bed_h", numbers={"avg": avg}))
    return out


def gen_sleep_timing(ctx: Context) -> List[Insight]:
    n = ctx.m.nights
    if n is None or len(n) < 20:
        return []
    fmt = ctx.fmt
    n = n.copy()
    n["wd"] = pd.to_datetime(n["wake_date"]).dt.day_of_week
    recent = n[pd.to_datetime(n["wake_date"]) >= ctx.last_day - pd.Timedelta(days=730)]
    if len(recent) >= 20:
        n = recent
    bed = float(n["bedtime_h"].median())
    wake = float(n["wake_h"].median())
    sd_bed = float(n["bedtime_h"].std() * 60)
    # Free days = mornings of Sat/Sun (nights of Fri/Sat)
    free = n["wd"].isin([5, 6])
    detail = (f"Typical bedtime {fmt.clock(bed)}, typical wake-up {fmt.clock(wake)} (last two years, "
              f"{len(n)} nights). Your bedtime varies by about ±{sd_bed:.0f} minutes from night to night.")
    numbers = {"bedtime": bed, "wake": wake, "sd_bed_min": sd_bed}
    tone, score = "neutral", 45.0
    if free.sum() >= 8 and (~free).sum() >= 8:
        mid_free = float(n.loc[free, "midsleep_h"].median())
        mid_work = float(n.loc[~free, "midsleep_h"].median())
        jetlag = (mid_free - mid_work) * 60
        wake_shift = (n.loc[free, "wake_h"].median() - n.loc[~free, "wake_h"].median()) * 60
        detail += (f" On weekends you wake about {fmt.minutes(abs(wake_shift))} "
                   f"{'later' if wake_shift >= 0 else 'earlier'}; the middle of your sleep shifts by "
                   f"{fmt.minutes(abs(jetlag))} (sometimes called “social jet lag”).")
        numbers["social_jetlag_min"] = jetlag
        if abs(jetlag) >= 120:
            tone, score = "attention", 55
    if sd_bed >= 75:
        tone, score = "attention", max(score, 55)
        detail += " More regular sleep timing is linked with better sleep quality and overall health."
    return [Insight("sleep_timing", "sleep", f"You usually go to bed around {fmt.clock(bed)} and get up at {fmt.clock(wake)}",
                    detail, tone=tone, score=score, metric="bedtime_h", numbers=numbers)]


def gen_sleep_trend(ctx: Context) -> List[Insight]:
    d = ctx.daily
    if "sleep_any_h" not in d:
        return []
    g = d["sleep_any_h"].groupby(d.index.year)
    yearly = g.mean()[g.count() >= 20]
    if len(yearly) < 3:
        return []
    fmt = ctx.fmt
    first_y, last_y = yearly.index[0], yearly.index[-1]
    change = yearly.iloc[-1] - yearly.iloc[0]
    if abs(change) < 0.25:
        return []
    label = "time in bed / sleep"
    title = (f"You're getting {fmt.minutes(abs(change) * 60)} {'more' if change > 0 else 'less'} {label} than in "
             f"{first_y}")
    detail = ("Average per recorded night by year: " +
              ", ".join(f"{y}: {fmt.hours(v)}" for y, v in yearly.items()) + ".")
    return [Insight("sleep_trend", "sleep", title, detail, score=45 + min(20, abs(change) * 20), metric="sleep_any_h")]


def gen_sleep_stages(ctx: Context) -> List[Insight]:
    n = ctx.m.nights
    if n is None or len(n) == 0:
        return []
    st = n[n["kind"] == "stages"]
    if len(st) < 10:
        return []
    total = st["sleep_h"]
    deep = float((st["deep_h"] / total).median() * 100)
    rem = float((st["rem_h"] / total).median() * 100)
    core = float((st["core_h"] / total).median() * 100)
    awake = float(st["awake_h"].median() * 60)
    detail = (f"On {len(st)} nights with stage data: deep {deep:.0f}%, REM {rem:.0f}%, core (light) {core:.0f}% of "
              f"sleep, and about {awake:.0f} minutes awake. In adults REM is typically around 20–25% of sleep and "
              "deep sleep 10–20%, with deep sleep declining with age.")
    return [Insight("sleep_stages", "sleep", f"Your sleep is about {deep:.0f}% deep and {rem:.0f}% REM", detail,
                    score=40, caveat="Wrist-based sleep staging is an estimate and less precise than a sleep lab.",
                    numbers={"deep": deep, "rem": rem, "core": core, "awake_min": awake})]


# ---------------------------------------------------------------------------
# Workouts
# ---------------------------------------------------------------------------
def gen_workouts(ctx: Context) -> List[Insight]:
    w = ctx.m.workouts
    if w is None or len(w) < 5:
        return []
    fmt = ctx.fmt
    hours = w.groupby("label")["duration_min"].sum().sort_values(ascending=False) / 60
    total_h = hours.sum()
    top = hours.index[0]
    per_year = w.groupby(w["date"].dt.year).size()
    title = (f"{fmt.num(len(w))} workouts logged — {top.lower()} is your favourite "
             f"({hours.iloc[0] / total_h * 100:.0f}% of workout time)")
    top3 = ", ".join(f"{k} {v:.0f} h" for k, v in hours.head(4).items())
    detail = (f"{fmt.num(total_h)} hours in total: {top3}. Most workouts in a year: {int(per_year.max())} "
              f"({int(per_year.idxmax())}).")
    rec = w[w["date"] >= ctx.recent.start]
    prev = w[(w["date"] >= ctx.previous.start) & (w["date"] <= ctx.previous.end)]
    if len(rec) or len(prev):
        detail += f" Last 12 months: {len(rec)} workouts ({len(rec) / 52:.1f} a week)"
        if len(prev):
            detail += f", versus {len(prev)} the year before"
        detail += "."
    if ctx.m.info.get("workout_duplicates_removed"):
        detail += (f" ({ctx.m.info['workout_duplicates_removed']} duplicate copies of the same session from other "
                   "apps were ignored.)")
    return [Insight("workouts", "workouts", title, detail, score=50, numbers={"hours": hours.to_dict()})]


def gen_running(ctx: Context) -> List[Insight]:
    w = ctx.m.workouts
    if w is None or len(w) == 0:
        return []
    runs = w[(w["activity"] == "Running") & (w["distance_km"] >= 2) & w["pace_min_per_km"].between(2.5, 12)]
    if len(runs) < 15 or (runs["date"].max() - runs["date"].min()).days < 180:
        return []
    s = runs.set_index("date")["pace_min_per_km"].sort_index()
    tr = trend(s, min_points=15)
    if tr is None:
        return []
    fmt = ctx.fmt
    change_s = (tr.end_value - tr.start_value) * 60
    clear = tr.significant and abs(change_s) >= 10
    if clear:
        title = (f"Your running pace {'improved' if change_s < 0 else 'slowed'} by about {abs(change_s):.0f} s/km "
                 f"({fmt.pace(tr.start_value)} → {fmt.pace(tr.end_value)})")
        tone = "positive" if change_s < 0 else "neutral"
        score = 55 + min(25, abs(change_s) / 4)
    else:
        title = f"Your running pace has held around {fmt.pace(float(s.median()))}"
        tone, score = "neutral", 35
    detail = (f"Based on {len(runs)} runs of 2 km or more between {fmt.date(s.index[0])} and {fmt.date(s.index[-1])}, "
              f"totalling {fmt.distance(runs['distance_km'].sum(), 0)}.")
    return [Insight("running", "workouts", title, detail, tone=tone, score=score,
                    numbers={"change_s_per_km": change_s, "runs": len(runs), "significant": clear})]


def gen_consistency(ctx: Context) -> List[Insight]:
    wk = ctx.weekly
    if "workouts" not in wk:
        return []
    tracked = wk[wk["worn_days"] >= 5]
    rec = tracked[tracked.index >= ctx.recent.start]
    if len(rec) < 12:
        return []
    share = float((rec["workouts"].fillna(0) >= 1).mean() * 100)
    flags = (tracked["workouts"].fillna(0) >= 2)
    best, s, e = longest_streak(flags)
    fmt = ctx.fmt
    detail = f"In the last 12 months you did at least one workout in {share:.0f}% of weeks you wore the Watch."
    if best >= 3:
        detail += f" Your longest run of weeks with 2+ workouts was {best} weeks (from {fmt.date(s)})."
    return [Insight("consistency", "workouts", f"You worked out in {share:.0f}% of weeks", detail,
                    tone="positive" if share >= 75 else "neutral", score=40, numbers={"share": share})]


# ---------------------------------------------------------------------------
# Relationships, periods, recent changes
# ---------------------------------------------------------------------------
def gen_relationships(ctx: Context, findings: List[R.Finding]) -> List[Insight]:
    out = []
    fmt = ctx.fmt
    sig = [f for f in findings if f.significant]
    for f in sorted(sig, key=lambda f: -abs(f.rho))[:6]:
        p = f.pair
        if f.level == "daily":
            amount = fmt.delta(p.y, abs(f.diff)).lstrip("+−")
            better_word = ("more" if f.diff > 0 else "less") if p.y == "sleep_any_h" else \
                ("higher" if f.diff > 0 else "lower")
            title = f"{_cap(p.x_words)} and {p.y_words}: {amount} {better_word}"
        else:
            title = f"Long-term: more {p.x_words}, {'higher' if f.rho > 0 else 'lower'} {p.y_words}"
        detail = R.describe(f, ctx) + (f" (a {f.strength} association, ρ = {f.rho:+.2f}, {f.n} "
                                       f"{'days' if f.level == 'daily' else 'months'}).")
        out.append(Insight(f"rel_{p.x}_{p.y}_{f.level}", "relationships", title, detail,
                           score=45 + min(30, abs(f.rho) * 60), caveat="An association in your data, not proof "
                           "that one causes the other.", numbers={"rho": f.rho, "q": f.q, "n": f.n}))
    return out


def gen_periods(ctx: Context, periods: List[P.Period]) -> List[Insight]:
    if not periods:
        return []
    fmt = ctx.fmt
    latest = periods[0]
    ctx_bits = []
    if "hrv" in latest.context and latest.context["hrv"] <= -10:
        ctx_bits.append(f"HRV was {abs(latest.context['hrv']):.0f}% lower")
    if "steps" in latest.context and latest.context["steps"] <= -20:
        ctx_bits.append(f"you took {abs(latest.context['steps']):.0f}% fewer steps")
    if "sleep_any_h" in latest.context and latest.context["sleep_any_h"] >= 0.5:
        ctx_bits.append(f"you slept {fmt.minutes(latest.context['sleep_any_h'] * 60)} longer")
    detail = (f"HealthLens found {len(periods)} stretch{'es' if len(periods) > 1 else ''} where your resting heart "
              f"rate ran well above your own recent baseline. The most recent: {fmt.span(latest.start, latest.end)}, "
              f"peaking at {fmt.value('resting_hr', latest.peak)} (usual ≈ {fmt.value('resting_hr', latest.baseline)})")
    detail += (", while " + " and ".join(ctx_bits) + "." if ctx_bits else ".")
    detail += (" Patterns like this often line up with a cold or flu, a hard training block, travel, heat, alcohol "
               "or stress — the list in the Heart section may help you recall what was going on.")
    return [Insight("periods", "heart", f"{len(periods)} periods of raised resting heart rate", detail,
                    score=50 + min(20, latest.excess * 2), numbers={"count": len(periods)})]


RECENT_WORDS = {
    "steps": ("steps", "more", "fewer"), "exercise_min": ("exercise", "more", "less"),
    "active_kcal": ("active energy", "more", "less"), "resting_hr": ("resting heart rate", "higher", "lower"),
    "hrv": ("HRV", "higher", "lower"), "walking_hr": ("walking heart rate", "higher", "lower"),
    "sleep_any_h": ("sleep / time in bed", "more", "less"), "bedtime_h": ("bedtime", "later", "earlier"),
    "vo2max": ("cardio fitness", "higher", "lower"), "weight_kg": ("weight", "higher", "lower"),
    "resp_rate": ("sleeping respiratory rate", "higher", "lower"),
}


def gen_recent(ctx: Context, recent: List[P.Recent]) -> List[Insight]:
    notable = [r for r in recent if abs(r.z) >= 1.5]
    if not notable:
        return []
    fmt = ctx.fmt
    notable.sort(key=lambda r: -abs(r.z))
    bits = []
    for r in notable[:4]:
        words, up, down = RECENT_WORDS[r.key]
        if r.key == "bedtime_h":
            bits.append(f"{words} {fmt.minutes(abs(r.recent - r.usual) * 60)} {up if r.recent > r.usual else down}")
        else:
            bits.append(f"{words} {fmt.value(r.key, r.recent)} vs your usual {fmt.value(r.key, r.usual)}")
    title = "Last 30 days: " + ", ".join(
        f"{RECENT_WORDS[r.key][1] if r.recent > r.usual else RECENT_WORDS[r.key][2]} {RECENT_WORDS[r.key][0]}"
        for r in notable[:3])
    detail = ("Compared with your previous 12 months: " + "; ".join(bits) +
              ". Recent stretches are short, so treat this as a nudge to reflect rather than a verdict.")
    return [Insight("recent", "trends", title, detail, score=58, numbers={"keys": [r.key for r in notable]})]


# ---------------------------------------------------------------------------
GENERATORS: List[Callable[[Context], List[Insight]]] = [
    gen_coverage, gen_steps, gen_exercise, gen_rings, gen_weekday, gen_time_of_day, gen_seasonality, gen_chapters,
    gen_resting_hr, gen_hrv, gen_vo2, gen_walking_hr, gen_hr_notifications, gen_spo2, gen_weight,
    gen_sleep_coverage, gen_sleep_duration, gen_sleep_timing, gen_sleep_trend, gen_sleep_stages,
    gen_workouts, gen_running, gen_consistency,
]

TREND_KEYS = ["steps", "exercise_min", "active_kcal", "resting_hr", "walking_hr", "hrv", "vo2max", "weight_kg",
              "sleep_any_h", "in_bed_h", "sleep_h", "bedtime_h", "workout_min", "stand_hours", "flights",
              "walking_speed_kmh", "resp_rate", "spo2", "env_audio_db"]


def analyse(m: Metrics, units: Optional[str] = None) -> Analysis:
    ctx = build_context(m, units)
    errors: List[str] = []
    insights: List[Insight] = []

    def safe(fn, *args, default=None):
        try:
            return fn(*args)
        except Exception:  # keep the report going; record what failed
            errors.append(f"{getattr(fn, '__name__', fn)}: {traceback.format_exc(limit=3)}")
            return [] if default is None else default

    for gen in GENERATORS:
        insights += safe(gen, ctx)
    findings = safe(R.daily_findings, ctx) + safe(R.monthly_findings, ctx)
    insights += safe(gen_relationships, ctx, findings)
    periods = safe(P.elevated_rhr_periods, ctx)
    insights += safe(gen_periods, ctx, periods)
    recent = safe(P.recent_changes, ctx)
    insights += safe(gen_recent, ctx, recent)
    chapters = {k: safe(P.chapters, ctx, k) for k in ("steps", "exercise_min", "resting_hr")}
    trends = {}
    for key in TREND_KEYS:
        if key in ctx.monthly:
            t = safe(trend, ctx.monthly[key].dropna())
            if t:
                trends[key] = t
    return Analysis(ctx=ctx, insights=insights, kpis=safe(kpis, ctx), yearly=safe(yearly_table, ctx, default=pd.DataFrame()),
                    records=safe(personal_records, ctx), periods=periods, chapters=chapters, relationships=findings,
                    recent=recent, trends=trends, errors=errors,
                    generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
