"""Assemble the offline HTML report."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..analysis import Analysis
from ..analysis.relationships import describe
from ..catalog import ALL_METRICS, DOMAIN_LABELS, label_for_type
from .charts import Charts

PKG = Path(__file__).resolve().parent.parent
TEMPLATES = PKG / "templates"
STATIC = PKG / "static"

# Blocks every network request from the page: the report can never phone home.
CSP = ("default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; "
       "img-src data: blob:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'")

TILE_LABELS = {"steps": "Steps / day", "exercise_min": "Exercise / day", "resting_hr": "Resting heart rate",
               "hrv": "HRV (SDNN)", "vo2max": "Cardio fitness (VO₂ max)", "workouts_n": "Workouts / week",
               "weight_kg": "Weight"}
PERCENT_DELTA = {"steps", "exercise_min", "workouts_n", "active_kcal"}


def plotly_js() -> str:
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _json_for_script(obj) -> str:
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


# ---------------------------------------------------------------------------
def _sparkline(series: pd.Series, n_recent: int = 12, months: int = 24) -> Optional[dict]:
    s = series.dropna().tail(months)
    if len(s) < 4:
        return None
    w, h, pad = 200.0, 32.0, 3.0
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo or 1.0
    xs = np.linspace(0, w, len(s))
    ys = h - pad - (s.to_numpy() - lo) / span * (h - 2 * pad)
    pts = [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)]
    split = max(0, len(pts) - n_recent)
    return {"context": " ".join(pts[: split + 1]) if split > 0 else "", "current": " ".join(pts[split:])}


def _tile(a: Analysis, k: dict) -> dict:
    fmt = a.ctx.fmt
    key = k["key"]
    m = ALL_METRICS[key]
    label = k["label"] if key == "sleep_any_h" else TILE_LABELS.get(key, m.label)
    v = k["value"]
    if m.unit == "h":
        number, unit = fmt.hours(v), ""
    elif key == "workouts_n":
        number, unit = fmt.num(v, 1), ""
    elif key == "exercise_min":
        number, unit = fmt.num(v), "min"
    else:
        number, unit = fmt.value(key, v, with_unit=False), fmt.unit_label(key)
    delta_text, cls = "No earlier year to compare", ""
    d = k.get("delta")
    if d is not None and k.get("previous"):
        prev = k["previous"]
        if key in PERCENT_DELTA:
            pct = 100 * d / prev if prev else 0
            small = abs(pct) < 2
            amount = f"{abs(pct):.0f}%"
        else:
            disp = fmt.delta(key, d)
            amount = disp.lstrip("+−")
            small = abs(d) < (0.08 if m.unit == "h" else (10 ** -m.decimals) * 0.5 + (0.5 if m.decimals == 0 else 0))
        if small:
            delta_text = "About the same as the year before"
        else:
            arrow = "▲" if d > 0 else "▼"
            delta_text = f"{arrow} {amount} vs the year before"
            if m.better in ("higher", "lower"):
                good = (d > 0) == (m.better == "higher")
                cls = "good" if good else "bad"
    return {"label": label, "number": number, "unit": unit, "delta": delta_text, "cls": cls,
            "spark": _sparkline(k["spark"])}


def _insight_dict(i) -> dict:
    return {"title": i.title, "detail": i.detail, "tone": i.tone, "caveat": i.caveat, "domain": i.domain,
            "domain_label": DOMAIN_LABELS.get(i.domain, i.domain.title()) if i.domain != "relationships"
            else "Connections"}


def _workout_table(a: Analysis) -> List[list]:
    w = a.ctx.m.workouts
    fmt = a.ctx.fmt
    if w is None or len(w) == 0:
        return []
    rows = []
    for label, g in sorted(w.groupby("label"), key=lambda t: -t[1]["duration_min"].sum()):
        dist = g["distance_km"].sum(min_count=1)
        rows.append([label, f"{len(g):,}", fmt.num(g["duration_min"].sum() / 60, 0),
                     fmt.minutes(g["duration_min"].mean()),
                     fmt.distance(dist, 0) if pd.notna(dist) and dist > 0 else "–",
                     fmt.num(g["avg_hr"].mean()) if g["avg_hr"].notna().any() else "–",
                     fmt.date(g["date"].max())])
    return rows


def _periods_table(a: Analysis) -> List[list]:
    fmt = a.ctx.fmt
    rows = []
    for p in a.periods:
        c = p.context
        rows.append([fmt.span(p.start, p.end), f"{p.days}", fmt.value("resting_hr", p.peak),
                     fmt.value("resting_hr", p.baseline),
                     fmt.pct(c["hrv"], 0, signed=True) if "hrv" in c else "–",
                     fmt.pct(c["steps"], 0, signed=True) if "steps" in c else "–",
                     fmt.delta("sleep_any_h", c["sleep_any_h"]) if "sleep_any_h" in c else "–"])
    return rows


def _relationship_rows(a: Analysis) -> List[dict]:
    rows = []
    for f in sorted(a.relationships, key=lambda f: (not f.significant, -abs(f.rho))):
        text = describe(f, a.ctx) if f.significant and math.isfinite(f.diff) else "No clear link in your data."
        rows.append({"question": f.pair.question, "answer": text, "clear": f.significant,
                     "rho": f"{f.rho:+.2f}", "n": f"{f.n:,} {'days' if f.level == 'daily' else 'months'}"})
    return rows


def _yearly_rows(a: Analysis):
    y = a.yearly
    fmt = a.ctx.fmt
    if y is None or len(y) == 0:
        return [], []
    spec = [("steps", "Steps / day"), ("distance_km", "Distance on foot / day"), ("exercise_min", "Exercise / day"),
            ("active_kcal", "Active energy / day"), ("workouts_n", "Workouts"), ("workout_min", "Workout hours"),
            ("resting_hr", "Resting heart rate"), ("walking_hr", "Walking heart rate"), ("hrv", "HRV"),
            ("vo2max", "VO₂ max"), ("weight_kg", "Weight"), ("sleep_h", "Measured sleep"),
            ("in_bed_h", "Time in bed"), ("bedtime_h", "Average bedtime"), ("move_ring_pct", "Move ring closed"),
            ("exercise_ring_pct", "Exercise ring closed"), ("stand_ring_pct", "Stand ring closed"),
            ("worn_days", "Days Watch worn"), ("nights", "Nights with sleep data")]
    years = [str(i) for i in y.index]
    rows = []
    for key, label in spec:
        if key not in y or y[key].notna().sum() == 0:
            continue
        vals = y[key]
        cells = []
        better = ALL_METRICS[key].better if key in ALL_METRICS else "neutral"
        best = None
        if better in ("higher", "lower") and vals.notna().sum() >= 2:
            best = vals.idxmax() if better == "higher" else vals.idxmin()
        for yr, v in vals.items():
            if pd.isna(v):
                cells.append({"text": "–", "best": False})
                continue
            if key == "workout_min":
                text = fmt.num(v / 60)
            elif key in ("workouts_n", "worn_days", "nights"):
                text = fmt.num(v)
            elif key.endswith("_ring_pct"):
                text = fmt.pct(v)
            else:
                text = fmt.value(key, v)
            cells.append({"text": text, "best": yr == best})
        rows.append({"label": label, "cells": cells})
    return years, rows


def _types_table(a: Analysis) -> List[list]:
    meta = a.ctx.m.meta
    src_names = dict(zip(a.ctx.m.sources["source_id"].astype(str), a.ctx.m.sources["name"]))
    fmt = a.ctx.fmt
    rows = []
    for short, info in sorted(meta.get("types", {}).items(), key=lambda kv: -kv[1]["count"]):
        first = pd.Timestamp(int(info["first"]), unit="s") if info.get("first") is not None else None
        last = pd.Timestamp(int(info["last"]), unit="s") if info.get("last") is not None else None
        top = sorted(info.get("sources", {}).items(), key=lambda kv: -kv[1])[:3]
        rows.append([info.get("label") or label_for_type(short), f"{info['count']:,}", fmt.date(first),
                     fmt.date(last), ", ".join(src_names.get(s, s) for s, _ in top)])
    return rows


def _stage_split(a: Analysis) -> Optional[List[dict]]:
    n = a.ctx.m.nights
    if n is None or len(n) == 0:
        return None
    st = n[n["kind"] == "stages"]
    if len(st) < 5:
        return None
    parts = [("Deep", st["deep_h"].mean(), 4), ("Core", st["core_h"].mean(), 3), ("REM", st["rem_h"].mean(), 1),
             ("Awake", st["awake_h"].mean(), 0)]
    total = sum(p[1] for p in parts if pd.notna(p[1]))
    out = []
    for name, v, ordi in parts:
        if pd.isna(v) or v <= 0:
            continue
        share = 100 * v / total
        out.append({"name": name, "share": share, "label": f"{share:.0f}%" if share >= 7 else "",
                    "ord": ordi, "hours": a.ctx.fmt.hours(v)})
    return out


def build_report(a: Analysis, demo: bool = False, fragment: bool = False) -> str:
    ch = Charts(a)
    ctx = a.ctx
    fmt = ctx.fmt
    meta = ctx.m.meta

    charts = {
        "steps_daily": ch.daily_trend("steps_daily", "steps", 30, "mean"),
        "steps_by_year": ch.steps_by_year(),
        "exercise_weekly": ch.exercise_weekly(),
        "rings_monthly": ch.rings_monthly(),
        "activity_heatmap": ch.activity_heatmap(),
        "rhr_daily": ch.daily_trend("rhr_daily", "resting_hr", 30, "median", shade_periods=True),
        "hrv_daily": ch.daily_trend("hrv_daily", "hrv", 30, "median"),
        "vo2": ch.vo2(),
        "walking_hr": ch.daily_trend("walking_hr", "walking_hr", 60, "median"),
        "hr_by_hour": ch.hr_by_hour(),
        "sleep_nights": ch.sleep_nights(),
        "sleep_window_monthly": ch.sleep_window_monthly(),
        "sleep_weekday": ch.sleep_weekday(),
        "sleep_stages": ch.sleep_stages(),
        "sleep_coverage": ch.sleep_coverage(),
        "workouts_monthly": ch.workouts_monthly(),
        "running_pace": ch.running_pace(),
        "hr_zones": ch.hr_zones(),
        "seasonality": ch.seasonality(),
        "devices": ch.devices(),
        "availability": ch.availability(),
    }
    small = []
    for key, title in (("steps", "Steps per day"), ("exercise_min", "Exercise minutes per day"),
                       ("resting_hr", "Resting heart rate"), ("hrv", "HRV"), ("vo2max", "VO₂ max"),
                       ("sleep_any_h", "Sleep / time in bed"), ("weight_kg", "Weight"),
                       ("walking_speed_kmh", "Walking speed")):
        cid = ch.monthly_small(key)
        if cid:
            small.append({"id": cid, "title": title, "unit": fmt.unit_label(key)})

    insights = a.insights
    by = lambda dom: [_insight_dict(i) for i in sorted(insights, key=lambda i: -i.score) if i.domain == dom]
    coverage = next((i for i in insights if i.id == "coverage"), None)
    sleep_cov = next((i for i in insights if i.id == "sleep_coverage"), None)
    years_label = f"{ctx.first_day.year}–{ctx.last_day.year}" if ctx.first_day.year != ctx.last_day.year \
        else str(ctx.first_day.year)
    yr_cols, yr_rows = _yearly_rows(a)
    dedup = a.ctx.m.info.get("dedup", {}).get("steps")
    dedup_text = None
    if dedup and dedup.get("raw_total"):
        saved = 100 * (1 - dedup["dedup_total"] / dedup["raw_total"])
        dedup_text = (f"Your iPhone and Watch often counted the same steps: adding every sample up would give "
                      f"{fmt.num(dedup['raw_total'])} steps in total, merging by source priority gives "
                      f"{fmt.num(dedup['dedup_total'])} ({saved:.0f}% were overlapping duplicates).")

    context = {
        "title": "HealthLens report" + (" (demo)" if demo else ""),
        "years_label": years_label,
        "demo": demo,
        "fragment": fragment,
        "csp": CSP,
        "version": __version__,
        "generated": f"{datetime.now().day} {datetime.now().strftime('%b %Y, %H:%M')}",
        "period": f"{fmt.date(ctx.first_day)} – {fmt.date(ctx.last_day)}",
        "coverage": _insight_dict(coverage) if coverage else None,
        "devices_text": "; ".join(coverage.numbers.get("devices", [])) if coverage else "",
        "tiles": [_tile(a, k) for k in a.kpis],
        "key": [_insight_dict(i) for i in a.key_findings(8)],
        "sections": {d: by(d) for d in ("activity", "heart", "body", "sleep", "workouts", "trends",
                                        "relationships")},
        "sleep_cov": _insight_dict(sleep_cov) if sleep_cov else None,
        "charts": charts,
        "small": small,
        "stage_split": _stage_split(a),
        "workout_rows": _workout_table(a),
        "period_rows": _periods_table(a),
        "relationship_rows": _relationship_rows(a),
        "yearly_cols": yr_cols, "yearly_rows": yr_rows,
        "records": a.records,
        "types_rows": _types_table(a),
        "dedup_text": dedup_text,
        "dupes": a.ctx.m.info.get("workout_duplicates_removed", 0),
        "errors": a.errors,
        "meta": meta,
        "units": fmt.units.system,
        "css": _read("base.css"),
        "theme_js": _read("theme.js"),
        "plotly_js": plotly_js(),
        "charts_json": _json_for_script(ch.specs),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(("html", "j2")),
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template("report.html.j2").render(**context)


def write_report(a: Analysis, path: Path, demo: bool = False, fragment: bool = False) -> Path:
    html = build_report(a, demo=demo, fragment=fragment)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
