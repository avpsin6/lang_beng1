"""Build the dashboard payload (daily metrics, nights, workouts) and page."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..analysis import Analysis
from ..analysis.core import MEDIAN, TOTALS, numeric_columns
from ..catalog import ALL_METRICS, DOMAIN_LABELS
from ..report.builder import CSP, STATIC, TEMPLATES, _json_for_script, _read, plotly_js

SKIP = {"move_goal_kcal", "exercise_goal_min", "stand_goal_h", "hr_min", "hr_max"}
DOMAIN_ORDER = ["activity", "heart", "sleep", "workouts", "body", "respiratory", "mobility", "hearing", "mind",
                "data"]
DEFAULTS = ["steps", "resting_hr", "sleep_any_h", "exercise_min", "hrv"]


def _vals(arr, nd: int) -> list:
    out = []
    for v in arr:
        f = float(v) if v is not None else float("nan")
        out.append(round(f, nd) if math.isfinite(f) else None)
    return out


def payload(a: Analysis) -> dict:
    ctx = a.ctx
    fmt = ctx.fmt
    d = ctx.daily
    metrics: List[dict] = []
    for key in numeric_columns(d):
        m = ALL_METRICS[key]
        if key in SKIP or d[key].notna().sum() < 5:
            continue
        s = d[key].astype(float)
        if key == "exercise_min":
            s = s.where(ctx.worn)
        if m.unit in ("h", "clock"):
            kind, vals, unit, nd = ("hours" if m.unit == "h" else "clock"), s.to_numpy(), "", 3
        else:
            kind, unit, nd = "number", fmt.unit_label(key), m.decimals + 1
            vals = np.asarray(fmt.units.convert(s.to_numpy(), m.unit), dtype=float)
        agg = "total" if key in TOTALS else ("median" if key in MEDIAN else "mean")
        metrics.append({"key": key, "label": m.label, "domain": m.domain,
                        "domainLabel": DOMAIN_LABELS.get(m.domain, m.domain.title()), "unit": unit, "kind": kind,
                        "decimals": m.decimals, "agg": agg, "better": m.better, "description": m.description,
                        "values": _vals(vals, nd)})
    metrics.sort(key=lambda x: (DOMAIN_ORDER.index(x["domain"]) if x["domain"] in DOMAIN_ORDER else 99))

    nights = []
    n = ctx.m.nights
    if n is not None and len(n):
        for _, r in n.sort_values("wake_date").iterrows():
            nights.append({"date": pd.Timestamp(r["wake_date"]).strftime("%Y-%m-%d"), "kind": r["kind"],
                           "bed": _vals([r["bedtime_h"]], 3)[0], "wake": _vals([r["wake_h"]], 3)[0],
                           "sleep": _vals([r["sleep_h"]], 2)[0], "inbed": _vals([r["in_bed_h"]], 2)[0]})
    workouts = []
    w = ctx.m.workouts
    if w is not None and len(w):
        for _, r in w.sort_values("start").iterrows():
            dist = r.get("distance_km")
            workouts.append({
                "date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), "type": r["label"],
                "min": _vals([r["duration_min"]], 1)[0],
                "dist": _vals([fmt.units.convert(dist, "km") if pd.notna(dist) else None], 2)[0],
                "kcal": _vals([r.get("energy_kcal")], 0)[0], "hr": _vals([r.get("avg_hr")], 0)[0],
                "pace": fmt.pace(r.get("pace_min_per_km")) if pd.notna(r.get("pace_min_per_km")) else None})
    keys = {m["key"] for m in metrics}
    return {"dates": [x.strftime("%Y-%m-%d") for x in d.index], "metrics": metrics, "nights": nights,
            "workouts": workouts, "distUnit": fmt.units.label("km"), "clock24": fmt.clock24,
            "defaults": [k for k in DEFAULTS if k in keys] or [m["key"] for m in metrics[:1]]}


def build_dashboard(a: Analysis, demo: bool = False, fragment: bool = False) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(("html", "j2")),
                      trim_blocks=True, lstrip_blocks=True)
    now = datetime.now()
    return env.get_template("dashboard.html.j2").render(
        title="HealthLens dashboard" + (" (demo)" if demo else ""), demo=demo, fragment=fragment, csp=CSP,
        version=__version__, generated=f"{now.day} {now.strftime('%b %Y, %H:%M')}",
        period=f"{a.ctx.fmt.date(a.ctx.first_day)} – {a.ctx.fmt.date(a.ctx.last_day)}",
        css=_read("base.css"), theme_js=_read("theme.js"), dashboard_js=_read("dashboard.js"),
        plotly_js=plotly_js(), data_json=_json_for_script(payload(a)))


def write_dashboard(a: Analysis, path: Path, demo: bool = False, fragment: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard(a, demo=demo, fragment=fragment), encoding="utf-8")
    return path
