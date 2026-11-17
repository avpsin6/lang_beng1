"""Chart specifications for the report (plain Plotly JSON + table views).

Colours are tokens (``@series-1``, ``@muted/0.12``, ``%seq``) resolved in the
browser for the current light/dark theme (see static/theme.js). Every chart
also carries a ``table`` — the accessible twin shown by "Show table".
Design rules (from the data-viz method): one y-axis per chart, thin marks,
2 px lines, hairline grid, a legend only for 2+ series, colour by identity.
"""

from __future__ import annotations

import calendar
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..analysis import Analysis
from ..analysis.fmt import Formatter
from ..catalog import ALL_METRICS, workout_label

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPIKES = {"showspikes": True, "spikemode": "across", "spikesnap": "cursor", "spikecolor": "@muted",
          "spikethickness": 1, "spikedash": "solid"}


def _num(x, nd: int = 2):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _list(values, nd: int = 2) -> list:
    return [_num(v, nd) for v in values]


def _dates(idx) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in idx]


class Charts:
    def __init__(self, a: Analysis):
        self.a = a
        self.ctx = a.ctx
        self.fmt: Formatter = a.ctx.fmt
        self.units = self.fmt.units
        self.specs: Dict[str, dict] = {}

    # -------------------------------------------------------------- helpers
    def add(self, cid: str, data: list, layout: dict, table: Optional[dict], height: int = 300) -> str:
        self.specs[cid] = {"data": data, "layout": layout, "table": table, "height": height}
        return cid

    def conv(self, key: str, values):
        m = ALL_METRICS.get(key)
        arr = np.asarray(values, dtype=float)
        if m is None or m.unit in ("h", "clock", "count", "events"):
            return arr
        return np.asarray(self.units.convert(arr, m.unit), dtype=float)

    def unit(self, key: str) -> str:
        return self.fmt.unit_label(key)

    def decimals(self, key: str) -> int:
        m = ALL_METRICS.get(key)
        return m.decimals if m else 1

    def d3fmt(self, key: str) -> str:
        d = self.decimals(key)
        return ",.0f" if d == 0 else f",.{d}f"

    # -------------------------------------------------------------- daily trend (band + rolling line)
    def daily_trend(self, cid: str, key: str, window: int = 30, stat: str = "mean", start=None,
                    ref: Optional[float] = None, ref_label: str = "", shade_periods: bool = False,
                    height: int = 300) -> Optional[str]:
        d = self.ctx.daily
        if key not in d or d[key].notna().sum() < 20:
            return None
        s = d[key].astype(float)
        if key in ("exercise_min",):
            s = s.where(self.ctx.worn)
        if start is not None:
            s = s.loc[start:]
        s_disp = pd.Series(self.conv(key, s.to_numpy()), index=s.index)
        roll = s_disp.rolling(window, min_periods=max(5, window // 3))
        center = roll.median() if stat == "median" else roll.mean()
        lo, hi = roll.quantile(0.25), roll.quantile(0.75)
        x = _dates(s.index)
        nd = max(self.decimals(key), 0)
        unit = self.unit(key)
        u = f" {unit}" if unit else ""
        fmt = self.d3fmt(key)
        data = [
            {"type": "scatter", "x": x, "y": _list(hi, nd + 1), "mode": "lines", "line": {"width": 0},
             "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter", "x": x, "y": _list(lo, nd + 1), "mode": "lines", "line": {"width": 0},
             "fill": "tonexty", "fillcolor": "@series-1/0.10", "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter", "x": x, "y": _list(center, nd + 1), "mode": "lines", "name": f"{window}-day {stat}",
             "line": {"color": "@series-1", "width": 2},
             "hovertemplate": f"%{{y:{fmt}}}{u} ({window}-day {stat})<extra></extra>"},
            {"type": "scatter", "x": x, "y": _list(s_disp, nd + 1), "mode": "markers", "name": "That day",
             "marker": {"size": 6, "opacity": 0, "color": "@series-1"},
             "hovertemplate": f"%{{y:{fmt}}}{u} that day<extra></extra>"},
        ]
        layout = {"hovermode": "x unified", "xaxis": dict(SPIKES, type="date", hoverformat="%-d %b %Y"),
                  "yaxis": {"tickformat": fmt if nd == 0 else None, "ticksuffix": u if len(u) < 6 else ""}}
        shapes, annotations = [], []
        if ref is not None:
            ref_disp = float(self.conv(key, [ref])[0])
            shapes.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": ref_disp, "y1": ref_disp,
                           "line": {"color": "@ink-2", "width": 1}})
            annotations.append({"xref": "paper", "x": 1, "y": ref_disp, "text": ref_label, "showarrow": False,
                                "xanchor": "right", "yanchor": "bottom", "font": {"size": 11, "color": "@ink-2"}})
        if shade_periods and self.a.periods:
            for p in self.a.periods:
                shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": p.start.strftime("%Y-%m-%d"),
                               "x1": (p.end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "y0": 0, "y1": 1,
                               "fillcolor": "@critical/0.12", "line": {"width": 0}, "layer": "below"})
        if shapes:
            layout["shapes"] = shapes
        if annotations:
            layout["annotations"] = annotations
        mon = s_disp.groupby(s_disp.index.values.astype("datetime64[M]")).agg(["mean", "median", "count"])
        rows = [[pd.Timestamp(i).strftime("%b %Y"),
                 self.fmt.num(r["median" if stat == "median" else "mean"], nd), int(r["count"])]
                for i, r in mon.iterrows() if r["count"] > 0]
        label = ALL_METRICS[key].label
        table = {"columns": ["Month", f"{label} ({stat}{', ' + unit if unit else ''})", "Days with data"],
                 "rows": rows[::-1]}
        return self.add(cid, data, layout, table, height)

    # -------------------------------------------------------------- activity
    def steps_by_year(self) -> Optional[str]:
        y = self.a.yearly
        if y is None or len(y) == 0 or "steps" not in y or y["steps"].notna().sum() < 1:
            return None
        y = y[y["steps"].notna()]
        years = [str(i) for i in y.index]
        data = [{"type": "bar", "x": years, "y": _list(y["steps"], 0), "marker": {"color": "@series-1"},
                 "text": [self.fmt.num(v) for v in y["steps"]], "textposition": "outside", "cliponaxis": False,
                 "textfont": {"color": "@ink-2", "size": 11},
                 "hovertemplate": "%{x}: %{y:,.0f} steps a day<extra></extra>"}]
        layout = {"xaxis": {"type": "category"}, "yaxis": {"tickformat": ",.0f", "rangemode": "tozero"},
                  "bargap": 0.45}
        table = {"columns": ["Year", "Average steps per day", "Days with data"],
                 "rows": [[i, self.fmt.num(r["steps"]), int(r.get("n_steps", 0) or 0)] for i, r in y.iterrows()]}
        return self.add("steps_by_year", data, layout, table, 280)

    def exercise_weekly(self) -> Optional[str]:
        wk = self.ctx.weekly
        if "exercise_min" not in wk:
            return None
        wk = wk[wk["worn_days"] >= 5].tail(104)
        if len(wk) < 8:
            return None
        met = wk["exercise_min"] >= 150
        x = _dates(wk.index)
        data = [
            {"type": "bar", "x": [d for d, m in zip(x, met) if m], "y": _list(wk.loc[met, "exercise_min"], 0),
             "name": "150 min or more", "marker": {"color": "@series-1"},
             "hovertemplate": "Week of %{x|%-d %b %Y}: %{y:,.0f} min<extra></extra>"},
            {"type": "bar", "x": [d for d, m in zip(x, met) if not m], "y": _list(wk.loc[~met, "exercise_min"], 0),
             "name": "Below 150 min", "marker": {"color": "@muted"},
             "hovertemplate": "Week of %{x|%-d %b %Y}: %{y:,.0f} min<extra></extra>"},
        ]
        layout = {"showlegend": True, "barmode": "overlay", "bargap": 0.25, "xaxis": {"type": "date"},
                  "yaxis": {"rangemode": "tozero", "ticksuffix": " min"},
                  "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 150, "y1": 150,
                              "line": {"color": "@ink-2", "width": 1}}],
                  "annotations": [{"xref": "paper", "x": 0, "y": 150, "text": "150 min (WHO minimum)",
                                   "showarrow": False, "xanchor": "left", "yanchor": "bottom",
                                   "bgcolor": "@surface",
                                   "font": {"size": 11, "color": "@ink-2"}}]}
        table = {"columns": ["Week starting", "Exercise minutes", "Days worn"],
                 "rows": [[self.fmt.date(i), self.fmt.num(r["exercise_min"]), int(r["worn_days"])]
                          for i, r in wk.iloc[::-1].iterrows()]}
        return self.add("exercise_weekly", data, layout, table, 280)

    def rings_monthly(self) -> Optional[str]:
        d = self.ctx.daily
        cols = [("move_ring_pct", "Move"), ("exercise_ring_pct", "Exercise"), ("stand_ring_pct", "Stand")]
        if any(c not in d for c, _ in cols):
            return None
        keys = d.index.values.astype("datetime64[M]")
        data, rows = [], {}
        for i, (c, name) in enumerate(cols):
            closed = (d[c] >= 100).where(d[c].notna())
            g = closed.groupby(keys)
            share = (g.mean() * 100).where(g.count() >= 10).dropna()
            if len(share) < 3:
                return None
            data.append({"type": "scatter", "mode": "lines", "x": _dates(share.index), "y": _list(share, 1),
                         "name": name, "line": {"color": f"@series-{i + 1}", "width": 2},
                         "hovertemplate": f"{name}: %{{y:.0f}}%<extra></extra>"})
            for k, v in share.items():
                rows.setdefault(k, {})[name] = v
        layout = {"showlegend": True, "hovermode": "x unified", "xaxis": dict(SPIKES, type="date",
                                                                              hoverformat="%b %Y"),
                  "yaxis": {"range": [0, 104], "ticksuffix": "%"}}
        table = {"columns": ["Month", "Move closed", "Exercise closed", "Stand closed"],
                 "rows": [[pd.Timestamp(k).strftime("%b %Y")] + [self.fmt.pct(v.get(n)) if n in v else None
                                                                  for n in ("Move", "Exercise", "Stand")]
                          for k, v in sorted(rows.items(), reverse=True)]}
        return self.add("rings_monthly", data, layout, table, 280)

    def activity_heatmap(self) -> Optional[str]:
        h = self.ctx.m.hourly
        if h is None or len(h) == 0 or h["steps"].notna().sum() < 500:
            return None
        h = h[h["date"] >= self.ctx.last_day - pd.Timedelta(days=730)]
        days = h.groupby(h["date"].dt.day_of_week)["date"].nunique()
        tot = h.groupby([h["date"].dt.day_of_week, "hour"])["steps"].sum().unstack(fill_value=0)
        tot = tot.reindex(index=range(7), columns=range(24), fill_value=0)
        z = tot.div(days.reindex(range(7)).replace(0, np.nan), axis=0)
        hours = [self.fmt.hour_label(i) for i in range(24)]
        data = [{"type": "heatmap", "z": [_list(r, 0) for r in z.to_numpy()], "x": hours, "y": WEEKDAYS,
                 "colorscale": "%seq", "xgap": 2, "ygap": 2, "zmin": 0,
                 "colorbar": {"thickness": 8, "outlinewidth": 0, "tickfont": {"color": "@muted", "size": 10},
                              "tickformat": ",.0f", "len": 0.9},
                 "hovertemplate": "%{y} %{x}: %{z:,.0f} steps on average<extra></extra>"}]
        layout = {"yaxis": {"autorange": "reversed", "showgrid": False, "type": "category"},
                  "xaxis": {"showgrid": False, "type": "category", "tickangle": 0, "nticks": 8}}
        table = {"columns": ["Day"] + hours,
                 "rows": [[WEEKDAYS[i]] + [self.fmt.num(v) for v in z.iloc[i]] for i in range(7)]}
        return self.add("activity_heatmap", data, layout, table, 260)

    # -------------------------------------------------------------- heart
    def vo2(self) -> Optional[str]:
        d = self.ctx.daily
        if "vo2max" not in d or d["vo2max"].notna().sum() < 5:
            return None
        s = d["vo2max"].dropna()
        roll = s.rolling("120D", min_periods=3).median()
        data = [
            {"type": "scatter", "mode": "markers", "x": _dates(s.index), "y": _list(s, 1), "name": "Estimate",
             "marker": {"color": "@series-1/0.55", "size": 7, "line": {"color": "@surface", "width": 1}},
             "hovertemplate": "%{x|%-d %b %Y}: %{y:.1f}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "x": _dates(roll.index), "y": _list(roll, 2),
             "name": "Typical (4-month median)", "line": {"color": "@series-1", "width": 2},
             "hovertemplate": "Typical: %{y:.1f}<extra></extra>"},
        ]
        layout = {"showlegend": True, "xaxis": {"type": "date"}, "yaxis": {"ticksuffix": ""}}
        prof = self.ctx.m.profile
        from ..analysis.insights import friend_median
        ref = friend_median(prof.age, prof.sex)
        if ref is not None:
            layout["shapes"] = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": ref, "y1": ref,
                                 "line": {"color": "@ink-2", "width": 1}}]
            layout["annotations"] = [{"xref": "paper", "x": 0, "y": ref, "xanchor": "left", "yanchor": "bottom",
                                      "showarrow": False, "font": {"size": 11, "color": "@ink-2"},
                                      "text": f"≈ median for your age & sex ({ref:.0f})"}]
        table = {"columns": ["Date", "VO₂ max (mL/kg/min)"],
                 "rows": [[self.fmt.date(i), self.fmt.num(v, 1)] for i, v in s.iloc[::-1].items()]}
        return self.add("vo2", data, layout, table, 300)

    def hr_by_hour(self) -> Optional[str]:
        h = self.ctx.m.hourly
        if h is None or len(h) == 0 or h["hr"].notna().sum() < 500:
            return None
        h = h[(h["date"] >= self.ctx.last_day - pd.Timedelta(days=730)) & h["hr"].notna()]
        weekend = h["date"].dt.day_of_week >= 5
        hours = [self.fmt.hour_label(i) for i in range(24)]
        data, rows = [], []
        series = {}
        for i, (mask, name) in enumerate(((~weekend, "Weekdays"), (weekend, "Weekends"))):
            prof = h[mask].groupby("hour")["hr"].mean().reindex(range(24))
            series[name] = prof
            data.append({"type": "scatter", "mode": "lines", "x": hours, "y": _list(prof, 1), "name": name,
                         "line": {"color": f"@series-{i + 1}", "width": 2, "shape": "spline", "smoothing": 0.6},
                         "connectgaps": False, "hovertemplate": f"{name}: %{{y:.0f}} bpm<extra></extra>"})
        layout = {"showlegend": True, "hovermode": "x unified", "xaxis": dict(SPIKES, type="category", nticks=8),
                  "yaxis": {"ticksuffix": " bpm"}}
        table = {"columns": ["Hour", "Weekdays (bpm)", "Weekends (bpm)"],
                 "rows": [[hours[i], self.fmt.num(series["Weekdays"].iloc[i]),
                           self.fmt.num(series["Weekends"].iloc[i])] for i in range(24)]}
        return self.add("hr_by_hour", data, layout, table, 280)

    # -------------------------------------------------------------- sleep
    def sleep_nights(self) -> Optional[str]:
        n = self.ctx.m.nights
        if n is None or len(n) < 5:
            return None
        n = n.copy()
        n["wake_date"] = pd.to_datetime(n["wake_date"])
        n["value"] = n["sleep_h"].fillna(n["in_bed_h"])
        measured = n[n["kind"] != "inbed"]
        inbed = n[n["kind"] == "inbed"]
        allv = n.set_index("wake_date")["value"].sort_index()
        roll = allv.rolling("30D", min_periods=5).mean()
        data = []
        if len(inbed):
            data.append({"type": "scatter", "mode": "markers", "x": _dates(inbed["wake_date"]),
                         "y": _list(inbed["value"], 2), "name": "Time in bed (iPhone)",
                         "marker": {"color": "@series-2/0.7", "size": 6, "line": {"color": "@surface", "width": 1}},
                         "hovertemplate": "%{x|%-d %b %Y}: %{y:.1f} h in bed<extra></extra>"})
        if len(measured):
            data.append({"type": "scatter", "mode": "markers", "x": _dates(measured["wake_date"]),
                         "y": _list(measured["value"], 2), "name": "Measured sleep",
                         "marker": {"color": "@series-1", "size": 7, "line": {"color": "@surface", "width": 1}},
                         "hovertemplate": "%{x|%-d %b %Y}: %{y:.1f} h asleep<extra></extra>"})
        data.append({"type": "scatter", "mode": "lines", "x": _dates(roll.index), "y": _list(roll, 2),
                     "name": "30-day average", "line": {"color": "@muted", "width": 2},
                     "hovertemplate": "30-day average: %{y:.1f} h<extra></extra>"})
        layout = {"showlegend": True, "xaxis": {"type": "date"}, "yaxis": {"ticksuffix": " h"},
                  "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 7, "y1": 7,
                              "line": {"color": "@ink-2", "width": 1}}],
                  "annotations": [{"xref": "paper", "x": 1, "y": 7, "text": "7 h", "showarrow": False,
                                   "xanchor": "right", "yanchor": "bottom", "font": {"size": 11, "color": "@ink-2"}}]}
        rows = [[self.fmt.date(r["wake_date"]), {"stages": "Watch (stages)", "asleep": "Watch / app",
                                                 "inbed": "iPhone (in bed)"}[r["kind"]],
                 self.fmt.hours(r["sleep_h"]) if pd.notna(r["sleep_h"]) else None,
                 self.fmt.hours(r["in_bed_h"]) if pd.notna(r["in_bed_h"]) else None,
                 self.fmt.clock(r["bedtime_h"]), self.fmt.clock(r["wake_h"])]
                for _, r in n.sort_values("wake_date", ascending=False).iterrows()]
        table = {"columns": ["Morning of", "Source", "Asleep", "In bed", "Bedtime", "Woke"], "rows": rows}
        return self.add("sleep_nights", data, layout, table, 320)

    def _clock_axis(self, lo: float, hi: float) -> dict:
        start = int(math.floor(lo))
        end = int(math.ceil(hi))
        step = 1 if end - start <= 8 else 2
        vals = list(range(start, end + 1, step))
        return {"autorange": "reversed", "tickvals": vals, "ticktext": [self.fmt.clock(v) for v in vals],
                "range": [end + 0.25, start - 0.25]}

    def sleep_window_monthly(self) -> Optional[str]:
        n = self.ctx.m.nights
        if n is None or len(n) < 20:
            return None
        n = n.copy()
        n["month"] = pd.to_datetime(n["wake_date"]).values.astype("datetime64[M]")
        g = n.groupby("month").agg(bed=("bedtime_h", "median"), wake=("wake_h", "median"), n=("wake_h", "size"))
        g = g[g["n"] >= 5]
        if len(g) < 3:
            return None
        up = g["wake"] + 24
        hover = [f"{pd.Timestamp(i).strftime('%b %Y')}: bed {self.fmt.clock(b)} → up {self.fmt.clock(w)} "
                 f"({int(c)} nights)" for i, b, w, c in zip(g.index, g["bed"], g["wake"], g["n"])]
        data = [{"type": "bar", "x": _dates(g.index), "base": _list(g["bed"], 3), "y": _list(up - g["bed"], 3),
                 "marker": {"color": "@series-1"}, "customdata": hover,
                 "hovertemplate": "%{customdata}<extra></extra>"}]
        layout = {"xaxis": {"type": "date"}, "yaxis": self._clock_axis(g["bed"].min(), up.max()), "bargap": 0.3}
        table = {"columns": ["Month", "Median bedtime", "Median wake-up", "Nights"],
                 "rows": [[pd.Timestamp(i).strftime("%b %Y"), self.fmt.clock(r["bed"]), self.fmt.clock(r["wake"]),
                           int(r["n"])] for i, r in g.iloc[::-1].iterrows()]}
        return self.add("sleep_window_monthly", data, layout, table, 300)

    def sleep_weekday(self) -> Optional[str]:
        n = self.ctx.m.nights
        if n is None or len(n) < 21:
            return None
        n = n.copy()
        n["wd"] = pd.to_datetime(n["wake_date"]).dt.day_of_week
        g = n.groupby("wd").agg(bed=("bedtime_h", "median"), wake=("wake_h", "median"), n=("wake_h", "size"))
        g = g.reindex(range(7)).dropna()
        if len(g) < 7:
            return None
        labels = [f"{WEEKDAYS[(i - 1) % 7]}→{WEEKDAYS[i]}" for i in g.index]
        up = g["wake"] + 24
        hover = [f"Night {lab}: bed {self.fmt.clock(b)} → up {self.fmt.clock(w)}" for lab, b, w in
                 zip(labels, g["bed"], g["wake"])]
        data = [{"type": "bar", "x": labels, "base": _list(g["bed"], 3), "y": _list(up - g["bed"], 3),
                 "marker": {"color": "@series-1"}, "customdata": hover, "hovertemplate": "%{customdata}<extra></extra>"}]
        layout = {"xaxis": {"type": "category"}, "yaxis": self._clock_axis(g["bed"].min(), up.max()), "bargap": 0.45}
        table = {"columns": ["Night", "Median bedtime", "Median wake-up", "Nights"],
                 "rows": [[lab, self.fmt.clock(r["bed"]), self.fmt.clock(r["wake"]), int(r["n"])]
                          for lab, (_, r) in zip(labels, g.iterrows())]}
        return self.add("sleep_weekday", data, layout, table, 280)

    STAGES = [("deep_h", "Deep", "@ord-4"), ("core_h", "Core", "@ord-3"), ("rem_h", "REM", "@ord-1"),
              ("awake_h", "Awake", "@ord-0")]

    def sleep_stages(self) -> Optional[str]:
        n = self.ctx.m.nights
        if n is None:
            return None
        st = n[n["kind"] == "stages"].sort_values("wake_date").tail(45)
        if len(st) < 5:
            return None
        x = [self.fmt.date(d) for d in st["wake_date"]]     # one slot per tracked night, gaps removed
        data = []
        for col, name, color in self.STAGES:
            data.append({"type": "bar", "x": x, "y": _list(st[col], 3), "name": name,
                         "marker": {"color": color, "line": {"color": "@surface", "width": 1}},
                         "hovertemplate": f"{name}: %{{y:.2f}} h<extra></extra>"})
        layout = {"showlegend": True, "barmode": "stack", "bargap": 0.2, "hovermode": "x unified",
                  "xaxis": {"type": "category", "nticks": 6, "tickangle": 0}, "yaxis": {"ticksuffix": " h"},
                  "legend": {"traceorder": "normal"}}
        table = {"columns": ["Morning of", "Deep", "Core", "REM", "Awake"],
                 "rows": [[self.fmt.date(r["wake_date"])] + [self.fmt.hours(r[c]) for c, _, _ in self.STAGES]
                          for _, r in st.iloc[::-1].iterrows()]}
        return self.add("sleep_stages", data, layout, table, 300)

    def sleep_coverage(self) -> Optional[str]:
        n = self.ctx.m.nights
        if n is None or len(n) == 0:
            return None
        n = n.copy()
        n["year"] = pd.to_datetime(n["wake_date"]).dt.year
        c = n.groupby(["year", "kind"]).size().unstack(fill_value=0)
        years = sorted(set(self.ctx.daily.index.year))
        c = c.reindex(years, fill_value=0)
        data = []
        names = [("stages", "Watch with stages"), ("asleep", "Watch / app (asleep)"), ("inbed", "iPhone time in bed")]
        for i, (k, name) in enumerate(names):
            if k in c:
                data.append({"type": "bar", "x": [str(y) for y in c.index], "y": c[k].tolist(), "name": name,
                             "marker": {"color": f"@series-{i + 1}", "line": {"color": "@surface", "width": 1}},
                             "hovertemplate": f"{name}: %{{y}} nights<extra></extra>"})
        layout = {"showlegend": True, "barmode": "stack", "bargap": 0.45, "hovermode": "x unified",
                  "xaxis": {"type": "category"}, "yaxis": {"rangemode": "tozero"}}
        table = {"columns": ["Year"] + [nm for k, nm in names if k in c] + ["Nights in year"],
                 "rows": [[str(y)] + [int(c.loc[y, k]) for k, nm in names if k in c] +
                          [366 if calendar.isleap(int(y)) else 365]
                          for y in c.index]}
        return self.add("sleep_coverage", data, layout, table, 260)

    # -------------------------------------------------------------- workouts
    def workouts_monthly(self) -> Optional[str]:
        w = self.ctx.m.workouts
        if w is None or len(w) < 5:
            return None
        w = w.copy()
        top = w.groupby("label")["duration_min"].sum().sort_values(ascending=False)
        keep = list(top.index[:5])
        w["group"] = np.where(w["label"].isin(keep), w["label"], "Other")
        w["month"] = w["date"].values.astype("datetime64[M]")
        g = w.groupby(["month", "group"])["duration_min"].sum().unstack(fill_value=0) / 60
        full = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(full, fill_value=0)
        data = []
        groups = keep + (["Other"] if "Other" in g else [])
        for i, name in enumerate(groups):
            color = "@muted" if name == "Other" else f"@series-{i + 1}"
            data.append({"type": "bar", "x": _dates(g.index), "y": _list(g[name], 2), "name": name,
                         "marker": {"color": color, "line": {"color": "@surface", "width": 1}},
                         "hovertemplate": f"{name}: %{{y:.1f}} h<extra></extra>"})
        layout = {"showlegend": True, "barmode": "stack", "bargap": 0.15, "hovermode": "x unified",
                  "xaxis": {"type": "date", "hoverformat": "%b %Y"}, "yaxis": {"ticksuffix": " h"}}
        table = {"columns": ["Month"] + groups + ["Total"],
                 "rows": [[pd.Timestamp(i).strftime("%b %Y")] + [self.fmt.num(r[c], 1) for c in groups] +
                          [self.fmt.num(r[groups].sum(), 1)] for i, r in g.iloc[::-1].iterrows()]}
        return self.add("workouts_monthly", data, layout, table, 300)

    def running_pace(self) -> Optional[str]:
        w = self.ctx.m.workouts
        if w is None or len(w) == 0:
            return None
        runs = w[(w["activity"] == "Running") & (w["distance_km"] >= 2) & w["pace_min_per_km"].between(2.5, 12)]
        if len(runs) < 8:
            return None
        imperial = self.units.system == "imperial"
        factor = 1.609344 if imperial else 1.0
        pace = runs.set_index("date")["pace_min_per_km"].sort_index() * factor
        roll = pace.rolling(15, min_periods=5, center=True).median()
        lo, hi = float(pace.min()), float(pace.max())
        step = 0.25 if hi - lo <= 1.5 else (0.5 if hi - lo <= 4 else 1.0)
        vals = [float(v) for v in np.arange(math.floor(lo / step) * step, math.ceil(hi / step) * step + 0.01, step)]
        ticktext = [self.fmt.pace(v / factor).split(" ")[0] for v in vals]
        label = "/mi" if imperial else "/km"
        custom = [f"{self.fmt.date(d)}: {self.fmt.pace(p / factor)} over {self.fmt.distance(dist, 2)}"
                  for d, p, dist in zip(pace.index, pace, runs.sort_values("date")["distance_km"])]
        data = [
            {"type": "scatter", "mode": "markers", "x": _dates(pace.index), "y": _list(pace, 3), "name": "Run",
             "marker": {"color": "@series-1/0.55", "size": 7, "line": {"color": "@surface", "width": 1}},
             "customdata": custom, "hovertemplate": "%{customdata}<extra></extra>"},
            {"type": "scatter", "mode": "lines", "x": _dates(roll.index), "y": _list(roll, 3),
             "name": "Typical (15-run median)", "line": {"color": "@series-1", "width": 2}, "hoverinfo": "skip"},
        ]
        layout = {"showlegend": True, "xaxis": {"type": "date"},
                  "yaxis": {"range": [vals[-1] + step / 4, vals[0] - step / 4], "tickvals": vals,
                            "ticktext": ticktext, "title": {"text": f"pace {label} (faster ↑)"}}}
        table = {"columns": ["Date", "Distance", "Duration", "Pace", "Avg HR"],
                 "rows": [[self.fmt.date(r["date"]), self.fmt.distance(r["distance_km"], 2),
                           self.fmt.minutes(r["duration_min"]), self.fmt.pace(r["pace_min_per_km"]),
                           self.fmt.num(r["avg_hr"]) if pd.notna(r["avg_hr"]) else None]
                          for _, r in runs.sort_values("date", ascending=False).iterrows()]}
        return self.add("running_pace", data, layout, table, 300)

    def hr_zones(self) -> Optional[str]:
        w = self.ctx.m.workouts
        cols = [f"z{i}_min" for i in range(1, 6)]
        if w is None or len(w) == 0 or any(c not in w for c in cols) or w[cols].notna().sum().sum() == 0:
            return None
        g = w.groupby(w["date"].dt.year)[cols].sum()
        g = g[g.sum(axis=1) > 60]
        if len(g) < 1:
            return None
        share = g.div(g.sum(axis=1), axis=0) * 100
        names = ["Zone 1 (<60% max)", "Zone 2 (60–70%)", "Zone 3 (70–80%)", "Zone 4 (80–90%)", "Zone 5 (>90%)"]
        data = []
        for i, (c, name) in enumerate(zip(cols, names)):
            data.append({"type": "bar", "x": [str(y) for y in share.index], "y": _list(share[c], 1), "name": name,
                         "marker": {"color": f"@ord-{i}", "line": {"color": "@surface", "width": 1}},
                         "hovertemplate": f"{name}: %{{y:.0f}}% of workout time<extra></extra>"})
        layout = {"showlegend": True, "barmode": "stack", "bargap": 0.45, "hovermode": "x unified",
                  "xaxis": {"type": "category"}, "yaxis": {"range": [0, 100], "ticksuffix": "%"}}
        table = {"columns": ["Year"] + names + ["Hours with HR"],
                 "rows": [[str(y)] + [self.fmt.pct(share.loc[y, c]) for c in cols] +
                          [self.fmt.num(g.loc[y].sum() / 60, 1)] for y in share.index]}
        return self.add("hr_zones", data, layout, table, 280)

    # -------------------------------------------------------------- long term
    def monthly_small(self, key: str) -> Optional[str]:
        mon = self.ctx.monthly
        if key not in mon or mon[key].notna().sum() < 6:
            return None
        s = mon[key].dropna()
        vals = self.conv(key, s.to_numpy())
        nd = self.decimals(key)
        unit = self.unit(key)
        u = f" {unit}" if unit else ""
        fmt = self.d3fmt(key)
        if ALL_METRICS[key].unit == "h":
            hover = [f"{pd.Timestamp(i).strftime('%b %Y')}: {self.fmt.hours(v)}" for i, v in zip(s.index, vals)]
        else:
            hover = [f"{pd.Timestamp(i).strftime('%b %Y')}: {self.fmt.value(key, v0)}" for i, v0 in s.items()]
        data = [{"type": "scatter", "mode": "lines", "x": _dates(s.index), "y": _list(vals, nd + 1),
                 "line": {"color": "@series-1", "width": 2}, "fill": "tozeroy" if key in ("steps", "exercise_min")
                 else "none", "fillcolor": "@series-1/0.08", "customdata": hover,
                 "hovertemplate": "%{customdata}<extra></extra>"}]
        layout = {"xaxis": {"type": "date"}, "yaxis": {"tickformat": fmt if nd == 0 else None,
                                                       "rangemode": "tozero" if key in ("steps", "exercise_min")
                                                       else "normal"},
                  "margin": {"l": 4, "r": 8, "t": 4, "b": 4}}
        chapters = self.a.chapters.get(key) or []
        if len(chapters) > 1:
            layout["shapes"] = [{"type": "line", "xref": "x", "yref": "paper",
                                 "x0": c.start.strftime("%Y-%m-%d"), "x1": c.start.strftime("%Y-%m-%d"),
                                 "y0": 0, "y1": 1, "line": {"color": "@axis", "width": 1}} for c in chapters[1:]]
        table = {"columns": ["Month", f"{ALL_METRICS[key].label}{' (' + unit + ')' if unit else ''}"],
                 "rows": [[pd.Timestamp(i).strftime("%b %Y"), self.fmt.num(v, nd)]
                          for i, v in zip(s.index[::-1], vals[::-1])]}
        return self.add(f"monthly_{key}", data, layout, table, 200)

    def seasonality(self) -> Optional[str]:
        ins = next((i for i in self.a.insights if i.id == "seasonality"), None)
        if ins is None:
            return None
        prof = ins.numbers["profile"]
        vals = [(prof.get(m, np.nan) - 1) * 100 for m in range(1, 13)]
        colors = ["@div-pos" if v >= 0 else "@div-neg" for v in vals]
        data = [{"type": "bar", "x": MONTHS, "y": _list(vals, 1), "marker": {"color": colors},
                 "hovertemplate": "%{x}: %{y:+.0f}% vs your yearly average<extra></extra>"}]
        layout = {"xaxis": {"type": "category"}, "yaxis": {"ticksuffix": "%", "zeroline": True,
                                                           "zerolinecolor": "@axis", "zerolinewidth": 1},
                  "bargap": 0.4}
        table = {"columns": ["Month", "Steps vs yearly average"],
                 "rows": [[m, self.fmt.pct(v, 0, signed=True)] for m, v in zip(MONTHS, vals)]}
        return self.add("seasonality", data, layout, table, 260)

    # -------------------------------------------------------------- data
    def devices(self) -> Optional[str]:
        src = self.ctx.m.sources
        if src is None or len(src) == 0 or "first" not in src:
            return None
        s = src[(src["records"] >= 50) & src["first"].notna()].copy()
        if len(s) == 0:
            return None
        s["label"] = np.where(s["friendly"].astype(str).str.len() > 0, s["friendly"], s["name"])
        s.loc[s["kind"] == "app", "label"] = s.loc[s["kind"] == "app", "name"]
        s = s.sort_values("first")
        data = []
        for i, (kind, name) in enumerate((("watch", "Apple Watch"), ("phone", "iPhone"), ("app", "Apps"),
                                          ("manual", "Manual entries"))):
            sub = s[s["kind"] == kind]
            if len(sub) == 0:
                continue
            starts = pd.to_datetime(sub["first"].astype(np.int64), unit="s")
            ends = pd.to_datetime(sub["last"].astype(np.int64), unit="s")
            dur = ((ends - starts).dt.total_seconds() * 1000).clip(lower=86400000).tolist()
            hover = [f"{lab}: {self.fmt.date(a)} – {self.fmt.date(b)} · {int(r):,} records"
                     for lab, a, b, r in zip(sub["label"], starts, ends, sub["records"])]
            data.append({"type": "bar", "orientation": "h", "y": sub["label"].tolist(), "x": dur,
                         "base": [d.strftime("%Y-%m-%d") for d in starts], "name": name,
                         "marker": {"color": "@muted" if kind == "manual" else f"@series-{i + 1}"},
                         "customdata": hover, "hovertemplate": "%{customdata}<extra></extra>"})
        layout = {"showlegend": len(data) > 1, "xaxis": {"type": "date"},
                  "yaxis": {"type": "category", "autorange": "reversed", "automargin": True}, "bargap": 0.35}
        rows = [[r["label"], {"watch": "Apple Watch", "phone": "iPhone", "app": "App", "manual": "Manual"}
                 .get(r["kind"], r["kind"]), self.fmt.date(pd.Timestamp(int(r["first"]), unit="s")),
                 self.fmt.date(pd.Timestamp(int(r["last"]), unit="s")), f"{int(r['records']):,}",
                 r["versions"] or ""] for _, r in s.iterrows()]
        table = {"columns": ["Source", "Kind", "First", "Last", "Records", "Software versions"], "rows": rows}
        return self.add("devices", data, layout, table, max(160, 46 * len(s) + 60))

    def availability(self) -> Optional[str]:
        d = self.ctx.daily
        rows_def = [("steps", "Steps"), ("hr_mean", "Heart rate"), ("resting_hr", "Resting HR"), ("hrv", "HRV"),
                    ("vo2max", "VO₂ max"), ("exercise_min", "Exercise"), ("sleep_any_h", "Sleep / in bed"),
                    ("sleep_h", "Measured sleep"), ("workouts_n", "Workouts"), ("weight_kg", "Weight"),
                    ("walking_speed_kmh", "Walking speed"), ("env_audio_db", "Noise")]
        rows_def = [(k, l) for k, l in rows_def if k in d]
        if not rows_def:
            return None
        years = sorted(set(d.index.year))
        z, text = [], []
        for k, _ in rows_def:
            series = d[k] if k != "workouts_n" else d[k].where(d[k] > 0)
            g = series.notna().groupby(d.index.year)
            share = (g.sum() / g.size() * 100).reindex(years)
            z.append(_list(share, 0))
            text.append([f"{int(g.sum().get(y, 0))} days" for y in years])
        data = [{"type": "heatmap", "z": z, "x": [str(y) for y in years], "y": [l for _, l in rows_def],
                 "colorscale": "%seq", "zmin": 0, "zmax": 100, "xgap": 2, "ygap": 2, "customdata": text,
                 "colorbar": {"thickness": 8, "outlinewidth": 0, "ticksuffix": "%",
                              "tickfont": {"color": "@muted", "size": 10}},
                 "hovertemplate": "%{y}, %{x}: %{z:.0f}% of days (%{customdata})<extra></extra>"}]
        layout = {"yaxis": {"autorange": "reversed", "type": "category", "showgrid": False},
                  "xaxis": {"type": "category", "showgrid": False}}
        table = {"columns": ["Data"] + [str(y) for y in years],
                 "rows": [[lab] + [f"{v:.0f}%" if v is not None else "–" for v in zrow]
                          for (k, lab), zrow in zip(rows_def, z)]}
        return self.add("availability", data, layout, table, max(220, 26 * len(rows_def) + 70))
