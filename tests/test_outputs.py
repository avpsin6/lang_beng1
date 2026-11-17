from __future__ import annotations

import json
import re

import numpy as np
import pytest

from healthlens import cli, units
from healthlens.analysis import analyse
from healthlens.dashboard import build_dashboard
from healthlens.dashboard.builder import payload
from healthlens.metrics import compute
from healthlens.report import build_report

EXTERNAL = re.compile(r"""<(?:script|link|img|iframe|a)\b[^>]*(?:src|href)\s*=\s*["']?(?:https?:)?//""", re.IGNORECASE)


def markup(html: str) -> str:
    """The page without the inlined script bodies (plotly.js mentions URLs in strings)."""
    return re.sub(r"<script\b[^>]*>.*?</script>", "<script></script>", html, flags=re.S)


@pytest.fixture(scope="module")
def analysis(long_store):
    return analyse(compute(long_store))


def _embedded(html: str, element_id: str):
    m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % element_id, html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_percent_units_are_scaled():
    out = units.convert(np.array([0.27, 0.97]), np.array(["%", "%"], dtype=object), "%")
    assert out.tolist() == pytest.approx([27.0, 97.0])


def test_report_is_self_contained(analysis):
    html = build_report(analysis, demo=True)
    assert "Content-Security-Policy" in html and "connect-src" in html
    assert not EXTERNAL.search(markup(html))              # nothing loaded from the network
    assert "Not medical advice" in html and "Demo report" in html
    for section in ("summary", "activity", "heart", "sleep", "workouts", "years", "connections", "data"):
        assert f'id="{section}"' in html
    specs = _embedded(html, "hl-charts")                  # valid JSON: no NaN / Infinity
    assert {"steps_daily", "rhr_daily", "sleep_nights", "workouts_monthly", "devices"} <= set(specs)
    for cid, spec in specs.items():
        assert spec["table"]["rows"], cid                 # every chart has a table view
        assert f'data-chart="{cid}"' in html


def test_report_escapes_untrusted_names(analysis):
    src = analysis.ctx.m.sources
    original = src["name"].copy()
    try:
        src.loc[src["kind"] == "watch", "name"] = '<img src=x onerror=alert(1)></script>'
        src.loc[src["kind"] == "watch", "friendly"] = ""
        html = build_report(analysis)
    finally:
        src["name"] = original
    assert "<img src=x" not in markup(html)
    body = html.split('id="hl-charts">')[1].split("</script>")[0]
    json.loads(body.replace("<\\/", "</"))                # a hostile name cannot end the data block


def test_dashboard_payload_and_page(analysis):
    data = payload(analysis)
    n = len(data["dates"])
    keys = {m["key"] for m in data["metrics"]}
    assert {"steps", "resting_hr", "sleep_any_h"} <= keys
    assert all(len(m["values"]) == n for m in data["metrics"])
    ds = next(m for m in data["metrics"] if m["key"] == "double_support_pct")
    assert 20 < np.nanmean([v for v in ds["values"] if v is not None]) < 35
    assert data["workouts"] and data["nights"]
    html = build_dashboard(analysis, demo=True)
    assert not EXTERNAL.search(markup(html)) and "connect-src" in html
    assert _embedded(html, "hl-data")["defaults"]


def test_cli_end_to_end(tmp_path, small_export, capsys):
    path, _, _ = small_export
    home = tmp_path / "home"
    assert cli.main(["--home", str(home), "run", str(path), "--no-open"]) == 0
    assert (home / "reports" / "report.html").stat().st_size > 1_000_000
    assert (home / "reports" / "dashboard.html").exists()
    assert cli.main(["--home", str(home), "info"]) == 0
    assert cli.main(["--home", str(home), "report", "--units", "imperial", "--no-open"]) == 0
    assert cli.main(["--home", str(home), "export-csv"]) == 0
    assert (home / "csv" / "daily.csv").exists()
    assert cli.main(["--home", str(home), "clear"]) == 1          # needs --yes
    assert cli.main(["--home", str(home), "clear", "--yes"]) == 0
    assert not (home / "cache").exists()
    assert cli.main(["--home", str(home), "report", "--no-open"]) == 2   # nothing imported: clear error
    assert "healthlens import" in capsys.readouterr().err
    assert cli.main(["--home", str(home), "run", str(tmp_path / "missing.zip"), "--no-open"]) == 2
