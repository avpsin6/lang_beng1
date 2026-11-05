from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from healthlens.analysis import analyse
from healthlens.analysis.fmt import Formatter
from healthlens.analysis.insights import friend_median
from healthlens.analysis.stats import (bh_adjust, block_bootstrap_diff, change_points, mann_kendall_p, spearman,
                                       theil_sen, trend)
from healthlens.metrics import compute
from healthlens.units import DisplayUnits


# --------------------------------------------------------------------------- stats
def test_theil_sen_is_robust_to_outliers():
    x = np.arange(50, dtype=float)
    y = 2 * x + 1
    y[[5, 17, 33]] = [500, -400, 900]
    slope, intercept = theil_sen(x, y)
    assert slope == pytest.approx(2.0) and intercept == pytest.approx(1.0)


def test_mann_kendall_detects_trend_but_not_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 60)
    assert mann_kendall_p(noise) > 0.05
    assert mann_kendall_p(noise + np.linspace(0, 4, 60)) < 0.001


def test_trend_reports_change_per_series():
    idx = pd.date_range("2018-01-01", periods=48, freq="MS")
    s = pd.Series(70 - 0.1 * np.arange(48), index=idx)
    t = trend(s)
    assert t.significant and t.change == pytest.approx(-4.7, abs=0.05)


def test_spearman_effective_n_shrinks_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    walk_x = np.cumsum(rng.normal(size=400))
    walk_y = np.cumsum(rng.normal(size=400))
    c = spearman(walk_x, walk_y)
    assert c.n_eff < 50                         # two random walks: tiny effective sample size
    x = rng.normal(size=400)
    c2 = spearman(x, x * 0.5 + rng.normal(size=400))
    assert c2.rho > 0.3 and c2.p_value < 1e-6


def test_bh_adjust_matches_reference():
    q = bh_adjust([0.01, 0.04, 0.03, 0.20])
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-4)


def test_change_points_find_level_shifts():
    y = np.r_[np.full(15, 10.0), np.full(12, 6.0), np.full(18, 12.0)] + np.random.default_rng(3).normal(0, 0.3, 45)
    assert change_points(y) == [15, 27]
    assert change_points(np.random.default_rng(4).normal(0, 1, 40)) == []


def test_block_bootstrap_difference():
    idx = pd.date_range("2020-01-01", periods=300)
    weekend = pd.Series(idx.day_of_week >= 5, index=idx)
    values = pd.Series(np.where(weekend, 9000.0, 7000.0), index=idx) + np.random.default_rng(5).normal(0, 800, 300)
    d = block_bootstrap_diff(values, weekend)
    assert d.clear and 1700 < d.diff < 2300 and d.low < d.diff < d.high


def test_friend_reference_lookup():
    assert friend_median(35, "male") == 42.4
    assert friend_median(52, "female") == 23.4
    assert friend_median(None, "male") is None and friend_median(35, None) is None


def test_formatter():
    f = Formatter(DisplayUnits("imperial"), clock24=False)
    assert f.clock(23.5) == "11:30 pm" and f.clock(24.25) == "12:15 am"
    assert f.hours(7.42) == "7 h 25 min"
    assert f.distance(10) == "6.2 mi"
    assert f.pace(5.0) == "8:03 /mi"
    assert f.value("weight_kg", 80) == "176.4 lb"
    assert Formatter(DisplayUnits("metric")).clock(23.5) == "23:30"


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def long_analysis(long_store):
    return analyse(compute(long_store))


def test_analysis_runs_without_errors(long_analysis):
    a = long_analysis
    assert a.errors == []
    ids = {i.id for i in a.insights}
    for required in ("coverage", "steps", "exercise", "resting_hr", "hrv", "vo2max", "sleep_coverage",
                     "workouts", "chapters", "periods"):
        assert required in ids, required
    assert len(a.key_findings()) == 8
    assert {k["key"] for k in a.kpis} >= {"steps", "resting_hr", "hrv"}


def test_chapters_find_the_lockdown_dip(long_analysis):
    ch = long_analysis.chapters["steps"]
    starts = [c.start for c in ch]
    assert any(abs((s - pd.Timestamp("2020-03-01")).days) <= 62 for s in starts)
    assert any(abs((s - pd.Timestamp("2021-06-01")).days) <= 62 for s in starts)


def test_raised_rhr_periods_match_simulated_illness(long_analysis, long_export):
    _, truth, _ = long_export
    ill_days = set(truth.loc[truth["ill"], "date"])
    periods = long_analysis.periods
    assert periods
    hits = sum(any(p.start - pd.Timedelta(days=2) <= d <= p.end + pd.Timedelta(days=2) for d in ill_days)
               for p in periods)
    assert hits / len(periods) >= 0.8


def test_long_term_relationship_found(long_analysis):
    monthly = {(f.pair.x, f.pair.y): f for f in long_analysis.relationships if f.level == "monthly"}
    f = monthly[("exercise_min", "resting_hr")]
    assert f.rho < -0.3 and f.significant


def test_yearly_table(long_analysis):
    y = long_analysis.yearly
    assert list(y.index) == [2018, 2019, 2020, 2021, 2022]
    assert y.loc[2020, "steps"] < y.loc[2019, "steps"]          # lockdown year
    assert y.loc[2022, "workouts_n"] > 0


FORBIDDEN = re.compile(r"\byou have (?!been|had|a |an |\d)|\bdiagnosed\b|\bdisease\b|\bdisorder\b|\bdangerous\b",
                       re.IGNORECASE)


def test_wording_is_descriptive_not_diagnostic(long_analysis):
    for i in long_analysis.insights:
        text = f"{i.title} {i.detail} {i.caveat or ''}"
        assert not FORBIDDEN.search(text), text


def test_short_export_degrades_gracefully(small_store):
    a = analyse(compute(small_store))
    assert a.errors == []
    assert any(i.id == "coverage" for i in a.insights)


def test_tiny_handcrafted_export(tmp_path):
    from healthlens.parser import import_export
    from healthlens.store import HealthStore
    from tests.test_parser import HANDCRAFTED

    xml = tmp_path / "export.xml"
    xml.write_text(HANDCRAFTED, encoding="utf-8")
    import_export(xml, tmp_path / "cache")
    a = analyse(compute(HealthStore(tmp_path / "cache")))
    assert a.errors == []
