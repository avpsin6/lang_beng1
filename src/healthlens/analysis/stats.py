"""Small, dependency-free statistics toolkit.

Health time series are autocorrelated (today resembles yesterday), which makes
naive p-values far too optimistic. Helpers here therefore use effective sample
sizes, block bootstraps and Benjamini-Hochberg correction where it matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def norm_sf(z: float) -> float:
    """Survival function of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0
    return float((x[1:] * x[:-1]).sum() / denom)


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------
@dataclass
class Trend:
    slope: float            # units per x-unit (we use years)
    intercept: float
    p_value: float
    n: int
    start_value: float      # fitted value at the first x
    end_value: float        # fitted value at the last x

    @property
    def change(self) -> float:
        return self.end_value - self.start_value

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 400_000, seed: int = 0) -> Tuple[float, float]:
    """Robust slope (median of pairwise slopes) and intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, float(np.nanmedian(y)) if n else float("nan")
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(i), max_pairs, replace=False)
        i, j = i[pick], j[pick]
    dx = x[j] - x[i]
    ok = dx != 0
    slopes = (y[j][ok] - y[i][ok]) / dx[ok]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def mann_kendall_p(y: np.ndarray) -> float:
    """Two-sided Mann-Kendall p-value with an autocorrelation-adjusted variance."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 6:
        return 1.0
    i, j = np.triu_indices(n, k=1)
    s = float(np.sign(y[j] - y[i]).sum())
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var = (n * (n - 1) * (2 * n + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18.0
    # variance inflation from lag-1 autocorrelation of detrended data
    slope, intercept = theil_sen(np.arange(n), y)
    r1 = lag1_autocorr(y - (slope * np.arange(n) + intercept))
    if r1 > 0:
        var *= (1 + r1) / (1 - r1)
    if var <= 0:
        return 1.0
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    return float(min(1.0, 2 * norm_sf(abs(z))))


def trend(series: pd.Series, min_points: int = 8) -> Optional[Trend]:
    """Trend of a time-indexed series (e.g. monthly means), slope per year."""
    s = series.dropna()
    if len(s) < min_points:
        return None
    x = (s.index - s.index[0]).days.to_numpy(dtype=float) / 365.25
    y = s.to_numpy(dtype=float)
    slope, intercept = theil_sen(x, y)
    p = mann_kendall_p(y)
    return Trend(slope=slope, intercept=intercept, p_value=p, n=len(s),
                 start_value=intercept + slope * x[0], end_value=intercept + slope * x[-1])


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
@dataclass
class Correlation:
    rho: float
    n: int
    n_eff: float
    p_value: float
    q_value: float = 1.0


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[Correlation]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10:
        return None
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    rho = float(np.corrcoef(rx, ry)[0, 1])
    r1x, r1y = lag1_autocorr(x), lag1_autocorr(y)
    factor = (1 - r1x * r1y) / (1 + r1x * r1y) if r1x * r1y > -1 else 1.0
    n_eff = max(4.0, n * min(1.0, factor))
    rho_c = max(-0.999999, min(0.999999, rho))
    z = math.atanh(rho_c) * math.sqrt(max(n_eff - 3, 1) / 1.06)
    return Correlation(rho=rho, n=n, n_eff=n_eff, p_value=float(min(1.0, 2 * norm_sf(abs(z)))))


def bh_adjust(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg false-discovery-rate adjusted q-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------
@dataclass
class Difference:
    diff: float
    low: float
    high: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float

    @property
    def clear(self) -> bool:
        """95 % interval excludes zero."""
        return self.low > 0 or self.high < 0


def block_bootstrap_diff(values: pd.Series, group: pd.Series, n_boot: int = 800, block: int = 7,
                         seed: int = 0) -> Optional[Difference]:
    """Mean(values | group) - mean(values | ~group) with a moving-block bootstrap CI.

    ``values`` and ``group`` are daily series on the same index. Resampling
    blocks of consecutive days keeps the autocorrelation structure.
    """
    df = pd.DataFrame({"v": values, "g": group}).dropna()
    if len(df) < 20:
        return None
    v = df["v"].to_numpy(dtype=float)
    g = df["g"].to_numpy(dtype=bool)
    if g.sum() < 5 or (~g).sum() < 5:
        return None
    n = len(v)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    idx = np.minimum(idx, n - 1)
    vb, gb = v[idx], g[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = (vb * gb).sum(1) / gb.sum(1)
        mb = (vb * ~gb).sum(1) / (~gb).sum(1)
    d = ma - mb
    d = d[np.isfinite(d)]
    if len(d) < n_boot * 0.5:
        return None
    mean_a, mean_b = float(v[g].mean()), float(v[~g].mean())
    return Difference(diff=mean_a - mean_b, low=float(np.percentile(d, 2.5)), high=float(np.percentile(d, 97.5)),
                      n_a=int(g.sum()), n_b=int((~g).sum()), mean_a=mean_a, mean_b=mean_b)


# ---------------------------------------------------------------------------
# Change points (binary segmentation on a short series such as monthly means)
# ---------------------------------------------------------------------------
def change_points(y: np.ndarray, min_size: int = 3, max_points: int = 3, penalty: Optional[float] = None) -> List[int]:
    """Indices where the mean level shifts (start index of each new segment)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * min_size + 1:
        return []
    var = float(np.var(np.diff(y)) / 2) if n > 2 else float(np.var(y))
    if var <= 0:
        return []
    pen = penalty if penalty is not None else 3.0 * var * math.log(n)

    def sse(a: int, b: int) -> float:
        seg = y[a:b]
        return float(((seg - seg.mean()) ** 2).sum())

    segments = [(0, n)]
    points: List[int] = []
    while len(points) < max_points:
        best = None
        for a, b in segments:
            if b - a < 2 * min_size:
                continue
            base = sse(a, b)
            for k in range(a + min_size, b - min_size + 1):
                gain = base - sse(a, k) - sse(k, b)
                if best is None or gain > best[0]:
                    best = (gain, k, a, b)
        if best is None or best[0] < pen:
            break
        _, k, a, b = best
        points.append(k)
        segments.remove((a, b))
        segments += [(a, k), (k, b)]
    return sorted(points)


# ---------------------------------------------------------------------------
def robust_baseline(series: pd.Series, window: int = 60, lag: int = 7, min_periods: int = 20):
    """Rolling median and robust SD of the *preceding* window (excluding the last ``lag`` days)."""
    shifted = series.shift(lag)
    med = shifted.rolling(window, min_periods=min_periods).median()
    mad = (shifted - med).abs().rolling(window, min_periods=min_periods).median() * 1.4826
    return med, mad


def detrend_rolling(series: pd.Series, window: int = 29) -> pd.Series:
    """Remove slow changes (seasons, fitness) with a centred rolling median."""
    base = series.rolling(window, center=True, min_periods=max(5, window // 3)).median()
    return series - base
