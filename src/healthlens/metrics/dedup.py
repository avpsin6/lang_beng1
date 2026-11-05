"""Source-priority de-duplication for cumulative quantities.

Your iPhone and Apple Watch both count steps (and distance, flights...). The raw
export contains *both* sets of samples, so naively summing them double counts.
Apple's Health app resolves this by source priority: take the top source's
samples, then let lower-priority sources fill only the time *not already
covered*. We implement the same idea:

1. Sources are ordered Watch > iPhone > other apps > manual entries (ties:
   the source with more samples first).
2. For each source in order, every sample keeps only the fraction of its
   duration that is not covered by higher-priority samples (values are
   prorated linearly in time). Instantaneous samples count only if they fall
   outside covered time.
3. "Coarse" samples (longer than 3 h, e.g. an app writing a daily total) are
   handled differently: they contribute ``max(0, value - higher-priority
   total within their span)`` — otherwise proration would inflate totals.

Everything is vectorised with numpy (sorted interval arrays + searchsorted),
so millions of samples take seconds.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

COARSE_SECONDS = 3 * 3600


def _union(a: np.ndarray, b: np.ndarray, s: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Union of sorted disjoint intervals (a, b) with arbitrary intervals (s, e)."""
    starts = np.concatenate([a, s])
    ends = np.concatenate([b, e])
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    run_max = np.maximum.accumulate(ends)
    new = np.empty(len(starts), dtype=bool)
    new[0] = True
    new[1:] = starts[1:] > run_max[:-1]
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(starts) - 1]
    return starts[first], run_max[last]


def _covered_upto(a: np.ndarray, b: np.ndarray, prefix: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Total covered length in (-inf, x] for sorted disjoint intervals."""
    idx = np.searchsorted(a, x, side="right") - 1
    out = np.zeros(len(x), dtype=np.float64)
    ok = idx >= 0
    i = idx[ok]
    out[ok] = prefix[i] + np.minimum(x[ok], b[i]) - a[i]
    return out


def _inside(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(a, x, side="right") - 1
    ok = idx >= 0
    res = np.zeros(len(x), dtype=bool)
    res[ok] = x[ok] < b[idx[ok]]
    return res


def source_order(df: pd.DataFrame) -> list:
    counts = df.groupby("source_id").size()
    prio = df.groupby("source_id")["priority"].first()
    return sorted(counts.index.tolist(), key=lambda s: (int(prio[s]), -int(counts[s]), s))


def dedup_values(df: pd.DataFrame) -> np.ndarray:
    """Return the de-duplicated contribution of every sample (same order as ``df``).

    ``df`` needs columns ``start``, ``end`` (seconds), ``value``, ``source_id``
    and ``priority``.
    """
    n = len(df)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    start = df["start"].to_numpy(dtype=np.float64)
    end = df["end"].to_numpy(dtype=np.float64)
    value = df["value"].to_numpy(dtype=np.float64)
    sid = df["source_id"].to_numpy()
    a = np.empty(0)
    b = np.empty(0)
    acc_mid = np.empty(0)     # midpoints of accepted contributions (sorted)
    acc_val = np.empty(0)
    for s_id in source_order(df):
        m = np.flatnonzero(sid == s_id)
        s, e, v = start[m], end[m], value[m]
        dur = e - s
        if len(a):
            prefix = np.r_[0.0, np.cumsum(b - a)[:-1]]
            cov = _covered_upto(a, b, prefix, e) - _covered_upto(a, b, prefix, s)
            safe = np.where(dur > 0, dur, 1.0)
            frac = np.where(dur > 0, 1.0 - cov / safe, (~_inside(a, b, s)).astype(np.float64))
            frac = np.clip(frac, 0.0, 1.0)
        else:
            frac = np.ones(len(m))
        contrib = v * frac
        coarse = dur > COARSE_SECONDS
        if coarse.any() and len(acc_mid):
            cum = np.r_[0.0, np.cumsum(acc_val)]
            lo = np.searchsorted(acc_mid, s[coarse], side="left")
            hi = np.searchsorted(acc_mid, e[coarse], side="right")
            higher = cum[hi] - cum[lo]
            contrib[coarse] = np.maximum(0.0, v[coarse] - higher)
        out[m] = contrib
        # update coverage and accepted contributions
        pos = dur > 0
        a, b = _union(a, b, s[pos], e[pos])
        mids = (s + e) / 2.0
        order = np.argsort(np.concatenate([acc_mid, mids]), kind="stable")
        acc_mid = np.concatenate([acc_mid, mids])[order]
        acc_val = np.concatenate([acc_val, contrib])[order]
    return out
