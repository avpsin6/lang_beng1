from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from healthlens.parser import import_export
from healthlens.store import HealthStore
from healthlens.synthetic import SyntheticConfig, generate_export


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("synthetic")


@pytest.fixture(scope="session")
def small_export(synth_dir):
    """~5 months of synthetic data spanning watchOS 8 -> 9 (sleep stages appear) and a DST change."""
    cfg = SyntheticConfig(start=date(2022, 8, 1), end=date(2022, 12, 31), seed=3, sleep_tracking="often")
    path = synth_dir / "small" / "export.zip"
    truth = generate_export(path, cfg)
    return path, truth, cfg


@pytest.fixture(scope="session")
def small_store(small_export, synth_dir):
    path, truth, cfg = small_export
    cache = synth_dir / "small" / "cache"
    import_export(path, cache, force=True)
    return HealthStore(cache)


@pytest.fixture(scope="session")
def long_export(synth_dir):
    """~4.5 years (phone-only era, watch arrives, lockdown, running phase) with rare sleep tracking."""
    cfg = SyntheticConfig(start=date(2018, 3, 1), end=date(2022, 9, 30), seed=11, hr_every_min=40,
                          sleep_tracking="rare")
    path = synth_dir / "long" / "export.zip"
    truth = generate_export(path, cfg)
    return path, truth, cfg


@pytest.fixture(scope="session")
def long_store(long_export, synth_dir):
    path, truth, cfg = long_export
    cache = synth_dir / "long" / "cache"
    import_export(path, cache, force=True)
    return HealthStore(cache)
