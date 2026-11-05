"""Where HealthLens keeps its files.

Everything lives in one *workspace* folder, by default ``~/HealthLens`` —
deliberately outside the code repository so personal data can't be committed
by accident. Override with ``--home`` or the ``HEALTHLENS_HOME`` env var.

    ~/HealthLens/
        cache/            parsed export (Parquet) + meta.json
        cache/derived/    daily / sleep / workout tables (recomputed on demand)
        reports/          report.html, dashboard.html
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def default_home() -> Path:
    env = os.environ.get("HEALTHLENS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / "HealthLens"


@dataclass
class Workspace:
    home: Path

    @classmethod
    def at(cls, home: Optional[os.PathLike] = None) -> "Workspace":
        return cls(Path(home).expanduser().resolve() if home else default_home())

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def derived(self) -> Path:
        return self.cache / "derived"

    @property
    def reports(self) -> Path:
        return self.home / "reports"

    def ensure(self) -> "Workspace":
        self.reports.mkdir(parents=True, exist_ok=True)
        return self
