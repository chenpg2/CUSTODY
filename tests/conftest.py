"""Fixtures shared by the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from quickstart import fabricate  # noqa: E402


@pytest.fixture(scope="session")
def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fabricated centre: fresh cycles and the frozen transfers that follow."""
    return fabricate(n_patients=300, seed=0)
