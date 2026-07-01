# tests/test_bake_black_sea.py
"""Unit tests for bake_black_sea helpers (no data files required)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`
import scripts.bake_black_sea as b


def _curve_df():
    return pd.DataFrame(
        {
            "age_ka": [9.0, 0.0],
            "level_m_global_gradual": [-35.0, 0.0],
            "level_m_global_catastrophic": [-100.0, 0.0],
        }
    )


def test_pontic_level_gradual_vs_catastrophic_diverge_at_9ka():
    df = _curve_df()
    assert b.pontic_level(df, 9.0, "gradual") == pytest.approx(-35.0)
    assert b.pontic_level(df, 9.0, "catastrophic") == pytest.approx(-100.0)


def test_pontic_level_unknown_scenario_raises():
    with pytest.raises(ValueError):
        b.pontic_level(_curve_df(), 9.0, "banana")


def test_pontic_level_sign_guard_rejects_positive_level():
    bad = pd.DataFrame({"age_ka": [9.0], "level_m_global_gradual": [5.0]})
    with pytest.raises(AssertionError):
        b.pontic_level(bad, 9.0, "gradual")
