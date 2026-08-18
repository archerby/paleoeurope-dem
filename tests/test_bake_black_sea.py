# tests/test_bake_black_sea.py
"""Unit tests for bake_black_sea helpers (no data files required)."""
import ast
import dataclasses
import hashlib
import importlib
import inspect
import logging
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`
import scripts.bake_black_sea as b
from paleoeurope.gia.ice6g_topo import (
    SMOOTHING_RIM_CELLS,
    IceSurfaceUnmaskedWarning,
    minimum_reference_cells,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------------------
# Defect 1a — the per-scalar bound is a DATUM-SANITY check, not a sign guard
# ---------------------------------------------------------------------------


def test_datum_sanity_range_table_covers_pontic_and_caspian():
    """The bound is table-driven so a third basin needs no logic change."""
    ranges = b._DATUM_SANITY_RANGE_M
    assert ranges["pontic"] == (-150.0, 1.0)
    assert ranges["caspian"] == (-160.0, 60.0)


def test_pontic_level_accepts_plausible_negative_level():
    """A valid Pontic level inside -150..+1 m passes the bound unchanged."""
    ok = pd.DataFrame({"age_ka": [16.5], "level_m_global_gradual": [-30.0]})
    assert b.pontic_level(ok, 16.5, "gradual") == pytest.approx(-30.0)


def test_datum_bound_rejects_out_of_range_pontic_level():
    """+120 m is outside the Pontic datum window — caught as a datum violation.

    NOT advertised as a sign-flip guard: it only fires because the Pontic basin
    happens to be single-signed.  See the two-signed Caspian tests below.
    """
    bad = pd.DataFrame({"age_ka": [9.0], "level_m_global_gradual": [120.0]})
    with pytest.raises(ValueError, match="pontic"):
        b.pontic_level(bad, 9.0, "gradual")


def test_datum_bound_rejects_blowups_for_both_basins():
    """-3000 m and +900 m are datum/unit blow-ups in ANY basin -> ValueError."""
    for basin in ("pontic", "caspian"):
        for level in (-3000.0, 900.0):
            df = pd.DataFrame({"age_ka": [9.0], "level_m_global_gradual": [level]})
            with pytest.raises(ValueError, match="datum"):
                b.basin_level(df, 9.0, "gradual", basin=basin)


def test_khvalynian_highstand_passes_for_caspian_but_not_pontic():
    """+35 m (HESS 2024 Khvalynian central highstand) is Caspian-plausible."""
    khval = pd.DataFrame({"age_ka": [16.0], "level_m_global_gradual": [35.0]})
    assert b.basin_level(khval, 16.0, "gradual", basin="caspian") == pytest.approx(35.0)
    with pytest.raises(ValueError):
        b.basin_level(khval, 16.0, "gradual", basin="pontic")


def test_caspian_deep_lowstand_and_undated_cap_bracket_the_datum_window():
    """-100 m lowstand and the undated +50 m cap both stay inside the window."""
    for level in (-100.0, 50.0):
        df = pd.DataFrame({"age_ka": [10.0], "level_m_global_gradual": [level]})
        assert b.basin_level(df, 10.0, "gradual", basin="caspian") == pytest.approx(level)
    too_deep = pd.DataFrame({"age_ka": [10.0], "level_m_global_gradual": [-200.0]})
    with pytest.raises(ValueError):
        b.basin_level(too_deep, 10.0, "gradual", basin="caspian")


@pytest.mark.parametrize("flipped", [-48.0, -50.0, -35.0, -22.0, -20.0, -14.0])
def test_datum_bound_cannot_detect_a_caspian_sign_flip(flipped):
    """HONEST NEGATIVE: every Khvalynian ladder rung has an in-window twin.

    +48/+50, +34/+36 (Talginka), +20/+22 (Buynaksk), +14/+15 (Turkmenian) all
    mirror into -160..+60, so the per-scalar bound accepts them.  This test
    documents WHY the real guard has to be curve-level.
    """
    df = pd.DataFrame({"age_ka": [14.0], "level_m_global_gradual": [flipped]})
    assert b.basin_level(df, 14.0, "gradual", basin="caspian") == pytest.approx(flipped)


def test_basin_level_unknown_basin_raises():
    with pytest.raises(ValueError, match="basin"):
        b.basin_level(_curve_df(), 9.0, "gradual", basin="banana")


def test_level_guard_is_not_an_assert_statement():
    """`assert` is stripped under python -O; the guard must not rely on it."""
    src = (
        inspect.getsource(b.basin_level)
        + inspect.getsource(b.pontic_level)
        + inspect.getsource(b._check_datum_sanity)
        + inspect.getsource(b.validate_basin_curve)
    )
    assert "assert " not in src


def test_datum_sanity_docstring_disclaims_sign_flip_detection():
    """The bound must not advertise a power it does not have."""
    doc = (b._check_datum_sanity.__doc__ or "").lower()
    assert "sign flip" in doc
    assert "validate_basin_curve" in doc


# ---------------------------------------------------------------------------
# Defect 1b — curve-level validation is the real sign-flip guard
# ---------------------------------------------------------------------------


def _khvalynian_df():
    """Plausible Khvalynian Caspian curve: -28 at 0 ka, +35 highstand, ladder."""
    return pd.DataFrame(
        {
            # LGM Atelian lowstand -> Early Khvalynian rise -> Talginka +35 ->
            # Buynaksk +20 -> Turkmenian +14 -> Makhachkala 0 -> modern -28.
            "age_ka": [22.0, 18.0, 16.0, 14.0, 13.0, 12.0, 9.0, 5.0, 0.0],
            "level_m_global_gradual": [
                -80.0, -50.0, 35.0, 20.0, 14.0, 0.0, -50.0, -30.0, -28.0
            ],
            "level_m_global_catastrophic": [
                -80.0, -50.0, 35.0, 20.0, 14.0, 0.0, -50.0, -30.0, -28.0
            ],
        }
    )


def _negate_levels(df):
    out = df.copy()
    for col in out.columns:
        if col.startswith("level_m_global_"):
            out[col] = -out[col]
    return out


def test_validate_basin_curve_accepts_plausible_khvalynian_curve():
    """Requirement 4 — the shape the future _build_caspian_curve.py must pass."""
    df = _khvalynian_df()
    for scenario in ("gradual", "catastrophic"):
        b.validate_basin_curve(df, "caspian", scenario)


def test_validate_basin_curve_rejects_negated_khvalynian_curve():
    """Requirement 1 — a wholesale negation is named as a sign flip."""
    flipped = _negate_levels(_khvalynian_df())
    with pytest.raises(ValueError, match="(?i)sign flip"):
        b.validate_basin_curve(flipped, "caspian", "gradual")


def test_validate_basin_curve_rejects_positive_caspian_anchor():
    """Requirement 2 — 0 ka at +28 m instead of -28 m is rejected."""
    df = _khvalynian_df()
    df.loc[df["age_ka"] == 0.0, "level_m_global_gradual"] = 28.0
    with pytest.raises(ValueError, match="0 ka"):
        b.validate_basin_curve(df, "caspian", "gradual")


def test_validate_basin_curve_rejects_caspian_curve_without_positive_max():
    """Requirement 3 — a Khvalynian curve must rise above the +23 m Manych sill.

    Anchor and mirror both pass here (0 ka is a correct -28 m); only the
    highstand-sign check can reject it.
    """
    df = _khvalynian_df()
    for col in ("level_m_global_gradual", "level_m_global_catastrophic"):
        df[col] = [-80.0, -50.0, -5.0, -10.0, -14.0, -20.0, -50.0, -30.0, -28.0]
    with pytest.raises(ValueError, match="(?i)highstand"):
        b.validate_basin_curve(df, "caspian", "gradual")


def _real_pontic_df():
    if not b.PONTIC_CSV.exists():
        pytest.skip(f"Pontic curve CSV not available at {b.PONTIC_CSV}")
    return pd.read_csv(b.PONTIC_CSV, comment="#")


def test_validate_basin_curve_accepts_the_real_pontic_curve():
    """Requirement 5 — regression: the working basin still validates."""
    df = _real_pontic_df()
    for scenario in ("gradual", "catastrophic"):
        b.validate_basin_curve(df, "pontic", scenario)


def test_validate_basin_curve_rejects_the_negated_real_pontic_curve():
    """Requirement 6 — the same curve, negated, must not validate."""
    flipped = _negate_levels(_real_pontic_df())
    for scenario in ("gradual", "catastrophic"):
        with pytest.raises(ValueError):
            b.validate_basin_curve(flipped, "pontic", scenario)


def test_mirror_test_is_blind_at_a_zero_anchor_but_the_datum_window_is_not():
    """HONEST NEGATIVE: -0.0 == 0.0, so a Pontic flip cannot trip the mirror test.

    A flipped Pontic curve is rejected by the datum window instead (that basin
    IS single-signed, and its window is asymmetric).  Pinning this so nobody
    later "simplifies" the two checks into one and loses Pontic coverage.
    """
    flipped = _negate_levels(_real_pontic_df())
    with pytest.raises(ValueError, match="(?i)datum") as exc:
        b.validate_basin_curve(flipped, "pontic", "gradual")
    assert "sign flip" not in str(exc.value).lower()


def test_validate_basin_curve_rejects_curve_without_a_zero_ka_anchor():
    df = _khvalynian_df()
    df = df[df["age_ka"] > 0.0]
    with pytest.raises(ValueError, match="(?i)anchor"):
        b.validate_basin_curve(df, "caspian", "gradual")


def test_validate_basin_curve_rejects_unknown_basin_and_scenario():
    df = _khvalynian_df()
    with pytest.raises(ValueError, match="basin"):
        b.validate_basin_curve(df, "banana", "gradual")
    with pytest.raises(ValueError, match="scenario"):
        b.validate_basin_curve(df, "caspian", "banana")


# ---------------------------------------------------------------------------
# Defect 1c (round 3) — the three scalar checks leak on SUBSET sign flips
#
# Mirror and anchor read curve[0 ka]; the old highstand check read max(curve)
# and passed as long as ANY positive rung survived.  A flip confined to a block
# of rows, with the anchor left intact, was therefore invisible.  Two new checks
# replace that: Check A (the highstand must clear the measured Manych sill) and
# Check B (physical continuity — |dlevel|/|dage| between age-sorted neighbours).
# ---------------------------------------------------------------------------

# The documented Khvalynian ladder, gradual = low bound of each published
# range, catastrophic = high bound:
#   +48/+50 undated cap, +34/+36 Talginka, +20/+22 Buynaksk, +14/+15 Turkmenian,
#   +4/+6, 0/-2, -5/-6, -10/-12, -16/-17, and the 0 ka anchor at -28.
# The five transgressive rungs sit on a 0.5-ka spacing (this is the spacing the
# attack arithmetic assumes: negating +48 next to +34 is an 82 m step in 0.5 ka
# = 164 m/ka).  The -16/-17 rung sits at 11.0 ka, i.e. BELOW the 11.7 ka
# Pleistocene/Holocene boundary, so attack 3 and attack 4 stay distinct.
_LADDER_AGES = [16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 11.0, 0.0]
_LADDER_GRADUAL = [48.0, 34.0, 20.0, 14.0, 4.0, 0.0, -5.0, -10.0, -16.0, -28.0]
_LADDER_CATASTROPHIC = [50.0, 36.0, 22.0, 15.0, 6.0, -2.0, -6.0, -12.0, -17.0, -28.0]


def _ladder_df():
    """The upright Khvalynian ladder — the shape a real Caspian curve must have."""
    return pd.DataFrame(
        {
            "age_ka": list(_LADDER_AGES),
            "level_m_global_gradual": list(_LADDER_GRADUAL),
            "level_m_global_catastrophic": list(_LADDER_CATASTROPHIC),
        }
    )


def _forge(df, fn):
    """Rewrite every level column through ``fn(age, level) -> level``."""
    out = df.copy()
    for col in out.columns:
        if col.startswith("level_m_global_"):
            out[col] = [
                float(fn(float(a), float(v)))
                for a, v in zip(out["age_ka"], out[col])
            ]
    return out


def _both_scenarios_reject(df, basin="caspian"):
    messages = []
    for scenario in ("gradual", "catastrophic"):
        with pytest.raises(ValueError) as exc:
            b.validate_basin_curve(df, basin, scenario)
        messages.append(str(exc.value))
    return messages


# --- mandatory regression test 1 — THE definition of done -------------------


def test_attack_a2_arslanov_typo_only_the_top_rung_negated_is_rejected():
    """A2: ONLY the +48/+50 rows are negated (the real arslanov2016 typo).

    The anchor is untouched, and +34/+36 survives as a positive maximum ABOVE
    the +23 m Manych sill, so Check A cannot see this.  Only physical
    continuity can: -48 next to +34 across 0.5 ka is 164 m/ka.
    """
    forged = _forge(_ladder_df(), lambda a, v: -v if v >= 48.0 else v)
    messages = _both_scenarios_reject(forged)
    for msg in messages:
        assert "m/ka" in msg, msg


# --- mandatory regression tests 2-5 -----------------------------------------


def test_attack_a2b_whole_highstand_shoulder_negated_is_rejected():
    """A2b: negate every rung >= +14 m, so max collapses to +4/+6 m."""
    forged = _forge(_ladder_df(), lambda a, v: -v if v >= 14.0 else v)
    assert max(forged["level_m_global_catastrophic"]) == pytest.approx(6.0)
    for msg in _both_scenarios_reject(forged):
        assert "sill" in msg.lower(), msg


def test_attack_a7_anchor_intact_pleistocene_half_flipped_is_rejected():
    """A7: 0 ka anchor left intact, every row at >= 11.7 ka negated."""
    forged = _forge(_ladder_df(), lambda a, v: -v if a >= 11.7 else v)
    assert forged.loc[forged["age_ka"] == 0.0, "level_m_global_gradual"].iloc[0] == -28.0
    _both_scenarios_reject(forged)


def test_attack_a9e_every_row_flipped_except_the_anchor_is_rejected():
    """A9e: every row negated EXCEPT the 0 ka anchor."""
    forged = _forge(_ladder_df(), lambda a, v: v if a == 0.0 else -v)
    assert forged.loc[forged["age_ka"] == 0.0, "level_m_global_gradual"].iloc[0] == -28.0
    _both_scenarios_reject(forged)


def test_attack_a9i_every_positive_rung_squashed_is_rejected():
    """A9i: every positive rung squashed to +0.1 m — a basin that never spills."""
    forged = _forge(_ladder_df(), lambda a, v: 0.1 if v > 0.0 else v)
    for msg in _both_scenarios_reject(forged):
        assert "sill" in msg.lower(), msg


# --- mandatory regression test 6 --------------------------------------------


def test_upright_khvalynian_ladder_is_accepted():
    """The documented ladder itself must pass both new checks."""
    df = _ladder_df()
    for scenario in ("gradual", "catastrophic"):
        b.validate_basin_curve(df, "caspian", scenario)


# --- mandatory regression tests 7 & 8 (the ones that matter most) ----------


def test_real_pontic_curve_still_accepted_after_the_new_checks():
    df = _real_pontic_df()
    for scenario in ("gradual", "catastrophic"):
        b.validate_basin_curve(df, "pontic", scenario)


def test_real_pontic_curve_negated_is_still_rejected_after_the_new_checks():
    flipped = _negate_levels(_real_pontic_df())
    for scenario in ("gradual", "catastrophic"):
        with pytest.raises(ValueError):
            b.validate_basin_curve(flipped, "pontic", scenario)


# --- Check A: the sill threshold and its opt-out ----------------------------


def test_manych_sill_constant_is_the_measured_value():
    assert b.MANYCH_SILL_M == pytest.approx(23.0)


def test_highstand_check_uses_the_sill_not_merely_a_positive_max():
    """A +10 m peak is positive but never clears the +23 m sill -> reject."""
    df = _forge(_ladder_df(), lambda a, v: 10.0 if v > 10.0 else v)
    assert max(df["level_m_global_gradual"]) == pytest.approx(10.0)
    for msg in _both_scenarios_reject(df):
        assert "23" in msg, msg


def test_spillway_requirement_is_on_by_default_and_opt_out_is_explicit():
    """Silence must not buy a weaker check; a caller must say so in code."""
    sig = inspect.signature(b.validate_basin_curve)
    assert sig.parameters["require_spillway_highstand"].default is True
    non_spilling = _forge(_ladder_df(), lambda a, v: min(v, 5.0))
    with pytest.raises(ValueError, match="(?i)sill"):
        b.validate_basin_curve(non_spilling, "caspian", "gradual")
    b.validate_basin_curve(
        non_spilling, "caspian", "gradual", require_spillway_highstand=False
    )


# --- Check B: the continuity limit, chosen by measurement -------------------


def _measured_max_rate(df):
    import numpy as np

    worst = 0.0
    for col in ("level_m_global_gradual", "level_m_global_catastrophic"):
        d = df[["age_ka", col]].dropna().sort_values("age_ka")
        a = d["age_ka"].to_numpy(dtype=float)
        lv = d[col].to_numpy(dtype=float)
        da, dl = np.diff(a), np.diff(lv)
        keep = da > 0
        if keep.any():
            worst = max(worst, float((abs(dl[keep]) / da[keep]).max()))
    return worst


def test_rate_limit_table_clears_the_measured_real_pontic_curve():
    """The Pontic limit must sit ABOVE the real curve's own worst rate."""
    measured = _measured_max_rate(_real_pontic_df())
    # 9.4->9.5 ka gradual = 100.0 m/ka; 7.15->7.5 ka catastrophic = 226.857 m/ka.
    assert measured == pytest.approx(226.857, abs=0.01)
    assert b._MAX_LEVEL_RATE_M_PER_KA["pontic"] > measured


def test_caspian_rate_limit_is_lethal_to_the_arslanov_typo():
    """82-86 m across 0.5 ka is 164-172 m/ka; the Caspian limit must be below."""
    assert b._MAX_LEVEL_RATE_M_PER_KA["caspian"] < 164.0
    # ...and comfortably above the upright ladder's own worst step (28 m/ka).
    assert b._MAX_LEVEL_RATE_M_PER_KA["caspian"] > 2.0 * _measured_max_rate(_ladder_df())


def test_a_single_shared_rate_limit_would_be_unable_to_do_both_jobs():
    """HONEST NEGATIVE, pinned: the two constraints need per-basin constants.

    The real Pontic curve contains a genuine 226.9 m/ka event (the catastrophic
    reconnection), while the arslanov2016 attack is only 164 m/ka.  One constant
    admitting the former necessarily admits the latter.
    """
    assert _measured_max_rate(_real_pontic_df()) > 164.0
    assert (
        b._MAX_LEVEL_RATE_M_PER_KA["pontic"] > b._MAX_LEVEL_RATE_M_PER_KA["caspian"]
    )


def test_continuity_check_is_order_independent():
    """Shuffled row order must give the same verdict (ages are sorted first)."""
    forged = _forge(_ladder_df(), lambda a, v: -v if v >= 48.0 else v)
    shuffled = forged.sample(frac=1.0, random_state=7).reset_index(drop=True)
    with pytest.raises(ValueError, match="m/ka"):
        b.validate_basin_curve(shuffled, "caspian", "gradual")
    upright = _ladder_df().sample(frac=1.0, random_state=3).reset_index(drop=True)
    b.validate_basin_curve(upright, "caspian", "gradual")


def test_duplicate_ages_do_not_divide_by_zero():
    df = _ladder_df()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    b.validate_basin_curve(dup, "caspian", "gradual")


def test_single_row_curve_does_not_crash_the_continuity_check():
    one = pd.DataFrame({"age_ka": [0.0], "level_m_global_gradual": [-28.0]})
    b.validate_basin_curve(
        one, "caspian", "gradual", require_spillway_highstand=False
    )


def test_non_finite_rows_are_filtered_before_differencing_but_still_rejected():
    """+inf must not poison the rate arithmetic, and must still be rejected."""
    df = _ladder_df()
    df.loc[df["age_ka"] == 15.0, "level_m_global_gradual"] = float("inf")
    with pytest.raises(ValueError, match="(?i)datum"):
        b.validate_basin_curve(df, "caspian", "gradual")


# --- the anchor window must read EVERY row inside it ------------------------


def test_duplicated_near_present_row_with_a_flipped_value_is_rejected():
    """A second row inside the anchor tolerance carrying +28 must not hide."""
    df = _ladder_df()
    extra = pd.DataFrame(
        {
            "age_ka": [0.1],
            "level_m_global_gradual": [28.0],
            "level_m_global_catastrophic": [28.0],
        }
    )
    df = pd.concat([df, extra], ignore_index=True)
    # min(abs(age)) still selects the untouched 0.0 row, so the old
    # single-point anchor read this curve as clean.
    with pytest.raises(ValueError, match="(?i)sign flip|0 ka"):
        b.validate_basin_curve(df, "caspian", "gradual")


# --- docstring honesty -------------------------------------------------------


def test_validate_basin_curve_docstring_states_the_known_limit_precisely():
    doc = (b.validate_basin_curve.__doc__ or "")
    low = doc.lower()
    assert "known limit" in low
    assert "manych" in low and "23" in doc
    # it must name what each check READS, and what stays invisible
    assert "0 ka" in doc
    assert "adjacent" in low or "neighbour" in low
    assert "invisible" in low or "cannot" in low


def test_manych_sill_constant_records_its_provenance():
    src = Path(b.__file__).read_text(encoding="utf-8")
    head, _, _ = src.partition("MANYCH_SILL_M = 23.0")
    tail = head[-900:]
    assert "gebco" in tail.lower()
    assert "44.119" in tail and "45.773" in tail
    assert "2026-07-28" in tail


# ---------------------------------------------------------------------------
# Regression — pontic_level() must be byte-identical to the historical values
# ---------------------------------------------------------------------------

# epoch -> (gradual, catastrophic), verified before the guard redesign.
_HISTORICAL_PONTIC_LEVELS = {
    22: (-120.0, -120.0),
    18: (-115.0, -115.0),
    16.5: (-30.0, -30.0),
    14.5: (-32.5, -32.5),
    12: (-60.0, -60.0),
    11: (-78.2, -78.2),
    10: (-80.0, -80.0),
    9.5: (-40.0, -100.0),
    9: (-35.0, -100.0),
    7.5: (-22.0, -100.0),
    5: (-10.0, -10.0),
    2: (-4.0, -4.0),
    0: (0.0, 0.0),
}


def test_pontic_level_matches_the_historical_13x2_value_set():
    df = _real_pontic_df()
    assert list(_HISTORICAL_PONTIC_LEVELS) == b.EPOCHS_KA
    for epoch, (grad, cat) in _HISTORICAL_PONTIC_LEVELS.items():
        assert b.pontic_level(df, epoch, "gradual") == pytest.approx(grad)
        assert b.pontic_level(df, epoch, "catastrophic") == pytest.approx(cat)


def test_level_guard_survives_python_dash_O():
    """End-to-end: the guard still raises when assertions are disabled."""
    prog = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import pandas as pd
        import scripts.bake_black_sea as b

        bad = pd.DataFrame({"age_ka": [9.0], "level_m_global_gradual": [120.0]})
        try:
            b.pontic_level(bad, 9.0, "gradual")
        except ValueError:
            print("GUARDED")
        else:
            print("LEAKED")

        flipped = pd.DataFrame({
            "age_ka": [16.0, 0.0],
            "level_m_global_gradual": [-35.0, 28.0],
        })
        try:
            b.validate_basin_curve(flipped, "caspian", "gradual")
        except ValueError:
            print("CURVE_GUARDED")
        else:
            print("CURVE_LEAKED")
        """
    )
    res = subprocess.run(
        [sys.executable, "-O", "-c", prog, str(_REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "GUARDED" in res.stdout
    assert "CURVE_GUARDED" in res.stdout


# ---------------------------------------------------------------------------
# Defect 2 — env-var indirection for the data/output roots
# ---------------------------------------------------------------------------


def test_roots_default_to_historical_repo_relative_locations(monkeypatch):
    """Unset env vars must reproduce today's paths, expressed repo-relatively.

    TWO THINGS THIS USED TO GET WRONG, both found on CT 129 paleo-compute
    (2026-08-18), where the repo lives at ``/opt/paleo/paleoeurope-dem`` and
    ``/opt/Paleo-Europe_3D`` is a symlink to ``/opt/paleo/Paleo-Europe_3D``:

    1. It read ``b.PALEO_DATA_ROOT`` / ``b.PALEO_OUT_ROOT`` as imported, i.e. with
       whatever ``PALEO_DATA_ROOT`` / ``PALEO_OUT_ROOT`` the shell happened to
       export -- so a test whose whole subject is the DEFAULT was reporting on the
       override.  The env is now cleared and the module reloaded, the same way
       ``test_paleo_*_env_redirects_*`` below does it in the other direction.
    2. It compared a ``.resolve()``d actual against an unresolved expected.  On a
       box where any component of the sibling path is a symlink those are two
       spellings of one directory and the assertion fails on the spelling.  The
       expected side is resolved too, so the check still fails if the default
       points at a *different* directory -- which is the thing it is for.
    """
    monkeypatch.delenv("PALEO_DATA_ROOT", raising=False)
    monkeypatch.delenv("PALEO_OUT_ROOT", raising=False)
    reloaded = importlib.reload(b)
    try:
        assert reloaded.PALEO_DATA_ROOT == (
            _REPO_ROOT.parents[1] / "Paleo-Europe_3D"
        ).resolve()
        assert reloaded.PALEO_OUT_ROOT == _REPO_ROOT.parent.resolve()
        assert reloaded.OUT_BASE == (
            _REPO_ROOT.parent / "black_sea_corrected" / "topo"
        ).resolve()
        assert reloaded.ICE6G_DIR == (
            reloaded.PALEO_DATA_ROOT / "data" / "raw" / "ICE6G"
        )
        assert reloaded.BS_DEM == (
            reloaded.PALEO_DATA_ROOT / "data" / "raw" / "GEBCO" / "bs_gmrt_bsproper.tif"
        )
    finally:
        monkeypatch.undo()
        importlib.reload(b)


def test_no_hardcoded_drive_absolute_paths_in_module():
    src = Path(b.__file__).read_text(encoding="utf-8")
    assert "D:/pasha" not in src
    assert "D:\\pasha" not in src


def test_paleo_out_root_env_redirects_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("PALEO_OUT_ROOT", str(tmp_path / "renders"))
    reloaded = importlib.reload(b)
    try:
        assert reloaded.PALEO_OUT_ROOT == (tmp_path / "renders").resolve()
        assert reloaded.OUT_BASE == (
            tmp_path / "renders" / "black_sea_corrected" / "topo"
        ).resolve()
    finally:
        monkeypatch.delenv("PALEO_OUT_ROOT")
        importlib.reload(b)


def test_paleo_data_root_env_redirects_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("PALEO_DATA_ROOT", str(tmp_path / "tank"))
    reloaded = importlib.reload(b)
    try:
        root = (tmp_path / "tank").resolve()
        assert reloaded.PALEO_DATA_ROOT == root
        assert reloaded.ICE6G_DIR == root / "data" / "raw" / "ICE6G"
        assert reloaded.SPRATT_CSV.is_relative_to(root)
        assert reloaded.PONTIC_CSV.is_relative_to(root)
        assert reloaded.PALEOCOASTLINES_SHP.is_relative_to(root)
    finally:
        monkeypatch.delenv("PALEO_DATA_ROOT")
        importlib.reload(b)


# ===========================================================================
# TRACK B — the ice-surface guard must reach the code path that actually bakes
#
# ``gia_delta_subset`` is the ONLY place in this project that runs GIA on the
# real ICE-6G distribution.  It called ``solid_displacement`` with no ``sftgif``,
# so every raster the Black-Sea bake produces was built with the ice-surface
# guard switched off.  Numerically harmless for THIS bbox (the window is ice-free
# at every shipped epoch, proved below) and therefore exactly the kind of latent
# defect that only bites once someone points the bbox somewhere colder.
# ===========================================================================


# --- synthetic global ICE-6G fixtures ---------------------------------------
#
# 1-degree global grid: n_lon == 2*n_lat and n_lat >= 90, which is what
# ``global_latitude_grid`` accepts, so the datum can be derived from it exactly
# as it is from the real 1080x2160 files.  Longitudes run 0..360 like the real
# distribution.  Tropical continental land (|lat| <= 20, lon <= 60, topo_0 > 0)
# supplies the far-field reference set; the Black-Sea window is deliberately
# OUTSIDE the +/-25 deg band so it can never leak into the datum.

#: 0.5-degree global grid.  n_lon == 2*n_lat and n_lat >= 90 is what
#: ``global_latitude_grid`` accepts.  Half a degree rather than one because
#: ``SMOOTHING_RIM_CELLS`` = 8 is calibrated in CELLS on the 10-arcmin
#: distribution (~150 km); on a 1-degree grid the same 8 cells would be 8 degrees
#: and would swallow the whole 10-degree-tall window.
_FAKE_N_LAT, _FAKE_N_LON = 360, 720
#: Δz the fixture plants inside the Black-Sea window, in metres.
_FAKE_LOCAL_DZ_M = 7.0
#: Ice-surface height the fixture plants over its glaciated cells.  Stands in for
#: the +1711 m Fennoscandian / +2736 m Hudson readings the real files give.
_FAKE_ICE_SURFACE_M = 2000.0


def _fake_ice6g_arrays(*, offset_m, local_dz_m, ice_in_window, ice_lat_min=None):
    """Synthetic global ICE-6G fields.

    *ice_in_window* glaciates the whole Black-Sea window; *ice_lat_min* glaciates
    only the part of it at or above that latitude, so a fixture can straddle a
    palaeo-ice margin.
    """
    lat = -90.0 + (np.arange(_FAKE_N_LAT) + 0.5) * (180.0 / _FAKE_N_LAT)
    lon = (np.arange(_FAKE_N_LON) + 0.5) * (360.0 / _FAKE_N_LON)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    tropical_land = (np.abs(lat2d) <= 20.0) & (lon2d <= 60.0)
    in_window = (
        (lat2d >= b.LAT_MIN - b.BUF)
        & (lat2d <= b.LAT_MAX + b.BUF)
        & (lon2d >= b.LON_MIN - b.BUF)
        & (lon2d <= b.LON_MAX + b.BUF)
    )
    topo_0 = np.where(tropical_land, 500.0, -3000.0)
    topo_0 = np.where(in_window, 100.0, topo_0)

    if ice_lat_min is not None:
        ice = in_window & (lat2d >= float(ice_lat_min))
    else:
        ice = in_window & bool(ice_in_window)
    topo_t = topo_0 + offset_m + np.where(in_window, local_dz_m, 0.0)
    # Over ice, ICE-6G ``Topo`` is the ICE SURFACE, not bedrock — the whole point.
    topo_t = np.where(ice, topo_0 + offset_m + _FAKE_ICE_SURFACE_M, topo_t)

    sftlf = np.where(topo_0 > 0.0, 100.0, 0.0)
    sftgif = np.where(ice, 100.0, 0.0)
    return lat, lon, topo_0, topo_t, sftlf, sftgif


def _write_fake_ice6g(
    directory,
    epoch,
    *,
    offset_m,
    local_dz_m=_FAKE_LOCAL_DZ_M,
    ice_in_window=False,
    ice_lat_min=None,
    with_sftgif=True,
):
    xr = pytest.importorskip("xarray")
    lat, lon, topo_0, topo_t, sftlf, sftgif = _fake_ice6g_arrays(
        offset_m=offset_m,
        local_dz_m=local_dz_m,
        ice_in_window=ice_in_window,
        ice_lat_min=ice_lat_min,
    )
    data = {
        "sftlf": (("lat", "lon"), sftlf.astype("f4")),
        "Topo": (("lat", "lon"), topo_t.astype("f4")),
        "Orog": (("lat", "lon"), np.maximum(topo_t, 0.0).astype("f4")),
    }
    if with_sftgif:
        data["sftgif"] = (("lat", "lon"), sftgif.astype("f4"))
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / b._epoch_filename(epoch)
    ds.to_netcdf(path)
    return path


def _fake_ice6g_dir(tmp_path, monkeypatch, *, epoch, offset_m=80.0, **kwargs):
    """Two-file synthetic ICE-6G directory (epoch + 0 ka), wired into the module."""
    directory = tmp_path / "ICE6G"
    _write_fake_ice6g(directory, epoch, offset_m=offset_m, **kwargs)
    _write_fake_ice6g(directory, 0, offset_m=0.0, local_dz_m=0.0, ice_in_window=False,
                      with_sftgif=kwargs.get("with_sftgif", True))
    monkeypatch.setattr(b, "ICE6G_DIR", directory)
    return directory


def _fake_dem(tmp_path, *, lat_lo=42.0, lat_hi=45.0, lon_lo=30.0, lon_hi=35.0, n=24):
    """Small modern-DEM GeoTIFF wholly inside the BS model window."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    path = tmp_path / "fake_bs_dem.tif"
    dem = np.full((n, n), -50.0, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": n,
        "width": n,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_bounds(lon_lo, lat_lo, lon_hi, lat_hi, n, n),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(dem, 1)
    return path


def test_fake_ice6g_fixture_supplies_a_usable_far_field_reference_set():
    """Guard the guard: if the fixture had no tropical land the datum would raise."""
    _, _, topo_0, _, _, _ = _fake_ice6g_arrays(
        offset_m=80.0, local_dz_m=_FAKE_LOCAL_DZ_M, ice_in_window=False
    )
    from paleoeurope.gia.ice6g_topo import far_field_reference_mask

    ref = far_field_reference_mask(topo_0)
    assert int(ref.sum()) > minimum_reference_cells(topo_0.size)


# --- requirement 1: the ice information must actually be passed through -------


def _real_ice6g_or_skip(epoch=16.5):
    for ep in (epoch, 0):
        path = b.ICE6G_DIR / b._epoch_filename(ep)
        if not path.exists():
            pytest.skip(f"real ICE-6G file not available at {path}")
    return b.ICE6G_DIR


def test_gia_delta_subset_does_not_warn_about_unmasked_ice_surface():
    """THE DEFECT.  The only real-data GIA call in the project ran unguarded."""
    _real_ice6g_or_skip(16.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IceSurfaceUnmaskedWarning)
        b.gia_delta_subset(16.5)


def test_gia_delta_subset_hands_solid_displacement_the_ice_fields(tmp_path, monkeypatch):
    """``sftgif`` for BOTH endpoints reaches ``solid_displacement``."""
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_in_window=True)
    seen = {}
    real = b.solid_displacement

    def spy(topo_t, topo_0, offset, **kwargs):
        seen.update(kwargs)
        seen["dz_shape"] = np.asarray(topo_t).shape
        return real(topo_t, topo_0, offset, **kwargs)

    monkeypatch.setattr(b, "solid_displacement", spy)
    b.gia_delta_subset(16.5)
    assert seen.get("sftgif_t") is not None, "sftgif at epoch t was not passed"
    assert seen.get("sftgif_0") is not None, "sftgif at 0 ka was not passed"
    assert np.asarray(seen["sftgif_t"]).shape == seen["dz_shape"]
    assert np.asarray(seen["sftgif_0"]).shape == seen["dz_shape"]


def test_glaciated_window_comes_back_as_nan_not_as_an_ice_surface(tmp_path, monkeypatch):
    """A glaciated cell must be NaN, never the +2000 m ice surface."""
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_in_window=True)
    dz, _, _ = b.gia_delta_subset(16.5)
    assert not np.isfinite(dz).any(), "glaciated window must be entirely unknown"
    assert not (np.nan_to_num(dz, nan=0.0) > 1000.0).any()


def test_ice_free_window_is_unaffected_by_the_guard(tmp_path, monkeypatch):
    """The same fixture with no ice returns the planted Δz exactly, no NaN."""
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_in_window=False)
    dz, _, _ = b.gia_delta_subset(16.5)
    assert np.isfinite(dz).all()
    assert dz == pytest.approx(_FAKE_LOCAL_DZ_M, abs=1e-3)


def test_missing_ice_information_is_refused_rather_than_silently_warned(
    tmp_path, monkeypatch
):
    """A file with no ``sftgif`` means the guard is UNKNOWABLE — refuse.

    The retired behaviour was a ``UserWarning`` nobody read, on every epoch of
    every run.  Rule: never let an unknown pass as a number.  The opt-out has to
    be written down in code.
    """
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, with_sftgif=False)
    with pytest.raises(ValueError, match="(?i)sftgif|ice"):
        b.gia_delta_subset(16.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IceSurfaceUnmaskedWarning)
        dz, _, _ = b.gia_delta_subset(16.5, require_ice_information=False)
    assert np.isfinite(dz).all()


def test_ice_information_requirement_is_on_by_default():
    sig = inspect.signature(b.gia_delta_subset)
    assert sig.parameters["require_ice_information"].default is True


# --- requirement 2: the datum is GLOBAL, the displacement is LOCAL -----------


def test_datum_is_taken_from_the_global_field_and_displacement_from_the_subset(
    tmp_path, monkeypatch
):
    """Pinned because getting this backwards IS the retired deep-ocean bug.

    ``implied_sea_surface_offset`` must see the whole 180x360 grid; the regional
    10x17 cut must only ever reach ``solid_displacement``.
    """
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)
    shapes = {}
    real_offset, real_disp = b.implied_sea_surface_offset, b.solid_displacement

    def spy_offset(topo_t, topo_0, *a, **kw):
        shapes["offset"] = (np.asarray(topo_t).shape, np.asarray(topo_0).shape)
        return real_offset(topo_t, topo_0, *a, **kw)

    def spy_disp(topo_t, topo_0, offset, **kw):
        shapes["disp"] = (np.asarray(topo_t).shape, np.asarray(topo_0).shape)
        return real_disp(topo_t, topo_0, offset, **kw)

    monkeypatch.setattr(b, "implied_sea_surface_offset", spy_offset)
    monkeypatch.setattr(b, "solid_displacement", spy_disp)
    dz, lats, lons = b.gia_delta_subset(16.5)

    assert shapes["offset"] == ((_FAKE_N_LAT, _FAKE_N_LON),) * 2
    assert shapes["disp"] == ((lats.size, lons.size),) * 2
    assert lats.size < _FAKE_N_LAT and lons.size < _FAKE_N_LON
    # ...and the numbers prove it: the planted local Δz survives only because the
    # datum came from the tropics.  A datum taken from the window itself would
    # cancel it to 0 m (or raise for want of a far field).
    assert dz == pytest.approx(_FAKE_LOCAL_DZ_M, abs=1e-3)


def test_a_regional_field_cannot_define_the_datum(tmp_path, monkeypatch):
    """HONEST NEGATIVE: swapping the two would not silently mis-bake, it raises."""
    from paleoeurope.gia.ice6g_topo import implied_sea_surface_offset

    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)
    dz, lats, lons = b.gia_delta_subset(16.5)
    regional = np.zeros((lats.size, lons.size))
    with pytest.raises(ValueError, match="(?i)global|reference set"):
        implied_sea_surface_offset(regional, regional)


# --- requirement 3: the bake's NaN policy ------------------------------------


def test_bake_nan_policy_table_has_no_zero_option():
    assert b.BAKE_GIA_NAN_POLICIES == ("raise", "nodata")
    assert "zero" not in b.BAKE_GIA_NAN_POLICIES
    assert b.BAKE_GIA_NAN_POLICY == "raise"


def test_bake_nan_policies_match_the_correction_pipeline():
    """Drift guard: the bake and the tile pipeline must offer the same policies."""
    from paleoeurope.gia.correction_pipeline import GIA_NAN_POLICIES

    assert b.BAKE_GIA_NAN_POLICIES == GIA_NAN_POLICIES


def test_build_paleo_dem_rejects_an_unknown_policy(tmp_path, monkeypatch):
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)
    monkeypatch.setattr(b, "BS_DEM", _fake_dem(tmp_path))
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")
    for bad in ("zero", "fill", ""):
        with pytest.raises(ValueError, match="(?i)gia_nan_policy"):
            b.build_paleo_dem(16.5, gia_nan_policy=bad)


def test_a_glaciated_cell_cannot_be_baked_as_bedrock(tmp_path, monkeypatch):
    """THE definition of done for requirement 3 — default policy refuses."""
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_in_window=True)
    monkeypatch.setattr(b, "BS_DEM", _fake_dem(tmp_path))
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")
    with pytest.raises(b.UnknownGiaInBakeError) as exc:
        b.build_paleo_dem(16.5)
    msg = str(exc.value)
    assert "ice" in msg.lower()
    assert "0 m" in msg or "zero" in msg.lower()
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.tif"))


def test_unknown_gia_in_bake_error_is_a_valueerror():
    assert issubclass(b.UnknownGiaInBakeError, ValueError)


def test_nodata_policy_writes_nodata_never_a_number(tmp_path, monkeypatch, caplog):
    """The opt-out records the hole as the declared nodata, not as 0 m of Δz."""
    rasterio = pytest.importorskip("rasterio")
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_in_window=True)
    monkeypatch.setattr(b, "BS_DEM", _fake_dem(tmp_path))
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")

    with caplog.at_level(logging.WARNING):
        out = b.build_paleo_dem(16.5, gia_nan_policy="nodata")
    assert "nodata" in caplog.text.lower()

    with rasterio.open(out) as ds:
        arr = ds.read(1)
        assert ds.nodata == pytest.approx(b.BS_NODATA_M)
    # The whole window is glaciated in this fixture, so EVERY cell is unknown.
    assert (arr == np.float32(b.BS_NODATA_M)).all()
    # and specifically: not the modern DEM with 0 m of correction, and nowhere
    # near the +2000 m ice surface.
    assert not np.isclose(arr, -50.0).any()
    assert not (arr > 1000.0).any()


def test_nodata_policy_does_not_leak_the_ice_surface_across_the_margin(
    tmp_path, monkeypatch
):
    """Masking BEFORE smoothing: no finite cell may carry ice-margin contamination.

    Half the window is glaciated, half is not.  A plain σ=2 Gaussian over the
    unmasked field smears the ~+2000 m step outward; the masked smooth must not.
    """
    rasterio = pytest.importorskip("rasterio")
    # Glaciate only the top three rows of the window (>= 47.5 N).  With the 8-cell
    # rim that puts the unknown/known boundary near 43.5 N, so a tile running
    # 40-47 N genuinely straddles it.
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5, ice_lat_min=47.5)
    monkeypatch.setattr(
        b, "BS_DEM", _fake_dem(tmp_path, lat_lo=40.0, lat_hi=47.0, n=48)
    )
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")
    out = b.build_paleo_dem(16.5, gia_nan_policy="nodata")

    with rasterio.open(out) as ds:
        arr = ds.read(1)
    known = arr != np.float32(b.BS_NODATA_M)
    assert known.any() and (~known).any(), "fixture must straddle the ice margin"
    # modern DEM is -50 m and the true Δz is +7 m, so the honest answer is -43 m.
    # 1 m of tolerance leaves no room for a smeared +2000 m step.
    assert np.abs(arr[known] - (-50.0 + _FAKE_LOCAL_DZ_M)).max() < 1.0


def test_the_modern_dems_own_holes_are_not_rewritten_as_nodata(tmp_path, monkeypatch):
    """The GMRT tile carries NaN of its own, and it has always been written as NaN.

    Rewriting it to BS_NODATA_M would change what
    ``render_paleocostline_epoch(force_dem_mask=True)`` draws — NaN never compares
    below the sea level, -9999 always does — so only the GIA hole may be
    substituted.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)  # ice-free window
    dem_path = tmp_path / "holey_dem.tif"
    n = 16
    dem = np.full((n, n), -50.0, dtype=np.float32)
    dem[0, 0] = np.nan
    profile = {
        "driver": "GTiff", "height": n, "width": n, "count": 1, "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_bounds(30.0, 42.0, 35.0, 45.0, n, n),
        "nodata": None,
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(dem, 1)
    monkeypatch.setattr(b, "BS_DEM", dem_path)
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")

    out = b.build_paleo_dem(16.5)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    assert np.isnan(arr[0, 0]), "the DEM's own hole must stay NaN, not become nodata"
    assert not (arr == np.float32(b.BS_NODATA_M)).any()
    assert arr[1, 1] == pytest.approx(-50.0 + _FAKE_LOCAL_DZ_M, abs=1e-3)


def test_build_paleo_dem_masks_before_smoothing_with_the_documented_rim():
    """The smoothing must be the ice-aware one, with the 8-cell rim, σ handled once."""
    src = inspect.getsource(b.build_paleo_dem)
    assert "smooth_preserving_ice_mask" in src
    assert "SMOOTHING_RIM_CELLS" in src
    # apply_gia_delta would otherwise apply a SECOND, NaN-propagating Gaussian.
    assert "sigma_blur=0.0" in src
    assert SMOOTHING_RIM_CELLS == 8


def test_module_never_turns_an_unknown_into_a_number():
    src = Path(b.__file__).read_text(encoding="utf-8")
    for banned in ("nan_to_num", "fillna", "np.isnan(dz)] = 0"):
        assert banned not in src, banned


def test_model_grid_must_bracket_the_tile_grid(tmp_path, monkeypatch):
    """``apply_gia_delta`` fills out-of-bounds Δz with 0.0 — a zero-filled unknown.

    The bake must refuse a tile that reaches outside the model window instead of
    silently baking 0 m of GIA there.
    """
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")
    # Window is LAT_MIN-BUF..LAT_MAX+BUF = 39..49; this tile runs to 55 N.
    monkeypatch.setattr(
        b, "BS_DEM", _fake_dem(tmp_path, lat_lo=42.0, lat_hi=55.0, n=16)
    )
    with pytest.raises(ValueError, match="(?i)outside|bracket|extrapolat"):
        b.build_paleo_dem(16.5)


def test_the_real_black_sea_tile_is_bracketed_by_the_model_window():
    """Regression: the guard above must never fire for the shipped tile."""
    rasterio = pytest.importorskip("rasterio")
    if not b.BS_DEM.exists():
        pytest.skip(f"real BS DEM not available at {b.BS_DEM}")
    with rasterio.open(b.BS_DEM) as ds:
        bounds = ds.bounds
    assert b.LAT_MIN - b.BUF <= bounds.bottom and bounds.top <= b.LAT_MAX + b.BUF
    assert b.LON_MIN - b.BUF <= bounds.left and bounds.right <= b.LON_MAX + b.BUF


# --- requirement 4: the Black Sea window is bit-identical --------------------

#: ``gia_delta_subset`` output measured on the real ICE-6G files 2026-07-30,
#: BEFORE the ice fields were threaded through: sha256 of the float32 array
#: bytes, then (min, max, mean) in metres.  The window is ice-free at every
#: shipped epoch, so passing ``sftgif`` must change nothing whatsoever — these
#: hashes are the no-regression evidence for the 13 rasters about to be re-baked.
_BS_DZ_BASELINE = {
    22: ("c46d13bac6be9f2bc038017e0c1f24277a0a7fc20024fbe27b8f0fdf3eceefe1",
         1.091156, 23.259857, 8.673641),
    18: ("a388b18f8e28db225b211cf27881eb8717f9532e7746e95b3a1372d88d2186e1",
         1.325134, 24.317810, 9.325067),
    16.5: ("21feb75a613253309e1ad5c3827ab0bc2df8044f6db60f54abf7773967750860",
           1.979919, 25.218689, 10.337350),
    14.5: ("c108007092b993bce06f711ad7add482b8e2215da6ae48ba25a3d666b3fd1669",
           2.693756, 25.653961, 11.252694),
    12: ("7f81e0541873005950edaa7ade5f283729eb321aa0fed9f4b2727292e8af1cc8",
         3.178345, 22.994263, 11.266631),
    0: ("fa50ab915a7f7457913d3b731979ebc80a68972522c9dd331161cf619e84cd37",
        0.000000, 0.000000, 0.000000),
}


@pytest.mark.parametrize("epoch", sorted(_BS_DZ_BASELINE))
def test_gia_delta_subset_is_bit_identical_to_the_pre_guard_values(epoch):
    _real_ice6g_or_skip(epoch)
    want_hash, want_min, want_max, want_mean = _BS_DZ_BASELINE[epoch]
    dz, _, _ = b.gia_delta_subset(epoch)
    assert dz.dtype == np.float32
    assert np.isfinite(dz).all(), "the Black Sea window has no glaciated cell"
    got = hashlib.sha256(np.ascontiguousarray(dz).tobytes()).hexdigest()
    assert got == want_hash, (
        f"{epoch} ka Δz changed: min={dz.min():.6f} max={dz.max():.6f} "
        f"mean={dz.mean():.6f} vs baseline {want_min}/{want_max}/{want_mean}"
    )
    assert float(dz.min()) == pytest.approx(want_min, abs=5e-6)
    assert float(dz.max()) == pytest.approx(want_max, abs=5e-6)


def test_black_sea_window_is_ice_free_at_every_shipped_epoch():
    """Why ``"raise"`` is the right default for THIS bake, measured not assumed."""
    xr = pytest.importorskip("xarray")
    _real_ice6g_or_skip(16.5)
    for epoch in b.EPOCHS_KA:
        path = b.ICE6G_DIR / b._epoch_filename(epoch)
        if not path.exists():
            pytest.skip(f"real ICE-6G file not available at {path}")
        with xr.open_dataset(path) as ds:
            sub = ds.sel(
                lat=slice(b.LAT_MIN - b.BUF, b.LAT_MAX + b.BUF),
                lon=slice(b.LON_MIN - b.BUF, b.LON_MAX + b.BUF),
            )
            assert float(np.nanmax(sub["sftgif"].values)) == 0.0, epoch


# --- requirement 5: the corrected datum, and the range it now returns --------


def test_gia_delta_subset_returns_the_corrected_datum_range_at_16_5ka():
    """Δz over the BS window at 16.5 ka is [+1.98, +25.22] m.

    Under the retired deep-ocean datum the same window read
    [-27.25, -4.01] m — the whole field 29.232 m too low, which is what
    over-flooded 47 712 km² of shelf.  Sign and offset are both pinned here.
    """
    _real_ice6g_or_skip(16.5)
    dz, _, _ = b.gia_delta_subset(16.5)
    assert float(dz.min()) == pytest.approx(1.979919, abs=1e-4)
    assert float(dz.max()) == pytest.approx(25.218689, abs=1e-4)
    assert float(dz.min()) > 0.0, "the corrected datum lifts the whole window"
    # the buggy values, explicitly excluded
    assert float(dz.min()) != pytest.approx(1.979919 - 29.232, abs=0.1)


def test_the_datum_call_still_uses_no_deep_threshold():
    """The retired positional argument must not have crept back in."""
    src = inspect.getsource(b.gia_delta_subset)
    assert "deep_threshold" not in src
    assert "implied_sea_surface_offset" in src


# ===========================================================================
# TAIL 2 — the Spratt & Lisiecki curve: which file, and no silent clamping
#
# Two divergent "Spratt" CSVs shipped in this project and the bake read the
# wrong one:
#
#   data/raw/sea_level_curves/spratt_lisiecki_2016_simplified.csv   0-20 ka, 1 ka
#   data/sea_level_curves/spratt_lisiecki_2016_simplified.csv       0-26 ka, 0.5 ka
#
# NEITHER is Spratt & Lisiecki.  Both are hand-drawn round-number curves whose
# values do not appear in the published stack at any age (16 ka: -105 m in the
# file vs -98.06 m published; 10 ka: -50 m vs -24.59 m).  The published series
# ships as ``spratt_lisiecki_2016_full.csv``, a bit-exact transcription of the
# ``SeaLev_shortPC1`` column of the NOAA/WDS-Paleo distribution
# ``spratt2016.txt`` (study 19982; Spratt & Lisiecki 2016, Clim. Past 12,
# 1079-1092, doi:10.5194/cp-12-1079-2016).
#
# The 0-20 ka file also truncates below every requested epoch older than 20 ka,
# and ``get_sea_level_for_epoch`` is nearest-neighbour: at 21 ka the nearest row
# is exactly 1000 yr away, its warning tests ``> 1000``, and the clamp is
# therefore SILENT.  That is the failure mode pinned here.
# ===========================================================================

_NOAA_SPRATT_TXT = (
    b.PALEO_DATA_ROOT / "data" / "raw" / "sea_level_curves" / "spratt2016.txt"
)
_FABRICATED_SPRATT = (
    b.PALEO_DATA_ROOT
    / "data"
    / "raw"
    / "sea_level_curves"
    / "spratt_lisiecki_2016_simplified.csv"
)


def _spratt_rows(path: Path):
    return pd.read_csv(
        path, comment="#", header=None,
        names=["age_bp", "sea_level_m", "uncertainty_m"],
    )


def test_spratt_csv_points_at_the_published_series_not_a_simplified_sketch():
    assert b.SPRATT_CSV.name == "spratt_lisiecki_2016_full.csv"
    assert "simplified" not in b.SPRATT_CSV.name


def test_spratt_csv_is_bit_exact_with_the_noaa_shortpc1_column():
    """Provenance, not vibes: every row must equal the published column."""
    if not (b.SPRATT_CSV.exists() and _NOAA_SPRATT_TXT.exists()):
        pytest.skip("NOAA spratt2016.txt / curve CSV not present in this checkout")
    published = {}
    for line in _NOAA_SPRATT_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] == "age_calkaBP":
            assert parts[1] == "SeaLev_shortPC1"
            assert parts[2] == "SeaLev_shortPC1_err_sig"
            continue
        published[float(parts[0]) * 1000.0] = (float(parts[1]), float(parts[2]))

    df = _spratt_rows(b.SPRATT_CSV)
    assert len(df) == 31, "0-30 ka at 1 ka steps"
    for _, row in df.iterrows():
        want_sl, want_err = published[float(row.age_bp)]
        assert float(row.sea_level_m) == pytest.approx(want_sl, abs=1e-9), row.age_bp
        assert float(row.uncertainty_m) == pytest.approx(want_err, abs=1e-9), row.age_bp


def test_the_curve_the_bake_loads_matches_the_values_its_docstring_cites():
    """The module docstring already quoted the PUBLISHED numbers.

    "16 ka: global ~-98 m ... 10 ka: global ~-25 m" — the retired file returned
    -105 and -50 there.  Docstring and data now agree.
    """
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    assert "-98 m" in b.__doc__ and "-25 m" in b.__doc__
    assert get_sl(16.0) == pytest.approx(-98.06, abs=0.5)
    assert get_sl(10.0) == pytest.approx(-24.59, abs=0.5)


def get_sl(epoch_ka: float) -> float:
    from paleoeurope.viz.paleocostline_render import get_sea_level_for_epoch
    return get_sea_level_for_epoch(epoch_ka, b.SPRATT_CSV)


def test_no_shipped_epoch_is_silently_clamped():
    """Every epoch in EPOCHS_KA must be BRACKETED by the curve, not clamped."""
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    b.check_spratt_coverage(b.EPOCHS_KA)  # must not raise
    for epoch in b.EPOCHS_KA:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_sl(epoch)
        assert not [w for w in caught if issubclass(w.category, UserWarning)], epoch


def test_coverage_check_raises_on_the_truncated_curve_that_used_to_ship(tmp_path):
    """The 0-20 ka file clamps 22 ka to the 20 ka row and must now be rejected."""
    truncated = tmp_path / "truncated.csv"
    truncated.write_text(
        "# 0-20 ka only\n"
        + "\n".join(f"{a},{-a / 160.0:.2f},1" for a in range(0, 20_001, 1000))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(b.SprattCoverageError) as exc:
        b.check_spratt_coverage([22.0, 18.0], truncated)
    assert "22" in str(exc.value)
    b.check_spratt_coverage([18.0, 0.0], truncated)  # inside the range: fine


def test_coverage_check_rejects_a_gap_wider_than_half_a_step(tmp_path):
    """A curve that BRACKETS an epoch but only every 5 ka is still not a lookup."""
    coarse = tmp_path / "coarse.csv"
    coarse.write_text("0,0,1\n5000,-10,1\n25000,-120,1\n", encoding="utf-8")
    with pytest.raises(b.SprattCoverageError):
        b.check_spratt_coverage([16.5], coarse)


def test_spratt_present_datum_offset_is_pinned_and_declared():
    """The published stack does NOT pass through 0 m at 0 ka: it reads +8.49 m.

    1sigma there is +/-5.23 m, so 0 is well inside the published uncertainty.
    Whether to subtract this offset (making the F4 column commensurate with
    offset(t), which is 0 at 0 ka by construction) is a SCIENCE decision and is
    deliberately NOT taken here — it is only made impossible to take silently.
    """
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    assert b.spratt_present_datum_m() == pytest.approx(8.49, abs=1e-9)


def test_the_fabricated_simplified_curve_is_no_longer_reachable_from_the_bake():
    """No CODE path may name it.  Comments about it are the point — keep those."""
    code = "\n".join(
        line
        for line in Path(b.__file__).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "spratt_lisiecki_2016_simplified.csv" not in code
    assert not b.SPRATT_CSV.name.startswith("spratt_lisiecki_2016_simplified")


def test_half_step_epochs_are_interpolated_not_snapped_to_the_younger_node():
    """16.5 / 14.5 / 9.5 / 7.5 ka sit exactly between two 1-ka nodes.

    This test used to pin the OPPOSITE: ``get_sea_level_for_epoch`` was
    nearest-neighbour, ``idxmin`` broke the tie toward the YOUNGER (shallower)
    node, and this docstring said "pinned so a future switch to interpolation is a
    deliberate, visible change".  That switch was made deliberately on 2026-07-31
    (``paleocostline_render.get_sea_level_for_epoch``: the tie was systematically
    one-signed, +1.5 to +5.0 m too shallow at exactly these four epochs), so the
    pin now records the new behaviour — visibly, in the same test.

    Published nodes must still come back bit-exact.
    """
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    for half, young, old in ((16.5, 16.0, 17.0), (14.5, 14.0, 15.0), (9.5, 9.0, 10.0)):
        mid = 0.5 * (get_sl(young) + get_sl(old))
        assert get_sl(half) == pytest.approx(mid, abs=1e-9), half
        assert get_sl(half) != pytest.approx(get_sl(young)), half
    assert get_sl(16.0) == pytest.approx(-98.06, abs=1e-9)


# ===========================================================================
# TRACK A - the F2 -> F1 threshold conversion, and the comparison frames
#
# ``2026-07-30-vertical-datum-contract.md`` is a DOCUMENT; until now it was
# implemented nowhere.  The bake passed the raw F2 CSV level straight through as
# the closed-basin threshold, which is exactly the first row of the contract's
# own "what breaks" table (section 8): the DEM was fixed, the threshold was not,
# and the hero frames under-flood by ~12 m.
#
# The conversion is T(t) = level(t) + dz_solid(x_marker, t), applied at the
# THRESHOLD and never in the curve (section 8 again: editing the level column
# breaks ``validate_basin_curve``'s F2 semantics, and the level column is the
# cited literature value).
# ===========================================================================


def _real_ice6g_epochs_or_skip():
    for ep in b.EPOCHS_KA:
        path = b.ICE6G_DIR / b._epoch_filename(ep)
        if not path.exists():
            pytest.skip(f"real ICE-6G file not available at {path}")


# --- the three rules, and that a caller can see which one applied ------------


def test_threshold_rules_are_exactly_three_and_named():
    assert b.THRESHOLD_RULES == ("marker", "locus", "ocean_offset")


def test_bosphorus_row_is_converted_with_its_own_marker():
    """16.5 ka: the row's marker IS the Bosphorus sill, so the term is exact."""
    _real_ice6g_epochs_or_skip()
    t = b.pontic_threshold(_real_pontic_df(), 16.5, "gradual")
    assert t.rule == "marker"
    assert t.frame == "F2"
    assert t.level_m == pytest.approx(-30.0)
    assert t.conversion_m == pytest.approx(11.979, abs=5e-3)
    assert t.threshold_m == pytest.approx(-18.02, abs=5e-3)
    assert "29.1" in t.provenance and "41.2" in t.provenance


def test_unknown_marker_row_falls_back_to_its_documented_locus_with_a_bracket():
    """22 ka: no marker, but the seismic-wedge LOCUS is documented."""
    _real_ice6g_epochs_or_skip()
    t = b.pontic_threshold(_real_pontic_df(), 22, "gradual")
    assert t.rule == "locus"
    assert t.threshold_m == pytest.approx(-104.17, abs=0.05)
    lo, hi = t.bracket_m
    assert lo == pytest.approx(-108.90, abs=0.05)
    assert hi == pytest.approx(-101.51, abs=0.05)
    assert t.n_cells and t.n_cells > 10
    assert "wedge" in t.provenance.lower()


def test_every_locus_row_carries_its_own_p5_p95_bracket():
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    for epoch in (22, 18, 14.5, 12, 11):
        t = b.pontic_threshold(df, epoch, "gradual")
        assert t.rule == "locus", epoch
        lo, hi = t.bracket_m
        assert lo < t.threshold_m < hi, epoch


# --- the rule that matters most: no marker, no locus -> RAISE ---------------


def _unregistered_marker_source_df():
    """A curve row whose ``marker_source`` matches no prefix in the table.

    This test used to pin the refusal on the REAL 10 / 9.5 ka rows.  On 2026-08-01
    those rows got documented loci (the two papers were read; neither prints a
    coordinate, so they yielded AREAS), and the shipped curve now contains no
    unconvertible row at all.  Pinning the refusal on a synthetic row keeps the
    rule itself under test without freezing the curve's current contents — a
    future row with a new provenance must still hit it.
    """
    return pd.DataFrame(
        {
            "age_ka": [9.0, 0.0],
            "level_m_global_gradual": [-35.0, 0.0],
            "level_m_global_catastrophic": [-100.0, 0.0],
            "marker_lon": ["UNKNOWN", "NOT_APPLICABLE"],
            "marker_lat": ["UNKNOWN", "NOT_APPLICABLE"],
            "marker_source": [
                "UNKNOWN:a_provenance_nobody_has_registered_a_locus_for",
                "NOT_APPLICABLE_F4_Spratt",
            ],
        }
    )


def test_a_row_with_neither_marker_nor_locus_raises_rather_than_substituting():
    _real_ice6g_epochs_or_skip()
    df = _unregistered_marker_source_df()
    for scenario in b.SCENARIOS:
        with pytest.raises(b.UnconvertibleThresholdError) as exc:
            b.pontic_threshold(df, 9.0, scenario)
        assert "marker" in str(exc.value).lower(), scenario
        assert "locus" in str(exc.value).lower(), scenario


def test_the_bosphorus_cell_is_never_substituted_for_an_unlocated_row():
    """The audited failure mode: that substitution is systematically one-signed."""
    _real_ice6g_epochs_or_skip()
    with pytest.raises(b.UnconvertibleThresholdError):
        b.pontic_threshold(_unregistered_marker_source_df(), 9.0, "gradual")
    src = inspect.getsource(b.pontic_threshold)
    assert "over-flood" in src.lower() or "one-signed" in src.lower()


def test_unconvertible_threshold_error_is_a_valueerror():
    assert issubclass(b.UnconvertibleThresholdError, ValueError)


# --- F4 rows: the reconnected basin IS the ocean, so -offset(t) is exact -----


def test_f4_rows_threshold_on_minus_offset_not_on_the_curve_level():
    """5 / 2 / 0 ka: an INTENDED-OCEAN threshold, switched to F3 per contract 2."""
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    for epoch in (5, 2, 0):
        t = b.pontic_threshold(df, epoch, "gradual")
        assert t.rule == "ocean_offset", epoch
        assert t.frame == "F4"
        assert t.threshold_m == pytest.approx(-b.sea_surface_offset(epoch), abs=1e-6)
    assert b.pontic_threshold(df, 5, "gradual").threshold_m != pytest.approx(-10.0)


def test_a_mixed_row_is_resolved_per_scenario_not_per_row():
    """9 ka: gradual is already marine (F4); catastrophic is still a deep F2 lake."""
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    grad = b.pontic_threshold(df, 9, "gradual")
    cat = b.pontic_threshold(df, 9, "catastrophic")
    assert (grad.rule, grad.frame) == ("ocean_offset", "F4")
    # The catastrophic column used to RAISE here; since 2026-08-01 it converts
    # over Lericolais's dune/terrace locus.  The point of the test is unchanged:
    # one row, two frames, resolved per COLUMN and never per row.
    assert (cat.rule, cat.frame) == ("locus", "F2")
    assert "nw_outer_shelf_lericolais_dune_terrace" in cat.provenance
    assert cat.threshold_m < grad.threshold_m - 50.0


def test_offset_is_zero_at_present_by_construction():
    _real_ice6g_epochs_or_skip()
    assert b.sea_surface_offset(0) == pytest.approx(0.0, abs=1e-9)


# --- the conversion is at the THRESHOLD, never in the curve -----------------


def test_the_level_column_is_never_rewritten_by_the_conversion():
    """Contract section 8: converting in the curve breaks validate_basin_curve."""
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    before = df.copy(deep=True)
    for epoch, (grad, cat) in _HISTORICAL_PONTIC_LEVELS.items():
        for scenario, want in (("gradual", grad), ("catastrophic", cat)):
            try:
                t = b.pontic_threshold(df, epoch, scenario)
            except b.UnconvertibleThresholdError:
                continue
            assert t.level_m == pytest.approx(want), (epoch, scenario)
    pd.testing.assert_frame_equal(df, before)


def test_the_conversion_is_additive_for_every_f2_row():
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    for epoch in (22, 18, 16.5, 14.5, 12, 11):
        t = b.pontic_threshold(df, epoch, "gradual")
        assert t.frame == "F2"
        assert t.threshold_m == pytest.approx(t.level_m + t.conversion_m, abs=1e-9)
        assert t.conversion_m > 0.0, epoch


def test_threshold_guard_is_not_an_assert_statement():
    src = inspect.getsource(b.pontic_threshold) + inspect.getsource(
        b.sea_surface_offset
    )
    assert "assert " not in src


# --- the eustatic series is 0-referenced ------------------------------------


def test_eustatic_series_is_zero_referenced_to_present():
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    assert b.eustatic_level_m(0) == pytest.approx(0.0, abs=1e-9)
    # ...and the published stack itself still reads +8.49 m there.
    assert b.spratt_present_datum_m() == pytest.approx(8.49, abs=1e-9)


def test_zero_referencing_moves_the_lgm_frame_to_the_documented_value():
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    # 22 ka is a published node: -125.82 - 8.49.
    assert b.eustatic_level_m(22) == pytest.approx(-134.31, abs=1e-2)
    # 16.5 ka is a half-step, so it inherits the interpolation the render module
    # switched to on 2026-07-31: mean(-98.06, -107.30) - 8.49.
    assert b.eustatic_level_m(16.5) == pytest.approx(-111.17, abs=1e-2)


# --- the deliberately-wrong frames stay eustatic ----------------------------


def test_the_wrong_frames_still_threshold_on_the_eustatic_curve():
    """They are the wrong-physics demonstration; making them right kills them."""
    src = inspect.getsource(b.main)
    wrong = src[src.index("WRONG"):]
    assert "eustatic_level_m" in wrong
    assert "sea_surface_offset" not in wrong


def test_the_wrong_frames_call_site_names_both_documents():
    src = inspect.getsource(b.main)
    wrong = src[src.index("WRONG"):]
    low = wrong.lower()
    assert "2026-07-30-vertical-datum-contract" in low
    assert "verifier" in low


# --- provenance reaches the raster and the log ------------------------------


def test_build_paleo_dem_writes_the_threshold_provenance_into_the_raster(
    tmp_path, monkeypatch
):
    rasterio = pytest.importorskip("rasterio")
    _fake_ice6g_dir(tmp_path, monkeypatch, epoch=16.5)
    monkeypatch.setattr(b, "BS_DEM", _fake_dem(tmp_path))
    monkeypatch.setattr(b, "PALEO_TIF_DIR", tmp_path / "out")
    out = b.build_paleo_dem(
        16.5, provenance={"threshold_gradual": "-18.02 m rule=marker"}
    )
    with rasterio.open(out) as ds:
        tags = ds.tags()
    assert tags["threshold_gradual"] == "-18.02 m rule=marker"


def test_every_threshold_reports_its_rule_in_one_line():
    _real_ice6g_epochs_or_skip()
    t = b.pontic_threshold(_real_pontic_df(), 16.5, "gradual")
    assert t.rule in b.THRESHOLD_RULES
    assert t.rule in t.provenance


# --- the ignored primary input: drift must be catchable ---------------------

#: sha256 over the PARSED, canonicalised rows of ``spratt_lisiecki_2016_full.csv``
#: (age,level,err at 6 dp, ascending, newline-joined).  ``spratt2016.txt`` lives
#: under the same gitignored ``data/raw/`` tree, so the row-by-row provenance test
#: above SKIPS in any checkout that does not happen to carry it - and skipping is
#: not catching.  This pin needs only the one file the bake actually reads.
_SPRATT_CANONICAL_SHA256 = (
    "9ef3a77f96e5cce9e202ca50c483c14923d13b271fad5ca234627a722529b484"
)


def test_spratt_csv_content_is_pinned_without_needing_the_noaa_file():
    if not b.SPRATT_CSV.exists():
        pytest.skip("curve CSV not present in this checkout")
    df = _spratt_rows(b.SPRATT_CSV).sort_values("age_bp")
    canon = "\n".join(
        f"{float(r.age_bp):.6f},{float(r.sea_level_m):.6f},{float(r.uncertainty_m):.6f}"
        for r in df.itertuples()
    )
    got = hashlib.sha256(canon.encode("ascii")).hexdigest()
    assert got == _SPRATT_CANONICAL_SHA256


# ===========================================================================
# FIX 1 - the 10.5-7.5 ka window's LOCI are registered, so the six frames that
# used to refuse now convert.
#
# Both papers behind that window were acquired and read in full on 2026-08-01
# (Giosan, Filip & Constantinescu 2009 QSR 28:1-6; Lericolais, Bulois, Gillet &
# Guichard 2009 GPC 66:65-75).  NEITHER PRINTS A COORDINATE - confirmed by
# reading - so marker_lon/marker_lat stay the literal UNKNOWN and the fallback
# stays rule="locus", never rule="marker".  What the read bought is two
# DOCUMENTED AREAS, georeferenced off each paper's own printed graticule.
# ===========================================================================


def test_the_two_papers_loci_are_registered_with_their_documented_boxes():
    lericolais = b._LOCI["nw_outer_shelf_lericolais_dune_terrace"]
    assert lericolais.lon_deg == (29.5, 31.0)
    assert lericolais.lat_deg == (43.5, 44.5)
    assert lericolais.modern_depth_m == (-115.0, -90.0)

    giosan = b._LOCI["danube_delta_separation_surface"]
    assert giosan.lon_deg == (28.8, 30.0)
    assert giosan.lat_deg == (44.6, 45.4)
    assert giosan.modern_depth_m == (-45.0, 5.0)


def test_every_registered_marker_source_prefix_points_at_a_locus_that_exists():
    """The silent-reintroduction failure mode: a prefix aimed at a missing locus.

    ``_locus_for_marker_source`` returns the NAME, and ``locus_solid_displacement``
    only discovers that the name is unknown when a row actually reaches it - i.e.
    at bake time, on one epoch, as a ``ValueError`` that reads nothing like "you
    renamed a locus".  Check the whole table statically instead.
    """
    for prefix, locus_name in b._MARKER_SOURCE_LOCI:
        assert locus_name in b._LOCI, (prefix, locus_name)
    # ...and every locus name is its own key, so a lookup cannot land elsewhere.
    for key, locus in b._LOCI.items():
        assert locus.name == key


def test_the_four_new_prefixes_resolve_to_the_documented_loci():
    resolve = b._locus_for_marker_source
    assert (
        resolve("UNKNOWN:lericolais2009_nw_outer_shelf_-100m_dune_terrace_LOCUS_ONLY")
        == "nw_outer_shelf_lericolais_dune_terrace"
    )
    assert (
        resolve("UNKNOWN:giosan2009_danube_delta_separation_surface_LOCUS_ONLY")
        == "danube_delta_separation_surface"
    )
    assert (
        resolve("UNKNOWN:bracket_midpoint_held_flat_from_the_10.9ka_anchor(...)")
        == "basin_shelf_band"
    )
    assert (
        resolve("UNKNOWN:gradual_tail_ramp_toward_the_9.4ka_reconnection(...)")
        == "basin_shelf_band"
    )


def test_each_new_locus_why_states_its_georeferencing_honestly():
    lericolais = b._LOCI["nw_outer_shelf_lericolais_dune_terrace"].why.lower()
    # why it is WIDER than the pseudo-3D mosaic the paper actually publishes
    assert "mosaic" in lericolais
    assert "graticule" in lericolais
    giosan = b._LOCI["danube_delta_separation_surface"].why.lower()
    assert "0.05" in giosan  # the stated georeferencing precision, in degrees
    assert "graticule" in giosan


def test_the_lericolais_locus_tracks_the_single_cell_under_the_mosaic():
    """The box is widened because 1 cell < locus_solid_displacement's minimum of 5.

    Widening is only defensible if it does not MOVE the answer, so measure the
    difference against the one 10-arcmin cell the 20x5 km mosaic actually falls
    in (30.1667 E / 43.9167 N).  Sub-0.12 m at every epoch that uses it.
    """
    _real_ice6g_epochs_or_skip()
    for epoch, want in ((10.0, -0.114), (9.5, -0.110), (9.0, -0.103), (7.5, -0.080)):
        median, _bracket, n_cells = b.locus_solid_displacement(
            epoch, "nw_outer_shelf_lericolais_dune_terrace"
        )
        cell, _block = b.marker_solid_displacement(epoch, 30.1667, 43.9167)
        assert n_cells == 7, epoch
        assert median - cell == pytest.approx(want, abs=5e-3), epoch


def test_the_six_frames_that_used_to_refuse_now_convert():
    """Each with rule="locus" - never "marker": no coordinate was ever printed."""
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    expected = {
        (10.0, "gradual"): ("basin_shelf_band", -68.26, (-72.31, -64.64)),
        (10.0, "catastrophic"): ("basin_shelf_band", -68.26, (-72.31, -64.64)),
        (9.5, "gradual"): ("basin_shelf_band", -29.02, (-32.80, -25.68)),
        (9.5, "catastrophic"): (
            "nw_outer_shelf_lericolais_dune_terrace",
            -85.21,
            (-86.03, -84.60),
        ),
        (9.0, "catastrophic"): (
            "nw_outer_shelf_lericolais_dune_terrace",
            -86.55,
            (-87.30, -85.99),
        ),
        (7.5, "catastrophic"): (
            "nw_outer_shelf_lericolais_dune_terrace",
            -90.03,
            (-90.59, -89.61),
        ),
    }
    for (epoch, scenario), (locus, want_t, (want_lo, want_hi)) in expected.items():
        t = b.pontic_threshold(df, epoch, scenario)
        assert t.rule == "locus", (epoch, scenario)
        assert t.frame == "F2", (epoch, scenario)
        assert locus in t.provenance, (epoch, scenario)
        assert t.threshold_m == pytest.approx(want_t, abs=0.05), (epoch, scenario)
        lo, hi = t.bracket_m
        assert lo == pytest.approx(want_lo, abs=0.05), (epoch, scenario)
        assert hi == pytest.approx(want_hi, abs=0.05), (epoch, scenario)


def test_the_two_10ka_columns_get_the_SAME_threshold():
    """Do not "fix" this.  Their level is -80 m in BOTH columns, and -80 is
    neither Giosan's -30 nor Lericolais's -100: it is the 10.9 ka central value
    held flat, itself the midpoint of BLVK9814's measured -55 mbsl ceiling and the
    -100 m lowstand, i.e. from no source.  Charging gradual to Giosan and
    catastrophic to Lericolais would MANUFACTURE a ~3.9 m difference out of an
    attribution that neither column's -80 m supports.
    """
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    grad = b.pontic_threshold(df, 10.0, "gradual")
    cat = b.pontic_threshold(df, 10.0, "catastrophic")
    assert grad.level_m == cat.level_m == pytest.approx(-80.0)
    assert grad.threshold_m == pytest.approx(cat.threshold_m, abs=1e-12)
    assert grad.bracket_m == cat.bracket_m


def test_nothing_is_unconvertible_any_more_so_the_plain_bake_does_not_abort():
    _real_ice6g_epochs_or_skip()
    _thresholds, unconvertible = b.preflight_thresholds(_real_pontic_df())
    assert unconvertible == {}


def test_registering_the_loci_left_the_twenty_already_baked_thresholds_untouched():
    """Bit-for-bit: a new locus must not perturb a frame it does not govern."""
    _real_ice6g_epochs_or_skip()
    df = _real_pontic_df()
    for epoch, want in (
        (22, -104.17),
        (18, -98.23),
        (16.5, -18.02),
        (14.5, -18.57),
        (12, -46.13),
        (11, -65.25),
        (9, -3.74),
        (7.5, 2.49),
        (5, 2.33),
        (2, 0.63),
        (0, 0.00),
    ):
        t = b.pontic_threshold(df, epoch, "gradual")
        assert t.threshold_m == pytest.approx(want, abs=0.005), epoch


# ===========================================================================
# T3 — the two curve-reading guards, and the band on the Pontic side
#
# Both guards were LATENT SUBSTITUTIONS, and neither changes a number this bake
# has ever produced: all 13 shipped epochs land on exact curve rows and on exact
# ICE-6G epochs.  What changes is what happens OFF that grid, which is where the
# animation is going.
# ===========================================================================


def _real_curve_or_skip():
    if not b.PONTIC_CSV.exists():
        pytest.skip(f"real Pontic curve not available: {b.PONTIC_CSV}")
    return pd.read_csv(b.PONTIC_CSV, comment="#")


# --- guard 1: the nearest-age lookup is bounded -----------------------------


def test_basin_level_refuses_a_far_row():
    """16.75 ka used to silently take -58.3 m; it now refuses.

    The 17.0 and 16.5 ka rows are EQUIDISTANT from 16.75 ka and read -58.3 m and
    -30.0 m, so ``idxmin`` returned whichever came first in the DataFrame — 28.3 m
    of threshold decided by row order.  Neither is defensible: the two rows are not
    even the same kind of number (a five-study seismic-wedge lowstand vs the modern
    Bosphorus sill depth used as an overflow proxy), so there is nothing to pick
    between and nothing to interpolate.
    """
    df = _real_curve_or_skip()
    # the substitution that used to happen, still visible in the data
    assert float(df.loc[df["age_ka"] == 17.0, "level_m_global_gradual"].iloc[0]) == -58.3
    assert float(df.loc[df["age_ka"] == 16.5, "level_m_global_gradual"].iloc[0]) == -30.0
    with pytest.raises(b.CurveAgeGapError, match="no row at 16.75 ka"):
        b.basin_level(df, 16.75, "gradual")
    # ...and the epochs the curve DOES carry are unaffected
    assert b.basin_level(df, 17.0, "gradual") == pytest.approx(-58.3)
    assert b.basin_level(df, 16.5, "gradual") == pytest.approx(-30.0)


def test_the_age_tolerance_cannot_reach_a_second_row():
    """0.01 ka is the curves' own age resolution and is below half their spacing.

    The tightest ADJACENT spacing is 0.05 ka in the Caspian curve and 0.10 ka in
    the Pontic one, so the tolerance can never make the lookup ambiguous by a
    hair — which is why it can be this tight without being arbitrary.
    """
    assert b._CURVE_AGE_TOL_KA == 0.01
    df = _real_curve_or_skip()
    ages = np.sort(pd.to_numeric(df["age_ka"]).to_numpy(dtype=float))
    spacing = np.diff(ages)
    assert float(spacing[spacing > 0].min()) >= 2 * b._CURVE_AGE_TOL_KA


def test_a_tiny_age_offset_is_still_the_same_row():
    """Float noise must not become a refusal: 16.5 +/- 1e-9 is the 16.5 ka row."""
    df = _real_curve_or_skip()
    assert b.basin_level(df, 16.5 + 1e-9, "gradual") == pytest.approx(-30.0)
    assert b.basin_level(df, 16.5 - 1e-9, "gradual") == pytest.approx(-30.0)


def test_duplicate_ages_that_disagree_are_refused_rather_than_ordered():
    """The other way row order used to decide a level."""
    dup = pd.DataFrame(
        {
            "age_ka": [9.0, 9.0, 0.0],
            "level_m_global_gradual": [-35.0, -60.0, 0.0],
            "level_m_global_catastrophic": [-100.0, -100.0, 0.0],
        }
    )
    with pytest.raises(b.CurveAgeGapError, match="disagree"):
        b.basin_level(dup, 9.0, "gradual")
    # identical duplicates are not a conflict and still resolve
    same = dup.copy()
    same.loc[1, "level_m_global_gradual"] = -35.0
    assert b.basin_level(same, 9.0, "gradual") == pytest.approx(-35.0)


def test_the_age_guard_is_not_an_assert_statement():
    """``assert`` is stripped under python -O; this guard protects a threshold."""
    src = inspect.getsource(b._curve_row_index) + inspect.getsource(b.basin_level)
    assert "assert " not in src


def test_every_shipped_epoch_still_lands_on_an_exact_row():
    """The guard costs this bake nothing, and that is asserted rather than hoped."""
    df = _real_curve_or_skip()
    ages = pd.to_numeric(df["age_ka"]).to_numpy(dtype=float)
    for epoch in b.EPOCHS_KA:
        assert float(np.abs(ages - epoch).min()) == 0.0, epoch


# --- guard 2: off-grid ICE-6G epochs, tagged rather than fatal ---------------


def test_pontic_quarter_ka_epoch_resolves():
    """14.7 ka: a row the curve HAS, at an epoch ICE-6G does not ship.

    RECONCILED FROM THE BRIEF, which named 16.75 ka for this test.  16.75 ka
    cannot be the case: the Pontic curve has no row there either (its grid is the
    same 0.5 ka as ICE-6G's), so the honest answer at 16.75 is the refusal the
    test above pins, not a threshold.  14.7 and 7.15 ka are the epochs where the
    curve genuinely carries a level and only the GIA file is missing — which is
    exactly the defect item 9 describes, and there are 2 such rows, not 8.

    The value is a cross-check as well as a smoke test: -17.278 m reproduces the
    contract's own sec.5a-bis per-row figure of -17.28 m for 14.7 ka, measured
    independently and by a different route.
    """
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(14.5)).exists():
        pytest.skip("real ICE-6G not available")
    assert 14.7 not in b.ice6g_ages_ka()
    assert float(np.abs(pd.to_numeric(df["age_ka"]) - 14.7).min()) == 0.0

    t = b.pontic_threshold(df, 14.7, "gradual")
    assert t.rule == "marker"
    assert t.threshold_m == pytest.approx(-17.28, abs=0.01)
    assert t.ice6g_interpolated is True
    assert "linear in time between ICE-6G 14.5 and 15 ka" in t.ice6g_time_note
    assert t.ice6g_time_note in t.provenance


def test_an_on_grid_epoch_is_not_tagged_as_interpolated():
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(16.5)).exists():
        pytest.skip("real ICE-6G not available")
    t = b.pontic_threshold(df, 16.5, "gradual")
    assert t.ice6g_interpolated is False
    assert "no time interpolation" in t.ice6g_time_note
    assert t.threshold_m == pytest.approx(-18.02, abs=5e-3)


def test_the_f4_rule_also_survives_an_off_grid_epoch():
    """7.15 ka is F4 and off the ICE-6G grid, so ``-offset(t)`` had to blend too."""
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(7.0)).exists():
        pytest.skip("real ICE-6G not available")
    t = b.pontic_threshold(df, 7.15, "gradual")
    assert t.rule == "ocean_offset"
    assert t.ice6g_interpolated is True
    blend = 0.7 * b.sea_surface_offset(7.0) + 0.3 * b.sea_surface_offset(7.5)
    assert t.threshold_m == pytest.approx(-blend, abs=1e-9)


def test_the_bracket_blend_is_linear_and_exact_at_its_endpoints():
    if not (b.ICE6G_DIR / b._epoch_filename(16.5)).exists():
        pytest.skip("real ICE-6G not available")
    assert b._ice6g_bracket(16.5) == (16.5, 16.5, 0.0)
    assert b.ice6g_is_interpolated(16.5) is False
    lo, hi, w = b._ice6g_bracket(16.25)
    assert (lo, hi) == (16.0, 16.5) and w == pytest.approx(0.5)
    assert b.ice6g_is_interpolated(16.25) is True
    mid = 0.5 * (b.sea_surface_offset(16.0) + b.sea_surface_offset(16.5))
    assert b.sea_surface_offset(16.25) == pytest.approx(mid, abs=1e-9)


def test_the_ice6g_age_cache_is_keyed_on_the_directory():
    """A module-level path that tests repoint must not be cached once, globally.

    A single-slot cache would answer a question about one ICE-6G distribution with
    the contents of another — the same class of error as reading a level off the
    wrong row.
    """
    assert b._ice6g_ages_in.cache_info() is not None
    src = inspect.getsource(b.ice6g_ages_ka)
    assert "_ice6g_ages_in(ICE6G_DIR)" in src


# --- the ROW ENVELOPE: read, converted with the central's own term, or refused --


def test_envelope_columns_are_the_two_the_curves_carry():
    assert b.ENVELOPE_COLUMNS == ("unc_low", "unc_high")
    assert b.ENVELOPE_RULES == (
        "envelope_as_carried_in_the_curve",
        "not_applicable_F4",
        "refused",
    )


def test_the_provenance_constant_no_longer_asserts_an_absence():
    """C3.  ``UNKNOWN:not_recorded_in_the_curve`` was a FALSE provenance claim.

    The curves have no ``unc_source`` / ``unc_interp_flag`` COLUMN — that is what
    the constant may say, and all it may say.  The ``note`` column DOES record the
    edges' sources in prose on the rows the headline number rests on, so a blanket
    "not recorded in the curve" asserted an absence the file contradicts, which is
    the same class of error as substituting a number.
    """
    assert b.ENVELOPE_PROVENANCE_NO_COLUMN == (
        "NOT_MACHINE_READABLE:the curve has no unc_source/unc_interp_flag column"
    )
    assert "not_recorded_in_the_curve" not in b.ENVELOPE_PROVENANCE_NO_COLUMN
    assert not hasattr(b, "BAND_PROVENANCE")
    # The retired literal may still be QUOTED in a comment or a docstring recording
    # why it was withdrawn — that is the audit trail.  What it may not be is
    # EMITTED, so the scan is over string literals that are not docstrings: no
    # string this module ships to a reader may carry the false claim.
    tree = ast.parse(Path(b.__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and (doc := ast.get_docstring(node, clean=False)) is not None:
            docstrings.add(doc)
    emitted = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "UNKNOWN:not_recorded_in_the_curve" in node.value
        and node.value not in docstrings
    ]
    assert not emitted, emitted
    env = b.basin_level_envelope(_envelope_df(-40.0, -30.0), 9.0)
    assert "not_recorded_in_the_curve" not in env.provenance


def test_the_note_column_records_edge_provenance_on_the_caspian_rows_that_matter():
    """C3, measured on the real file rather than asserted.

    The 15.0 ka note names ``unc_low`` and where it comes from; the 14.0 ka note
    records a carried conflict on the same edge.  Both are rows the spill window
    depends on.  A provenance statement that says "not recorded" over these is
    false, so :func:`envelope_note_provenance` reports per row.
    """
    csv = (
        b.PALEO_DATA_ROOT
        / "docs"
        / "superpowers"
        / "research"
        / "2026-07-28-caspian-khvalynian-level-curve.csv"
    )
    if not csv.exists():
        pytest.skip("Caspian curve not available")
    df = pd.read_csv(csv, comment="#")
    named = sorted(
        float(df.loc[i, "age_ka"])
        for i in df.index
        if b.envelope_note_provenance(df, i)[0]
    )
    assert named == [14.0, 15.0, 17.4, 18.0, 20.0]
    in_note, statement = b.envelope_note_provenance(
        df, b._curve_row_index(df, 15.0)
    )
    assert in_note
    assert "unc_low" in statement and "do not report the provenance as absent" in statement
    assert b.ENVELOPE_PROVENANCE_NO_COLUMN in statement
    # ... and a row whose note is silent says so, without claiming the curve is.
    quiet, quiet_statement = b.envelope_note_provenance(
        df, b._curve_row_index(df, 16.5)
    )
    assert not quiet
    assert "Other rows of the same curve do record theirs in prose" in quiet_statement


def test_the_caspian_pair_is_an_envelope_over_every_level_column():
    """C1, the headline, measured: one pair per ROW bracketing all six columns.

    This is what makes it an INTER-SCHOOL SCENARIO ENVELOPE rather than the
    uncertainty of any one scenario — at 16.0 ka it brackets ``defensible`` +35.0
    and ``maximal`` +48.0 at the same time.
    """
    csv = (
        b.PALEO_DATA_ROOT
        / "docs"
        / "superpowers"
        / "research"
        / "2026-07-28-caspian-khvalynian-level-curve.csv"
    )
    if not csv.exists():
        pytest.skip("Caspian curve not available")
    df = pd.read_csv(csv, comment="#")
    columns = b.level_columns(df)
    assert len(columns) == 6
    for i in df.index:
        lo = float(df.loc[i, "unc_low"])
        hi = float(df.loc[i, "unc_high"])
        bracketed, scope = b.envelope_scope(df, i, lo, hi)
        assert bracketed == columns, float(df.loc[i, "age_ka"])
        assert scope.startswith("INTER-SCHOOL SCENARIO ENVELOPE")
        assert "NOT the uncertainty of any single scenario column" in scope
    row = b._curve_row_index(df, 16.0)
    assert (float(df.loc[row, "unc_low"]), float(df.loc[row, "unc_high"])) == (
        22.0,
        50.0,
    )
    assert float(df.loc[row, "level_m_global_defensible"]) == 35.0
    assert float(df.loc[row, "level_m_global_maximal"]) == 48.0


def test_the_pontic_pair_is_NOT_an_envelope_over_both_its_columns():
    """The scope is a MEASUREMENT, and here it measures the other answer.

    Same code, same two column names, different curve: at 9.0 / 8.5 / 8.0 / 7.5 ka
    the Pontic pair was written for ``gradual`` and does not contain
    ``catastrophic``, so the scope sentence must NOT claim a shared envelope.
    """
    df = _real_curve_or_skip()
    for epoch in (9.0, 8.5, 8.0, 7.5):
        i = b._curve_row_index(df, epoch)
        lo, hi = float(df.loc[i, "unc_low"]), float(df.loc[i, "unc_high"])
        bracketed, scope = b.envelope_scope(df, i, lo, hi)
        assert bracketed == ("level_m_global_gradual",), epoch
        assert scope.startswith("ROW-LEVEL PAIR, NOT AN ENVELOPE"), epoch
        assert "brackets 1 of 2 level columns" in scope, epoch
    # ... and where it does contain both, it says so.
    i = b._curve_row_index(df, 16.5)
    _cols, scope = b.envelope_scope(
        df, i, float(df.loc[i, "unc_low"]), float(df.loc[i, "unc_high"])
    )
    assert scope.startswith("INTER-SCHOOL SCENARIO ENVELOPE")


def test_the_envelope_object_carries_no_scenario_of_its_own():
    """C1 as an API fact: there is no ``LevelEnvelope.scenario`` to stamp.

    ``scenario`` selects only the column whose central is containment-CHECKED, and
    the field that records that is named for what it is.  A field called
    ``scenario`` on a row-level object is what let the report assert a per-scenario
    band in the first place.
    """
    fields = {f.name for f in dataclasses.fields(b.LevelEnvelope)}
    assert "scenario" not in fields
    assert "containment_checked_against" in fields
    assert {"scope", "bracketed_columns", "all_level_columns"} <= fields
    env = b.basin_level_envelope(_envelope_df(-40.0, -30.0), 9.0, "gradual")
    assert env.containment_checked_against == "level_m_global_gradual"


def test_band_that_does_not_contain_its_central_is_refused():
    # NAME KEPT VERBATIM from the phase brief's C-BAND acceptance table, even
    # though the object is now called an envelope: that table is the acceptance
    # contract and a renamed assertion is an unrun one.  The FIELDS are renamed;
    # this identifier is a handle the gate holds.
    """The Pontic catastrophic column at 9.0 / 8.5 / 8.0 / 7.5 ka.

    The pair there was written for the GRADUAL column; the catastrophic level is
    -100 m, 60-83 m below it.  Two of the four (9.0 and 7.5 ka) are BAKED epochs,
    so this is not a hypothetical.  The frame is still produced — the Pontic
    threshold does not depend on the envelope — and the refusal is recorded instead
    of raised, with all four numeric fields ``None``.
    """
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(9.0)).exists():
        pytest.skip("real ICE-6G not available")
    want = {
        9.0: "level_m_global_catastrophic = -100.0, envelope [-40.0, -30.0]",
        8.5: "level_m_global_catastrophic = -100.0, envelope [-35.0, -25.0]",
        8.0: "level_m_global_catastrophic = -100.0, envelope [-30.0, -20.0]",
        7.5: "level_m_global_catastrophic = -100.0, envelope [-27.0, -17.0]",
    }
    for epoch, detail in want.items():
        t = b.pontic_threshold(df, epoch, "catastrophic")
        assert t.envelope_status == (
            "REFUSED: envelope does not contain the central it was checked against "
            f"({detail})"
        ), epoch
        assert t.envelope_rule == "refused", epoch
        assert (
            t.envelope_lo_m,
            t.envelope_hi_m,
            t.threshold_envelope_lo_m,
            t.threshold_envelope_hi_m,
        ) == (None, None, None, None), epoch
        # the frame itself is still produced, on the central line
        assert t.threshold_m == pytest.approx(t.level_m + t.conversion_m)
        assert t.envelope_status in t.provenance, epoch
        assert "REFUSED" in t.envelope_scope, epoch


def test_the_envelope_status_is_per_column_not_per_row():
    """The two columns of the same row land on two DIFFERENT statuses.

    Measured, and it explains the defect rather than merely recording it: by 8.0 ka
    the gradual column has already reconnected, so it is F4 and its threshold is
    ``-offset(t)`` — nothing there uses the level's envelope at all.  The pair
    sitting in those rows brackets that gradual line ([-30, -20] at 8.0 ka), and the
    only column still asking for one is the catastrophic one, which is a deep lake
    at -100 m.  Hence a pair 70 m away from the central it is paired with.
    """
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(8.0)).exists():
        pytest.skip("real ICE-6G not available")
    grad = b.pontic_threshold(df, 8.0, "gradual")
    cat = b.pontic_threshold(df, 8.0, "catastrophic")
    assert grad.frame == "F4" and grad.envelope_rule == "not_applicable_F4"
    assert cat.frame == "F2" and cat.envelope_rule == "refused"
    row = df.loc[b._curve_row_index(df, 8.0)]
    assert (float(row["unc_low"]), float(row["unc_high"])) == (-30.0, -20.0)
    assert float(row["level_m_global_gradual"]) == -25.0
    assert float(row["level_m_global_catastrophic"]) == -100.0


def test_an_f4_row_gets_four_none_envelope_fields_and_its_own_rule():
    """Converting the LEVEL's envelope on a row whose threshold is -offset(t) would
    put a bar on a number nothing uses."""
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(5.0)).exists():
        pytest.skip("real ICE-6G not available")
    t = b.pontic_threshold(df, 5.0, "gradual")
    assert t.rule == "ocean_offset" and t.frame == "F4"
    assert t.envelope_rule == "not_applicable_F4"
    assert (
        t.envelope_lo_m,
        t.envelope_hi_m,
        t.threshold_envelope_lo_m,
        t.threshold_envelope_hi_m,
    ) == (None, None, None, None)
    assert "NOT_APPLICABLE" in t.envelope_status
    assert "NOT_APPLICABLE" in t.envelope_scope


def test_both_envelope_edges_are_converted_with_the_central_s_own_term():
    """Contract sec.8, edge for edge — one conversion term for all three."""
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(16.5)).exists():
        pytest.skip("real ICE-6G not available")
    for epoch in (22, 18, 16.5, 14.5, 12):
        t = b.pontic_threshold(df, epoch, "gradual")
        assert t.envelope_rule == "envelope_as_carried_in_the_curve", epoch
        assert t.threshold_envelope_lo_m - t.envelope_lo_m == pytest.approx(
            t.conversion_m
        ), epoch
        assert t.threshold_m - t.level_m == pytest.approx(t.conversion_m), epoch
        assert t.threshold_envelope_hi_m - t.envelope_hi_m == pytest.approx(
            t.conversion_m
        ), epoch
        assert b.ENVELOPE_PROVENANCE_NO_COLUMN in t.envelope_provenance
        assert t.envelope_scope, epoch


def _envelope_df(low, high, level=-35.0):
    return pd.DataFrame(
        {
            "age_ka": [9.0, 0.0],
            "level_m_global_gradual": [level, 0.0],
            "level_m_global_catastrophic": [-100.0, 0.0],
            "unc_low": [low, -5.0],
            "unc_high": [high, 0.0],
        }
    )


def test_a_missing_nan_or_inverted_envelope_edge_raises():
    """Three ways an envelope fails to be one.  None of them is repaired."""
    with pytest.raises(b.MissingEnvelopeError, match="carries no"):
        b.basin_level_envelope(
            _envelope_df(-40.0, -30.0).drop(columns=["unc_low"]), 9.0
        )
    with pytest.raises(b.MissingEnvelopeError, match="not a number"):
        b.basin_level_envelope(_envelope_df(np.nan, -30.0), 9.0)
    with pytest.raises(b.MissingEnvelopeError, match="inverted"):
        b.basin_level_envelope(_envelope_df(-30.0, -40.0), 9.0)


def test_a_degenerate_envelope_is_refused_rather_than_reported_as_width_zero():
    """C6.  ``unc_low == unc_high == central`` passed every check and reported 0.0 m.

    That is a band asserting CERTAINTY: it forces low == central == high, so
    ``verdict_is_envelope_robust`` is unconditionally True and the triple stops
    being three independent readings.  Neither shipped curve has such a row, which
    is why refusing costs nothing today and catches a future edit.
    """
    with pytest.raises(b.MissingEnvelopeError, match="degenerate"):
        b.basin_level_envelope(_envelope_df(-35.0, -35.0), 9.0)
    exc = pytest.raises(
        b.MissingEnvelopeError, b.basin_level_envelope, _envelope_df(-35.0, -35.0), 9.0
    )
    assert "claim of CERTAINTY" in str(exc.value)
    assert "width 0.0 m" in str(exc.value)
    # neither shipped curve is degenerate anywhere — the measurement, not a hope
    df = _real_curve_or_skip()
    assert not (
        pd.to_numeric(df["unc_low"]) == pd.to_numeric(df["unc_high"])
    ).any()


def test_a_usable_envelope_reports_its_width_and_its_measured_scope():
    env = b.basin_level_envelope(_envelope_df(-40.0, -30.0), 9.0)
    assert (env.low_m, env.central_m, env.high_m) == (-40.0, -35.0, -30.0)
    assert env.width_m == pytest.approx(10.0)
    assert env.rule == "envelope_as_carried_in_the_curve"
    assert b.ENVELOPE_PROVENANCE_NO_COLUMN in env.provenance
    # the fixture's catastrophic column is -100 m, outside [-40, -30]
    assert env.bracketed_columns == ("level_m_global_gradual",)
    assert env.scope.startswith("ROW-LEVEL PAIR, NOT AN ENVELOPE")


def test_the_envelope_is_read_off_the_same_row_as_the_level():
    """One lookup, so a level and its envelope can never come from different rows."""
    src = inspect.getsource(b.basin_level_envelope)
    assert "_curve_row_index(" in src
    with pytest.raises(b.CurveAgeGapError):
        b.basin_level_envelope(_envelope_df(-40.0, -30.0), 5.0)


def test_a_blown_up_envelope_edge_is_refused_not_clipped():
    """A datum/unit error on an EDGE is still a datum/unit error."""
    env = b.LevelEnvelope(
        epoch_ka=9.0,
        containment_checked_against="level_m_global_gradual",
        basin="pontic",
        central_m=-35.0,
        low_m=-3500.0,
        high_m=-30.0,
        width_m=3470.0,
        rule=b.ENVELOPE_RULES[0],
        provenance=b.ENVELOPE_PROVENANCE_NO_COLUMN,
        bracketed_columns=("level_m_global_gradual",),
        all_level_columns=("level_m_global_gradual",),
        scope="test",
        provenance_in_note=False,
    )
    with pytest.raises(b.MissingEnvelopeError, match="sanity window"):
        b._convert_envelope(env, 10.0, "9 ka / gradual")


def test_the_envelope_never_falls_back_on_the_central_or_a_neighbour():
    """The two substitutions the class names, asserted as absences in the source."""
    src = inspect.getsource(b.basin_level_envelope) + inspect.getsource(
        b.pontic_threshold
    )
    for forbidden in ("unc_low.fillna", "unc_high.fillna", ".ffill(", ".bfill(", "interpolate("):
        assert forbidden not in src
    doc = b.MissingEnvelopeError.__doc__ or ""
    assert "neighbouring row" in doc and "28 m to 2 m" in doc


def test_duplicate_rows_that_disagree_on_the_MARKER_are_ambiguous_too():
    """C9.  The scan used to check levels and edges and ignore the marker.

    A marker decides ``conversion_m``, i.e. the whole F2 -> F1 term, so two rows
    that agree on every number and name different markers are exactly as ambiguous
    as two that disagree on a level — and which one applied was decided by
    DataFrame row order.  Neither shipped curve has duplicate ages (asserted), so
    this changes no shipped number.
    """
    df = pd.DataFrame(
        {
            "age_ka": [9.0, 9.0],
            "level_m_global_gradual": [-35.0, -35.0],
            "unc_low": [-40.0, -40.0],
            "unc_high": [-30.0, -30.0],
            "marker_lon": [29.10, 29.33],
            "marker_lat": [41.20, 44.13],
            "marker_source": ["Bosphorus sill", "BLVK9814"],
        }
    )
    with pytest.raises(b.CurveAgeGapError, match="marker_lon"):
        b._curve_row_index(df, 9.0)
    # identical markers on both rows are not ambiguous
    same = df.copy()
    same.loc[1, ["marker_lon", "marker_lat", "marker_source"]] = [
        29.10,
        41.20,
        "Bosphorus sill",
    ]
    assert b._curve_row_index(same, 9.0) in (0, 1)
    real = _real_curve_or_skip()
    assert not pd.to_numeric(real["age_ka"]).duplicated().any()


# --- the plain bake still runs ----------------------------------------------


def test_preflight_still_resolves_every_frame_with_the_envelope_in_place():
    """26/26, 0 unconvertible: none of the above may cost the bake a frame."""
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(16.5)).exists():
        pytest.skip("real ICE-6G not available")
    resolved, unconvertible = b.preflight_thresholds(df)
    assert len(resolved) == len(b.EPOCHS_KA) * len(b.SCENARIOS) == 26
    assert unconvertible == {}
    # ... including the four frames whose envelope is refused
    for epoch in (9.0, 7.5):
        assert resolved[(epoch, "catastrophic")].envelope_rule == "refused"


def test_preflight_records_a_missing_curve_row_instead_of_dying_on_it():
    """An animation sweeping a grid finer than the curve needs the refusal as DATA.

    One row per frame, prefixed with the class name so a missing row and an
    unconvertible level — which want different fixes — are never confused.
    """
    df = _real_curve_or_skip()
    if not (b.ICE6G_DIR / b._epoch_filename(16.5)).exists():
        pytest.skip("real ICE-6G not available")
    resolved, unconvertible = b.preflight_thresholds(df, epochs=[16.5, 16.75])
    assert set(resolved) == {(16.5, "gradual"), (16.5, "catastrophic")}
    assert set(unconvertible) == {(16.75, "gradual"), (16.75, "catastrophic")}
    for message in unconvertible.values():
        assert message.startswith("CurveAgeGapError: ")
        assert "no row at 16.75 ka" in message
