# tests/test_bake_black_sea.py
"""Unit tests for bake_black_sea helpers (no data files required)."""
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


def test_roots_default_to_historical_repo_relative_locations():
    """Unset env vars must reproduce today's paths, expressed repo-relatively."""
    assert b.PALEO_DATA_ROOT == _REPO_ROOT.parents[1] / "Paleo-Europe_3D"
    assert b.PALEO_OUT_ROOT == _REPO_ROOT.parent
    assert b.OUT_BASE == _REPO_ROOT.parent / "black_sea_corrected" / "topo"
    assert b.ICE6G_DIR == b.PALEO_DATA_ROOT / "data" / "raw" / "ICE6G"
    assert b.BS_DEM == b.PALEO_DATA_ROOT / "data" / "raw" / "GEBCO" / "bs_gmrt_bsproper.tif"


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
