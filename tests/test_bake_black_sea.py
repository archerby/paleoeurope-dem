# tests/test_bake_black_sea.py
"""Unit tests for bake_black_sea helpers (no data files required)."""
import importlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`
import scripts.bake_black_sea as b

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
