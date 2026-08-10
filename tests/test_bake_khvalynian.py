# tests/test_bake_khvalynian.py
"""Tests for scripts/bake_khvalynian.py — the Khvalynian (Caspian) bake.

Three groups, and the split matters:

* **pure** — grid, guards, dataclasses, sign logic.  No data files.
* **@integration** — the real ICE-6G distribution, the real curve CSV and the real
  33 MB working COG.  These are where the *measurements* live: the sill's sign, the
  sill/locus differential, the ice margin clipping the scene.  A claim this module
  makes in prose is pinned here or it is not made.
* **raster** — the full write path (windowed strips, tags, NaN policy) against a
  SMALL GeoTIFF fixture inside the real extent, so it exercises real code at test
  speed instead of 7800x4320.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`

import scripts.bake_black_sea as bs  # noqa: E402
import scripts.bake_khvalynian as k  # noqa: E402
from scripts.bake_black_sea import (  # noqa: E402
    MANYCH_SILL_M,
    UnconvertibleThresholdError,
    UnknownGiaInBakeError,
)

# WHY THE MODULE HANDLE AND NOT A SECOND `from ... import`.
# ``test_bake_black_sea.py`` calls ``importlib.reload`` on ``bake_black_sea`` (to
# prove the PALEO_*_ROOT env indirection works).  Reload re-executes the module
# INTO THE SAME namespace, so every class in it becomes a NEW object while this
# file's import-time binding keeps the old one — and a function DEFINED in that
# module resolves its globals at CALL time, so it raises the new class.  An
# exception raised by ``basin_level_band`` therefore has to be looked up as
# ``bs.MissingEnvelopeError`` inside the test body, not bound at import.
#
# ``UnconvertibleThresholdError`` above is safe imported: the code that raises it
# in these tests lives in ``bake_khvalynian``, whose globals are never reloaded.

_DATA_AVAILABLE = k.CASPIAN_CSV.exists() and k.ICE6G_DIR.exists()
_DEM_AVAILABLE = k.KHV_DEM.exists()

integration = pytest.mark.skipif(
    not _DATA_AVAILABLE, reason="ICE-6G / Caspian curve not available"
)
needs_dem = pytest.mark.skipif(
    not _DEM_AVAILABLE, reason=f"working raster not available: {k.KHV_DEM}"
)


@pytest.fixture(scope="module")
def curve():
    if not _DATA_AVAILABLE:
        pytest.skip("Caspian curve not available")
    return k.load_curve()


# ===========================================================================
# The epoch grid — 0.25 ka across the event, and no smuggled interpolation
# ===========================================================================


def test_epoch_grid_is_quarter_ka_across_the_spill_window():
    """17-13 ka is sampled at exactly 0.25 ka: the event is not between frames."""
    lo, hi = k.FINE_WINDOW_KA
    fine = sorted(e for e in k.EPOCHS_KA if lo <= e <= hi)
    assert fine == [round(lo + k.FINE_STEP_KA * i, 2) for i in range(17)]
    steps = np.diff(fine)
    assert np.allclose(steps, k.FINE_STEP_KA)


def test_epoch_grid_is_coarser_outside_the_spill_window():
    lo, hi = k.FINE_WINDOW_KA
    outside = sorted(e for e in k.EPOCHS_KA if e < lo or e > hi)
    gaps = np.diff(outside)
    assert gaps.min() >= 0.5, f"a sub-0.5 ka gap outside the fine window: {outside}"


def test_epoch_grid_is_strictly_sorted_and_unique():
    assert sorted(set(k.EPOCHS_KA)) == sorted(k.EPOCHS_KA)


@integration
def test_every_baked_epoch_has_an_exact_curve_row(curve):
    """No epoch relies on the nearest-age lookup interpolating a level.

    ``basin_level`` is nearest-age and never fails, so an off-grid epoch would
    silently take a neighbouring row's level.  This grid was chosen so that cannot
    happen; the guarantee is asserted, not assumed.
    """
    ages = curve["age_ka"].to_numpy(dtype=float)
    misses = [e for e in k.EPOCHS_KA if np.abs(ages - e).min() > 1e-9]
    assert not misses, f"epochs with no exact curve row: {misses}"


# ===========================================================================
# ICE-6G time interpolation
# ===========================================================================


@integration
def test_shipped_ice6g_ages_step_half_a_ka_then_one():
    ages = k.ice6g_ages_ka()
    assert ages[0] == 0.0 and ages[-1] == 26.0
    assert 16.5 in ages and 16.25 not in ages


@integration
def test_bracket_of_a_shipped_epoch_is_the_epoch_itself():
    assert k._ice6g_bracket(16.5) == (16.5, 16.5, 0.0)


@integration
def test_bracket_of_a_quarter_ka_epoch_is_its_two_neighbours():
    lo, hi, w = k._ice6g_bracket(16.25)
    assert (lo, hi) == (16.0, 16.5)
    assert w == pytest.approx(0.5)


@integration
def test_bracket_refuses_an_epoch_outside_the_shipped_range():
    with pytest.raises(ValueError, match="outside the shipped ICE-6G range"):
        k._ice6g_bracket(30.0)


@integration
def test_time_interpolation_note_distinguishes_exact_from_blended():
    assert "no time interpolation" in k._time_interpolation_note(16.5)
    assert "linear in time" in k._time_interpolation_note(16.25)


@integration
def test_offset_on_grid_reproduces_the_shipped_datum_values():
    """The datum fix's own numbers, recomputed here rather than quoted.

    22 / 18 / 16.5 / 14.5 / 12 ka offsets are the values the corrected Pontic bake
    ran on.  If this module ever computed the datum from a regional cut (the bug
    retired 2026-07-30) these would move by tens of metres.
    """
    assert k.sea_surface_offset(18.0) == pytest.approx(81.959, abs=0.01)
    assert k.sea_surface_offset(16.5) == pytest.approx(77.906, abs=0.01)
    assert k.sea_surface_offset(14.5) == pytest.approx(66.373, abs=0.01)
    assert k.sea_surface_offset(12.0) == pytest.approx(30.005, abs=0.01)


@integration
def test_quarter_ka_offset_is_the_linear_blend_of_its_brackets():
    blend = 0.5 * (k.sea_surface_offset(16.0) + k.sea_surface_offset(16.5))
    assert k.sea_surface_offset(16.25) == pytest.approx(blend, abs=1e-9)


@integration
def test_mwp1a_makes_the_14_25_ka_frame_a_midpoint_across_a_20_m_step():
    """Stated in the docstring, measured here: the closing epoch is the soft one."""
    step = k.sea_surface_offset(14.5) - k.sea_surface_offset(14.0)
    assert step > 15.0, f"MWP-1A step is {step:.1f} m, docstring says ~20 m"
    assert "linear in time" in k._time_interpolation_note(14.25)


# ===========================================================================
# The sill — its value, and above all its SIGN
# ===========================================================================


@needs_dem
def test_manych_sill_elevation_is_read_from_the_raster_and_is_positive():
    """The sign of the event, established from the raster, not quoted from a paper.

    +23 m means the crest stands ABOVE present sea level, which is why an endorheic
    Caspian at -28 m does not spill today.  A negative reading would make the sill a
    submarine channel and this render would have no subject.
    """
    value = k.manych_sill_elevation_m()
    assert value > 0.0, f"the Manych saddle read {value:+.2f} m — sign inverted?"
    assert value == pytest.approx(MANYCH_SILL_M, abs=1.0)


@needs_dem
def test_the_sill_read_does_not_load_the_raster_whole():
    """A 1x1 windowed read: the working raster is 7800x4320 and stays on disk."""
    with rasterio.open(k.KHV_DEM) as ds:
        assert (ds.width, ds.height) == (7800, 4320)
        assert ds.block_shapes[0] == (512, 512)
    assert k.manych_sill_elevation_m() > 0.0


@needs_dem
def test_the_working_raster_covers_caspian_manych_and_pontic():
    with rasterio.open(k.KHV_DEM) as ds:
        b = ds.bounds
    assert (b.left, b.bottom, b.right, b.top) == (
        k.LON_MIN,
        k.LAT_MIN,
        k.LON_MAX,
        k.LAT_MAX,
    )
    # the Manych saddle, the Caspian southern deep and the Bosphorus are all inside
    for lon, lat in ((44.119, 45.773), (51.0, 38.0), (29.1, 41.2)):
        assert b.left <= lon <= b.right and b.bottom <= lat <= b.top


def test_the_continuous_path_crest_is_carried_and_is_above_the_point_sill():
    """+23.00 is the max of per-column minima, i.e. a LOWER BOUND on the crest."""
    assert k.MANYCH_SILL_PATH_M > MANYCH_SILL_M


# ===========================================================================
# The locus, and the differential that licenses it
# ===========================================================================


@integration
def test_the_locus_band_contains_enough_cells_to_have_a_median():
    _median, (_p5, _p95), n = k.locus_solid_displacement(16.5)
    assert n >= k._MIN_LOCUS_CELLS * 100


@integration
@pytest.mark.parametrize(
    "epoch,expected",
    [
        (18.0, -0.292),
        (17.5, -0.287),
        (17.0, -0.229),
        (16.5, -0.243),
        (16.0, -0.270),
        (15.5, -0.205),
        (15.0, -0.171),
        (14.5, -0.146),
        (14.0, +0.177),
        (13.5, +0.167),
        (13.0, +0.155),
        (12.0, +0.148),
    ],
)
def test_sill_locus_differential_is_the_measured_value(epoch, expected):
    """Measured on THIS module's locus — not inherited from the datum contract.

    The contract's §6 update reports +0.03..+0.18 m for a different band.  This
    band gives -0.29..+0.18 m: same order, but two-signed where the contract's is
    one-signed.  The conclusion (a band locus is good enough for the Caspian, where
    the same substitution costs the Pontic up to 5.8 m) survives; the bracket is
    the one measured here and is what SillTest carries.
    """
    differential, _bracket = k.sill_marker_differential(epoch)
    assert differential == pytest.approx(expected, abs=0.005)


@integration
def test_the_differential_stays_far_below_the_at_sill_tolerance():
    """This is the whole licence for using a band instead of per-row markers."""
    worst = max(
        abs(k.sill_marker_differential(e)[0]) for e in (18.0, 16.5, 14.5, 13.0)
    )
    assert worst < k.AT_SILL_TOL_M


@integration
def test_the_differential_bracket_brackets_the_differential():
    differential, (lo, hi) = k.sill_marker_differential(16.5)
    assert lo <= differential <= hi
    assert hi - lo > 0.0


@integration
def test_palaeo_sill_f1_reproduces_the_shipped_curve_column():
    """23.00 + Δz(sill) must equal the CSV's own paleo_sill_m_asl, or one of us is
    on a different datum."""
    for epoch, expected in ((21.0, 24.02), (18.0, 24.62), (16.5, 25.70), (14.5, 26.94)):
        dz_sill, _block = k.sill_solid_displacement(epoch)
        assert MANYCH_SILL_M + dz_sill == pytest.approx(expected, abs=0.02)


# ===========================================================================
# The curve, its sign, and the lowstand that did not happen
# ===========================================================================


@integration
def test_the_shipped_curve_passes_every_guard(curve):
    assert len(curve) > 60
    for scenario in k.SCENARIOS:
        assert f"level_m_global_{scenario}" in curve.columns


@integration
def test_khvalynian_highstand_levels_are_POSITIVE(curve):
    """Sign vigilance.  A naive PDF extraction inverted every Caspian level once.

    arslanov2016 prints the Early Khvalynian stages positive in Table 1, Table 2 and
    the Fig. 1 caption; the Elsevier encoding renders '+' as U+00FE.  A re-inverted
    curve would put the highstand at -35/-48 m — below the modern surface — and this
    fails.
    """
    peak_rows = curve[(curve["age_ka"] >= 15.0) & (curve["age_ka"] <= 16.5)]
    assert (peak_rows["level_m_global_defensible"] > 20.0).all()
    assert (peak_rows["level_m_global_maximal"] > 20.0).all()


@integration
def test_no_deep_lgm_lowstand_in_the_shipped_curve(curve):
    k.check_no_lgm_lowstand(curve)  # must not raise


def test_check_no_lgm_lowstand_rejects_a_reimported_atelian_lowstand():
    """-100 m at 21.5 ka is the Atelian (MIS 4 / early MIS 3) mis-dated onto the LGM."""
    bad = pd.DataFrame(
        {
            "age_ka": [26.0, 21.0, 16.5, 0.0],
            "level_m_global_defensible": [5.0, -100.0, 27.0, -28.0],
            "level_m_global_maximal": [5.0, -100.0, 33.4, -28.0],
        }
    )
    with pytest.raises(ValueError, match="lowstand that did not happen"):
        k.check_no_lgm_lowstand(bad)


def test_check_no_lgm_lowstand_accepts_the_established_plus_5_to_plus_15_plateau():
    ok = pd.DataFrame(
        {
            "age_ka": [26.0, 22.0, 19.0],
            "level_m_global_defensible": [5.0, 6.3, 7.0],
            "level_m_global_maximal": [5.0, 6.3, 15.0],
        }
    )
    k.check_no_lgm_lowstand(ok)


def test_check_no_lgm_lowstand_rejects_an_unknown_scenario():
    with pytest.raises(ValueError, match="unknown scenario"):
        k.check_no_lgm_lowstand(
            pd.DataFrame({"age_ka": [26.0], "level_m_global_defensible": [5.0]}),
            scenarios=("banana",),
        )


# ===========================================================================
# The threshold: F2 -> F1
# ===========================================================================


@integration
def test_threshold_is_level_plus_the_locus_median_and_says_which_rule(curve):
    threshold = k.caspian_threshold(curve, 16.5, "defensible")
    median, _bracket, n = k.locus_solid_displacement(16.5)
    assert threshold.rule == "locus"
    assert threshold.frame == "F2"
    assert threshold.basin == "caspian"
    assert threshold.level_m == pytest.approx(27.0)
    assert threshold.conversion_m == pytest.approx(median)
    assert threshold.threshold_m == pytest.approx(27.0 + median)
    assert threshold.n_cells == n
    assert "rule=locus" in threshold.provenance


@integration
def test_the_conversion_is_ADDITIVE_so_a_positive_dz_raises_the_threshold(curve):
    """Sign of the conversion, stated as a test rather than trusted from prose.

    Δz > 0 means the ground stood higher than today, so the same present-day marker
    elevation implies a HIGHER palaeo water surface.  Subtracting would flip the
    spill verdict at every epoch of the window.
    """
    median, _bracket, _n = k.locus_solid_displacement(16.5)
    assert median > 0.0
    threshold = k.caspian_threshold(curve, 16.5, "defensible")
    assert threshold.threshold_m > threshold.level_m


@integration
def test_threshold_bracket_straddles_the_threshold(curve):
    t = k.caspian_threshold(curve, 16.5, "defensible")
    lo, hi = t.bracket_m
    assert lo <= t.threshold_m <= hi


@integration
def test_threshold_rejects_an_unknown_scenario(curve):
    with pytest.raises(ValueError, match="unknown scenario"):
        k.caspian_threshold(curve, 16.5, "gradual_but_misspelled")


@integration
def test_the_caspian_curve_still_has_no_marker_columns(curve):
    """The premise of the locus rule.  If this ever fails, the marker rule is live
    and its numbers must be re-checked — silently preferring markers would change
    every threshold."""
    assert not all(c in curve.columns for c in ("marker_lon", "marker_lat"))


def test_locus_refuses_when_the_band_is_too_small(monkeypatch):
    tiny = k.CaspianLocus(
        name="tiny",
        lon_deg=(44.0, 44.05),
        lat_deg=(45.0, 45.05),
        modern_elev_m=(1e9, 2e9),
        why="deliberately empty",
    )
    monkeypatch.setattr(
        k,
        "solid_displacement_field",
        lambda epoch: (
            np.zeros((3, 3)),
            np.array([44.9, 45.0, 45.1]),
            np.array([44.0, 44.1, 44.2]),
        ),
    )
    monkeypatch.setattr(
        k,
        "_open_khv_window",
        lambda age: type("W", (), {"topo_0": np.zeros((3, 3))})(),
    )
    with pytest.raises(UnconvertibleThresholdError, match="too few"):
        k.locus_solid_displacement(16.5, tiny)


# ===========================================================================
# THE SILL TEST — the first-class output
# ===========================================================================


def test_verdict_vocabulary_matches_the_curve_csv():
    assert set(k.SPILL_STATES) == {"spilling_west", "at_sill", "endorheic"}
    assert k._verdict(+5.0) == "spilling_west"
    assert k._verdict(-5.0) == "endorheic"
    assert k._verdict(0.0) == "at_sill"
    assert k._verdict(k.AT_SILL_TOL_M) == "at_sill"
    assert k._verdict(k.AT_SILL_TOL_M + 1e-9) == "spilling_west"


@integration
def test_sill_test_compares_two_F1_quantities(curve):
    t = k.sill_test(curve, 16.5, "defensible")
    assert t.surface_f1_m == pytest.approx(t.level_f2_m + t.dz_locus_m)
    assert t.sill_f1_m == pytest.approx(MANYCH_SILL_M + t.dz_sill_m)
    assert t.margin_m == pytest.approx(t.surface_f1_m - t.sill_f1_m)
    assert t.verdict == "spilling_west"


@integration
def test_the_hero_epoch_clears_the_sill_by_a_robust_margin(curve):
    """16.5 ka defensible: +27.0 m F2 level, crest +25.70 m F1 — the event."""
    t = k.sill_test(curve, 16.5, "defensible")
    assert t.sill_f1_m == pytest.approx(25.70, abs=0.02)
    assert t.margin_m > k.AT_SILL_TOL_M
    assert t.verdict_is_robust


@integration
def test_the_F1_and_F2_readings_agree_to_within_the_differential(curve):
    """Contract §6's shortcut ('compare against the MODERN sill') is the same test."""
    for epoch in (18.0, 16.5, 15.0, 14.25, 13.0):
        t = k.sill_test(curve, epoch, "defensible")
        assert t.margin_m - t.margin_modern_sill_m == pytest.approx(
            -t.differential_m, abs=1e-9
        )


@integration
def test_the_mixed_frame_reading_is_carried_and_is_the_pessimistic_one(curve):
    """An F2 level against an F1 sill discards the whole GIA term (~2.7 m at 16.5 ka)."""
    t = k.sill_test(curve, 16.5, "defensible")
    assert t.margin_mixed_frame_m < t.margin_m
    assert t.margin_m - t.margin_mixed_frame_m == pytest.approx(t.dz_locus_m)
    assert t.dz_locus_m == pytest.approx(2.94, abs=0.05)


@integration
def test_the_defensible_spill_window_is_16_5_to_14_25_ka(curve):
    """THE result of this bake, on this module's own arithmetic."""
    report = k.sill_report(curve, k.EPOCHS_KA, ("defensible",))
    assert k.spill_window_ka(report, "defensible") == (16.5, 14.25)


@integration
def test_the_maximal_scenario_opens_the_window_no_later_than_the_defensible(curve):
    report = k.sill_report(curve, k.EPOCHS_KA, k.SCENARIOS)
    d = k.spill_window_ka(report, "defensible")
    m = k.spill_window_ka(report, "maximal")
    assert m[0] >= d[0]


@integration
def test_the_lgm_plateau_never_spills(curve):
    """+5..+7 m against a +24 m crest: 17 m short.  No spill before the rise."""
    report = k.sill_report(curve, [26.0, 22.0, 20.0, 18.0], ("defensible", "maximal"))
    assert set(report["verdict"]) == {"endorheic"}


@integration
def test_the_continuous_path_crest_costs_exactly_the_closing_frame(curve):
    """+25.00 m is a higher bar than +23.00 m, so it cannot spill on MORE epochs —
    and on this curve it costs precisely one frame, the 14.25 ka closer."""
    report = k.sill_report(curve, k.EPOCHS_KA, ("defensible",))
    spills = set(report.loc[report["verdict"] == "spilling_west", "epoch_ka"])
    spills_path = set(report.loc[report["verdict_path"] == "spilling_west", "epoch_ka"])
    assert spills_path < spills
    assert spills - spills_path == {14.25}


@integration
def test_the_mixed_frame_reading_closes_the_window_one_frame_early(curve):
    """Contract §6's prediction, reproduced from the raw fields.

    Comparing the F2 level against the F1 ``paleo_sill_m_asl`` discards the whole
    GIA term on one side and loses the 14.25 ka frame.  This is the number the
    frame-correct test exists to recover; it is measured, not quoted.
    """
    report = k.sill_report(curve, k.EPOCHS_KA, ("defensible",))
    right = set(report.loc[report["verdict"] == "spilling_west", "epoch_ka"])
    mixed = set(report.loc[report["verdict_mixed_frame"] == "spilling_west", "epoch_ka"])
    assert right - mixed == {14.25}


@integration
def test_the_only_non_robust_verdict_is_the_mwp1a_epoch(curve):
    """14.0 ka: margin -1.18 m, inside the locus bracket — and the same epoch the
    curve's header flags as an ICE-6G MWP-1A artefact.  Two independent reasons the
    closing date is soft, and the flag is how a reader finds out."""
    report = k.sill_report(curve, k.EPOCHS_KA, k.SCENARIOS)
    soft = set(report.loc[~report["verdict_is_robust"], "epoch_ka"])
    assert soft == {14.0}


@integration
def test_the_report_is_one_row_per_epoch_and_scenario(curve):
    epochs = [16.5, 16.25, 16.0]
    report = k.sill_report(curve, epochs, k.SCENARIOS)
    assert len(report) == len(epochs) * len(k.SCENARIOS)
    assert set(report["epoch_ka"]) == set(epochs)
    for column in (
        "verdict",
        "verdict_path",
        "verdict_modern_sill",
        "verdict_mixed_frame",
        "verdict_is_robust",
        "differential_bracket_lo_m",
        "differential_bracket_hi_m",
        "gia_time_interpolation",
        "provenance",
    ):
        assert column in report.columns


@integration
def test_a_verdict_inside_the_locus_bracket_is_flagged_not_robust(curve):
    """The bracket is carried so a knife-edge epoch cannot be quoted as a date.

    16.75 ka defensible sits +21.4 m against a +25.71 m crest — 4 m short, robust.
    Construct the knife edge instead of hunting for one, so the flag is tested at
    the place it exists for.
    """
    knife = curve.copy()
    row = (knife["age_ka"] - 16.5).abs().idxmin()
    dz_sill, _b = k.sill_solid_displacement(16.5)
    median, _bracket, _n = k.locus_solid_displacement(16.5)
    knife.loc[row, "level_m_global_defensible"] = (
        MANYCH_SILL_M + dz_sill - median
    )  # margin exactly 0
    t = k.sill_test(knife, 16.5, "defensible")
    assert t.verdict == "at_sill"
    assert not t.verdict_is_robust


@integration
def test_report_verdicts_agree_with_the_shipped_modern_sill_column(curve):
    """Independent cross-check: the curve's own spill_state_modern_sill_defensible.

    Two different codebases, same physical test.  Disagreement outside the
    at_sill band would mean one of them is on a different frame.
    """
    report = k.sill_report(curve, k.EPOCHS_KA, ("defensible",))
    shipped = curve.set_index("age_ka")["spill_state_modern_sill_defensible"]
    disagree = []
    for _i, r in report.iterrows():
        want = str(shipped.loc[r.epoch_ka])
        if r.verdict_modern_sill != want and r.verdict_modern_sill != "at_sill":
            disagree.append((r.epoch_ka, r.verdict_modern_sill, want))
    assert not disagree, f"verdict disagreements with the shipped column: {disagree}"


# ===========================================================================
# The ice margin — the reason this bake's NaN default is not the Pontic bake's
# ===========================================================================


@integration
def test_the_window_straddles_the_ice_margin_before_17_ka():
    """Measured, and it falsifies bake_black_sea's 'the Caspian bbox is ice-free'.

    That claim was true of the narrow Caspian bbox.  This scene reaches 54 N / 25.5 E
    and clips the SE margin of the Fennoscandian sheet, which is exactly why the
    default policy here is 'nodata' and not 'raise'.
    """
    assert k.ice_cells_in_window(18.0) > 0
    assert k.ice_cells_in_window(17.5) > 0
    assert k.ice_cells_in_window(17.0) == 0
    assert k.ice_cells_in_window(16.5) == 0


@integration
def test_the_spill_window_itself_is_ice_free():
    """The frames that carry the event have no unknown Δz at all."""
    for epoch in k.EPOCHS_KA:
        if 13.0 <= epoch <= 17.0:
            assert k.ice_cells_in_window(epoch) == 0, f"{epoch} ka"


def test_the_default_nan_policy_is_nodata_and_zero_does_not_exist():
    from scripts.bake_black_sea import BAKE_GIA_NAN_POLICIES

    assert k.KHV_GIA_NAN_POLICY == "nodata"
    assert k.KHV_GIA_NAN_POLICY in BAKE_GIA_NAN_POLICIES
    assert "zero" not in BAKE_GIA_NAN_POLICIES


def test_khv_nodata_is_bound_to_the_pontic_bakes_sentinel():
    from scripts.bake_black_sea import BS_NODATA_M

    assert k.KHV_NODATA_M == BS_NODATA_M


# ===========================================================================
# The raster write path, against a small fixture inside the real extent
# ===========================================================================


@pytest.fixture()
def small_dem(tmp_path):
    """A 60x40 GeoTIFF over the Manych saddle, inside the real scene extent."""
    from rasterio.transform import from_origin

    west, north, res = 43.0, 47.0, 0.05
    data = np.linspace(-100.0, 100.0, 60 * 40).reshape(40, 60).astype(np.float32)
    path = tmp_path / "small.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=40,
        width=60,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, res, res),
        nodata=-32767.0,
    ) as dst:
        dst.write(data, 1)
    return path


@integration
def test_bake_writes_a_raster_carrying_the_pontic_tag_set_plus_the_sill_test(
    small_dem, tmp_path, curve
):
    tags = {
        "threshold_defensible": k.caspian_threshold(curve, 16.5, "defensible").provenance,
        "sill_test_defensible": k.sill_test(curve, 16.5, "defensible").provenance,
        "sill_verdict_defensible": k.sill_test(curve, 16.5, "defensible").verdict,
    }
    out = k.build_paleo_dem(
        16.5, provenance=tags, dem_path=small_dem, out_dir=tmp_path / "topo"
    )
    with rasterio.open(out) as ds:
        got = ds.tags()
        assert ds.nodata == k.KHV_NODATA_M
        assert ds.dtypes[0] == "float32"
        # Lossless float predictor: 2.6x smaller on the real 7800x4320 frame, same
        # bytes back.  It is a CREATION option, so it is not in ds.profile on read —
        # it surfaces in the IMAGE_STRUCTURE metadata domain.
        assert ds.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == "3"
        assert ds.tags(ns="IMAGE_STRUCTURE")["COMPRESSION"] == "DEFLATE"

    # the Pontic bake's key set
    for key in (
        "epoch_ka",
        "gia_model",
        "gia_datum",
        "gia_sea_surface_offset_m",
        "gia_nan_policy",
        "vertical_frame",
        "vertical_datum_contract",
    ):
        assert key in got, key
    # plus this bake's own
    for key in (
        "threshold_defensible",
        "sill_test_defensible",
        "sill_verdict_defensible",
        "gia_time_interpolation",
        "manych_sill_modern_m",
        "manych_sill_path_m",
    ):
        assert key in got, key

    assert got["epoch_ka"] == "16.5"
    assert got["gia_sea_surface_offset_m"].startswith("77.9")
    assert got["sill_verdict_defensible"] == "spilling_west"
    assert "SILL TEST" in got["sill_test_defensible"]
    assert got["manych_sill_modern_m"] == "23.00"


@integration
def test_striped_bake_is_identical_to_a_single_strip_bake(small_dem, tmp_path):
    """Windowed reads must not change the answer."""
    a = k.build_paleo_dem(
        16.5, dem_path=small_dem, out_dir=tmp_path / "a", strip_rows=40
    )
    b = k.build_paleo_dem(
        16.5, dem_path=small_dem, out_dir=tmp_path / "b", strip_rows=7
    )
    with rasterio.open(a) as da, rasterio.open(b) as db:
        assert np.array_equal(da.read(1), db.read(1))


@integration
def test_the_paleo_dem_is_the_modern_dem_plus_a_positive_dz_at_16_5_ka(
    small_dem, tmp_path
):
    out = k.build_paleo_dem(16.5, dem_path=small_dem, out_dir=tmp_path / "topo")
    with rasterio.open(small_dem) as src, rasterio.open(out) as dst:
        modern = src.read(1)
        paleo = dst.read(1)
    delta = paleo - modern
    assert np.isfinite(delta).all()
    # Δz over the Manych/Ergeni ground at 16.5 ka is ~+2.7..+3.0 m; the sign is the
    # point (the ground stood HIGHER), the magnitude is a sanity bound.
    assert 1.0 < float(delta.mean()) < 6.0


@integration
def test_bake_rejects_a_zero_nan_policy(small_dem, tmp_path):
    with pytest.raises(ValueError, match="no 'zero' policy"):
        k.build_paleo_dem(
            16.5,
            gia_nan_policy="zero",
            dem_path=small_dem,
            out_dir=tmp_path / "topo",
        )


@integration
def test_raise_policy_still_aborts_on_the_glaciated_epochs(small_dem, tmp_path):
    """'nodata' is a default, not a silencer: 'raise' is still available and fires."""
    with pytest.raises(UnknownGiaInBakeError, match="ICE SURFACE"):
        k.build_paleo_dem(
            18.0,
            gia_nan_policy="raise",
            dem_path=small_dem,
            out_dir=tmp_path / "topo",
        )


@integration
def test_nodata_policy_writes_the_sentinel_never_zero_correction(small_dem, tmp_path):
    """At 18 ka the hole is far from this fixture, so nothing here is nodata — but
    the tag must still record how many cells were unknown, so a reader of the .tif
    knows a hole existed somewhere in the scene."""
    out = k.build_paleo_dem(
        18.0, gia_nan_policy="nodata", dem_path=small_dem, out_dir=tmp_path / "topo"
    )
    with rasterio.open(out) as ds:
        assert ds.tags()["gia_nan_policy"] == "nodata"
        assert int(ds.tags()["gia_unknown_ice6g_cells"]) > 0


@integration
def test_bake_refuses_a_dem_that_reaches_outside_the_ice6g_window(tmp_path):
    """apply_gia_delta extrapolates as 0 m of GIA — a zero-filled unknown."""
    from rasterio.transform import from_origin

    path = tmp_path / "toobig.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-20.0, 70.0, 1.0, 1.0),
        nodata=-32767.0,
    ) as dst:
        dst.write(np.zeros((10, 10), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="model window"):
        k.build_paleo_dem(16.5, dem_path=path, out_dir=tmp_path / "topo")


# ===========================================================================
# main() — and the frame it deliberately does not render
# ===========================================================================


@integration
def test_main_report_only_computes_the_sill_report_and_writes_no_raster(tmp_path):
    """Was ``..._writes_the_sill_csv_and_no_raster``, and the rename is the fix.

    Writing the CSV is now gated behind ``write_report`` / ``--write`` — see
    ``test_sill_report_csv_write_is_gated`` for why merely asking the question
    must not overwrite the report of record.  What ``--sill-report-only`` still
    guarantees is the part that was always its job: the full report is computed
    and returned, and no raster is produced.
    """
    report = k.main(
        epochs=[16.5, 16.25],
        write_rasters=False,
        write_report=True,
        out_dir=tmp_path,
    )
    csv = tmp_path / k.SILL_REPORT_CSV.name
    assert csv.exists()
    assert len(pd.read_csv(csv)) == len(report) == 2 * len(k.SCENARIOS)
    assert not (tmp_path / "topo").exists()


def test_this_module_renders_nothing_because_the_scene_holds_two_basin_levels():
    """A structural assertion, not a style check.

    A single ``sea_level_override_m`` over 25.5-58 E would flood the Pontic to the
    Caspian's surface.  If a render call ever appears here it must come with a basin
    partition, and this test is where that decision has to be re-taken.
    """
    source = Path(k.__file__).read_text(encoding="utf-8")
    assert "render_paleocostline_epoch" not in source
    # the keyword APPLIED, not the name discussed in prose
    assert "sea_level_override_m=" not in source


# ===========================================================================
# T3 — THE CURVE'S PER-ROW LEVEL ENVELOPE.  The spill used to be a property of the
# ADOPTED CENTRAL curve alone; these tests pin what the curve's own unc_low /
# unc_high pair says about the same event, and — since 2026-08-03 — WHAT THAT PAIR
# IS.
#
# The two findings in two lines:
#   * the envelope's LOWER edge is endorheic at every epoch of the 43-epoch grid, in
#     both scenarios — so the honest caption is "at the central estimate it spilled;
#     at the lower edge of the envelope the curve carries, it did not";
#   * the pair is ONE PER ROW and brackets all six level columns, so it is an
#     INTER-SCHOOL SCENARIO ENVELOPE and not the uncertainty of either scenario.
#     Everything derived from it is identical across scenarios, and that has to be
#     visible in the data or a per-scenario report misstates it.
# ===========================================================================


#: The Caspian curve's four DISTINCT level columns.  ``gradual`` / ``catastrophic``
#: are GENERATED ALIASES of ``defensible`` / ``maximal`` (the CSV header says so),
#: so including them would test the same numbers twice and claim six checks.
_DISTINCT_LEVEL_COLUMNS = (
    "level_m_global_defensible",
    "level_m_global_maximal",
    "level_m_global_late_rise",
    "level_m_global_early_rise",
)


@pytest.fixture(scope="module")
def band_report(curve):
    """The full report, computed once — 43 epochs x 2 scenarios."""
    return k.sill_report(curve, k.EPOCHS_KA, k.SCENARIOS)


@integration
def test_unc_band_is_absolute_not_a_width(curve):
    """``unc_low`` / ``unc_high`` are LEVELS, not a +/- half-width.

    The whole triple rests on this reading.  Were they half-widths, ``unc_low``
    would be a small positive number and every envelope verdict below would be
    nonsense — so the reading is asserted on all 71 rows and all four distinct
    level columns rather than inferred from two spot values.
    """
    assert len(curve) == 71
    lo = pd.to_numeric(curve["unc_low"])
    hi = pd.to_numeric(curve["unc_high"])
    assert lo.notna().all() and hi.notna().all()
    for col in _DISTINCT_LEVEL_COLUMNS:
        level = pd.to_numeric(curve[col])
        bad = curve.index[~((lo <= level) & (level <= hi))]
        assert not len(bad), f"{col}: envelope misses its central at rows {list(bad)}"
    # And the width really does vary, which is WHY a neighbouring row's envelope is
    # never borrowed for a row whose own is unusable.
    width = hi - lo
    assert float(width.min()) == pytest.approx(2.0)
    assert float(width.max()) == pytest.approx(28.0)


@integration
def test_the_pair_is_one_per_row_and_brackets_every_level_column(curve):
    """C1, THE HEADLINE.  It is an inter-school ENVELOPE, not a per-scenario band.

    Three measurements, all on the shipped file:

    * every one of the SIX level columns lies inside the pair at every one of the
      71 rows — so the pair is not "the band around defensible" or "around
      maximal", it brackets both schools at once;
    * 63 distinct pairs serve 71 rows, i.e. it really is carried per row and is not
      a constant;
    * at 16.00 ka the pair [+22.0, +50.0] brackets defensible +35.0 AND maximal
      +48.0 — the single row that makes the point unarguable.
    """
    columns = bs.level_columns(curve)
    assert len(columns) == 6
    lo = pd.to_numeric(curve["unc_low"])
    hi = pd.to_numeric(curve["unc_high"])
    for col in columns:
        v = pd.to_numeric(curve[col])
        assert bool(((lo - 1e-9 <= v) & (v <= hi + 1e-9)).all()), col
    assert len(set(zip(lo, hi))) == 63
    row = bs._curve_row_index(curve, 16.0)
    assert (float(lo.loc[row]), float(hi.loc[row])) == (22.0, 50.0)
    assert float(curve.loc[row, "level_m_global_defensible"]) == 35.0
    assert float(curve.loc[row, "level_m_global_maximal"]) == 48.0


@integration
def test_the_envelope_triple_is_identical_between_the_two_scenarios(band_report):
    """C1's consequence, in the data.  One pair per row -> one triple per row.

    Nine reported fields carry the same value for ``defensible`` and ``maximal`` at
    all 43 epochs.  They repeat in a per-scenario report because the report has one
    row per scenario, NOT because the scenario changed anything — and the row says
    so via ``envelope_is_scenario_invariant`` and ``envelope_scope`` rather than
    leaving a reader of one row to conclude that +17.1 m is maximal's own bound.
    """
    d = band_report[band_report["scenario"] == "defensible"].set_index("epoch_ka")
    m = band_report[band_report["scenario"] == "maximal"].set_index("epoch_ka")
    assert len(d) == len(m) == 43
    for col in (
        "envelope_lo_f2_m",
        "envelope_hi_f2_m",
        "margin_envelope_lo_m",
        "margin_envelope_hi_m",
        "margin_joint_corner_m",
    ):
        assert float((d[col] - m[col]).abs().max()) < 1e-9, col
    for col in (
        "verdict_envelope_lo",
        "verdict_envelope_hi",
        "envelope_verdict",
        "verdict_is_envelope_robust",
        "verdict_is_jointly_robust",
    ):
        assert bool((d[col] == m[col]).all()), col
    assert bool(band_report["envelope_is_scenario_invariant"].all())
    assert set(band_report["envelope_brackets_n_columns"]) == {6}
    assert set(band_report["envelope_n_level_columns"]) == {6}
    # ... and the sentence that says it, on every row
    scopes = set(band_report["envelope_scope"])
    assert all(s.startswith("INTER-SCHOOL SCENARIO ENVELOPE") for s in scopes)
    assert all("NOT the uncertainty of any single scenario column" in s for s in scopes)


@integration
def test_the_report_never_calls_the_envelope_a_band(band_report):
    """The rename is the fix, so it is asserted on the SHIPPED COLUMN NAMES.

    A reader of ``khvalynian_sill_test.csv`` sees column names before docstrings.
    ``band_lo_f2_m`` next to ``scenario=maximal`` is the false claim; there must be
    no column left that spells it that way.
    """
    banded = [c for c in band_report.columns if "band" in str(c)]
    assert banded == []
    for column in (
        "envelope_lo_f2_m",
        "envelope_hi_f2_m",
        "envelope_width_m",
        "surface_envelope_lo_f1_m",
        "surface_envelope_hi_f1_m",
        "margin_envelope_lo_m",
        "margin_envelope_hi_m",
        "verdict_envelope_lo",
        "verdict_envelope_hi",
        "envelope_verdict",
        "verdict_is_envelope_robust",
        "envelope_scope",
        "envelope_rule",
        "envelope_provenance",
    ):
        assert column in band_report.columns, column


@integration
def test_band_lower_bound_never_spills(band_report):
    """THE finding.  Not "rarely": never, on either scenario, on the whole grid."""
    for scenario in k.SCENARIOS:
        assert (
            k.spill_window_ka(
                band_report, scenario, verdict_column="verdict_envelope_lo"
            )
            is None
        )
    assert set(band_report["verdict_envelope_lo"]) == {"endorheic"}


@integration
def test_band_upper_bound_spills_17_to_13_75(band_report):
    """The other edge, so the triple is bracketed above as well as below.

    Inside the 17-13 ka engine window the upper edge spills at every epoch down
    to 13.75 ka and is endorheic at the last three.  It ALSO spills at 17.5 and
    18.0 ka, outside that window — and those two are exactly the rows where that
    edge is CIRCULAR (see ``test_the_upper_edge_is_circular_where_it_stands_on_the_sill``),
    so they are recorded and refused rather than quoted as a wider window.
    """
    window = band_report[
        (band_report["epoch_ka"] >= 13.0) & (band_report["epoch_ka"] <= 17.0)
    ]
    spilling = sorted(
        {
            float(e)
            for e in window.loc[
                window["verdict_envelope_hi"] == "spilling_west", "epoch_ka"
            ]
        }
    )
    assert spilling == [round(13.75 + 0.25 * i, 2) for i in range(14)]
    endorheic = sorted(
        {
            float(e)
            for e in window.loc[
                window["verdict_envelope_hi"] == "endorheic", "epoch_ka"
            ]
        }
    )
    assert endorheic == [13.0, 13.25, 13.5]
    outside = band_report[band_report["epoch_ka"] > 17.0]
    assert set(
        outside.loc[outside["verdict_envelope_hi"] == "spilling_west", "epoch_ka"]
    ) == {17.5, 18.0}
    # none of the 17-13 ka window is circular, so THAT window is real evidence
    assert not window["envelope_hi_is_circular"].any()


@integration
def test_the_upper_edge_is_circular_where_it_stands_on_the_sill(curve, band_report):
    """C2.  ``verdict_envelope_hi`` is not evidence where unc_high IS the crest.

    Established by MEASUREMENT over the curve, not by a hardcoded list of ages:
    :func:`envelope_hi_circularity` asks (a) does ``unc_high`` coincide with a level
    column here, and (b) does it stand at a measured Manych crest within
    ``AT_SILL_TOL_M``.  (b) is the circularity test.

    Measured on the shipped curve, four rows have ``unc_high == early_rise``
    (20.0 / 17.7 / 17.5 / 17.4 ka) and four have ``unc_high`` at the +25.00 m crest
    (18.0 / 17.7 / 17.5 / 17.4 ka).  20.0 ka is NOT circular — there the shared
    value is +15.0 m, ten metres below the crest, so the test says ``endorheic`` and
    that is information.  18.0 ka IS circular without matching a column, which is
    why the rule keys on (b): its note records unc_high raised to +25 because a
    hiatus "cannot exclude a short excursion inside itself".

    Of the 43 BAKED epochs exactly 18.0 and 17.5 ka are affected — precisely the two
    where the upper edge spills outside the 17-13 ka window.
    """
    # (a) the coincidence, measured over every level column of the file
    hi = pd.to_numeric(curve["unc_high"])
    coincide = {
        col: sorted(
            float(a)
            for a in curve.loc[
                np.isclose(pd.to_numeric(curve[col]), hi), "age_ka"
            ]
        )
        for col in bs.level_columns(curve)
    }
    assert coincide["level_m_global_early_rise"] == [17.4, 17.5, 17.7, 20.0]
    for col, ages in coincide.items():
        if col != "level_m_global_early_rise":
            assert ages == [], col

    # (b) the circularity test itself, on the curve's own rows
    circular_rows = sorted(
        float(curve.loc[i, "age_ka"])
        for i in curve.index
        if k.envelope_hi_circularity(curve, float(curve.loc[i, "age_ka"]))[0]
    )
    assert circular_rows == [17.4, 17.5, 17.7, 18.0]
    # 20.0 ka shares its value with early_rise and is still informative
    is_circ, why = k.envelope_hi_circularity(curve, 20.0)
    assert not is_circ
    assert "+15.00 m is not at the measured Manych spillway crest" in why
    assert "level_m_global_early_rise" in why  # the coincidence is still reported

    # ... and in the REPORT, on the baked grid
    flagged = sorted(
        {
            float(e)
            for e in band_report.loc[band_report["envelope_hi_is_circular"], "epoch_ka"]
        }
    )
    assert flagged == [17.5, 18.0]
    for epoch in flagged:
        rows = band_report[band_report["epoch_ka"] == epoch]
        for _i, r in rows.iterrows():
            assert r.verdict_envelope_hi == "spilling_west"
            assert "CIRCULAR" in r.envelope_hi_circularity
            assert "NO INDEPENDENT EVIDENCE" in r.envelope_hi_circularity
            assert "Semikolennykh" in r.envelope_hi_circularity
            assert r.envelope_hi_circularity in r.provenance


@integration
def test_the_upper_edge_window_can_be_asked_for_without_the_circular_rows(band_report):
    """C2's operational half: the query has to be able to exclude them.

    Dropping them is EXPLICIT rather than default — a silent exclusion would be its
    own substitution — and the two windows differ by exactly the circular rows.
    """
    for scenario in k.SCENARIOS:
        raw = k.spill_window_ka(
            band_report, scenario, verdict_column="verdict_envelope_hi"
        )
        clean = k.spill_window_ka(
            band_report,
            scenario,
            verdict_column="verdict_envelope_hi",
            drop_circular=True,
        )
        assert raw == (18.0, 13.75)
        assert clean == (17.0, 13.75)
    # a pre-2026-08-03 report cannot answer the question and says so
    legacy = band_report.drop(columns=["envelope_hi_is_circular"])
    with pytest.raises(ValueError, match="envelope_hi_is_circular"):
        k.spill_window_ka(
            legacy, "defensible", "verdict_envelope_hi", drop_circular=True
        )


@integration
def test_joint_corner_is_not_robust_on_the_mwp1a_ramp(band_report):
    """The lower bound is endorheic everywhere — but not by much on the ramp.

    At five epochs the band's lower edge pushed by the locus's p95 stops being
    ``endorheic`` and reads ``at_sill``: it misses the crest by 7 to 18
    CENTIMETRES.  Those margins are what the caption's "as little as" clause
    exists to stop anyone rounding away, so they are pinned to the millimetre.

    Two independent readers reported this set; one quoted the corner range as
    -0.13..-0.07, which is a pair of EXTREMES and not the five values — the
    15.5 ka entry, -0.176 m, falls outside it.  Re-measured here from the CSV and
    the code, the per-epoch list is the reading that survives.
    """
    soft = sorted(
        {
            float(e)
            for e in band_report.loc[
                ~band_report["verdict_is_jointly_robust"], "epoch_ka"
            ]
        }
    )
    assert soft == [15.0, 15.25, 15.5, 15.75, 16.0]
    want = {
        16.00: -0.130,
        15.75: -0.169,
        15.50: -0.176,
        15.25: -0.125,
        15.00: -0.071,
    }
    for epoch, margin in want.items():
        rows = band_report[band_report["epoch_ka"] == epoch]
        assert len(rows) == 2, epoch  # both scenarios share one band
        for _i, r in rows.iterrows():
            assert r.margin_joint_corner_m == pytest.approx(margin, abs=5e-4), epoch
            assert k._verdict(r.margin_joint_corner_m) == "at_sill", epoch
            assert r.verdict_envelope_lo == "endorheic", epoch


@integration
def test_the_lower_bound_is_closest_at_16_ka_and_it_is_inside_the_sill_bar(
    band_report,
):
    """0.73 m at the locus MEDIAN — the number the caption quotes, and its bar."""
    window = band_report[
        (band_report["epoch_ka"] >= 13.0) & (band_report["epoch_ka"] <= 17.0)
    ]
    closest = window.loc[window["margin_envelope_lo_m"].abs().idxmin()]
    assert float(closest["epoch_ka"]) == 16.0
    assert float(closest["margin_envelope_lo_m"]) == pytest.approx(-0.730, abs=5e-4)
    assert abs(float(closest["margin_envelope_lo_m"])) < k.MANYCH_SILL_GIA_BAR_M


def test_verdict_is_robust_is_unchanged():
    """Its COMPUTATION is byte-identical; only its comment changed.

    ``verdict_is_robust`` covers the locus spread with the LEVEL HELD FIXED.  The
    band is a second, independent axis and must not have leaked into it — if it
    had, ``test_the_only_non_robust_verdict_is_the_mwp1a_epoch`` would be testing
    something else under the same name.
    """
    src = inspect.getsource(k.sill_test)
    assert (
        "    lo_b, hi_b = threshold.bracket_m if threshold.bracket_m else "
        "(surface, surface)\n"
        "    robust = _verdict(lo_b - sill_f1) == _verdict(hi_b - sill_f1) == "
        "_verdict(margin)\n"
    ) in src
    assert "verdict_is_robust=bool(robust)," in src
    assert src.count("\n    robust = ") == 1  # nothing else feeds the flag


@integration
def test_the_two_robustness_flags_answer_different_questions(band_report):
    """They are not redundant, and the measurement says so: disjoint epoch sets."""
    locus_soft = {
        float(e) for e in band_report.loc[~band_report["verdict_is_robust"], "epoch_ka"]
    }
    joint_soft = {
        float(e)
        for e in band_report.loc[~band_report["verdict_is_jointly_robust"], "epoch_ka"]
    }
    assert locus_soft == {14.0}
    assert joint_soft == {15.0, 15.25, 15.5, 15.75, 16.0}
    assert not (locus_soft & joint_soft)


@integration
def test_band_verdict_carries_the_disagreement_instead_of_averaging_it(curve):
    """At the hero epoch the triple straddles the sill, and the token says so."""
    t = k.sill_test(curve, 16.5, "defensible")
    assert t.verdict_envelope_lo == "endorheic"
    assert t.verdict == "spilling_west"
    assert t.verdict_envelope_hi == "spilling_west"
    assert t.envelope_verdict == "endorheic..spilling_west"
    assert not t.verdict_is_envelope_robust
    # ... and where all three agree it collapses to the single word.
    quiet = k.sill_test(curve, 0.0, "defensible")
    assert quiet.envelope_verdict == "endorheic"
    assert quiet.verdict_is_envelope_robust


@integration
def test_both_band_edges_are_converted_with_the_central_s_own_term(curve):
    """Contract sec.8, edge for edge: one ``conversion_m``, three surfaces.

    A per-edge Δz would make part of the band's WIDTH a GIA artefact.  So the
    three F1 surfaces must differ from their F2 levels by exactly the same
    number, and the band's width must survive the conversion unchanged.
    """
    for epoch in (17.0, 16.5, 15.0, 14.0, 13.0):
        t = k.sill_test(curve, epoch, "defensible")
        assert t.surface_envelope_lo_f1_m - t.envelope_lo_f2_m == pytest.approx(t.dz_locus_m)
        assert t.surface_f1_m - t.level_f2_m == pytest.approx(t.dz_locus_m)
        assert t.surface_envelope_hi_f1_m - t.envelope_hi_f2_m == pytest.approx(t.dz_locus_m)
        assert t.envelope_width_m == pytest.approx(
            t.surface_envelope_hi_f1_m - t.surface_envelope_lo_f1_m
        )
        assert t.envelope_lo_f2_m <= t.level_f2_m <= t.envelope_hi_f2_m


@integration
def test_the_caspian_threshold_does_not_catch_a_missing_band(curve):
    """THE ONE DELIBERATE ASYMMETRY, from the side that refuses.

    ``pontic_threshold`` RECORDS an unusable band and carries on, because nothing
    there thresholds on it.  Here both edges become verdicts, so an unusable band
    must stop the frame rather than produce two thirds of a fictional triple.
    """
    row = (curve["age_ka"] - 16.5).abs().idxmin()

    nan_edge = curve.copy()
    nan_edge.loc[row, "unc_low"] = np.nan
    with pytest.raises(bs.MissingEnvelopeError, match="not a number"):
        k.caspian_threshold(nan_edge, 16.5, "defensible")
    with pytest.raises(bs.MissingEnvelopeError):
        k.sill_test(nan_edge, 16.5, "defensible")

    inverted = curve.copy()
    inverted.loc[row, "unc_low"] = 43.0  # above unc_high = +42.9
    with pytest.raises(bs.MissingEnvelopeError, match="inverted"):
        k.caspian_threshold(inverted, 16.5, "defensible")

    dropped = curve.drop(columns=["unc_low", "unc_high"])
    with pytest.raises(bs.MissingEnvelopeError, match="carries no"):
        k.caspian_threshold(dropped, 16.5, "defensible")


@integration
def test_a_degenerate_envelope_stops_the_caspian_frame(curve):
    """C6.  ``unc_low == unc_high == central`` used to pass every check.

    It reported ``width_m = 0.0``, forced ``verdict_envelope_lo == verdict ==
    verdict_envelope_hi``, and made ``verdict_is_envelope_robust`` unconditionally
    True — a band asserting CERTAINTY, dressed as three agreeing readings.  Here
    both edges drive verdicts, so it must stop the frame.

    The shipped curve has no such row (asserted), so this costs the grid nothing.
    """
    lo = pd.to_numeric(curve["unc_low"])
    hi = pd.to_numeric(curve["unc_high"])
    assert not bool((lo == hi).any())

    pinned = curve.copy()
    row = (pinned["age_ka"] - 16.5).abs().idxmin()
    central = float(pinned.loc[row, "level_m_global_defensible"])
    pinned.loc[row, "unc_low"] = central
    pinned.loc[row, "unc_high"] = central
    with pytest.raises(bs.MissingEnvelopeError, match="degenerate"):
        k.caspian_threshold(pinned, 16.5, "defensible")
    with pytest.raises(bs.MissingEnvelopeError, match="claim of CERTAINTY"):
        k.sill_test(pinned, 16.5, "defensible")


@integration
def test_a_threshold_without_a_locus_bracket_refuses_instead_of_reading_certain(
    curve, monkeypatch
):
    """C7.  Both robustness flags were built on a silent ``(surface, surface)``.

    With no ``bracket_m`` the p5/p95 offsets collapse to 0.0: ``verdict_is_robust``
    becomes unconditionally True and ``verdict_is_jointly_robust`` degenerates into
    a copy of ``verdict_envelope_lo``.  Two flags whose whole job is to say "this
    answer is inside the uncertainty" would read "certain", with nothing in the row
    to show it.

    No rule ``caspian_threshold`` implements can produce an unbracketed threshold —
    which is exactly why the fallback's failure was invisible — so the refusal is
    provoked by handing ``sill_test`` a threshold with ``bracket_m=None`` rather
    than by faking a curve row the file could never contain.
    """
    real = k.caspian_threshold(curve, 16.5, "defensible")
    assert real.bracket_m is not None  # every shipped row is bracketed
    stripped = dataclasses.replace(real, bracket_m=None)
    monkeypatch.setattr(k, "caspian_threshold", lambda *a, **kw: stripped)
    with pytest.raises(k.MissingLocusBracketError, match="unconditionally True"):
        k.sill_test(curve, 16.5, "defensible")


def test_the_bracket_guard_sits_in_front_of_the_untouched_fallback():
    """C7's constraint: ``verdict_is_robust``'s arithmetic may not be edited.

    The guard is a refusal placed BEFORE the pre-existing expression, so the two
    lines ``test_verdict_is_robust_is_unchanged`` pins stay byte-identical and the
    fallback becomes provably unreachable instead of silently wrong.
    """
    src = inspect.getsource(k.sill_test)
    assert "if not threshold.bracket_m:" in src
    assert src.index("if not threshold.bracket_m:") < src.index(
        "    lo_b, hi_b = threshold.bracket_m"
    )


@integration
def test_the_caspian_side_refuses_an_epoch_the_curve_has_no_row_for(curve):
    """The shared age guard, from this bake's side.

    Every one of the 43 baked epochs lands on an exact row, so this costs the grid
    nothing — but an epoch between rows must refuse rather than borrow, here as on
    the Pontic side.
    """
    ages = pd.to_numeric(curve["age_ka"]).to_numpy(dtype=float)
    assert float(np.abs(ages - 16.37).min()) > 0.01  # genuinely between rows
    with pytest.raises(bs.CurveAgeGapError, match="no row at 16.37 ka"):
        k.caspian_threshold(curve, 16.37, "defensible")


@integration
def test_a_band_that_does_not_contain_its_central_stops_the_caspian_frame(curve):
    """The refusal the Pontic side records is RAISED here."""
    broken = curve.copy()
    row = (broken["age_ka"] - 16.5).abs().idxmin()
    broken.loc[row, "unc_high"] = 20.0  # the central is +27.0
    with pytest.raises(
        bs.MissingEnvelopeError, match="does not contain the central it was checked"
    ):
        k.caspian_threshold(broken, 16.5, "defensible")


@integration
def test_the_band_is_never_repaired_from_a_neighbouring_row(curve):
    """The specific substitution the class forbids, tested as a behaviour.

    16.5 ka and 16.25 ka carry bands 25.8 m and 27.3 m wide with different edges.
    Breaking one must not quietly produce the other's numbers.
    """
    broken = curve.copy()
    row = (broken["age_ka"] - 16.5).abs().idxmin()
    broken.loc[row, "unc_low"] = np.nan
    with pytest.raises(bs.MissingEnvelopeError):
        k.caspian_threshold(broken, 16.5, "defensible")
    neighbour = k.caspian_threshold(curve, 16.25, "defensible")
    hero = k.caspian_threshold(curve, 16.5, "defensible")
    assert neighbour.envelope_lo_m != hero.envelope_lo_m
    assert neighbour.envelope_hi_m != hero.envelope_hi_m


@integration
def test_the_envelope_provenance_says_only_what_is_true_per_row(band_report, curve):
    """C3.  RAISE-not-compute item 13, corrected: "not recorded" was FALSE.

    What is true, and all that is: the curve has no machine-readable ``unc_source``
    / ``unc_interp_flag`` column.  What is NOT true is that the sources are
    unrecorded — the ``note`` column records them in prose on the rows the headline
    number rests on, and asserting an absence there is a false provenance claim, the
    same class of error as substituting a number.

    So the per-row field distinguishes the two, measured: 4 of the 43 baked epochs
    carry a note naming an edge column (14.0 / 15.0 / 18.0 / 20.0 ka), and every row
    still carries the standing "no column" fact.
    """
    assert set(band_report["envelope_rule"]) == {"envelope_as_carried_in_the_curve"}
    # the standing, TRUE fact, on every row
    for text in set(band_report["envelope_provenance"]):
        assert bs.ENVELOPE_PROVENANCE_NO_COLUMN in text
        assert "not_recorded_in_the_curve" not in text
    # the two columns whose ABSENCE is the reason it is not machine-readable
    assert "unc_source" not in curve.columns
    assert "unc_interp_flag" not in curve.columns
    # ... and the per-row measurement of what the note DOES record
    noted = sorted(
        {
            float(e)
            for e in band_report.loc[
                band_report["envelope_provenance_in_note"], "epoch_ka"
            ]
        }
    )
    assert noted == [14.0, 15.0, 18.0, 20.0]
    fifteen = band_report[band_report["epoch_ka"] == 15.0].iloc[0]
    assert "do not report the provenance as absent" in fifteen.envelope_provenance
    # the sentence in the curve this test exists to stop anyone contradicting
    row = bs._curve_row_index(curve, 15.0)
    note = str(curve.loc[row, "note"])
    assert "unc_low carries the +22 m bottom edge" in note
    assert "+35 m is its TOP EDGE" in note


@integration
def test_a_synthesized_low_scenario_column_is_refused_by_the_curve_validator(curve):
    """RAISE-not-compute item 14: FORBIDDEN, and here is the guard that forbids it.

    The obvious shortcut for "render the lower bound" is to add a level column
    made of ``unc_low`` and bake it like any other scenario.  It cannot be done,
    and not as a matter of taste: max(unc_low) = +22.0 m is below the Manych sill,
    so ``validate_basin_curve``'s Check A rejects the file outright.  The
    conversion belongs at the threshold, never in the curve (contract sec.8).
    """
    assert float(pd.to_numeric(curve["unc_low"]).max()) == pytest.approx(22.0)
    assert float(pd.to_numeric(curve["unc_low"]).max()) < MANYCH_SILL_M
    synthetic = curve.copy()
    synthetic["level_m_global_lowband"] = pd.to_numeric(curve["unc_low"])
    with pytest.raises(ValueError, match="below the measured Manych sill"):
        bs.validate_basin_curve(synthetic, "caspian", "lowband")


#: The 22 columns the report of record carried before the envelope work.  This is
#: the set the phase brief protects ("C may not change the existing SillTest column
#: set"), so it is written down once and asserted, not left to a spot check.
_PRE_ENVELOPE_COLUMNS = (
    "epoch_ka",
    "scenario",
    "level_f2_m",
    "surface_f1_m",
    "dz_locus_m",
    "dz_sill_m",
    "differential_m",
    "sill_f1_m",
    "sill_path_f1_m",
    "margin_m",
    "verdict",
    "margin_path_m",
    "verdict_path",
    "verdict_is_robust",
    "margin_modern_sill_m",
    "verdict_modern_sill",
    "margin_mixed_frame_m",
    "verdict_mixed_frame",
    "gia_time_interpolation",
    "provenance",
    "differential_bracket_lo_m",
    "differential_bracket_hi_m",
)


@integration
def test_the_report_carries_the_envelope_columns_and_keeps_the_old_ones(band_report):
    for column in (
        "verdict_envelope_lo",
        "verdict_envelope_hi",
        "envelope_verdict",
        "verdict_is_envelope_robust",
        "verdict_is_jointly_robust",
        "envelope_lo_f2_m",
        "envelope_hi_f2_m",
        "envelope_width_m",
        "surface_envelope_lo_f1_m",
        "surface_envelope_hi_f1_m",
        "margin_envelope_lo_m",
        "margin_envelope_hi_m",
        "margin_joint_corner_m",
        "envelope_rule",
        "envelope_provenance",
        "envelope_provenance_in_note",
        "envelope_scope",
        "envelope_is_scenario_invariant",
        "envelope_brackets_n_columns",
        "envelope_n_level_columns",
        "envelope_hi_is_circular",
        "envelope_hi_circularity",
    ):
        assert column in band_report.columns, column
    # this task was ADDITIVE over the report of record: all 22 pre-existing columns
    # are still here, under their original names
    for column in _PRE_ENVELOPE_COLUMNS:
        assert column in band_report.columns, column
    assert len(_PRE_ENVELOPE_COLUMNS) == 22


@integration
def test_spill_window_rejects_a_column_that_is_not_a_verdict(band_report):
    with pytest.raises(ValueError, match="unknown verdict column"):
        k.spill_window_ka(band_report, "defensible", verdict_column="margin_m")


# ===========================================================================
# The caption — the one sentence this report licenses
# ===========================================================================


@integration
def test_the_caption_states_both_edges_and_names_the_envelope_correctly(band_report):
    """THE DELIVERABLE SENTENCE: both edges, the right name, the circular refusal."""
    caption = k.sill_caption(band_report)
    # the central windows
    assert "defensible 16.5->14.25 ka" in caption
    assert "maximal 16.75->14.25 ka" in caption
    # C1 — named for what it is, and its scope stated
    assert "NOT a per-scenario band" in caption
    assert "ONE pair carried per curve ROW" in caption
    assert "bracket 6 of 6 level columns at once" in caption
    assert "inter-school scenario envelope as carried in the curve" in caption
    assert "the same for every scenario in this report (defensible/maximal)" in caption
    # BOTH edges, with their F2 values
    assert "At its LOWER edge (+22.0 m F2 at 16 ka)" in caption
    assert "never cleared the Manych at any of the 17 epochs of 17-13 ka" in caption
    assert "as little as 0.73 m at 16 ka" in caption
    assert "inside the sill's own +/-2 m GIA bar" in caption
    assert "At its UPPER edge (+50.0 m F2 at the same epoch)" in caption
    assert (
        "clears the crest over defensible 17->13.75 ka, maximal 17->13.75 ka"
        in caption
    )
    assert "counting only rows where that edge is independent of the sill" in caption
    # C2 — the circular rows named and refused, not silently dropped
    assert "The upper edge ALSO reads spilling_west at 17.5/18 ka" in caption
    assert "NOT independent evidence and is not cited as any" in caption
    assert "unc_high stands on the Manych crest itself" in caption
    assert "Semikolennykh et al. 2022" in caption
    assert "testing it against the sill is circular" in caption
    # C3 — the provenance claim that survives measurement
    assert (
        "NOT_MACHINE_READABLE:the curve has no unc_source/unc_interp_flag" in caption
    )
    assert (
        "4 of 43 epochs in this report record their edges' sources in prose"
        in caption
    )
    assert "'not recorded in the curve' would be false" in caption
    assert "UNKNOWN:not_recorded_in_the_curve" not in caption
    # the word RAISE-not-compute item 13 forbids
    assert "published" not in caption


@integration
def test_the_caption_never_prints_an_epoch_without_saying_whose_it_is(band_report):
    """C8.  It used to take idxmin over ALL scenarios and print a bare epoch.

    Printed immediately after per-scenario spill windows, that epoch could only be
    read as belonging to whichever scenario was named last.  Printing one epoch is
    legitimate only if the lower edge is the same for every scenario — which is a
    property of this curve and therefore a thing to CHECK.  When the check fails the
    caption names the scenario instead.
    """
    caption = k.sill_caption(band_report)
    assert "the same for every scenario in this report" in caption
    assert "max spread across scenarios 0.00e+00 m" in caption

    # force the other branch: make one scenario's lower edge differ
    skewed = band_report.copy()
    mask = (skewed["scenario"] == "maximal") & (skewed["epoch_ka"] == 16.0)
    skewed.loc[mask, "margin_envelope_lo_m"] = -0.10
    other = k.sill_caption(skewed)
    assert "on the maximal scenario" in other
    assert "the lower edge is NOT the same across scenarios" in other


@integration
def test_the_caption_names_only_the_epochs_the_report_actually_covers(curve):
    """A two-epoch report must not license a sentence about seventeen."""
    narrow = k.sill_report(curve, [16.5, 16.25], k.SCENARIOS)
    caption = k.sill_caption(narrow)
    assert "at any of the 2 epochs of 16.5-16.25 ka" in caption
    # ... and the "inside the bar" clause is a COMPARISON, not a reassurance:
    # over those two epochs the lower edge misses by 2.24 m, outside the bar.
    assert "OUTSIDE the sill's own +/-2 m GIA bar" in caption
    # no circular row in this slice, and the caption SAYS so rather than staying
    # silent — a missing clause is indistinguishable from an unasked question
    assert (
        "No row of this report has an upper edge standing on the Manych crest"
        in caption
    )


@integration
def test_the_caption_refuses_a_report_that_does_not_cover_the_window(curve):
    outside = k.sill_report(curve, [22.0, 20.0], ("defensible",))
    with pytest.raises(ValueError, match="covers no epoch"):
        k.sill_caption(outside)


@integration
def test_the_caption_refuses_a_report_that_cannot_answer_the_circularity_question(
    band_report,
):
    """A pre-2026-08-03 report has no circularity column, so it licenses no sentence."""
    legacy = band_report.drop(columns=["envelope_hi_is_circular"])
    with pytest.raises(ValueError, match="envelope_hi_is_circular"):
        k.sill_caption(legacy)


@integration
def test_the_caption_refuses_to_assert_a_lower_bound_it_did_not_measure(band_report):
    """If the lower edge ever spills, the sentence is false and is not printed."""
    lying = band_report.copy()
    lying.loc[lying["epoch_ka"] == 16.0, "verdict_envelope_lo"] = "spilling_west"
    with pytest.raises(ValueError, match="lower edge spills"):
        k.sill_caption(lying)


# ===========================================================================
# main() — asking a question must not write; BAKING must
# ===========================================================================


@integration
def test_sill_report_csv_write_is_gated(tmp_path):
    """``--sill-report-only`` used to clobber the report as a SIDE EFFECT.

    That CSV is where the renderer's sill expectation is read from, so "compute
    this in seconds and show me" quietly rewrote the number the next frame is
    checked against.  The report is still computed, printed and returned; writing
    it from a READ-ONLY QUERY needs ``write_report=True`` (``--write`` on the CLI).
    """
    report = k.main(epochs=[16.5, 16.25], write_rasters=False, out_dir=tmp_path)
    assert len(report) == 2 * len(k.SCENARIOS)
    assert not (tmp_path / k.SILL_REPORT_CSV.name).exists()
    assert list(tmp_path.iterdir()) == []


@integration
def test_the_write_flag_is_what_turns_the_report_write_on(tmp_path):
    report = k.main(
        epochs=[16.5, 16.25], write_rasters=False, write_report=True, out_dir=tmp_path
    )
    csv = tmp_path / k.SILL_REPORT_CSV.name
    assert csv.exists()
    written = pd.read_csv(csv)
    assert len(written) == len(report) == 2 * len(k.SCENARIOS)
    assert "verdict_envelope_lo" in written.columns
    assert not (tmp_path / "topo").exists()


@integration
@needs_dem
def test_a_real_bake_refreshes_the_report_without_being_asked(tmp_path):
    """C5.  THE GATE WAS ON THE WRONG THING and this is the half that was broken.

    ``python scripts/bake_khvalynian.py --epochs 16.5`` wrote rasters and left the
    PREVIOUS report on disk — and the renderer reads its sill expectation out of
    that report.  Fresh rasters beside a stale report is a silent frame/verdict
    mismatch, which is strictly worse than the side effect the gate was added to
    stop.  So ``write_report=None`` (the default) means "follow ``write_rasters``".

    Asserted through a real bake of ONE epoch, with a STALE file planted first so
    the test can tell "refreshed" from "never written".
    """
    stale = tmp_path / k.SILL_REPORT_CSV.name
    stale.write_text("epoch_ka,scenario\nSTALE,STALE\n", encoding="utf-8")
    report = k.main(epochs=[16.5], out_dir=tmp_path)
    assert (tmp_path / "topo" / "khv_16.5ka.tif").exists()
    written = pd.read_csv(stale)
    assert "STALE" not in set(written.get("scenario", []))
    assert len(written) == len(report) == len(k.SCENARIOS)
    assert "envelope_hi_is_circular" in written.columns


def test_the_gate_follows_the_bake_and_write_can_only_add_a_write():
    """The gate's contract, asserted on the signature and the CLI wiring.

    ``--write`` may turn a write ON (a read-only query that wants to save) and may
    never turn one OFF — a real bake owes a fresh report whether the flag is there
    or not, so the CLI passes ``None`` rather than ``False``.
    """
    src = inspect.getsource(k._cli)
    assert '"--write"' in src
    assert "write_report=True if args.write else None" in src
    assert "write_report=args.write" not in src
    assert inspect.signature(k.main).parameters["write_report"].default is None
    body = inspect.getsource(k.main)
    assert (
        "write_report = bool(write_rasters) if write_report is None "
        "else bool(write_report)"
    ) in body


# ===========================================================================
# The ICE-6G bracket after the move — same numbers, one implementation
# ===========================================================================


def test_the_bracket_is_the_one_in_bake_black_sea():
    """It was WRITTEN here and now lives next to the Pontic threshold that needs it.

    Two copies of a time-interpolation rule is two rules that can disagree about
    what "linear in time" means, so this asserts there is only ONE definition —
    by source, because an ``is`` check would be defeated by the ``importlib.reload``
    the sibling test file performs.
    """
    source = Path(k.__file__).read_text(encoding="utf-8")
    assert "def _ice6g_bracket(" not in source
    assert "def ice6g_ages_ka(" not in source
    assert "    _ice6g_bracket,\n" in source  # imported from bake_black_sea
    assert "    ice6g_ages_ka,\n" in source
    assert k._ice6g_bracket(16.5) == (16.5, 16.5, 0.0)


@integration
def test_an_off_grid_epoch_is_tagged_as_interpolated_on_the_threshold(curve):
    """Every quarter-ka epoch of the window blends two ICE-6G epochs, and says so."""
    on_grid = k.caspian_threshold(curve, 16.5, "defensible")
    assert on_grid.ice6g_interpolated is False
    assert "no time interpolation" in on_grid.ice6g_time_note

    off_grid = k.caspian_threshold(curve, 16.25, "defensible")
    assert off_grid.ice6g_interpolated is True
    assert "linear in time between ICE-6G 16 and 16.5 ka" in off_grid.ice6g_time_note


# ===========================================================================
# THE MESHER TAG SET — what a 4D reader needs and the bake used to throw away
# ===========================================================================
#
# Two defects are pinned below, and they are different KINDS of defect.
#
# The nodata tag was not FALSE — 101 really is the number of ICE-6G model cells
# whose Topo is the ice surface.  It was UNLABELLED, in a file where the obvious
# reading of "cells" is the raster's own, and under that reading it understates the
# hole by a factor of ~3000.  So these tests assert both counts are carried under
# names that say which unit each is in, and that the raster count is measured off the
# pixels rather than derived from the model grid — the derivation does not work, and
# the ratio clause says by how much.
#
# The water levels were ABSENT (Pontic), UNNAMED (Mediterranean) or reachable only by
# parsing a 1 299-character English sentence (Caspian).  These tests assert a machine
# can now read all three, and that where one cannot be established the tag says so
# with that epoch's own measured reason instead of holding a number.


def _tiny_transform():
    """1-degree cells anchored at the scene's NW corner, so a bbox is readable."""
    from rasterio.transform import from_origin

    return from_origin(25.5, 54.0, 1.0, 1.0)


def _sheets(tags):
    return {b: json.loads(tags[f"water_level_{b}_json"]) for b in k.BASIN_SHEETS}


# ---------------------------------------------------------------------------
# The census arithmetic, on arrays small enough to count by hand
# ---------------------------------------------------------------------------


def test_census_counts_cells_and_boxes_them_by_cell_edges():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = True
    mask[2, 3] = True
    c = k.nodata_census_from_mask(mask, _tiny_transform(), -9999.0)
    assert (c.cells, c.cells_total) == (2, 20)
    assert (c.row_min, c.row_max, c.col_min, c.col_max) == (1, 2, 2, 3)
    # EDGES, not centres: the box must CONTAIN both cells, so it runs from the NW
    # corner of (row 1, col 2) to the SE corner of (row 2, col 3).  A centre-based
    # box would understate the hole by half a cell on every side.
    assert (c.lon_min, c.lon_max) == (27.5, 29.5)
    assert (c.lat_min, c.lat_max) == (51.0, 53.0)
    assert c.touches_edge is False
    assert c.edges_touched == ()


def test_census_reports_each_border_separately_and_says_which_are_touched():
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, :3] = True  # north border
    mask[:2, 0] = True  # west border
    c = k.nodata_census_from_mask(mask, _tiny_transform(), -9999.0)
    assert (c.edge_north, c.edge_west, c.edge_south, c.edge_east) == (3, 2, 0, 0)
    assert c.touches_edge is True
    assert set(c.edges_touched) == {"north", "west"}


def test_an_absent_hole_is_a_STATE_and_never_a_row_zero():
    """The refuse-or-disclose rule applied to a STATE rather than to a float.

    ``row_min = 0`` is a real place — the top row of the frame — so a census that
    spelled "no hole" as a box at the origin would be indistinguishable from a hole
    in the corner.  Absence is ``None``, and the tags built from it say NONE /
    NOT_APPLICABLE in words.
    """
    c = k.nodata_census_from_mask(np.zeros((4, 5), dtype=bool), _tiny_transform(), -1.0)
    assert c.cells == 0
    assert (c.row_min, c.col_min, c.lon_min, c.lat_max) == (None, None, None, None)
    assert c.touches_edge is None  # not False: the question does not apply

    tags = k.nodata_tags(
        16.5,
        c,
        model_unknown=0,
        model_total=24960,
        model_grid="test",
        gia_nan_policy="nodata",
    )
    assert tags["nodata_bbox_rowcol"].startswith("NONE:")
    assert tags["nodata_bbox_lonlat"].startswith("NONE:")
    assert tags["nodata_touches_raster_edge"].startswith("NOT_APPLICABLE:")
    assert tags["nodata_fill_policy"].startswith("NOT_APPLICABLE:")


def test_census_refuses_a_non_2d_mask():
    with pytest.raises(ValueError, match="2-D"):
        k.nodata_census_from_mask(np.zeros(4, dtype=bool), _tiny_transform(), -1.0)


# ---------------------------------------------------------------------------
# The tag that lied: two units, both named, both counted
# ---------------------------------------------------------------------------


def test_both_counts_are_carried_and_their_names_say_which_unit_each_is_in():
    mask = np.zeros((40, 50), dtype=bool)
    mask[0, :10] = True
    c = k.nodata_census_from_mask(mask, _tiny_transform(), -9999.0)
    tags = k.nodata_tags(
        16.5,
        c,
        model_unknown=101,
        model_total=24960,
        model_grid="ICE-6G_C VM5a, cell 10.0x10.0 arcmin",
        gia_nan_policy="nodata",
    )
    # the raster-cell count, measured on the raster
    assert tags["nodata_raster_cells"] == "10"
    assert tags["nodata_raster_cells_total"] == "2000"
    # the MODEL-cell count, under a name that says which grid it counts
    assert tags["gia_unknown_ice6g_model_cells"] == "101"
    assert tags["gia_unknown_ice6g_model_cells_total"] == "24960"
    # and the legacy name is disambiguated rather than left to be misread
    assert "MODEL cells of 10 arcmin (NOT raster cells)" in (
        tags["gia_unknown_ice6g_cells_unit"]
    )
    assert "nodata_raster_cells" in tags["gia_unknown_ice6g_cells_unit"]
    note = tags["nodata_units_note"]
    assert "TWO DIFFERENT UNITS" in note
    assert "gia_unknown_ice6g_model_cells=101 of 24960" in note
    assert "nodata_raster_cells=10 of 2000" in note
    assert "MEASURED FROM THIS FILE" in note


def test_a_hole_on_the_edge_is_declared_as_extrapolation_not_interpolation():
    """Why the edge contact is worth a tag at all, said in the data.

    A hole enclosed by data can be interpolated across.  A hole that reaches the
    frame's border cannot: there is nothing on the far side, so any fill is an
    extrapolation, and a mesher that quietly in-paints it is inventing terrain.  The
    shipped rasters are in the second case, on two borders.
    """
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :4] = True
    c = k.nodata_census_from_mask(mask, _tiny_transform(), -9999.0)
    tags = k.nodata_tags(
        16.5,
        c,
        model_unknown=1,
        model_total=10,
        model_grid="x",
        gia_nan_policy="nodata",
    )
    assert tags["nodata_touches_raster_edge"] == "true"
    assert tags["nodata_fill_policy"].startswith("EXTRAPOLATION:")
    assert "north border 4 of 10" in tags["nodata_fill_policy"]

    interior = np.zeros((10, 10), dtype=bool)
    interior[4:6, 4:6] = True
    ci = k.nodata_census_from_mask(interior, _tiny_transform(), -9999.0)
    ti = k.nodata_tags(
        16.5,
        ci,
        model_unknown=1,
        model_total=10,
        model_grid="x",
        gia_nan_policy="nodata",
    )
    assert ti["nodata_touches_raster_edge"] == "false"
    assert ti["nodata_fill_policy"].startswith("INTERIOR:")


#: Digit-bearing text ``nodata_tags`` is allowed to contain literally.  Every entry is
#: a NAME or a structural convention, not a figure about a raster: a name cannot go
#: stale when the data changes, and a figure can.
_ALLOWED_LITERAL_DIGITS = ("ICE-6G", "ice6g", "band 1", "0-based")


def test_the_templates_hardcode_no_figure_about_a_raster():
    """A template interpolates; it does not carry a figure of its own.

    Six sentences have been retracted in this project for shipping a number that was
    impossible on the set the sentence itself named, so this is checked structurally
    rather than by comparing two sample outputs: every string literal in
    ``nodata_tags`` is scanned, and a digit in one has to belong to a NAME.

    It has already caught one: the note used to say "10-arcmin MODEL cells" and
    "15-arcsec RASTER cells".  Both were true of the shipped pair of grids and both
    would have survived a resample and started lying — the same failure mode, in the
    same sentence, as the tag this pass exists to repair.  Both are now measured, one
    off the ICE-6G coordinate vectors and one off the raster's own transform.
    """
    import ast

    tree = ast.parse(inspect.getsource(k.nodata_tags).lstrip())
    # An f-string's format spec (".6f") is a Constant too, and it is a rounding
    # instruction rather than a claim about anything, so it is excluded by position.
    specs = {
        id(sub)
        for node in ast.walk(tree)
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None
        for sub in ast.walk(node.format_spec)
    }
    offenders = []
    for node in ast.walk(tree):
        if id(node) in specs:
            continue
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        text = node.value
        for allowed in _ALLOWED_LITERAL_DIGITS:
            text = text.replace(allowed, "")
        if any(ch.isdigit() for ch in text):
            offenders.append(node.value)
    assert not offenders, f"hardcoded figures in a template: {offenders}"


def test_the_two_grid_resolutions_in_the_note_are_the_measured_ones():
    """Both sides of the unit comparison are read, not recited.

    Constructed on a grid that is deliberately NOT the shipped 15-arcsec one, so a
    recited "15 arcsec" would be visibly wrong.
    """
    from rasterio.transform import from_origin

    c = k.nodata_census_from_mask(
        np.zeros((10, 10), dtype=bool), from_origin(25.5, 54.0, 0.25, 0.25), -9999.0
    )
    assert c.cell_lat == pytest.approx(0.25) and c.cell_lon == pytest.approx(0.25)
    note = k.nodata_tags(
        16.5,
        c,
        model_unknown=0,
        model_total=7,
        model_grid="g",
        gia_nan_policy="nodata",
    )["nodata_units_note"]
    assert "900 arcsec" in note  # 0.25 deg, this fixture's own cell
    assert "15-arcsec" not in note and "10-arcmin" not in note


# ---------------------------------------------------------------------------
# Three machine-readable water surfaces
# ---------------------------------------------------------------------------


@integration
def test_all_three_water_sheets_are_machine_readable_at_a_hero_epoch(curve):
    """The Caspian was prose, the Mediterranean was unnamed, the Pontic was absent."""
    tags = k.water_level_tags(16.5, caspian_curve=curve)
    assert set(tags["water_level_basins"].split(",")) == set(k.BASIN_SHEETS)
    sheets = _sheets(tags)
    for basin, sheet in sheets.items():
        assert sheet["basin"] == basin
        assert sheet["unit"] == "m"
        assert sheet["epoch_ka"] == 16.5
        assert "F1" in sheet["frame"] and "PRESENT-DAY geoid" in sheet["frame"]
        assert "H_paleo < water_level_m" in sheet["comparable_with"]
        for record in sheet["levels"]:
            assert record["status"] == "OK"
            assert isinstance(record["water_level_m"], float)
            assert record["rule"] and "rule=" in record["rule"]
            assert record["refusal"] is None

    caspian = {r["scenario"]: r["water_level_m"] for r in sheets["caspian"]["levels"]}
    assert set(caspian) == set(k.SCENARIOS)
    assert caspian["defensible"] == pytest.approx(29.944, abs=1e-3)
    assert caspian["maximal"] == pytest.approx(36.344, abs=1e-3)
    # the Pontic stands ~48 m below the Caspian at 16.5 ka: that gap IS the hero shot,
    # and it is why one scalar sea level cannot be laid across this frame.
    pontic = {r["scenario"]: r["water_level_m"] for r in sheets["pontic"]["levels"]}
    assert pontic["gradual"] == pytest.approx(-18.020, abs=1e-3)
    assert caspian["defensible"] - pontic["gradual"] > 40.0


@integration
def test_the_mediterranean_sheet_is_the_negated_offset_and_finally_says_so():
    """It always shipped as ``-gia_sea_surface_offset_m`` with nothing naming it.

    Contract §2: ``-offset(t)`` is the EXACT land/sea companion of this DEM for an
    ICE-6G-ocean cell, and it needs no uniformity assumption.  The number was already
    in the file; what was missing was any tag saying that is what it is, so a mesher
    had to have read the contract to use it.
    """
    for epoch in (18.0, 16.5, 9.0, 0.0):
        sheet = _sheets(k.water_level_tags(epoch))["mediterranean"]
        assert len(sheet["levels"]) == 1, "a global mean has no competing schools"
        record = sheet["levels"][0]
        assert record["scenario"] == "all"
        assert record["water_level_m"] == pytest.approx(-k.sea_surface_offset(epoch))
        assert "ocean_offset" in record["rule"] and "EXACT" in record["rule"]
        assert "gia_sea_surface_offset_m" in record["rule"]
        assert "no competing schools" in sheet["scenario_axis"]


@integration
def test_after_reconnection_the_pontic_and_the_mediterranean_share_one_surface():
    """Not a coincidence for a reader to re-derive: the rule field states it."""
    sheets = _sheets(k.water_level_tags(0.0))
    med = sheets["mediterranean"]["levels"][0]["water_level_m"]
    for record in sheets["pontic"]["levels"]:
        assert record["water_level_m"] == pytest.approx(med)
        assert "RECONNECTED" in record["rule"]
    assert "rule_changes_with_epoch" in sheets["pontic"]


@integration
def test_the_pontic_rule_changes_with_epoch_and_the_tag_names_which_applied():
    rules = {}
    for epoch in (18.0, 16.5, 0.0):
        record = _sheets(k.water_level_tags(epoch))["pontic"]["levels"][0]
        rules[epoch] = record["rule"].split()[0]
    assert rules == {18.0: "rule=locus", 16.5: "rule=marker", 0.0: "rule=ocean_offset"}


# ---------------------------------------------------------------------------
# The Pontic gap: an explicit null and the measured reason, never an interpolant
# ---------------------------------------------------------------------------


@integration
def test_the_pontic_is_refused_at_exactly_the_eight_quarter_ka_epochs():
    """Measured off the two files, not asserted from the curve's nominal step.

    The Pontic curve is on a 0.5 ka grid (plus odd rows at 7.15, 9.4, 10.9 and
    14.7 ka) and this bake samples 0.25 ka across 17-13 ka, so eight of the 43 baked
    epochs have no row at all.  Which eight is a property of two files.
    """
    refused = []
    for epoch in k.EPOCHS_KA:
        sheet = _sheets(k.water_level_tags(epoch))["pontic"]
        nulls = [r for r in sheet["levels"] if r["water_level_m"] is None]
        if nulls:
            refused.append(epoch)
            assert len(nulls) == len(k.PONTIC_SCENARIOS)
    assert refused == [16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25]
    assert len(refused) == 8


@integration
def test_a_refused_pontic_level_is_null_with_a_reason_and_never_a_number():
    for record in _sheets(k.water_level_tags(16.75))["pontic"]["levels"]:
        assert record["water_level_m"] is None
        assert record["bracket_m"] is None
        assert record["status"] == "REFUSED"
        assert record["rule"] is None
        assert "No value is written and none is guessed" in record["refusal"]


@integration
def test_the_refusal_names_the_rule_boundary_and_the_width_of_the_choice():
    """The two reasons interpolation is forbidden here, MEASURED at this epoch.

    Contract §5a-bis and §7.1: the rows bracketing 16.75 ka are converted by different
    estimators — ``locus`` at 17.0 ka, ``marker`` at 16.5 ka — so an interpolant
    belongs to neither; and their levels are 28.3 m apart, a gap a nearest-row
    fallback would resolve by DataFrame row order.
    """
    reason = _sheets(k.water_level_tags(16.75))["pontic"]["levels"][0]["refusal"]
    assert "DIFFERENT rules" in reason
    assert "marker at 16.5 ka" in reason and "locus at 17 ka" in reason
    assert "28.30 m apart" in reason
    assert "-30.00 m at 16.5 ka" in reason and "-58.30 m at 17 ka" in reason
    # and it refuses to call either bracketing row "the nearest": both are 0.25 ka
    # away, which is exactly the tie DataFrame order would break.
    assert "no such thing as THE nearest row here" in reason


@integration
def test_the_refusal_does_not_reuse_one_epochs_argument_at_another():
    """14.75 ka is bracketed by two ``marker`` rows whose levels agree to 0.00 m.

    The rule-boundary / 28.3 m argument is TRUE at 16.75 ka and false here, so a
    canned reason string would be a false claim at most of the eight refused epochs.
    Each epoch's reason is built from that epoch's own measurements.
    """
    reason = _sheets(k.water_level_tags(14.75))["pontic"]["levels"][0]["refusal"]
    assert "DIFFERENT rules" not in reason
    assert "both bracketing rows are converted by rule marker" in reason
    assert "0.00 m apart" in reason
    assert "28.30" not in reason


# ---------------------------------------------------------------------------
# The two shipped precisions of one number
# ---------------------------------------------------------------------------


@integration
def test_the_two_shipped_precisions_of_the_caspian_surface_agree(curve):
    """+29.944 in one prose tag and +29.94 in another are ONE number, rounded twice.

    Both strings already ship on every raster — ``threshold_<scenario>`` writes the
    surface to 3 dp and ``sill_test_<scenario>`` writes it to 2 — and nothing checked
    they were the same quantity.  The machine-readable tag ends the ambiguity by
    carrying the float itself; this asserts both prose renderings are roundings of
    exactly that float, at all 43 epochs in both scenarios, so a reader who quotes
    either one is quoting the tag.
    """
    fine_re = re.compile(r"=\s*([+-]\d+\.\d+) m \[p5-p95")
    coarse_re = re.compile(r"surface_F1 ([+-]\d+\.\d+) m")
    checked = 0
    for epoch in k.EPOCHS_KA:
        sheet = _sheets(k.water_level_tags(epoch, caspian_curve=curve))["caspian"]
        exact = {r["scenario"]: r["water_level_m"] for r in sheet["levels"]}
        for scenario in k.SCENARIOS:
            fine = fine_re.search(
                k.caspian_threshold(curve, epoch, scenario).provenance
            )
            coarse = coarse_re.search(k.sill_test(curve, epoch, scenario).provenance)
            assert fine and coarse, f"unparseable provenance at {epoch} ka/{scenario}"
            value = exact[scenario]
            # Both prose strings must be renderings OF THIS FLOAT.  Compared as
            # strings, and never by re-rounding one into the other's precision --
            # see the double-rounding test below for why that distinction is not
            # pedantry.
            assert fine.group(1) == f"{value:+.3f}", (
                f"{epoch} ka/{scenario}: threshold_* says {fine.group(1)}, the tag "
                f"carries {value!r}"
            )
            assert coarse.group(1) == f"{value:+.2f}", (
                f"{epoch} ka/{scenario}: sill_test_* says {coarse.group(1)}, the tag "
                f"carries {value!r}"
            )
            checked += 1
    assert checked == len(k.EPOCHS_KA) * len(k.SCENARIOS)


@integration
def test_re_rounding_one_prose_precision_into_the_other_is_off_by_a_centimetre(curve):
    """WHY the machine-readable float had to be added, in one epoch.

    At 14.25 ka the surface is 30.32546... m.  ``threshold_*`` renders it +30.325 and
    ``sill_test_*`` renders it +30.33; both are correct roundings of that one number.
    But a reader who takes the 3 dp string and rounds it to 2 gets **30.32**, because
    "30.325" as a decimal literal is a different float from the one that produced it.
    Neither prose tag is wrong and the pair still cannot be reconciled by arithmetic
    on the printed text — which is exactly what a machine-readable value is for.
    """
    value = k.caspian_threshold(curve, 14.25, "defensible").threshold_m
    assert f"{value:+.3f}" == "+30.325"
    assert f"{value:+.2f}" == "+30.33"
    assert round(float("30.325"), 2) == 30.32  # the trap, stated
    record = _sheets(k.water_level_tags(14.25, caspian_curve=curve))["caspian"]["levels"][0]
    assert record["water_level_m"] == value


@integration
def test_the_hero_epochs_two_precisions_are_the_ones_that_shipped(curve):
    """The exact pair named in the audit, so a regression is recognisable by sight."""
    assert (
        "= +29.944 m [p5-p95"
        in k.caspian_threshold(curve, 16.5, "defensible").provenance
    )
    assert "surface_F1 +29.94 m" in k.sill_test(curve, 16.5, "defensible").provenance
    record = _sheets(k.water_level_tags(16.5, caspian_curve=curve))["caspian"]["levels"][0]
    assert f"{record['water_level_m']:+.3f}" == "+29.944"
    assert f"{record['water_level_m']:+.2f}" == "+29.94"


# ---------------------------------------------------------------------------
# ASCII, determinism, idempotency
# ---------------------------------------------------------------------------


def test_a_non_ascii_tag_value_is_refused_because_gdal_corrupts_it():
    """Measured: a U+2014 written through ``update_tags`` reads back as U+FFFD.

    The shipped ``threshold_*`` tags already carry one such replacement character
    where an em dash was meant.  A tag is data, and silently mangled data is the
    defect class this phase exists to remove.
    """
    with pytest.raises(ValueError, match="non-ASCII"):
        k._ascii_tag("demo", "an em dash — here")
    assert k._ascii_tag("demo", "plain ascii") == "plain ascii"


def test_borrowed_non_ascii_text_is_escaped_losslessly_rather_than_edited():
    """``CurveAgeGapError``'s own message contains an em dash.

    It is somebody else's sentence, quoted into a refusal, so it may be neither
    mangled by GDAL nor silently rewritten with a hyphen.  ``ensure_ascii`` escapes it
    on the way in and ``json.loads`` restores the exact character on the way out.
    """
    payload = k._json_tag({"quote": "before — after"})
    assert payload.isascii()
    assert "\\u2014" in payload
    assert json.loads(payload)["quote"] == "before — after"


@integration
def test_every_tag_this_pass_writes_is_ascii_and_every_json_tag_parses(curve):
    census = k.nodata_census_from_mask(
        np.zeros((10, 10), dtype=bool), _tiny_transform(), -9999.0
    )
    tags = k.mesher_tags(
        16.75, census, pixel_sha256="0" * 64, dem_path=k.KHV_DEM, caspian_curve=curve
    )
    for name, value in tags.items():
        assert name.isascii() and value.isascii(), name
    for basin in k.BASIN_SHEETS:
        json.loads(tags[f"water_level_{basin}_json"])


@integration
def test_the_tag_set_is_a_pure_function_of_its_inputs(curve):
    """Idempotency of the in-place pass reduces to determinism of this dict.

    ``sort_keys`` makes the JSON byte-stable, so a second run over an already-tagged
    raster computes identical strings, compares equal, and writes nothing.
    """
    census = k.nodata_census_from_mask(
        np.zeros((10, 10), dtype=bool), _tiny_transform(), -9999.0
    )
    kwargs = dict(pixel_sha256="a" * 64, dem_path=k.KHV_DEM, caspian_curve=curve)
    assert k.mesher_tags(16.5, census, **kwargs) == k.mesher_tags(16.5, census, **kwargs)


@integration
def test_the_tag_set_indexes_itself(curve):
    census = k.nodata_census_from_mask(
        np.zeros((10, 10), dtype=bool), _tiny_transform(), -9999.0
    )
    tags = k.mesher_tags(
        16.5, census, pixel_sha256="a" * 64, dem_path=k.KHV_DEM, caspian_curve=curve
    )
    assert set(tags["mesher_tags"].split(",")) == set(tags)
    assert tags["mesher_tag_schema"] == k.MESHER_TAG_SCHEMA


# ---------------------------------------------------------------------------
# The write path: future bakes carry the same tags, measured on their own pixels
# ---------------------------------------------------------------------------


@integration
def test_a_fresh_bake_carries_the_mesher_tag_set(small_dem, tmp_path, curve):
    out = k.build_paleo_dem(
        16.5, dem_path=small_dem, out_dir=tmp_path / "topo", caspian_curve=curve
    )
    with rasterio.open(out) as ds:
        tags = ds.tags()
    for key in (
        "mesher_tag_schema",
        "pixel_sha256",
        "dem_source_sha256",
        "nodata_raster_cells",
        "nodata_bbox_rowcol",
        "nodata_touches_raster_edge",
        "gia_unknown_ice6g_model_cells",
        "nodata_units_note",
        "water_level_caspian_json",
        "water_level_mediterranean_json",
        "water_level_pontic_json",
    ):
        assert key in tags, key
    # the legacy tag and its newly-named twin carry the same number, as they must:
    # the defect was the missing unit, not the value.
    assert tags["gia_unknown_ice6g_cells"] == tags["gia_unknown_ice6g_model_cells"]
    assert tags["vertical_frame"].startswith("F0")


@integration
def test_the_pixel_sha256_tag_is_the_hash_of_the_band_as_read_back(small_dem, tmp_path):
    """Accumulated strip by strip while writing; must equal the finished band's hash.

    If it did not, the additive pass would be comparing its own fresh read against a
    number that meant something else, and the whole safety argument would be void.
    """
    out = k.build_paleo_dem(
        16.5, dem_path=small_dem, out_dir=tmp_path / "topo", strip_rows=7
    )
    with rasterio.open(out) as ds:
        band = np.ascontiguousarray(ds.read(1))
        assert ds.tags()["pixel_sha256"] == hashlib.sha256(band.tobytes()).hexdigest()


@integration
def test_the_bake_measures_nodata_on_its_own_pixels_not_on_the_model_grid(
    tmp_path, curve
):
    """A fixture inside the ice hole, where the two counts are both non-zero and differ.

    This is the defect in one assertion: the model-cell count and the raster-cell
    count are different numbers about different grids, and the raster one can only
    come from the raster.
    """
    from rasterio.transform import from_origin

    path = tmp_path / "nw.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(26.0, 53.9, 0.01, 0.01),
        nodata=-32767.0,
    ) as dst:
        dst.write(np.zeros((60, 60), dtype=np.float32), 1)
    out = k.build_paleo_dem(
        18.0, dem_path=path, out_dir=tmp_path / "topo", caspian_curve=curve
    )
    with rasterio.open(out) as ds:
        tags = ds.tags()
        measured = int((ds.read(1) == np.float32(k.KHV_NODATA_M)).sum())
    assert int(tags["nodata_raster_cells"]) == measured > 0
    assert int(tags["gia_unknown_ice6g_model_cells"]) == 101
    assert int(tags["nodata_raster_cells"]) != int(tags["gia_unknown_ice6g_model_cells"])


# ---------------------------------------------------------------------------
# The additive pass over an already-baked raster
# ---------------------------------------------------------------------------


@pytest.fixture()
def baked(small_dem, tmp_path, curve):
    """A raster in the SHIPPED state: the legacy tag set and nothing else.

    Built by baking normally — with the same ``provenance`` mapping ``main()`` passes,
    so the sill verdicts are present as they are on the shipped rasters — and then
    stripping the mesher keys back off.  The fixture is therefore the real pre-pass
    state rather than a hand-written approximation, and ``mesher_tags`` naming its own
    keys is what makes the strip exact.
    """
    sill = {}
    for scenario in k.SCENARIOS:
        test = k.sill_test(curve, 18.0, scenario)
        sill[f"threshold_{scenario}"] = k.caspian_threshold(
            curve, 18.0, scenario
        ).provenance
        sill[f"sill_test_{scenario}"] = test.provenance
        sill[f"sill_verdict_{scenario}"] = test.verdict
    out = k.build_paleo_dem(
        18.0,
        dem_path=small_dem,
        out_dir=tmp_path / "topo",
        provenance=sill,
        caspian_curve=curve,
    )
    with rasterio.open(out) as ds:
        owned = set(ds.tags()["mesher_tags"].split(","))
        keep = {key: v for key, v in ds.tags().items() if key not in owned}
        profile = ds.profile
        band = ds.read(1)
    stripped = tmp_path / "legacy.tif"
    with rasterio.open(stripped, "w", **profile) as dst:
        dst.write(band, 1)
        dst.update_tags(**keep)
    return stripped


@integration
def test_the_additive_pass_adds_the_tags_and_changes_no_pixel(baked, tmp_path):
    with rasterio.open(baked) as ds:
        before = hashlib.sha256(np.ascontiguousarray(ds.read(1)).tobytes()).hexdigest()
    result = k.retag_paleo_dem(baked, sidecar_dir=tmp_path / "sidecars")
    assert result.pixels_unchanged
    assert result.pixel_sha256_before == before
    assert result.tags_changed == result.tags_written > 0
    assert result.already_current is False
    with rasterio.open(baked) as ds:
        tags = ds.tags()
        after = hashlib.sha256(np.ascontiguousarray(ds.read(1)).tobytes()).hexdigest()
    assert after == before
    assert tags["pixel_sha256"] == before
    assert "water_level_pontic_json" in tags
    # everything the bake had written is still there
    assert tags["sill_verdict_defensible"] and tags["threshold_defensible"]
    assert tags["gia_unknown_ice6g_cells"] == "101"


@integration
def test_the_additive_pass_is_idempotent(baked, tmp_path):
    first = k.retag_paleo_dem(baked, sidecar_dir=tmp_path / "sidecars")
    with rasterio.open(baked) as ds:
        after_first = dict(ds.tags())
    second = k.retag_paleo_dem(baked, sidecar_dir=tmp_path / "sidecars")
    with rasterio.open(baked) as ds:
        after_second = dict(ds.tags())
    assert second.already_current is True
    assert second.tags_changed == 0
    assert after_second == after_first
    assert second.pixel_sha256_after == first.pixel_sha256_before


@integration
def test_the_pass_snapshots_the_original_tags_once_and_never_overwrites_them(
    baked, tmp_path
):
    """A re-run must not replace the genuine original with a copy of its own output."""
    sidecars = tmp_path / "sidecars"
    k.retag_paleo_dem(baked, sidecar_dir=sidecars)
    snapshot = sidecars / f"{baked.stem}.tags-before.json"
    saved = json.loads(snapshot.read_text(encoding="ascii"))
    assert "water_level_pontic_json" not in saved["tags"]
    assert saved["tags"]["sill_verdict_defensible"]
    k.retag_paleo_dem(baked, sidecar_dir=sidecars)
    assert json.loads(snapshot.read_text(encoding="ascii")) == saved


@integration
def test_the_pass_refuses_a_raster_that_does_not_say_which_epoch_it_is(
    small_dem, tmp_path
):
    """The epoch comes from the tag, never the filename: a rename must not age a file."""
    with rasterio.open(small_dem) as ds:
        profile, band = ds.profile, ds.read(1)
    profile["nodata"] = k.KHV_NODATA_M
    path = tmp_path / "khv_16.5ka.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(band.astype("float32"), 1)
    with pytest.raises(ValueError, match="no 'epoch_ka' tag"):
        k.retag_paleo_dem(path, sidecar_dir=tmp_path / "s")


@integration
def test_a_dry_run_writes_neither_tags_nor_sidecar(baked, tmp_path):
    sidecars = tmp_path / "sidecars"
    with rasterio.open(baked) as ds:
        before = dict(ds.tags())
    result = k.retag_paleo_dem(baked, dry_run=True, sidecar_dir=sidecars)
    with rasterio.open(baked) as ds:
        assert dict(ds.tags()) == before
    assert not sidecars.exists()
    assert result.tags_changed > 0 and result.pixels_unchanged


@integration
def test_the_pass_checks_the_legacy_count_instead_of_copying_it_forward(
    baked, tmp_path
):
    """``legacy_tag_agrees`` is a measurement per epoch, not an assumption."""
    with rasterio.open(baked, "r+") as ds:
        ds.update_tags(gia_unknown_ice6g_cells="7")
    result = k.retag_paleo_dem(baked, sidecar_dir=tmp_path / "s")
    assert result.legacy_tag_agrees is False
    assert result.gia_unknown_ice6g_model_cells == 101
    with rasterio.open(baked) as ds:
        # the legacy tag is left exactly as found; nothing is quietly repaired, and
        # the disagreement travels in the manifest instead of being papered over.
        assert ds.tags()["gia_unknown_ice6g_cells"] == "7"
        assert ds.tags()["gia_unknown_ice6g_model_cells"] == "101"


@integration
def test_a_pixel_change_during_the_pass_would_raise(baked, tmp_path, monkeypatch):
    """The assertion that IS the safety of the pass, exercised.

    ``PixelDataChangedError`` is unreachable in normal operation — GDAL rewrites only
    the TIFF directory — which is exactly why it is worth proving it fires.  An
    unreachable guard that does not work is indistinguishable from no guard.
    """
    real = k.read_band
    calls = {"n": 0}

    def shifting(path, **kwargs):
        band, transform, nodata, tags = real(path, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1:  # the read-BACK sees a different array
            band = band + np.float32(1.0)
        return np.ascontiguousarray(band), transform, nodata, tags

    monkeypatch.setattr(k, "read_band", shifting)
    with pytest.raises(k.PixelDataChangedError, match="must not touch a single"):
        k.retag_paleo_dem(baked, sidecar_dir=tmp_path / "s")


@integration
def test_a_transport_hiccup_is_retried_but_a_dead_raster_is_reported(
    baked, tmp_path, monkeypatch
):
    """The retry is a TRANSPORT retry and it relaxes nothing.

    The first full pass died on its first file with ``TIFFFillTile: got 0 bytes,
    expected 101230`` on the SMB-mounted product tree, and that same file read
    cleanly on its own seconds later.  A hiccup must not cost 1.8 h — but a raster
    that never reads must be named, not skipped.
    """
    real = rasterio.open
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise rasterio.errors.RasterioIOError("Read failed. See previous exception")
        return real(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", flaky)
    band, _transform, _nodata, tags = k.read_band(baked, attempts=3)
    assert band.shape and tags["epoch_ka"] == "18"
    assert calls["n"] == 2  # one failure, one success

    def always_dead(*args, **kwargs):
        raise rasterio.errors.RasterioIOError("Read failed. See previous exception")

    monkeypatch.setattr(rasterio, "open", always_dead)
    with pytest.raises(RuntimeError, match="NOT skipped silently"):
        k.read_band(baked, attempts=2)


@integration
def test_retag_all_writes_a_manifest_and_refuses_a_missing_raster(
    baked, tmp_path, curve
):
    topo = tmp_path / "grid"
    topo.mkdir()
    for epoch in (18.0, 16.75):
        k.build_paleo_dem(epoch, dem_path=baked, out_dir=topo, caspian_curve=curve)
    manifest = k.retag_all(
        topo_dir=topo,
        manifest_path=tmp_path / "manifest.csv",
        sidecar_dir=tmp_path / "s",
    )
    assert len(manifest) == 2
    assert manifest["pixels_unchanged"].all()
    assert set(manifest["epoch_ka"]) == {18.0, 16.75}
    # 16.75 ka has no Pontic row: both scenarios refused, each with its own reason
    row = manifest.loc[manifest["epoch_ka"] == 16.75].iloc[0]
    assert row["pontic_levels_refused"] == len(k.PONTIC_SCENARIOS)
    assert row["pontic_levels_resolved"] == 0
    assert (tmp_path / "manifest.csv").exists()

    with pytest.raises(FileNotFoundError, match="do not exist"):
        k.retag_all(topo_dir=topo, epochs=[5.0], manifest_path=tmp_path / "m2.csv")


# ---------------------------------------------------------------------------
# What the additive pass was not allowed to move
# ---------------------------------------------------------------------------


def test_the_sill_constants_are_untouched_by_the_tag_pass():
    """C-BAND's serialization rule, still binding: these are read-only here."""
    assert MANYCH_SILL_M == 23.0
    assert k.MANYCH_SILL_PATH_M == 25.00
    assert k.AT_SILL_TOL_M == 0.5


def test_the_tag_pass_owns_no_elevation_arithmetic():
    """No function added by this phase may compute an elevation.

    The assessment established the geometry is finished, so the pass is additive by
    construction as well as by intent: the only array operation anywhere in it is a
    comparison against the declared nodata sentinel.
    """
    source = inspect.getsource(k.retag_paleo_dem) + inspect.getsource(k.mesher_tags)
    for forbidden in ("apply_gia_delta", "dst.write", "smooth_preserving_ice_mask"):
        assert forbidden not in source, forbidden
    assert 'rasterio.open(path, "r+")' in inspect.getsource(k.retag_paleo_dem)
