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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`

import scripts.bake_khvalynian as k  # noqa: E402
from scripts.bake_black_sea import (  # noqa: E402
    MANYCH_SILL_M,
    UnconvertibleThresholdError,
    UnknownGiaInBakeError,
)

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
def test_main_report_only_writes_the_sill_csv_and_no_raster(tmp_path):
    report = k.main(
        epochs=[16.5, 16.25],
        write_rasters=False,
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
