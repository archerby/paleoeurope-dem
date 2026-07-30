"""Tests for the ICE-6G ``Topo`` → present-datum solid displacement helpers.

WHY THIS FILE WAS REWRITTEN
--------------------------
The previous version of this file *asserted the bug*.  ``implied_sea_surface_offset``
used to take its datum from the median of ``ΔTopo`` over **deep ocean**, which forces
``median(Δz_solid) = 0`` over the ocean floor *by construction*.  But the ocean floor
does move: unloaded of ~120 m of water at the LGM it stands ~+30 m higher than today
(≈ 0.24 × the sea-surface change — the hydro-isostatic ratio).  Zeroing that number
by fiat pushed it onto the continents with the sign flipped, so every paleo-DEM built
through ``solid_displacement`` carried a spurious continental *lowering* of ~−30 m at
the LGM.  The old ``test_far_field_solid_displacement_near_zero_real_data`` demanded
|Δz_solid| < 8 m at three deep-ocean points at 18 ka — i.e. it demanded the bug.
Those points are now asserted to be **+28…+32 m**, which is the physics.

The tests here are therefore written so that the two candidate formulas give
*different* answers (nothing in the old suite could tell them apart), and so that the
old formula cannot be reintroduced by retuning a bound.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from paleoeurope.gia.ice6g_topo import (
    FAR_FIELD_ICE_BUFFER_KM,
    FAR_FIELD_LAT_BAND_DEG,
    IceSurfaceUnmaskedWarning,
    distance_to_ice_km,
    ever_glaciated_mask,
    far_field_reference_mask,
    global_latitude_grid,
    ice_surface_mask,
    implied_sea_surface_offset,
    smooth_preserving_ice_mask,
    solid_displacement,
)

# ===========================================================================
# Synthetic global grid used by the unit tests
# ===========================================================================
#
# 180 × 360 cell-centred 1° grid: ``lon == 2 × lat`` in size, which is what makes
# the latitudes inferable from the array shape alone (see ``global_latitude_grid``).
# ``|lat| ≤ 25`` yields 50 × 360 = 18 000 tropical cells, above the 10 000-cell
# floor the estimator insists on.

M, N = 180, 360
SEA_SURFACE_RISE = 100.0    # ΔW: the sea surface stood 100 m lower at epoch t
OCEAN_FLOOR_UPLIFT = 30.0   # the ocean floor stood 30 m higher (water load absent)


def _synthetic_pair(land_uplift: float = 0.0, land_rows: slice | None = None):
    """Return ``(topo_t, topo_0)`` for a grid with tropical land and deep ocean.

    Tropical land (|lat| ≤ 25) has Δz_solid = *land_uplift* (0 by default: a stable
    craton).  Everything else is deep ocean at −4000 m with Δz_solid = +30 m.  The
    contemporaneous sea surface is 100 m lower at epoch *t*, so ``ΔTopo`` = +100 over
    the land and +130 over the ocean.

    A deep-ocean estimator returns 130.0 here; the correct far-field continental
    estimator returns 100.0.  Every downstream number differs by that 30 m.
    """
    lat2d = global_latitude_grid((M, N))
    tropics = np.abs(lat2d) <= 25.0
    if land_rows is not None:
        keep = np.zeros_like(tropics)
        keep[land_rows, :] = True
        tropics = tropics & keep

    topo_0 = np.full((M, N), -4000.0)
    topo_0[tropics] = 300.0                      # dry tropical continent today

    dz_solid = np.full((M, N), OCEAN_FLOOR_UPLIFT)
    dz_solid[tropics] = land_uplift

    topo_t = topo_0 + SEA_SURFACE_RISE + dz_solid
    return topo_t, topo_0


def _synthetic_masks(topo_0):
    """``sftlf``/``sftgif`` percent fields consistent with :func:`_synthetic_pair`."""
    land = (topo_0 > 0.0).astype(np.float32) * 100.0
    ice = np.zeros_like(land)
    lat2d = global_latitude_grid(topo_0.shape)
    ice[lat2d > 60.0] = 100.0                     # a northern ice sheet
    return land, ice


# ===========================================================================
# The discriminating test: which reference set sets the datum?
# ===========================================================================


def test_offset_is_far_field_tropical_land_median_not_deep_ocean():
    """THE test that tells the two formulas apart.

    Tropical land Δz_solid = 0, ocean floor Δz_solid = +30, sea surface 100 m lower.
    The offset must be 100.0 (the sea-surface change), **not** 130.0 (which is the
    sea-surface change plus the ocean-floor rebound the old estimator absorbed).
    """
    topo_t, topo_0 = _synthetic_pair()
    offset = implied_sea_surface_offset(topo_t, topo_0)
    assert offset == pytest.approx(SEA_SURFACE_RISE, abs=1e-9)
    assert offset != pytest.approx(SEA_SURFACE_RISE + OCEAN_FLOOR_UPLIFT, abs=1.0)


def test_solid_displacement_is_zero_over_land_and_positive_over_ocean():
    """Δz_solid = 0 over the stable tropics and +30 over the ocean floor.

    The ocean-floor rebound must be *present in the output*, not zeroed.  This is
    the assertion that makes the old formula unreintroducible: under it the ocean
    cells read exactly 0.0 and the land cells read −30.
    """
    topo_t, topo_0 = _synthetic_pair()
    dz = solid_displacement(
        topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0), sftgif_t=np.zeros((M, N))
    )
    lat2d = global_latitude_grid((M, N))
    tropics = np.abs(lat2d) <= 25.0
    assert np.allclose(dz[tropics], 0.0, atol=1e-9)
    assert np.allclose(dz[~tropics], OCEAN_FLOOR_UPLIFT, atol=1e-9)


def test_solid_displacement_recovers_uplift_over_land():
    """A tropical craton that additionally rose 50 m must report +50, not +150."""
    topo_t, topo_0 = _synthetic_pair()
    # A non-reference mid-latitude land patch with 50 m of genuine solid uplift.
    topo_t, topo_0 = topo_t.copy(), topo_0.copy()
    topo_0[140:150, 100:110] = 200.0
    topo_t[140:150, 100:110] = 200.0 + SEA_SURFACE_RISE + 50.0
    offset = implied_sea_surface_offset(topo_t, topo_0)
    dz = solid_displacement(topo_t, topo_0, offset, sftgif_t=np.zeros((M, N)))
    assert dz[145, 105] == pytest.approx(50.0, abs=1e-9)


# ===========================================================================
# The reference set itself
# ===========================================================================


def test_reference_mask_applies_all_five_conditions():
    topo_t, topo_0 = _synthetic_pair()
    sftlf, sftgif = _synthetic_masks(topo_0)
    lat2d = global_latitude_grid((M, N))
    ever_ice = sftgif > 50
    dist = distance_to_ice_km(ever_ice)

    ref = far_field_reference_mask(
        topo_0, sftlf_t=sftlf, sftlf_0=sftlf, sftgif_t=sftgif,
        ever_ice=ever_ice, dist_to_ice_km=dist,
    )
    assert ref.any()
    assert np.all(np.abs(lat2d[ref]) <= FAR_FIELD_LAT_BAND_DEG)
    assert np.all(topo_0[ref] > 0.0)              # genuinely continental today
    assert not np.any(ever_ice[ref])              # never glaciated in the suite
    assert np.all(dist[ref] >= FAR_FIELD_ICE_BUFFER_KM)
    assert np.all(sftlf[ref] > 50.0)              # land at both endpoints


def test_reference_mask_excludes_present_day_shelf():
    """``topo_0 > 0``: present-day shelf sits in the hydro-isostatic transition zone."""
    topo_t, topo_0 = _synthetic_pair()
    topo_0 = topo_0.copy()
    topo_0[90:95, :20] = -60.0                    # a tropical shelf strip
    ref = far_field_reference_mask(topo_0)
    assert not ref[90:95, :20].any()


def test_offset_refuses_a_regional_subset():
    """A regional window has no tropical far field — refuse, do not improvise.

    This is what stops the North Sea test fixture (66×67 cells, deep cells only in
    the Norwegian Trench) from silently producing a datum.
    """
    topo_t, topo_0 = _synthetic_pair()
    with pytest.raises(ValueError, match="global"):
        implied_sea_surface_offset(topo_t[40:60, 100:140], topo_0[40:60, 100:140])


def test_minimum_reference_cells_scales_with_the_grid():
    """The requirement is coverage, not a raw count.

    Rewritten 2026-07-30.  The threshold used to be an absolute
    ``MIN_REFERENCE_CELLS = 10_000``, which was a hidden dependency on the
    10-arcmin shipped resolution: a legitimately global 2° grid holds only
    ~4 500 cells in the entire ``|lat| <= 25`` band, so an absolute floor at
    10 000 refused every coarse global field no matter how much tropical land it
    carried — while a 0.25° *regional* cut with 10 000 ocean cells would have
    sailed straight through.  Cell count is the wrong axis.
    """
    from paleoeurope.gia.ice6g_topo import (
        MIN_REFERENCE_CELLS_FLOOR,
        minimum_reference_cells,
    )

    # Real ICE-6G 10 arcmin grid: at least as strict as the 10 000 it replaced.
    assert minimum_reference_cells(1080 * 2160) >= 10_000
    # A coarse but genuinely global 2° grid is now admissible…
    assert minimum_reference_cells(91 * 180) == MIN_REFERENCE_CELLS_FLOOR
    # …and the floor still stops a degenerate handful of cells defining a datum.
    assert minimum_reference_cells(10) == MIN_REFERENCE_CELLS_FLOOR
    # Monotonic in grid size.
    assert minimum_reference_cells(64_800) <= minimum_reference_cells(2_332_800)


def test_offset_refuses_when_reference_set_too_small():
    """Fewer far-field cells than required → refuse, whoever sets the bar.

    720 land cells on a 64 800-cell grid clears the scaled requirement (324), so
    this exercises the explicit-override path: a caller that pins a stricter bar
    must still be honoured.
    """
    topo_t, topo_0 = _synthetic_pair(land_rows=slice(90, 92))   # 2 rows = 720 cells
    with pytest.raises(ValueError, match="reference"):
        implied_sea_surface_offset(topo_t, topo_0, min_reference_cells=10_000)


def test_offset_accepts_a_reference_set_that_clears_the_scaled_bar():
    """The same 720 cells suffice under the default, grid-scaled requirement."""
    topo_t, topo_0 = _synthetic_pair(land_rows=slice(90, 92))
    assert np.isfinite(implied_sea_surface_offset(topo_t, topo_0))


def test_offset_refuses_non_global_grid_without_explicit_lat():
    topo_t, topo_0 = _synthetic_pair()
    trimmed_t, trimmed_0 = topo_t[:, :300], topo_0[:, :300]
    with pytest.raises(ValueError, match="global"):
        implied_sea_surface_offset(trimmed_t, trimmed_0)
    # …but an explicit latitude grid is accepted.
    lat2d = global_latitude_grid((M, N))[:, :300]
    assert implied_sea_surface_offset(trimmed_t, trimmed_0, lat=lat2d) == pytest.approx(
        SEA_SURFACE_RISE, abs=1e-9
    )


def test_offset_accepts_an_explicit_reference_mask():
    """Escape hatch for synthetic fixtures: the caller declares its own far field."""
    topo_t, topo_0 = _synthetic_pair()
    mask = np.zeros((M, N), bool)
    mask[80:100, :] = True                        # the tropical band, declared by hand
    assert implied_sea_surface_offset(
        topo_t, topo_0, reference_mask=mask, min_reference_cells=100
    ) == pytest.approx(SEA_SURFACE_RISE, abs=1e-9)


def test_offset_ignores_nan_reference_cells():
    """ICE-6G_C ``Topo`` declares ``_FillValue = 1e+20``; a stray NaN must not poison."""
    topo_t, topo_0 = _synthetic_pair()
    topo_t = topo_t.copy()
    topo_t[85:95, :50] = np.nan
    assert implied_sea_surface_offset(topo_t, topo_0) == pytest.approx(
        SEA_SURFACE_RISE, abs=1e-9
    )


def test_retired_deep_threshold_argument_is_refused_loudly():
    """The old third positional argument must not be silently accepted.

    ``implied_sea_surface_offset(topo_t, topo_0, -1000.0)`` used to select the deep
    ocean, i.e. the bug.  Accepting and ignoring it would let old call sites keep
    reading as if the datum were still deep-ocean-referenced.
    """
    topo_t, topo_0 = _synthetic_pair()
    with pytest.raises(TypeError, match="deep_threshold"):
        implied_sea_surface_offset(topo_t, topo_0, -1000.0)


def test_global_latitude_grid_is_symmetric_and_cell_centred():
    lat2d = global_latitude_grid((M, N))
    assert lat2d.shape == (M, N)
    assert lat2d[0, 0] == pytest.approx(-90.0 + 0.5 * 180.0 / M)
    assert np.allclose(lat2d[0, :], lat2d[0, 0])
    assert np.allclose(np.abs(lat2d[0, 0]), np.abs(lat2d[-1, 0]))


def test_distance_to_ice_km_wraps_in_longitude():
    """The dateline must not act as a false barrier."""
    ice = np.zeros((M, N), bool)
    ice[90, 0] = True
    dist = distance_to_ice_km(ice)
    # Cell (90, N-1) is one cell west of (90, 0) across the seam.
    assert dist[90, N - 1] == pytest.approx(dist[90, 1], rel=1e-9)


def test_ever_glaciated_mask_is_the_union_over_epochs():
    a = np.zeros((4, 4), np.float32); a[0, 0] = 100.0
    b = np.zeros((4, 4), np.float32); b[1, 1] = 100.0
    c = np.zeros((4, 4), np.float32); c[1, 1] = 40.0     # below the 50 % threshold
    ever = ever_glaciated_mask([a, b, c])
    assert ever[0, 0] and ever[1, 1]
    assert ever.sum() == 2


# ===========================================================================
# The ice-surface trap
# ===========================================================================


def test_solid_displacement_is_nan_over_ice_covered_cells():
    """Over ``sftgif > 0`` cells ``Topo`` is the ICE SURFACE, not bedrock → NaN.

    Bedrock under ice is not recoverable from the shipped 10 arcmin distribution
    (its only data variables are sftlf/sftgif/Topo/Orog, and Topo == Orog == the ice
    surface there), so the only honest answer is "no data".
    """
    topo_t, topo_0 = _synthetic_pair()
    _sftlf, sftgif = _synthetic_masks(topo_0)
    dz = solid_displacement(
        topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0),
        sftgif_t=sftgif, sftgif_0=np.zeros_like(sftgif),
    )
    ice = sftgif > 50
    assert np.all(np.isnan(dz[ice]))
    assert np.all(np.isfinite(dz[~ice]))


def test_solid_displacement_masks_present_day_ice_too():
    """A cell iced *today* also has ``Topo(0)`` = ice surface → the delta is junk."""
    topo_t, topo_0 = _synthetic_pair()
    sftgif_0 = np.zeros((M, N), np.float32)
    sftgif_0[170:, :] = 100.0
    dz = solid_displacement(
        topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0),
        sftgif_t=np.zeros((M, N)), sftgif_0=sftgif_0,
    )
    assert np.all(np.isnan(dz[170:, :]))


def test_solid_displacement_warns_when_no_ice_information_is_supplied():
    """No masks → the ice-surface trap is live, and the caller must be told."""
    topo_t, topo_0 = _synthetic_pair()
    offset = implied_sea_surface_offset(topo_t, topo_0)
    with pytest.warns(IceSurfaceUnmaskedWarning, match="(?i)ice surface"):
        solid_displacement(topo_t, topo_0, offset)


def test_ice_mask_dilation_covers_the_smoothing_rim():
    ice = np.zeros((40, 40), bool)
    ice[20, 20] = True
    mask = ice_surface_mask(ice_mask_t=ice, dilate_cells=3)
    assert mask[20, 23] and mask[17, 20]
    assert not mask[20, 25]


def test_smoothing_does_not_leak_ice_surface_values_across_the_margin():
    """σ=2 on a raw ΔTopo field smears the ~+1800 m ice-margin step ~130 km.

    Measured on the real files at 21 ka, a plain ``gaussian_filter`` over an
    unmasked field differs from the NaN-aware answer by >100 m three cells past the
    margin and by >1 m out to six or seven cells.  ``smooth_preserving_ice_mask``
    must renormalise over the ice-free cells only, so an ice-free cell next to a
    1000 m ice step stays small.
    """
    from scipy.ndimage import gaussian_filter

    dz = np.zeros((60, 60))
    ice = np.zeros((60, 60), bool)
    ice[:30, :] = True
    dz[:30, :] = 1000.0                            # the ice-surface plateau
    dz_masked = np.where(ice, np.nan, dz)

    naive = gaussian_filter(dz, sigma=2.0, mode="nearest")
    clean = smooth_preserving_ice_mask(dz_masked, sigma=2.0)

    assert naive[31, 30] > 100.0                   # the leak, unmasked
    assert abs(clean[31, 30]) < 1e-9               # …gone
    assert np.all(np.isnan(clean[ice]))            # ice stays no-data


# ===========================================================================
# Real ICE-6G_C data
# ===========================================================================

ICE6G_DIR = Path(r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/ICE6G")

#: All 48 shipped epochs: 0–20.5 ka at 0.5 ka, then 21–26 ka at 1 ka.
ALL_EPOCHS: tuple[float, ...] = tuple(
    [round(0.5 * i, 1) for i in range(0, 42)] + [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
)

requires_ice6g = pytest.mark.skipif(
    not (ICE6G_DIR / "I6_C.VM5a_10min.18.nc").exists(),
    reason="ICE-6G data not present",
)


def _epoch_file(epoch: float) -> Path:
    return ICE6G_DIR / f"I6_C.VM5a_10min.{epoch:g}.nc"


@pytest.fixture(scope="module")
def ice6g_static():
    """Epoch-independent fields: ``lat``, ``lon``, ``Topo(0)``, ``sftlf(0)``,
    ``sftgif(0)``, ``ever_ice`` (union over all 48 epochs) and ``dist_to_ice_km``.

    The 48-file ``sftgif`` scan takes ~2.5 s and the wrapped EDT ~1 s, so this is
    computed once per module rather than cached on disk.
    """
    if not _epoch_file(18).exists():
        pytest.skip("ICE-6G data not present")
    ice_fields = []
    for epoch in ALL_EPOCHS:
        with xr.open_dataset(_epoch_file(epoch)) as ds:
            ice_fields.append(ds["sftgif"].values)
    ever_ice = ever_glaciated_mask(ice_fields)
    with xr.open_dataset(_epoch_file(0)) as ds0:
        out = {
            "lat": ds0["lat"].values.astype(np.float64),
            "lon": ds0["lon"].values.astype(np.float64),
            "topo_0": ds0["Topo"].values.astype(np.float64),
            "sftlf_0": ds0["sftlf"].values,
            "sftgif_0": ds0["sftgif"].values,
        }
    out["ever_ice"] = ever_ice
    out["dist_to_ice_km"] = distance_to_ice_km(ever_ice)
    return out


def _epoch_fields(epoch: float):
    with xr.open_dataset(_epoch_file(epoch)) as ds:
        return (
            ds["Topo"].values.astype(np.float64),
            ds["sftlf"].values,
            ds["sftgif"].values,
        )


@requires_ice6g
@pytest.mark.integration
@pytest.mark.slow
def test_acceptance_A_reference_set_median_is_zero_at_every_epoch(ice6g_static):
    """CRITERION A — the acceptance criterion, at all 48 epochs 0 → 26 ka.

    |median Δz_solid over the far-field ice-free tropical reference set| ≤ 1.0 m.

    Tolerance: the median is 0 *by construction* when the estimator uses the full
    five-condition mask, so 1.0 m is not a measurement tolerance but a regression
    tripwire.  It is sized to the 2.06 m spread between independent continental
    blocks (Africa 77.79, Australia 77.24, S America 77.58, India/SE Asia 79.30 m
    at 16.5 ka), i.e. ±1 m about the median — the honest systematic uncertainty on
    the offset.  The old estimator fails this at 45 of 48 epochs, reaching
    −31.80 m at 23 ka.
    """
    worst = (None, 0.0)
    for epoch in ALL_EPOCHS:
        topo_t, sftlf_t, sftgif_t = _epoch_fields(epoch)
        offset = implied_sea_surface_offset(topo_t, ice6g_static["topo_0"])
        dz = solid_displacement(
            topo_t, ice6g_static["topo_0"], offset,
            sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"],
        )
        ref = far_field_reference_mask(
            ice6g_static["topo_0"],
            sftlf_t=sftlf_t,
            sftlf_0=ice6g_static["sftlf_0"],
            sftgif_t=sftgif_t,
            ever_ice=ice6g_static["ever_ice"],
            dist_to_ice_km=ice6g_static["dist_to_ice_km"],
        )
        assert ref.sum() > 100_000, f"{epoch} ka: reference set collapsed to {ref.sum()}"
        med = float(np.nanmedian(dz[ref]))
        if abs(med) > abs(worst[1]):
            worst = (epoch, med)
        assert abs(med) <= 1.0, (
            f"{epoch:g} ka: median Δz_solid over the far-field reference set is "
            f"{med:+.3f} m, must be ~0 — the datum is not land-referenced"
        )
    # Measured 2026-07-30: the worst residual over all 48 epochs is +0.005 m.
    assert abs(worst[1]) < 0.05, f"worst residual {worst[1]:+.4f} m at {worst[0]} ka"


@requires_ice6g
@pytest.mark.integration
@pytest.mark.slow
def test_masked_and_unmasked_reference_set_offsets_agree(ice6g_static):
    """Justifies the mask-free compatibility path used by the production callers.

    ``bake_black_sea.gia_delta_subset`` and ``Ice6gLoader.get_fields`` call
    ``implied_sea_surface_offset(topo_t, topo_0)`` with no mask fields, so the
    estimator falls back to ``|lat| ≤ 25 & topo_0 > 0``, dropping three of G1's
    five conditions.  That fallback is only defensible if it agrees with the full
    five-condition reference set.  Measured over all 48 epochs the largest
    disagreement is 0.005 m — five orders of magnitude below the ~30 m bias being
    fixed, and three below the ±1 m inter-continental systematic.
    """
    worst = 0.0
    for epoch in ALL_EPOCHS:
        topo_t, sftlf_t, sftgif_t = _epoch_fields(epoch)
        degraded = implied_sea_surface_offset(topo_t, ice6g_static["topo_0"])
        full = implied_sea_surface_offset(
            topo_t,
            ice6g_static["topo_0"],
            sftlf_t=sftlf_t,
            sftlf_0=ice6g_static["sftlf_0"],
            sftgif_t=sftgif_t,
            ever_ice=ice6g_static["ever_ice"],
            dist_to_ice_km=ice6g_static["dist_to_ice_km"],
        )
        worst = max(worst, abs(degraded - full))
    assert worst < 0.05, f"fallback disagrees with the full reference set by {worst:.4f} m"


@requires_ice6g
@pytest.mark.integration
@pytest.mark.slow
def test_acceptance_B_ocean_floor_rebound_is_present_not_zeroed(ice6g_static):
    """CRITERION B — the criterion that makes the bug unreintroducible.

    Criterion A alone is satisfiable by the *old* code with a relabelled reference
    set, because "deep ocean" was the old estimator's reference set and its median
    Δz_solid is 0.000 there at all 48 epochs by construction.  So we also demand
    that the ocean-floor rebound is PRESENT: median Δz_solid over deep ocean must
    be positive and in [+20, +35] m for 16.5 ≤ t ≤ 23 ka.  Measured: +29.23 m at
    16.5 ka rising to +31.80 m at 23 ka.  ≈ 0.24 × the sea-surface change, the
    expected hydro-isostatic ratio.
    """
    deep = ice6g_static["topo_0"] < -1000.0
    for epoch in [e for e in ALL_EPOCHS if 16.5 <= e <= 23.0]:
        topo_t, _sftlf_t, sftgif_t = _epoch_fields(epoch)
        offset = implied_sea_surface_offset(topo_t, ice6g_static["topo_0"])
        dz = solid_displacement(
            topo_t, ice6g_static["topo_0"], offset,
            sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"],
        )
        med = float(np.nanmedian(dz[deep]))
        assert 20.0 <= med <= 35.0, (
            f"{epoch:g} ka: median Δz_solid over deep ocean is {med:+.3f} m; the "
            "ocean floor must stand HIGHER than today at a lowstand (water load "
            "absent).  0.0 means the deep-ocean datum bug is back."
        )


@requires_ice6g
@pytest.mark.integration
def test_acceptance_C_offset_is_zero_at_0ka_and_falls_monotonically(ice6g_static):
    """CRITERION C, corrected.

    ``offset(0 ka)`` must be exactly 0.0, and the offset must be non-increasing
    from 26 ka to 6.5 ka.

    G1 specified "monotonically non-increasing from 26 → 0 ka apart from the
    14.5 → 14.0 MWP-1A step".  That is FALSE for the land-referenced offset and
    the failure is physical, not a bug: measured, the offset reaches a *minimum* of
    −3.54 m at 6.5 ka and then rises back to 0.0 at 0 ka.  That is the far-field
    mid-Holocene highstand (ocean siphoning / continental levering) — the tropical
    land surface stood a few metres above present sea level in the mid-Holocene.
    Asserting monotonicity through the Holocene would encode a wrong expectation,
    so the monotone claim is asserted only over 26 → 6.5 ka.
    """
    offsets = {}
    for epoch in ALL_EPOCHS:
        topo_t, _sftlf_t, _sftgif_t = _epoch_fields(epoch)
        offsets[epoch] = implied_sea_surface_offset(topo_t, ice6g_static["topo_0"])

    assert offsets[0.0] == 0.0, f"offset(0 ka) = {offsets[0.0]!r}, must be exactly 0.0"

    glacial = sorted((e for e in ALL_EPOCHS if e >= 6.5), reverse=True)
    for older, younger in zip(glacial, glacial[1:]):
        assert offsets[younger] <= offsets[older] + 1e-6, (
            f"offset rose from {offsets[older]:.3f} m at {older:g} ka to "
            f"{offsets[younger]:.3f} m at {younger:g} ka"
        )
    # The Holocene reversal is real and is asserted, so it cannot be "fixed" away.
    assert min(offsets.values()) == pytest.approx(-3.543, abs=0.05)
    assert min(offsets, key=offsets.get) == pytest.approx(6.5)


@requires_ice6g
@pytest.mark.integration
def test_far_field_deep_ocean_points_at_18ka_show_the_rebound(ice6g_static):
    """REWRITTEN WITH INVERTED INTENT — was ``test_far_field_solid_displacement_
    near_zero_real_data``, which demanded |Δz_solid| < 8 m at these three points.

    That assertion *was* the bug: it demanded the ocean floor not move.  Measured
    after the fix these three far-field abyssal points read +30.76, +28.41 and
    +31.46 m at 18 ka, and the offset is 81.96 m (not the 90–130 m the old test
    demanded, which was the deep-ocean median).  The bounds below are NOT retuned
    versions of the old ones — the intent is reversed.
    """
    topo_t, _sftlf_t, sftgif_t = _epoch_fields(18.0)
    topo_0 = ice6g_static["topo_0"]
    offset = implied_sea_surface_offset(topo_t, topo_0)
    assert 78.0 < offset < 86.0, f"18 ka land-referenced offset {offset:.2f} m"
    dz = solid_displacement(
        topo_t, topo_0, offset, sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"]
    )
    da = xr.DataArray(
        dz, coords={"lat": ice6g_static["lat"], "lon": ice6g_static["lon"]},
        dims=("lat", "lon"),
    )
    for lat, lon in [(-20.0, 200.0), (-30.0, 80.0), (-30.0, 340.0)]:
        v = float(da.sel(lat=lat, lon=lon, method="nearest"))
        assert 28.0 <= v <= 32.0, (
            f"far-field abyssal Δz_solid = {v:+.2f} m at ({lat},{lon}); the unloaded "
            "ocean floor must stand ~+30 m higher than today at 18 ka, and reading "
            "~0 there means the datum is deep-ocean-referenced again"
        )


# --- the Manych regression: the test that protects the science ----------------

MANYCH_LAT, MANYCH_LON = 45.773, 44.119    # GEBCO 2024 sill point, 15 arcsec
MANYCH_SILL_MODERN_M = 23.00               # per-column-minima maximum (lower bound)


def _sill_asl(dz, ice6g_static) -> float:
    """``23.00 m + Δz_solid`` bilinearly sampled at the Manych sill point."""
    da = xr.DataArray(
        dz, coords={"lat": ice6g_static["lat"], "lon": ice6g_static["lon"]},
        dims=("lat", "lon"),
    )
    return MANYCH_SILL_MODERN_M + float(
        da.interp(lat=MANYCH_LAT, lon=MANYCH_LON, method="linear")
    )


@requires_ice6g
@pytest.mark.integration
def test_manych_palaeo_sill_at_16_5ka(ice6g_static):
    """The palaeo-sill stood ABOVE its modern +23.00 m, not 26 m below it.

    The buggy engine put it at −3.52 m a.s.l. at 16.5 ka, which would let the
    Caspian spill west at essentially any level above −3.5 m.  Corrected it is
    +25.72 m — the fix makes the Khvalynian spill HARDER, and it agrees with the
    independently-derived science file (``2026-07-28-caspian-khvalynian-level-
    curve.csv``: ``paleo_sill_m_asl`` = 25.79 at 16.14 ka) and with the project's
    stated palaeo-sill range +25.7 … +26.9 m.

    ±0.5 m here is an interpolation/regression tolerance only.  The physical
    uncertainty is larger: ±1–2 m, because a single scalar offset cannot resolve
    the spatially varying geoid change 1560 km from the ice margin.
    """
    topo_t, _sftlf_t, sftgif_t = _epoch_fields(16.5)
    topo_0 = ice6g_static["topo_0"]
    dz = solid_displacement(
        topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0),
        sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"],
    )
    assert _sill_asl(dz, ice6g_static) == pytest.approx(25.72, abs=0.5)


@requires_ice6g
@pytest.mark.integration
@pytest.mark.slow
def test_manych_palaeo_sill_never_drops_below_modern_through_the_spill_window(
    ice6g_static,
):
    """Through 24 → 13 ka the sill stood above +23.00 m at every epoch.

    Local GIA at Manych is small and positive (+1.0 to +3.1 m over 24–15.5 ka).
    The buggy engine returned a *negative* sill elevation at all of these epochs.
    The sill cell is ice-free at all 48 epochs (``sftgif`` = 0, ``ever_ice``
    False), so this number is not touched by the ice-surface mask.
    """
    assert not ice6g_static["ever_ice"][
        int(np.argmin(np.abs(ice6g_static["lat"] - MANYCH_LAT))),
        int(np.argmin(np.abs(ice6g_static["lon"] - MANYCH_LON))),
    ], "the Manych sill cell is glaciated somewhere in the suite — mask applies"

    topo_0 = ice6g_static["topo_0"]
    for epoch in [e for e in ALL_EPOCHS if 13.0 <= e <= 24.0]:
        topo_t, _sftlf_t, sftgif_t = _epoch_fields(epoch)
        dz = solid_displacement(
            topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0),
            sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"],
        )
        sill = _sill_asl(dz, ice6g_static)
        assert sill > MANYCH_SILL_MODERN_M, (
            f"{epoch:g} ka: palaeo-sill {sill:+.2f} m a.s.l. is below the modern "
            f"{MANYCH_SILL_MODERN_M:.2f} m — the datum bias is back"
        )


# --- the ice-surface trap, on real data --------------------------------------

FENNOSCANDIA_LAT, FENNOSCANDIA_LON = 62.0, 15.0
EUROPE_WINDOW = (45.0, 72.0, 350.0, 40.0)   # lat_min, lat_max, lon_min, lon_max (0..360)


@requires_ice6g
@pytest.mark.integration
def test_fennoscandia_under_ice_is_nan_not_an_ice_surface_elevation(ice6g_static):
    """At 62 N / 15 E, 21 ka, ``sftgif`` = 100 and the unmasked Δz_solid is
    +1711 m — that is the ICE SURFACE (``Topo(21)`` = +2101.6 over ``Topo(0)`` =
    +303.0), and ``correction_pipeline`` would add ICE-7G thickness on top of it,
    counting the ice column twice.  It must be NaN.
    """
    topo_t, _sftlf_t, sftgif_t = _epoch_fields(21.0)
    topo_0 = ice6g_static["topo_0"]
    offset = implied_sea_surface_offset(topo_t, topo_0)

    i = int(np.argmin(np.abs(ice6g_static["lat"] - FENNOSCANDIA_LAT)))
    j = int(np.argmin(np.abs(ice6g_static["lon"] - FENNOSCANDIA_LON)))
    assert sftgif_t[i, j] > 50.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IceSurfaceUnmaskedWarning)
        unmasked = solid_displacement(topo_t, topo_0, offset)
    assert unmasked[i, j] > 1500.0, "sanity: the trap is real in the raw field"

    dz = solid_displacement(
        topo_t, topo_0, offset, sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"]
    )
    assert np.isnan(dz[i, j])


@requires_ice6g
@pytest.mark.integration
@pytest.mark.slow
def test_no_finite_europe_displacement_exceeds_250m_at_any_epoch(ice6g_static):
    """After masking, nothing in the Europe window looks like an ice sheet.

    Genuine ice-free Δz_solid there peaks at +94 m over all 48 epochs (+71 m at
    21 ka).  Unmasked the same window reaches +2537 m.  250 m is a wide margin
    around the physics that still catches any ice-surface leak by an order of
    magnitude.
    """
    lat, lon = ice6g_static["lat"], ice6g_static["lon"]
    lat_sel = (lat >= EUROPE_WINDOW[0]) & (lat <= EUROPE_WINDOW[1])
    lon_sel = (lon >= EUROPE_WINDOW[2]) | (lon <= EUROPE_WINDOW[3])
    window = np.ix_(lat_sel, lon_sel)
    topo_0 = ice6g_static["topo_0"]

    for epoch in ALL_EPOCHS:
        topo_t, _sftlf_t, sftgif_t = _epoch_fields(epoch)
        dz = solid_displacement(
            topo_t, topo_0, implied_sea_surface_offset(topo_t, topo_0),
            sftgif_t=sftgif_t, sftgif_0=ice6g_static["sftgif_0"],
        )
        finite = dz[window]
        finite = finite[np.isfinite(finite)]
        assert finite.max() <= 250.0, (
            f"{epoch:g} ka: finite Δz_solid reaches {finite.max():.1f} m in the "
            "Europe window — an ice-surface elevation leaked through as bedrock"
        )


@requires_ice6g
@pytest.mark.integration
def test_north_sea_fixture_is_not_accepted_as_a_datum_source():
    """The synthetic North Sea fixture must not be able to produce an offset.

    ``tests/fixtures/ice6g/ICE6G_fixture.nc`` is a 66×67 subset (lat 57.08–67.92,
    lon 1–12) whose only deep cells are the Norwegian Trench — maximally
    near-field.  Under the old estimator it happily produced a number.
    """
    fixture = Path(__file__).parent / "fixtures" / "ice6g" / "ICE6G_fixture.nc"
    if not fixture.exists():
        pytest.skip("ICE6G fixture not present")
    with xr.open_dataset(fixture) as ds:
        topo = ds["Topo"]
        topo_0 = topo.isel(time=0).values.astype(np.float64)
        topo_t = topo.isel(time=-1).values.astype(np.float64)
    with pytest.raises(ValueError):
        implied_sea_surface_offset(topo_t, topo_0)
