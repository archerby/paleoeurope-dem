"""
tests.test_ice6g_loader_ice_guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Does the *ice half* of the 2026-07-30 GIA datum fix actually reach production?

:mod:`paleoeurope.gia.ice6g_topo` grew an ice-surface guard (``solid_displacement``
returns NaN where ICE-6G ``Topo`` is the ice surface) and a mask-aware smoother
(``smooth_preserving_ice_mask``).  Neither is worth anything unless
:meth:`paleoeurope.gia.ice6g_loader.Ice6gLoader.get_fields` — the only production
entry point into that module, used by ``correction_pipeline``, ``distributed.tasks``
and ``scripts/run_paleo.py`` — passes the ice fields through and smooths *after*
masking.  These tests pin exactly that, plus the two things that must not regress:

* the sea-surface datum is derived from the file's **real latitudes**, not guessed
  from the array shape (a global-extent grid that is not exactly ``n_lon == 2·n_lat``
  must still work);
* on an **ice-free** window the answer is bit-for-bit the plain
  ``gaussian_filter(dz, sigma=2.0)`` that this loader has always returned.

Since 2026-07-30 it also covers the two defects that kept ``get_fields`` from ever
running on the **real** distribution:

* the shipped ``I6_C.VM5a_10min.<epoch>.nc`` files hold **one epoch each and no time
  dimension**, so the old ``_detect_dim(ds, ("time","Time","t"))`` raised ``KeyError``
  on every one of them.  The loader now has an explicit epoch-pair API
  (:meth:`Ice6gLoader.from_epoch_files`, :meth:`Ice6gLoader.from_epoch_directory`);
* those files run ``lon 0..360`` while every raster in this project runs −180..180, so
  a negative-longitude bbox used to select the wrong columns — a *truncated* window
  for Doggerland, and a silently **empty** one for anything wholly west of Greenwich.

Deliberately in its own file: ``tests/test_gia.py`` and ``tests/test_gia_topo.py`` are
being worked on concurrently.

The fixture is synthetic but reproduces the three traps of the real
``I6_C.VM5a_10min.*.nc`` files: ``Topo`` referenced to the *contemporaneous* sea
surface, ``Orog`` clamped to 0 over water, and ``Topo`` = the **ice surface** wherever
``sftgif > 0``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from paleoeurope.gia.ice6g_loader import (
    GiaDeltaHasIceSurfaceNaNWarning,
    Ice6gLoader,
    ice6g_epoch_filename,
)
from paleoeurope.gia.ice6g_topo import (
    global_latitude_grid,
    implied_sea_surface_offset,
    solid_displacement,
)

# --- fixture geometry -------------------------------------------------------
#
# 0.5° in latitude by 1.0° in longitude: global in *extent* but 360×360, so
# ``n_lon == 2·n_lat`` is false and the shape-inferred latitude fallback refuses it.
# The loader must therefore be passing ``lat=`` explicitly.  The tropical band holds
# 100 rows × 360 cols = 36 000 land cells, comfortably over MIN_REFERENCE_CELLS.
LAT_STEP, LON_STEP = 0.5, 1.0
SEA_SURFACE_OFFSET = 112.0     # eustatic lowstand baked into ``Topo`` at −18 ka
DEEP_TOPO_0 = -3000.0          # far-field deep ocean
TROPIC_TOPO_0 = 200.0          # far-field ice-free tropical continental land
REGION_TOPO_0 = 100.0          # the test region: land today
REGION_DZ_SOLID = 40.0         # true solid uplift over the test region at −18 ka
ICE_THICKNESS = 1800.0         # ice column at −18 ka poleward of ICE_EDGE_LAT
ICE_EDGE_LAT = 60.0

# Region straddling the ice margin, and an ice-free tropical control window.
REGION_BOUNDS = (0.0, 50.0, 30.0, 70.0)    # lon_min, lat_min, lon_max, lat_max
TROPICAL_BOUNDS = (5.0, 2.0, 25.0, 20.0)


def _make_global_ice6g_nc(path: Path) -> Path:
    """Write a global-extent, two-epoch, ICE-6G-shaped NetCDF with an ice surface."""
    lats = np.arange(-90.0 + LAT_STEP / 2, 90.0, LAT_STEP, dtype=np.float64)
    lons = np.arange(LON_STEP / 2, 360.0, LON_STEP, dtype=np.float64)
    times = np.array([0.0, -18.0], dtype=np.float64)

    lon2d, lat2d = np.meshgrid(lons, lats)
    tropics = np.abs(lat2d) <= 25.0
    region = (
        (lon2d >= REGION_BOUNDS[0])
        & (lon2d <= REGION_BOUNDS[2])
        & (lat2d >= REGION_BOUNDS[1])
        & (lat2d <= REGION_BOUNDS[3])
    )
    ice_t = lat2d > ICE_EDGE_LAT

    topo_0 = np.full(lat2d.shape, DEEP_TOPO_0)
    topo_0[tropics] = TROPIC_TOPO_0
    topo_0[region] = REGION_TOPO_0

    dz_solid = np.zeros_like(topo_0)
    dz_solid[region] = REGION_DZ_SOLID

    # The trap: at −18 ka ``Topo`` is the ICE SURFACE poleward of ICE_EDGE_LAT.
    topo_t = topo_0 + SEA_SURFACE_OFFSET + dz_solid
    topo_t[ice_t] = topo_0[ice_t] + SEA_SURFACE_OFFSET + dz_solid[ice_t] + ICE_THICKNESS

    topo = np.stack([topo_0, topo_t])
    sftlf = np.stack([(topo_0 > 0).astype(np.float64) * 100.0] * 2)
    sftgif = np.stack([np.zeros_like(topo_0), ice_t.astype(np.float64) * 100.0])

    ds = xr.Dataset(
        {
            "Topo": (("time", "lat", "lon"), topo.astype(np.float32)),
            "Orog": (("time", "lat", "lon"), np.clip(topo, 0.0, None).astype(np.float32)),
            "sftlf": (("time", "lat", "lon"), sftlf.astype(np.float32)),
            "sftgif": (("time", "lat", "lon"), sftgif.astype(np.float32)),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture(scope="module")
def global_nc(tmp_path_factory) -> Path:
    return _make_global_ice6g_nc(
        tmp_path_factory.mktemp("ice6g") / "ice6g_global_with_ice.nc"
    )


def _fields(nc: Path, bounds):
    loader = Ice6gLoader(nc)
    try:
        return loader.get_fields(epoch_ka=18, bounds=bounds)
    finally:
        loader.close()


# ===========================================================================
# Job 2 — the datum must use the file's real latitudes
# ===========================================================================


def test_shape_inference_would_refuse_this_grid(global_nc):
    """Guard for the guard: the fixture really is unrecognisable by shape alone."""
    with xr.open_dataset(global_nc) as ds:
        shape = ds["Topo"].isel(time=0).shape
    assert shape == (360, 360)
    with pytest.raises(ValueError, match="not a recognisable global"):
        global_latitude_grid(shape)


def test_datum_resolves_from_real_latitudes(global_nc):
    """``get_fields`` must pass ``lat=`` through, so a 360×360 global grid works.

    With the shape-inferred fallback this raises ValueError before computing
    anything.  Over the tropical control window the answer is then Δz_solid == 0,
    which also proves the datum itself is right: any error in the offset shows up
    here one-for-one.
    """
    _lats, _lons, dz, _ice_h = _fields(global_nc, TROPICAL_BOUNDS)
    assert np.isfinite(dz).all()
    np.testing.assert_allclose(dz, 0.0, atol=1e-3)


# ===========================================================================
# Job 3 — the ice-surface guard must reach production
# ===========================================================================


def test_glaciated_cells_are_nan_not_ice_surface(global_nc):
    """Where ``sftgif > 0`` the loader must return NaN, not a +1800 m ice step."""
    lats, _lons, dz, _ice_h = _fields(global_nc, REGION_BOUNDS)

    ice_rows = lats > ICE_EDGE_LAT
    assert ice_rows.any() and not ice_rows.all(), "window must straddle the margin"

    assert np.isnan(dz[ice_rows]).all(), (
        "cells glaciated at 18 ka came back with a number; ICE-6G 'Topo' is the ice "
        "surface there, so the loader is reporting an ice column as bedrock uplift"
    )
    # …and the NaN must not have eaten the ice-free side (a plain gaussian_filter
    # over a NaN-holed field propagates NaN across the whole kernel support).
    assert np.isfinite(dz[~ice_rows]).all()


def test_no_ice_surface_leaks_across_the_margin(global_nc):
    """Masking must happen BEFORE smoothing: the +1800 m step must not smear out.

    The true Δz_solid is a uniform +40 m everywhere on the ice-free side of the
    margin inside the region.  Smoothing an *unmasked* field with σ=2.0 pulls the ice
    column into those cells (hundreds of metres one to three cells out); masking
    first, then renormalising the kernel over finite cells only, leaves them at +40.
    """
    lats, lons, dz, _ice_h = _fields(global_nc, REGION_BOUNDS)

    # Stay ≥ 8 cells (the σ=2 kernel's reach) away from every edge of the region that
    # is NOT the ice margin, so the only discontinuity in range is the ice itself:
    # 8 cells is 4° in latitude (0.5° cells) and 8° in longitude (1.0° cells).
    interior = (
        (lats[:, None] > REGION_BOUNDS[1] + 5.0)
        & (lats[:, None] <= ICE_EDGE_LAT)
        & (lons[None, :] > REGION_BOUNDS[0] + 9.0)
        & (lons[None, :] < REGION_BOUNDS[2] - 9.0)
    )
    assert interior.sum() > 50

    worst = float(np.nanmax(np.abs(dz[interior] - REGION_DZ_SOLID)))
    assert worst < 1.0, (
        f"ice-free cells inside the region are off the true Δz_solid="
        f"{REGION_DZ_SOLID} m by up to {worst:.1f} m — the {ICE_THICKNESS} m ice "
        "surface is leaking across the margin, i.e. the field is being smoothed "
        "before it is masked"
    )
    # Nothing anywhere in the window may carry an ice-column magnitude.
    assert float(np.nanmax(np.abs(dz))) < 100.0


def test_nan_output_is_announced_not_silently_returned(global_nc):
    """A NaN-bearing Δz must warn, because a downstream nan_to_num reads as 0 m GIA."""
    loader = Ice6gLoader(global_nc)
    try:
        with pytest.warns(GiaDeltaHasIceSurfaceNaNWarning, match="ICE SURFACE"):
            loader.get_fields(epoch_ka=18, bounds=REGION_BOUNDS)
    finally:
        loader.close()


def test_ice_free_window_does_not_warn(global_nc):
    """The NaN warning must be specific — an ice-free window must not raise it."""
    loader = Ice6gLoader(global_nc)
    try:
        with pytest.warns() as record:
            loader.get_fields(epoch_ka=18, bounds=TROPICAL_BOUNDS)
    finally:
        loader.close()
    assert not [w for w in record if issubclass(w.category, GiaDeltaHasIceSurfaceNaNWarning)]


# ===========================================================================
# Backward compatibility — the smoother must not change the ice-free answer
# ===========================================================================


def test_ice_free_window_matches_the_plain_sigma2_gaussian(global_nc):
    """On a window with no ice, output == ``gaussian_filter(Δz_solid, sigma=2.0)``.

    ``smooth_preserving_ice_mask`` is only allowed to differ from the loader's
    historical bare ``gaussian_filter`` where NaN exists — including in boundary
    handling, hence ``mode="reflect"``.  A window with a real step edge inside it is
    used so the filter is not a no-op.
    """
    bounds = (20.0, 40.0, 40.0, 55.0)   # crosses the region's lon=30 / lat=50 edges
    lats, lons, dz, _ice_h = _fields(global_nc, bounds)
    assert np.isfinite(dz).all()

    with xr.open_dataset(global_nc) as ds:
        topo_0 = ds["Topo"].isel(time=0).values.astype(np.float64)
        topo_t = ds["Topo"].isel(time=1).values.astype(np.float64)
        lat = ds["lat"].values
        lon = ds["lon"].values
        sftgif_t = ds["sftgif"].isel(time=1).values.astype(np.float64)
        sftgif_0 = ds["sftgif"].isel(time=0).values.astype(np.float64)

    offset = implied_sea_surface_offset(topo_t, topo_0, lat=lat)
    assert offset == pytest.approx(SEA_SURFACE_OFFSET, abs=1e-3)
    dz_ref_global = solid_displacement(
        topo_t, topo_0, offset, sftgif_t=sftgif_t, sftgif_0=sftgif_0
    )
    sub = dz_ref_global[np.ix_(np.isin(lat, lats), np.isin(lon, lons))]
    assert np.isfinite(sub).all()

    # The step edge must be inside the window, or this test proves nothing.
    assert sub.max() - sub.min() > 10.0
    np.testing.assert_allclose(dz, gaussian_filter(sub, sigma=2.0), atol=1e-4)


# ===========================================================================
# Defect 2 — longitude convention.  0..360 file, −180..180 request.
# ===========================================================================
#
# The synthetic fixture above already runs lon 0.5..359.5, exactly like the real
# files, so the wrap can be pinned here without touching the disk.


def test_bbox_west_of_greenwich_is_not_silently_empty(global_nc):
    """A wholly negative-longitude bbox must return real columns, or raise.

    Before the fix ``(lon >= -11) & (lon <= -4)`` matched **nothing** on a 0..360
    grid, and ``isel`` with an all-False mask happily returned zero-column arrays.
    The caller (``correction_pipeline`` over Ireland, or any Europe-wide extent)
    then got empty arrays with no error at all.
    """
    bounds = (-11.0, 2.0, -4.0, 20.0)   # tropical band → ice-free, datum derivable
    _lats, lons, dz, _ice_h = _fields(global_nc, bounds)

    assert lons.size > 0 and dz.shape[1] == lons.size, (
        "a west-of-Greenwich window came back with zero columns; the 0..360 file "
        "longitudes were compared raw against a negative request"
    )
    assert lons.min() >= -12.0 and lons.max() <= -3.0, (
        f"longitudes must be returned in the requested frame, got {lons.min()}..{lons.max()}"
    )
    np.testing.assert_allclose(dz, 0.0, atol=1e-3)


def test_bbox_straddling_the_prime_meridian_is_contiguous(global_nc):
    """Doggerland-shaped request: the western half must not be dropped.

    ``(lon >= -5) & (lon <= 9)`` is *not* empty on a 0..360 grid — it matches
    0..9 — so this half of the defect never even looked broken.  It silently
    truncated the window at Greenwich.
    """
    bounds = (-4.0, 2.0, 8.0, 20.0)
    _lats, lons, dz, _ice_h = _fields(global_nc, bounds)

    assert (lons < 0).any(), "the western half of the window was dropped"
    assert (lons > 0).any()
    assert np.all(np.diff(lons) > 0), f"longitudes are not ascending: {lons}"
    # No 350°-wide jump: the antimeridian split must have been re-joined.
    assert float(np.max(np.diff(lons))) < 2.0 * LON_STEP
    np.testing.assert_allclose(dz, 0.0, atol=1e-3)


def test_degenerate_selection_raises_rather_than_returning_empty(global_nc):
    """A bbox off the file's grid must raise, never hand back empty arrays."""
    loader = Ice6gLoader(global_nc)
    try:
        with pytest.raises(ValueError, match="(?i)selects no|empty|outside"):
            loader.get_fields(epoch_ka=18, bounds=(5.0, 95.0, 25.0, 99.0))
    finally:
        loader.close()


# ===========================================================================
# Defect 1 — the real per-epoch files.  One epoch each, no time dimension.
# ===========================================================================
#
# Resolved through ``PALEO_DATA_ROOT`` with a repo-relative fallback, the same
# indirection ``scripts/bake_black_sea.py`` uses.  A hardcoded ``D:/…`` literal here
# would make every one of these tests skip *silently* on any other machine (CI box
# LXC 128 included) while still reporting green — that has happened before.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PALEO_DATA_ROOT = _REPO_ROOT.parents[1] / "Paleo-Europe_3D"


def _ice6g_raw_dir() -> Path:
    root = Path(os.environ.get("PALEO_DATA_ROOT") or _DEFAULT_PALEO_DATA_ROOT)
    return root.resolve() / "data" / "raw" / "ICE6G"


ICE6G_RAW_DIR = _ice6g_raw_dir()
REAL_EPOCH_KA = 16.5
REAL_REFERENCE_KA = 0.0
_REAL_FILES = (
    ICE6G_RAW_DIR / ice6g_epoch_filename(REAL_EPOCH_KA),
    ICE6G_RAW_DIR / ice6g_epoch_filename(REAL_REFERENCE_KA),
)

requires_real_ice6g = pytest.mark.skipif(
    not all(p.exists() for p in _REAL_FILES),
    reason=f"real ICE-6G per-epoch files not found under {ICE6G_RAW_DIR} "
           "(set PALEO_DATA_ROOT to the Paleo-Europe_3D tree)",
)

# Manych spillway — the corridor the Khvalynian transgression drains through.
MANYCH_LON, MANYCH_LAT = 44.119, 45.773
MANYCH_BOUNDS = (42.0, 44.0, 46.5, 47.5)
MANYCH_MODERN_SILL_M = 23.00
#: Δz_solid at the 10 arcmin cell nearest the sill at 16.5 ka, from the corrected
#: far-field datum (77.9058 m).  Raw value +2.7010 m; ``get_fields`` returns it after
#: the σ=2 Gaussian, which over this ice-free corridor moves it by +0.025 m.
MANYCH_DZ_16K5_M = 2.701
#: What the retired deep-ocean datum produced here.  Kept only as a tripwire.
MANYCH_DZ_DEEP_OCEAN_ARTEFACT_M = -26.5


def _real_pair_loader() -> Ice6gLoader:
    return Ice6gLoader.from_epoch_directory(
        ICE6G_RAW_DIR, REAL_EPOCH_KA, reference_ka=REAL_REFERENCE_KA
    )


@requires_real_ice6g
def test_real_per_epoch_files_open_at_all():
    """The shipped files have no ``time`` dimension; opening one must not KeyError."""
    loader = _real_pair_loader()
    try:
        lats, lons, dz, _ice_h = loader.get_fields(
            epoch_ka=REAL_EPOCH_KA, bounds=MANYCH_BOUNDS
        )
    finally:
        loader.close()
    assert dz.shape == (lats.size, lons.size)
    assert lats.size > 5 and lons.size > 5


@requires_real_ice6g
def test_manych_sill_matches_the_corrected_datum():
    """+2.701 m of UPLIFT at the sill at 16.5 ka — not the −26.5 m the bug produced.

    This is the whole point of the 2026-07-30 datum fix reaching production: the
    palaeo-sill stands at 23.00 + 2.70 = 25.70 m a.s.l., i.e. **higher** than today,
    so the Khvalynian spill west is harder than a modern-sill reading, not easier.
    """
    loader = _real_pair_loader()
    try:
        lats, lons, dz, _ice_h = loader.get_fields(
            epoch_ka=REAL_EPOCH_KA, bounds=MANYCH_BOUNDS
        )
    finally:
        loader.close()

    i = int(np.argmin(np.abs(lats - MANYCH_LAT)))
    j = int(np.argmin(np.abs(lons - MANYCH_LON)))
    value = float(dz[i, j])

    assert np.isfinite(value), "the sill cell is ice-free at 16.5 ka; NaN is a bug"
    assert value == pytest.approx(MANYCH_DZ_16K5_M, abs=0.05), (
        f"Δz_solid at the Manych sill is {value:.4f} m, expected "
        f"{MANYCH_DZ_16K5_M} m (± the σ=2 smoothing).  If it is near "
        f"{MANYCH_DZ_DEEP_OCEAN_ARTEFACT_M} m the deep-ocean datum is back."
    )
    sill = MANYCH_MODERN_SILL_M + value
    assert sill == pytest.approx(25.70, abs=0.05), (
        f"palaeo-sill {sill:.2f} m a.s.l.; the corrected value is 25.70 m"
    )


@requires_real_ice6g
def test_negative_longitude_bbox_on_the_real_files():
    """Doggerland on the real 0..360 grid: never empty, never truncated."""
    loader = _real_pair_loader()
    try:
        _lats, lons, dz, _ice_h = loader.get_fields(
            epoch_ka=REAL_EPOCH_KA, bounds=(-4.0, 51.0, 8.0, 57.0)
        )
    finally:
        loader.close()

    assert lons.size > 0 and dz.size > 0, "Doggerland window came back empty"
    assert (lons < 0).any(), "the western half of Doggerland was silently dropped"
    assert np.all(np.diff(lons) > 0)
    assert lons.min() >= -5.001 and lons.max() <= 9.001


@requires_real_ice6g
def test_glaciated_cells_are_nan_on_the_real_files():
    """Fennoscandia at 16.5 ka: ``sftgif > 0`` → NaN, not a ~+1700 m ice column."""
    loader = _real_pair_loader()
    try:
        with pytest.warns(GiaDeltaHasIceSurfaceNaNWarning, match="ICE SURFACE"):
            _lats, _lons, dz, _ice_h = loader.get_fields(
                epoch_ka=REAL_EPOCH_KA, bounds=(10.0, 58.0, 30.0, 72.0)
            )
    finally:
        loader.close()

    assert np.isnan(dz).any(), "no NaN over an ice sheet — the guard is not wired in"
    assert np.isfinite(dz).any(), "the whole window is NaN — the guard is too greedy"
    assert float(np.nanmax(np.abs(dz))) < 500.0, (
        "an ice-column magnitude survived into Δz_solid"
    )


@requires_real_ice6g
def test_epoch_mismatch_between_request_and_file_is_refused():
    """Asking a 16.5 ka pair for 21 ka must raise, not answer with the wrong epoch."""
    loader = _real_pair_loader()
    try:
        with pytest.raises(ValueError, match="(?i)epoch"):
            loader.get_fields(epoch_ka=21, bounds=MANYCH_BOUNDS)
    finally:
        loader.close()


@requires_real_ice6g
def test_single_epoch_file_alone_refuses_with_guidance():
    """One file cannot make a delta.  The refusal must name the epoch-pair API."""
    loader = Ice6gLoader(_REAL_FILES[0])
    try:
        with pytest.raises(ValueError) as exc:
            loader.get_fields(epoch_ka=REAL_EPOCH_KA, bounds=MANYCH_BOUNDS)
    finally:
        loader.close()
    msg = str(exc.value)
    assert "from_epoch_files" in msg or "from_epoch_directory" in msg, (
        f"the refusal must point at the epoch-pair API; got: {msg}"
    )


def test_epoch_filename_convention():
    """Integers bare, half-ka to one decimal — the shipped naming, exactly."""
    assert ice6g_epoch_filename(16) == "I6_C.VM5a_10min.16.nc"
    assert ice6g_epoch_filename(16.0) == "I6_C.VM5a_10min.16.nc"
    assert ice6g_epoch_filename(16.5) == "I6_C.VM5a_10min.16.5.nc"
    assert ice6g_epoch_filename(0) == "I6_C.VM5a_10min.0.nc"
    assert ice6g_epoch_filename(0.5) == "I6_C.VM5a_10min.0.5.nc"


def test_missing_epoch_file_raises_naming_the_path(tmp_path):
    """A missing epoch must fail at construction with the path, not later."""
    with pytest.raises(FileNotFoundError, match="I6_C.VM5a_10min.16.5.nc"):
        Ice6gLoader.from_epoch_directory(tmp_path, 16.5, reference_ka=0.0)


# ===========================================================================
# The 43-epoch fixture must keep loading through the time axis
# ===========================================================================


def test_multi_epoch_fixture_still_loads(ice6g_path):
    """``tests/fixtures/ice6g/ICE6G_fixture.nc`` — 43 epochs on a ``time`` axis.

    It is a *regional* cut (57–68 °N, 1–12 °E), so it holds no far-field tropical
    land and the datum cannot be derived from it — that refusal is correct and is
    pinned in ``tests/test_gia.py``.  Here it is supplied explicitly, which is the
    documented escape hatch, so that the **time-axis path itself** stays covered.
    """
    if not ice6g_path.exists():
        pytest.skip("ICE-6G fixture file not found")

    with xr.open_dataset(ice6g_path) as ds:
        assert ds.sizes["time"] == 43

    loader = Ice6gLoader(ice6g_path)
    try:
        with pytest.warns(GiaDeltaHasIceSurfaceNaNWarning):
            lats, lons, dz, ice_h = loader.get_fields(
                epoch_ka=18,
                bounds=(4.0, 59.0, 10.0, 65.0),
                sea_surface_offset=81.958,   # measured once on the global 18 ka file
            )
    finally:
        loader.close()

    assert dz.shape == (lats.size, lons.size) == ice_h.shape
    assert lats.size > 0 and lons.size > 0
    assert np.isnan(dz).any(), "Scandinavia at 18 ka is glaciated; expected NaN"


def test_multi_epoch_fixture_without_a_datum_still_refuses(ice6g_path):
    """The regional cut must not improvise a datum just because the API grew."""
    if not ice6g_path.exists():
        pytest.skip("ICE-6G fixture file not found")

    loader = Ice6gLoader(ice6g_path)
    try:
        with pytest.raises(ValueError, match="(?i)reference set|global"):
            loader.get_fields(epoch_ka=18, bounds=(4.0, 59.0, 10.0, 65.0))
    finally:
        loader.close()
