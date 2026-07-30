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

Deliberately in its own file: ``tests/test_gia.py`` and ``tests/test_gia_topo.py`` are
being worked on concurrently.

The fixture is synthetic but reproduces the three traps of the real
``I6_C.VM5a_10min.*.nc`` files: ``Topo`` referenced to the *contemporaneous* sea
surface, ``Orog`` clamped to 0 over water, and ``Topo`` = the **ice surface** wherever
``sftgif > 0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from paleoeurope.gia.ice6g_loader import (
    GiaDeltaHasIceSurfaceNaNWarning,
    Ice6gLoader,
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
