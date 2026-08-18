"""
tests/test_gia.py

Unit tests for paleoeurope.gia: apply_gia_delta, IceEnvelope, Ice6gLoader.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from paleoeurope.gia.correction_matrix import build_correction_matrix
from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.envelope import IceEnvelope
from paleoeurope.gia.ice6g_topo import implied_sea_surface_offset, solid_displacement
from paleoeurope.utils.grid import make_pixel_coords

# Synthetic model grid covering the test tile
MODEL_LATS = np.array([50.0, 50.5, 51.0, 51.5, 52.0, 52.5])
MODEL_LONS = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])


# ===========================================================================
# apply_gia_delta
# ===========================================================================


class TestApplyGiaDelta:
    def _make_inputs(self, delta_val: float = -50.0, ice_val: float = 0.0):
        """Return a minimal set of inputs for apply_gia_delta."""
        modern = np.zeros((10, 10), dtype=np.float32)
        dz = np.full((len(MODEL_LATS), len(MODEL_LONS)), delta_val, dtype=np.float32)
        ice_h = np.full((len(MODEL_LATS), len(MODEL_LONS)), ice_val, dtype=np.float32)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (10, 10))
        return modern, dz, ice_h, tile_lats, tile_lons

    def test_uniform_delta_applied(self):
        """Uniform Δz should shift all non-ice pixels by exactly that amount."""
        modern, dz, ice_h, tile_lats, tile_lons = self._make_inputs(delta_val=-50.0)
        paleo, ice_tile = apply_gia_delta(
            modern, MODEL_LATS, MODEL_LONS, dz, ice_h, tile_lats, tile_lons
        )
        assert np.all(ice_tile == 0.0), "No ice expected"
        np.testing.assert_array_almost_equal(paleo, -50.0, decimal=3)

    def test_zero_delta_is_identity(self):
        """Δz == 0 should reproduce the modern DEM."""
        modern = np.random.default_rng(7).random((10, 10)).astype(np.float32) * 200
        dz = np.zeros((len(MODEL_LATS), len(MODEL_LONS)), dtype=np.float32)
        ice_h = np.zeros_like(dz)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (10, 10))
        paleo, ice_tile = apply_gia_delta(modern, MODEL_LATS, MODEL_LONS, dz, ice_h, tile_lats, tile_lons)
        np.testing.assert_array_almost_equal(paleo, modern, decimal=3)

    def test_full_ice_coverage_returns_ice_tile(self):
        """Full ice coverage → ice_tile should reflect the ice thickness."""
        modern, dz, _, tile_lats, tile_lons = self._make_inputs()
        ice_h = np.full((len(MODEL_LATS), len(MODEL_LONS)), 2000.0, dtype=np.float32)
        paleo, ice_tile = apply_gia_delta(modern, MODEL_LATS, MODEL_LONS, dz, ice_h, tile_lats, tile_lons)
        assert np.all(ice_tile == 2000.0), "All pixels should be ice-covered with 2000m"
        assert not np.isnan(paleo).any(), "Paleo bedrock should not be NaN"

    def test_output_shape_preserved(self):
        """Output shape must match the modern DEM shape."""
        modern = np.zeros((15, 20), dtype=np.float32)
        dz = np.zeros((len(MODEL_LATS), len(MODEL_LONS)), dtype=np.float32)
        ice_h = np.zeros_like(dz)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (15, 20))
        paleo, ice_tile = apply_gia_delta(modern, MODEL_LATS, MODEL_LONS, dz, ice_h, tile_lats, tile_lons)
        assert paleo.shape == (15, 20)
        assert ice_tile.shape == (15, 20)

    def test_descending_model_lats_handled(self):
        """apply_gia_delta must work even if model_lats is descending."""
        modern = np.zeros((10, 10), dtype=np.float32)
        dz = np.full((len(MODEL_LATS), len(MODEL_LONS)), -30.0, dtype=np.float32)
        ice_h = np.zeros_like(dz)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (10, 10))
        paleo, _ = apply_gia_delta(
            modern, MODEL_LATS[::-1], MODEL_LONS, dz[::-1, :], ice_h[::-1, :], tile_lats, tile_lons
        )
        np.testing.assert_array_almost_equal(paleo, -30.0, decimal=3)


# ===========================================================================
# IceEnvelope
# ===========================================================================


class TestIceEnvelope:
    def _make_envelope(self, ice_val: float, dz_val: float = 0.0) -> IceEnvelope:
        ice_h = np.full((len(MODEL_LATS), len(MODEL_LONS)), ice_val, dtype=np.float32)
        dz = np.full((len(MODEL_LATS), len(MODEL_LONS)), dz_val, dtype=np.float32)
        return IceEnvelope(MODEL_LATS.copy(), MODEL_LONS.copy(), ice_h, dz)

    def test_no_ice_returns_bedrock(self):
        """Zero ice height → surface equals bedrock (modern + dz)."""
        env = self._make_envelope(ice_val=0.0, dz_val=-20.0)
        modern = np.ones((10, 10), dtype=np.float32) * 100.0
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (10, 10))
        surface = env.surface_elevation(modern, tile_lats, tile_lons)
        np.testing.assert_array_almost_equal(surface, 80.0, decimal=3)

    def test_full_ice_returns_ice_surface(self):
        """Full ice → surface elevation = bedrock + ice_height."""
        env = self._make_envelope(ice_val=1000.0, dz_val=-50.0)
        modern = np.zeros((10, 10), dtype=np.float32)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (10, 10))
        surface = env.surface_elevation(modern, tile_lats, tile_lons)
        # bedrock = 0 - 50 = -50; ice_surface = -50 + 1000 = 950
        np.testing.assert_array_almost_equal(surface, 950.0, decimal=1)

    def test_output_shape(self):
        """Output shape must match modern DEM shape."""
        env = self._make_envelope(ice_val=500.0)
        modern = np.zeros((12, 15), dtype=np.float32)
        tile_lats, tile_lons = make_pixel_coords((0.0, 51.0, 1.0, 52.0), (12, 15))
        surface = env.surface_elevation(modern, tile_lats, tile_lons)
        assert surface.shape == (12, 15)


# ===========================================================================
# Ice6gLoader (uses synthetic .nc)
# ===========================================================================


class TestIce6gLoader:
    def test_loader_requires_existing_file(self, tmp_path):
        """Ice6gLoader should raise FileNotFoundError for missing file."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        with pytest.raises(FileNotFoundError):
            Ice6gLoader(tmp_path / "nonexistent.nc")

    def test_regional_fixture_cannot_define_the_global_datum(self, ice6g_path):
        """A regional ICE-6G cut must be REFUSED, not silently improvised on.

        Rewritten 2026-07-30.  This test used to assert ``|Δz|.max() > 0`` for
        Norway at 21 ka off ``ICE6G_fixture.nc`` — a 66×67 cut spanning
        57–68 N / 1–12 E.  That assertion passed under the retired estimator
        only because it took its datum from whatever deep cells happened to be
        inside the window: a *regional* deep-ocean median, ice-proximal and
        therefore biased, which is precisely the defect that put a ~−30 m
        spurious lowering into every baked BlackSea raster.

        The sea-surface datum is a **global far-field** quantity.  A window with
        no ice-free tropical continental land cannot define it, and the honest
        behaviour is to say so rather than to return a plausible-looking number.
        So the invariant worth pinning here is the refusal itself — the physics
        is exercised on global synthetic grids by
        :class:`TestIce6gLoaderDeltaIsSolidDisplacement` and on the real files by
        :class:`TestIce6gLoaderRealData`.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        if not ice6g_path.exists():
            pytest.skip("ICE-6G fixture file not found")

        loader = Ice6gLoader(ice6g_path)
        with pytest.raises(ValueError) as exc:
            loader.get_fields(
                epoch_ka=21,
                bounds=(4.0, 59.0, 10.0, 65.0),   # Norway — inside fixture coverage
            )
        msg = str(exc.value)
        assert "reference set" in msg or "global" in msg, (
            "a regional cut must be refused with a message that names the "
            f"reason (reference set / global datum); got: {msg}"
        )


# ===========================================================================
# Ice6gLoader — the GIA delta must be the ``ice6g_topo`` Δz_solid, never ΔOrog
# ===========================================================================

# Real ICE-6G_C VM5a distribution (one NetCDF per epoch, no time dimension).
#
# Resolved through the same indirection as ``scripts/bake_black_sea.py`` so that
# no drive letter is baked in: ``PALEO_DATA_ROOT`` wins, otherwise the sibling
# ``Paleo-Europe_3D`` tree next to this repository.  A hardcoded ``D:/…`` here
# would make the one real-data test skip *silently* on every other machine
# (CI box LXC 128 included) while still reporting green.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PALEO_DATA_ROOT = _REPO_ROOT.parents[1] / "Paleo-Europe_3D"


def _resolve_ice6g_raw_dir() -> Path:
    """Directory holding ``I6_C.VM5a_10min.<epoch>.nc``, env-overridable."""
    root = Path(os.environ.get("PALEO_DATA_ROOT") or _DEFAULT_PALEO_DATA_ROOT)
    return root.resolve() / "data" / "raw" / "ICE6G"


ICE6G_RAW_DIR = _resolve_ice6g_raw_dir()
_ICE6G_RAW_EPOCHS = (
    ICE6G_RAW_DIR / "I6_C.VM5a_10min.16.5.nc",
    ICE6G_RAW_DIR / "I6_C.VM5a_10min.0.nc",
)

# Manych spillway — the corridor the Khvalynian transgression drains through.
MANYCH_LON, MANYCH_LAT = 44.119, 45.773
MANYCH_BOUNDS = (42.0, 44.0, 46.5, 47.5)   # lon_min, lat_min, lon_max, lat_max

# Synthetic scenario constants (metres).
SEA_SURFACE_OFFSET = 112.0   # eustatic lowstand baked into ICE-6G ``Topo``
REGION_DZ_SOLID = -30.0      # true solid displacement over the test region
REGION_TOPO_0 = -50.0        # region is shallow water today → ``Orog`` == 0 there
DEEP_TOPO_0 = -3000.0        # far-field deep ocean

# --- Far-field reference set -------------------------------------------------
#
# The sea-surface datum is the median ΔTopo over ice-free tropical CONTINENTAL
# land (|lat| <= 25, topo_0 > 0, never glaciated, far from ice).  Every builder
# below must therefore carry some, or ``implied_sea_surface_offset`` correctly
# refuses to invent a datum.  These fixtures originally held no land above sea
# level anywhere — topo_0 was −3000 m ocean plus a −50 m shelf — so the
# reference set came out empty.
#
# The band is placed at |lat| <= 20 (strictly inside the 25° selector, so no
# cell is marginal), across lon 0–120°E, and given zero solid displacement:
# the datum measured over it is exactly SEA_SURFACE_OFFSET, which is what the
# assertions downstream expect.  It cannot collide with the Manych test region
# (lat 35–55) or the synthetic ice cap (lat > 60).
TROPICS_LAT_ABS = 20.0
TROPICS_LON_MAX = 120.0
TROPICS_TOPO_0 = 200.0       # continental interior, well above sea level


def _add_tropical_reference_land(
    topo_0: np.ndarray, dz_solid: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray
) -> None:
    """Stamp the far-field reference band into *topo_0* / *dz_solid* in place."""
    band = (
        (np.abs(lat2d) <= TROPICS_LAT_ABS)
        & (lon2d >= 0.0)
        & (lon2d <= TROPICS_LON_MAX)
    )
    topo_0[band] = TROPICS_TOPO_0
    dz_solid[band] = 0.0


def _make_ice6g_like_nc(
    path: Path,
    *,
    ice_vars: tuple[str, ...] = ("sice",),
    include_orog: bool = True,
    thickness_value: float = 1500.0,
) -> Path:
    """Write a tiny ICE-6G_C-shaped NetCDF with the real ``Orog``-clamping signature.

    Two epochs (0 and −18 ka) on a coarse global grid.  ``Topo`` carries
    bathymetry; ``Orog`` is ``max(Topo, 0)`` — clamped to 0 over water, exactly
    as in the distributed ``I6_C.VM5a_10min.*.nc`` files.  The far field rises by
    :data:`SEA_SURFACE_OFFSET` with no solid motion, while a region around the
    Manych corridor additionally moves by :data:`REGION_DZ_SOLID`.

    Consequences that the loader must reproduce:
      * ``Δz_solid``  == ``REGION_DZ_SOLID``      (−30 m: the correct answer)
      * ``ΔOrog``     == +32 m over that region   (the wrong-physics answer)
    """
    lats = np.arange(-80.0, 80.1, 2.0, dtype=np.float32)
    lons = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
    times = np.array([0.0, -18.0], dtype=np.float32)

    lon2d, lat2d = np.meshgrid(lons, lats)
    region = (lon2d >= 30.0) & (lon2d <= 60.0) & (lat2d >= 35.0) & (lat2d <= 55.0)

    topo_0 = np.full(lat2d.shape, DEEP_TOPO_0, dtype=np.float64)
    topo_0[region] = REGION_TOPO_0

    dz_solid = np.zeros_like(topo_0)
    dz_solid[region] = REGION_DZ_SOLID
    _add_tropical_reference_land(topo_0, dz_solid, lat2d, lon2d)

    topo_t = topo_0 + SEA_SURFACE_OFFSET + dz_solid

    topo = np.stack([topo_0, topo_t]).astype(np.float32)
    data_vars = {"Topo": (("time", "lat", "lon"), topo)}
    if include_orog:
        data_vars["Orog"] = (("time", "lat", "lon"), np.clip(topo, 0.0, None))
    for name in ice_vars:
        if name == "sftgif":
            # Real ICE-6G icemask: percent, 100 = ice.
            mask = np.zeros_like(topo)
            mask[1][lat2d > 60.0] = 100.0
            data_vars[name] = (("time", "lat", "lon"), mask)
        else:
            # *thickness_value* is written verbatim, so a caller can encode the
            # field in centimetres (e.g. 300 000 cm) and exercise the loader's
            # cm→m heuristic.
            thick = np.zeros_like(topo)
            thick[1][lat2d > 60.0] = thickness_value
            data_vars[name] = (("time", "lat", "lon"), thick)

    ds = xr.Dataset(data_vars, coords={"time": times, "lat": lats, "lon": lons})
    ds.to_netcdf(path)
    ds.close()
    return path


# --- Structured field used by the σ=2.0 smoothing tests ---------------------
#
# ``_make_ice6g_like_nc`` is deliberately *uniform* over the requested bbox,
# which makes ``gaussian_filter`` a mathematical no-op there: a test built on it
# cannot tell a smoothed loader from an unsmoothed one.  These constants build a
# field with a real step edge *inside* the bbox instead.  The 2° grid above only
# yields a 3×3 subset over MANYCH_BOUNDS±1°, far too coarse to resolve a σ=2.0
# kernel, so the structured file is written at 0.5°.
STEP_RES_DEG = 0.5
STEP_EDGE_LON = 44.0        # inside MANYCH_BOUNDS, ≥3σ from both subset edges
STEP_DZ_WEST = -40.0        # solid displacement west of the edge
STEP_DZ_EAST = 0.0          # …and east of it


def _make_ice6g_like_nc_stepped(path: Path) -> Path:
    """Like :func:`_make_ice6g_like_nc` but with a Δz_solid **step edge** in-bbox.

    Same ``Orog``-clamping signature and same far-field :data:`SEA_SURFACE_OFFSET`,
    but ``Δz_solid`` jumps from :data:`STEP_DZ_WEST` to :data:`STEP_DZ_EAST` across
    :data:`STEP_EDGE_LON`.  The discontinuity sits well inside the requested
    subset, so a σ=2.0 Gaussian has a large, measurable effect and the smoothing
    step can no longer be deleted without a test noticing.
    """
    lats = np.arange(-80.0, 80.01, STEP_RES_DEG, dtype=np.float32)
    lons = np.arange(0.0, 360.0, STEP_RES_DEG, dtype=np.float32)
    times = np.array([0.0, -18.0], dtype=np.float32)

    lon2d, lat2d = np.meshgrid(lons, lats)
    region = (lon2d >= 30.0) & (lon2d <= 60.0) & (lat2d >= 35.0) & (lat2d <= 55.0)

    topo_0 = np.full(lat2d.shape, DEEP_TOPO_0, dtype=np.float64)
    topo_0[region] = REGION_TOPO_0

    dz_solid = np.zeros_like(topo_0)
    dz_solid[region & (lon2d < STEP_EDGE_LON)] = STEP_DZ_WEST
    dz_solid[region & (lon2d >= STEP_EDGE_LON)] = STEP_DZ_EAST
    _add_tropical_reference_land(topo_0, dz_solid, lat2d, lon2d)

    topo_t = topo_0 + SEA_SURFACE_OFFSET + dz_solid
    topo = np.stack([topo_0, topo_t]).astype(np.float32)

    thick = np.zeros_like(topo)
    thick[1][lat2d > 60.0] = 1500.0

    ds = xr.Dataset(
        {
            "Topo": (("time", "lat", "lon"), topo),
            "Orog": (("time", "lat", "lon"), np.clip(topo, 0.0, None)),
            "sice": (("time", "lat", "lon"), thick),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


class TestIce6gLoaderSmoothing:
    """The σ=2.0 Gaussian on the model grid must actually be applied.

    Regression guard: with a spatially *uniform* Δz_solid over the bbox the
    filter is a no-op, so every previous assertion here passed unchanged when
    ``gaussian_filter`` was deleted from ``ice6g_loader.get_fields``.  These
    tests run on a field with a step edge inside the bbox, where the filter
    changes the answer by ~10 m.
    """

    @staticmethod
    def _reference_unsmoothed(nc: Path, lats_sub, lons_sub) -> np.ndarray:
        """Δz_solid on the requested subset, recomputed with **no** smoothing."""
        with xr.open_dataset(nc) as ds:
            topo_0 = ds["Topo"].isel(time=0).values.astype(np.float64)
            topo_t = ds["Topo"].isel(time=1).values.astype(np.float64)
            lat = ds["lat"].values
            lon = ds["lon"].values
        # Pass the real latitudes: these fixtures are global in coverage but not
        # cell-centred at a resolution the shape heuristic recognises, and the
        # production loader passes ``lat=`` for exactly the same reason.
        offset = implied_sea_surface_offset(topo_t, topo_0, lat=lat)
        dz_global = solid_displacement(topo_t, topo_0, offset)
        return dz_global[np.ix_(np.isin(lat, lats_sub), np.isin(lon, lons_sub))]

    def test_step_edge_is_actually_smoothed(self, tmp_path):
        """Loader output must differ from the raw field, and match σ=2.0 exactly.

        (a) at the discontinuity the smoothed answer departs from the raw one by
            metres — proof the filter ran;
        (b) it equals ``gaussian_filter(reference, sigma=2.0)`` to 1e-3 — proof it
            ran with the documented σ and nothing else.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc_stepped(tmp_path / "ice6g_step_edge.nc")
        loader = Ice6gLoader(nc)
        try:
            lats_sub, lons_sub, dz, _ice_h = loader.get_fields(
                epoch_ka=18, bounds=MANYCH_BOUNDS
            )
        finally:
            loader.close()

        dz_ref = self._reference_unsmoothed(nc, lats_sub, lons_sub)

        # Sanity: the synthetic field really is discontinuous inside the bbox.
        assert dz_ref.min() == pytest.approx(STEP_DZ_WEST, abs=1e-3)
        assert dz_ref.max() == pytest.approx(STEP_DZ_EAST, abs=1e-3)

        # (a) smoothing is observable — an unsmoothed loader fails here.
        max_dev = float(np.abs(dz - dz_ref).max())
        assert max_dev > 1.0, (
            f"loader output is within {max_dev:.4f} m of the UNSMOOTHED field "
            "across a 40 m step edge — the σ=2.0 gaussian_filter is not being "
            "applied in Ice6gLoader.get_fields"
        )

        # (b) …and it is exactly the documented σ=2.0 kernel.
        np.testing.assert_allclose(
            dz, gaussian_filter(dz_ref, sigma=2.0), atol=1e-3
        )

    def test_smoothing_conserves_the_field_mean_and_bounds(self, tmp_path):
        """A Gaussian neither creates extrema nor shifts the bulk level."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc_stepped(tmp_path / "ice6g_step_edge_stats.nc")
        loader = Ice6gLoader(nc)
        try:
            lats_sub, lons_sub, dz, _ice_h = loader.get_fields(
                epoch_ka=18, bounds=MANYCH_BOUNDS
            )
        finally:
            loader.close()

        dz_ref = self._reference_unsmoothed(nc, lats_sub, lons_sub)
        assert dz.min() >= STEP_DZ_WEST - 1e-3
        assert dz.max() <= STEP_DZ_EAST + 1e-3
        # Strictly inside the raw range → the sharp edge was rounded off.
        assert dz.max() < STEP_DZ_EAST - 1e-3 or dz.min() > STEP_DZ_WEST + 1e-3
        assert float(dz.mean()) == pytest.approx(float(dz_ref.mean()), abs=2.0)


class TestIce6gLoaderDeltaIsSolidDisplacement:
    """The loader's Δz must come from :mod:`paleoeurope.gia.ice6g_topo`."""

    def test_delta_equals_solid_displacement_not_orog(self, tmp_path):
        """ΔOrog is clamped to 0 over water — the loader must not use it.

        Over the synthetic Manych region the correct Δz_solid is −30 m while
        ΔOrog is +32 m.  A loader on the ``Orog`` path returns +32 and would
        raise the +23 m sill to ≈ +55 m, so the spill could never happen.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc(tmp_path / "ice6g_orog_clamped.nc")
        loader = Ice6gLoader(nc)
        try:
            _lats, _lons, dz, _ice_h = loader.get_fields(
                epoch_ka=18, bounds=MANYCH_BOUNDS
            )
        finally:
            loader.close()

        assert np.allclose(dz, REGION_DZ_SOLID, atol=1e-3), (
            f"expected Δz_solid={REGION_DZ_SOLID} m over the region, got "
            f"[{dz.min():.2f}, {dz.max():.2f}] m — the loader is not using "
            "ice6g_topo.solid_displacement"
        )
        # The wrong-physics answer must be nowhere in the output.
        orog_delta = SEA_SURFACE_OFFSET + REGION_DZ_SOLID + REGION_TOPO_0  # +32 m
        assert not np.any(np.isclose(dz, orog_delta, atol=1e-3)), (
            "loader returned the ΔOrog value — the Orog branch is still live"
        )

    def test_delta_matches_ice6g_topo_reference_computation(self, tmp_path):
        """Independently recompute Δz_solid via ice6g_topo and compare.

        The reference is smoothed with the same σ=2.0 the loader documents.  On
        *this* fixture the field is spatially uniform over the bbox, so the
        filter is a no-op and this test says nothing about smoothing — that is
        :class:`TestIce6gLoaderSmoothing`'s job, on a stepped field.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc(tmp_path / "ice6g_reference.nc")

        with xr.open_dataset(nc) as ds:
            topo_0 = ds["Topo"].isel(time=0).values.astype(np.float64)
            topo_t = ds["Topo"].isel(time=1).values.astype(np.float64)
            lat = ds["lat"].values
            lon = ds["lon"].values
            offset = implied_sea_surface_offset(topo_t, topo_0, lat=lat)
            dz_ref_global = solid_displacement(topo_t, topo_0, offset)

        assert offset == pytest.approx(SEA_SURFACE_OFFSET, abs=1e-3)

        loader = Ice6gLoader(nc)
        try:
            lats_sub, lons_sub, dz, _ice_h = loader.get_fields(
                epoch_ka=18, bounds=MANYCH_BOUNDS
            )
        finally:
            loader.close()

        lat_sel = np.isin(lat, lats_sub)
        lon_sel = np.isin(lon, lons_sub)
        dz_ref = gaussian_filter(dz_ref_global[np.ix_(lat_sel, lon_sel)], sigma=2.0)
        assert dz.shape == dz_ref.shape
        np.testing.assert_allclose(dz, dz_ref, atol=1e-3)

    def test_sea_surface_offset_uses_the_global_grid(self, tmp_path):
        """The requested bbox has no deep ocean; the offset must still resolve.

        ``implied_sea_surface_offset`` needs far-field deep-ocean cells.  If the
        loader computed it on the spatial subset it would raise ValueError here.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc(tmp_path / "ice6g_no_local_deep.nc")
        loader = Ice6gLoader(nc)
        try:
            _lats, _lons, dz, _ice_h = loader.get_fields(
                epoch_ka=18, bounds=MANYCH_BOUNDS
            )
        finally:
            loader.close()
        assert np.isfinite(dz).all()

    def test_missing_topo_raises_naming_available_variables(self, tmp_path):
        """No ``Topo`` → a clear ValueError, not a silent wrong-physics fallback."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        lats = np.arange(-80.0, 80.1, 2.0, dtype=np.float32)
        lons = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
        times = np.array([0.0, -18.0], dtype=np.float32)
        shape = (len(times), len(lats), len(lons))
        ds = xr.Dataset(
            {
                "stgr": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
                "sealevel": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
                "sice": (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )
        nc = tmp_path / "ice6g_no_topo.nc"
        ds.to_netcdf(nc)
        ds.close()

        loader = Ice6gLoader(nc)
        try:
            with pytest.raises(ValueError, match="Topo"):
                loader.get_fields(epoch_ka=18, bounds=MANYCH_BOUNDS)
        finally:
            loader.close()


class TestIce6gLoaderIceDetection:
    """Ice detection must survive the variable names the real files actually use."""

    def test_real_icemask_variable_name_is_accepted(self, tmp_path):
        """``I6_C.VM5a_10min.*.nc`` carries ``sftgif``, never ``sice``."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc(tmp_path / "ice6g_sftgif.nc", ice_vars=("sftgif",))
        loader = Ice6gLoader(nc)
        try:
            with pytest.warns(UserWarning, match="thickness"):
                _lats, _lons, dz, ice_h = loader.get_fields(
                    epoch_ka=18, bounds=MANYCH_BOUNDS
                )
        finally:
            loader.close()

        assert np.allclose(dz, REGION_DZ_SOLID, atol=1e-3)
        assert ice_h.shape == dz.shape
        assert np.isfinite(ice_h).all()
        assert (ice_h >= 0.0).all()

    def test_missing_ice_variable_does_not_kill_the_gia_delta(self, tmp_path):
        """No ice variable at all → warn, return zeros, keep Δz usable."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = _make_ice6g_like_nc(tmp_path / "ice6g_no_ice.nc", ice_vars=())
        loader = Ice6gLoader(nc)
        try:
            with pytest.warns(UserWarning):
                _lats, _lons, dz, ice_h = loader.get_fields(
                    epoch_ka=18, bounds=MANYCH_BOUNDS
                )
        finally:
            loader.close()

        assert np.allclose(dz, REGION_DZ_SOLID, atol=1e-3)
        assert ice_h.shape == dz.shape
        assert np.all(ice_h == 0.0)

    @staticmethod
    def _ice_max(nc: Path) -> float:
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        loader = Ice6gLoader(nc)
        try:
            _lats, _lons, _dz, ice_h = loader.get_fields(
                epoch_ka=18, bounds=(0.0, 62.0, 10.0, 70.0)
            )
        finally:
            loader.close()
        return float(ice_h.max())

    def test_thickness_variable_still_preferred_over_the_mask(self, tmp_path):
        """``sice`` wins over ``sftgif`` when both are present.

        1 500 m sits *below* the 5 000 m cm-heuristic cut-off, so this case
        pins the metres branch: the value must come through un-divided.
        """
        nc = _make_ice6g_like_nc(
            tmp_path / "ice6g_both.nc",
            ice_vars=("sftgif", "sice"),
            thickness_value=1500.0,
        )
        assert self._ice_max(nc) == pytest.approx(1500.0)

    def test_centimetre_encoded_thickness_is_converted_to_metres(self, tmp_path):
        """The ``/100`` branch, which the metres case above never reaches.

        ICE-6G_C VM5a stores ``sice`` in centimetres (max ~300 000 cm at LGM).
        300 000 exceeds the 5 000 m cut-off, so the loader must return 3 000 m.
        """
        nc = _make_ice6g_like_nc(
            tmp_path / "ice6g_cm.nc",
            ice_vars=("sftgif", "sice"),
            thickness_value=300_000.0,
        )
        assert self._ice_max(nc) == pytest.approx(3000.0), (
            "cm→m conversion did not run: a 300 000 cm (3 000 m) LGM ice sheet "
            "must not be reported as 300 000 m"
        )

    def test_metres_encoded_thickness_is_not_divided(self, tmp_path):
        """Guard the heuristic's other side: 4 900 m is thick, not centimetres."""
        nc = _make_ice6g_like_nc(
            tmp_path / "ice6g_thick_m.nc",
            ice_vars=("sice",),
            thickness_value=4900.0,
        )
        assert self._ice_max(nc) == pytest.approx(4900.0)


class TestIce6gRawDirResolution:
    """The real-data path must be resolvable off this box, not only on D:."""

    def test_no_hardcoded_drive_absolute_paths_in_this_module(self):
        """No Windows drive-absolute data path may survive in this file.

        The needles are assembled at runtime so that this assertion does not
        match its own source text.
        """
        src = Path(__file__).read_text(encoding="utf-8")
        drive = "D" + ":"
        for sep in ("/", "\\"):
            needle = drive + sep + "pasha"
            assert needle not in src, f"drive-absolute path {needle!r} in test_gia.py"

    def test_default_is_repo_relative(self, monkeypatch):
        """Without the env var the sibling ``Paleo-Europe_3D`` tree is used.

        The expected side is ``.resolve()``d because the actual one is:
        ``_resolve_ice6g_raw_dir`` resolves the root, and on CT 129 paleo-compute
        ``/opt/Paleo-Europe_3D`` is a symlink to ``/opt/paleo/Paleo-Europe_3D``, so
        the two are one directory under two spellings and the old assertion failed
        on the spelling alone.  Resolving both sides keeps the check that matters:
        a default pointing at a DIFFERENT directory still fails.

        The sibling name is spelled out here rather than taken from
        ``_DEFAULT_PALEO_DATA_ROOT``: reusing that name moves both sides of the
        assertion together and the test degenerates into a tautology.
        """
        monkeypatch.delenv("PALEO_DATA_ROOT", raising=False)
        assert _resolve_ice6g_raw_dir() == (
            (_REPO_ROOT.parents[1] / "Paleo-Europe_3D").resolve()
            / "data" / "raw" / "ICE6G"
        )

    def test_paleo_data_root_env_redirects_the_raw_dir(self, monkeypatch, tmp_path):
        """``PALEO_DATA_ROOT`` moves the whole tree (findus: /mnt/tank)."""
        monkeypatch.setenv("PALEO_DATA_ROOT", str(tmp_path / "tank"))
        assert _resolve_ice6g_raw_dir() == (
            (tmp_path / "tank").resolve() / "data" / "raw" / "ICE6G"
        )


@pytest.mark.skipif(
    not all(p.exists() for p in _ICE6G_RAW_EPOCHS),
    reason="real ICE-6G_C VM5a files not present",
)
class TestIce6gLoaderRealData:
    """Integration flavour against the on-disk ICE-6G_C VM5a distribution."""

    @staticmethod
    def _build_two_epoch_nc(dest: Path) -> Path:
        """Concatenate 16.5 ka + 0 ka into the multi-epoch layout the loader expects.

        The real distribution ships one file per epoch with no ``time`` dimension,
        so it is stitched here.  Decimated 2× in both axes to stay light; the
        global deep ocean is preserved because the sea-surface offset needs it.
        """
        epoch_t, epoch_0 = _ICE6G_RAW_EPOCHS
        with xr.open_dataset(epoch_t) as dt, xr.open_dataset(epoch_0) as d0:
            sl = {"lat": slice(None, None, 2), "lon": slice(None, None, 2)}
            topo = np.stack(
                [
                    d0["Topo"].isel(sl).values.astype(np.float32),
                    dt["Topo"].isel(sl).values.astype(np.float32),
                ]
            )
            orog = np.stack(
                [
                    d0["Orog"].isel(sl).values.astype(np.float32),
                    dt["Orog"].isel(sl).values.astype(np.float32),
                ]
            )
            gif = np.stack(
                [
                    d0["sftgif"].isel(sl).values.astype(np.float32),
                    dt["sftgif"].isel(sl).values.astype(np.float32),
                ]
            )
            lat = dt["lat"].isel(lat=slice(None, None, 2)).values
            lon = dt["lon"].isel(lon=slice(None, None, 2)).values

        ds = xr.Dataset(
            {
                "Topo": (("time", "lat", "lon"), topo),
                "Orog": (("time", "lat", "lon"), orog),
                "sftgif": (("time", "lat", "lon"), gif),
            },
            coords={
                "time": np.array([0.0, -16.5], dtype=np.float32),
                "lat": lat,
                "lon": lon,
            },
        )
        ds.to_netcdf(dest)
        ds.close()
        return dest

    def test_manych_corridor_rises_slightly_at_16_5_ka(self, tmp_path):
        """Measured on the real grid: Δz_solid ≈ +2.7 m at the sill cell.

        Rewritten 2026-07-30 — this test used to assert Δz_solid < 0 ("must be
        negative, measured −26.5 m").  That −26.5 m was the **bug**: the retired
        estimator referenced its datum to the deep-ocean median, and the
        far-field ocean floor stands ~+29 m higher at 16.5 ka because the water
        load is absent, so the estimator zeroed a real signal and pushed it onto
        the continents sign-flipped.  Referenced instead to ice-free tropical
        continental land, the Manych corridor shows a small *uplift*.

        Direction matters, and it is counter-intuitive: the bug placed the
        palaeo-sill at ≈ −3.5 m a.s.l., i.e. ~26 m LOWER than today, which made
        the Khvalynian spill spuriously trivial.  The corrected sill stands at
        ≈ +25.7 m — higher than the modern +23.00 m — so the fix makes the spill
        HARDER to achieve, not easier.  Anything that drags this number back
        towards −26 m has reintroduced the deep-ocean datum.
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        nc = self._build_two_epoch_nc(tmp_path / "ice6g_16.5ka_0ka.nc")
        loader = Ice6gLoader(nc)
        try:
            with pytest.warns(UserWarning):
                lats, lons, dz, _ice_h = loader.get_fields(
                    epoch_ka=16.5, bounds=MANYCH_BOUNDS
                )
        finally:
            loader.close()

        sill = float(
            xr.DataArray(dz, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
            .sel(lat=MANYCH_LAT, lon=MANYCH_LON, method="nearest")
        )
        assert sill > 0.0, (
            f"Manych Δz_solid at 16.5 ka must be a small POSITIVE uplift "
            f"(measured +2.7 m at the sill cell); got {sill:+.2f} m. A negative "
            "value near −26 m means the deep-ocean datum is back."
        )
        assert 0.5 < sill < 8.0, (
            f"Manych Δz_solid {sill:+.2f} m out of range — expected a few metres "
            "of uplift; ±1–2 m of near-field systematic is accepted"
        )
        # The number that actually gates the animation: the palaeo-sill must sit
        # ABOVE the modern +23.00 m, so overtopping it requires a real highstand.
        palaeo_sill = 23.00 + sill
        assert palaeo_sill > 23.00, (
            f"palaeo-sill {palaeo_sill:.2f} m must exceed the modern +23.00 m"
        )
        assert 24.0 < palaeo_sill < 30.0, (
            f"palaeo-sill {palaeo_sill:.2f} m a.s.l. outside the corrected "
            "+25.7…+26.9 m band (±1–2 m near-field systematic)"
        )


# ==========================================================================
# Ice7gLoader (uses real fixture .nc with time_ka)
# ==========================================================================


class TestIce7gLoader:
    def test_get_thickness_fixture(self, ice7g_path):
        """get_thickness must return a 2-D (lat, lon) field for a given epoch."""
        from paleoeurope.gia.ice7g_loader import Ice7gLoader

        if not ice7g_path.exists():
            pytest.skip("ICE-7G fixture file not found")

        loader = Ice7gLoader(ice7g_path)
        lats, lons, ice_h = loader.get_thickness(
            epoch_ka=12.0,
            bounds=(4.0, 59.0, 10.0, 65.0),  # Norway — inside fixture coverage
        )
        assert ice_h.ndim == 2
        assert ice_h.shape == (len(lats), len(lons))
        assert np.isfinite(ice_h).all()
        assert (ice_h >= 0.0).all()


# ==========================================================================
# build_correction_matrix edge cases
# ==========================================================================


class TestBuildCorrectionMatrix:
    def test_cubic_falls_back_on_tiny_grid(self):
        """Cubic interpolation should not crash on a 3×3 source grid."""
        lats = np.array([61.5, 62.5, 63.5], dtype=float)
        lons = np.array([5.5, 6.5, 7.5], dtype=float)
        field = np.zeros((3, 3), dtype=np.float32)
        arr, _tf = build_correction_matrix(
            field,
            lats,
            lons,
            west=6.0,
            south=62.0,
            east=7.0,
            north=63.0,
            res_deg=1.0 / 60.0,
            interp_method="cubic",
        )
        assert arr.ndim == 2
        assert arr.shape[0] > 0 and arr.shape[1] > 0
