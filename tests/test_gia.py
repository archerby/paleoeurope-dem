"""
tests/test_gia.py

Unit tests for paleoeurope.gia: apply_gia_delta, IceEnvelope, Ice6gLoader.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoeurope.gia.correction_matrix import build_correction_matrix
from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.envelope import IceEnvelope
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

    def test_get_fields_fixture(self, ice6g_path):
        """get_fields should return arrays with correct shapes from fixture data."""
        from paleoeurope.gia.ice6g_loader import Ice6gLoader

        if not ice6g_path.exists():
            pytest.skip("ICE-6G fixture file not found")

        loader = Ice6gLoader(ice6g_path)
        lats, lons, dz, ice_h = loader.get_fields(
            epoch_ka=21,
            bounds=(4.0, 59.0, 10.0, 65.0),   # Norway — inside fixture coverage
        )
        assert dz.ndim == 2
        assert ice_h.ndim == 2
        assert dz.shape == ice_h.shape
        assert len(lats) == dz.shape[0]
        assert len(lons) == dz.shape[1]
        # Real GIA signal: Norway at 21 ka has significant bedrock depression
        # (stgr = Topo[21ka] − Topo[0ka] from real ICE-6G_C VM5a data).
        assert np.abs(dz).max() > 0.0, (
            "GIA delta (stgr) must be non-zero for Norway at 21 ka; "
            "check that ICE6G_fixture.nc was generated from real data via "
            "prepare_test_fixtures.py and that ice6g_loader reads 'stgr'."
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
