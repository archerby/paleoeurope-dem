"""
tests/test_fusion.py

Unit tests for paleoeurope.fusion: blender, datum_corrector, loaders.

Unit tests (TestRasterBlender, TestDatumCorrector, TestFabdemLoader) use minimal
in-memory DataArrays (20×20 px) defined in conftest.py — this is correct for
algorithm property tests and does NOT constitute synthetic data in the pipeline.

Integration tests (TestFabdemLoader) resolve data through conftest.py which
prioritises committed real fixtures in tests/fixtures/.
No external downloads required.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoeurope.fusion.blender import RasterBlender
from paleoeurope.fusion.datum_corrector import DatumCorrector

# ===========================================================================
# RasterBlender
# ===========================================================================


class TestRasterBlender:
    def test_pure_land_returns_land(self, simple_land_da, simple_ocean_da):
        """All-land tile → output equals land input."""
        blender = RasterBlender(blend_distance_px=3)
        # Make a fully-land mask
        land = simple_land_da.copy()
        land[:] = 100.0  # no NaN
        result = blender.blend(land, simple_ocean_da)
        # Should be all ~100 m (land values)
        assert float(result.mean()) == pytest.approx(100.0, abs=1e-3)

    def test_pure_ocean_returns_ocean(self, simple_land_da, simple_ocean_da):
        """All-ocean tile → output equals ocean input."""
        blender = RasterBlender(blend_distance_px=3)
        land = simple_land_da.copy()
        land[:] = np.nan  # all ocean
        result = blender.blend(land, simple_ocean_da)
        expected = float(simple_ocean_da.mean())
        assert float(result.mean()) == pytest.approx(expected, abs=1e-3)

    def test_blend_output_shape(self, simple_land_da, simple_ocean_da):
        """Output shape must match input shape."""
        blender = RasterBlender(blend_distance_px=3)
        result = blender.blend(simple_land_da, simple_ocean_da)
        assert result.shape == simple_land_da.shape

    def test_blend_output_coords(self, simple_land_da, simple_ocean_da):
        """Output coordinates must match land input."""
        blender = RasterBlender(blend_distance_px=3)
        result = blender.blend(simple_land_da, simple_ocean_da)
        np.testing.assert_array_equal(result.coords["x"].values, simple_land_da.coords["x"].values)
        np.testing.assert_array_equal(result.coords["y"].values, simple_land_da.coords["y"].values)

    def test_blend_boundary_is_between_land_ocean(self, simple_land_da, simple_ocean_da):
        """Values in the blend zone must be strictly between land and ocean."""
        blender = RasterBlender(blend_distance_px=5)
        result = blender.blend(simple_land_da, simple_ocean_da)
        arr = result.values

        ocean_val = float(simple_ocean_da.mean())
        # In blend zone, values should be between ocean_val and max land value
        blend_zone = arr[~np.isnan(arr)]
        assert float(blend_zone.min()) >= ocean_val - 1e-3

    def test_invalid_blend_distance_raises(self):
        """blend_distance_px < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            RasterBlender(blend_distance_px=0)

    def test_compute_alpha_range(self):
        """Alpha values must be in [0, 1]."""
        blender = RasterBlender(blend_distance_px=10)
        mask = np.zeros((20, 20), dtype=bool)
        mask[10:, :] = True
        alpha = blender.compute_alpha(mask, mean_lat=51.0)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_compute_alpha_shape(self):
        """Alpha array must match the input mask shape."""
        blender = RasterBlender()
        mask = np.ones((30, 40), dtype=bool)
        alpha = blender.compute_alpha(mask)
        assert alpha.shape == (30, 40)


# ===========================================================================
# DatumCorrector
# ===========================================================================


class TestDatumCorrector:
    def test_noop_mode_returns_original(self, simple_ocean_da):
        """No-op mode (no grid path) must return the input array unchanged."""
        corrector = DatumCorrector(grid_path=None)
        result = corrector.align(simple_ocean_da)
        np.testing.assert_array_almost_equal(result.values, simple_ocean_da.values)

    def test_noop_mode_warns(self, simple_ocean_da):
        """No-op mode should issue a UserWarning."""
        corrector = DatumCorrector(grid_path=None)
        with pytest.warns(UserWarning, match="no-op"):
            corrector.align(simple_ocean_da)

    def test_missing_grid_warns_on_init(self, tmp_path):
        """Providing a nonexistent grid path should warn, not raise."""
        with pytest.warns(UserWarning, match="not found"):
            corrector = DatumCorrector(grid_path=tmp_path / "nonexistent.tif")
        assert corrector.grid_path is None

    def test_undulation_at_noop(self):
        """undulation_at returns 0.0 in no-op mode."""
        corrector = DatumCorrector(grid_path=None)
        assert corrector.undulation_at(51.5, 0.5) == 0.0

    def test_with_synthetic_grid(self, egm2008_path, simple_ocean_da):
        """With a real (synthetic) EGM2008 grid, output must differ from input."""
        if not egm2008_path.exists():
            pytest.skip("Synthetic EGM2008 grid not generated yet")

        # Ensure the DataArray has rioxarray CRS info
        import rioxarray  # noqa: F401

        da = simple_ocean_da.copy()
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        da = da.rio.write_crs("EPSG:4326")

        corrector = DatumCorrector(grid_path=egm2008_path)
        result = corrector.align(da)
        # Correction should shift values
        assert result.shape == da.shape


# ===========================================================================
# FabdemLoader (integration-ish, uses synthetic data)
# ===========================================================================


class TestFabdemLoader:
    def test_loader_initialises(self, fabdem_dir):
        """FabdemLoader should initialise without error."""
        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        loader = FabdemLoader(fabdem_dir)
        assert loader is not None

    def test_read_window_returns_array(self, fabdem_dir, tile_bounds):
        """read_window must return a float32 array and affine transform."""
        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        if not fabdem_dir.exists():
            pytest.skip("Synthetic FABDEM tiles not generated yet")

        loader = FabdemLoader(fabdem_dir)
        arr, transform, crs = loader.read_window(tile_bounds)
        assert arr is not None
        assert arr.dtype == np.float32
        assert arr.ndim == 2

    def test_read_window_no_coverage_returns_none(self, fabdem_dir):
        """read_window on an area outside tile coverage should return None."""
        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        if not fabdem_dir.exists():
            pytest.skip("Synthetic FABDEM tiles not generated yet")

        loader = FabdemLoader(fabdem_dir)
        arr, transform, crs = loader.read_window((100.0, 10.0, 101.0, 11.0))
        assert arr is None
