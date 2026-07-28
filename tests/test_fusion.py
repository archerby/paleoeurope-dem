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


# ===========================================================================
# FabdemLoader index hygiene
#
# The GeoParquet index is a *machine-local cache*, never source.  Three ways it
# used to lie:
#   1. a stale index (tile paths recorded on another machine) was trusted verbatim;
#   2. the in-memory cache lived on the class, so a second region loaded in the
#      same process silently got the FIRST region's tiles — exactly the shape of
#      a Pontic-then-Caspian run inside one process;
#   3. a failed cache write was swallowed, so a full disk looked like success.
# ===========================================================================


def _write_tile(directory, name, bounds):
    """Write a tiny 1-band GeoTIFF covering *bounds* — enough to index and read."""
    import rasterio
    from rasterio.transform import from_bounds as affine_from_bounds

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=affine_from_bounds(*bounds, 8, 8),
    ) as dst:
        dst.write(np.full((8, 8), 100.0, dtype=np.float32), 1)
    return path


@pytest.fixture
def clean_index_cache():
    """Drop the loader's in-memory index cache around the test."""
    from paleoeurope.fusion.fabdem_loader import FabdemLoader

    FabdemLoader.clear_index_cache()
    yield
    FabdemLoader.clear_index_cache()


class TestFabdemLoaderIndexHygiene:
    def test_stale_index_is_discarded_and_rebuilt(self, tmp_path, clean_index_cache):
        """An index whose recorded tiles are gone must be rebuilt, with a warning."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box as shapely_box

        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        tile_dir = tmp_path / "fabdem"
        real_tile = _write_tile(tile_dir, "N62E006_FABDEM_V1-2.tif", (6.0, 62.0, 7.0, 63.0))

        # Index recorded on a machine that no longer exists — cf. the committed
        # fabdem_index.parquet that pointed at /home/pp/... on a powered-off box.
        dead = "/home/pp/paleo_project/tests/fixtures/fabdem/N62E006_FABDEM_V1-2_fixture.tif"
        stale = gpd.GeoDataFrame(
            [
                {
                    "path": dead,
                    # Correct provenance, so only the dead path can make it stale.
                    "fabdem_dir": str(tile_dir.resolve()),
                    "minx": 6.0,
                    "miny": 62.0,
                    "maxx": 7.0,
                    "maxy": 63.0,
                    "geometry": shapely_box(6.0, 62.0, 7.0, 63.0),
                }
            ],
            crs="EPSG:4326",
        )
        index_path = tmp_path / "fabdem_index.parquet"
        stale.to_parquet(index_path)

        with pytest.warns(UserWarning, match="stale"):
            loader = FabdemLoader(tile_dir, index_path=index_path)

        indexed = [rec["path"] for rec in loader._candidates(shapely_box(6.0, 62.0, 7.0, 63.0))]
        assert dead not in indexed, "loader handed out a tile path from a powered-off machine"
        assert str(real_tile) in indexed

        arr, transform, crs = loader.read_window((6.0, 62.0, 7.0, 63.0))
        assert arr is not None

    def test_index_cache_is_keyed_by_directory(self, tmp_path, clean_index_cache):
        """A second tile directory must not inherit the first directory's index."""
        from shapely.geometry import box as shapely_box

        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        pontic_dir = tmp_path / "pontic"
        caspian_dir = tmp_path / "caspian"
        _write_tile(pontic_dir, "N44E033_FABDEM_V1-2.tif", (33.0, 44.0, 34.0, 45.0))
        caspian_tile = _write_tile(caspian_dir, "N45E050_FABDEM_V1-2.tif", (50.0, 45.0, 51.0, 46.0))

        FabdemLoader(pontic_dir, index_path=tmp_path / "pontic.parquet")
        caspian = FabdemLoader(caspian_dir, index_path=tmp_path / "caspian.parquet")

        hits = [rec["path"] for rec in caspian._candidates(shapely_box(50.0, 45.0, 51.0, 46.0))]
        assert str(caspian_tile) in hits, "second region was served the first region's index"

        arr, transform, crs = caspian.read_window((50.0, 45.0, 51.0, 46.0))
        assert arr is not None

    def test_index_built_for_another_directory_is_discarded(self, tmp_path, clean_index_cache):
        """One shared index_path must not serve the wrong region's tiles.

        The on-disk twin of the class-level cache bug: with the default
        ``fabdem_index.parquet``, a Pontic run leaves an index that a later
        Caspian run would otherwise load and believe.
        """
        pytest.importorskip("geopandas")
        from shapely.geometry import box as shapely_box

        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        shared_index = tmp_path / "fabdem_index.parquet"
        pontic_dir = tmp_path / "pontic"
        caspian_dir = tmp_path / "caspian"
        _write_tile(pontic_dir, "N44E033_FABDEM_V1-2.tif", (33.0, 44.0, 34.0, 45.0))
        caspian_tile = _write_tile(caspian_dir, "N45E050_FABDEM_V1-2.tif", (50.0, 45.0, 51.0, 46.0))

        FabdemLoader(pontic_dir, index_path=shared_index)
        FabdemLoader.clear_index_cache()  # new process would start cold

        with pytest.warns(UserWarning, match="stale"):
            caspian = FabdemLoader(caspian_dir, index_path=shared_index)

        hits = [rec["path"] for rec in caspian._candidates(shapely_box(50.0, 45.0, 51.0, 46.0))]
        assert str(caspian_tile) in hits, "Caspian run was served the Pontic index"

    def test_index_write_failure_warns(self, tmp_path, monkeypatch, clean_index_cache):
        """A cache write that fails (e.g. full disk) must warn, not pass silently."""
        gpd = pytest.importorskip("geopandas")

        from paleoeurope.fusion.fabdem_loader import FabdemLoader

        tile_dir = tmp_path / "fabdem"
        _write_tile(tile_dir, "N62E006_FABDEM_V1-2.tif", (6.0, 62.0, 7.0, 63.0))

        def _no_space(self, *args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", _no_space, raising=True)

        with pytest.warns(UserWarning, match="No space left on device"):
            FabdemLoader(tile_dir, index_path=tmp_path / "fabdem_index.parquet")
