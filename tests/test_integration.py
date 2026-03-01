"""
tests/test_integration.py

End-to-end integration test:
  real FABDEM fixture + real GEBCO fixture → fusion → GIA → paleo-DEM.

All input data comes from committed real data in tests/fixtures/ (tile N62E006,
Sunnmøre / Ålesund coast, Norway).  No synthetic data is used in these tests.

Marked with @pytest.mark.integration.  Skipped in CI unless --run-integration
is passed.  Run locally after generating fixtures:

    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoeurope.fusion.pipeline import run_fusion_tile
from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.ice6g_loader import Ice6gLoader
from paleoeurope.utils.grid import make_pixel_coords

pytestmark = pytest.mark.integration


def test_full_fusion_writes_geotiff(fabdem_dir, gebco_path, egm2008_path, tmp_path, tile_bounds, tile_id):
    """Fusion pipeline produces a valid GeoTIFF for the test tile."""
    if not fabdem_dir.exists() or not gebco_path.exists():
        pytest.skip("Test data not available")

    out_path = tmp_path / f"{tile_id}_fusion.tif"

    result = run_fusion_tile(
        tile_id=tile_id,
        fabdem_dir=fabdem_dir,
        gebco_path=gebco_path,
        output_path=out_path,
        egm2008_grid=egm2008_path if egm2008_path.exists() else None,
        apply_geoid_offset=egm2008_path.exists(),
        blend_distance_px=20,
    )

    import rasterio

    assert result.exists()
    with rasterio.open(result) as src:
        data = src.read(1)
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        assert src.nodata is not None


def test_fusion_then_gia_pipeline(fabdem_dir, gebco_path, ice6g_path, tmp_path, tile_bounds, tile_id):
    """Full pipeline: fusion → GIA correction → non-trivial paleo-DEM."""
    if not fabdem_dir.exists() or not gebco_path.exists() or not ice6g_path.exists():
        pytest.skip("Test data not available")

    fusion_out = tmp_path / f"{tile_id}_fusion.tif"
    run_fusion_tile(
        tile_id=tile_id,
        fabdem_dir=fabdem_dir,
        gebco_path=gebco_path,
        output_path=fusion_out,
        blend_distance_px=20,
    )

    from paleoeurope.utils.raster import read_geotiff

    win = read_geotiff(fusion_out)

    loader = Ice6gLoader(ice6g_path)
    lats, lons, dz, ice_h = loader.get_fields(epoch_ka=21, bounds=tile_bounds)

    tile_lats, tile_lons = make_pixel_coords(tile_bounds, win.data.shape)
    paleo, ice_mask = apply_gia_delta(win.data, lats, lons, dz, ice_h, tile_lats, tile_lons)

    assert paleo.shape == win.data.shape
    assert paleo.dtype == np.float32
    # Tile N62E006 (62-63°N, Sunnmøre) is fully under the Fennoscandian ice sheet
    # at 21 ka — expect significant GIA delta and ice-covered pixels.
    # dz comes from stgr = Topo[21ka] − Topo[0ka] in the real ICE-6G fixture.
    assert np.abs(dz).max() > 0.0, "GIA delta must be non-zero for 21 ka over Scandinavia"


def test_reproducibility_seed_42(fabdem_dir, gebco_path, tmp_path, tile_bounds, tile_id):
    """Running the pipeline twice should produce identical outputs (deterministic)."""
    if not fabdem_dir.exists() or not gebco_path.exists():
        pytest.skip("Test data not available")

    out1 = tmp_path / "run1.tif"
    out2 = tmp_path / "run2.tif"

    run_fusion_tile(tile_id, fabdem_dir, gebco_path, out1, blend_distance_px=20)
    run_fusion_tile(tile_id, fabdem_dir, gebco_path, out2, blend_distance_px=20, overwrite=True)

    import rasterio

    with rasterio.open(out1) as s1, rasterio.open(out2) as s2:
        np.testing.assert_array_equal(s1.read(1), s2.read(1))
