"""
paleoeurope.fusion.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orchestrate the full fusion for a single 1°×1° tile.

The pipeline steps:
 1. Load FABDEM window (bare-earth land).
 2. Load GEBCO window (bathymetry), reprojected to match FABDEM.
 3. Optionally apply an EGM2008-based geoid/undulation offset to GEBCO.
 4. Alpha-blend the two layers.
 5. Write the output Cloud-Optimised GeoTIFF.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS

from paleoeurope.fusion.blender import RasterBlender
from paleoeurope.fusion.datum_corrector import DatumCorrector
from paleoeurope.fusion.fabdem_loader import FabdemLoader
from paleoeurope.fusion.gebco_loader import GebcoLoader

logger = logging.getLogger(__name__)

# Resolution of FABDEM: 3 arc-seconds ≈ 1/1200 degree
_FABDEM_RES_DEG = 1.0 / 1200.0
_PIXELS_PER_DEGREE = 3600  # standard 1 arc-sec grid


def run_fusion_tile(
    tile_id: str,
    fabdem_dir: str | Path,
    gebco_path: str | Path,
    output_path: str | Path,
    egm2008_grid: str | Path | None = None,
    apply_geoid_offset: bool | None = None,
    blend_distance_px: int = 50,
    tid_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run the complete fusion pipeline for one tile.

    Parameters
    ----------
    tile_id : str
        SRTM-style tile ID, e.g. ``"N51E000"``.
    fabdem_dir : str or Path
        Directory containing FABDEM GeoTIFF tiles.
    gebco_path : str or Path
        Path to the GEBCO 2024 GeoTIFF or NetCDF.
    output_path : str or Path
        Destination path for the fused GeoTIFF.
    egm2008_grid : str or Path, optional
        Path to an EGM2008 undulation grid.
    apply_geoid_offset : bool, optional
        Whether to apply the EGM2008-based undulation as a vertical offset to
        GEBCO. Defaults to ``False`` unless explicitly enabled via the
        environment variable ``PALEO_APPLY_GEOID_CORRECTION=1``.
    blend_distance_px : int, optional
        Half-width of the coastal blending zone in pixels. Default 50.
    tid_path : str or Path, optional
        Path to the GEBCO TID grid for land masking.
    overwrite : bool, optional
        Skip processing if the output file already exists.  Default ``False``.

    Returns
    -------
    Path
        Path to the written output GeoTIFF.

    Raises
    ------
    ValueError
        If *tile_id* cannot be parsed.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info("Skipping %s — already exists at %s", tile_id, output_path)
        return output_path

    bounds = _tile_id_to_bounds(tile_id)
    logger.info("Fusion: %s  bounds=%s", tile_id, bounds)

    # 1 — Load FABDEM
    fab_loader = FabdemLoader(fabdem_dir)
    fab_arr, fab_transform, fab_crs = fab_loader.read_window(bounds)

    if fab_arr is None:
        logger.warning("No FABDEM coverage for %s — skipping.", tile_id)
        return output_path  # empty / skipped

    height, width = fab_arr.shape
    fab_crs_obj = fab_crs or CRS.from_epsg(4326)

    # Wrap as DataArray with proper coords
    fab_da = _arr_to_da(fab_arr, bounds, height, width)

    # 2 — Load GEBCO
    geb_loader = GebcoLoader(gebco_path, tid_path=tid_path)
    geb_arr = geb_loader.read_window(
        bounds=bounds,
        target_shape=(height, width),
        target_transform=fab_transform,
        target_crs=fab_crs_obj,
    )
    geb_da = _arr_to_da(geb_arr, bounds, height, width)

    # 3 — Optional EGM2008-based geoid/undulation offset
    if apply_geoid_offset is None:
        apply_geoid_offset = os.getenv("PALEO_APPLY_GEOID_CORRECTION", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    if apply_geoid_offset:
        corrector = DatumCorrector(egm2008_grid)
        geb_da = corrector.align(geb_da)
    else:
        logger.info("Geoid offset disabled (PALEO_APPLY_GEOID_CORRECTION=0)")

    # 4 — Alpha blend
    blender = RasterBlender(blend_distance_px=blend_distance_px)
    merged_da = blender.blend(fab_da, geb_da)

    # 5 — Write GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_arr = np.asarray(merged_da, dtype=np.float32)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=fab_crs_obj,
        transform=fab_transform,
        nodata=-9999.0,
        compress="LZW",
    ) as dst:
        out = np.where(np.isnan(merged_arr), -9999.0, merged_arr).astype(np.float32)
        dst.write(out, 1)

    logger.info("Written: %s", output_path)
    return output_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _tile_id_to_bounds(tile_id: str) -> tuple[float, float, float, float]:
    """Parse a tile ID like ``N51E000`` → ``(0.0, 51.0, 1.0, 52.0)``."""
    import re

    m = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile_id.upper())
    if not m:
        raise ValueError(
            f"Cannot parse tile_id '{tile_id}'.  Expected format: N51E000."
        )
    lat_dir, lat_v, lon_dir, lon_v = m.groups()
    lat = int(lat_v) * (1 if lat_dir == "N" else -1)
    lon = int(lon_v) * (1 if lon_dir == "E" else -1)
    return (float(lon), float(lat), float(lon + 1), float(lat + 1))


def _arr_to_da(
    arr: np.ndarray,
    bounds: tuple[float, float, float, float],
    height: int,
    width: int,
) -> xr.DataArray:
    """Wrap a numpy array in an xarray DataArray with geographic coords."""
    minx, miny, maxx, maxy = bounds
    lons = np.linspace(minx, maxx, width, endpoint=False) + (maxx - minx) / (2 * width)
    lats = np.linspace(maxy, miny, height, endpoint=False) - (maxy - miny) / (2 * height)
    return xr.DataArray(arr, dims=["y", "x"], coords={"y": lats, "x": lons})
