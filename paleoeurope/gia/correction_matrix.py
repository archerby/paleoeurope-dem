"""
paleoeurope.gia.correction_matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-level helpers for building and reading GIA / ice-thickness
**correction matrices** — uniform-grid arrays that store either the
isostatic-rebound offset Δz_GIA or the ice-thickness H_ice for a single epoch.

The matrix workflow mirrors production SYNC mode exactly:

1. ``build_correction_matrix()`` — interpolates a coarse ICE-6G model field
   (0.5° or 1° grid) onto a fine uniform output grid (default 10 arcsec =
   1/360°, matching the production DMK resolution).  Optionally applies a
   Gaussian smoothing kernel.

2. ``read_correction_matrix()`` — window-reads a tile-sized slice of an
   in-memory correction matrix using ``rasterio`` bilinear resampling,
   identical to the path taken when reading from a GeoTIFF on disk.

Naming
------
Throughout this module the term *correction_matrix* (abbreviated CM) replaces
the legacy term *DMK* used in earlier development notebooks.

References
----------
- Lambeck et al. (2014) ICE-6G_C VM5a model.
- Peltier et al. (2015) doi:10.1002/2014JB011176.
- Roy & Peltier (2018) doi:10.1016/j.quascirev.2017.11.016.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.transform
import rasterio.windows
from rasterio.io import MemoryFile
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

__all__ = ["build_correction_matrix", "read_correction_matrix", "write_correction_matrix"]

logger = logging.getLogger(__name__)

# Production default: 10 arcsec ≈ 1/360°  (~300 m @ 60°N).
# Notebooks may override to 1/60 (1 arcmin ≈ 1.8 km) for quick single-tile
# demos where GIA signal is long-wavelength.
CM_RESOLUTION_DEG: float = 1.0 / 360.0


def write_correction_matrix(
    matrix_arr: np.ndarray,
    matrix_transform: "rasterio.transform.Affine",
    output_path: str,
    *,
    compress: str = "lzw",
) -> None:
    """Write a correction matrix to a GeoTIFF.

    This mirrors production's on-disk DMK workflow (Stage 0 in geo_07):
    build a global matrix once per epoch, then window-read slices per tile.
    """
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": int(matrix_arr.shape[1]),
        "height": int(matrix_arr.shape[0]),
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": matrix_transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0.0,
    }

    import os
    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(matrix_arr.astype(np.float32), 1)
    try:
        os.chmod(out, 0o666)
    except Exception:
        pass


def build_correction_matrix(
    field_2d: np.ndarray,
    field_lats: np.ndarray,
    field_lons: np.ndarray,
    west: float,
    south: float,
    east: float,
    north: float,
    res_deg: float = CM_RESOLUTION_DEG,
    gaussian_sigma: Optional[float] = None,
    interp_method: str = "linear",
) -> tuple[np.ndarray, "rasterio.transform.Affine"]:
    """Interpolate a coarse model field to a uniform output grid.

    Mirrors ``GIAModel.build_correction_matrix()`` and
    ``IceThicknessModel.build_ice_matrix()`` from ``scripts/geo/deformation.py``.

    The routine:
    1. Ensures *field_lats* is ascending (silently flips if necessary).
    2. Builds a regular output grid covering ``[west, east] × [south, north]``
       at resolution *res_deg*.
    3. Interpolates *field_2d* onto the output grid using *interp_method*
       (``"linear"`` for GIA bedrock, ``"cubic"`` for ICE thickness to avoid
       the linear-ramp artefact at ice-sheet margins).
    4. Optionally applies a Gaussian smoothing kernel of width *gaussian_sigma*
       (in grid cells, pixel-space — as used by IceThicknessModel).

    Parameters
    ----------
    field_2d : np.ndarray, shape (M, N)
        2-D scalar field (e.g. Δz bedrock or H_ice) on the model grid.
    field_lats : np.ndarray, shape (M,)
        Latitude grid of the model (degrees, monotonic).
    field_lons : np.ndarray, shape (N,)
        Longitude grid of the model (degrees, monotonic).
    west, south, east, north : float
        Output bounding box (degrees).  Typically padded ±0.5° around
        the tile to avoid boundary artefacts.
    res_deg : float, optional
        Output grid resolution in degrees.  Default ``1/360`` (10 arcsec).
    interp_method : str, optional
        SciPy ``RegularGridInterpolator`` method.  Use ``"linear"`` (default)
        for GIA bedrock delta and ``"cubic"`` for ICE thickness to produce a
        physically correct parabolic ice-dome profile at the margin.
    gaussian_sigma : float or None, optional
        If given, a Gaussian filter of this width (in output *pixels*) is
        applied after interpolation.  Pass e.g. ``1.5`` for H_ice smoothing.
        Values below zero are clipped to zero after smoothing.

    Returns
    -------
    matrix_arr : np.ndarray, dtype float32, shape (n_rows, n_cols)
        The interpolated correction matrix.
    matrix_transform : rasterio.transform.Affine
        Affine geotransform so the array can be window-read by
        :func:`read_correction_matrix`.

    Examples
    --------
    >>> import numpy as np
    >>> lats = np.array([60.0, 60.5, 61.0])
    >>> lons = np.array([6.0, 6.5, 7.0])
    >>> field = np.zeros((3, 3), dtype=np.float32)
    >>> arr, tf = build_correction_matrix(field, lats, lons, 5.5, 61.5, 7.5, 63.5)
    >>> arr.shape
    (720, 720)
    """
    # Ensure ascending latitude order
    if field_lats[0] > field_lats[-1]:
        field_lats = field_lats[::-1]
        field_2d = field_2d[::-1, :]

    # SciPy cubic interpolation requires at least 4 points per dimension.
    # The open-demo ICE-7G fixture is intentionally tiny (3×3); fall back to
    # linear in that case while keeping production behaviour unchanged.
    effective_method = interp_method
    if interp_method == "cubic" and (len(field_lats) < 4 or len(field_lons) < 4):
        logger.info(
            "build_correction_matrix: input grid too small for cubic (lats=%d lons=%d); "
            "falling back to linear",
            len(field_lats),
            len(field_lons),
        )
        effective_method = "linear"

    n_rows = max(int(round((north - south) / res_deg)), 1)
    n_cols = max(int(round((east - west) / res_deg)), 1)

    out_lats = np.linspace(north - res_deg / 2.0, south + res_deg / 2.0, n_rows)
    out_lons = np.linspace(west + res_deg / 2.0, east - res_deg / 2.0, n_cols)

    lon_g, lat_g = np.meshgrid(out_lons, out_lats)
    pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])

    interp = RegularGridInterpolator(
        (field_lats, field_lons),
        field_2d.astype(np.float64),
        method=effective_method,
        bounds_error=False,
        fill_value=0.0,
    )
    matrix_arr = interp(pts).reshape(n_rows, n_cols).astype(np.float32)

    del lon_g, lat_g, pts, interp

    if gaussian_sigma is not None and gaussian_sigma > 0:
        matrix_arr = gaussian_filter(matrix_arr.astype(np.float64),
                                     sigma=gaussian_sigma).astype(np.float32)
        np.clip(matrix_arr, 0.0, None, out=matrix_arr)

    matrix_transform = rasterio.transform.from_bounds(
        west, south, east, north, n_cols, n_rows
    )

    logger.debug(
        "build_correction_matrix: shape=%s  res=%.6f°  range=[%.2f, %.2f]",
        matrix_arr.shape,
        res_deg,
        float(matrix_arr.min()),
        float(matrix_arr.max()),
    )

    return matrix_arr, matrix_transform


def read_correction_matrix(
    matrix_arr: np.ndarray,
    matrix_transform: "rasterio.transform.Affine",
    bounds: "rasterio.coords.BoundingBox",
    h: int,
    w: int,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
) -> np.ndarray:
    """Window-read a tile-sized slice of an in-memory correction matrix.

    Uses a ``rasterio.MemoryFile`` so the resampling path is
    identical to reading from a GeoTIFF on disk — matching production
    SYNC-mode ``_read_dmk()`` exactly.

    Pass ``resampling=rasterio.enums.Resampling.cubic`` when reading ICE
    thickness to eliminate the linear-ramp artefact caused by low-resolution
    (1°) ICE-7G grids being bilinearly stretched.

    Parameters
    ----------
    matrix_arr : np.ndarray, shape (n_rows, n_cols)
        The correction matrix produced by :func:`build_correction_matrix`.
    matrix_transform : rasterio.transform.Affine
        The corresponding affine geotransform.
    bounds : rasterio.coords.BoundingBox
        Spatial extent of the target tile.
    h, w : int
        Height and width of the target tile in pixels.

    resampling : rasterio.enums.Resampling, optional
        Rasterio resampling algorithm.  Default ``bilinear`` (for GIA).
        Use ``cubic`` for ice-thickness matrices.

    Returns
    -------
    np.ndarray, dtype float32, shape (h, w)
        A resampled slice of the correction matrix, NaN-free
        (NaN values are replaced by 0.0).

    Examples
    --------
    >>> import rasterio
    >>> arr, tf = build_correction_matrix(field, lats, lons, 5.5, 61.5, 7.5, 63.5)
    >>> with rasterio.open('N62E006_fusion.tif') as src:
    ...     bounds = src.bounds
    ...     h, w   = src.height, src.width
    >>> delta = read_correction_matrix(arr, tf, bounds, h, w)
    >>> delta.shape
    (3600, 3600)
    """
    n_rows, n_cols = matrix_arr.shape
    mem_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": n_cols,
        "height": n_rows,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": matrix_transform,
    }
    with MemoryFile() as mf:
        with mf.open(**mem_profile) as dst:
            dst.write(matrix_arr[np.newaxis, :, :])
        with mf.open() as src:
            win = rasterio.windows.from_bounds(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                transform=src.transform,
            )
            result = src.read(
                1,
                window=win,
                out_shape=(h, w),
                resampling=resampling,
                masked=False,
            ).astype(np.float32)

    np.nan_to_num(result, nan=0.0, copy=False)
    return result
