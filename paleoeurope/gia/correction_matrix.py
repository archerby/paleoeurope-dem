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

Unknown values
--------------
NaN in a correction matrix means **unknown**, and this module never converts it
into a number.  A Δz_GIA of ``0.0`` is a legitimate measurement — the far-field
tropical reference set sits there by construction of the datum — so a zero-fill
is not a neutral placeholder, it is a false measurement, and it lands on exactly
the formerly glaciated ground where the isostatic signal is largest.  The NaN
originates in :func:`paleoeurope.gia.ice6g_topo.solid_displacement`, which refuses
to report a solid-surface displacement where ICE-6G ``Topo`` is the ice surface.

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

__all__ = [
    "build_correction_matrix",
    "read_correction_matrix",
    "write_correction_matrix",
    "CorrectionMatrixHasUnknownCellsError",
]

logger = logging.getLogger(__name__)


class CorrectionMatrixHasUnknownCellsError(ValueError):
    """A caller asked for a NaN-free correction matrix and the matrix has NaN.

    Raised by :func:`read_correction_matrix` when ``preserve_nan=False``.  The
    NaN is not a nodata artefact: :func:`paleoeurope.gia.ice6g_topo.solid_displacement`
    emits it over cells where ICE-6G ``Topo`` is the **ice surface**, i.e. where
    these files carry no solid-surface elevation at all.  Zero-filling it would
    record 0 m of isostatic correction — *"nothing happened here"* — over exactly
    the formerly glaciated ground where the isostatic signal is largest.  So the
    opt-out is a refusal, not a fill.
    """

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
    nodata: float = float("nan"),
) -> None:
    """Write a correction matrix to a GeoTIFF.

    This mirrors production's on-disk DMK workflow (Stage 0 in geo_07):
    build a global matrix once per epoch, then window-read slices per tile.

    Parameters
    ----------
    nodata : float, optional
        Nodata value declared in the GeoTIFF header.  Default ``nan``.

        It used to be ``0.0``, which conflated the two things this module now
        keeps apart: for a Δz_GIA matrix ``0.0`` is a perfectly ordinary
        *measured* value (far-field tropical land sits there by construction of
        the datum), so declaring it nodata told every masked reader to throw away
        the cells where the answer is "no rebound" and to keep the NaN cells where
        the answer is "unknown".  ``nan`` marks exactly the unknown cells — the
        ice-surface holes from
        :func:`paleoeurope.gia.ice6g_topo.solid_displacement` — and nothing else.
        Pass ``0.0`` explicitly only to reproduce a pre-2026-07-30 artefact.
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
        "nodata": nodata,
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
    oob_fill: float = 0.0,
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
    oob_fill : float, optional
        Value written where the output grid falls outside the model grid's
        extent.  Default ``0.0`` for backwards compatibility, but note that this
        is the same "unknown looks like no correction" conflation the rest of this
        module now refuses: 0 m of Δz_GIA is a legitimate measured value.  The
        production caller pads the model subset by 1° beyond the output bbox so
        the case never arises; whenever it *does* arise a WARNING is logged with
        the cell count, so it can no longer happen silently.  Pass
        ``float("nan")`` to make it explicit instead.

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
        fill_value=oob_fill,
    )
    matrix_arr = interp(pts).reshape(n_rows, n_cols).astype(np.float32)

    del lon_g, lat_g, pts, interp

    # Out-of-extent cells get *filled*, not measured.  Never let that be silent:
    # with the default oob_fill=0.0 those cells are indistinguishable from
    # "measured zero rebound".  Counted from the 1-D coordinate vectors, so this
    # costs nothing.
    n_in_rows = int(np.count_nonzero(
        (out_lats >= field_lats[0]) & (out_lats <= field_lats[-1])
    ))
    n_in_cols = int(np.count_nonzero(
        (out_lons >= field_lons[0]) & (out_lons <= field_lons[-1])
    ))
    n_oob = n_rows * n_cols - n_in_rows * n_in_cols
    if n_oob:
        logger.warning(
            "build_correction_matrix: %d of %d output cells (%.1f%%) fall outside "
            "the model grid extent [%.3f, %.3f]x[%.3f, %.3f] and were FILLED with "
            "%r — these carry no measurement.  Widen the model subset (production "
            "pads by 1°) or pass oob_fill=float('nan').",
            n_oob, n_rows * n_cols, 100.0 * n_oob / (n_rows * n_cols),
            float(field_lats[0]), float(field_lats[-1]),
            float(field_lons[0]), float(field_lons[-1]),
            oob_fill,
        )

    if gaussian_sigma is not None and gaussian_sigma > 0:
        matrix_arr = gaussian_filter(matrix_arr.astype(np.float64),
                                     sigma=gaussian_sigma).astype(np.float32)
        np.clip(matrix_arr, 0.0, None, out=matrix_arr)

    matrix_transform = rasterio.transform.from_bounds(
        west, south, east, north, n_cols, n_rows
    )

    # Range over the *finite* cells: a plain .min()/.max() reports NaN for the
    # whole matrix as soon as one ice-surface hole is present, which hides the
    # very information this line exists to show.
    n_unknown = int(np.count_nonzero(~np.isfinite(matrix_arr)))
    if n_unknown < matrix_arr.size:
        lo = float(np.nanmin(matrix_arr))
        hi = float(np.nanmax(matrix_arr))
    else:
        lo = hi = float("nan")
    logger.debug(
        "build_correction_matrix: shape=%s  res=%.6f°  finite range=[%.2f, %.2f]  "
        "unknown(NaN)=%d",
        matrix_arr.shape,
        res_deg,
        lo,
        hi,
        n_unknown,
    )

    return matrix_arr, matrix_transform


def read_correction_matrix(
    matrix_arr: np.ndarray,
    matrix_transform: "rasterio.transform.Affine",
    bounds: "rasterio.coords.BoundingBox",
    h: int,
    w: int,
    resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.bilinear,
    preserve_nan: bool = True,
) -> np.ndarray:
    """Window-read a tile-sized slice of an in-memory correction matrix.

    Uses a ``rasterio.MemoryFile`` so the resampling path is
    identical to reading from a GeoTIFF on disk — matching production
    SYNC-mode ``_read_dmk()`` exactly.

    Pass ``resampling=rasterio.enums.Resampling.cubic`` when reading ICE
    thickness to eliminate the linear-ramp artefact caused by low-resolution
    (1°) ICE-7G grids being bilinearly stretched.

    NaN policy
    ----------
    **NaN is data and is passed through.**  Until 2026-07-30 this function ended
    with ``np.nan_to_num(result, nan=0.0, copy=False)``, which quietly defeated the
    ice-surface guard one hop upstream:
    :func:`paleoeurope.gia.ice6g_topo.solid_displacement` returns NaN where ICE-6G
    ``Topo`` is the **ice surface** and the files therefore hold no solid-surface
    elevation at all, and this call turned every such cell into ``0.0`` — a
    confident *"no isostatic correction here"* over precisely the formerly
    glaciated ground where the correction is largest (median +1141 m of
    ice-surface offset at 21 ka).  Measured on a 60x60 N62E006 tile with a single
    glaciated model node, 900 of 3600 cells (25 %) arrived as exactly 0.0.

    NaN in the result therefore means **unknown**, never "zero".  Consumers must
    either restrict themselves to the finite cells or source Δz from ICE-7G's
    bedrock-deformation field; :func:`paleoeurope.gia.correction_pipeline.run_single_tile_epochs`
    does the former and refuses to write an unknown cell as a number.

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
    preserve_nan : bool, optional
        Default ``True`` — the honest behaviour: unknown cells stay NaN.

        ``False`` is **not** a zero-fill.  It is an assertion by the caller that
        this matrix has no unknown cells; if any are present the call raises
        :class:`CorrectionMatrixHasUnknownCellsError` naming the count, because
        the only two ways to "remove" a NaN here are to refuse it or to lie about
        it.  On a NaN-free matrix both modes return the identical array, so
        existing NaN-free callers are unaffected either way.

    Returns
    -------
    np.ndarray, dtype float32, shape (h, w)
        A resampled slice of the correction matrix.  **May contain NaN**, meaning
        *unknown* (ice surface, no bedrock in the source files) — see the NaN
        policy above.

    Raises
    ------
    CorrectionMatrixHasUnknownCellsError
        If ``preserve_nan=False`` and the resampled slice contains NaN.

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

    n_unknown = int(np.count_nonzero(~np.isfinite(result)))
    if n_unknown and not preserve_nan:
        raise CorrectionMatrixHasUnknownCellsError(
            f"read_correction_matrix: {n_unknown} of {result.size} cells "
            f"({100.0 * n_unknown / result.size:.1f}%) of this tile are unknown "
            "(NaN) because ICE-6G 'Topo' is the ICE SURFACE there and these files "
            "carry no bedrock-under-ice field.  preserve_nan=False asserts a "
            "NaN-free matrix and is refused rather than honoured by zero-filling: "
            "0.0 is a legitimate measured Δz_GIA, so a fill here would record 'no "
            "isostatic correction' over formerly glaciated ground.  Either keep "
            "preserve_nan=True and skip the unknown cells, or take Δz from "
            "ICE-7G's bedrock-deformation field."
        )
    if n_unknown:
        logger.info(
            "read_correction_matrix: %d of %d tile cells (%.1f%%) are unknown "
            "(ice-surface NaN) and are returned as NaN, not 0.0",
            n_unknown, result.size, 100.0 * n_unknown / result.size,
        )
    return result
