"""
paleoeurope.gia.correction_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level pipeline: apply GIA correction and ice-thickness overlay to a single
fused DEM tile for a list of epochs.

This module wraps the per-epoch loop that was previously inlined in notebooks,
so it can be:

- Called from the publication notebook with a single function call.
- Imported from Celery tasks without duplication.
- Unit-tested with fixture data independently of any notebook state.

Naming
------
``correction_matrix`` replaces the legacy term ``DMK`` used in earlier
development notebooks.

Usage (notebook §4 pattern)
----------------------------
>>> from paleoeurope.gia.correction_pipeline import run_single_tile_epochs
>>> epochs_data = run_single_tile_epochs(
...     fusion_path  = OUTPUT_DIR / 'N62E006_fusion.tif',
...     ice6g_path   = FIXTURES / 'ice6g' / 'ICE6G_fixture.nc',
...     ice7g_path   = FIXTURES / 'ice7g' / 'ICE7G_fixture.nc',
...     output_dir   = OUTPUT_DIR,
...     tile_id      = 'N62E006',
...     epochs_ka    = [0.0, 8.0, 12.0, 21.0],
... )
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.transform

from paleoeurope.gia.correction_matrix import (
    CM_RESOLUTION_DEG,
    build_correction_matrix,
    read_correction_matrix,
    write_correction_matrix,
)
from paleoeurope.gia.ice6g_loader import Ice6gLoader
from paleoeurope.gia.ice7g_loader import Ice7gLoader
from paleoeurope.ice.envelope import apply_envelope_method
from paleoeurope.utils.raster import RasterWindow, read_geotiff

__all__ = ["run_single_tile_epochs", "EpochResult"]

logger = logging.getLogger(__name__)

# ── Defaults matching production SYNC mode ────────────────────────────────────
DEFAULT_PADDING_DEG: float = 0.5   # bbox padding for correction matrix extent
DEFAULT_ICE_SIGMA: float = 1.5     # Gaussian σ for ice-thickness smoothing
DEFAULT_ICE_THRESHOLD_M: float = 10.0  # pixels with ice > 10 m get surface overlay


class EpochResult:
    """Container for per-epoch outputs from :func:`run_single_tile_epochs`.

    Attributes
    ----------
    epoch_ka : float          Target epoch in kiloyears BP.
    paleo_path : Path         Written ``*_t{N}ka.tif`` (surface = bedrock + ice).
    ice_path : Path           Written ``*_t{N}ka_ice.tif`` sidecar.
    ice_mask : np.ndarray     Bool array – True where ice was present.
    ice_h_tile : np.ndarray   Ice thickness on the tile grid (m, float32).
    paleo_bedrock : np.ndarray  Bedrock surface before ice overlay (m, float32).
    dz_min : float            Minimum GIA bedrock Δz on model field (m).
    dz_max : float            Maximum GIA bedrock Δz on model field (m).
    """

    __slots__ = (
        "epoch_ka", "paleo_path", "ice_path",
        "ice_mask", "ice_h_tile", "paleo_bedrock",
        "dz_min", "dz_max",
    )

    def __init__(
        self,
        epoch_ka: float,
        paleo_path: Path,
        ice_path: Path,
        ice_mask: np.ndarray,
        ice_h_tile: np.ndarray,
        paleo_bedrock: np.ndarray,
        dz_min: float,
        dz_max: float,
    ) -> None:
        self.epoch_ka      = epoch_ka
        self.paleo_path    = paleo_path
        self.ice_path      = ice_path
        self.ice_mask      = ice_mask
        self.ice_h_tile    = ice_h_tile
        self.paleo_bedrock = paleo_bedrock
        self.dz_min        = dz_min
        self.dz_max        = dz_max

    def to_dict(self) -> dict:
        """Return a dict compatible with the legacy *epochs_data* format."""
        return {
            "ice_mask":      self.ice_mask,
            "ice_h_tile":    self.ice_h_tile,
            "paleo_bedrock": self.paleo_bedrock,
            "dz_min":        self.dz_min,
            "dz_max":        self.dz_max,
        }


def run_single_tile_epochs(
    fusion_path: str | Path,
    ice6g_path: str | Path,
    ice7g_path: str | Path,
    output_dir: str | Path,
    tile_id: str,
    epochs_ka: list[float],
    *,
    padding_deg: float = DEFAULT_PADDING_DEG,
    ice_sigma: float = DEFAULT_ICE_SIGMA,
    ice_threshold_m: float = DEFAULT_ICE_THRESHOLD_M,
    blend_ice: bool = True,
    transition_depth_m: float = 200.0,
    cm_res_deg: float = CM_RESOLUTION_DEG,
    compress: str = "lzw",
    verbose: bool = True,
) -> dict[float, EpochResult]:
    """Apply GIA correction for each epoch and write output tiles.

    This function encapsulates the per-epoch loop that was previously inlined
    in the publication notebook (§4).  It mirrors production SYNC mode from
    ``scripts/geo/deformation.py``:

    * Epoch 0 ka: copied as-is (no deformation).
    * Other epochs:

      1. Build **GIA correction matrix** from model bedrock Δz.
      2. Build **ice correction matrix** from model H_ice (Gaussian-smoothed).
      3. Window-read GIA matrix at tile resolution via bilinear resampling;
         ice-thickness matrix via **cubic** resampling (avoids the linear-ramp
         artefact caused by ICE-7G's coarse 1° grid).
      4. Compose surface: ``bedrock + ice`` where ``ice > ice_threshold_m``.
      5. Write ``*_t{N}ka.tif``  (surface = bedrock + ice).
      6. Write ``*_t{N}ka_ice.tif``  ice-thickness sidecar.

    Parameters
    ----------
    fusion_path : str or Path
        Path to the modern fused DEM tile (``*_fusion.tif``).
    ice6g_path : str or Path
        Path to the ICE-6G NetCDF fixture or full dataset.
    output_dir : str or Path
        Directory for output tiles.  Created if needed.
    tile_id : str
        Tile identifier (e.g. ``'N62E006'``).  Used for output filenames.
    epochs_ka : list[float]
        List of target epochs in kiloyears BP (e.g. ``[0.0, 8.0, 12.0, 21.0]``).
    padding_deg : float, optional
        Padding around the tile bbox for the correction matrix extent.
        Default ``0.5°``.
    ice_sigma : float, optional
        Gaussian σ for H_ice smoothing (pixel units on model grid).
        Default ``1.5``.  Set to ``0.0`` to disable.
    ice_threshold_m : float, optional
        Minimum ice thickness (m) for surface overlay.  Default ``10.0``.
    cm_res_deg : float, optional
        Correction matrix resolution in degrees.  Default ``1/360`` (10 arcsec).
    compress : str, optional
        GeoTIFF compression codec.  Default ``'lzw'``.
    verbose : bool, optional
        Print progress messages.  Default ``True``.

    Returns
    -------
    dict[float, EpochResult]
        Mapping from epoch (ka) to an :class:`EpochResult` object.
        Call ``.to_dict()`` on each value for the legacy ``epochs_data`` format.
    """
    fusion_path = Path(fusion_path)
    ice6g_path  = Path(ice6g_path)
    ice7g_path  = Path(ice7g_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read modern fused DEM once ────────────────────────────────────────────
    win: RasterWindow = read_geotiff(fusion_path)
    with rasterio.open(fusion_path) as _src:
        tile_profile = _src.profile.copy()
        tile_bounds  = _src.bounds
        tile_h, tile_w = _src.height, _src.width

    tile_profile.update(
        dtype="float32",
        nodata=-9999.0,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    # Padded bbox for correction matrices
    cm_west  = tile_bounds.left   - padding_deg
    cm_south = tile_bounds.bottom - padding_deg
    cm_east  = tile_bounds.right  + padding_deg
    cm_north = tile_bounds.top    + padding_deg

    gia_loader = Ice6gLoader(ice6g_path)
    ice_loader = Ice7gLoader(ice7g_path)
    results: dict[float, EpochResult] = {}

    try:
        for epoch_ka in epochs_ka:
            paleo_path = output_dir / f"{tile_id}_t{int(epoch_ka):02d}ka.tif"
            ice_path   = output_dir / f"{tile_id}_t{int(epoch_ka):02d}ka_ice.tif"

            if verbose:
                print(f"\n{'─'*60}\n  Epoch {int(epoch_ka)} ka\n{'─'*60}")

            # ── Epoch 0: modern baseline ──────────────────────────────────────
            if epoch_ka == 0.0:
                shutil.copy2(str(fusion_path), str(paleo_path))
                empty = np.zeros((tile_h, tile_w), dtype=np.float32)
                results[0.0] = EpochResult(
                    epoch_ka=0.0,
                    paleo_path=paleo_path,
                    ice_path=paleo_path,   # no ice sidecar for epoch-0
                    ice_mask=empty.astype(bool),
                    ice_h_tile=empty,
                    paleo_bedrock=win.data.copy(),
                    dz_min=0.0,
                    dz_max=0.0,
                )
                if verbose:
                    print("  ✅ copied (no deformation)")
                continue

            # ── Get ICE-6G fields ─────────────────────────────────────────────
            lats6, lons6, dz, _ice6g_h_m = gia_loader.get_fields(
                epoch_ka=epoch_ka,
                bounds=(cm_west, cm_south, cm_east, cm_north),
            )

            # ── Get ICE-7G thickness field ───────────────────────────────────
            lats7, lons7, ice7_h_m = ice_loader.get_thickness(
                epoch_ka=epoch_ka,
                bounds=(cm_west, cm_south, cm_east, cm_north),
                gaussian_sigma=None,  # smoothing is applied after interpolation
            )

            # ── Build GIA correction matrix (bedrock uplift) ──────────────────
            gia_matrix, gia_tf = build_correction_matrix(
                dz, lats6, lons6,
                cm_west, cm_south, cm_east, cm_north,
                res_deg=cm_res_deg,
            )

            gia_dmk_path = output_dir / f"{tile_id}_gia_dmk_t{int(epoch_ka):02d}ka.tif"
            write_correction_matrix(gia_matrix, gia_tf, str(gia_dmk_path), compress=compress)

            # ── Build ice correction matrix (ice thickness) ───────────────────
            # Use cubic interpolation to produce a physically correct
            # parabolic ice-dome profile instead of the linear ramp that
            # bilinear gives when stretching a coarse 1° ICE-7G grid.
            ice_matrix, ice_tf = build_correction_matrix(
                ice7_h_m, lats7, lons7,
                cm_west, cm_south, cm_east, cm_north,
                res_deg=cm_res_deg,
                gaussian_sigma=ice_sigma if ice_sigma > 0 else None,
                interp_method="cubic",
            )

            ice_dmk_path = output_dir / f"{tile_id}_ice7g_dmk_t{int(epoch_ka):02d}ka.tif"
            write_correction_matrix(ice_matrix, ice_tf, str(ice_dmk_path), compress=compress)

            # ── Window-read matrices to tile grid (production behaviour) ─────
            dz_tile = read_correction_matrix(
                gia_matrix, gia_tf, tile_bounds, tile_h, tile_w,
                resampling=rasterio.enums.Resampling.bilinear,
            )
            ice_h_tile = read_correction_matrix(
                ice_matrix, ice_tf, tile_bounds, tile_h, tile_w,
                resampling=rasterio.enums.Resampling.cubic,
            )

            nodata_val = float(tile_profile.get("nodata", -9999.0))
            nodata_mask = np.isnan(win.data) | (win.data.astype(np.float32) == np.float32(nodata_val))

            modern_dem = win.data.astype(np.float32)
            modern_dem = np.where(nodata_mask, np.nan, modern_dem)

            paleo_bedrock = (modern_dem + dz_tile).astype(np.float32)
            paleo_bedrock = np.where(nodata_mask, np.nan, paleo_bedrock)

            ice_h_tile = np.clip(ice_h_tile, 0.0, None).astype(np.float32)
            ice_h_tile = np.where(nodata_mask, 0.0, ice_h_tile)

            ice_mask = ice_h_tile >= float(ice_threshold_m)
            ice_for_surface = np.where(ice_mask, ice_h_tile, 0.0).astype(np.float32)

            if blend_ice:
                paleo_surface = apply_envelope_method(
                    paleo_bedrock=paleo_bedrock,
                    ice_thickness=ice_for_surface,
                    modern_bedrock=modern_dem,
                    t_tr=float(transition_depth_m),
                ).astype(np.float32)
            else:
                paleo_surface = paleo_bedrock.copy()
                paleo_surface[ice_mask] = paleo_bedrock[ice_mask] + ice_h_tile[ice_mask]

            paleo_surface = np.where(nodata_mask, nodata_val, paleo_surface).astype(np.float32)

            # ── Write ice sidecar and paleo tile ─────────────────────────────
            ice_profile = tile_profile.copy()
            ice_profile["nodata"] = 0.0
            _write_single_band_geotiff(ice_path, ice_h_tile, ice_profile)
            _write_single_band_geotiff(paleo_path, paleo_surface, tile_profile)

            results[float(epoch_ka)] = EpochResult(
                epoch_ka=float(epoch_ka),
                paleo_path=paleo_path,
                ice_path=ice_path,
                ice_mask=ice_mask,
                ice_h_tile=ice_h_tile,
                paleo_bedrock=paleo_bedrock,
                dz_min=float(np.min(gia_matrix)),
                dz_max=float(np.max(gia_matrix)),
            )

            if verbose:
                print(f"  ✅ wrote {paleo_path.name}  (+ {ice_path.name})")

            # Reduce peak RAM for notebook kernels
            del gia_matrix, ice_matrix, dz_tile, ice_h_tile, paleo_surface
            gc.collect()

        return results

    finally:
        try:
            gia_loader.close()
        except Exception:
            pass
        try:
            ice_loader.close()
        except Exception:
            pass


def _write_single_band_geotiff(path: Path, arr2d: np.ndarray, profile: dict) -> None:
    prof = profile.copy()
    prof.update(count=1, dtype="float32")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr2d.astype(np.float32), 1)

