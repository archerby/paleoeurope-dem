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

__all__ = [
    "run_single_tile_epochs",
    "EpochResult",
    "UnknownGiaCorrectionError",
    "GIA_NAN_POLICIES",
]

logger = logging.getLogger(__name__)

# ── Defaults matching production SYNC mode ────────────────────────────────────
DEFAULT_PADDING_DEG: float = 0.5   # bbox padding for correction matrix extent
DEFAULT_ICE_SIGMA: float = 1.5     # Gaussian σ for ice-thickness smoothing
DEFAULT_ICE_THRESHOLD_M: float = 10.0  # pixels with ice > 10 m get surface overlay

#: Accepted values for ``run_single_tile_epochs(gia_nan_policy=...)``.
#:
#: ``"raise"`` (default)
#:     Any unknown Δz cell aborts the tile with
#:     :class:`UnknownGiaCorrectionError`.  Correct for every tile that should be
#:     ice-free at the epoch — the Black Sea / Manych corridor, all of the
#:     Khvalynian work — where an unknown cell means a bug or the wrong epoch,
#:     not a fact about the world.
#: ``"nodata"``
#:     The legitimate case: a tile that genuinely straddles a palaeo-ice margin
#:     (Fennoscandia at 21 ka).  Unknown cells are written as the tile's declared
#:     nodata value and reported in ``EpochResult.gia_unknown_mask``.  They are
#:     never written as 0 m of correction and never left as a bare NaN inside a
#:     raster whose header declares a different nodata.
#:
#: There is deliberately no ``"zero"`` policy.
GIA_NAN_POLICIES: tuple[str, ...] = ("raise", "nodata")


class UnknownGiaCorrectionError(ValueError):
    """``dz_tile`` holds cells with no solid-surface displacement at all.

    ICE-6G ``Topo`` is the **ice surface** wherever the icemask is set, so
    :func:`paleoeurope.gia.ice6g_topo.solid_displacement` returns NaN there and
    these files ship no bedrock-under-ice field to fall back on.  Adding such a
    cell to a modern DEM is meaningless; zero-filling it first — which
    ``read_correction_matrix`` used to do — is worse, because the result then reads
    as "this ground did not move" over exactly the formerly glaciated ground that
    moved the most.

    Resolve it by one of:

    * restricting the tile/epoch to ice-free ground (what every Khvalynian /
      Black Sea tile already is);
    * sourcing Δz from ICE-7G's bedrock-deformation field, the same source this
      module already uses for thickness;
    * passing ``gia_nan_policy="nodata"`` to accept the hole and have it recorded
      as nodata — an honest "unknown", not a number.
    """


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
                              NaN where the GIA correction is unknown.
    dz_min : float            Minimum GIA bedrock Δz over the **known** cells of
                              the model field (m); NaN only if none are known.
    dz_max : float            Maximum GIA bedrock Δz over the known cells (m).
    gia_unknown_mask : np.ndarray
                              Bool array — True where no solid-surface
                              displacement exists because ICE-6G ``Topo`` is the
                              ice surface there.  These cells are *unknown*, not
                              zero; they are written to the paleo tile as its
                              nodata value.  All-False for ice-free tiles and for
                              epoch 0.
    """

    __slots__ = (
        "epoch_ka", "paleo_path", "ice_path",
        "ice_mask", "ice_h_tile", "paleo_bedrock",
        "dz_min", "dz_max", "gia_unknown_mask",
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
        gia_unknown_mask: np.ndarray | None = None,
    ) -> None:
        self.epoch_ka      = epoch_ka
        self.paleo_path    = paleo_path
        self.ice_path      = ice_path
        self.ice_mask      = ice_mask
        self.ice_h_tile    = ice_h_tile
        self.paleo_bedrock = paleo_bedrock
        self.dz_min        = dz_min
        self.dz_max        = dz_max
        self.gia_unknown_mask = (
            np.zeros(np.shape(ice_mask), dtype=bool)
            if gia_unknown_mask is None
            else gia_unknown_mask
        )

    def to_dict(self) -> dict:
        """Return a dict compatible with the legacy *epochs_data* format."""
        return {
            "ice_mask":      self.ice_mask,
            "ice_h_tile":    self.ice_h_tile,
            "paleo_bedrock": self.paleo_bedrock,
            "dz_min":        self.dz_min,
            "dz_max":        self.dz_max,
            "gia_unknown_mask": self.gia_unknown_mask,
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
    gia_nan_policy: str = "raise",
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
    gia_nan_policy : {'raise', 'nodata'}, optional
        What to do when ``dz_tile`` contains cells with no solid-surface
        displacement — see :data:`GIA_NAN_POLICIES`.  Default ``'raise'``.
        Zero-filling is not on offer in either mode.
    verbose : bool, optional
        Print progress messages.  Default ``True``.

    Returns
    -------
    dict[float, EpochResult]
        Mapping from epoch (ka) to an :class:`EpochResult` object.
        Call ``.to_dict()`` on each value for the legacy ``epochs_data`` format.

    Raises
    ------
    UnknownGiaCorrectionError
        With ``gia_nan_policy='raise'``, when the GIA correction is unknown over
        any cell of the tile (ICE-6G ``Topo`` is the ice surface there).
    ValueError
        If *gia_nan_policy* is not one of :data:`GIA_NAN_POLICIES`, or if the
        ICE-7G thickness read yields non-finite cells (thickness has no
        legitimate "unknown": the sidecar declares ``nodata=0.0``, so a NaN there
        would be silently written as *ice-free*, dropping ice rather than
        flagging it).

    Notes
    -----
    Unknown-value policy.  ``dz_tile`` comes from
    :func:`~paleoeurope.gia.correction_matrix.read_correction_matrix`, which as of
    2026-07-30 is NaN-preserving.  Before that it ended in
    ``np.nan_to_num(result, nan=0.0)``, so the ice-surface guard in
    :func:`~paleoeurope.gia.ice6g_topo.solid_displacement` never reached
    production: an unknown Δz became ``0.0`` and ``paleo_bedrock = modern + 0``
    silently claimed the ground had not moved.  Nothing in this function may
    convert an unknown cell into a number; it either aborts or records nodata.
    """
    if gia_nan_policy not in GIA_NAN_POLICIES:
        raise ValueError(
            f"gia_nan_policy={gia_nan_policy!r} is not one of {GIA_NAN_POLICIES}. "
            "There is no zero-fill policy: 0.0 is a legitimate measured Δz_GIA, so "
            "filling an unknown cell with it would record 'this ground did not "
            "move' over formerly glaciated ground."
        )

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

            # ── Unknown Δz policy ────────────────────────────────────────────
            # NaN here means "ICE-6G Topo is the ice surface, there is no solid
            # surface in these files", not "no correction".  It must never be
            # added to the DEM as a number, and it must never reach the output
            # raster as a bare NaN under a header that declares nodata=-9999.
            gia_unknown = ~np.isfinite(dz_tile)
            n_gia_unknown = int(np.count_nonzero(gia_unknown & ~nodata_mask))
            if n_gia_unknown:
                detail = (
                    f"epoch {epoch_ka} ka, tile {tile_id}: the GIA correction is "
                    f"unknown over {n_gia_unknown} of {gia_unknown.size} cells "
                    f"({100.0 * n_gia_unknown / gia_unknown.size:.1f}%) that hold "
                    "valid modern elevation, because ICE-6G 'Topo' is the ICE "
                    "SURFACE there and these files carry no bedrock-under-ice "
                    "field."
                )
                if gia_nan_policy == "raise":
                    raise UnknownGiaCorrectionError(
                        detail
                        + "  Restrict this tile/epoch to ice-free ground, take Δz "
                        "from ICE-7G's bedrock-deformation field, or pass "
                        "gia_nan_policy='nodata' to record the hole as nodata.  It "
                        "will not be filled with 0 m."
                    )
                logger.warning(
                    "%s  gia_nan_policy='nodata': written as nodata=%r, NOT as 0 m "
                    "of correction.", detail, nodata_val,
                )

            # ICE-7G thickness has no legitimate unknown, and the sidecar declares
            # nodata=0.0 — so a NaN slipping through would be written as *ice-free*
            # and would drop ice instead of flagging it.  Refuse outright.
            n_ice_unknown = int(np.count_nonzero(~np.isfinite(ice_h_tile)))
            if n_ice_unknown:
                raise ValueError(
                    f"epoch {epoch_ka} ka, tile {tile_id}: ICE-7G thickness is "
                    f"non-finite over {n_ice_unknown} of {ice_h_tile.size} cells.  "
                    "Thickness has no 'unknown' state in this pipeline: the "
                    "sidecar's nodata is 0.0, so clipping/zeroing these cells "
                    "would record them as ICE-FREE and silently remove ice.  Fix "
                    "the ICE-7G source or extent instead."
                )

            # Everything downstream of here treats "unknown modern elevation" and
            # "unknown GIA correction" identically: both are nodata, neither is 0.
            unknown_mask = nodata_mask | gia_unknown

            modern_dem = win.data.astype(np.float32)
            modern_dem = np.where(nodata_mask, np.nan, modern_dem)

            paleo_bedrock = (modern_dem + dz_tile).astype(np.float32)
            paleo_bedrock = np.where(unknown_mask, np.nan, paleo_bedrock)

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

            paleo_surface = np.where(unknown_mask, nodata_val, paleo_surface).astype(np.float32)

            # Belt and braces: nothing non-finite may leave this function inside a
            # raster whose header declares a finite nodata.  A residual NaN would
            # be read back as a valid float by any consumer using masked=False.
            residual = ~np.isfinite(paleo_surface)
            if residual.any():
                paleo_surface = np.where(residual, nodata_val, paleo_surface).astype(np.float32)
                logger.warning(
                    "epoch %s ka, tile %s: %d non-finite cells survived the "
                    "surface composition and were written as nodata=%r",
                    epoch_ka, tile_id, int(np.count_nonzero(residual)), nodata_val,
                )

            # ── Write ice sidecar and paleo tile ─────────────────────────────
            ice_profile = tile_profile.copy()
            ice_profile["nodata"] = 0.0
            _write_single_band_geotiff(ice_path, ice_h_tile, ice_profile)
            _write_single_band_geotiff(paleo_path, paleo_surface, tile_profile)

            # Range over the KNOWN cells.  A plain min()/max() returns NaN for the
            # whole matrix as soon as one ice-surface hole exists, which destroys
            # the diagnostic; the count of unknown cells is carried separately by
            # ``gia_unknown_mask`` rather than being smuggled into the range.
            if np.isfinite(gia_matrix).any():
                dz_min, dz_max = float(np.nanmin(gia_matrix)), float(np.nanmax(gia_matrix))
            else:
                dz_min = dz_max = float("nan")

            results[float(epoch_ka)] = EpochResult(
                epoch_ka=float(epoch_ka),
                paleo_path=paleo_path,
                ice_path=ice_path,
                ice_mask=ice_mask,
                ice_h_tile=ice_h_tile,
                paleo_bedrock=paleo_bedrock,
                dz_min=dz_min,
                dz_max=dz_max,
                gia_unknown_mask=gia_unknown,
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

