"""
paleoeurope.fusion.gebco_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load GEBCO 2024 bathymetry tiles and apply TID-based masking.

GEBCO (General Bathymetric Chart of the Oceans) provides global continuous
ocean depth / land elevation at 15 arc-second resolution.  The Type IDentifer
(TID) grid indicates data provenance; TID=0 marks direct bathymetric survey
coverage.

Reference: GEBCO Compilation Group (2024) https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds


class GebcoLoader:
    """Load a window of GEBCO bathymetry, reproject, and optionally mask by TID.

    Parameters
    ----------
    gebco_path : str or Path
        Path to the GEBCO GeoTIFF or NetCDF file.
    tid_path : str or Path, optional
        Path to the accompanying GEBCO TID grid.  If provided, land pixels
        (TID == 0 and elevation > 0) are masked to NaN.
    nodata : float, optional
        NoData fill value. Default ``-32767`` (GEBCO convention).

    Examples
    --------
    >>> loader = GebcoLoader("data/raw/gebco/GEBCO_2024.tif")
    >>> target_transform = rasterio.transform.from_bounds(0, 51, 1, 52, 3600, 3600)
    >>> arr = loader.read_window(
    ...     bounds=(0.0, 51.0, 1.0, 52.0),
    ...     target_shape=(3600, 3600),
    ...     target_transform=target_transform,
    ...     target_crs="EPSG:4326",
    ... )
    >>> arr.shape
    (3600, 3600)
    """

    def __init__(
        self,
        gebco_path: str | Path,
        tid_path: str | Path | None = None,
        nodata: float = -32767.0,
    ) -> None:
        self.gebco_path = Path(gebco_path)
        self.tid_path = Path(tid_path) if tid_path else None
        self.nodata = nodata

        if not self.gebco_path.exists():
            raise FileNotFoundError(f"GEBCO file not found: {gebco_path}")

        # Guardrail: avoid accidentally using the GEBCO TID grid as elevation.
        # This can silently produce nonsensical fusion results.
        if "tid" in self.gebco_path.stem.lower():
            raise ValueError(
                "gebco_path looks like a GEBCO TID/provenance grid. "
                "Pass the GEBCO elevation grid as gebco_path and the TID grid as tid_path. "
                f"Got gebco_path={self.gebco_path}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_window(
        self,
        bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int],
        target_transform: object,
        target_crs: object,
    ) -> np.ndarray:
        """Read, reproject, and optionally TID-mask a GEBCO window.

        Parameters
        ----------
        bounds : tuple
            ``(minx, miny, maxx, maxy)`` in the coordinate system of *target_crs*.
        target_shape : tuple of (height, width)
            Pixel dimensions of the output array.
        target_transform : rasterio.Affine
            Affine transform of the output grid.
        target_crs : rasterio.CRS or str
            Coordinate reference system of the output grid.

        Returns
        -------
        np.ndarray
            ``float32`` array of shape ``(height, width)``.  Ocean pixels have
            elevation ≤ 0; land cells from GEBCO are *not* removed here (the
            blender handles the merge).  NaN where no source data.
        """
        destination = np.zeros(target_shape, dtype=np.float32)

        with rasterio.open(self.gebco_path) as src:
            pad = 0.1
            minx, miny, maxx, maxy = bounds
            src_window = from_bounds(
                minx - pad, miny - pad, maxx + pad, maxy + pad,
                transform=src.transform,
            )

            raw = src.read(1, window=src_window, masked=False).astype(np.float32)

            if src.nodata is not None:
                raw[raw == src.nodata] = np.nan
            else:
                raw[raw == self.nodata] = np.nan

            src_transform = src.window_transform(src_window)
            src_crs = src.crs

        # Reproject onto target grid
        reproject(
            source=raw,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=WarpResampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        # Optional TID masking: remove GEBCO land cells (TID == 0, elev > 0)
        if self.tid_path is not None and self.tid_path.exists():
            destination = self._apply_tid_mask(destination, bounds, target_shape, target_transform, target_crs)

        return destination

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_tid_mask(
        self,
        data: np.ndarray,
        bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int],
        target_transform: object,
        target_crs: object,
    ) -> np.ndarray:
        """Mask GEBCO land pixels based on the TID grid.

        Parameters
        ----------
        data : np.ndarray
            Already-reprojected GEBCO array (float32).
        bounds, target_shape, target_transform, target_crs :
            Same as in :meth:`read_window`.

        Returns
        -------
        np.ndarray
            *data* with TID==0 land pixels set to NaN.
        """
        tid_dest = np.zeros(target_shape, dtype=np.float32)
        minx, miny, maxx, maxy = bounds
        pad = 0.1

        with rasterio.open(self.tid_path) as src:  # type: ignore[arg-type]
            src_window = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, transform=src.transform)
            tid_raw = src.read(1, window=src_window).astype(np.float32)
            src_transform = src.window_transform(src_window)
            src_crs = src.crs

        reproject(
            source=tid_raw,
            destination=tid_dest,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=WarpResampling.nearest,
        )

        # TID == 0 with positive elevation = land-based sounding → mask out
        land_tid_mask = (np.round(tid_dest).astype(int) == 0) & (data > 0)
        data[land_tid_mask] = np.nan

        return data
