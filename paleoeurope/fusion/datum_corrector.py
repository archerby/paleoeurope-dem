"""
paleoeurope.fusion.datum_corrector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply an EGM2008-based geoid/undulation offset to GEBCO elevation data.

IMPORTANT: Whether any geoid/undulation offset should be applied (and with
which sign) depends on what each dataset actually stores (orthometric,
ellipsoidal, or mean sea level heights) and must be validated for the
specific GEBCO--FABDEM pairing. The public demo pipeline keeps this step
optional and disabled by default.

Candidate formula::

    H_EGM2008 = H_MSL − N(φ, λ)

where N is the geoid undulation from the EGM2008 grid.

Reference: Pavlis et al. (2012) https://doi.org/10.1029/2011JB008916
Grid download: https://earth-info.nga.mil/index.php?dir=wgs84
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


class DatumCorrector:
    """Correct GEBCO MSL elevations to EGM2008.

    Parameters
    ----------
    grid_path : str or Path, optional
        Path to the EGM2008 geoid undulation grid (GeoTIFF, Float32, arc-minute
        or coarser resolution).  If ``None``, the environment variable
        ``VERTICAL_GRID_PATH`` is checked.  If still missing, the corrector
        applies a no-op (with a warning) so the pipeline can continue.

    Examples
    --------
    >>> import xarray as xr, numpy as np
    >>> da = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"])
    >>> corrector = DatumCorrector(grid_path=None)   # no-op mode
    >>> corrected = corrector.align(da)
    >>> np.allclose(corrected.values, 0.0)
    True
    """

    def __init__(self, grid_path: str | Path | None = None) -> None:
        candidate = grid_path or os.environ.get("VERTICAL_GRID_PATH")
        self.grid_path: Path | None = Path(candidate) if candidate else None

        if self.grid_path is not None and not self.grid_path.exists():
            warnings.warn(
                f"EGM2008 grid not found at {self.grid_path}. "
                "DatumCorrector will run in no-op mode.",
                UserWarning,
                stacklevel=2,
            )
            self.grid_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, da: xr.DataArray) -> xr.DataArray:
        """Apply EGM2008 geoid correction to *da*.

        Parameters
        ----------
        da : xr.DataArray
            Input elevation array (MSL, float32).  Must have ``x`` and ``y``
            coordinate arrays (decimal degrees, EPSG:4326).

        Returns
        -------
        xr.DataArray
            Corrected elevation array (EGM2008, float32), same shape and coords
            as *da*.  If the grid is unavailable, returns *da* unchanged with a
            warning.
        """
        if self.grid_path is None:
            warnings.warn(
                "DatumCorrector running in no-op mode (no EGM2008 grid). "
                "Geoid correction NOT applied.",
                UserWarning,
                stacklevel=2,
            )
            return da

        try:
            import rasterio
            from rasterio.windows import from_bounds as win_from_bounds
            from scipy.interpolate import RegularGridInterpolator

            # Derive bounds from x/y coords (set by _make_da) to avoid
            # requiring rioxarray just for this step.
            if "x" in da.coords and "y" in da.coords:
                xs = da.coords["x"].values
                ys = da.coords["y"].values
                dx = abs(xs[1] - xs[0]) if len(xs) > 1 else 0.0
                dy = abs(ys[1] - ys[0]) if len(ys) > 1 else 0.0
                bounds = (
                    float(xs.min()) - dx / 2,
                    float(ys.min()) - dy / 2,
                    float(xs.max()) + dx / 2,
                    float(ys.max()) + dy / 2,
                )
            else:
                bounds = da.rio.bounds()
            buffer = 0.05

            with rasterio.open(self.grid_path) as src:
                window = win_from_bounds(
                    bounds[0] - buffer,
                    bounds[1] - buffer,
                    bounds[2] + buffer,
                    bounds[3] + buffer,
                    src.transform,
                )
                geoid_raw = src.read(1, window=window, masked=True)
                win_transform = src.window_transform(window)

            rows, cols = geoid_raw.shape
            win_lons = win_transform[2] + np.arange(cols) * win_transform[0]
            win_lats = win_transform[5] + np.arange(rows) * win_transform[4]

            # Build interpolator (lat ascending)
            interp = RegularGridInterpolator(
                (win_lats[::-1], win_lons),
                np.flipud(geoid_raw.filled(0.0)),
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            lons = da.coords["x"].values if "x" in da.coords else _lons_from_da(da)
            lats = da.coords["y"].values if "y" in da.coords else _lats_from_da(da)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

            N = interp(pts).reshape(da.shape).astype(np.float32)

            corrected = da - N
            corrected.attrs = da.attrs
            return corrected

        except Exception as exc:
            warnings.warn(
                f"DatumCorrector.align failed ({exc}); returning input unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return da

    def undulation_at(self, lat: float, lon: float) -> float:
        """Return the EGM2008 geoid undulation N at a single point.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates in decimal degrees.

        Returns
        -------
        float
            Geoid undulation N in metres, or 0.0 if the grid is not available.
        """
        if self.grid_path is None:
            return 0.0

        import rasterio
        from rasterio.windows import from_bounds as win_from_bounds
        from scipy.interpolate import RegularGridInterpolator

        buf = 0.1
        with rasterio.open(self.grid_path) as src:
            window = win_from_bounds(lon - buf, lat - buf, lon + buf, lat + buf, src.transform)
            raw = src.read(1, window=window, masked=True)
            wt = src.window_transform(window)

        rows, cols = raw.shape
        lons = wt[2] + np.arange(cols) * wt[0]
        lats = wt[5] + np.arange(rows) * wt[4]
        interp = RegularGridInterpolator(
            (lats[::-1], lons), np.flipud(raw.filled(0.0)), bounds_error=False, fill_value=0.0
        )
        return float(interp([[lat, lon]])[0])


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _lons_from_da(da: xr.DataArray) -> np.ndarray:
    import rasterio  # noqa: F401

    t = da.rio.transform()
    width = da.shape[-1]
    return t[2] + np.arange(width) * t[0]


def _lats_from_da(da: xr.DataArray) -> np.ndarray:
    t = da.rio.transform()
    height = da.shape[-2]
    return t[5] + np.arange(height) * t[4]
