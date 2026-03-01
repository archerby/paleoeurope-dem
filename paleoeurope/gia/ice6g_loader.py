"""
paleoeurope.gia.ice6g_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load ICE-6G_C (VM5a) model fields from the Peltier et al. (2015) NetCDF.

The NetCDF contains:
- ``sealevel`` — relative sea-level change (m) on a 0.5° grid
- ``iceheight`` — ice-sheet thickness (m), same grid
- ``RSL`` — relative sea level (m)
- ``time`` — model time in ka BP (negative = past), typically 0 to −26 ka

Peltier et al. (2015) https://doi.org/10.1002/2014JB011176
Data: https://www.physics.utoronto.ca/~peltier/data.html
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


class Ice6gLoader:
    """Load ICE-6G_C VM5a deformation/ice fields for a bounding box and epoch.

    Parameters
    ----------
    nc_path : str or Path
        Path to the ``ICE-6G_C_VM5a_O512.nc`` file (or equivalent).

    Examples
    --------
    >>> loader = Ice6gLoader("data/raw/ice6g/ICE-6G_C_VM5a_O512.nc")
    >>> dz, ice_h = loader.get_fields(epoch_ka=21, bounds=(-5, 47, 10, 60))
    >>> dz.shape  # depends on model resolution
    (26, 30)
    """

    def __init__(self, nc_path: str | Path) -> None:
        self.nc_path = Path(nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"ICE-6G file not found: {nc_path}")
        self._ds: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fields(
        self,
        epoch_ka: float,
        bounds: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return GIA deformation and ice thickness for one epoch and bbox.

        Parameters
        ----------
        epoch_ka : float
            Target epoch in ka BP (e.g. ``21`` for the LGM, ``0`` for modern).
        bounds : tuple
            ``(lon_min, lat_min, lon_max, lat_max)`` in EPSG:4326 degrees.

        Returns
        -------
        lats : np.ndarray  shape (M,)
        lons : np.ndarray  shape (N,)
        delta_z : np.ndarray  shape (M, N)
            GIA-related elevation change at *epoch_ka* relative to 0 ka (metres).
            Positive = land was higher (isostatic rebound not yet completed).
        ice_height : np.ndarray  shape (M, N)
            Ice-sheet thickness (m) at *epoch_ka*.  0 where ice-free.
        """
        ds = self._open()

        lon_min, lat_min, lon_max, lat_max = bounds
        buf = 1.0  # 1° buffer for interpolation safety

        # Subset by lat/lon — handle different possible dimension names
        lat_dim = _detect_dim(ds, ("lat", "latitude", "Lat"))
        lon_dim = _detect_dim(ds, ("lon", "longitude", "Lon"))
        time_dim = _detect_dim(ds, ("time", "Time", "t"))

        lat = ds[lat_dim].values
        lon = ds[lon_dim].values

        lat_sel = (lat >= lat_min - buf) & (lat <= lat_max + buf)
        lon_sel = (lon >= lon_min - buf) & (lon <= lon_max + buf)

        ds_sub = ds.isel({lat_dim: lat_sel, lon_dim: lon_sel})

        # Select nearest epoch
        time_vals = ds_sub[time_dim].values  # expected in ka (negative = past)
        target_time = _ka_to_model_time(epoch_ka, time_vals)
        time_idx = int(np.argmin(np.abs(time_vals - target_time)))
        ds_t = ds_sub.isel({time_dim: time_idx})
        ds_0 = ds_sub.isel({time_dim: int(np.argmin(np.abs(time_vals)))})

        # ── GIA bedrock delta ─────────────────────────────────────────────────
        # Production pipeline (geo_07 / scripts/geo/deformation.py GIAModel)
        # uses ``Orog(t) − Orog(0)`` — the BEDROCK orographic delta that
        # captures only crustal deformation (GIA isostatic flexure).
        # This excludes the ice-surface contribution stored in ``Topo``.
        # Equivalent to GIAModel.build_correction_matrix() in the Celery pipeline.
        #
        # Fallback chain:
        #   1. Orog(t) − Orog(0)   ← preferred: bedrock-only GIA delta
        #   2. stgr = Topo(t)−Topo(0) ← includes ice surface; only if Orog absent
        #   3. RSL(t) − RSL(0)      ← last resort
        #
        # Gaussian smoothing (σ=2.0 on the ~10-arcmin grid ≈ 36 km blur) is
        # applied in all cases — physically correct for long-wavelength
        # lithospheric flexure and matches GIAModel.build_correction_matrix().
        ice_var = _detect_var(ds, ("sice", "iceheight", "ice", "thick", "H_ice", "hice"))

        if "Orog" in ds_sub.data_vars:
            dz_raw = (ds_t["Orog"].values.astype(np.float64)
                      - ds_0["Orog"].values.astype(np.float64))
        elif "stgr" in ds_sub.data_vars:
            warnings.warn(
                "Orog not found — falling back to stgr = Topo(t)−Topo(0). "
                "stgr includes ice-surface changes and is not a pure bedrock "
                "GIA delta.",
                UserWarning,
                stacklevel=3,
            )
            dz_raw = ds_t["stgr"].values.astype(np.float64)
        else:
            rsl_var = _detect_var(ds, ("sealevel", "RSL", "rsl", "geoid"))
            dz_raw = (ds_t[rsl_var] - ds_0[rsl_var]).values.astype(np.float64)

        # Gaussian smoothing on the coarse model grid (σ=2.0 → ~36 km at
        # 10 arcmin): same as GIAModel.build_correction_matrix(sigma=2.0).
        dz = gaussian_filter(dz_raw, sigma=2.0).astype(np.float32)

        ice_h_raw = ds_t[ice_var].values.astype(np.float32)

        # ICE-6G_C VM5a stores ice-sheet thickness in **centimetres** in the
        # ``sice`` variable (max ~300 000 cm = 3 000 m at LGM).  Convert to
        # metres so that callers can use physically meaningful thresholds
        # (e.g. ``ice_threshold_m = 1.0`` means ≥ 1 m of ice).
        # The conversion is only applied when the raw maximum exceeds the
        # largest physically possible ice thickness in metres (~5 000 m), which
        # reliably distinguishes a cm-encoded field from a metres-encoded one.
        if ice_h_raw.max() > 5_000.0:
            ice_h = ice_h_raw / 100.0   # cm → m
        else:
            ice_h = ice_h_raw           # already in metres

        lats_sub = ds_sub[lat_dim].values
        lons_sub = ds_sub[lon_dim].values

        return lats_sub, lons_sub, dz, ice_h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_dataset(self.nc_path, engine="netcdf4")
        return self._ds

    def close(self) -> None:
        """Release the open NetCDF file handle."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------


def _detect_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError(f"Cannot find dimension; tried: {candidates}.  Available: {list(ds.dims)}")


def _detect_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Cannot find variable; tried: {candidates}.  Available: {list(ds.data_vars)}")


def _ka_to_model_time(epoch_ka: float, time_vals: np.ndarray) -> float:
    """Map a user-supplied epoch (positive ka BP) to the model's time axis."""
    # ICE-6G uses negative ka for past and 0 for present
    if time_vals.min() < 0:
        return -float(epoch_ka)
    # Some files use positive ka convention
    return float(epoch_ka)
