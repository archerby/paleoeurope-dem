"""
paleoeurope.gia.ice7g_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load ICE-7G model ice thickness fields (H_ice) for a bounding box and epoch.

Production model (see notebooks/geo_07_debbug_topo_epochs.ipynb) uses ICE-7G
*ice thickness* as a separate global matrix (ICE-DMK), alongside ICE-6G for
bedrock (GIA) correction.

This loader supports two common ICE-7G layouts:

1) A directory of per-epoch NetCDF files named like:
   ``I7G_NA.VM7_1deg.{epoch}.nc`` (epoch in ka BP, integer).

2) A single NetCDF file with a time dimension.

The only required data variable is ``stgit`` (ice thickness in metres).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Ice7gLayout:
    kind: str  # "file" | "dir"
    path: Path


class Ice7gLoader:
    """Load ICE-7G `stgit` thickness for an epoch and bbox."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ICE-7G path not found: {p}")

        if p.is_dir():
            self._layout = Ice7gLayout("dir", p)
            self._ds: xr.Dataset | None = None
        else:
            self._layout = Ice7gLayout("file", p)
            self._ds = None

    def get_thickness(
        self,
        *,
        epoch_ka: float,
        bounds: tuple[float, float, float, float],
        gaussian_sigma: float | None = None,
        buffer_deg: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lats, lons, ice_thickness_m) for one epoch and bbox."""

        ds = self._open_dataset_for_epoch(epoch_ka)

        lon_min, lat_min, lon_max, lat_max = bounds

        lat_dim = _detect_dim(ds, ("lat", "latitude", "Lat"))
        lon_dim = _detect_dim(ds, ("lon", "longitude", "Lon"))

        lat = ds[lat_dim].values
        lon = ds[lon_dim].values

        lat_sel = (lat >= lat_min - buffer_deg) & (lat <= lat_max + buffer_deg)
        lon_sel = (lon >= lon_min - buffer_deg) & (lon <= lon_max + buffer_deg)

        ds_sub = ds.isel({lat_dim: lat_sel, lon_dim: lon_sel})

        ice_var = _detect_var(ds_sub, ("stgit",))
        ice_da = ds_sub[ice_var]

        # Support both per-epoch files (2-D) and multi-epoch files (3-D).
        # ICE7G_fixture.nc uses a `time_ka` dimension.
        if ice_da.ndim == 3:
            time_dim = _detect_time_dim(ds_sub, ice_da.dims, lat_dim=lat_dim, lon_dim=lon_dim)
            if time_dim is None:
                raise ValueError(
                    "ICE-7G stgit is 3-D but a time dimension could not be identified. "
                    f"dims={ice_da.dims}"
                )
            ice_da = _select_time_slice(ice_da, time_dim=time_dim, epoch_ka=epoch_ka)

        ice_h = ice_da.values.astype(np.float32)

        # Some ICE-7G files store thickness as (lat, lon); some as (lon, lat)
        # if incorrectly encoded. We only support the common (lat, lon) order.
        if ice_h.ndim != 2:
            raise ValueError(f"ICE-7G stgit must be 2-D after time selection, got shape={ice_h.shape}")

        if gaussian_sigma is not None and gaussian_sigma > 0:
            ice_h = gaussian_filter(ice_h.astype(np.float64), sigma=gaussian_sigma).astype(np.float32)
            np.clip(ice_h, 0.0, None, out=ice_h)

        lats_sub = ds_sub[lat_dim].values
        lons_sub = ds_sub[lon_dim].values

        return lats_sub, lons_sub, ice_h

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_dataset_for_epoch(self, epoch_ka: float) -> xr.Dataset:
        if self._layout.kind == "file":
            return self._open_single_file(epoch_ka)
        return self._open_epoch_file(epoch_ka)

    def _open_single_file(self, epoch_ka: float) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_dataset(self._layout.path, engine="netcdf4")

        ds = self._ds

        # Common ICE-7G time coordinate names include `time_ka`.
        time_dim = next(
            (
                d
                for d in (
                    "time_ka",
                    "age_ka",
                    "time",
                    "Time",
                    "t",
                    "epoch",
                    "epochs",
                    "ka",
                    "age",
                )
                if d in ds.dims or d in ds.coords
            ),
            None,
        )

        # Fallback: infer time dim from stgit dims.
        if time_dim is None and "stgit" in ds.data_vars and ds["stgit"].ndim == 3:
            extras = [
                d
                for d in ds["stgit"].dims
                if d not in ("lat", "latitude", "Lat", "lon", "longitude", "Lon")
            ]
            if len(extras) == 1:
                time_dim = extras[0]

        if time_dim is None or "stgit" not in ds.data_vars or time_dim not in ds["stgit"].dims:
            return ds

        da = ds["stgit"]
        da_sel = _select_time_slice(da, time_dim=time_dim, epoch_ka=epoch_ka)
        return da_sel.to_dataset(name="stgit")


def _detect_time_dim(
    ds: xr.Dataset,
    ice_dims: tuple[str, ...],
    *,
    lat_dim: str,
    lon_dim: str,
) -> str | None:
    # Prefer explicit known names.
    for name in (
        "time_ka",
        "age_ka",
        "time",
        "Time",
        "t",
        "epoch",
        "epochs",
        "ka",
        "age",
    ):
        if name in ice_dims and (name in ds.dims or name in ds.coords):
            return name

    # Generic fallback: identify the only non-lat/lon dimension.
    extras = [d for d in ice_dims if d not in (lat_dim, lon_dim)]
    if len(extras) == 1:
        return extras[0]
    return None


def _select_time_slice(da: xr.DataArray, *, time_dim: str, epoch_ka: float) -> xr.DataArray:
    """Select the closest time slice for the requested epoch (ka BP)."""
    if time_dim not in da.dims:
        return da
    if time_dim not in da.coords:
        # No coordinate values; fall back to index rounding.
        idx = int(np.clip(int(round(float(epoch_ka))), 0, da.sizes[time_dim] - 1))
        return da.isel({time_dim: idx})

    time_vals = np.asarray(da[time_dim].values)
    try:
        time_vals_f = time_vals.astype(float)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"ICE-7G time coordinate '{time_dim}' is non-numeric: {time_vals!r}") from e

    target_time = _ka_to_model_time(float(epoch_ka), time_vals_f)
    idx = int(np.argmin(np.abs(time_vals_f - target_time)))
    return da.isel({time_dim: idx})

    def _open_epoch_file(self, epoch_ka: float) -> xr.Dataset:
        epoch_i = int(round(float(epoch_ka)))
        patterns = [
            f"I7G_NA.VM7_1deg.{epoch_i}.nc",
            f"*{epoch_i}*.nc",
        ]
        for pat in patterns:
            hits = sorted(self._layout.path.glob(pat))
            if hits:
                return xr.open_dataset(hits[0], engine="netcdf4")
        raise FileNotFoundError(
            "ICE-7G epoch file not found in directory. "
            f"dir={self._layout.path} epoch={epoch_i} tried={patterns}"
        )


def _detect_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError(
        f"Cannot find dimension; tried: {candidates}. Available: {list(ds.dims)}"
    )


def _detect_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Cannot find variable; tried: {candidates}. Available: {list(ds.data_vars)}"
    )


def _ka_to_model_time(epoch_ka: float, time_vals: np.ndarray) -> float:
    if time_vals.min() < 0:
        return -float(epoch_ka)
    return float(epoch_ka)
