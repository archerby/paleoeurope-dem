"""
paleoeurope.gia.ice6g_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load ICE-6G_C (VM5a) model fields from the Peltier et al. (2015) NetCDF.

The NetCDF is expected to hold several epochs stacked on a ``time`` axis
(ka BP, negative = past, typically 0 to −26 ka) and must contain:

- ``Topo`` — topography/bathymetry (m) relative to the *contemporaneous* sea
  surface.  This is the only field the GIA delta is derived from; see
  :mod:`paleoeurope.gia.ice6g_topo`.

Optionally:

- ``sice``/``iceheight``/… — ice-sheet thickness (m or cm)
- ``sftgif`` — icemask (percent, 100 = ice); the real
  ``I6_C.VM5a_10min.<epoch>.nc`` files carry this and *no* thickness field.

``Orog`` is deliberately ignored: it is clamped to 0 over water, so an
``Orog(t) − Orog(0)`` delta is wrong exactly under a paleo-shoreline.

Peltier et al. (2015) https://doi.org/10.1002/2014JB011176
Data: https://www.physics.utoronto.ca/~peltier/data.html
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from paleoeurope.gia.ice6g_topo import implied_sea_surface_offset, solid_displacement

#: Genuine ice-*thickness* variable names (metres or centimetres).
ICE_THICKNESS_VARS = ("sice", "iceheight", "ice", "thick", "H_ice", "hice")

#: Ice *mask* variable names (percent, 100 = ice).  The distributed
#: ``I6_C.VM5a_10min.<epoch>.nc`` files carry ``sftgif`` and no thickness at all.
ICE_MASK_VARS = ("sftgif", "icemask", "sftgrf")


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
            Present-datum solid-earth vertical displacement ``Δz_solid`` at
            *epoch_ka* (metres), Gaussian-smoothed with σ=2.0 on the model grid.
            Positive = the solid surface stood higher than today.
        ice_height : np.ndarray  shape (M, N)
            Ice-sheet thickness (m) at *epoch_ka*.  0 where ice-free — and 0
            everywhere, with a warning, when the file carries only an icemask
            (``sftgif``) or no ice field at all.

        Raises
        ------
        ValueError
            If the file has no ``Topo`` field, or if the grid contains no
            deep-ocean cells from which to derive the sea-surface offset.
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
        zero_idx = int(np.argmin(np.abs(time_vals)))
        ds_t = ds_sub.isel({time_dim: time_idx})
        ds_0 = ds_sub.isel({time_dim: zero_idx})

        # ── GIA solid-earth delta ─────────────────────────────────────────────
        # Δz_solid = [Topo(t) − Topo(0)] − deep_ocean_median[Topo(t) − Topo(0)]
        # — see :mod:`paleoeurope.gia.ice6g_topo`.
        #
        # ``Orog(t) − Orog(0)`` is NOT an option: ICE-6G ``Orog`` is clamped to 0
        # over water, so ΔOrog vanishes (or flips sign) exactly under a paleo-
        # shoreline.  Measured over the Manych spillway at 16.5 ka the correct
        # Δz_solid is −26.5 m while ΔOrog is +80.6 m — enough to lift the +23 m
        # sill out of reach and make the Khvalynian spill west impossible at any
        # Caspian level, silently.  ``stgr`` (= bare ΔTopo) is not an option
        # either: it double-counts the ~112 m eustatic sea-surface change baked
        # into ``Topo``.  Neither branch is kept behind a flag.
        if "Topo" not in ds.data_vars:
            raise ValueError(
                "ICE-6G file lacks the 'Topo' field required for the GIA delta "
                "(Δz_solid = ΔTopo − deep-ocean median ΔTopo).  Neither 'Orog' "
                "(clamped to 0 over water) nor 'stgr'/'RSL' is an acceptable "
                f"substitute.  Variables present: {sorted(ds.data_vars)}"
            )

        # The sea-surface offset is a *global* far-field quantity, so it is taken
        # from the whole grid — a regional subset (e.g. the Manych corridor) has
        # no deep ocean at all and would make it undefined.
        topo_t_global = ds["Topo"].isel({time_dim: time_idx}).values.astype(np.float64)
        topo_0_global = ds["Topo"].isel({time_dim: zero_idx}).values.astype(np.float64)
        sea_surface_offset = implied_sea_surface_offset(topo_t_global, topo_0_global)

        dz_raw = solid_displacement(
            ds_t["Topo"].values.astype(np.float64),
            ds_0["Topo"].values.astype(np.float64),
            sea_surface_offset,
        )

        # Gaussian smoothing on the coarse model grid (σ=2.0 → ~36 km at
        # 10 arcmin): physically correct for long-wavelength lithospheric flexure
        # and matches GIAModel.build_correction_matrix(sigma=2.0).
        dz = gaussian_filter(dz_raw, sigma=2.0).astype(np.float32)

        ice_h = self._read_ice_thickness(ds, ds_t)

        lats_sub = ds_sub[lat_dim].values
        lons_sub = ds_sub[lon_dim].values

        return lats_sub, lons_sub, dz, ice_h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_ice_thickness(ds: xr.Dataset, ds_t: xr.Dataset) -> np.ndarray:
        """Ice-sheet thickness (m) on the epoch subset, degrading gracefully.

        Priority:
          1. a genuine thickness field (``sice`` and friends) — used as-is;
          2. an icemask (``sftgif``, percent) — a *mask* carries no thickness,
             so zeros are returned with a loud warning;
          3. nothing at all — zeros with a warning.

        Cases 2 and 3 must not kill the GIA delta: the production path
        (:mod:`paleoeurope.gia.correction_pipeline`) sources thickness from
        :class:`~paleoeurope.gia.ice7g_loader.Ice7gLoader` and discards this
        field anyway.
        """
        thickness_var = _detect_optional_var(ds, ICE_THICKNESS_VARS)
        if thickness_var is not None:
            ice_h_raw = ds_t[thickness_var].values.astype(np.float32)
            # ICE-6G_C VM5a stores ice-sheet thickness in **centimetres** in the
            # ``sice`` variable (max ~300 000 cm = 3 000 m at LGM).  Convert to
            # metres so that callers can use physically meaningful thresholds
            # (e.g. ``ice_threshold_m = 1.0`` means ≥ 1 m of ice).
            # The conversion is only applied when the raw maximum exceeds the
            # largest physically possible ice thickness in metres (~5 000 m),
            # which reliably distinguishes a cm-encoded field from a metres-
            # encoded one.
            if ice_h_raw.max() > 5_000.0:
                return ice_h_raw / 100.0   # cm → m
            return ice_h_raw               # already in metres

        mask_var = _detect_optional_var(ds, ICE_MASK_VARS)
        template = ds_t["Topo"] if "Topo" in ds_t else ds_t[list(ds_t.data_vars)[0]]
        zeros = np.zeros(template.shape, dtype=np.float32)

        if mask_var is not None:
            warnings.warn(
                f"ICE-6G file carries only the icemask '{mask_var}' (percent) and "
                f"no ice thickness variable {ICE_THICKNESS_VARS}; returning zero "
                "thickness.  Source ice thickness from Ice7gLoader instead.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"ICE-6G file has no ice thickness {ICE_THICKNESS_VARS} and no "
                f"icemask {ICE_MASK_VARS}; returning zero thickness.  The GIA "
                "delta is unaffected.",
                UserWarning,
                stacklevel=3,
            )
        return zeros

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


def _detect_optional_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    """First of *candidates* present in ``ds.data_vars``, or ``None``."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _detect_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    name = _detect_optional_var(ds, candidates)
    if name is None:
        raise KeyError(
            f"Cannot find variable; tried: {candidates}.  Available: {list(ds.data_vars)}"
        )
    return name


def _ka_to_model_time(epoch_ka: float, time_vals: np.ndarray) -> float:
    """Map a user-supplied epoch (positive ka BP) to the model's time axis."""
    # ICE-6G uses negative ka for past and 0 for present
    if time_vals.min() < 0:
        return -float(epoch_ka)
    # Some files use positive ka convention
    return float(epoch_ka)
