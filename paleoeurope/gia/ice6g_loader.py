"""
paleoeurope.gia.ice6g_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load ICE-6G_C (VM5a) model fields from the Peltier et al. (2015) NetCDF.

The NetCDF is expected to hold several epochs stacked on a ``time`` axis
(ka BP, negative = past, typically 0 to −26 ka) and must contain:

- ``Topo`` — topography/bathymetry (m) relative to the *contemporaneous* sea
  surface.  This is the only field the GIA delta is derived from; see
  :mod:`paleoeurope.gia.ice6g_topo`.

Optionally, and all of them are *used* when present:

- ``sice``/``iceheight``/… — ice-sheet thickness (m or cm)
- ``sftgif`` — icemask (percent, 100 = ice); the real
  ``I6_C.VM5a_10min.<epoch>.nc`` files carry this and *no* thickness field.
  It is not only reported: it drives the **ice-surface guard**, because where
  ``sftgif > 0`` ``Topo`` is the ice surface and not the solid surface, so
  ``Δz_solid`` there is returned as **NaN** rather than as a ~+1700 m ice step.
- ``sftlf`` — land fraction (percent); tightens the far-field reference set from
  which the sea-surface datum is taken.

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

from paleoeurope.gia.ice6g_topo import (
    ICE_PERCENT_THRESHOLD,
    implied_sea_surface_offset,
    smooth_preserving_ice_mask,
    solid_displacement,
)

#: Genuine ice-*thickness* variable names (metres or centimetres).
ICE_THICKNESS_VARS = ("sice", "iceheight", "ice", "thick", "H_ice", "hice")

#: Ice *mask* variable names (percent, 100 = ice).  The distributed
#: ``I6_C.VM5a_10min.<epoch>.nc`` files carry ``sftgif`` and no thickness at all.
ICE_MASK_VARS = ("sftgif", "icemask", "sftgrf")

#: Land-fraction variable names (percent, 100 = land).  Only used to tighten the
#: far-field reference set behind the sea-surface datum.
LAND_FRACTION_VARS = ("sftlf", "landmask", "landfrac")


class GiaDeltaHasIceSurfaceNaNWarning(UserWarning):
    """The returned ``Δz_solid`` contains NaN because ICE-6G ``Topo`` is ice there.

    Not a defect and not a nodata artefact: over cells glaciated at the epoch or
    today, ``Topo`` is the **ice surface**, so no solid-surface displacement exists
    in these files at all (they ship no bedrock-under-ice field).  A consumer must
    either restrict itself to the finite cells or source Δz from ICE-7G's
    bedrock-deformation field.  ``np.nan_to_num`` on this array writes "no GIA here"
    where the truth is "unknown", which reads as a valid 0 m and silently
    under-corrects formerly glaciated ground.  Filter this class deliberately.
    """


class Ice6gLoader:
    """Load ICE-6G_C VM5a deformation/ice fields for a bounding box and epoch.

    Parameters
    ----------
    nc_path : str or Path
        Path to the ``ICE-6G_C_VM5a_O512.nc`` file (or equivalent).

    Examples
    --------
    >>> loader = Ice6gLoader("data/raw/ice6g/ICE-6G_C_VM5a_O512.nc")
    >>> lats, lons, dz, ice_h = loader.get_fields(epoch_ka=21, bounds=(-5, 47, 10, 60))
    >>> dz.shape  # depends on model resolution
    (26, 30)

    ``dz`` is NaN wherever the file says there was ice at 21 ka or today — over
    Fennoscandia at 21 ka that is most of the window.  See :meth:`get_fields`.
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
        *,
        sea_surface_offset: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return GIA deformation and ice thickness for one epoch and bbox.

        Parameters
        ----------
        epoch_ka : float
            Target epoch in ka BP (e.g. ``21`` for the LGM, ``0`` for modern).
        bounds : tuple
            ``(lon_min, lat_min, lon_max, lat_max)`` in EPSG:4326 degrees.
        sea_surface_offset : float, optional
            Pre-measured sea-surface datum for this epoch, in metres.  Only for
            files that are a **regional clip**: the datum is a global far-field
            quantity, so a window with no ice-free tropical continental land
            cannot derive it and :func:`implied_sea_surface_offset` refuses.
            Supply a value measured once from the *global* file for the same
            epoch (e.g. ``+87.33`` m at 21 ka, ``+77.91`` m at 16.5 ka).  Leave
            it ``None`` for a global file — deriving it is always preferable to
            asserting it, and the refusal is what stops a regional cut from
            improvising a datum the way the retired deep-ocean estimator did.

        Returns
        -------
        lats : np.ndarray  shape (M,)
        lons : np.ndarray  shape (N,)
        delta_z : np.ndarray  shape (M, N)
            Present-datum solid-earth vertical displacement ``Δz_solid`` at
            *epoch_ka* (metres), Gaussian-smoothed with σ=2.0 on the model grid.
            Positive = the solid surface stood higher than today.

            **May contain NaN**, and the NaN is load-bearing: it marks cells that
            are glaciated at *epoch_ka* or today, where ICE-6G ``Topo`` is the ice
            surface rather than the solid surface (see
            :func:`paleoeurope.gia.ice6g_topo.solid_displacement`).  Masking happens
            *before* the Gaussian, so ice-surface magnitudes do not leak across the
            margin.  Callers must not ``nan_to_num`` this array — see
            :class:`GiaDeltaHasIceSurfaceNaNWarning`.  If the file carries no ice
            field at all the guard cannot be applied and
            :class:`~paleoeurope.gia.ice6g_topo.IceSurfaceUnmaskedWarning` is raised
            from ``solid_displacement`` instead.
        ice_height : np.ndarray  shape (M, N)
            Ice-sheet thickness (m) at *epoch_ka*.  0 where ice-free — and 0
            everywhere, with a warning, when the file carries only an icemask
            (``sftgif``) or no ice field at all.

        Raises
        ------
        ValueError
            If the file has no ``Topo`` field, or if the grid holds no far-field
            ice-free tropical continental land from which to derive the sea-surface
            datum — which is the case for any **regional** file, since the datum is
            a global quantity.

        Warns
        -----
        GiaDeltaHasIceSurfaceNaNWarning
            When the returned ``delta_z`` contains ice-surface NaN.
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
        # Δz_solid = [Topo(t) − Topo(0)] − sea_surface_offset(t), where
        # ``sea_surface_offset`` is the median ΔTopo over **far-field ice-free
        # tropical continental land** — |lat| ≤ 25°, topo_0 > 0, land at both
        # endpoints, never glaciated in the whole suite, ≥ 1500 km from any
        # ever-glaciated cell.  See :mod:`paleoeurope.gia.ice6g_topo`.
        #
        # It is emphatically NOT the deep-ocean median, which is what this module
        # used until 2026-07-30.  Relieved of ~120 m of water at the LGM the ocean
        # floor stands ~+30 m HIGHER than today (hydro-isostasy, ≈0.3 of the
        # far-field sea-surface change), so a deep-ocean datum set that rebound to
        # zero by fiat and pushed it onto the continents **with the sign flipped**:
        # every paleo-DEM built through here carried a spurious continental
        # lowering growing to −31.8 m at 23 ka.
        #
        # ``Orog(t) − Orog(0)`` is NOT an option either: ICE-6G ``Orog`` is clamped
        # to 0 over water, so ΔOrog vanishes (or flips sign) exactly under a paleo-
        # shoreline.  Measured over the Manych spillway at 16.5 ka the correct
        # Δz_solid is **+2.72 m** at the sill cell (modern elevation +23.00 m →
        # palaeo-sill +25.72 m a.s.l., ±1–2 m near-field) — mild UPLIFT, so the
        # corrected spill is *harder* than a modern-sill reading, not easier.
        # Re-measured 2026-07-30 on I6_C.VM5a_10min.{16.5,0}.nc: +2.7010 m at the
        # 10 arcmin cell nearest 44.119 °E / 45.773 °N, sea-surface datum 77.9058 m.
        # ΔOrog there is +80.6 m, enough to lift the sill out of reach and make the
        # Khvalynian spill west impossible at any Caspian level, silently.
        #
        # The −26.5 m that this comment block asserted as "the correct Δz_solid"
        # before 2026-07-30 was the **deep-ocean-datum artefact — the bug's own
        # output**, not a measurement.  Do not reinstate or re-derive it: it made
        # the Khvalynian spill look spuriously trivial by placing the palaeo-sill at
        # −3.52 m a.s.l., 26.5 m below the modern sill.
        #
        # ``stgr`` (= bare ΔTopo) is not an option either: it double-counts the
        # ~112 m eustatic sea-surface change baked into ``Topo``.  No branch is kept
        # behind a flag.
        if "Topo" not in ds.data_vars:
            raise ValueError(
                "ICE-6G file lacks the 'Topo' field required for the GIA delta "
                "(Δz_solid = ΔTopo − median ΔTopo over far-field ice-free tropical "
                "continental land).  Neither 'Orog' (clamped to 0 over water) nor "
                "'stgr'/'RSL' is an acceptable substitute.  Variables present: "
                f"{sorted(ds.data_vars)}"
            )

        # The sea-surface offset is a *global* far-field quantity, so it is taken
        # from the whole grid — a regional subset (e.g. the Manych corridor) holds
        # no tropical far field at all and would make it undefined.
        #
        # ``lat=`` is passed explicitly: ``implied_sea_surface_offset`` can infer
        # latitudes from the array shape, but that is a fallback for callers holding
        # bare arrays.  This one has the real coordinate vector in hand, so it must
        # not make the estimator guess (and a file whose grid is global in extent but
        # not exactly ``n_lon == 2·n_lat`` would otherwise be refused for no reason).
        topo_t_global = ds["Topo"].isel({time_dim: time_idx}).values.astype(np.float64)
        topo_0_global = ds["Topo"].isel({time_dim: zero_idx}).values.astype(np.float64)

        # Reference-set conditions 1 and 2b, applied only when the file carries the
        # fields (``far_field_reference_mask`` degrades condition-by-condition; the
        # measured cost of omitting these is ≤0.005 m, but the full form is preferred).
        # The "never glaciated in *any* epoch" union and the 1500 km ice buffer need
        # all 48 epochs at once, which is a multi-file scan this loader does not do.
        land_var = _detect_optional_var(ds, LAND_FRACTION_VARS)
        mask_var = _detect_optional_var(ds, ICE_MASK_VARS)
        sftlf_t_global = _percent_field(ds, land_var, time_dim, time_idx)
        sftlf_0_global = _percent_field(ds, land_var, time_dim, zero_idx)
        sftgif_t_global = _percent_field(ds, mask_var, time_dim, time_idx)

        if sea_surface_offset is None:
            sea_surface_offset = implied_sea_surface_offset(
                topo_t_global,
                topo_0_global,
                lat=lat,
                sftlf_t=sftlf_t_global,
                sftlf_0=sftlf_0_global,
                sftgif_t=sftgif_t_global,
            )
        else:
            # Regional-clip escape hatch.  A cut that holds no tropical far field
            # cannot derive the datum, and this project makes such cuts routinely
            # (the Manych corridor, the Black Sea window, the Norway test tile).
            # The caller supplies a value measured ONCE from the global file for
            # the same epoch; it is not a free parameter and it is not a fallback
            # that fires by itself — omitting it still raises, loudly, which is
            # what keeps a regional window from silently improvising a datum.
            sea_surface_offset = float(sea_surface_offset)

        # Where the file tells us there is ice — at *either* endpoint — ``Topo`` is
        # the ice surface, so ``solid_displacement`` returns NaN there instead of an
        # ice-column step of up to +3506 m that ``correction_pipeline`` would then add
        # ICE-7G thickness on top of, counting the ice twice.  Passing these masks is
        # what turns the guard on; without them ``solid_displacement`` only warns.
        ice_mask_t, ice_mask_0 = self._ice_surface_endpoints(ds, ds_t, ds_0)

        dz_raw = solid_displacement(
            ds_t["Topo"].values.astype(np.float64),
            ds_0["Topo"].values.astype(np.float64),
            sea_surface_offset,
            ice_mask_t=ice_mask_t,
            ice_mask_0=ice_mask_0,
        )

        # Gaussian smoothing on the coarse model grid (σ=2.0 → ~36 km at
        # 10 arcmin): physically correct for long-wavelength lithospheric flexure
        # and matches GIAModel.build_correction_matrix(sigma=2.0).
        #
        # ``smooth_preserving_ice_mask``, not a bare ``gaussian_filter``: the masked
        # cells must neither swallow their ice-free neighbours (a plain filter
        # propagates NaN over the whole 8-cell kernel support) nor — had the mask been
        # applied after smoothing — smear ice-surface magnitudes outward, measured at
        # >100 m three cells past the margin and >0.2 m out to eight.
        # ``mode="reflect"`` reproduces scipy's default, so on an ice-free window the
        # result is bit-for-bit the previous ``gaussian_filter(dz_raw, sigma=2.0)``.
        dz = smooth_preserving_ice_mask(dz_raw, 2.0, mode="reflect").astype(np.float32)

        # NaN is propagated to the caller as-is.  It is the honest answer: these files
        # carry no bedrock under ice, so there is no number to fill in.  Warn loudly,
        # because a downstream ``nan_to_num`` would turn "unknown" into a valid-looking
        # 0 m of GIA — see GiaDeltaHasIceSurfaceNaNWarning.
        n_nan = int(np.count_nonzero(~np.isfinite(dz)))
        if n_nan:
            warnings.warn(
                f"GIA delta for epoch {epoch_ka} ka over bounds {bounds} contains "
                f"{n_nan} NaN cells ({100.0 * n_nan / dz.size:.1f}% of the window) "
                "where ICE-6G 'Topo' is the ICE SURFACE and no bedrock is available "
                "in these files.  Restrict to the finite cells or take Δz from "
                "ICE-7G's bedrock-deformation field; do NOT nan_to_num this array, "
                "which would record 0 m of GIA over formerly glaciated ground.",
                GiaDeltaHasIceSurfaceNaNWarning,
                stacklevel=2,
            )

        ice_h = self._read_ice_thickness(ds, ds_t)

        lats_sub = ds_sub[lat_dim].values
        lons_sub = ds_sub[lon_dim].values

        return lats_sub, lons_sub, dz, ice_h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ice_surface_endpoints(
        ds: xr.Dataset, ds_t: xr.Dataset, ds_0: xr.Dataset
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Boolean "``Topo`` is an ice surface here" masks for both endpoints.

        Either endpoint being glaciated makes ``Topo(t) − Topo(0)`` meaningless, so
        both are needed.  Priority is the **opposite** of :meth:`_read_ice_thickness`,
        deliberately: that method wants a magnitude, this one wants the extent, and
        ``sftgif`` is ICE-6G's own statement of where its ``Topo`` switched to the ice
        surface — a thickness field may come from a different model or grid.

          1. an icemask (``sftgif``, percent) — what the real
             ``I6_C.VM5a_10min.<epoch>.nc`` files carry;
          2. a thickness field (``sice`` and friends) — any positive thickness means
             ``Topo`` is the ice surface there, whatever the units;
          3. nothing → ``(None, None)``, and :func:`solid_displacement` then emits
             :class:`~paleoeurope.gia.ice6g_topo.IceSurfaceUnmaskedWarning` rather
             than pretending the field is bedrock.

        Returned as booleans (not percent) so that the same code path serves both
        conventions; ``solid_displacement`` accepts either.
        """
        mask_var = _detect_optional_var(ds, ICE_MASK_VARS)
        if mask_var is not None:
            pct_t = _percent_field(ds_t, mask_var)
            pct_0 = _percent_field(ds_0, mask_var)
            return pct_t > ICE_PERCENT_THRESHOLD, pct_0 > ICE_PERCENT_THRESHOLD

        thickness_var = _detect_optional_var(ds, ICE_THICKNESS_VARS)
        if thickness_var is not None:
            thick_t = np.asarray(ds_t[thickness_var].values, dtype=np.float64)
            thick_0 = np.asarray(ds_0[thickness_var].values, dtype=np.float64)
            return thick_t > 0.0, thick_0 > 0.0

        return None, None

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


def _percent_field(
    ds: xr.Dataset,
    var: str | None,
    time_dim: str | None = None,
    time_idx: int | None = None,
) -> np.ndarray | None:
    """A 0–100 percent field (``sftlf``/``sftgif``) as float64, or ``None``.

    ``None`` in, ``None`` out, so a caller can hand the result straight to the
    optional ``sftlf_t=``/``sftgif_t=`` parameters of
    :func:`~paleoeurope.gia.ice6g_topo.implied_sea_surface_offset` and have the
    corresponding reference-set condition simply not be applied.

    A field encoded as a 0–1 *fraction* is scaled to percent: the shipped ICE-6G
    files are binary 0/100, but the 50 % thresholds downstream would silently
    exclude everything on a fraction-encoded file, which for ``sftgif`` would mean
    disabling the ice-surface guard.  The test is ``max ≤ 1.5``; a percent field
    whose window genuinely never exceeds 1 % is indistinguishable, and scaling it is
    harmless where it is 0 and conservative (more ice, more masking) where it is not.
    """
    if var is None:
        return None
    da = ds[var]
    if time_dim is not None and time_dim in da.dims:
        da = da.isel({time_dim: time_idx})
    arr = np.asarray(da.values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.max()) <= 1.5:
        return arr * 100.0
    return arr


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
