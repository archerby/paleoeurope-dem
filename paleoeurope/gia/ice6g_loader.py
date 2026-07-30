"""
paleoeurope.gia.ice6g_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load ICE-6G_C (VM5a) model fields from the Peltier et al. (2015) NetCDF.

Two on-disk layouts are supported, because the GIA delta needs **two** epochs and
the distribution does not ship them together:

*multi-epoch* — one NetCDF holding several epochs stacked on a ``time`` axis (ka BP,
negative = past).  ``Ice6gLoader(path)``; the 0 ka reference is the epoch nearest
zero on that axis.  ``tests/fixtures/ice6g/ICE6G_fixture.nc`` is this shape.

*per-epoch pair* — what Peltier actually distributes: ``I6_C.VM5a_10min.<epoch>.nc``,
**one epoch per file with no time dimension at all**, longitude 0..360, 1080×2160.
Use :meth:`Ice6gLoader.from_epoch_files` (two explicit paths) or
:meth:`Ice6gLoader.from_epoch_directory` (a directory + the two epochs).  There is no
sniffing and no default reference file: which epoch supplies the present-day datum is
the datum of the whole calculation, so the caller states it.  Until 2026-07-30 this
layout could not be opened at all — ``get_fields`` looked for a ``time`` dimension and
raised ``KeyError`` on every shipped file.

Longitude is normalised for the caller: bounds are given in the −180..180 convention
every raster in this project uses, and are mapped into whatever convention the file
runs, antimeridian split included.  A selection that comes out degenerate **raises**;
it used to return zero-column arrays, silently, for any bbox west of Greenwich.

Either layout must contain:

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

#: Dimension names a stacked multi-epoch file may use for its epoch axis.
TIME_DIMS = ("time", "Time", "t")

#: Filename template of the distributed per-epoch files.
ICE6G_EPOCH_TEMPLATE = "I6_C.VM5a_10min.{epoch}.nc"


def ice6g_epoch_token(epoch_ka: float) -> str:
    """Epoch as it appears in a distributed filename.

    The convention is not ``f"{epoch:g}"`` by accident — it is what is on disk:
    whole ka bare (``16`` → ``"16"``, ``0`` → ``"0"``), half-ka to exactly one
    decimal (``16.5`` → ``"16.5"``).  The suite runs 0–20.5 at 0.5 ka then 21–26 at
    1 ka, so nothing finer than a half-ka exists and anything finer is a caller
    error rather than a rounding opportunity.
    """
    value = float(epoch_ka)
    doubled = value * 2.0
    if not np.isfinite(value) or abs(doubled - round(doubled)) > 1e-9:
        raise ValueError(
            f"ICE-6G is distributed on a half-ka grid (0–20.5 at 0.5 ka, 21–26 at "
            f"1 ka); {epoch_ka!r} is not a half-ka epoch and no file is named for it."
        )
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def ice6g_epoch_filename(
    epoch_ka: float, template: str = ICE6G_EPOCH_TEMPLATE
) -> str:
    """Name of the distributed NetCDF holding *epoch_ka* alone."""
    return template.format(epoch=ice6g_epoch_token(epoch_ka))


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
        The NetCDF holding the target epoch.  A stacked multi-epoch file when
        *reference_path* is omitted; one of the distributed per-epoch files when it
        is given.
    reference_path : str or Path, optional
        The NetCDF holding the **present-day reference epoch**, for the per-epoch
        layout.  Prefer the named constructors :meth:`from_epoch_files` /
        :meth:`from_epoch_directory` over passing this by hand.
    epoch_ka : float, optional
        The epoch *nc_path* holds, when it is not recoverable from the filename.
        :meth:`get_fields` refuses a request for a different epoch.

    Examples
    --------
    Multi-epoch file (a ``time`` axis carries every epoch)::

        >>> loader = Ice6gLoader("data/raw/ice6g/ICE-6G_C_VM5a_O512.nc")
        >>> lats, lons, dz, ice_h = loader.get_fields(21, bounds=(-5, 47, 10, 60))

    The distributed per-epoch files (one epoch each, no time dimension)::

        >>> loader = Ice6gLoader.from_epoch_directory(
        ...     "data/raw/ICE6G", 16.5, reference_ka=0.0
        ... )
        >>> lats, lons, dz, ice_h = loader.get_fields(16.5, bounds=(42, 44, 46.5, 47.5))

    ``dz`` is NaN wherever the file says there was ice at the epoch or today — over
    Fennoscandia at 21 ka that is most of the window.  See :meth:`get_fields`.
    """

    def __init__(
        self,
        nc_path: str | Path,
        *,
        reference_path: str | Path | None = None,
        epoch_ka: float | None = None,
    ) -> None:
        self.nc_path = Path(nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"ICE-6G file not found: {nc_path}")

        self.reference_path = Path(reference_path) if reference_path is not None else None
        if self.reference_path is not None and not self.reference_path.exists():
            raise FileNotFoundError(
                f"ICE-6G reference-epoch file not found: {reference_path}"
            )

        if epoch_ka is None:
            epoch_ka = _epoch_from_filename(self.nc_path)
        self.epoch_ka = None if epoch_ka is None else float(epoch_ka)

        self._ds: xr.Dataset | None = None
        self._ds_ref: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Alternate constructors for the distributed per-epoch layout
    # ------------------------------------------------------------------

    @classmethod
    def from_epoch_files(
        cls,
        epoch_path: str | Path,
        reference_path: str | Path,
        *,
        epoch_ka: float | None = None,
    ) -> Ice6gLoader:
        """Two explicit files: the target epoch and the present-day reference.

        Both are required and neither is inferred.  ``Δz_solid`` is a *difference*,
        so a single per-epoch file cannot produce one, and which file stands for
        "today" is the datum the entire paleo-DEM is expressed against — guessing it
        (say, by globbing the directory and taking the numerically smallest epoch)
        would put an unstated assumption underneath every raster this project bakes.
        """
        return cls(epoch_path, reference_path=reference_path, epoch_ka=epoch_ka)

    @classmethod
    def from_epoch_directory(
        cls,
        directory: str | Path,
        epoch_ka: float,
        *,
        reference_ka: float,
        template: str = ICE6G_EPOCH_TEMPLATE,
    ) -> Ice6gLoader:
        """Resolve both per-epoch files inside *directory* by the shipped naming.

        ``reference_ka`` is keyword-only and has **no default**.  0.0 is the only
        value that makes physical sense today, but it is the present-day datum of
        the whole calculation and a caller must say so rather than inherit it from a
        default that a later edit could quietly change.
        """
        directory = Path(directory)
        epoch_path = directory / ice6g_epoch_filename(epoch_ka, template)
        reference_path = directory / ice6g_epoch_filename(reference_ka, template)
        for path, label in ((epoch_path, "epoch"), (reference_path, "reference epoch")):
            if not path.exists():
                raise FileNotFoundError(
                    f"ICE-6G {label} file not found: {path}.  The distributed files "
                    f"are named {template!r} with whole ka bare and half-ka to one "
                    "decimal (16 -> '16', 16.5 -> '16.5')."
                )
        return cls(epoch_path, reference_path=reference_path, epoch_ka=epoch_ka)

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
            Target epoch in ka BP (e.g. ``21`` for the LGM, ``0`` for modern).  In
            the per-epoch layout the loader already holds one specific epoch, and a
            request for a different one raises rather than silently answering with
            the epoch that happens to be open.
        bounds : tuple
            ``(lon_min, lat_min, lon_max, lat_max)`` in EPSG:4326 degrees, longitude
            in the **−180..180** convention this project's rasters use.  Longitudes
            are mapped into the file's own convention (the distributed files run
            0..360) and re-expressed in the requested frame on the way out, so a
            window straddling the prime meridian comes back contiguous and
            ascending.  A 1° buffer is added on every side for interpolation safety.
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
            If the file has no ``Topo`` field; if the grid holds no far-field
            ice-free tropical continental land from which to derive the sea-surface
            datum — which is the case for any **regional** file, since the datum is
            a global quantity; if *bounds* selects no cell of the grid; if a
            single-epoch file was opened without a reference epoch; or if
            *epoch_ka* disagrees with the epoch this loader holds.

        Warns
        -----
        GiaDeltaHasIceSurfaceNaNWarning
            When the returned ``delta_z`` contains ice-surface NaN.
        """
        # Both endpoints as time-free datasets, whatever the on-disk layout was.
        ds_t_global, ds_0_global = self._epoch_datasets(epoch_ka)

        lon_min, lat_min, lon_max, lat_max = bounds
        buf = 1.0  # 1° buffer for interpolation safety

        # Subset by lat/lon — handle different possible dimension names
        lat_dim = _detect_dim(ds_t_global, ("lat", "latitude", "Lat"))
        lon_dim = _detect_dim(ds_t_global, ("lon", "longitude", "Lon"))

        lat = np.asarray(ds_t_global[lat_dim].values, dtype=np.float64)
        lon = np.asarray(ds_t_global[lon_dim].values, dtype=np.float64)

        # Integer indices, not boolean masks: the longitude selection may have to
        # re-order the grid across the antimeridian, and a boolean mask cannot.
        lat_idx = _latitude_indices(lat, lat_min - buf, lat_max + buf, bounds)
        lon_idx, lons_sub = _longitude_indices(lon, lon_min - buf, lon_max + buf, bounds)
        lats_sub = lat[lat_idx]

        sel = {lat_dim: lat_idx, lon_dim: lon_idx}
        ds_t = ds_t_global.isel(sel)
        ds_0 = ds_0_global.isel(sel)

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
        for name, epoch_ds in (("epoch", ds_t_global), ("reference epoch", ds_0_global)):
            if "Topo" not in epoch_ds.data_vars:
                raise ValueError(
                    f"ICE-6G {name} file lacks the 'Topo' field required for the GIA "
                    "delta (Δz_solid = ΔTopo − median ΔTopo over far-field ice-free "
                    "tropical continental land).  Neither 'Orog' (clamped to 0 over "
                    "water) nor 'stgr'/'RSL' is an acceptable substitute.  Variables "
                    f"present: {sorted(epoch_ds.data_vars)}"
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
        topo_t_global = ds_t_global["Topo"].values.astype(np.float64)
        topo_0_global = ds_0_global["Topo"].values.astype(np.float64)
        if topo_t_global.shape != topo_0_global.shape:
            raise ValueError(
                "the epoch and reference-epoch grids differ: "
                f"{topo_t_global.shape} vs {topo_0_global.shape}.  ΔTopo is a "
                "cell-by-cell difference and cannot be taken across two grids."
            )

        # Reference-set conditions 1 and 2b, applied only when the file carries the
        # fields (``far_field_reference_mask`` degrades condition-by-condition; the
        # measured cost of omitting these is ≤0.005 m, but the full form is preferred).
        # The "never glaciated in *any* epoch" union and the 1500 km ice buffer need
        # all 48 epochs at once, which is a multi-file scan this loader does not do.
        land_var = _detect_optional_var(ds_t_global, LAND_FRACTION_VARS)
        mask_var = _detect_optional_var(ds_t_global, ICE_MASK_VARS)
        sftlf_t_global = _percent_field(ds_t_global, land_var)
        # The reference-epoch file is a separate file in the per-epoch layout, so it
        # is asked what it actually carries rather than assumed to mirror the other.
        sftlf_0_global = _percent_field(ds_0_global, _present(ds_0_global, land_var))
        sftgif_t_global = _percent_field(ds_t_global, mask_var)

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
        ice_mask_t, ice_mask_0 = self._ice_surface_endpoints(ds_t_global, ds_t, ds_0)

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

        ice_h = self._read_ice_thickness(ds_t_global, ds_t)

        return lats_sub, lons_sub, dz, ice_h

    # ------------------------------------------------------------------
    # Epoch resolution
    # ------------------------------------------------------------------

    def _epoch_datasets(self, epoch_ka: float) -> tuple[xr.Dataset, xr.Dataset]:
        """The target-epoch and reference-epoch datasets, both free of a time axis.

        Two layouts, one return shape.  The multi-epoch branch keeps behaving exactly
        as it always has (nearest epoch on the ``time`` axis; the epoch nearest zero
        as the reference); the per-epoch branch uses the two files it was handed.
        """
        ds = self._open()

        if self.reference_path is not None:
            self._check_requested_epoch(epoch_ka)
            return (
                _squeeze_time(ds, self.nc_path),
                _squeeze_time(self._open_reference(), self.reference_path),
            )

        time_dim = _detect_optional_dim(ds, TIME_DIMS)
        if time_dim is None:
            raise ValueError(
                f"{self.nc_path.name} has no epoch axis {TIME_DIMS} — it holds a "
                "single epoch, which is the layout Peltier distributes "
                "('I6_C.VM5a_10min.<epoch>.nc', one file per epoch).  Δz_solid is a "
                "difference between two epochs and cannot be taken from one file.  "
                "Use Ice6gLoader.from_epoch_files(epoch_path, reference_path) or "
                "Ice6gLoader.from_epoch_directory(dir, epoch_ka, reference_ka=0.0).  "
                f"Dimensions present: {sorted(ds.sizes)}"
            )

        time_vals = np.asarray(ds[time_dim].values)
        target_time = _ka_to_model_time(epoch_ka, time_vals)
        time_idx = int(np.argmin(np.abs(time_vals - target_time)))
        zero_idx = int(np.argmin(np.abs(time_vals)))
        return ds.isel({time_dim: time_idx}), ds.isel({time_dim: zero_idx})

    def _check_requested_epoch(self, epoch_ka: float) -> None:
        """Refuse a request for an epoch this loader does not hold.

        In the per-epoch layout the file *is* the epoch.  Answering a request for
        21 ka out of the 16.5 ka file — which is what silently ignoring ``epoch_ka``
        would do — mislabels every raster downstream with an epoch it is not.
        """
        if self.epoch_ka is None:
            return
        if abs(float(epoch_ka) - self.epoch_ka) > 1e-9:
            raise ValueError(
                f"this loader holds epoch {self.epoch_ka} ka ({self.nc_path.name}) "
                f"but {epoch_ka} ka was requested.  In the per-epoch layout one "
                "loader serves one epoch; construct another with "
                "Ice6gLoader.from_epoch_directory(dir, "
                f"{epoch_ka}, reference_ka=0.0)."
            )

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
            # In the per-epoch layout ``ds_0`` is a different file; if it does not
            # carry the mask, that endpoint is left unmasked (``None``) rather than
            # assumed ice-free — ``ice_surface_mask`` keeps the other endpoint's.
            pct_t = _percent_field(ds_t, _present(ds_t, mask_var))
            pct_0 = _percent_field(ds_0, _present(ds_0, mask_var))
            return (
                None if pct_t is None else pct_t > ICE_PERCENT_THRESHOLD,
                None if pct_0 is None else pct_0 > ICE_PERCENT_THRESHOLD,
            )

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

    def _open_reference(self) -> xr.Dataset:
        if self.reference_path is None:
            raise ValueError("no reference-epoch file was given")
        if self._ds_ref is None:
            self._ds_ref = xr.open_dataset(self.reference_path, engine="netcdf4")
        return self._ds_ref

    def close(self) -> None:
        """Release the open NetCDF file handle(s)."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None
        if self._ds_ref is not None:
            self._ds_ref.close()
            self._ds_ref = None


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------


def _detect_optional_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    """First of *candidates* present as a dimension or coordinate, or ``None``."""
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name
    return None


def _detect_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    name = _detect_optional_dim(ds, candidates)
    if name is None:
        raise KeyError(
            f"Cannot find dimension; tried: {candidates}.  Available: {list(ds.dims)}"
        )
    return name


def _present(ds: xr.Dataset, var: str | None) -> str | None:
    """*var* if this dataset actually carries it, else ``None``."""
    return var if var is not None and var in ds.data_vars else None


def _squeeze_time(ds: xr.Dataset, path: Path) -> xr.Dataset:
    """Drop a length-1 epoch axis; refuse a longer one.

    The distributed per-epoch files carry no time dimension at all, but a
    re-exported copy may keep it as a singleton.  A file with *several* epochs is
    not a per-epoch file, and picking one of them here would be exactly the guess
    the epoch-pair API exists to avoid.
    """
    time_dim = _detect_optional_dim(ds, TIME_DIMS)
    if time_dim is None or time_dim not in ds.sizes:
        return ds
    n = int(ds.sizes[time_dim])
    if n == 1:
        return ds.isel({time_dim: 0})
    raise ValueError(
        f"{path.name} carries {n} epochs on '{time_dim}', so it is not a per-epoch "
        "file.  Open it directly — Ice6gLoader(path) — and let the time axis select "
        "the epoch, instead of pairing it with a separate reference file."
    )


def _epoch_from_filename(path: Path) -> float | None:
    """Epoch in ka BP encoded in a distributed filename, or ``None``.

    Matches ``<anything>.<epoch>.nc`` where ``<epoch>`` is a whole or half ka, i.e.
    exactly the shipped ``I6_C.VM5a_10min.16.5.nc`` convention.  Anything else
    returns ``None`` and the loader simply does not epoch-check that file, rather
    than inventing an epoch from a name it does not recognise.
    """
    stem = Path(path).stem  # 'I6_C.VM5a_10min.16.5'
    token = stem.rsplit(".", 1)[-1] if "." in stem else stem
    try:
        value = float(token)
    except ValueError:
        return None
    doubled = value * 2.0
    if abs(doubled - round(doubled)) > 1e-9:
        return None
    return value


def _latitude_indices(
    lat: np.ndarray,
    lat_min: float,
    lat_max: float,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Indices of the rows inside ``[lat_min, lat_max]``; raise if there are none."""
    idx = np.nonzero((lat >= lat_min) & (lat <= lat_max))[0]
    if idx.size == 0:
        raise ValueError(
            f"bounds {bounds} select no latitude row of this ICE-6G grid "
            f"(file covers {lat.min():.4f}..{lat.max():.4f} °N).  Returning an empty "
            "window instead would hand the caller a zero-row array with no error."
        )
    return idx


def _longitude_indices(
    lon: np.ndarray,
    lon_min: float,
    lon_max: float,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap-aware column selection.  Returns ``(indices, longitudes)``.

    The distributed ICE-6G files run ``lon 0..360``; every raster in this project
    runs −180..180.  A raw ``(lon >= lon_min) & (lon <= lon_max)`` against a negative
    request therefore selected the wrong columns and said nothing about it: an
    Ireland-shaped bbox (−11..−4) matched **nothing** and produced zero-column
    arrays, while a Doggerland-shaped one (−4..8) matched only its eastern half and
    quietly dropped everything west of Greenwich.

    Each file longitude is re-expressed in the half-open 360° window that starts at
    *lon_min*, which is convention-agnostic — it works for a 0..360 file, a
    −180..180 file, and a request that crosses either seam.  The selected columns are
    then ordered by that re-expressed longitude, so the returned window is contiguous
    and ascending even where it spans the file's antimeridian, and the longitudes
    handed back are in the caller's own frame (negative west of Greenwich).

    Unwrapped values are returned bit-identical to the file's, so a window that never
    crosses a seam is unchanged by any of this.
    """
    if lon_max < lon_min:
        raise ValueError(
            f"bounds {bounds} have lon_max < lon_min; a longitude range that wraps "
            "must be written as an increasing pair (e.g. 170 -> 190, not 170 -> -170)."
        )

    shifted = lon_min + np.mod(lon - lon_min, 360.0)
    # np.mod on floats can land a whole multiple of 360 away from the original; snap
    # the unwrapped ones back so a non-wrapping window is byte-for-byte as on disk.
    shifted = np.where(np.abs(shifted - lon) < 1e-9, lon, shifted)

    idx = np.nonzero(shifted <= lon_max)[0]
    if idx.size == 0:
        raise ValueError(
            f"bounds {bounds} select no longitude column of this ICE-6G grid "
            f"(file covers {lon.min():.4f}..{lon.max():.4f} °E).  Returning an empty "
            "window instead would hand the caller a zero-column array with no error."
        )

    order = np.argsort(shifted[idx], kind="stable")
    idx = idx[order]
    return idx, shifted[idx]


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
