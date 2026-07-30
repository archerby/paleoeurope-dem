"""
paleoeurope.gia.ice6g_topo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert ICE-6G_C ``Topo`` fields into a present-day-datum solid-earth vertical
displacement, for use as the GIA delta in :func:`paleoeurope.gia.deformation.apply_gia_delta`.

Sign convention (unchanged, and the rest of the engine depends on it)::

    H_paleo = H_modern + Δz_solid          Δz_solid > 0 → the solid surface stood
                                           HIGHER than it does today.

Why not ``Orog``?
-----------------
ICE-6G ``Orog`` is clamped to 0 over water, so ``Orog(t) − Orog(0)`` vanishes (or
flips sign) exactly under a paleo-shoreline — precisely where the interesting line
is drawn.  ``Topo`` carries bathymetry, but it is expressed **relative to the
contemporaneous (epoch) sea surface**, so a bare ``ΔTopo`` double-counts the
eustatic sea-surface change.  One scalar per epoch has to be removed::

    Δz_solid(t) = [Topo(t) − Topo(0)] − sea_surface_offset(t)

The whole question is *which cells define* ``sea_surface_offset``.

THE REFERENCE SET, AND WHY IT IS NOT THE DEEP OCEAN
---------------------------------------------------
Until 2026-07-30 this module used ``median(ΔTopo)`` over **global deep ocean**
(``topo_0 < −1000 m``).  That is wrong, and wrong by ~30 m at the LGM.

Deep ocean is not a zero-displacement reference frame.  Relieved of ~120 m of
water at the LGM, the ocean floor stands **higher** than today — measured on these
files, the median Δz_solid over deep ocean is +31.5 m at 21 ka, +29.2 m at 16.5 ka,
+12.0 m at 10 ka, ≈ 0.24 × the sea-surface change, which is the expected
hydro-isostatic ratio.  Taking the datum from those cells sets that number to zero
by fiat and pushes it onto the continents **with the sign flipped**: every
paleo-DEM built through this module carried a spurious continental *lowering*
growing from 0 m at 0 ka to −31.8 m at 23 ka.  At the Manych spillway the buggy
engine placed the palaeo-sill at −3.5 m a.s.l. at 16.5 ka instead of +25.7 m.

The reference set used here instead is **far-field, ice-free, tropical continental
land**, on the native ICE-6G grid, all five conditions:

1. **Land at both endpoints** — ``sftlf(t) > 50`` and ``sftlf(0) > 50``.
   (``sftlf``/``sftgif`` are binary 0/100 percent in these files.)
2. **Never glaciated anywhere in the suite** — the *union* of ``sftgif > 50`` over
   all 48 epochs is excluded (429 737 cells, 18.4 % of the grid), not merely the ice
   at epoch *t*: a deglaciated cell is still relaxing and is not a zero reference.
3. **Low latitude** — ``|lat| ≤ 25°``.
4. **Genuinely continental today** — ``topo_0 > 0``, so present-day shelf, which
   sits in the hydro-isostatic transition zone, is excluded.
5. **Ice buffer** — ``≥ 1500 km`` from any ever-glaciated cell (longitudinally
   wrapped EDT, so the dateline is not a false barrier).

Over stable tropical continental interiors Δz_solid ≈ 0 genuinely holds, so
``median(ΔTopo)`` there *is* the sea-surface change and nothing else.  Yield:
159 790 cells at 16.5 ka out of 2 332 800, IQR of ΔTopo across them 2.1 m.

RESIDUAL UNCERTAINTY — DO NOT OVERCLAIM
---------------------------------------
* **±1 m systematic on the offset.**  Independent continental blocks disagree by
  2.06 m at 16.5 ka (Africa 77.79, Australia 77.24, S America 77.58, India/SE Asia
  79.30 m).  That spread, not the median's standard error, is the honest bar.
* **±1–2 m near-field, and it is the one that matters here.**  A single scalar
  offset assumes the sea-surface/geoid change ΔW is spatially uniform.  It is not.
  The 25–45 °N band — the same distance class as the Manych sill, 1560 km from the
  nearest ever-glaciated cell — gives an apparent offset 2.1 m above the tropics.
  Part of that is genuine forebulge Δz_solid we want to keep and part is geoid ΔW we
  want to remove, and ``sftlf``/``sftgif``/``Topo``/``Orog`` cannot separate them.
  The shipped 10 arcmin distribution carries no sea-level or geoid field, so this
  cannot be resolved from these files.  Since the Khvalynian spill question turns
  on a −3.79 m margin, ±1–2 m must be quoted wherever the corrected sill is.
* **Temporal aliasing at 14.5 → 13.5 ka.**  ICE-6G_C puts ~20.4 m of far-field
  sea-surface rise into the single 14.5 → 14.0 ka step (MWP-1A), ~6× the local rate.
  The corrected sill shows a non-monotone +2.4 m excursion at 14.0 ka that partly
  retreats by 13.5 ka.  That is model temporal resolution, not a datable event;
  interpolate through the interval rather than stepping, and prefer the 13.5 ka row.
* **A late-Holocene reversal is real.**  The offset is *not* monotone all the way to
  0 ka: it bottoms out at −3.54 m at 6.5 ka and returns to 0.000 m at 0 ka.  That
  is the far-field mid-Holocene highstand (ocean siphoning / continental levering),
  not an artefact.
* **25 and 26 ka are undiagnosed.**  Local Δz_solid at Manych there is −0.8 and
  −5.0 m against a stable +1.0…+1.4 m through 24–19 ka.  Possibly a genuine
  early-glacial signal, possibly an ICE-6G_C spin-up edge effect.  Do not quote them.

THE ICE-SURFACE TRAP
--------------------
Where ``sftgif > 0``, ICE-6G ``Topo`` is the **ice surface, not bedrock** — verified
directly: at the Byrd Subglacial Basin ``Topo(0)`` = +1514 m where the bedrock is
about −1500 m, and ``Topo == Orog`` exactly at every present-day ice point.  There
is **no bedrock-under-ice field in the shipped 10 arcmin files at all** (the only
data variables are ``sftlf``, ``sftgif``, ``Topo``, ``Orog``), so bedrock is not
recoverable here.  Over cells glaciated at *t* but ice-free today the unmasked
Δz_solid reads a median of +1141 m at 21 ka and up to +3506 m; at 62 °N / 15 °E it
reads +1711 m, and :mod:`paleoeurope.gia.correction_pipeline` would then *add*
ICE-7G thickness on top of that, counting the ice column twice.

So :func:`solid_displacement` returns **NaN** over ice, given the masks, and *warns*
(:class:`IceSurfaceUnmaskedWarning`) when it is called without them.  Ice-free
ground under formerly glaciated ice needs Δz from ICE-7G's bedrock-deformation
field instead; this module will not invent it.

Smoothing must come **after** masking: a plain ``gaussian_filter(σ=2)`` over an
unmasked field leaks the ~+1800 m ice-margin step outward, measured on the real
files at 21 ka, by >100 m three cells past the margin, >1 m out to six or seven
cells and >0.2 m out to eight.  Use :func:`smooth_preserving_ice_mask`.

Peltier et al. (2015) https://doi.org/10.1002/2014JB011176
Data: https://www.physics.utoronto.ca/~peltier/data.html
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

__all__ = [
    "FAR_FIELD_LAT_BAND_DEG",
    "FAR_FIELD_ICE_BUFFER_KM",
    "MIN_REFERENCE_CELLS",
    "MIN_REFERENCE_FRACTION",
    "MIN_REFERENCE_CELLS_FLOOR",
    "minimum_reference_cells",
    "LAND_PERCENT_THRESHOLD",
    "ICE_PERCENT_THRESHOLD",
    "ICE6G_10MIN_CELL_KM",
    "SMOOTHING_RIM_CELLS",
    "IceSurfaceUnmaskedWarning",
    "global_latitude_grid",
    "ever_glaciated_mask",
    "distance_to_ice_km",
    "far_field_reference_mask",
    "implied_sea_surface_offset",
    "ice_surface_mask",
    "solid_displacement",
    "smooth_preserving_ice_mask",
]

#: Latitude half-band of the far-field reference set.  Sensitivity: sweeping
#: 10°→40° moves the 16.5 ka offset by only 0.62 m (29.501 → 28.883), drifting
#: monotonically toward zero as the forebulge leaks in — the expected direction.
FAR_FIELD_LAT_BAND_DEG = 25.0

#: Minimum great-circle-ish distance from any ever-glaciated cell.  Measured, this
#: is a near no-op for the tropical band (which is already 4346 km from ice on the
#: median: 0/500/1000/1500/2000 km all give −29.232 m at 16.5 ka).  It is kept
#: because it makes the criterion self-documenting and becomes load-bearing the
#: moment anyone widens :data:`FAR_FIELD_LAT_BAND_DEG`.
FAR_FIELD_ICE_BUFFER_KM = 1500.0

#: Refuse to derive a global datum from fewer cells than this.  A regional window
#: (the Manych corridor, a North Sea fixture) has no tropical far field at all and
#: must raise rather than improvise — the old estimator silently accepted any grid
#: that happened to contain one deep cell.
#:
#: The requirement is *coverage*, not a raw count, so it scales with the grid.  An
#: absolute 10 000 was a hidden dependency on the 10-arcmin shipped resolution: a
#: legitimately global 2° grid has only ~4 500 cells in the whole |lat| <= 25 band,
#: so an absolute floor at 10 000 refused every coarse global field regardless of
#: how much tropical land it carried — while a 0.25° regional cut with 10 000 cells
#: of ocean would have sailed through.  Cell count is the wrong axis.
#:
#: :data:`MIN_REFERENCE_FRACTION` reproduces the old strictness where it was
#: calibrated: 0.005 x 2 332 800 (1080x2160) = 11 664, i.e. slightly tighter than
#: the 10 000 it replaces.  :data:`MIN_REFERENCE_CELLS_FLOOR` keeps a small
#: absolute floor so a degenerate handful of cells can never define a datum.
#: A grid with no tropical ice-free land still raises, which is the invariant that
#: matters.
MIN_REFERENCE_FRACTION = 0.005
MIN_REFERENCE_CELLS_FLOOR = 100

#: Retained for callers that pin an explicit count.  Prefer letting
#: :func:`minimum_reference_cells` scale it to the grid.
MIN_REFERENCE_CELLS = 10_000


def minimum_reference_cells(n_total: int) -> int:
    """Reference-set minimum for a grid of *n_total* cells.

    ``max(MIN_REFERENCE_CELLS_FLOOR, MIN_REFERENCE_FRACTION * n_total)`` — see
    :data:`MIN_REFERENCE_FRACTION` for why this is a fraction and not a count.
    """
    return max(MIN_REFERENCE_CELLS_FLOOR, int(MIN_REFERENCE_FRACTION * int(n_total)))

#: ``sftlf``/``sftgif`` are binary 0/100 percent in the shipped files; 50 splits them.
LAND_PERCENT_THRESHOLD = 50.0
ICE_PERCENT_THRESHOLD = 50.0

#: Nominal cell size of the 10 arcmin grid, used for the ice-distance transform.
#: Longitude spacing is treated as its equatorial value, which OVER-estimates purely
#: zonal distances at high latitude.  For a tropical target and polar ice the path is
#: near-meridional so the approximation is tight, and where it is loose it claims
#: more separation than exists — hence the latitude-band sensitivity cross-check.
ICE6G_10MIN_CELL_KM = 18.53

#: Cells beyond an ice margin that a σ=2 Gaussian can reach: ``ceil(truncate·σ)`` = 8
#: for scipy's default ``truncate=4.0``.  Measured contamination of a naive filter
#: over an unmasked field exceeds 1 m out to 6–7 cells and 0.2 m out to 8.
SMOOTHING_RIM_CELLS = 8


class IceSurfaceUnmaskedWarning(UserWarning):
    """``solid_displacement`` was called with no ice information.

    Any cell glaciated at epoch *t* or today then carries the **ice surface**, not
    bedrock, in the returned array.  Pass ``sftgif_t``/``sftgif_0`` (or explicit
    boolean masks) to get NaN there instead.  Filter or escalate this class
    deliberately — do not blanket-ignore it.
    """


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def global_latitude_grid(shape: tuple[int, int]) -> np.ndarray:
    """Latitude of every cell centre of a **global** cell-centred lat/lon grid.

    Only used when a caller supplies bare ``Topo`` arrays with no coordinates (the
    two production call sites do exactly that, both with the full global field).
    Accepted only when ``n_lon == 2 · n_lat`` and the grid is not tiny, which the
    real 10 arcmin distribution satisfies (1080 × 2160, cell centres −89.9167 …
    +89.9167).  Only ``|lat|`` is ever used downstream, so north-up vs south-up
    orientation is immaterial.

    Raises
    ------
    ValueError
        If *shape* is not a recognisable global grid.  Pass ``lat=`` explicitly
        rather than letting anything guess.
    """
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D (lat, lon) grid, got shape {shape}")
    n_lat, n_lon = shape
    if n_lon != 2 * n_lat or n_lat < 90:
        raise ValueError(
            f"grid {n_lat}×{n_lon} is not a recognisable global cell-centred "
            "lat/lon grid (expected n_lon == 2·n_lat, n_lat ≥ 90; the real ICE-6G "
            "10 arcmin grid is 1080×2160).  The sea-surface datum is a GLOBAL "
            "far-field quantity and cannot be derived from a regional subset — "
            "pass the full global field, or an explicit lat=/reference_mask=."
        )
    step = 180.0 / n_lat
    lat_1d = -90.0 + (np.arange(n_lat) + 0.5) * step
    return np.repeat(lat_1d[:, None], n_lon, axis=1)


def _as_lat_grid(lat: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if lat is None:
        return global_latitude_grid(shape)
    lat = np.asarray(lat, dtype=np.float64)
    if lat.ndim == 1:
        if lat.size != shape[0]:
            raise ValueError(
                f"1-D lat has {lat.size} entries but the grid has {shape[0]} rows"
            )
        return np.repeat(lat[:, None], shape[1], axis=1)
    if lat.shape != shape:
        raise ValueError(f"lat grid {lat.shape} does not match field shape {shape}")
    return lat


def ever_glaciated_mask(
    sftgif_fields: Iterable[np.ndarray],
    *,
    ice_percent_threshold: float = ICE_PERCENT_THRESHOLD,
) -> np.ndarray:
    """Union of ``sftgif > threshold`` over *all* epochs — the "ever ice" mask.

    Condition 2 of the reference set.  Excluding only the ice present at epoch *t*
    would admit deglaciated cells that are still relaxing, which are not a
    zero-displacement reference.  On the 48 shipped epochs this marks 429 737 cells
    (18.42 % of the grid); the scan costs ~2.5 s, so it is cheap enough to compute
    per session, but it is epoch-independent and worth hoisting out of any loop.
    """
    ever: np.ndarray | None = None
    for field in sftgif_fields:
        flagged = np.asarray(field) > ice_percent_threshold
        ever = flagged if ever is None else (ever | flagged)
    if ever is None:
        raise ValueError("ever_glaciated_mask() needs at least one sftgif field")
    return ever


def distance_to_ice_km(
    ever_ice: np.ndarray, *, cell_km: float = ICE6G_10MIN_CELL_KM
) -> np.ndarray:
    """Distance from each cell to the nearest ever-glaciated cell, in km.

    Euclidean transform on the grid with longitudinal wrap-around (half-width pad),
    so the Pacific dateline is not a false barrier.  See
    :data:`ICE6G_10MIN_CELL_KM` for the metric's known bias.
    """
    ever_ice = np.asarray(ever_ice, dtype=bool)
    if not ever_ice.any():
        return np.full(ever_ice.shape, np.inf)
    pad = ever_ice.shape[1] // 2
    wrapped = np.concatenate(
        [ever_ice[:, -pad:], ever_ice, ever_ice[:, :pad]], axis=1
    )
    dist = distance_transform_edt(~wrapped, sampling=(cell_km, cell_km))
    return dist[:, pad : pad + ever_ice.shape[1]]


# ---------------------------------------------------------------------------
# The reference set
# ---------------------------------------------------------------------------


def far_field_reference_mask(
    topo_0: np.ndarray,
    *,
    lat: np.ndarray | None = None,
    sftlf_t: np.ndarray | None = None,
    sftlf_0: np.ndarray | None = None,
    sftgif_t: np.ndarray | None = None,
    ever_ice: np.ndarray | None = None,
    dist_to_ice_km: np.ndarray | None = None,
    lat_band: float = FAR_FIELD_LAT_BAND_DEG,
    ice_buffer_km: float = FAR_FIELD_ICE_BUFFER_KM,
) -> np.ndarray:
    """Boolean mask of far-field ice-free tropical continental land.

    The two unconditional conditions — ``|lat| ≤ lat_band`` and ``topo_0 > 0`` — are
    always applied.  Each of the other three (land at both endpoints, never
    glaciated, ice buffer) is applied **only if its field is supplied**, so a caller
    holding just ``Topo`` still gets a defensible reference set.

    That degradation is deliberate and is measured, not assumed: over all 48 shipped
    epochs the mask-free offset differs from the full five-condition offset by at
    most **0.005 m** — five orders of magnitude below the ~30 m bias this module
    exists to remove, and three below the ±1 m inter-continental systematic.  The
    reason is that the tropical band is already 4346 km from ice on the median and
    contains almost no ever-glaciated cells, so conditions 1, 2 and 5 are nearly
    vacuous *there*; they are not vacuous if ``lat_band`` is widened, which is why
    they remain part of the definition.

    Prefer the full form.  ``tests/test_gia_topo.py::
    test_masked_and_unmasked_reference_set_offsets_agree`` is the guard that keeps
    the short form honest.
    """
    topo_0 = np.asarray(topo_0, dtype=np.float64)
    lat2d = _as_lat_grid(lat, topo_0.shape)

    ref = (np.abs(lat2d) <= float(lat_band)) & (topo_0 > 0.0)

    if sftlf_t is not None:
        ref &= np.asarray(sftlf_t) > LAND_PERCENT_THRESHOLD
    if sftlf_0 is not None:
        ref &= np.asarray(sftlf_0) > LAND_PERCENT_THRESHOLD
    if sftgif_t is not None:
        ref &= np.asarray(sftgif_t) <= ICE_PERCENT_THRESHOLD
    if ever_ice is not None:
        ref &= ~np.asarray(ever_ice, dtype=bool)
    if dist_to_ice_km is not None:
        ref &= np.asarray(dist_to_ice_km) >= float(ice_buffer_km)

    return ref


def implied_sea_surface_offset(
    topo_t: np.ndarray,
    topo_0: np.ndarray,
    deep_threshold: float | None = None,
    *,
    lat: np.ndarray | None = None,
    sftlf_t: np.ndarray | None = None,
    sftlf_0: np.ndarray | None = None,
    sftgif_t: np.ndarray | None = None,
    ever_ice: np.ndarray | None = None,
    dist_to_ice_km: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    lat_band: float = FAR_FIELD_LAT_BAND_DEG,
    ice_buffer_km: float = FAR_FIELD_ICE_BUFFER_KM,
    min_reference_cells: int | None = None,
) -> float:
    """Epoch sea-surface change baked into ICE-6G ``Topo``, in metres.

    ``median(topo_t − topo_0)`` over the far-field ice-free tropical continental
    reference set documented in the module docstring.  Positive at past lowstands
    (+81.96 m at 18 ka, +87.33 m at 21 ka).  Subtract it from ``ΔTopo`` to get
    Δz_solid — see :func:`solid_displacement`.

    .. warning::
       **API CHANGE, 2026-07-30.**  This used to be the median over *deep ocean*
       (``topo_0 < deep_threshold``), which is the datum bug: the ocean floor is not
       a zero-displacement frame.  The values returned are now ~30 m smaller at the
       LGM (18 ka: 81.96 m, was 112.25 m).  Positional arity is preserved and the
       two production call sites — ``Ice6gLoader.get_fields`` and
       ``bake_black_sea.gia_delta_subset``, both of which already pass the **global**
       ``Topo`` field — keep working unchanged and are silently corrected.
       ``deep_threshold`` is refused rather than ignored, so no old call site can
       keep reading as though the datum were still deep-ocean-referenced.
       **Every raster baked before this date is too low by the bias** (−30.3 m at
       18 ka, −29.2 at 16.5, −26.8 at 14.5, −17.8 at 12) and must be re-baked.

    Parameters
    ----------
    topo_t, topo_0 : np.ndarray
        ICE-6G ``Topo`` at the target epoch and at 0 ka (same shape, metres).
        Must be the **global** field unless ``lat``/``reference_mask`` is given.
    deep_threshold : None
        Retired.  Passing anything raises :class:`TypeError`.
    lat : np.ndarray, optional
        Cell-centre latitudes, 1-D (per row) or 2-D.  Inferred from the array shape
        for a global grid; see :func:`global_latitude_grid`.
    sftlf_t, sftlf_0, sftgif_t, ever_ice, dist_to_ice_km : np.ndarray, optional
        The remaining reference-set conditions.  Supply all five for the full
        definition; see :func:`far_field_reference_mask` on what omitting them costs.
    reference_mask : np.ndarray, optional
        Escape hatch: use exactly these cells and skip mask construction entirely.
        Intended for synthetic fixtures, which must declare their own far field
        rather than have one guessed for them.
    min_reference_cells : int
        Refuse if the reference set is smaller than this.

    Raises
    ------
    TypeError
        If the retired ``deep_threshold`` argument is supplied.
    ValueError
        If the grid is not global and no ``lat``/``reference_mask`` was given, or if
        the reference set has fewer than *min_reference_cells* usable cells.
    """
    if deep_threshold is not None:
        raise TypeError(
            "implied_sea_surface_offset() no longer accepts 'deep_threshold': the "
            "sea-surface datum is no longer taken from deep ocean.  The ocean floor "
            "stands ~+30 m higher than today at the LGM (water load absent), so a "
            "deep-ocean median absorbed that rebound and pushed it onto the "
            "continents with the sign flipped.  The datum is now the median ΔTopo "
            "over far-field ice-free tropical continental land; drop the argument, "
            "and re-bake anything built with the old one."
        )

    topo_t = np.asarray(topo_t, dtype=np.float64)
    topo_0 = np.asarray(topo_0, dtype=np.float64)
    if topo_t.shape != topo_0.shape:
        raise ValueError(f"shape mismatch: topo_t {topo_t.shape} vs topo_0 {topo_0.shape}")

    if reference_mask is not None:
        ref = np.asarray(reference_mask, dtype=bool)
        if ref.shape != topo_t.shape:
            raise ValueError(
                f"reference_mask {ref.shape} does not match field {topo_t.shape}"
            )
    else:
        ref = far_field_reference_mask(
            topo_0,
            lat=lat,
            sftlf_t=sftlf_t,
            sftlf_0=sftlf_0,
            sftgif_t=sftgif_t,
            ever_ice=ever_ice,
            dist_to_ice_km=dist_to_ice_km,
            lat_band=lat_band,
            ice_buffer_km=ice_buffer_km,
        )

    delta = topo_t - topo_0
    usable = ref & np.isfinite(delta)
    n = int(usable.sum())
    # Scaled to the grid — an absolute count is a hidden resolution dependency.
    # See MIN_REFERENCE_FRACTION.  An explicit min_reference_cells still wins.
    required = (
        minimum_reference_cells(delta.size)
        if min_reference_cells is None
        else int(min_reference_cells)
    )
    if n < required:
        raise ValueError(
            f"far-field reference set has only {n} usable cells (need "
            f"{required} for a {delta.size}-cell grid): there is no tropical "
            "ice-free continental land in this grid, so the sea-surface datum is "
            "undefined here.  Pass the GLOBAL Topo field, or an explicit "
            "reference_mask= if you are supplying a synthetic fixture."
        )
    # nanmedian: ICE-6G_C ``Topo`` declares _FillValue = 1e+20, so a stray fill/NaN
    # cell is ignored instead of poisoning the datum for the whole epoch.
    return float(np.nanmedian(delta[usable]))


# ---------------------------------------------------------------------------
# Solid displacement, and the ice-surface trap
# ---------------------------------------------------------------------------


def ice_surface_mask(
    sftgif_t: np.ndarray | None = None,
    sftgif_0: np.ndarray | None = None,
    *,
    ice_mask_t: np.ndarray | None = None,
    ice_mask_0: np.ndarray | None = None,
    ice_percent_threshold: float = ICE_PERCENT_THRESHOLD,
    dilate_cells: int = 0,
) -> np.ndarray | None:
    """Cells where ICE-6G ``Topo`` is the ice surface rather than the solid surface.

    A cell is flagged if it is glaciated at epoch *t* **or** today — either endpoint
    being an ice surface makes the difference meaningless.  Accepts percent fields
    (``sftgif``) or ready-made boolean masks.  Returns ``None`` when nothing was
    supplied, so callers can distinguish "no ice here" from "no ice information".

    ``dilate_cells`` grows the mask isotropically; pass
    :data:`SMOOTHING_RIM_CELLS` when the result will be convolved with a σ=2
    Gaussian by something that is not :func:`smooth_preserving_ice_mask`.
    """
    parts: list[np.ndarray] = []
    for percent in (sftgif_t, sftgif_0):
        if percent is not None:
            parts.append(np.asarray(percent) > ice_percent_threshold)
    for boolean in (ice_mask_t, ice_mask_0):
        if boolean is not None:
            parts.append(np.asarray(boolean, dtype=bool))
    if not parts:
        return None

    mask = parts[0]
    for extra in parts[1:]:
        mask = mask | extra
    if dilate_cells > 0:
        mask = binary_dilation(mask, iterations=int(dilate_cells))
    return mask


def solid_displacement(
    topo_t: np.ndarray,
    topo_0: np.ndarray,
    sea_surface_offset: float,
    *,
    sftgif_t: np.ndarray | None = None,
    sftgif_0: np.ndarray | None = None,
    ice_mask_t: np.ndarray | None = None,
    ice_mask_0: np.ndarray | None = None,
    dilate_cells: int = 0,
) -> np.ndarray:
    """Present-datum solid-earth vertical displacement ``Δz_solid``, in metres.

    ``Δz_solid = (topo_t − topo_0) − sea_surface_offset``, then **NaN wherever
    ``Topo`` is an ice surface rather than the solid surface**.  Positive = the solid
    surface stood higher than today; add to a present-datum modern DEM to obtain the
    paleo-DEM (``H_paleo = H_modern + Δz_solid``).  Far-field tropical land → ~0;
    unloaded ocean floor → ~+30 m at the LGM; near former ice loads → the isostatic
    signal.

    NaN over ice is not a placeholder for a number that exists somewhere: the
    shipped 10 arcmin files carry **no bedrock-under-ice field at all** (only
    ``sftlf``, ``sftgif``, ``Topo``, ``Orog``, and ``Topo == Orog ==`` the ice
    surface there).  A consumer that needs formerly glaciated ground must take Δz
    from ICE-7G's bedrock-deformation field — the same source
    :mod:`paleoeurope.gia.correction_pipeline` already uses for thickness — and must
    not ``nan_to_num`` this array, which would write ice-surface elevations into a
    variable called ``paleo_bedrock``.

    Parameters
    ----------
    sftgif_t, sftgif_0 : np.ndarray, optional
        ICE-6G icemask (percent) at epoch *t* and at 0 ka.
    ice_mask_t, ice_mask_0 : np.ndarray, optional
        Boolean equivalents, if the caller already has them.
    dilate_cells : int
        Grow the ice mask by this many cells before applying it.  Pass
        :data:`SMOOTHING_RIM_CELLS` if a plain Gaussian will be applied downstream;
        0 is correct with :func:`smooth_preserving_ice_mask`.

    Warns
    -----
    IceSurfaceUnmaskedWarning
        When no ice information is supplied at all, so any glaciated cell in the
        output holds an ice-surface elevation (median +1141 m at 21 ka, max +3506 m).
        Kept as a warning rather than an error only to preserve the existing
        signature for callers that pass regional subsets with no ``sftgif`` to hand.
    """
    dz = (np.asarray(topo_t, dtype=np.float64) - np.asarray(topo_0, dtype=np.float64))
    dz = dz - float(sea_surface_offset)

    mask = ice_surface_mask(
        sftgif_t,
        sftgif_0,
        ice_mask_t=ice_mask_t,
        ice_mask_0=ice_mask_0,
        dilate_cells=dilate_cells,
    )
    if mask is None:
        warnings.warn(
            "solid_displacement() called with no ice information: over any cell "
            "glaciated at this epoch or today, ICE-6G 'Topo' is the ICE SURFACE, "
            "not the solid surface, so the returned array holds ice-surface "
            "elevations there (median +1141 m at 21 ka, max +3506 m) and must NOT "
            "be used as bedrock.  Pass sftgif_t=/sftgif_0= to get NaN instead.",
            IceSurfaceUnmaskedWarning,
            stacklevel=2,
        )
        return dz

    if mask.shape != dz.shape:
        raise ValueError(f"ice mask {mask.shape} does not match field {dz.shape}")
    return np.where(mask, np.nan, dz)


def smooth_preserving_ice_mask(
    dz: np.ndarray, sigma: float = 2.0, *, dilate_cells: int = 0, mode: str = "nearest"
) -> np.ndarray:
    """Gaussian-smooth ``Δz_solid`` without letting NaN (ice) cells leak or spread.

    A plain ``gaussian_filter`` does two wrong things to the output of
    :func:`solid_displacement`: NaNs propagate over the whole kernel support, and —
    if the mask was applied *after* smoothing instead — ice-surface values smear
    across the margin.  Measured on the real files at 21 ka, a naive σ=2 filter over
    an unmasked field differs from the answer below by a median of +169 m one cell
    past the ice margin, +33 m at three cells, and >1 m out to six or seven cells
    (p99 < 1 m only from seven cells out, <0.2 m from eight).

    Implementation: weighted convolution renormalised over the finite cells only
    (``G∗x̃ / G∗m``), then the mask re-applied.  Ice values never enter the sum, so
    no rim dilation is needed for correctness; ``dilate_cells`` is offered for
    callers who want to discard the margin cells whose smoothed value rests on
    one-sided support (they are legitimate but effectively less smoothed).
    """
    dz = np.asarray(dz, dtype=np.float64)
    finite = np.isfinite(dz)
    if finite.all():
        return gaussian_filter(dz, sigma=sigma, mode=mode)

    numer = gaussian_filter(np.where(finite, dz, 0.0), sigma=sigma, mode=mode)
    denom = gaussian_filter(finite.astype(np.float64), sigma=sigma, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numer / denom
    out = np.where(denom > 1e-6, out, np.nan)

    hole = ~finite
    if dilate_cells > 0:
        hole = binary_dilation(hole, iterations=int(dilate_cells))
    return np.where(hole, np.nan, out)
