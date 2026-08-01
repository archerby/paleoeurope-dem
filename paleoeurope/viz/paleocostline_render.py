"""
paleoeurope.viz.paleocostline_render
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composite per-epoch paleo-topography render.

Pipeline
--------
1. Build a merged DEM mosaic from per-tile GeoTIFFs (down-sampled to
   ``RenderConfig.max_px`` pixels on the longer axis).
2. Load ``Paleocoastlines.shp`` once; filter polygons to the sea-level step
   nearest to the Spratt & Lisiecki (2016) value for the requested epoch.
3. Rasterize land polygons onto the DEM grid to obtain a land/sea binary mask
   (falls back to DEM zero-contour if the shapefile is unavailable or covers
   less than 1 % of the scene).
4. LAND pixels  : hillshade × terrain colourmap by absolute elevation.
5. OCEAN pixels : blue gradient by bathymetric depth (square-root stretch).
6. ICE pixels   : solid white override on top of everything (ICE-7G H_ice > 10 m).
7. Export PNG and return the output ``Path``.

References
----------
Spratt, R. M. & Lisiecki, L. E. (2016). A Late Pleistocene sea-level stack.
    *Climate of the Past*, 12, 1079–1092. https://doi.org/10.5194/cp-12-1079-2016

Paleocoastlines GIS dataset — see docs/data_catalogue.md for citation and DOI.

Examples
--------
>>> from paleoeurope.viz import render_paleocostline_epoch, RenderConfig
>>> cfg = RenderConfig(max_px=1024, dpi=100)
>>> out = render_paleocostline_epoch(
...     epoch_ka=21,
...     tile_paths={"/path/to/N62E006_t21.tif": "/path/to/N62E006_t21.tif"},
...     sea_level_csv=Path("data/sea_level_curves/spratt_lisiecki_2016_simplified.csv"),
...     paleocoastlines_shp=Path("data/paleocostline/Paleocoastlines/Paleocoastlines/Paleocoastlines.shp"),
...     output_dir=Path("outputs"),
...     config=cfg,
... )
>>> print(out)   # PosixPath('outputs/paleo_render_21ka.png')
"""

from __future__ import annotations

import gc
import heapq
import json
import warnings
from collections import deque
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling as RIOResampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.merge import merge as rasterio_merge
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    gaussian_filter1d,
)
from scipy.ndimage import (
    label as ndi_label,
)
from shapely.geometry import box as shp_box
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    """Render parameters for :func:`render_paleocostline_epoch`.

    Parameters
    ----------
    max_px : int
        Maximum pixel count along the longer mosaic axis.  Larger mosaics are
        down-sampled at merge time.  Default 2048.
    dpi : int
        Output PNG DPI.  Default 150 (2100 × 2100 px at 14-inch figure size).
    fig_size_in : float
        Square figure side length in inches.  Default 14.
    vert_exag : float
        Vertical exaggeration for hillshade.  Default 8×.
    ice_threshold_m : float
        Ice-sheet pixels thicker than this value (m) receive the white override.
        Default 10 m.
    seam_sigma_px : float
        Gaussian σ for tile-seam blending along tile borders.  Default 5 px.
    land_elev_percentile_lo : float
        Lower colour-scale percentile for land elevation (clips deep valleys).
    land_elev_percentile_hi : float
        Upper colour-scale percentile for land elevation (clips alpine peaks).
    """

    max_px: int = 2048
    dpi: int = 150
    fig_size_in: float = 14.0
    vert_exag: float = 8.0
    ice_threshold_m: float = 10.0
    seam_sigma_px: float = 5.0
    land_elev_percentile_lo: float = 2.0
    land_elev_percentile_hi: float = 98.0
    extra_figure_title: str = ""
    ice_model_label: str = "ICE-7G"
    """Name of the GIA/ice model shown in the figure title. The Black Sea bake
    is ICE-6G_C, so it passes 'ICE-6G_C' to keep the annotation truthful."""
    ice_dome_sigma_px: float = 60.0
    """Gaussian σ (pixels) applied to the DEM *before* hillshade on ice-covered
    pixels.  Removes bedrock texture under the glacier so the dome surface
    looks smooth.  σ=60 at 2048 px / 1° tile ≈ 2–3 km smoothing radius at
    30 m FABDEM resolution after downsampling — enough to erase valleys and
    ridges while preserving the large-scale dome curvature.
    Mirrors IceThicknessModel.get_thickness_like(sigma=0.7 on 1°-grid → cubic
    resampling) that produces smooth organic dome shapes in production.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Widest spacing (years) between the two published nodes bracketing a requested
# epoch that still counts as READING the eustatic curve rather than inventing a
# number across a hole in it.  The published Spratt & Lisiecki (2016) stack is on
# a uniform 1 ka grid (NOAA/WDS-Paleo study 19982), so 1000 yr means "the two
# adjacent published samples"; anything wider means the curve does not resolve
# the epoch and :func:`get_sea_level_for_epoch` says so instead of answering.
MAX_CURVE_NODE_GAP_YR = 1_000.0


class SeaLevelCurveResolutionError(ValueError):
    """The eustatic curve does not resolve the requested epoch.

    Raised when the two curve rows bracketing the request are further apart than
    ``max_node_gap_yr``, i.e. when producing a value would mean bridging a hole
    in the curve rather than reading between two adjacent published samples.
    """


def get_sea_level_for_epoch(
    epoch_ka: float,
    csv_path: Path,
    *,
    max_node_gap_yr: float = MAX_CURVE_NODE_GAP_YR,
) -> float:
    """Return the Spratt & Lisiecki (2016) sea-level value for *epoch_ka*.

    The CSV must contain three columns (no header, ``#``-prefixed comment lines
    allowed): **age_BP** (years), **sea_level_m** (m relative to present),
    **uncertainty_m** (m, not used here).

    Interpolated, not snapped
    -------------------------
    Sea level is a continuous quantity sampled on a discrete age grid, so a
    request between two nodes is answered by **linear interpolation in age**
    between the bracketing rows.  On a node the published value is returned
    bit-exact, so nothing that asks for a published age moves.

    This replaces a nearest-neighbour lookup that was *systematically biased*.
    The published stack is on a uniform 1 ka grid, ``pandas.Series.idxmin``
    returns the FIRST row attaining the minimum, and the CSV is ascending in
    age — so an epoch landing exactly halfway between two nodes always resolved
    to the YOUNGER, i.e. SHALLOWER, node.  Four of the thirteen epochs the
    Black-Sea bake ships do exactly that, two of them the hero frames::

          ka    nearest-neighbour      interpolated      error
        16.5      -98.06  (16 ka)         -102.680     +4.620 m
        14.5      -76.64  (14 ka)          -81.605     +4.965 m
         9.5      -15.83  ( 9 ka)          -20.210     +4.380 m
         7.5       -6.11  ( 7 ka)           -7.600     +1.490 m

    and it did so *silently*: the miss is exactly 500 yr — exactly half a step —
    which every guard in the project tests with a strict ``>`` inequality.
    Interpolation removes the tie rather than arbitrating it, so there is no
    half-step boundary condition left to fall through.

    What it refuses to do
    ---------------------
    * **Bridge a hole.**  If the bracketing rows are more than *max_node_gap_yr*
      apart the request is not a lookup, and :class:`SeaLevelCurveResolutionError`
      is raised rather than a plausible-looking number returned.
    * **Clamp in silence.**  Off either end of the curve the endpoint value is
      still returned (callers such as the bake's coverage check turn this into a
      hard stop of their own), but *any* such extrapolation now warns.  The old
      warning fired only above a 1000 yr miss, so a 0-20 ka curve asked for
      21 ka — missing by exactly 1000 yr — clamped without a word.

    This function is for the CONTINUOUS eustatic curve only.  Do not reuse it for
    a step-keyed or stage-keyed series: the discrete coastline layers of
    :func:`filter_paleocostlines_for_epoch` and a closed-basin curve carrying a
    genuine reconnection discontinuity must both stay nearest-neighbour, because
    the value halfway between two of their rows does not exist.

    Parameters
    ----------
    epoch_ka : float
        Target epoch in ka BP (e.g. ``21.0``).
    csv_path : Path
        Path to the eustatic curve CSV (``spratt_lisiecki_2016_full.csv``).
    max_node_gap_yr : float, keyword-only
        Widest bracketing-node spacing (years) still considered a lookup.
        Defaults to :data:`MAX_CURVE_NODE_GAP_YR` (one step of the published
        1 ka grid).  Raise it only if you own a coarser curve and are prepared
        to say so in writing.

    Returns
    -------
    float
        Sea level in metres relative to present (negative = lower than today).

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    SeaLevelCurveResolutionError
        If the curve is empty, or if the bracketing rows are further apart than
        *max_node_gap_yr*.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sea-level CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        comment="#",
        header=None,
        names=["age_bp", "sea_level_m", "uncertainty_m"],
    )
    df = df[["age_bp", "sea_level_m"]].apply(pd.to_numeric, errors="coerce").dropna()
    # np.interp / np.searchsorted are undefined on unsorted x, and a curve file
    # is not required to be sorted.  Sort explicitly; drop duplicate ages so a
    # repeated row cannot produce a zero-width bracket.
    df = df.drop_duplicates(subset="age_bp").sort_values("age_bp")
    ages = df["age_bp"].to_numpy(dtype=float)
    levels = df["sea_level_m"].to_numpy(dtype=float)
    if ages.size == 0:
        raise SeaLevelCurveResolutionError(
            f"Sea-level CSV has no usable (age_bp, sea_level_m) rows: {csv_path}"
        )

    want = float(epoch_ka) * 1_000.0

    # ── Off the end: clamp, but never in silence ────────────────────────────
    if want < ages[0] or want > ages[-1]:
        endpoint_age = ages[0] if want < ages[0] else ages[-1]
        age_diff = abs(endpoint_age - want)
        warnings.warn(
            f"Spratt CSV: closest row is {int(endpoint_age)} BP "
            f"(requested {int(want)} BP, Δ = {age_diff:.0f} yr). "
            f"The curve covers {int(ages[0])}–{int(ages[-1])} BP and does not "
            "reach this epoch — using clamped endpoint value.",
            UserWarning,
            stacklevel=2,
        )
        return float(levels[0] if want < ages[0] else levels[-1])

    # ── On a node: the published value, untouched ───────────────────────────
    hit = np.flatnonzero(ages == want)
    if hit.size:
        return float(levels[hit[0]])

    # ── Between nodes: interpolate, unless that would bridge a hole ─────────
    hi = int(np.searchsorted(ages, want))
    lo = hi - 1
    gap = ages[hi] - ages[lo]
    if gap > max_node_gap_yr:
        raise SeaLevelCurveResolutionError(
            f"eustatic curve {csv_path} does not resolve {epoch_ka} ka "
            f"({int(want)} BP): the bracketing rows are {int(ages[lo])} BP and "
            f"{int(ages[hi])} BP, {gap:.0f} yr apart — wider than the "
            f"{max_node_gap_yr:.0f} yr budget. Interpolating across that hole "
            "would invent a number the curve does not contain. Supply a curve "
            "that samples this epoch, or pass an explicit max_node_gap_yr."
        )
    frac = (want - ages[lo]) / gap
    return float(levels[lo] + frac * (levels[hi] - levels[lo]))


def load_paleocostlines(shp_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and repair a Paleocoastlines vector file.

    Accepts any OGR-supported format: ``.shp``, ``.fgb`` (FlatGeobuf),
    ``.gpkg``, etc.  The bundled file ``data/paleocostlines/paleocoastlines_north_sea.fgb``
    covers −5°→15°E / 50°→63°N and can be used without any extra downloads.

    Returns a :class:`geopandas.GeoDataFrame` with a numeric ``Sea level``
    column, or ``None`` if loading fails.  Invalid geometries are repaired with
    :func:`shapely.validation.make_valid` and non-polygon results are dropped.

    Parameters
    ----------
    shp_path : Path
        Path to the Paleocoastlines vector file (".shp", ".fgb", or ".gpkg").

    Returns
    -------
    gpd.GeoDataFrame | None
        GeoDataFrame with columns ``geometry`` and ``Sea level`` (float, metres),
        or ``None`` on failure.
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        warnings.warn(f"Paleocoastlines shapefile not found: {shp_path}", stacklevel=2)
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf = gpd.read_file(shp_path, engine="pyogrio", on_invalid="ignore")

        gdf = gdf[~gdf.geometry.isna()].copy()

        invalid_mask = ~gdf.geometry.is_valid
        n_repaired = int(invalid_mask.sum())
        if n_repaired:
            gdf.loc[invalid_mask, "geometry"] = (
                gdf.loc[invalid_mask, "geometry"].apply(make_valid)
            )
            gdf = gdf.explode(index_parts=False)
            gdf = gdf[
                gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            ].copy()

        gdf["Sea level"] = pd.to_numeric(gdf["Sea level"], errors="coerce")
        gdf = gdf[gdf["Sea level"].notna()].copy()
        return gdf

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to load Paleocoastlines shapefile: {exc}",
            stacklevel=2,
        )
        return None


def filter_paleocostlines_for_epoch(
    gdf: Optional[gpd.GeoDataFrame],
    target_sl_m: float,
) -> tuple[Optional[gpd.GeoDataFrame], Optional[int]]:
    """Return land polygons for the sea-level step nearest to *target_sl_m*.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame | None
        Full Paleocoastlines GeoDataFrame from :func:`load_paleocostlines`.
    target_sl_m : float
        Target sea level in metres (e.g. ``-120.0`` for LGM).

    Returns
    -------
    land_gdf : gpd.GeoDataFrame | None
        Subset of *gdf* for the matching sea-level step, or ``None`` if *gdf*
        is ``None``.
    matched_level : int | None
        The sea-level value (metres) of the matching step, or ``None``.
    """
    if gdf is None:
        return None, None

    levels = sorted(gdf["Sea level"].dropna().unique())
    matched = int(min(levels, key=lambda x: abs(x - target_sl_m)))
    land = gdf[gdf["Sea level"] == matched].copy()
    return land, matched


def rasterize_land_mask(
    land_gdf: Optional[gpd.GeoDataFrame],
    transform: rasterio.transform.Affine,
    height: int,
    width: int,
    mosaic_crs,
    bounds: BoundingBox,
) -> Optional[np.ndarray]:
    """Rasterize land polygons onto the DEM grid.

    Parameters
    ----------
    land_gdf : gpd.GeoDataFrame | None
        Land polygons (from :func:`filter_paleocostlines_for_epoch`).
    transform : rasterio.transform.Affine
        Affine transform of the output grid.
    height, width : int
        Output raster dimensions (rows × columns).
    mosaic_crs
        CRS of the output grid.  The GeoDataFrame is reprojected if needed.
    bounds : rasterio.coords.BoundingBox
        Geographic extent of the output grid (used for spatial clipping).

    Returns
    -------
    np.ndarray[bool] | None
        Boolean mask where ``True`` = land.  Returns ``None`` if rasterization
        fails or no polygons intersect the scene.
    """
    if land_gdf is None or len(land_gdf) == 0:
        return None

    try:
        clip_box = shp_box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf_proj = land_gdf.to_crs(mosaic_crs)
        clipped = gdf_proj.clip(clip_box)
        if len(clipped) == 0:
            return None

        geoms = [
            (g, 1)
            for g in clipped.geometry
            if g is not None and not g.is_empty
        ]
        if not geoms:
            return None

        mask = rio_rasterize(
            geoms,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)
        return mask

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Land-mask rasterization failed ({exc}); falling back to DEM threshold.",
            stacklevel=2,
        )
        return None


def ocean_depth_from_surface(
    dem: np.ndarray,
    ocean_mask: np.ndarray,
    water_level: float | np.ndarray,
) -> np.ndarray:
    """Water-column depth below the epoch water surface (m, ``>= 0``) on ocean pixels.

    ``depth = clip(water_level - dem, 0, inf)`` inside ``ocean_mask``, else 0.

    ``water_level`` is the epoch/basin surface in the SAME datum as ``dem``
    (present global SL = 0).  Using the epoch surface — not the present 0 m datum —
    is what makes a lowered decoupled basin read as a *shallower* body, so a
    ~100 m drawdown is visible in the shading instead of looking as deep as today.

    It may be a **scalar** (one basin, one surface — the original behaviour, and
    the arithmetic below is unchanged for it) or an **array broadcastable to
    ``dem``** (several basins standing at different levels in one frame, plus the
    ramped spill corridor between them — see :func:`assign_basins`).  Cells whose
    surface is NaN (no basin owns them) are never in ``ocean_mask``, so they
    contribute 0.
    """
    return np.where(ocean_mask, np.clip(water_level - dem, 0.0, None), 0.0)


def dem_threshold_ocean(
    dem_sm: np.ndarray,
    nodata: np.ndarray,
    water_level: float | np.ndarray,
) -> np.ndarray:
    """Boolean ocean mask from the DEM: submerged where ``dem_sm < water_level``.

    ``water_level`` is the epoch/basin surface (present-datum).  Morphological
    closing then opening (2 iterations each) removes single-pixel speckle.  This
    is the basin-correct replacement for the old hard ``dem_sm < 0.0`` fallback,
    which silently ignored the override.

    ``water_level`` may be a **scalar** (one basin at one level — unchanged) or an
    **array broadcastable to ``dem_sm``** (the per-cell water surface produced by
    :func:`assign_basins` when two basins stand at different levels in the same
    frame).  A NaN entry means "no basin owns this cell": ``dem_sm < NaN`` is
    False, so such a cell is land, which is the honest reading — the frame carries
    no level for it.
    """
    ocean_raw = (~nodata) & (dem_sm < water_level)
    ocean = binary_closing(ocean_raw, iterations=2) & (~nodata)
    ocean = binary_opening(ocean, iterations=2).astype(bool)
    return ocean


# ---------------------------------------------------------------------------
# Two basins, two levels, one frame
# ---------------------------------------------------------------------------


class SillResolutionError(ValueError):
    """This frame's DEM disagrees with an independently measured sill elevation.

    ``spilling`` is only as honest as the grid it is observed on, and a spillway
    narrower than the working pixel is filled in by resampling.  Measured on the
    real Caspian↔Pontic raster (2026-08-01), the Manych saddle reads

    ==========================================================  =========
    native GEBCO 15 arcsec, untouched                            +25.00 m
    bilinear-downsampled to a 2000 px render mosaic              +30.00 m
    ... plus the renderer's ``dem_sm`` gaussian σ = 1.5 px       +33.35 m
    ... plus ``build_epoch_mosaic``'s tile-seam blend at 44 E    +38.52 m
    ==========================================================  =========

    — 13.5 m of purely numerical rise, which at a +35 m Khvalynian highstand
    turns a SPILLING frame into an endorheic one without a word.  A caller who
    has measured the sill at native resolution passes it as
    ``sill_expectations`` and gets this exception instead of a quietly wrong
    animation.
    """


class CorridorNotRampableError(NotImplementedError):
    """More than one pair of basins is spilling into another at this epoch.

    The linear water-surface ramp implemented by :func:`assign_basins` is defined
    for exactly **one** donor→receiver corridor.  With several simultaneous
    corridors the surface in each depends on how the discharges divide, which is a
    hydraulic question this module does not answer and will not guess at.  Pass
    ``corridor="step"`` to get every basin at its own flat level with the
    corridors *marked* instead.
    """


@dataclass(frozen=True)
class BasinSeed:
    """One named basin, located by a point that is under its own water.

    Parameters
    ----------
    name : str
        Basin name, e.g. ``"caspian"``.  Must be unique within one call and is
        what appears in the figure, the sidecar JSON and the sill keys.
    lon, lat : float
        Seed coordinates **in the CRS of the DEM being assigned** (EPSG:4326 for
        every raster in this project).  The cell containing this point is the
        growth origin of the basin.
    level_m : float
        This basin's water surface for the epoch, in the SAME vertical frame as
        the DEM — i.e. F1 for the paleo-DEM, which for an F2 literature level
        means ``level + Δz_solid(x_marker, t)`` has ALREADY been applied by the
        caller.  See ``2026-07-30-vertical-datum-contract.md`` §1 and §5a-bis;
        this function does no frame conversion and cannot detect a wrong one.
    """

    name: str
    lon: float
    lat: float
    level_m: float


@dataclass(frozen=True)
class BasinField:
    """The per-cell result of :func:`assign_basins`.

    Attributes
    ----------
    basin_id : np.ndarray[int32]
        ``0`` = no basin owns this cell (nodata, above the flood ceiling, or in a
        depression that reaches neither seed).  ``i+1`` = owned by ``names[i]``.
    names, levels_m : tuple
        The seeds, in input order.
    bottleneck_m : np.ndarray[float64]
        For each owned cell, the lowest elevation a path from its seed can be held
        under — the minimax ("bottleneck") elevation.  ``inf`` where unowned.
        This is what makes the sill measurable: the saddle between two basins is
        the lowest ``max(bottleneck_a, bottleneck_b)`` over the cells where their
        territories touch.
    water_level_m : np.ndarray[float64]
        The water SURFACE at each cell: the owning basin's level, except inside a
        ramped spill corridor where it is interpolated between donor and receiver.
        NaN where no basin owns the cell.
    water_mask : np.ndarray[bool]
        ``dem < water_level_m``.  This is the two-basin generalisation of the
        single scalar threshold.
    corridor_mask : np.ndarray[bool]
        Cells of the spill corridor.  In ``"ramp"`` mode these are the corridor
        cells that are actually wet; in ``"step"`` mode they are the cells the
        spill would flow through, marked but NOT given a made-up surface.
    spilling : bool
        **Observed from the terrain**, not read off a curve: True iff at least one
        pair of seeded fills merges at the epoch's own levels.
    connected_groups : tuple[tuple[str, ...], ...]
        Basins grouped by hydrological connection at this epoch.  Two separate
        basins give ``(("caspian",), ("pontic",))``; once they spill,
        ``(("caspian", "pontic"),)``.
    sill_m, sill_lonlat : dict
        Per unordered basin pair (key = the two names, sorted): the measured
        saddle elevation and where it is.  **NaN / None when the saddle lies above
        the flood ceiling** — the algorithm never invents one it did not reach.
    corridor_mode : {"none", "ramp", "step"}
    ceiling_m : float
        The elevation the ownership flood was capped at.
    """

    basin_id: np.ndarray
    names: tuple[str, ...]
    levels_m: tuple[float, ...]
    bottleneck_m: np.ndarray
    water_level_m: np.ndarray
    water_mask: np.ndarray
    corridor_mask: np.ndarray
    spilling: bool
    connected_groups: tuple[tuple[str, ...], ...]
    sill_m: dict = dataclass_field(default_factory=dict)
    sill_lonlat: dict = dataclass_field(default_factory=dict)
    corridor_mode: str = "none"
    ceiling_m: float = float("nan")


def _seed_rowcol(
    transform: rasterio.transform.Affine,
    lon: float,
    lat: float,
    shape: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Cell containing (*lon*, *lat*); raises if that is off the raster."""
    height, width = shape
    col_f, row_f = ~transform * (float(lon), float(lat))
    row, col = int(np.floor(row_f)), int(np.floor(col_f))
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"basin seed {name!r} at {lon:.4f} E / {lat:.4f} N falls outside the "
            f"raster (it maps to row {row}, col {col} of a {height}x{width} grid). "
            "A seed off the frame would silently produce an empty basin."
        )
    return row, col


def _priority_flood_ownership(
    dem: np.ndarray,
    seed_rc: Sequence[tuple[int, int]],
    ceiling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow every seed by ascending bottleneck elevation; return (owner, bottleneck).

    This is the classic priority-flood / flooding-watershed: the frontier cell that
    can be reached while staying LOWEST is always claimed next, so a cell ends up
    owned by the basin it drains to — the water divide, not a bounding box.  The
    value recorded per cell is the minimax elevation along the path that claimed
    it, which is exactly what a saddle measurement needs.

    Deterministic: heap entries are ``(bottleneck, flat_index, label)`` and a cell
    is pushed at most once, so ``flat_index`` breaks every tie the same way on
    every run and on every machine.  4-connectivity, matching
    :func:`scipy.ndimage.label`'s default structure so the ownership flood and the
    spill-connectivity test agree about what "connected" means.

    Cells above *ceiling*, and NaN cells, are never claimed.
    """
    height, width = dem.shape
    flat = np.ascontiguousarray(dem, dtype=np.float64).ravel()
    # Python lists are ~4x faster than numpy scalar indexing in this inner loop,
    # which runs once per cell of a 2000-px mosaic.
    elevations = flat.tolist()
    passable = (np.isfinite(flat) & (flat <= ceiling)).tolist()
    owner = [0] * flat.size
    bottleneck = [float("inf")] * flat.size

    heap: list[tuple[float, int, int]] = []
    for label, (row, col) in enumerate(seed_rc, start=1):
        idx = row * width + col
        owner[idx] = label
        bottleneck[idx] = elevations[idx]
        heapq.heappush(heap, (elevations[idx], idx, label))

    push, pop = heapq.heappush, heapq.heappop
    last_col = width - 1
    last_row = height - 1
    while heap:
        here, idx, label = pop(heap)
        row = idx // width
        col = idx - row * width
        for nbr in (
            idx - width if row else -1,
            idx + width if row < last_row else -1,
            idx - 1 if col else -1,
            idx + 1 if col < last_col else -1,
        ):
            if nbr < 0 or owner[nbr] or not passable[nbr]:
                continue
            owner[nbr] = label
            deeper = elevations[nbr]
            reach = here if here > deeper else deeper
            bottleneck[nbr] = reach
            push(heap, (reach, nbr, label))

    return (
        np.asarray(owner, dtype=np.int32).reshape(height, width),
        np.asarray(bottleneck, dtype=np.float64).reshape(height, width),
    )


def _measure_sills(
    basin_id: np.ndarray,
    bottleneck: np.ndarray,
    n_basins: int,
) -> dict[tuple[int, int], tuple[float, tuple[int, int]]]:
    """Saddle elevation + location for every pair of basin territories that touch.

    The saddle between two basins is ``min`` over all 4-adjacent cell pairs
    straddling their common border of ``max(bottleneck_a, bottleneck_b)``, and its
    location is the higher of that winning pair — i.e. the actual crest cell.
    Pairs whose territories never touch (because the flood ceiling cut the ridge
    off) are simply absent, and the caller reports them as unknown.
    """
    best: dict[tuple[int, int], tuple[float, tuple[int, int]]] = {}
    height, width = basin_id.shape
    rows, cols = np.indices((height, width))

    for axis in (0, 1):
        if axis == 0:
            lab_a, lab_b = basin_id[:-1, :], basin_id[1:, :]
            bot_a, bot_b = bottleneck[:-1, :], bottleneck[1:, :]
            row_a, row_b = rows[:-1, :], rows[1:, :]
            col_a, col_b = cols[:-1, :], cols[1:, :]
        else:
            lab_a, lab_b = basin_id[:, :-1], basin_id[:, 1:]
            bot_a, bot_b = bottleneck[:, :-1], bottleneck[:, 1:]
            row_a, row_b = rows[:, :-1], rows[:, 1:]
            col_a, col_b = cols[:, :-1], cols[:, 1:]

        touching = (lab_a > 0) & (lab_b > 0) & (lab_a != lab_b)
        if not touching.any():
            continue
        crest = np.maximum(bot_a, bot_b)[touching]
        take_b = (bot_b >= bot_a)[touching]
        crest_row = np.where(take_b, row_b[touching], row_a[touching])
        crest_col = np.where(take_b, col_b[touching], col_a[touching])
        pair_lo = np.minimum(lab_a, lab_b)[touching]
        pair_hi = np.maximum(lab_a, lab_b)[touching]
        key = pair_lo.astype(np.int64) * (n_basins + 1) + pair_hi

        for encoded in np.unique(key):
            sel = key == encoded
            winner = int(np.argmin(crest[sel]))
            value = float(crest[sel][winner])
            where = (int(crest_row[sel][winner]), int(crest_col[sel][winner]))
            pair = (int(encoded // (n_basins + 1)), int(encoded % (n_basins + 1)))
            if pair not in best or value < best[pair][0]:
                best[pair] = (value, where)

    return best


def _geodesic_hops(region: np.ndarray, source: np.ndarray) -> np.ndarray:
    """4-connected hop count from *source* to every cell of *region*, staying inside it.

    Cells of *region* adjacent to *source* are at distance 1.  Unreachable cells
    are ``inf``.  This is a graph distance in PIXELS, not a metric length and not a
    hydraulic path — it exists only to parameterise the corridor ramp from one end
    to the other, and inside a channel it is monotone along the channel, which is
    all the ramp needs.
    """
    height, width = region.shape
    dist = np.full(region.size, np.inf)
    inside = region.ravel()

    adjacent = np.zeros_like(region)
    adjacent[:-1, :] |= source[1:, :]
    adjacent[1:, :] |= source[:-1, :]
    adjacent[:, :-1] |= source[:, 1:]
    adjacent[:, 1:] |= source[:, :-1]
    start = np.flatnonzero((region & adjacent).ravel())
    dist[start] = 1.0

    queue = deque(int(i) for i in start)
    last_row, last_col = height - 1, width - 1
    while queue:
        idx = queue.popleft()
        here = dist[idx]
        row = idx // width
        col = idx - row * width
        for nbr in (
            idx - width if row else -1,
            idx + width if row < last_row else -1,
            idx - 1 if col else -1,
            idx + 1 if col < last_col else -1,
        ):
            if nbr < 0 or not inside[nbr] or dist[nbr] <= here + 1.0:
                continue
            dist[nbr] = here + 1.0
            queue.append(nbr)

    return dist.reshape(height, width)


def assign_basins(
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    seeds: Sequence[BasinSeed],
    *,
    ceiling_margin_m: float = 100.0,
    corridor: str = "ramp",
    sill_expectations: Optional[dict] = None,
    sill_tolerance_m: float = 2.0,
) -> BasinField:
    """Assign every cell to the basin it drains to, and threshold each at its own level.

    This is the multi-basin generalisation of a single scalar sea level.  The hero
    frame of the Khvalynian reconstruction — the Caspian near +35 m and the Pontic
    near −30 m **in one image**, with water crossing the Manych — is inexpressible
    with one number, and this is the object that expresses it.

    How membership is decided
    -------------------------
    By a seeded priority flood over the DEM (:func:`_priority_flood_ownership`),
    not by a bounding box: the frontier cell reachable at the LOWEST elevation is
    always claimed next, so a cell ends up in the basin whose seed it can reach
    without crossing a higher barrier — the drainage divide.  A consequence worth
    having is that the **saddle between two basins falls out of the same flood**,
    so the sill elevation reported here is *measured off the terrain* rather than
    quoted from a document.

    How the spill is decided
    ------------------------
    Also from the terrain: two basins are connected iff the fill of the higher one
    **at its own level** reaches the other one's seed (``scipy.ndimage.label`` on
    ``dem < max(level_i, level_j)``).  That is the "the two fills merged" test, and
    it is the animation's whole subject, so it is a first-class output
    (:attr:`BasinField.spilling`, :attr:`BasinField.connected_groups`) rather than
    an assertion carried in from a level curve.

    The corridor between two levels
    -------------------------------
    Between a donor basin at +35 m and a receiver at −30 m the water surface is a
    **gradient, not a step, and not a 60 m wall of blue**.  With ``corridor="ramp"``
    (default) each corridor cell gets

    ``L = level_donor + (level_receiver − level_donor) · d_donor / (d_donor + d_receiver)``

    where the two ``d`` are 4-connected hop distances measured *inside the corridor*
    from the donor's water and from the receiver's water.  The profile is linear in
    along-channel graph distance, monotone, and exact at both ends.  It also fixes
    the failure mode a naive "flood everything below the donor level" produces —
    filling the whole receiving basin to the donor's surface — because far from the
    channel ``d_donor ≫ d_receiver``, so ``L`` collapses onto the receiver's level
    and those cells stay dry.

    **What this ramp is not.**  It is not hydraulics.  There is no discharge, no
    channel roughness, no backwater: the lateral extent of the flow is set here by
    where the terrain lies below a linearly-interpolated surface, and a real
    spillway's width is set by how much water is moving.  A linear water-surface
    profile is the standard first-order assumption for a uniform channel and it is
    defensible for a figure; it is not a flood model.  If several corridors are
    open at once the interpolation is undefined and this function
    :class:`raises <CorridorNotRampableError>` — pass ``corridor="step"`` to get
    every basin flat at its own level with the corridor cells MARKED in
    :attr:`BasinField.corridor_mask` instead of drawn as an invented surface.

    Parameters
    ----------
    dem : np.ndarray
        Elevation grid, metres, present-day datum; NaN = nodata.  Pass the SAME
        array the land/sea threshold will run on (the renderer passes its smoothed
        ``dem_sm``), so ownership, sill and mask cannot disagree.
    transform : rasterio.transform.Affine
        Affine transform of *dem*, used only to locate the seeds and to report the
        sill positions.
    seeds : sequence of BasinSeed
        One per basin.  Levels must already be in the DEM's vertical frame.
    ceiling_margin_m : float
        The ownership flood stops at ``max(level) + ceiling_margin_m``.  Cells above
        it are unowned (and land at every basin's level, so nothing is lost) — the
        cap is what keeps the flood proportional to the interesting part of the
        scene.  Raise it if you need a saddle that stands far above every level;
        a saddle above the ceiling is reported as NaN, never guessed.
    corridor : {"ramp", "step"}
    sill_expectations : mapping {(name, name): metres} | None
        Independently measured saddle elevations, e.g. the Manych sill read
        straight off native-resolution GEBCO.  Each is compared against what THIS
        grid resolves and :class:`SillResolutionError` is raised on a
        disagreement wider than *sill_tolerance_m* — see that exception for the
        13.5 m of resampling-induced sill rise that motivates it.  Key order does
        not matter.  A saddle the flood never reached also raises: "I could not
        see it" must not read as "it agrees".
    sill_tolerance_m : float
        Default 2.0 m.

    Raises
    ------
    ValueError
        No seeds; duplicate basin names; a seed off the raster; a seed on nodata;
        an expectation naming a basin that is not in *seeds*; or a seed standing
        ABOVE its own level — that last one would quietly produce an empty basin,
        which is precisely the class of silent wrongness this project refuses.
    SillResolutionError
        A sill this grid resolves disagrees with a supplied measurement.
    CorridorNotRampableError
        ``corridor="ramp"`` with more than one spilling pair.
    """
    if corridor not in ("ramp", "step"):
        raise ValueError(f"corridor={corridor!r} is not one of 'ramp', 'step'")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("assign_basins needs at least one BasinSeed")
    names = [s.name for s in seeds]
    if len(set(names)) != len(names):
        raise ValueError(f"basin names must be unique, got {names!r}")

    dem = np.asarray(dem, dtype=np.float64)
    height, width = dem.shape
    seed_rc: list[tuple[int, int]] = []
    for seed in seeds:
        row, col = _seed_rowcol(transform, seed.lon, seed.lat, (height, width), seed.name)
        value = dem[row, col]
        if not np.isfinite(value):
            raise ValueError(
                f"basin seed {seed.name!r} at {seed.lon:.4f} E / {seed.lat:.4f} N "
                "lands on a nodata cell of the DEM."
            )
        if value >= seed.level_m:
            raise ValueError(
                f"basin seed {seed.name!r} sits at {value:+.2f} m, which is above "
                f"its own level {seed.level_m:+.2f} m. A seed must be under its "
                "basin's water; on dry land it grows an empty basin and the frame "
                "silently loses a sea."
            )
        seed_rc.append((row, col))
    if len(set(seed_rc)) != len(seed_rc):
        raise ValueError(
            "two basin seeds fall in the same DEM cell; they cannot be told apart"
        )

    levels = [float(s.level_m) for s in seeds]
    ceiling = float(max(levels) + ceiling_margin_m)
    basin_id, bottleneck = _priority_flood_ownership(dem, seed_rc, ceiling)

    # ── Sills, measured ──────────────────────────────────────────────────────
    touching = _measure_sills(basin_id, bottleneck, len(seeds))
    sill_m: dict[tuple[str, str], float] = {}
    sill_lonlat: dict[tuple[str, str], tuple[float, float] | None] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            key = tuple(sorted((names[i], names[j])))
            found = touching.get((i + 1, j + 1))
            if found is None:
                sill_m[key] = float("nan")
                sill_lonlat[key] = None
            else:
                value, (row, col) = found
                sill_m[key] = value
                lon, lat = transform * (col + 0.5, row + 0.5)
                sill_lonlat[key] = (float(lon), float(lat))

    for raw_pair, expected in (sill_expectations or {}).items():
        key = tuple(sorted(raw_pair))
        if key not in sill_m:
            raise ValueError(
                f"sill_expectations key {tuple(raw_pair)!r} is not a basin pair of "
                f"this call; the basins are {tuple(names)!r}."
            )
        measured = sill_m[key]
        if not np.isfinite(measured):
            raise SillResolutionError(
                f"a sill of {float(expected):+.2f} m was supplied for "
                f"{key[0]}<->{key[1]}, but their territories never met: the "
                f"saddle stands above this call's flood ceiling of "
                f"{ceiling:+.2f} m. Raise ceiling_margin_m rather than reading "
                "'not measured' as 'agrees'."
            )
        if abs(measured - float(expected)) > sill_tolerance_m:
            raise SillResolutionError(
                f"this grid resolves the {key[0]}<->{key[1]} sill at "
                f"{measured:+.2f} m, but it was measured at "
                f"{float(expected):+.2f} m — a gap of "
                f"{measured - float(expected):+.2f} m, wider than the "
                f"{sill_tolerance_m:.2f} m tolerance. A spillway narrower than "
                "the working pixel is filled in by resampling and the sill rises, "
                "so a frame built on this grid can report a basin as endorheic "
                "when the terrain spills. Render at a resolution that resolves the "
                "channel, or widen sill_tolerance_m in writing."
            )

    # ── Spill, observed ──────────────────────────────────────────────────────
    # Two basins are connected iff filling the HIGHER one to its own level reaches
    # the other one's seed.  The lower basin's own fill can never cross the sill,
    # so this single directed test is the whole of the symmetric question.
    component_cache: dict[float, np.ndarray] = {}

    def components(level: float) -> np.ndarray:
        cached = component_cache.get(level)
        if cached is None:
            cached, _ = ndi_label(np.isfinite(dem) & (dem < level))
            component_cache[level] = cached
        return cached

    merged_pairs: list[tuple[int, int]] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            labelled = components(max(levels[i], levels[j]))
            a = labelled[seed_rc[i]]
            if a != 0 and a == labelled[seed_rc[j]]:
                merged_pairs.append((i, j))

    parent = list(range(len(seeds)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merged_pairs:
        parent[find(i)] = find(j)
    grouped: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        grouped.setdefault(find(i), []).append(name)
    connected_groups = tuple(sorted(tuple(sorted(g)) for g in grouped.values()))

    # ── Water surface ────────────────────────────────────────────────────────
    level_lookup = np.concatenate(([np.nan], np.asarray(levels, dtype=np.float64)))
    water_level = level_lookup[basin_id]
    corridor_mask = np.zeros(dem.shape, dtype=bool)
    corridor_mode = "none"

    if merged_pairs:
        if corridor == "ramp" and len(merged_pairs) > 1:
            spelled = ", ".join(f"{names[i]}<->{names[j]}" for i, j in merged_pairs)
            raise CorridorNotRampableError(
                f"{len(merged_pairs)} basin pairs are connected at this epoch "
                f"({spelled}). The linear ramp is defined for ONE donor->receiver "
                "corridor; with several the surface in each depends on how the "
                "discharge divides, which this module does not model. Pass "
                "corridor='step' to render every basin flat at its own level with "
                "the corridors marked."
            )
        i, j = merged_pairs[0]
        donor, receiver = (i, j) if levels[i] >= levels[j] else (j, i)
        if levels[donor] > levels[receiver]:
            spill_labels = components(levels[donor])
            extent = spill_labels == spill_labels[seed_rc[donor]]
            donor_water = extent & (basin_id == donor + 1)
            receiver_labels = components(levels[receiver])
            receiver_water = receiver_labels == receiver_labels[seed_rc[receiver]]
            candidates = extent & ~donor_water & ~receiver_water

            if candidates.any():
                if corridor == "step":
                    corridor_mode = "step"
                    corridor_mask = candidates
                else:
                    corridor_mode = "ramp"
                    d_donor = _geodesic_hops(candidates, donor_water)
                    d_receiver = _geodesic_hops(candidates, receiver_water)
                    reach_d = np.isfinite(d_donor)
                    reach_r = np.isfinite(d_receiver)
                    both = reach_d & reach_r
                    frac = np.full(dem.shape, np.nan)
                    frac[both] = d_donor[both] / (d_donor[both] + d_receiver[both])
                    # A pocket that only the donor can reach is a BAY of the donor
                    # lake and stands at the donor's level; one only the receiver
                    # can reach never saw the spill at all.
                    frac[reach_d & ~reach_r] = 0.0
                    frac[~reach_d & reach_r] = 1.0
                    ramp = levels[donor] + (levels[receiver] - levels[donor]) * frac
                    ramped = candidates & np.isfinite(frac)
                    water_level = np.where(ramped, ramp, water_level)
                    corridor_mask = ramped & (dem < water_level)

    water_mask = np.isfinite(dem) & (dem < water_level)

    return BasinField(
        basin_id=basin_id,
        names=tuple(names),
        levels_m=tuple(levels),
        bottleneck_m=bottleneck,
        water_level_m=water_level,
        water_mask=water_mask,
        corridor_mask=corridor_mask,
        spilling=bool(merged_pairs),
        connected_groups=connected_groups,
        sill_m=sill_m,
        sill_lonlat=sill_lonlat,
        corridor_mode=corridor_mode,
        ceiling_m=ceiling,
    )


def build_epoch_mosaic(
    tile_paths: dict[str, str | Path],
    config: Optional[RenderConfig] = None,
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, BoundingBox, object, float]:
    """Merge per-tile GeoTIFFs into a single down-sampled mosaic array.

    Tile boundary seams are smoothed with a 1-D Gaussian filter applied in
    strips along each integer degree boundary.

    Parameters
    ----------
    tile_paths : dict[str, str | Path]
        Dict whose *values* are paths to paleo-DEM GeoTIFFs.  Ice-sheet side-car
        files are discovered automatically as ``<tile>_ice.tif``.
    config : RenderConfig | None
        Render configuration.  Defaults to :class:`RenderConfig` with all
        defaults if ``None``.

    Returns
    -------
    dem : np.ndarray[float32], shape (H, W)
        Merged DEM in metres (``np.nan`` for nodata).
    ice : np.ndarray[float32], shape (H, W)
        Ice-thickness array in metres (zeros where no sidecar exists).
    transform : rasterio.transform.Affine
        Affine transform of the mosaic.
    bounds : rasterio.coords.BoundingBox
        Geographic extent of the mosaic.
    crs
        CRS of the mosaic.
    resolution : float
        Pixel size in degrees.

    Raises
    ------
    FileNotFoundError
        If no valid tile files are found.
    """
    cfg = config or RenderConfig()

    valid = [Path(f) for f in tile_paths.values() if Path(f).exists()]
    if not valid:
        raise FileNotFoundError("No valid tiles found in tile_paths.")

    dem_ds = [rasterio.open(f) for f in valid]
    try:
        crs = dem_ds[0].crs
        west = min(d.bounds.left for d in dem_ds)
        east = max(d.bounds.right for d in dem_ds)
        south = min(d.bounds.bottom for d in dem_ds)
        north = max(d.bounds.top for d in dem_ds)
        bounds = BoundingBox(west, south, east, north)

        nat_res = dem_ds[0].res[0]
        n_px = max((east - west) / nat_res, (north - south) / nat_res)
        step = max(1.0, n_px / cfg.max_px)
        tgt_res = nat_res * step

        mosaic, xform = rasterio_merge(
            dem_ds, res=tgt_res, resampling=RIOResampling.bilinear, nodata=-9999.0
        )
    finally:
        for d in dem_ds:
            d.close()

    mosaic = mosaic[0].astype(np.float32)
    mosaic[mosaic <= -9_000.0] = np.nan
    H, W = mosaic.shape

    # ── ICE sidecars ────────────────────────────────────────────────────────
    ice_paths = [
        Path(str(f).replace(".tif", "_ice.tif"))
        for f in valid
        if Path(str(f).replace(".tif", "_ice.tif")).exists()
    ]
    if ice_paths:
        ice_ds = [rasterio.open(f) for f in ice_paths]
        try:
            ice_m, _ = rasterio_merge(
                ice_ds, res=tgt_res, resampling=RIOResampling.bilinear, nodata=0.0
            )
        finally:
            for d in ice_ds:
                d.close()
        ice = np.clip(ice_m[0].astype(np.float32), 0.0, None)
    else:
        ice = np.zeros((H, W), dtype=np.float32)

    # ── Tile-seam Gaussian blending ─────────────────────────────────────────
    sig = cfg.seam_sigma_px
    hw = int(np.ceil(sig * 3))
    dem_s = mosaic.copy()
    nd = np.isnan(dem_s)
    dem_s[nd] = 0.0

    for lat in range(int(np.ceil(south)), int(north) + 1):
        r = int(round((north - lat) / tgt_res))
        r0, r1 = max(0, r - hw), min(H, r + hw + 1)
        if r0 < r1:
            dem_s[r0:r1, :] = gaussian_filter1d(dem_s[r0:r1, :], sigma=sig, axis=0)

    for lon in range(int(np.ceil(west)), int(east) + 1):
        c = int(round((lon - west) / tgt_res))
        c0, c1 = max(0, c - hw), min(W, c + hw + 1)
        if c0 < c1:
            dem_s[:, c0:c1] = gaussian_filter1d(dem_s[:, c0:c1], sigma=sig, axis=1)

    mosaic = np.where(nd, np.nan, dem_s)
    del dem_s
    gc.collect()

    return mosaic, ice, xform, bounds, crs, tgt_res


def render_paleocostline_epoch(
    epoch_ka: float,
    tile_paths: dict[str, str | Path],
    sea_level_csv: Path,
    paleocoastlines_shp: Path,
    output_dir: Path,
    config: Optional[RenderConfig] = None,
    *,
    data_source_label: str = "fusion",
    preloaded_coast_gdf: Optional[gpd.GeoDataFrame] = None,
    sea_level_override_m: Optional[float] = None,
    force_dem_mask: bool = False,
    basin_seeds: Optional[Sequence[BasinSeed]] = None,
    basin_ceiling_margin_m: float = 100.0,
    basin_corridor: str = "ramp",
    basin_sill_expectations: Optional[dict] = None,
    basin_sill_tolerance_m: float = 2.0,
) -> Path:
    """Render a composite paleo-topography image for one epoch.

    This is the top-level convenience function.  For batch rendering multiple
    epochs, call :func:`load_paleocostlines` once and pass the result via
    *preloaded_coast_gdf* to avoid re-reading the shapefile on each epoch.

    Parameters
    ----------
    epoch_ka : float
        Epoch to render in ka BP (e.g. ``21.0``).
    tile_paths : dict[str, str | Path]
        Dict mapping tile IDs to file paths of paleo-DEM GeoTIFFs for this
        epoch.  Ice-sheet sidecars (``<stem>_ice.tif``) are auto-detected.
    sea_level_csv : Path
        Path to ``spratt_lisiecki_2016_simplified.csv``.
    paleocoastlines_shp : Path
        Path to ``Paleocoastlines.shp``.
    output_dir : Path
        Directory where the output PNG will be written.
    config : RenderConfig | None
        Render parameters.  Uses defaults if ``None``.
    data_source_label : str
        Short label embedded in the figure title (e.g. ``"fusion"``).
    preloaded_coast_gdf : gpd.GeoDataFrame | None
        Pre-loaded Paleocoastlines GeoDataFrame (from a previous call to
        :func:`load_paleocostlines`).  If ``None`` the shapefile is loaded
        inside this call.
    sea_level_override_m : float | None
        Per-basin sea-level override in metres vs present global sea level
        (negative = below today).  When ``None`` (default) the global
        Spratt & Lisiecki (2016) value is used — identical, backward-compatible
        behaviour.  When given, this value (NOT the Spratt value) selects the
        coastline step via :func:`filter_paleocostlines_for_epoch`, and the
        override is recorded in the figure title / ``mask_src`` label so the
        frame is self-documenting.

        This exists because a sill-dammed basin (e.g. the Black Sea, an
        isolated lake until ~9 ka) must NOT inherit the global eustatic curve
        inside the decoupled basin — it has its own cited level curve.  Pass
        that basin's level for *epoch_ka* here.
    force_dem_mask : bool
        When True, skip the vector Paleocoastlines and derive land/sea directly
        from the paleo-DEM against ``selection_sl`` (``dem < selection_sl``).  For a
        decoupled basin this makes the shoreline follow the cited level continuously
        instead of snapping to the coarse global-eustatic vector steps (which barely
        move across a -30 to -115 m sweep and mix in the wrong open-ocean basin).
    basin_seeds : sequence of BasinSeed | None
        **Opt-in multi-basin path.**  ``None`` (default) leaves every line of the
        single-level behaviour above exactly as it was.  When given, each cell is
        thresholded against ITS OWN basin's level instead of one scalar, so two
        basins standing at different levels — the Caspian near +35 m and the Pontic
        near −30 m — can share one frame, with a ramped water surface across the
        spillway between them.  See :func:`assign_basins` for how membership, the
        sill and the spill are derived from the terrain, and for what the corridor
        ramp is and is not.

        Levels must already be in the paleo-DEM's vertical frame (F1).  This is the
        DEM-driven path by construction: the vector Paleocoastlines carry one global
        step per polygon and cannot express two surfaces, so they are not consulted.
        Mutually exclusive with *sea_level_override_m*.

        A machine-readable sidecar ``<png stem>.basins.json`` is written next to the
        PNG recording the levels, the measured sill(s), and whether the basins were
        observed to be spilling — the animation needs that per frame, and a figure
        is not a data format.
    basin_ceiling_margin_m : float
        Passed through to :func:`assign_basins`.
    basin_corridor : {"ramp", "step"}
        Passed through to :func:`assign_basins`.
    basin_sill_expectations, basin_sill_tolerance_m
        Passed through to :func:`assign_basins`.  **Use them for any frame whose
        subject is a spillway**: the mosaic this function builds is down-sampled,
        seam-blended and gaussian-smoothed, and a channel narrower than the
        resulting pixel is filled in — measured at +13.5 m of spurious sill rise
        across the Manych.  See :class:`SillResolutionError`.

    Returns
    -------
    Path
        Absolute path of the saved PNG file.

    Raises
    ------
    FileNotFoundError
        If no valid tiles are found for this epoch.
    ValueError
        If both *basin_seeds* and *sea_level_override_m* are given.
    """
    cfg = config or RenderConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sea level & coastline ────────────────────────────────────────────────
    # The global Spratt & Lisiecki value is always computed (it is the default
    # and remains the basis for the global-ocean datum).  A per-basin override,
    # when supplied, REPLACES it for the coastline-step selection only — this is
    # how a sill-dammed basin (e.g. the Black Sea, isolated until ~9 ka) is
    # driven by its own cited level curve instead of the global eustatic curve.
    if basin_seeds is not None and sea_level_override_m is not None:
        raise ValueError(
            "render_paleocostline_epoch got BOTH basin_seeds and "
            "sea_level_override_m. They are two different answers to 'what is the "
            "water surface here' and there is no defensible way to combine them: "
            "put every basin's level in its own BasinSeed."
        )

    spratt_sl = get_sea_level_for_epoch(epoch_ka, sea_level_csv)

    if sea_level_override_m is not None:
        selection_sl = float(sea_level_override_m)
        sl_source_label = f"basin-override {selection_sl:+.0f} m"
    else:
        selection_sl = spratt_sl
        sl_source_label = None  # default: label derived from the matched step

    if basin_seeds is None:
        coast_gdf = (
            preloaded_coast_gdf
            if preloaded_coast_gdf is not None
            else load_paleocostlines(paleocoastlines_shp)
        )
        land_gdf, coast_level = filter_paleocostlines_for_epoch(coast_gdf, selection_sl)
    else:
        # The vector Paleocoastlines carry ONE global sea-level step per polygon.
        # A frame with two basins at two levels has no such step, so the vector
        # path is not "unavailable" here — it is inapplicable, and saying so is
        # cheaper than loading a shapefile whose answer would be discarded.
        land_gdf, coast_level = None, None

    # ── DEM mosaic ───────────────────────────────────────────────────────────
    dem, ice, xform, bounds, crs, _res = build_epoch_mosaic(tile_paths, cfg)

    # ── Compositing ─────────────────────────────────────────────────────────
    H, W = dem.shape
    nodata = np.isnan(dem)

    dem_sm = gaussian_filter(np.where(nodata, 0.0, dem), sigma=1.5)
    dem_sm = np.where(nodata, np.nan, dem_sm)
    dem_f = np.where(nodata, 0.0, dem)
    dem_fs = np.where(nodata, 0.0, dem_sm)

    # Ice mask from sidecar
    ice_mask = (~nodata) & (ice > cfg.ice_threshold_m)

    # ── Ice dome smoothing ────────────────────────────────────────────────────
    # Production IceThicknessModel smooths the ice surface before reprojection:
    # gaussian_filter(sigma=0.7 on 1°-grid) → cubic bicubic reproject.
    # At tile resolution the equivalent is a large-sigma smooth on the DEM that
    # replaces the jagged bedrock texture under the glacier with a smooth dome.
    # This is the most visible part of the pipeline — without it the ice surface
    # shows fjord/mountain topography instead of a clean dome shape.
    if ice_mask.any():
        # Heavily smooth the ice surface to remove bedrock texture
        dem_ice_input = np.where(nodata, 0.0, dem)
        dem_ice_dome = gaussian_filter(dem_ice_input, sigma=cfg.ice_dome_sigma_px)
        # Blend: full dome smoothing where ice is thick, taper at margins
        ice_h_norm = np.clip(ice / max(float(ice[ice_mask].max()), 1.0), 0.0, 1.0)
        dem_dome_blend = np.where(
            ice_mask,
            dem_ice_dome * ice_h_norm + dem_f * (1.0 - ice_h_norm),
            dem_f,
        )
        # Use blended dome surface for hillshade only on ice pixels
        dem_fs = np.where(ice_mask, dem_dome_blend, dem_fs)
        del dem_ice_input, dem_ice_dome, dem_dome_blend, ice_h_norm

    # Land / sea mask. force_dem_mask -> DEM-driven at selection_sl (basin path);
    # otherwise vector first, DEM fallback (also at selection_sl, not a hard 0 m).
    vector_land = None if force_dem_mask else rasterize_land_mask(
        land_gdf, xform, H, W, crs, bounds
    )

    basin_field: Optional[BasinField] = None
    if basin_seeds is not None:
        # Ownership, sill and mask are all derived from the SAME array the
        # single-basin path thresholds (dem_sm), so they cannot disagree.
        basin_field = assign_basins(
            dem_sm,
            xform,
            basin_seeds,
            ceiling_margin_m=basin_ceiling_margin_m,
            corridor=basin_corridor,
            sill_expectations=basin_sill_expectations,
            sill_tolerance_m=basin_sill_tolerance_m,
        )
        ocean_mask = dem_threshold_ocean(dem_sm, nodata, basin_field.water_level_m)
        # The de-speckle above is a single-pixel-noise filter; the spill corridor is
        # the SUBJECT of this frame and a narrow one must not be opened away, so it
        # is restored verbatim.
        ocean_mask = ocean_mask | (basin_field.corridor_mask & (~nodata))
        land_mask = (~nodata) & (~ocean_mask)
        mask_src = "DEM contour, per-basin: " + " / ".join(
            f"{name} {level:+.2f} m"
            for name, level in zip(basin_field.names, basin_field.levels_m)
        )
    elif vector_land is not None:
        land_mask = (~nodata) & vector_land
        ocean_mask = (~nodata) & (~vector_land)
        mask_src = f"Paleocoastlines {coast_level:+d} m"
        if sl_source_label is not None:
            mask_src += f" [{sl_source_label}]"
    else:
        ocean_mask = dem_threshold_ocean(dem_sm, nodata, selection_sl)
        land_mask = (~nodata) & (~ocean_mask)
        mask_src = (
            f"DEM contour {selection_sl:+.0f} m"
            if force_dem_mask
            else f"DEM threshold {selection_sl:+.0f} m (vector unavailable)"
        )
        if sl_source_label is not None:
            mask_src += f" [{sl_source_label}]"

    # Hillshade
    cos_lat = np.cos(np.radians((bounds.bottom + bounds.top) / 2.0))
    dx_m = abs(xform.a) * 111_320.0 * cos_lat
    dy_m = abs(xform.e) * 111_320.0
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_fs, vert_exag=cfg.vert_exag, dx=dx_m, dy=dy_m)
    hs3 = np.stack([hs] * 3, axis=-1)

    # Land colour
    # Exclude ice-covered pixels from the elevation normalization so that ice
    # elevations (e.g. 3000-5400 m at LGM) don't compress the land colour range.
    land_non_ice = land_mask & (~ice_mask)
    if land_non_ice.any():
        lv_min = float(np.nanpercentile(dem[land_non_ice], cfg.land_elev_percentile_lo))
        lv_max = float(np.nanpercentile(dem[land_non_ice], cfg.land_elev_percentile_hi))
    elif land_mask.any():
        lv_min = float(np.nanpercentile(dem[land_mask], cfg.land_elev_percentile_lo))
        lv_max = float(np.nanpercentile(dem[land_mask], cfg.land_elev_percentile_hi))
    else:
        lv_min, lv_max = 0.0, 800.0
    l_norm = np.clip((dem_f - lv_min) / max(lv_max - lv_min, 1.0), 0.0, 1.0)
    l_base = plt.cm.terrain(l_norm * 0.57 + 0.25)[..., :3]
    land_rgb = np.clip(l_base * (hs3 * 0.72 + 0.28), 0.0, 1.0)

    # Ocean colour — sqrt-stretched depth BELOW THE EPOCH SURFACE (selection_sl),
    # so a lowered decoupled basin reads as shallower, not as deep as today.
    water_surface = (
        selection_sl if basin_field is None else basin_field.water_level_m
    )
    depth = ocean_depth_from_surface(dem_f, ocean_mask, water_surface)
    if ocean_mask.any():
        max_depth = max(float(np.percentile(depth[ocean_mask], 95)), 1.0)
    else:
        max_depth = 1.0
    depth_norm = np.clip(np.sqrt(depth / max_depth), 0.0, 1.0)
    ocean_rgb = plt.cm.Blues(depth_norm * 0.60 + 0.30)[..., :3]

    # Ice colour — snow/blue-ice palette driven by hillshade from smoothed dome
    # Matches the visual convention used in paleo-DEM publications:
    # thin ice → steel-blue, thick ice → white, with subtle hillshade shading.
    if ice_mask.any():
        # Ice hillshade already baked into hs3 (computed from smoothed dem_fs)
        # Normalise ice thickness: 0 (threshold) → 1 (max) → maps to colour range
        ice_depth_norm = np.clip(
            ice / max(float(ice[ice_mask].max()), 1.0), 0.0, 1.0
        )
        # plt.cm.Blues_r: low value = dark blue, high = white
        # We map 0→0.3 (thin ice = steel blue) … 1→0.92 (thick ice ≈ white)
        ice_cmap_val = ice_depth_norm * 0.62 + 0.30
        ice_base_rgb = plt.cm.Blues_r(ice_cmap_val)[..., :3]
        # Blend with hillshade (weaker exaggeration on ice → 0.5 ambient + 0.5 shaded)
        ice_rgb = np.clip(ice_base_rgb * (hs3 * 0.55 + 0.45), 0.0, 1.0)
    else:
        ice_rgb = None

    # Composite RGBA
    composite = np.full((H, W, 4), [0.06, 0.06, 0.06, 1.0], dtype=np.float32)
    composite[ocean_mask, :3] = ocean_rgb[ocean_mask]
    composite[land_mask, :3] = land_rgb[land_mask]
    if ice_mask.any() and ice_rgb is not None:
        composite[ice_mask, :3] = ice_rgb[ice_mask]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(cfg.fig_size_in, cfg.fig_size_in), dpi=cfg.dpi
    )
    ax.imshow(
        composite,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="equal",
        interpolation="bilinear",
    )
    del composite

    if ocean_mask.any():
        sm = plt.cm.ScalarMappable(
            cmap="Blues",
            norm=mcolors.Normalize(vmin=0.0, vmax=max_depth),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
        depth_label = (
            "Depth below basin surface (m)"
            if (sea_level_override_m is not None or basin_field is not None)
            else "Depth below sea level (m)"
        )
        cb.set_label(depth_label, fontsize=11)

    n_valid = max(int((~nodata).sum()), 1)
    pct_land = land_mask.sum() / n_valid * 100
    pct_sea = ocean_mask.sum() / n_valid * 100
    pct_ice = ice_mask.sum() / n_valid * 100

    legend_items = [
        Line2D([0], [0], color="#2a6db5", lw=6,
               label=(f"{'Lake' if sea_level_override_m is not None else 'Ocean'}"
                      f"  ({mask_src})")),
        Line2D([0], [0], color="#6b8f3a", lw=6, label="Land  (hillshade + terrain)"),
    ]
    if basin_field is not None:
        legend_items.append(
            Line2D([0], [0], color="#2a6db5", lw=6,
                   label=_basin_legend_label(basin_field))
        )
    if ice_mask.any():
        legend_items.append(
            Line2D([0], [0], color="#c6e2f7", lw=6,
                   label=f"Ice sheet  (ICE-6G sice > {cfg.ice_threshold_m:.0f} m, "
                         f"dome-smoothed σ={cfg.ice_dome_sigma_px:.0f} px)")
        )

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=9,
        framealpha=0.75,
        facecolor="#1a1a1a",
        labelcolor="white",
    )

    title_lines = [
        f"Paleo Render  –  {epoch_ka} ka BP  [{data_source_label.upper()}]",
        f"Coastline: {mask_src}  |  GIA + {cfg.ice_model_label} corrected DEM",
        (
            f"land={pct_land:.0f}%  sea={pct_sea:.0f}%  ice={pct_ice:.0f}%  |  "
            f"{bounds.left:.1f}–{bounds.right:.1f}°E  "
            f"{bounds.bottom:.1f}–{bounds.top:.1f}°N"
        ),
    ]
    if basin_field is not None:
        title_lines.append(_basin_title_line(basin_field))
    if cfg.extra_figure_title:
        title_lines.append(cfg.extra_figure_title)
    ax.set_title("\n".join(title_lines), fontsize=10, pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    out_path = output_dir / f"paleo_render_{epoch_ka}ka.png"
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    if basin_field is not None:
        _write_basin_sidecar(
            out_path.parent / f"{out_path.stem}.basins.json",
            epoch_ka,
            basin_field,
            nodata,
        )

    gc.collect()
    return out_path.resolve()


def _basin_title_line(field: BasinField) -> str:
    """One-line summary of the multi-basin state, for the figure title."""
    state = "SPILLING" if field.spilling else "endorheic"
    groups = " | ".join("+".join(g) for g in field.connected_groups)
    sills = ", ".join(
        f"{'~'.join(pair)} sill "
        + ("above flood ceiling" if not np.isfinite(value) else f"{value:+.2f} m")
        for pair, value in sorted(field.sill_m.items())
    )
    return f"Basins: {state} ({groups})  |  {sills}  |  corridor: {field.corridor_mode}"


def _basin_legend_label(field: BasinField) -> str:
    corridor = {
        "ramp": "corridor: linear water-surface ramp (not hydraulics)",
        "step": "corridor: MARKED, not surfaced (step)",
        "none": "no corridor",
    }[field.corridor_mode]
    return f"{'spilling' if field.spilling else 'basins separate'}  ({corridor})"


def _write_basin_sidecar(
    path: Path,
    epoch_ka: float,
    field: BasinField,
    nodata: np.ndarray,
) -> None:
    """Record the multi-basin state next to the PNG, machine-readably.

    A figure is not a data format.  The animation needs, per frame, which basins
    existed, at what level, whether they were connected, and where the sill was —
    so it is written out rather than left to be re-derived or read off a picture.
    """
    valid = int((~nodata).sum())
    payload = {
        "epoch_ka": float(epoch_ka),
        "spilling": bool(field.spilling),
        "connected_groups": [list(g) for g in field.connected_groups],
        "flood_ceiling_m": float(field.ceiling_m),
        "basins": [
            {
                "name": name,
                "level_m": float(level),
                "cells_owned": int((field.basin_id == i + 1).sum()),
                "cells_water": int(
                    (field.water_mask & (field.basin_id == i + 1)).sum()
                ),
            }
            for i, (name, level) in enumerate(zip(field.names, field.levels_m))
        ],
        "sills": [
            {
                "pair": list(pair),
                "elevation_m": (None if not np.isfinite(value) else float(value)),
                "lon": (
                    None
                    if field.sill_lonlat.get(pair) is None
                    else field.sill_lonlat[pair][0]
                ),
                "lat": (
                    None
                    if field.sill_lonlat.get(pair) is None
                    else field.sill_lonlat[pair][1]
                ),
                "note": (
                    "measured from this frame's DEM by seeded priority flood; "
                    "null = the saddle stands above the flood ceiling and was NOT "
                    "guessed"
                ),
            }
            for pair, value in sorted(field.sill_m.items())
        ],
        "corridor": {
            "mode": field.corridor_mode,
            "cells": int(field.corridor_mask.sum()),
            "note": (
                "ramp = water surface interpolated linearly in along-channel graph "
                "distance between the donor and receiver levels; it is a "
                "first-order water-surface profile, NOT a hydraulic model (no "
                "discharge, no roughness, no backwater). step = every basin flat "
                "at its own level and the corridor only marked."
            ),
        },
        "valid_cells": valid,
        "vertical_frame": (
            "basin levels are consumed AS GIVEN, in the DEM's own frame (F1). No "
            "frame conversion happens here; see "
            "Paleo-Europe_3D/docs/superpowers/research/"
            "2026-07-30-vertical-datum-contract.md"
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
