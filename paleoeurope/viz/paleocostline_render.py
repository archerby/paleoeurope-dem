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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    water_level: float,
) -> np.ndarray:
    """Water-column depth below the epoch water surface (m, ``>= 0``) on ocean pixels.

    ``depth = clip(water_level - dem, 0, inf)`` inside ``ocean_mask``, else 0.

    ``water_level`` is the epoch/basin surface in the SAME datum as ``dem``
    (present global SL = 0).  Using the epoch surface — not the present 0 m datum —
    is what makes a lowered decoupled basin read as a *shallower* body, so a
    ~100 m drawdown is visible in the shading instead of looking as deep as today.
    """
    return np.where(ocean_mask, np.clip(water_level - dem, 0.0, None), 0.0)


def dem_threshold_ocean(
    dem_sm: np.ndarray,
    nodata: np.ndarray,
    water_level: float,
) -> np.ndarray:
    """Boolean ocean mask from the DEM: submerged where ``dem_sm < water_level``.

    ``water_level`` is the epoch/basin surface (present-datum).  Morphological
    closing then opening (2 iterations each) removes single-pixel speckle.  This
    is the basin-correct replacement for the old hard ``dem_sm < 0.0`` fallback,
    which silently ignored the override.
    """
    ocean_raw = (~nodata) & (dem_sm < water_level)
    ocean = binary_closing(ocean_raw, iterations=2) & (~nodata)
    ocean = binary_opening(ocean, iterations=2).astype(bool)
    return ocean


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

    Returns
    -------
    Path
        Absolute path of the saved PNG file.

    Raises
    ------
    FileNotFoundError
        If no valid tiles are found for this epoch.
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
    spratt_sl = get_sea_level_for_epoch(epoch_ka, sea_level_csv)

    if sea_level_override_m is not None:
        selection_sl = float(sea_level_override_m)
        sl_source_label = f"basin-override {selection_sl:+.0f} m"
    else:
        selection_sl = spratt_sl
        sl_source_label = None  # default: label derived from the matched step

    coast_gdf = (
        preloaded_coast_gdf
        if preloaded_coast_gdf is not None
        else load_paleocostlines(paleocoastlines_shp)
    )
    land_gdf, coast_level = filter_paleocostlines_for_epoch(coast_gdf, selection_sl)

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

    if vector_land is not None:
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
    depth = ocean_depth_from_surface(dem_f, ocean_mask, selection_sl)
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
            if sea_level_override_m is not None
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
    gc.collect()
    return out_path.resolve()
