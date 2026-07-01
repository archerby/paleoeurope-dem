#!/usr/bin/env python
"""
bake_black_sea.py
=================

Build a GIA-corrected paleo-DEM of the **Black Sea proper** per epoch from the
GMRT bathymetry/topography tile + ICE-6G_C bedrock orography, then render each
epoch's topo frame TWICE:

  * **WRONG**  — using the global eustatic (Spratt & Lisiecki 2016) sea level.
  * **CORRECT** — using the cited Pontic / Neoeuxinian basin level curve.

WHY
---
The Black Sea was a **sill-dammed lake**, decoupled from the global ocean until
reconnection at ~9 ka.  Inside that closed basin the land/sea boundary must be
driven by the basin's OWN level curve, NOT the global eustatic curve.  Whenever
the two curves disagree (e.g. 16 ka: global ~-98 m vs Pontic ~-30 m;
10 ka: global ~-25 m vs Pontic ~-80 m) the corrected coastline differs
dramatically.  This script visualises that scientific correction.

Because the whole scene is Black-Sea-proper water (one decoupled basin), a
single basin level is valid for the entire frame.

PALEO-DEM
---------
paleo_dem(t) = H_modern + Δz_solid(t)   (present-datum solid-earth displacement)
Δz_solid(t)  = [Topo(t) - Topo(0)] - sea_surface_offset(t)   (ICE-6G_C Topo)
sea_surface_offset(t) = median over deep ocean of [Topo(t) - Topo(0)]
    (Orog is clamped to 0 over water -> GIA vanishes under the sea; bare ΔTopo
     double-counts the epoch sea surface. See paleoeurope.gia.ice6g_topo.)
ice_height = 0 everywhere (no ice sheet over the Black Sea).

OUTPUT
------
  D:/pasha/RESEARCH/renders/black_sea_corrected/topo/correct/*.png
  D:/pasha/RESEARCH/renders/black_sea_corrected/topo/wrong/*.png
  D:/pasha/RESEARCH/renders/black_sea_corrected/topo/black_sea_topo_comparison.png

Run (repo NOT pip-installed):
    PYTHONPATH=/d/pasha/RESEARCH/renders/paleoeurope-dem \
        python scripts/bake_black_sea.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run directly (repo is not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.ice6g_topo import implied_sea_surface_offset, solid_displacement
from paleoeurope.viz.paleocostline_render import (
    RenderConfig,
    get_sea_level_for_epoch,
    load_paleocostlines,
    render_paleocostline_epoch,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BS_DEM = Path(
    r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/GEBCO/bs_gmrt_bsproper.tif"
)
ICE6G_DIR = Path(r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/ICE6G")
PONTIC_CSV = Path(
    r"D:/pasha/RESEARCH/Paleo-Europe_3D/docs/superpowers/research"
    r"/2026-06-21-black-sea-level-curve.csv"
)
# Simplified curve: 0 ka -> 0 m (the full CSV has a +8.49 m present-datum offset
# and must NOT be used for the global-eustatic reference frames).
SPRATT_CSV = Path(
    r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/sea_level_curves"
    r"/spratt_lisiecki_2016_simplified.csv"
)
PALEOCOASTLINES_SHP = Path(
    r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/paleocostline"
    r"/Paleocoastlines/Paleocoastlines/Paleocoastlines.shp"
)

OUT_BASE = Path(r"D:/pasha/RESEARCH/renders/black_sea_corrected/topo")
PALEO_TIF_DIR = _REPO_ROOT / "outputs" / "_bs_paleo"

EPOCHS_KA = [18, 16, 12, 10, 9, 0]

# Reconnection scenarios exposed by the cited curve (carry both, assert neither).
SCENARIOS = ("gradual", "catastrophic")

# Region buffer (deg) for the GIA subset around the BS bbox 27-42E / 40-48N.
LAT_MIN, LAT_MAX = 40.0, 48.0
LON_MIN, LON_MAX = 27.0, 42.0
BUF = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_filename(epoch: float) -> str:
    """ICE-6G filename token for an epoch.

    Integers use the bare int (``16`` -> ``...16.nc``); half-ka use one decimal
    (``0.5`` -> ``...0.5.nc``).  ``0`` -> ``...0.nc``.
    """
    if float(epoch).is_integer():
        return f"I6_C.VM5a_10min.{int(epoch)}.nc"
    return f"I6_C.VM5a_10min.{epoch}.nc"


def pontic_level(df: pd.DataFrame, epoch: float, scenario: str = "gradual") -> float:
    """Nearest-age Pontic basin level (m vs present global SL) for *epoch*.

    *scenario* selects the reconnection column (``gradual`` or ``catastrophic``).
    A sign-flip guard asserts the level is at or below present sea level (curve §3):
    a positive value would put the shoreline above the surrounding terrain.
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    idx = (df["age_ka"] - float(epoch)).abs().idxmin()
    level = float(df.loc[idx, col])
    assert level <= 1.0, (
        f"basin level {level:+.1f} m > +1 m at {epoch} ka — sign-flip guard "
        "(curve doc §3)"
    )
    return level


def gia_delta_subset(epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dz_solid, model_lats, model_lons) for the BS region at *epoch*.

    dz_solid = present-datum solid-earth displacement (Topo-based), subset to the
    BS bbox +/- BUF.  The sea-surface offset is computed from the GLOBAL Topo field
    (deep-ocean median) so the datum is internally consistent with ICE-6G.
    """
    f_t = ICE6G_DIR / _epoch_filename(epoch)
    f_0 = ICE6G_DIR / _epoch_filename(0)
    lat_lo, lat_hi = LAT_MIN - BUF, LAT_MAX + BUF
    lon_lo, lon_hi = LON_MIN - BUF, LON_MAX + BUF

    with xr.open_dataset(f_t) as ds_t, xr.open_dataset(f_0) as ds_0:
        # Global sea-surface offset (deep-ocean median of ΔTopo).
        offset = implied_sea_surface_offset(
            ds_t["Topo"].values.astype(np.float64),
            ds_0["Topo"].values.astype(np.float64),
        )
        sub_t = ds_t.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        sub_0 = ds_0.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        topo_t = sub_t["Topo"].values.astype(np.float64)
        topo_0 = sub_0["Topo"].values.astype(np.float64)
        model_lats = sub_t["lat"].values.astype(np.float64)
        model_lons = sub_t["lon"].values.astype(np.float64)

    dz = solid_displacement(topo_t, topo_0, offset).astype(np.float32)  # (M, N)
    return dz, model_lats, model_lons


def build_paleo_dem(epoch: float) -> Path:
    """Bake the GIA-corrected paleo-DEM GeoTIFF for *epoch*; return its path."""
    with rasterio.open(BS_DEM) as ds:
        modern_dem = ds.read(1).astype(np.float32)
        transform = ds.transform
        crs = ds.crs
        width, height = ds.width, ds.height

    # Column centres (ascending) and row centres (DESCENDING north->south).
    cols = np.arange(width)
    rows = np.arange(height)
    tile_lons = (transform * (cols + 0.5, np.zeros_like(cols) + 0.5))[0]
    tile_lats = (transform * (np.zeros_like(rows) + 0.5, rows + 0.5))[1]

    dz, model_lats, model_lons = gia_delta_subset(epoch)
    ice_height = np.zeros_like(dz)

    paleo_dem, _ice = apply_gia_delta(
        modern_dem,
        model_lats,
        model_lons,
        dz,
        ice_height,
        tile_lats,
        tile_lons,
        sigma_blur=2.0,
    )

    PALEO_TIF_DIR.mkdir(parents=True, exist_ok=True)
    out_tif = PALEO_TIF_DIR / f"bs_{epoch}ka.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(paleo_dem.astype(np.float32), 1)

    return out_tif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pontic_df = pd.read_csv(PONTIC_CSV, comment="#")

    out_correct = OUT_BASE / "correct"
    out_wrong = OUT_BASE / "wrong"
    out_correct.mkdir(parents=True, exist_ok=True)
    out_wrong.mkdir(parents=True, exist_ok=True)

    # Load the Paleocoastlines vector ONCE and reuse for every render.
    print("Loading Paleocoastlines vector (once) ...")
    coast_gdf = load_paleocostlines(PALEOCOASTLINES_SHP)
    if coast_gdf is None:
        raise SystemExit("FATAL: Paleocoastlines vector failed to load.")
    avail_levels = sorted(coast_gdf["Sea level"].dropna().unique())
    print(f"  loaded {len(coast_gdf)} polygons; "
          f"{len(avail_levels)} sea-level steps "
          f"[{int(min(avail_levels))}..{int(max(avail_levels))} m]")

    cfg = RenderConfig(max_px=2000)

    print()
    header = (f"{'epoch':>6} | {'global_sl':>10} | {'bs_sl':>8} | "
              f"{'wrong_coast':>12} | {'correct_coast':>14}")
    print(header)
    print("-" * len(header))

    rows = []  # for the montage
    for epoch in EPOCHS_KA:
        paleo_tif = build_paleo_dem(epoch)

        bs_sl = pontic_level(pontic_df, epoch)
        global_sl = get_sea_level_for_epoch(epoch, SPRATT_CSV)

        # Nearest Paleocoastlines step each render will actually snap to.
        wrong_coast = int(min(avail_levels, key=lambda x: abs(x - global_sl)))
        correct_coast = int(min(avail_levels, key=lambda x: abs(x - bs_sl)))

        # CORRECT — Pontic basin override.
        correct_png = render_paleocostline_epoch(
            epoch,
            {"bs": str(paleo_tif)},
            sea_level_csv=SPRATT_CSV,
            paleocoastlines_shp=PALEOCOASTLINES_SHP,
            output_dir=out_correct,
            config=cfg,
            data_source_label="Pontic",
            sea_level_override_m=bs_sl,
            preloaded_coast_gdf=coast_gdf,
        )

        # WRONG — global eustatic (no override).
        wrong_png = render_paleocostline_epoch(
            epoch,
            {"bs": str(paleo_tif)},
            sea_level_csv=SPRATT_CSV,
            paleocoastlines_shp=PALEOCOASTLINES_SHP,
            output_dir=out_wrong,
            config=cfg,
            data_source_label="global-eustatic",
            sea_level_override_m=None,
            preloaded_coast_gdf=coast_gdf,
        )

        print(f"{epoch:>6} | {global_sl:>+10.2f} | {bs_sl:>+8.2f} | "
              f"{wrong_coast:>+12d} | {correct_coast:>+14d}")

        rows.append(
            dict(
                epoch=epoch,
                global_sl=global_sl,
                bs_sl=bs_sl,
                wrong_coast=wrong_coast,
                correct_coast=correct_coast,
                wrong_png=Path(wrong_png),
                correct_png=Path(correct_png),
            )
        )

    # ── Comparison montage ──────────────────────────────────────────────────
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, r in enumerate(rows):
        wrong_img = mpimg.imread(r["wrong_png"])
        correct_img = mpimg.imread(r["correct_png"])

        ax_w = axes[i, 0]
        ax_c = axes[i, 1]
        ax_w.imshow(wrong_img)
        ax_c.imshow(correct_img)
        for ax in (ax_w, ax_c):
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0:
            ax_w.set_title("WRONG — global eustatic", fontsize=13, color="#a11")
            ax_c.set_title("CORRECT — Pontic basin", fontsize=13, color="#161")

        ax_w.set_ylabel(
            f"{r['epoch']} ka\n"
            f"global {r['global_sl']:+.0f} m\n"
            f"(coast {r['wrong_coast']:+d} m)",
            fontsize=10, rotation=0, labelpad=46, va="center",
        )
        # Right-side annotation for the corrected level.
        ax_c.text(
            1.02, 0.5,
            f"Pontic {r['bs_sl']:+.0f} m\n(coast {r['correct_coast']:+d} m)",
            transform=ax_c.transAxes, fontsize=10, va="center", ha="left",
            color="#161",
        )

    fig.suptitle(
        "Black Sea paleo-topography: scientific correction of the decoupled "
        "basin level\n"
        "Left = WRONG global eustatic (Spratt & Lisiecki 2016) applied inside "
        "the sill-dammed lake.\n"
        "Right = CORRECT cited Pontic / Neoeuxinian basin curve "
        "(gradual scenario). GMRT DEM + ICE-6G_C GIA.",
        fontsize=12, y=0.997,
    )
    fig.tight_layout(rect=(0.04, 0.0, 0.92, 0.985))

    montage = OUT_BASE / "black_sea_topo_comparison.png"
    fig.savefig(montage, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Montage written: {montage}")
    print(f"Correct frames:  {out_correct}")
    print(f"Wrong frames:    {out_wrong}")


if __name__ == "__main__":
    main()
