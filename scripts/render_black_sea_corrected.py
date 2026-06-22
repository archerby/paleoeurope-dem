#!/usr/bin/env python
"""
render_black_sea_corrected.py
=============================

Corrected Black Sea (Pontic / Neoeuxinian basin) paleogeographic map.

WHY
---
The existing ``black_sea`` 2D render series drove the land/sea boundary inside
the decoupled Pontic basin with the **global eustatic** (Spratt & Lisiecki,
2016) sea level.  That is scientifically wrong: the Black Sea was a
**sill-dammed lake**, isolated from the global ocean until reconnection at
~9 ka.  Its water level followed its OWN curve — sometimes far above, sometimes
far below the global eustatic curve during the deglacial window.

This script renders a side-by-side comparison, for a set of divergent epochs,
of the WRONG global-eustatic coastline vs the CORRECT Pontic-curve coastline,
using ONLY the real Paleocoastlines.shp vector steps (no DEM — the Black Sea
DEM tiles are not available locally).

The Pontic level curve is the cited synthesis
``2026-06-21-black-sea-level-curve.csv`` (gradual reconnection scenario, the
default ``level_m_global_gradual`` column; m vs present global sea level,
negative = below).

OUTPUT
------
A grid PNG (rows = epochs, 2 columns: WRONG global | CORRECT Pontic) written to
``D:/pasha/RESEARCH/renders/black_sea_corrected/``.

Run:
    python scripts/render_black_sea_corrected.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make the package importable when this script is run directly from anywhere
# (the repo is not pip-installed; `paleoeurope/` lives at the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from paleoeurope.viz.paleocostline_render import (
    filter_paleocostlines_for_epoch,
    get_sea_level_for_epoch,
    load_paleocostlines,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PALEOCOASTLINES_SHP = Path(
    r"D:\pasha\RESEARCH\Paleo-Europe_3D\data\raw\paleocostline"
    r"\Paleocoastlines\Paleocoastlines\Paleocoastlines.shp"
)
SPRATT_CSV = Path(
    r"D:\pasha\RESEARCH\renders\paleoeurope-dem"
    r"\data\sea_level_curves\spratt_lisiecki_2016_simplified.csv"
)
BLACK_SEA_CSV = Path(
    r"D:\pasha\RESEARCH\Paleo-Europe_3D\docs\superpowers\research"
    r"\2026-06-21-black-sea-level-curve.csv"
)
OUTPUT_DIR = Path(r"D:\pasha\RESEARCH\renders\black_sea_corrected")

# Black Sea region bounding box (lon0, lat0, lon1, lat1) — from the render titles.
BS_BBOX = (20.0, 33.0, 41.0, 54.0)

# Divergent deglacial epochs + 18 ka lowstand (agreement reference).
EPOCHS_KA = [18, 16, 14, 12, 10, 9]

# Black Sea curve column to use as default (gradual reconnection scenario).
BS_COLUMN = "level_m_global_gradual"

# Colours
LAND_COLOR = "#6b8f3a"   # green
SEA_COLOR = "#2a6db5"    # blue
EDGE_COLOR = "#1d1d1d"
BG_COLOR = "#cfe6f5"     # light water background (everything not land = sea/lake)


def bs_level_for_epoch(bs_df: pd.DataFrame, epoch_ka: float) -> tuple[float, str]:
    """Return (gradual BS level in m vs present global SL, basin_state) for the
    CSV row nearest *epoch_ka*."""
    idx = (bs_df["age_ka"] - epoch_ka).abs().idxmin()
    row = bs_df.loc[idx]
    return float(row[BS_COLUMN]), str(row["basin_state"])


def _plot_panel(ax, land_gdf, level_m, step_m, title_top, source_label, src_color):
    """Render one comparison panel: land polygons over a sea background, fixed
    BS-bbox extent so panels are directly comparable."""
    lon0, lat0, lon1, lat1 = BS_BBOX

    # Sea/lake background fills the whole extent.
    ax.set_facecolor(BG_COLOR)

    if land_gdf is not None and len(land_gdf) > 0:
        land_gdf.plot(
            ax=ax,
            facecolor=LAND_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.2,
        )

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.tick_params(labelsize=7)

    ax.set_title(
        f"{title_top}\n"
        f"level used: {level_m:+.1f} m  →  coastline step {step_m:+d} m\n"
        f"source: {source_label}",
        fontsize=8.5,
        color=src_color,
        pad=6,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"[1/4] Loading Paleocoastlines vector: {PALEOCOASTLINES_SHP}")
    coast_gdf = load_paleocostlines(PALEOCOASTLINES_SHP)
    if coast_gdf is None:
        raise SystemExit("FATAL: could not load Paleocoastlines.shp")

    lon0, lat0, lon1, lat1 = BS_BBOX
    bs_coast = coast_gdf.cx[lon0:lon1, lat0:lat1].copy()
    print(
        f"      total features={len(coast_gdf)}  in BS bbox={len(bs_coast)}  "
        f"steps={sorted(int(x) for x in bs_coast['Sea level'].unique())}"
    )

    print(f"[2/4] Loading Black Sea curve CSV (column '{BS_COLUMN}'): {BLACK_SEA_CSV}")
    bs_df = pd.read_csv(BLACK_SEA_CSV)

    # ── Per-epoch sea levels + coastline steps ───────────────────────────────
    print("[3/4] Computing per-epoch global vs Pontic sea levels & coastline steps")
    print(
        f"\n{'epoch':>6} | {'global_sl':>9} {'g_step':>7} | "
        f"{'bs_sl':>7} {'bs_step':>8} | {'d(step)':>8} | basin_state"
    )
    print("-" * 78)

    rows = []
    for ep in EPOCHS_KA:
        global_sl = get_sea_level_for_epoch(float(ep), SPRATT_CSV)
        bs_sl, state = bs_level_for_epoch(bs_df, float(ep))

        land_global, step_global = filter_paleocostlines_for_epoch(bs_coast, global_sl)
        land_bs, step_bs = filter_paleocostlines_for_epoch(bs_coast, bs_sl)

        d_step = step_bs - step_global
        verdict = (
            "BS deeper" if step_bs < step_global
            else "BS higher" if step_bs > step_global
            else "agree"
        )
        print(
            f"{ep:>4}ka | {global_sl:>9.1f} {step_global:>7d} | "
            f"{bs_sl:>7.1f} {step_bs:>8d} | {d_step:>+8d} | {state}  ({verdict})"
        )
        rows.append(
            dict(
                ep=ep,
                global_sl=global_sl,
                step_global=step_global,
                land_global=land_global,
                bs_sl=bs_sl,
                step_bs=step_bs,
                land_bs=land_bs,
                state=state,
            )
        )
    print()

    # ── Figure: grid (rows = epochs, 2 cols) ─────────────────────────────────
    print("[4/4] Rendering comparison figure")
    n = len(rows)
    fig, axes = plt.subplots(
        n, 2, figsize=(11, 4.4 * n), dpi=130, squeeze=False
    )

    for i, r in enumerate(rows):
        # LEFT: WRONG — global eustatic
        _plot_panel(
            axes[i][0],
            r["land_global"],
            r["global_sl"],
            r["step_global"],
            f"{r['ep']} ka — WRONG: global eustatic in the basin",
            "Spratt-Lisiecki (2016) global",
            "#a11a1a",
        )
        # RIGHT: CORRECT — Pontic curve
        _plot_panel(
            axes[i][1],
            r["land_bs"],
            r["bs_sl"],
            r["step_bs"],
            f"{r['ep']} ka — CORRECT: Pontic basin level "
            f"({r['state']})",
            "Pontic curve 2026-06-21 (gradual)",
            "#1a6b1a",
        )

    legend_handles = [
        Patch(facecolor=LAND_COLOR, edgecolor=EDGE_COLOR, label="Land"),
        Patch(facecolor=BG_COLOR, edgecolor=EDGE_COLOR, label="Sea / lake"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "Black Sea paleogeography corrected for per-basin sea level\n"
        "The Pontic basin was a sill-dammed lake until ~9 ka — driving its "
        "coastline with the GLOBAL eustatic curve (left) is wrong.\n"
        "The cited Pontic level curve (right) keeps the lake near its spill "
        "point during 16-14 ka, drops it to a deep lowstand (~ -80 m) before "
        "reconnection at 10 ka, then couples to the ocean at 9 ka.",
        fontsize=12.5,
        y=0.997,
        va="top",
    )

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.975))

    out_grid = OUTPUT_DIR / "black_sea_corrected_comparison.png"
    fig.savefig(out_grid, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved grid figure → {out_grid.resolve()}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
