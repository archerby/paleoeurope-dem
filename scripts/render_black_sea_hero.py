#!/usr/bin/env python
"""Black Sea coastline-correction HERO figure (real Paleocoastlines data).

For the divergent deglacial epochs, show the Pontic-curve-correct Black Sea
land/sea and SHADE the area where the (wrong) global-eustatic render disagrees.
This makes the deep-walled basin's subtle-but-real shoreline shift visible and
labels the direction of the error (global wrongly drains vs wrongly floods).

No DEM needed — uses only the global Paleocoastlines vector (which covers the BS)
queried at two different sea levels. Full hillshade topo render is gated on the
(missing) BS DEM tiles.
"""
from __future__ import annotations
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio.coords import BoundingBox

from paleoeurope.viz.paleocostline_render import (
    load_paleocostlines,
    filter_paleocostlines_for_epoch,
    rasterize_land_mask,
    get_sea_level_for_epoch,
)

SHP = Path(r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/paleocostline/Paleocoastlines/Paleocoastlines/Paleocoastlines.shp")
SPRATT = Path("data/sea_level_curves/spratt_lisiecki_2016_simplified.csv")
BSCURVE = Path(r"D:/pasha/RESEARCH/Paleo-Europe_3D/docs/superpowers/research/2026-06-21-black-sea-level-curve.csv")
OUT = Path(r"D:/pasha/RESEARCH/renders/black_sea_corrected/black_sea_hero.png")

# Black Sea bbox (from the existing render titles)
W_, S_, E_, N_ = 20.0, 33.0, 41.0, 54.0
EPOCHS = [16.0, 14.0, 12.0, 10.0]  # the deglacial crossover arc
PXDEG = 0.02
WIDTH = int(round((E_ - W_) / PXDEG))
HEIGHT = int(round((N_ - S_) / PXDEG))
TRANSFORM = rasterio.transform.from_bounds(W_, S_, E_, N_, WIDTH, HEIGHT)
BOUNDS = BoundingBox(W_, S_, E_, N_)


def bs_level(epoch_ka: float) -> float:
    df = pd.read_csv(BSCURVE, comment="#")
    i = (df["age_ka"] - epoch_ka).abs().idxmin()
    return float(df.loc[i, "level_m_global_gradual"])


def land_mask_at(gdf, level_m: float):
    land, step = filter_paleocostlines_for_epoch(gdf, level_m)
    m = rasterize_land_mask(land, TRANSFORM, HEIGHT, WIDTH, "EPSG:4326", BOUNDS)
    if m is None:
        m = np.zeros((HEIGHT, WIDTH), bool)
    return m, step


def main() -> None:
    print(f"grid {WIDTH}x{HEIGHT} over {W_}-{E_}E / {S_}-{N_}N")
    gdf = load_paleocostlines(SHP)
    print("coastline features:", len(gdf))

    fig, axes = plt.subplots(1, len(EPOCHS), figsize=(5.2 * len(EPOCHS), 5.8), dpi=130)
    extent = [W_, E_, S_, N_]
    # base colours
    C_LAND = np.array([0.80, 0.74, 0.55])   # tan
    C_SEA = np.array([0.20, 0.40, 0.62])    # deep blue (the lake at the CORRECT level)
    C_FLOOD = np.array([0.85, 0.15, 0.15])  # red: global says SEA, truth=land (global over-floods)
    C_DRAIN = np.array([0.95, 0.65, 0.10])  # orange: global says LAND, truth=sea (global over-drains)

    for ax, ep in zip(np.atleast_1d(axes), EPOCHS):
        g_sl = get_sea_level_for_epoch(ep, SPRATT)
        b_sl = bs_level(ep)
        g_land, g_step = land_mask_at(gdf, g_sl)
        p_land, p_step = land_mask_at(gdf, b_sl)

        over_flood = p_land & (~g_land)   # truth land, global sea  -> global over-floods (red)
        over_drain = g_land & (~p_land)   # truth sea, global land  -> global over-drains (orange)
        diff_px = int(over_flood.sum() + over_drain.sum())
        tot = HEIGHT * WIDTH
        verdict = ("DRAINS the lake" if over_drain.sum() > over_flood.sum()*1.3
                   else "FLOODS the lake" if over_flood.sum() > over_drain.sum()*1.3
                   else "~agrees")

        rgb = np.empty((HEIGHT, WIDTH, 3), np.float32)
        rgb[p_land] = C_LAND
        rgb[~p_land] = C_SEA
        rgb[over_flood] = C_FLOOD
        rgb[over_drain] = C_DRAIN
        ax.imshow(rgb, extent=extent, origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(
            f"{ep:.0f} ka BP   global {g_sl:+.0f} m (step {g_step:+d})  vs  Pontic {b_sl:+.0f} m (step {p_step:+d})\n"
            f"global eustatic render {verdict}   |   disagreement {100*diff_px/tot:.1f}% of frame",
            fontsize=9.5,
        )
        ax.set_xlabel("°E", fontsize=8); ax.set_ylabel("°N", fontsize=8)
        ax.tick_params(labelsize=7)
        print(f"{ep:>5.0f}ka  global {g_sl:+.0f}(step {g_step:+d})  Pontic {b_sl:+.0f}(step {p_step:+d})  "
              f"over_flood={int(over_flood.sum())} over_drain={int(over_drain.sum())}  -> {verdict}")

    legend = [
        Patch(facecolor=C_LAND, label="Land (Pontic-correct)"),
        Patch(facecolor=C_SEA, label="Black Sea lake (correct level)"),
        Patch(facecolor=C_FLOOD, label="Global render WRONGLY floods (was dry lake-bed)"),
        Patch(facecolor=C_DRAIN, label="Global render WRONGLY drains (was under the lake)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Black Sea coastline corrected to the cited Pontic level curve vs the (wrong) global eustatic curve\n"
        "Decoupled Neoeuxinian lake: ~16 ka it brimmed at its −30 m spill point (global wrongly drains it); "
        "by ~10 ka it fell to a deep ~−80 m lowstand (global wrongly floods it). Source: Pontic curve 2026-06-21 (gradual).",
        fontsize=11, y=1.02,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
