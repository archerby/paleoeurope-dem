#!/usr/bin/env python
"""
scripts/prepare_paleocostlines_clip.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clip the global Paleocoastlines SHP (Zickel et al. 2016) to the North Sea /
European shelf area and save as a compact FlatGeobuf for inclusion in the repo.

Typical usage
-------------
python scripts/prepare_paleocostlines_clip.py \\
    --src /path/to/Paleocoastlines/Paleocoastlines/Paleocoastlines.shp \\
    --out data/paleocostlines/paleocoastlines_north_sea.fgb

Requirements
------------
    geopandas >= 0.14, pyogrio >= 0.7, shapely >= 2.0

Output
------
FlatGeobuf with columns: ``Sea level`` (int, m) + ``geometry`` (Polygon/MultiPolygon)
Bounding box: −5°→15°E, 50°→63°N
Simplification: 0.001° ≈ 100 m (preserves features at 1 : 250 000 scale)

Source licence: CC BY 4.0 — Zickel et al. (2016) DOI: 10.5880/SFB806.19
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as shp_box
from shapely.validation import make_valid

# Default clip bounding box: North Sea + Danish / Norwegian Continental Shelf
warnings.filterwarnings("ignore")
DEFAULT_BBOX = (-5.0, 50.0, 15.0, 63.0)   # west, south, east, north
DEFAULT_SIMPLIFY = 0.001                    # degrees ≈ 100 m
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "paleocostlines" / "paleocoastlines_north_sea.fgb"


def prepare_clip(
    src: Path,
    out: Path,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    simplify_tol: float = DEFAULT_SIMPLIFY,
) -> Path:
    """Clip and simplify Paleocoastlines.shp, write FlatGeobuf.

    Parameters
    ----------
    src : Path
        Path to the source ``Paleocoastlines.shp``.
    out : Path
        Output ``.fgb`` path.
    bbox : tuple
        Clip bounding box ``(west, south, east, north)`` in EPSG:4326.
    simplify_tol : float
        Simplification tolerance in degrees.

    Returns
    -------
    Path
        Path to the written output file.
    """
    src, out = Path(src), Path(out)
    if not src.exists():
        raise FileNotFoundError(f"Source SHP not found: {src}")

    print(f"Reading {src.name} (filtered to bbox={bbox})…", file=sys.stderr)
    gdf = gpd.read_file(src, engine="pyogrio", on_invalid="ignore", bbox=bbox)
    print(f"  {len(gdf):,} rows in bbox", file=sys.stderr)

    # Drop null geometries
    gdf = gdf[gdf.geometry.notna()].copy()

    # Repair invalid geometries
    inv = ~gdf.geometry.is_valid
    n_inv = int(inv.sum())
    if n_inv:
        print(f"  Repairing {n_inv} invalid geometries…", file=sys.stderr)
        gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].map(
            lambda g: make_valid(g) if g is not None else g
        )
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[
            gdf.geometry.notna()
            & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()

    # Clip precisely
    print("  Clipping…", file=sys.stderr)
    clipped = gdf.clip(shp_box(*bbox))
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    print(f"  {len(clipped):,} rows after clip", file=sys.stderr)

    # Simplify
    if simplify_tol > 0:
        print(f"  Simplifying at {simplify_tol}°…", file=sys.stderr)
        clipped["geometry"] = clipped.geometry.simplify(simplify_tol, preserve_topology=True)
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()

    # Keep only essential columns
    keep = [c for c in ["Sea level", "geometry"] if c in clipped.columns]
    clipped = clipped[keep].copy()

    # Write
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out, driver="FlatGeobuf")
    size_mb = out.stat().st_size / 1e6
    print(f"Saved: {out}  ({size_mb:.1f} MB)", file=sys.stderr)

    steps = sorted(clipped["Sea level"].dropna().unique().tolist())
    print(f"Sea-level steps: {steps}", file=sys.stderr)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clip global Paleocoastlines SHP to European shelf FlatGeobuf."
    )
    p.add_argument("--src", required=True, type=Path, help="Path to Paleocoastlines.shp")
    p.add_argument("--out", default=DEFAULT_OUT, type=Path, help="Output .fgb path")
    p.add_argument(
        "--bbox",
        nargs=4, type=float, metavar=("W", "S", "E", "N"),
        default=list(DEFAULT_BBOX),
        help="Clip bounding box (default: %(default)s)",
    )
    p.add_argument(
        "--simplify", type=float, default=DEFAULT_SIMPLIFY,
        help="Simplification tolerance in degrees (default: %(default)s)",
    )
    args = p.parse_args()
    prepare_clip(args.src, args.out, tuple(args.bbox), args.simplify)


if __name__ == "__main__":
    main()
