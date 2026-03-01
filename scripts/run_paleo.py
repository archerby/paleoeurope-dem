#!/usr/bin/env python3
"""
scripts/run_paleo.py

CLI: apply GIA correction to fused tiles and produce a paleo-DEM for one epoch.

Requires fused tiles (output of run_fusion.py) in the configured output_dir.

Examples
--------
python scripts/run_paleo.py --epoch 21 --config configs/default.yml
python scripts/run_paleo.py --epoch 12 --tile N51E000 --config configs/default.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epoch", type=float, required=True, metavar="KA", help="Target epoch in ka BP (e.g. 21)")
    parser.add_argument("--config", default="configs/default.yml", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tile", metavar="TILE_ID", help="Process a single tile")
    group.add_argument("--tile-list", type=Path, metavar="FILE", help="Text file with tile IDs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from paleoeurope.gia.deformation import apply_gia_delta
    from paleoeurope.gia.ice6g_loader import Ice6gLoader
    from paleoeurope.utils.grid import make_pixel_coords, tile_id_to_bounds
    from paleoeurope.utils.raster import read_geotiff, write_geotiff

    fusion_dir = Path(cfg["paths"]["output_dir"])
    paleo_dir = Path(cfg["paths"]["paleo_dem_dir"]) / f"{int(args.epoch):02d}ka"
    paleo_dir.mkdir(parents=True, exist_ok=True)

    # Build tile list
    if args.tile:
        tiles = [args.tile]
    elif args.tile_list:
        tiles = [t.strip() for t in args.tile_list.read_text().splitlines() if t.strip()]
    else:
        tiles = [p.stem.replace("_fusion", "") for p in sorted(fusion_dir.glob("*_fusion.tif"))]

    logger.info("Epoch %.0f ka — %d tiles → %s", args.epoch, len(tiles), paleo_dir)

    ice_loader = Ice6gLoader(cfg["gia"]["ice6g_path"])
    success, errors = 0, 0
    t0 = time.perf_counter()

    for tile_id in tiles:
        out_path = paleo_dir / f"{tile_id}_t{int(args.epoch)}.tif"
        if out_path.exists() and not args.overwrite:
            logger.debug("Skipping %s (exists)", tile_id)
            success += 1
            continue

        fusion_path = fusion_dir / f"{tile_id}_fusion.tif"
        if not fusion_path.exists():
            logger.warning("Fusion tile missing: %s — skipping", fusion_path)
            errors += 1
            continue

        try:
            win = read_geotiff(fusion_path)
            bounds = tile_id_to_bounds(tile_id)
            lats, lons, dz, ice_h = ice_loader.get_fields(epoch_ka=args.epoch, bounds=bounds)
            tile_lats, tile_lons = make_pixel_coords(bounds, win.data.shape)
            paleo, _ = apply_gia_delta(win.data, lats, lons, dz, ice_h, tile_lats, tile_lons)
            write_geotiff(out_path, paleo, win.transform, win.crs)
            success += 1
        except Exception as exc:
            logger.error("FAILED %s: %s", tile_id, exc)
            errors += 1

    elapsed = time.perf_counter() - t0
    logger.info("Done %.1f s — success=%d errors=%d", elapsed, success, errors)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
