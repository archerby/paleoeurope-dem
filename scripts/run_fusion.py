#!/usr/bin/env python3
"""
scripts/run_fusion.py

CLI: run the FABDEM + GEBCO fusion for one tile or a tile-list file.

Examples
--------
# Single tile
python scripts/run_fusion.py --tile N51E000 --config configs/default.yml

# Multiple tiles from a file
python scripts/run_fusion.py --tile-list configs/europe_tiles.txt --config configs/europe_full.yml

# Quick test with fixture data (real 1°×1° subset committed to tests/fixtures/)
python scripts/run_fusion.py --tile-list tests/data/tile_list.txt --config configs/default.yml
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

from paleoeurope.fusion.pipeline import run_fusion_tile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tile", metavar="TILE_ID", help="Single tile ID, e.g. N51E000")
    group.add_argument("--tile-list", metavar="FILE", type=Path, help="Text file with one tile ID per line")
    parser.add_argument("--config", default="configs/default.yml", type=Path, help="YAML config file")
    parser.add_argument("--output-dir", type=Path, help="Override output directory from config")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing output tiles")
    args = parser.parse_args(argv)

    # Load config
    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tile list
    if args.tile:
        tiles = [args.tile]
    else:
        tiles = [t.strip() for t in args.tile_list.read_text().splitlines() if t.strip()]

    logger.info("Processing %d tiles → %s", len(tiles), output_dir)

    success, skipped, errors = 0, 0, 0
    t0 = time.perf_counter()

    for tile_id in tiles:
        out_path = output_dir / f"{tile_id}_fusion.tif"
        try:
            result = run_fusion_tile(
                tile_id=tile_id,
                fabdem_dir=cfg["paths"]["fabdem_dir"],
                gebco_path=cfg["paths"]["gebco_file"],
                output_path=out_path,
                egm2008_grid=cfg["fusion"].get("egm2008_grid"),
                apply_geoid_offset=cfg["fusion"].get("apply_geoid_offset", False),
                blend_distance_px=cfg["fusion"].get("blend_distance_px", 50),
                overwrite=args.overwrite,
            )
            if result.exists():
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("FAILED %s: %s", tile_id, exc)
            errors += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1f s — success=%d  skipped=%d  errors=%d",
        elapsed, success, skipped, errors,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
