#!/usr/bin/env python3
"""
scripts/run_full_pipeline.py

CLI: end-to-end pipeline — fusion for all tiles, then GIA for all epochs.

Equivalent to running run_fusion.py followed by run_paleo.py for each epoch.

Examples
--------
# Full Europe (requires real data):
python scripts/run_full_pipeline.py --config configs/europe_full.yml

# Quick demo with downloaded/real test data (tests/data/ built by generate_synthetic_data.py):
python scripts/run_full_pipeline.py --config configs/default.yml \\
    --tile-list tests/data/tile_list.txt --epochs 0 21
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yml", type=Path)
    parser.add_argument("--tile-list", type=Path, metavar="FILE")
    parser.add_argument("--epochs", nargs="+", type=float, metavar="KA",
                        help="Epochs to process (default: from config)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true", help="Skip fusion step")
    args = parser.parse_args(argv)

    import yaml
    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["gia"]["epochs"]
    tile_list_arg = ["--tile-list", str(args.tile_list)] if args.tile_list else []
    overwrite_arg = ["--overwrite"] if args.overwrite else []
    config_arg = ["--config", str(args.config)]

    # --- Step 1: Fusion ---
    if not args.skip_fusion:
        logger.info("=== Step 1/2: Fusion ===")
        rc = subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_fusion.py")]
            + config_arg + tile_list_arg + overwrite_arg
        )
        if rc != 0:
            logger.error("Fusion step failed (exit code %d)", rc)
            return rc

    # --- Step 2: GIA for each epoch ---
    logger.info("=== Step 2/2: GIA correction for epochs %s ka ===", epochs)
    for epoch in epochs:
        logger.info("  Epoch %.0f ka", epoch)
        rc = subprocess.call(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_paleo.py"),
             "--epoch", str(epoch)]
            + config_arg + tile_list_arg + overwrite_arg
        )
        if rc != 0:
            logger.error("GIA step failed for epoch %.0f ka (exit code %d)", epoch, rc)
            return rc

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
