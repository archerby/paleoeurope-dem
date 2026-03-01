"""
distributed/tasks.py

Optional Celery task layer for distributed full-Europe processing.

This module is only needed for large-scale parallel runs.
For single-tile or demo use, use scripts/run_fusion.py directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Celery app is only created if celery is installed
try:
    from celery import Celery

    celery_app = Celery("paleoeurope")
    celery_app.config_from_object("distributed.celery_config", silent=True)

    @celery_app.task(bind=True, max_retries=2, soft_time_limit=600)
    def fusion_tile_task(self, tile_id: str, config: dict) -> dict:
        """Celery task: run fusion pipeline for one tile.

        Parameters
        ----------
        tile_id : str
            e.g. ``"N51E000"``
        config : dict
            Merged config dict (from configs/default.yml or europe_full.yml).

        Returns
        -------
        dict
            ``{"tile_id": ..., "output_path": ..., "status": "ok"|"error"}``
        """
        from paleoeurope.fusion.pipeline import run_fusion_tile

        try:
            out = run_fusion_tile(
                tile_id=tile_id,
                fabdem_dir=config["paths"]["fabdem_dir"],
                gebco_path=config["paths"]["gebco_file"],
                output_path=Path(config["paths"]["output_dir"]) / f"{tile_id}_fusion.tif",
                egm2008_grid=config["fusion"].get("egm2008_grid"),
                apply_geoid_offset=config["fusion"].get("apply_geoid_offset", False),
                blend_distance_px=config["fusion"].get("blend_distance_px", 50),
            )
            return {"tile_id": tile_id, "output_path": str(out), "status": "ok"}
        except Exception as exc:
            logger.error("fusion_tile_task failed for %s: %s", tile_id, exc, exc_info=True)
            raise self.retry(exc=exc)

    @celery_app.task(bind=True, max_retries=2, soft_time_limit=600)
    def paleo_tile_task(self, tile_id: str, epoch_ka: float, config: dict) -> dict:
        """Celery task: apply GIA correction for one tile and epoch.

        Parameters
        ----------
        tile_id : str
        epoch_ka : float
            Target epoch in ka BP.
        config : dict

        Returns
        -------
        dict
        """
        from paleoeurope.gia.ice6g_loader import Ice6gLoader
        from paleoeurope.gia.deformation import apply_gia_delta
        from paleoeurope.utils.raster import read_geotiff, write_geotiff
        from paleoeurope.utils.grid import tile_id_to_bounds, make_pixel_coords
        import numpy as np

        try:
            fusion_path = Path(config["paths"]["output_dir"]) / f"{tile_id}_fusion.tif"
            win = read_geotiff(fusion_path)

            loader = Ice6gLoader(config["gia"]["ice6g_path"])
            bounds = tile_id_to_bounds(tile_id)
            lats, lons, dz, ice_h = loader.get_fields(epoch_ka=epoch_ka, bounds=bounds)

            tile_lats, tile_lons = make_pixel_coords(bounds, win.data.shape)
            paleo, _ = apply_gia_delta(win.data, lats, lons, dz, ice_h, tile_lats, tile_lons)

            epoch_str = f"{int(epoch_ka):02d}ka"
            out_dir = Path(config["paths"]["paleo_dem_dir"]) / epoch_str
            out_path = out_dir / f"{tile_id}_t{int(epoch_ka)}.tif"
            write_geotiff(out_path, paleo, win.transform, win.crs)

            return {"tile_id": tile_id, "epoch_ka": epoch_ka, "output_path": str(out_path), "status": "ok"}
        except Exception as exc:
            logger.error("paleo_tile_task failed for %s @%s ka: %s", tile_id, epoch_ka, exc, exc_info=True)
            raise self.retry(exc=exc)

except ImportError:
    logger.debug("celery not installed; distributed tasks unavailable")
