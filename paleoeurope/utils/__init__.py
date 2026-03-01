"""
paleoeurope.utils

Shared raster and grid utilities.
"""

from paleoeurope.utils.grid import (
    bounds_to_tile_ids,
    make_pixel_coords,
    tile_id_to_bounds,
)
from paleoeurope.utils.raster import (
    clip_to_bounds,
    make_dataarray,
    read_geotiff,
    reproject_array,
    write_geotiff,
)
from paleoeurope.utils.tile_index import (
    BBox,
    TileIndex,
    collect_epoch_tile_paths,
)

__all__ = [
    "write_geotiff",
    "read_geotiff",
    "reproject_array",
    "clip_to_bounds",
    "make_dataarray",
    "tile_id_to_bounds",
    "bounds_to_tile_ids",
    "make_pixel_coords",
    "TileIndex",
    "collect_epoch_tile_paths",
    "BBox",
]
