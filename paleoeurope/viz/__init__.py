"""
paleoeurope.viz
~~~~~~~~~~~~~~~

Visualization utilities for paleo-DEM reconstruction products.

Modules
-------
paleocostline_render
    Composite per-epoch render: hillshade terrain (land),  depth-gradient ocean,
    and optional solid-white ice-sheet layer.  Coastline mask is sourced from
    vector ``Paleocoastlines.shp`` matched to the Spratt & Lisiecki (2016)
    sea-level curve, with automatic fallback to a DEM zero-contour.
"""

from __future__ import annotations

from paleoeurope.viz.hillshade import (
    DEFAULT_VERT_EXAG,
    bw_hillshade,
)
from paleoeurope.viz.paleocostline_render import (
    RenderConfig,
    build_epoch_mosaic,
    filter_paleocostlines_for_epoch,
    get_sea_level_for_epoch,
    load_paleocostlines,
    rasterize_land_mask,
    render_paleocostline_epoch,
)

__all__ = [
    "RenderConfig",
    "build_epoch_mosaic",
    "filter_paleocostlines_for_epoch",
    "get_sea_level_for_epoch",
    "load_paleocostlines",
    "rasterize_land_mask",
    "render_paleocostline_epoch",
    "bw_hillshade",
    "DEFAULT_VERT_EXAG",
]
