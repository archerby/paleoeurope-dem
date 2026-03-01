"""
paleoeurope

Reproducible pipeline for generating multi-epoch paleo-DEM reconstructions
of Europe from the Last Glacial Maximum (~21 ka) to the present.

Sub-packages
------------
fusion
    FABDEM + GEBCO data loading, EGM2008 datum correction, alpha blending.
gia
    ICE-6G_C / ICE-7G_NA loading, delta-method GIA deformation, ice envelope.
utils
    Shared raster and grid utilities.
viz
    Visualization: composite hillshade + coastline renders per epoch.
    Entry point: ``paleoeurope.viz.render_paleocostline_epoch``.
"""

__version__ = "0.1.0-alpha"
__all__ = ["fusion", "gia", "utils", "viz"]
