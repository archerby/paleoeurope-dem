"""
paleoeurope.viz.hillshade
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure greyscale hillshade rendering for paleo-DEM tiles.

This module consolidates the ``bw_hillshade`` helper that was previously
duplicated verbatim across ``demo_pipeline.ipynb`` (as ``_hs()``) and
``04_paleocostline_demo.ipynb``.  Any divergence between the two
implementations is now impossible.

Usage
-----
>>> from paleoeurope.viz.hillshade import bw_hillshade
>>> hs = bw_hillshade(dem, vert_exag=8.0)           # → float32 array in [0, 1]
>>> ax.imshow(hs, cmap='gray', vmin=0.0, vmax=1.0)
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

__all__ = ["bw_hillshade", "DEFAULT_VERT_EXAG", "DEFAULT_AZDEG", "DEFAULT_ALTDEG"]

# Defaults used throughout the publication pipeline
DEFAULT_VERT_EXAG: float = 8.0
DEFAULT_AZDEG: float = 315.0   # north-west illumination
DEFAULT_ALTDEG: float = 35.0   # ~35° above horizon


def bw_hillshade(
    dem: np.ndarray,
    vert_exag: float = DEFAULT_VERT_EXAG,
    azdeg: float = DEFAULT_AZDEG,
    altdeg: float = DEFAULT_ALTDEG,
) -> np.ndarray:
    """Compute a pure greyscale hillshade from an elevation array.

    NaN pixels in *dem* are preserved as NaN in the output so callers can
    mask or replace them independently (e.g. as white ocean background).

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Elevation array in metres.  May contain NaN (water / nodata).
        Any floating-point dtype is accepted; internally promoted to float64
        for the hillshade computation.
    vert_exag : float, optional
        Vertical exaggeration factor.  Default ``8.0``.
    azdeg : float, optional
        Sun azimuth in degrees (0 = North, clockwise).  Default ``315.0``
        (north-west illumination, conventional for terrain maps).
    altdeg : float, optional
        Sun elevation above horizon in degrees.  Default ``35.0``.

    Returns
    -------
    np.ndarray, dtype float32, shape (H, W)
        Hillshade intensity in the range ``[0.0, 1.0]``.  NaN where the
        input was NaN.

    Examples
    --------
    >>> import numpy as np
    >>> dem = np.random.rand(100, 100).astype(np.float32) * 1000
    >>> hs  = bw_hillshade(dem)
    >>> hs.shape
    (100, 100)
    >>> float(hs.min()) >= 0.0 and float(hs.max()) <= 1.0
    True
    >>> dem_with_nan = dem.copy(); dem_with_nan[0, 0] = np.nan
    >>> np.isnan(bw_hillshade(dem_with_nan)[0, 0])
    True
    """
    ls = mcolors.LightSource(azdeg=azdeg, altdeg=altdeg)

    nan_mask = np.isnan(dem)
    # Replace NaN with 0 for the underlying hillshade compute (does not affect
    # gradient at valid pixels as long as patches of NaN are small relative to
    # the smoothing scale of the Lambertian reflectance).
    filled = np.where(nan_mask, 0.0, dem).astype(np.float64)

    hs = ls.hillshade(filled, vert_exag=vert_exag, dx=1.0, dy=1.0)

    # Restore NaN mask and return as float32
    result = np.where(nan_mask, np.nan, hs).astype(np.float32)
    return result
