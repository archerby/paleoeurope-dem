"""
paleoeurope.gia.deformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply the GIA delta method to produce a paleo-DEM for one epoch.

The delta method (e.g. Lambeck et al. 2014) calculates the elevation change
relative to the present as the difference in relative sea-level (RSL) between
the target epoch and 0 ka from an ice-sheet model:

    z_paleo(φ, λ, t) = z_modern(φ, λ) + Δz_GIA(φ, λ, t)

where Δz_GIA is bilinearly interpolated from the model grid to the tile grid.
Ice-covered pixels are excluded and handled separately in
:mod:`paleoeurope.gia.envelope`.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def apply_gia_delta(
    modern_dem: np.ndarray,
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    delta_z: np.ndarray,
    ice_height: np.ndarray,
    tile_lats: np.ndarray,
    tile_lons: np.ndarray,
    sigma_blur: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the GIA delta correction to a modern-DEM tile.

    Parameters
    ----------
    modern_dem : np.ndarray  shape (H, W)
        Modern fused elevation (m), float32.
    model_lats : np.ndarray  shape (M,)
        Latitude grid of the GIA model (ascending, degrees).
    model_lons : np.ndarray  shape (N,)
        Longitude grid of the GIA model (degrees).
    delta_z : np.ndarray  shape (M, N)
        GIA-related elevation change Δz(φ, λ, t) in metres
        (positive = paleo surface was *higher* than today).
    ice_height : np.ndarray  shape (M, N)
        Ice-sheet thickness H_ice(φ, λ, t) in metres at the target epoch.
    tile_lats : np.ndarray  shape (H,)
        Latitude coordinate of each pixel row (descending, degrees).
    tile_lons : np.ndarray  shape (W,)
        Longitude coordinate of each pixel column (degrees).
    sigma_blur : float, optional
        Standard deviation for Gaussian smoothing of the GIA delta field
        to prevent 10-minute grid artifacts and preserve mantle mass. Default 2.0.

    Returns
    -------
    paleo_dem : np.ndarray  shape (H, W)
        Paleo elevation in metres.
    ice_tile : np.ndarray  shape (H, W)
        Interpolated ice thickness (m) on the tile grid. A boolean ice mask can
        be derived as e.g. ``ice_tile >= 1.0``.

    Notes
    -----
    Model grids from ICE-6G_C are typically 0.5° or 1° resolution.  Bilinear
    interpolation to the 3 arc-sec (1/1200°) tile grid is accurate for GIA
    because GIA signals are long-wavelength (hundreds of km).

    Examples
    --------
    >>> import numpy as np
    >>> dem = np.zeros((5, 5), dtype=np.float32)
    >>> lats = np.array([60.0, 60.5])
    >>> lons = np.array([10.0, 10.5])
    >>> dz = np.full((2, 2), -50.0, dtype=np.float32)  # land was 50 m lower
    >>> ice_h = np.zeros((2, 2), dtype=np.float32)     # no ice
    >>> tile_lats = np.linspace(60.45, 60.05, 5)
    >>> tile_lons = np.linspace(10.05, 10.45, 5)
    >>> paleo, ice_tile = apply_gia_delta(dem, lats, lons, dz, ice_h, tile_lats, tile_lons)
    >>> np.allclose(paleo, -50.0)
    True
    >>> float(ice_tile.max())
    0.0
    """
    # Ensure lats are ascending for RegularGridInterpolator
    if model_lats[0] > model_lats[-1]:
        model_lats = model_lats[::-1]
        delta_z = delta_z[::-1, :]
        ice_height = ice_height[::-1, :]

    # Smooth the GIA delta field to preserve mass and prevent grid artifacts
    delta_z_smooth = gaussian_filter(delta_z.astype(np.float64), sigma=sigma_blur)

    # Interpolate Δz onto tile grid
    interp_dz = RegularGridInterpolator(
        (model_lats, model_lons),
        delta_z_smooth,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    interp_ice = RegularGridInterpolator(
        (model_lats, model_lons),
        ice_height.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Build meshgrid of tile coordinates
    # tile_lats may be descending — that's fine, we just build the grid
    lon_grid, lat_grid = np.meshgrid(tile_lons, tile_lats)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    dz_tile = interp_dz(pts).reshape(modern_dem.shape).astype(np.float32)
    ice_tile = interp_ice(pts).reshape(modern_dem.shape).astype(np.float32)

    # Apply delta correction
    paleo_dem = modern_dem + dz_tile

    ice_mask = ice_tile >= 1.0

    logger.debug(
        "GIA delta: Δz range=[%.1f, %.1f] m, ice_coverage=%.2f%%",
        float(dz_tile.min()),
        float(dz_tile.max()),
        100.0 * float(ice_mask.mean()),
    )

    return paleo_dem, ice_tile
