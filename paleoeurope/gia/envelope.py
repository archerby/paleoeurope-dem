"""
paleoeurope.gia.envelope
~~~~~~~~~~~~~~~~~~~~~~~~~

Ice-surface elevation for ice-covered pixels.

The ice *envelope* replaces NaN pixels (where H_ice > 0) with the surface
elevation of the ice sheet:

    z_ice_surface(φ, λ, t) = z_bedrock(φ, λ, t) + H_ice(φ, λ, t)

where z_bedrock is the isostatic bedrock elevation at epoch t (already
computed by :func:`~paleoeurope.gia.deformation.apply_gia_delta`).

This allows hillshading of the ice-sheet dome geometry for figures such as
fig06 and fig07 in the paper.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class IceEnvelope:
    """Compute the ice-surface elevation for glacier-covered tiles.

    Parameters
    ----------
    model_lats : np.ndarray  shape (M,)
        Latitude grid of the GIA model (degrees, ascending).
    model_lons : np.ndarray  shape (N,)
        Longitude grid of the GIA model (degrees).
    ice_height : np.ndarray  shape (M, N)
        Ice-sheet thickness H_ice(φ, λ, t) in metres.
    bedrock_delta : np.ndarray  shape (M, N)
        Isostatic bedrock correction Δz_GIA at the same epoch (metres).
        The bedrock elevation at epoch t is ``modern_dem + bedrock_delta``.

    Examples
    --------
    >>> import numpy as np
    >>> lats = np.array([59.0, 60.0, 61.0])
    >>> lons = np.array([10.0, 11.0])
    >>> ice_h = np.array([[0., 0.], [500., 1000.], [2000., 3000.]])
    >>> dz = np.zeros_like(ice_h)
    >>> env = IceEnvelope(lats, lons, ice_h, dz)
    >>> modern = np.zeros((3, 2))
    >>> surface = env.surface_elevation(modern, lats, lons)
    >>> surface[2, 1]  # 3000 m ice on 0 m bedrock
    3000.0
    """

    def __init__(
        self,
        model_lats: np.ndarray,
        model_lons: np.ndarray,
        ice_height: np.ndarray,
        bedrock_delta: np.ndarray,
    ) -> None:
        # Ensure ascending lats
        if model_lats[0] > model_lats[-1]:
            model_lats = model_lats[::-1]
            ice_height = ice_height[::-1, :]
            bedrock_delta = bedrock_delta[::-1, :]

        self._model_lats = model_lats
        self._model_lons = model_lons
        self._ice_height = ice_height.astype(np.float64)
        self._bedrock_delta = bedrock_delta.astype(np.float64)

        self._interp_ice = RegularGridInterpolator(
            (model_lats, model_lons), self._ice_height,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        self._interp_dz = RegularGridInterpolator(
            (model_lats, model_lons), self._bedrock_delta,
            method="linear", bounds_error=False, fill_value=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def surface_elevation(
        self,
        modern_dem: np.ndarray,
        tile_lats: np.ndarray,
        tile_lons: np.ndarray,
        ice_threshold_m: float = 1.0,
    ) -> np.ndarray:
        """Return the ice-surface elevation for glacier-covered pixels.

        Parameters
        ----------
        modern_dem : np.ndarray  shape (H, W)
            Modern bare-earth DEM (m), float32.
        tile_lats : np.ndarray  shape (H,)
            Latitude of each row (degrees).
        tile_lons : np.ndarray  shape (W,)
            Longitude of each column (degrees).
        ice_threshold_m : float, optional
            Minimum ice thickness to emit non-zero surface elevation. Default 1.

        Returns
        -------
        np.ndarray  shape (H, W)
            Ice-free pixels: same as modern_dem + Δz_GIA.
            Ice-covered pixels: bedrock + H_ice (ice surface).
            NaN where ice_height == 0 and modern_dem is NaN.
        """
        lon_grid, lat_grid = np.meshgrid(tile_lons, tile_lats)
        pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        ice_tile = self._interp_ice(pts).reshape(modern_dem.shape).astype(np.float32)
        dz_tile = self._interp_dz(pts).reshape(modern_dem.shape).astype(np.float32)

        bedrock = modern_dem + dz_tile
        ice_surface = bedrock + ice_tile

        result = bedrock.copy()
        ice_mask = ice_tile >= ice_threshold_m
        result[ice_mask] = ice_surface[ice_mask]

        return result
