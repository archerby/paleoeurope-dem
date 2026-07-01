"""
paleoeurope.gia.ice6g_topo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert ICE-6G_C ``Topo`` fields into a present-day-datum solid-earth vertical
displacement, for use as the GIA delta in :func:`paleoeurope.gia.deformation.apply_gia_delta`.

Why not ``Orog``?  ICE-6G ``Orog`` (orography) is clamped to 0 over water, so a
``Orog(t) - Orog(0)`` delta vanishes exactly under the sea floor — precisely where
a paleo-shoreline is drawn.  ``Topo`` carries bathymetry, but it is expressed
**relative to the contemporaneous (epoch) sea surface**: over far-field deep ocean
(no ice load, no solid motion) ``Topo(t) - Topo(0)`` equals the sea-surface change
(~ +112 m at 18 ka). We remove that component to recover the pure present-datum
solid displacement.

    sea_surface_offset(t) = median over deep ocean of [Topo(t) - Topo(0)]
    Δz_solid(t)           = [Topo(t) - Topo(0)] - sea_surface_offset(t)

Δz_solid is ~0 in the far field (verifiable) and the true isostatic signal near
former ice loads, in the present-day datum.  It is added to a present-datum
modern DEM to build the paleo-DEM.
"""

from __future__ import annotations

import numpy as np


def implied_sea_surface_offset(
    topo_t: np.ndarray,
    topo_0: np.ndarray,
    deep_threshold: float = -1000.0,
) -> float:
    """Median of ``topo_t - topo_0`` over deep-ocean cells (``topo_0 < deep_threshold``).

    This is the epoch sea-surface change baked into ICE-6G ``Topo`` (the far field
    has ~no solid motion, so the whole ΔTopo there is the moving sea surface).
    Global deep ocean is overwhelmingly far-field, so the median is robust without
    an explicit ice mask.

    Parameters
    ----------
    topo_t, topo_0 : np.ndarray
        ICE-6G ``Topo`` at the target epoch and at 0 ka (same shape, metres).
    deep_threshold : float
        ``topo_0`` below this (metres) counts as deep ocean.  Default −1000.

    Returns
    -------
    float
        Sea-surface offset in metres (positive at past lowstands).
    """
    delta = topo_t - topo_0
    deep = topo_0 < deep_threshold
    if not np.any(deep):
        raise ValueError("no deep-ocean cells found for the sea-surface offset")
    # nanmedian (not median): ICE-6G_C ``Topo`` declares _FillValue = 1e+20, so a
    # stray fill/NaN deep-ocean cell is ignored instead of poisoning the offset.
    return float(np.nanmedian(delta[deep]))


def solid_displacement(
    topo_t: np.ndarray,
    topo_0: np.ndarray,
    sea_surface_offset: float,
) -> np.ndarray:
    """Present-datum solid-earth vertical displacement ``Δz_solid``.

    ``Δz_solid = (topo_t - topo_0) - sea_surface_offset``.  Positive = the solid
    surface stood higher than today.  Far-field → ~0; near former ice loads → the
    isostatic signal.  Add to a present-datum modern DEM to obtain the paleo-DEM.
    """
    return (topo_t - topo_0) - sea_surface_offset
