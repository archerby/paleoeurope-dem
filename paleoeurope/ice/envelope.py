"""
paleoeurope.ice.envelope
~~~~~~~~~~~~~~~~~~~~~~~~

Implements the Envelope Method for ice sheet surface reconstruction.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

def apply_envelope_method(
    paleo_bedrock: np.ndarray,
    ice_thickness: np.ndarray,
    modern_bedrock: np.ndarray,
    t_tr: float = 200.0,
) -> np.ndarray:
    """Apply the Envelope Method to generate the final ice surface.

    The method transitions from the rough basal topography (paleo_bedrock) at the
    ice margin to a smooth, dome-like ice surface in the interior. The transition
    is controlled by the thickness of the ice and the parameter t_tr.

    Calculated as:
        Z_final = (1 - w) * S_rough + w * S_target
    where:
        w = clip(ice_thickness / t_tr, 0, 1)
        S_rough = paleo_bedrock + ice_thickness
        S_target = smoothed_paleo_bedrock + ice_thickness

    Parameters
    ----------
    paleo_bedrock : np.ndarray
        The high-resolution bedrock elevation at the target epoch (e.g. after GIA).
    ice_thickness : np.ndarray
        The ice-sheet thickness (H_ice).
    modern_bedrock : np.ndarray
        The modern bedrock, used as a proxy to extract the smooth target baseline
        (S_target = paleo_bedrock - modern + mean(modern) + ice_thickness).
    t_tr : float
        Transition depth in metres. Default is 200.0 based on plastic ice profile.

    Returns
    -------
    np.ndarray
        Final elevation model with ice surfaces smoothly blended.
    """

    # 1. Calculate weight
    w = np.clip(ice_thickness / t_tr, 0.0, 1.0)

    # 2. Define surface end-members
    s_rough = paleo_bedrock + ice_thickness

    # Target proxy: we subtract the modern high-frequency relief from paleo_bedrock
    # to approximate the smooth underlying GIA orog model, preserving mean elevation.
    mean_modern = np.nanmean(modern_bedrock)
    smoothed_bedrock_proxy = paleo_bedrock - modern_bedrock + mean_modern
    s_target = smoothed_bedrock_proxy + ice_thickness

    # 3. Blend
    z_final = (1.0 - w) * s_rough + w * s_target

    # 4. Mask
    ice_mask = ice_thickness >= 1.0

    final_dem = np.where(ice_mask, z_final, paleo_bedrock)

    logger.debug(
        "Envelope Method applied: t_tr=%.1f m, ice_coverage=%.2f%%",
        t_tr,
        100.0 * float(ice_mask.mean()) if ice_mask.size > 0 else 0.0,
    )

    return final_dem
