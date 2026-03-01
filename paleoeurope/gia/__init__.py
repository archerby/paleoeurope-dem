"""
paleoeurope.gia

Glacial Isostatic Adjustment (GIA) deformation of the modern DEM.

The delta method computes the elevation change Δz_GIA(φ, λ, t) between the
modern (0 ka) and a target epoch t by interpolating the ICE-6G_C or ICE-7G_NA
model onto the tile grid.  The corrected paleo-DEM is:

    z_paleo(φ, λ, t) = z_modern(φ, λ) + Δz_GIA(φ, λ, t)

Ice-covered pixels (H_ice > 0 at epoch t) are excluded from the paleo-DEM
surface and replaced with an ice-surface elevation from the ice-thickness
envelope.

Public API
----------
Ice6gLoader, Ice7gLoader
apply_gia_delta
IceEnvelope
"""

from paleoeurope.gia.correction_matrix import (
    CM_RESOLUTION_DEG,
    build_correction_matrix,
    read_correction_matrix,
    write_correction_matrix,
)
from paleoeurope.gia.correction_pipeline import (
    EpochResult,
    run_single_tile_epochs,
)
from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.envelope import IceEnvelope
from paleoeurope.gia.ice6g_loader import Ice6gLoader
from paleoeurope.gia.ice7g_loader import Ice7gLoader

__all__ = [
    "Ice6gLoader",
    "Ice7gLoader",
    "apply_gia_delta",
    "IceEnvelope",
    "build_correction_matrix",
    "read_correction_matrix",
    "write_correction_matrix",
    "CM_RESOLUTION_DEG",
    "run_single_tile_epochs",
    "EpochResult",
]
