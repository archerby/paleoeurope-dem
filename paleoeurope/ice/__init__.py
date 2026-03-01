"""
paleoeurope.ice
~~~~~~~~~~~~~~~~

Provides methods for ice sheet processing, specifically the Envelope Method
which ensures a smooth, physically accurate transition from the steep ice margin
(where basal topography is transmitted to the surface) to the thick interior dome.
"""

from .envelope import apply_envelope_method

__all__ = ["apply_envelope_method"]
