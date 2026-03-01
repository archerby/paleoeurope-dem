"""
paleoeurope.fusion.blender
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alpha-blend FABDEM (land) and GEBCO (ocean) across the coastline.

The blender computes a *signed Euclidean distance field* from the land mask
and maps it through a linear ramp to produce per-pixel alpha weights.  This
avoids hard edges at the coast that would appear as artefacts in hillshading.

At high latitudes, longitude degrees are compressed.  The distance transform
uses a latitude-adjusted aspect ratio so that the blend zone is circular in
physical space (metres) rather than in pixel space (degrees).

Algorithm
---------
1. Derive land mask from ``land.notnull()`` (or caller-supplied mask).
2. Compute ``dist_land`` = distance to nearest water pixel (EDT on land mask).
3. Compute ``dist_water`` = distance to nearest land pixel (EDT on inverted mask).
4. ``signed_dist = dist_land − dist_water``  (positive inside land).
5. ``alpha = clip((signed_dist + B) / (2B), 0, 1)`` where B = blend_distance_px.
6. ``result = land_filled · alpha + ocean · (1 − alpha)``.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import xarray as xr


class RasterBlender:
    """Blend a land DEM and an ocean DEM with a smooth coastal transition.

    Parameters
    ----------
    blend_distance_px : int
        Half-width of the blending zone in pixels.  Pixels farther than this
        distance from the coastline receive pure FABDEM (land) or pure GEBCO
        (ocean) values.  Default is 50 px (≈ 4.5 km at 3 arc-sec resolution).

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> land = xr.DataArray(np.full((10, 10), np.nan))
    >>> land[5:, :] = 100.0   # southern half is land
    >>> ocean = xr.DataArray(np.full((10, 10), -50.0))
    >>> blender = RasterBlender(blend_distance_px=3)
    >>> merged = blender.blend(land, ocean)
    """

    def __init__(self, blend_distance_px: int = 50) -> None:
        if blend_distance_px < 1:
            raise ValueError("blend_distance_px must be >= 1")
        self.blend_distance_px = blend_distance_px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        land: xr.DataArray,
        ocean: xr.DataArray,
        land_mask: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Produce a blended FABDEM–GEBCO composite.

        Parameters
        ----------
        land : xr.DataArray
            High-resolution bare-earth DEM (FABDEM).  NaN where ocean.
        ocean : xr.DataArray
            Bathymetry / ocean DEM (GEBCO), already resampled to match *land*.
        land_mask : xr.DataArray, optional
            Boolean mask where ``True`` means land.  Derived from
            ``land.notnull()`` if ``None``.

        Returns
        -------
        xr.DataArray
            Merged elevation array, same shape and coordinates as *land*.
        """
        if land_mask is None:
            land_mask = land.notnull()

        mask_arr: np.ndarray = np.asarray(land_mask, dtype=bool)

        # --- Short-circuit pure tiles ---
        if mask_arr.all():
            return land.copy()
        if not mask_arr.any():
            return ocean.copy()

        # --- Latitude-adjusted aspect ratio ---
        sampling = self._sampling(land)

        # --- Distance fields ---
        water_mask = ~mask_arr
        dist_land = scipy.ndimage.distance_transform_edt(mask_arr, sampling=sampling)
        dist_water = scipy.ndimage.distance_transform_edt(water_mask, sampling=sampling)

        # --- Alpha ramp ---
        B = float(self.blend_distance_px)
        signed = dist_land - dist_water
        alpha = np.clip((signed + B) / (2.0 * B), 0.0, 1.0).astype(np.float32)

        # --- Blend ---
        land_filled = np.nan_to_num(np.asarray(land, dtype=np.float32), nan=0.0)
        ocean_arr = np.asarray(ocean, dtype=np.float32)
        merged = land_filled * alpha + ocean_arr * (1.0 - alpha)

        return xr.DataArray(merged, coords=land.coords, dims=land.dims, attrs=land.attrs)

    def compute_alpha(
        self,
        land_mask: np.ndarray,
        mean_lat: float = 0.0,
    ) -> np.ndarray:
        """Return the alpha weight array for a given land mask.

        Useful for diagnostics and figure generation (fig03_alpha_blending).

        Parameters
        ----------
        land_mask : np.ndarray
            Boolean array, ``True`` = land.
        mean_lat : float
            Mean latitude of the tile in degrees (used for aspect correction).

        Returns
        -------
        np.ndarray
            Float32 array in [0, 1].
        """
        mask = mask_arr = np.asarray(land_mask, dtype=bool)
        water_mask = ~mask
        aspect = float(np.cos(np.radians(mean_lat)))
        sampling = (1.0, aspect if aspect > 0.01 else 0.01)

        dist_land = scipy.ndimage.distance_transform_edt(mask_arr, sampling=sampling)
        dist_water = scipy.ndimage.distance_transform_edt(water_mask, sampling=sampling)

        B = float(self.blend_distance_px)
        signed = dist_land - dist_water
        return np.clip((signed + B) / (2.0 * B), 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sampling(da: xr.DataArray) -> tuple[float, float]:
        """Return (y_spacing, x_spacing) for the distance transform.

        At mid-latitudes, 1° of longitude is shorter than 1° of latitude.
        We correct for this so the blend zone is isotropic in metres.
        """
        if "y" in da.coords:
            mean_lat = float(da.coords["y"].mean())
            aspect = float(np.cos(np.radians(mean_lat)))
            aspect = max(aspect, 0.01)  # guard against poles
            return (1.0, aspect)
        return (1.0, 1.0)
