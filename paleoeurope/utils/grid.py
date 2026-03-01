"""
paleoeurope.utils.grid
~~~~~~~~~~~~~~~~~~~~~~~~

Tile naming, bounding-box arithmetic, and pixel coordinate generation.

All functions are pure (no file I/O, no side effects) and accept / return
plain Python / numpy types.
"""

from __future__ import annotations

import re

import numpy as np

# Standard SRTM/FABDEM tile pattern: N51E000, S04W073, etc.
_TILE_RE = re.compile(r"^([NS])(\d{2})([EW])(\d{3})$", re.IGNORECASE)


# ------------------------------------------------------------------
# Tile ID ↔ bounding box
# ------------------------------------------------------------------


def tile_id_to_bounds(tile_id: str) -> tuple[float, float, float, float]:
    """Parse a standard SRTM tile ID to a bounding box.

    Parameters
    ----------
    tile_id : str
        Tile identifier such as ``"N51E000"`` (upper-case OK).

    Returns
    -------
    tuple
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.

    Raises
    ------
    ValueError
        If *tile_id* does not match the expected pattern.

    Examples
    --------
    >>> tile_id_to_bounds("N51E000")
    (0.0, 51.0, 1.0, 52.0)
    >>> tile_id_to_bounds("S04W073")
    (-73.0, -4.0, -72.0, -3.0)
    """
    m = _TILE_RE.match(tile_id.strip())
    if not m:
        raise ValueError(
            f"Cannot parse tile_id '{tile_id}'.  "
            "Expected format: [NS]dd[EW]ddd  e.g. N51E000."
        )
    lat_dir, lat_v, lon_dir, lon_v = m.groups()
    lat = int(lat_v) * (1 if lat_dir.upper() == "N" else -1)
    lon = int(lon_v) * (1 if lon_dir.upper() == "E" else -1)
    return (float(lon), float(lat), float(lon + 1), float(lat + 1))


def bounds_to_tile_ids(
    bounds: tuple[float, float, float, float],
    tile_size: float = 1.0,
) -> list[str]:
    """Return a list of tile IDs that cover *bounds*.

    Parameters
    ----------
    bounds : tuple
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.
    tile_size : float, optional
        Tile side length in degrees.  Default ``1.0``.

    Returns
    -------
    list of str
        Tile IDs in row-major order (north → south, west → east).

    Examples
    --------
    >>> tiles = bounds_to_tile_ids((0.0, 51.0, 2.0, 53.0))
    >>> sorted(tiles)
    ['N51E000', 'N51E001', 'N52E000', 'N52E001']
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    tile_ids: list[str] = []

    lat = np.floor(lat_min / tile_size) * tile_size
    while lat < lat_max:
        lon = np.floor(lon_min / tile_size) * tile_size
        while lon < lon_max:
            tile_ids.append(_coords_to_tile_id(float(lat), float(lon)))
            lon += tile_size
        lat += tile_size

    return tile_ids


def _coords_to_tile_id(lat: float, lon: float) -> str:
    """Format SW-corner coordinates as a tile ID string."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{lat_dir}{abs(int(lat)):02d}{lon_dir}{abs(int(lon)):03d}"


# ------------------------------------------------------------------
# Pixel coordinate generation
# ------------------------------------------------------------------


def make_pixel_coords(
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate latitude and longitude arrays for a raster grid.

    Parameters
    ----------
    bounds : tuple
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.
    shape : tuple (height, width)
        Raster dimensions in pixels.
    center : bool, optional
        If ``True`` (default), coordinates point to the pixel centre.
        If ``False``, they point to the pixel's top-left corner.

    Returns
    -------
    lats : np.ndarray  shape (height,)
        Latitudes in descending order (north → south, like a raster).
    lons : np.ndarray  shape (width,)
        Longitudes in ascending order (west → east).

    Examples
    --------
    >>> lats, lons = make_pixel_coords((0, 51, 1, 52), (3600, 3600))
    >>> lats[0] > lats[-1]   # descending
    True
    >>> lons[0] < lons[-1]   # ascending
    True
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    height, width = shape

    lon_step = (lon_max - lon_min) / width
    lat_step = (lat_max - lat_min) / height

    if center:
        lons = lon_min + lon_step * (np.arange(width) + 0.5)
        lats = lat_max - lat_step * (np.arange(height) + 0.5)
    else:
        lons = lon_min + lon_step * np.arange(width)
        lats = lat_max - lat_step * np.arange(height)

    return lats, lons
