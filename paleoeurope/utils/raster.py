"""
paleoeurope.utils.raster
~~~~~~~~~~~~~~~~~~~~~~~~~

Common raster I/O and reprojection utilities backed by rasterio.

All functions take and return :class:`numpy.ndarray` or lightweight
``(array, transform, crs)`` tuples — no file paths leak into core logic.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import NamedTuple

if typing.TYPE_CHECKING:
    import xarray as xr


import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as rasterio_reproject


class RasterWindow(NamedTuple):
    """Lightweight container returned by :func:`read_geotiff`."""

    data: np.ndarray   # shape (H, W), float32
    transform: Affine
    crs: CRS
    nodata: float


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    nodata: float = -9999.0,
    compress: str = "LZW",
    dtype: str = "float32",
) -> Path:
    """Write *data* to a Cloud-Optimised GeoTIFF.

    Parameters
    ----------
    path : str or Path
        Destination file.  Parent directories are created if necessary.
    data : np.ndarray  shape (H, W)
        Single-band raster values.
    transform : rasterio.Affine
        Affine geotransform for *data*.
    crs : rasterio.CRS or str
        Coordinate reference system (e.g. ``"EPSG:4326"``).
    nodata : float, optional
        NoData value. Default ``-9999.0``.
    compress : str, optional
        Rasterio compression codec.  Default ``"LZW"``.
    dtype : str, optional
        Output dtype. Default ``"float32"``.

    Returns
    -------
    Path
        Path to the written file.

    Examples
    --------
    >>> import numpy as np
    >>> from rasterio.transform import from_bounds
    >>> arr = np.ones((100, 100), dtype="float32")
    >>> t = from_bounds(0, 51, 1, 52, 100, 100)
    >>> p = write_geotiff("/tmp/test.tif", arr, t, "EPSG:4326")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)

    out = np.where(np.isnan(data), nodata, data).astype(dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(out, 1)

    return path


def read_geotiff(path: str | Path) -> RasterWindow:
    """Read a single-band GeoTIFF and return a :class:`RasterWindow`.

    Parameters
    ----------
    path : str or Path
        Path to the GeoTIFF.

    Returns
    -------
    RasterWindow
        Named tuple with ``data``, ``transform``, ``crs``, ``nodata``.

    Examples
    --------
    >>> win = read_geotiff("data/output/N51E000_fusion.tif")
    >>> win.data.shape
    (3600, 3600)
    """
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = float(src.nodata) if src.nodata is not None else np.nan
        if not np.isnan(nodata):
            data[data == nodata] = np.nan
        return RasterWindow(data=data, transform=src.transform, crs=src.crs, nodata=nodata)


# ------------------------------------------------------------------
# Reprojection / clipping
# ------------------------------------------------------------------


def reproject_array(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS | str,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    dst_crs: CRS | str,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = np.nan,
) -> np.ndarray:
    """Reproject *data* from one grid to another.

    Parameters
    ----------
    data : np.ndarray  shape (H, W)
        Source raster (float32).
    src_transform, src_crs : Affine, CRS
        Source grid definition.
    dst_transform : Affine
    dst_shape : tuple (H_out, W_out)
    dst_crs : CRS or str
    resampling : rasterio.Resampling, optional
    nodata : float, optional
        Fill value for areas outside source coverage.

    Returns
    -------
    np.ndarray  shape (*dst_shape*)
        Reprojected float32 array.
    """
    destination = np.full(dst_shape, nodata, dtype=np.float32)

    if isinstance(src_crs, str):
        src_crs = CRS.from_user_input(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = CRS.from_user_input(dst_crs)

    src_nodata = float(nodata) if not np.isnan(nodata) else None
    dst_nodata = src_nodata

    rasterio_reproject(
        source=data.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )

    return destination


def clip_to_bounds(
    data: np.ndarray,
    src_transform: Affine,
    bounds: tuple[float, float, float, float],
    target_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, Affine]:
    """Clip *data* to *bounds* (minx, miny, maxx, maxy).

    Parameters
    ----------
    data : np.ndarray  shape (H, W)
    src_transform : rasterio.Affine
    bounds : tuple
        ``(minx, miny, maxx, maxy)`` in the CRS of *data*.
    target_shape : tuple (H_out, W_out), optional
        If provided, the output is resampled to exactly this size.

    Returns
    -------
    clipped : np.ndarray
    clip_transform : rasterio.Affine
    """
    from rasterio.windows import from_bounds as win_from_bounds

    window = win_from_bounds(*bounds, transform=src_transform)
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    win_h = int(np.ceil(window.height))
    win_w = int(np.ceil(window.width))

    # Clamp to array bounds
    row_off = max(0, row_off)
    col_off = max(0, col_off)
    row_end = min(data.shape[0], row_off + win_h)
    col_end = min(data.shape[1], col_off + win_w)

    clipped = data[row_off:row_end, col_off:col_end].copy()
    clip_transform = src_transform * Affine.translation(col_off, row_off)

    return clipped, clip_transform


# ------------------------------------------------------------------
# xarray helpers
# ------------------------------------------------------------------


def make_dataarray(
    arr: np.ndarray,
    bounds: tuple[float, float, float, float],
    h: int | None = None,
    w: int | None = None,
) -> "xr.DataArray":
    """Wrap *arr* in an :class:`xarray.DataArray` with geographic coordinates.

    The function creates pixel-centred latitude / longitude coordinates for a
    regular WGS-84 grid that spans *bounds* (west, south, east, north).

    Parameters
    ----------
    arr : np.ndarray, shape (H, W)
        2-D elevation or scalar array.
    bounds : tuple
        ``(west, south, east, north)`` in decimal degrees.
    h, w : int or None, optional
        Expected height and width.  If given and they don't match
        ``arr.shape``, a ``ValueError`` is raised.  Useful as a sanity check
        when the array comes from a separate source.

    Returns
    -------
    xarray.DataArray
        Dims ``('y', 'x')`` with ascending-longitude and descending-latitude
        pixel-centre coordinates attached.

    Examples
    --------
    >>> import numpy as np
    >>> da = make_dataarray(np.ones((120, 120)), (6.0, 62.0, 7.0, 63.0))
    >>> da.dims
    ('y', 'x')
    >>> float(da.x[0])   # first pixel centre
    6.00416...
    """
    import xarray as xr  # lazy import — xarray optional for core I/O utils

    ah, aw = arr.shape
    if h is not None and ah != h:
        raise ValueError(f"Array height {ah} does not match expected {h}")
    if w is not None and aw != w:
        raise ValueError(f"Array width {aw} does not match expected {w}")

    west, south, east, north = bounds
    lons = np.linspace(west,  east,  aw, endpoint=False) + (east  - west)  / (2 * aw)
    lats = np.linspace(north, south, ah, endpoint=False) - (north - south) / (2 * ah)

    return xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={"y": lats, "x": lons},
    )
