"""
paleoeurope.fusion.fabdem_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load FABDEM v1.2 tiles into numpy arrays.

FABDEM (Forest- and Building-removed DEM) is a bare-earth 1 arc-second DEM
derived from Copernicus GLO-30. Tiles follow the standard SRTM naming
convention (e.g. N51E000 = 51–52°N, 0–1°E).

Reference: Hawker et al. (2022) https://doi.org/10.5194/essd-14-4677-2022
Data DOI:  https://doi.org/10.5523/bris.25wfy0f9ukibjp75gb8laq1ghq
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:  # pragma: no cover
    HAS_GEOPANDAS = False


class FabdemLoader:
    """Load FABDEM tiles covering an arbitrary bounding box.

    Parameters
    ----------
    fabdem_dir : str or Path
        Directory (or tree) containing FABDEM GeoTIFF tiles.
    index_path : str or Path, optional
        Path for the cached spatial index (GeoParquet). Created on first use.
        Default: ``fabdem_index.parquet`` in the current working directory.

    Notes
    -----
    On first instantiation the loader scans ``fabdem_dir`` and builds a
    spatial index from tile filenames (fast, O(N)) or rasterio metadata
    (slow fallback).  The index is cached in memory for the process lifetime
    and optionally persisted to *index_path* as GeoParquet.

    FABDEM tiles use the convention ``Nxx Eyyyy`` — the letters indicate the
    SW corner of the tile, and each tile covers exactly 1° × 1°.

    Examples
    --------
    >>> loader = FabdemLoader("data/raw/fabdem/")
    >>> arr, transform, crs = loader.read_window((0.0, 51.0, 1.0, 52.0))
    >>> arr.shape
    (3600, 3600)
    """

    _index_cache: object = None  # class-level in-memory cache
    _TILE_PATTERN = re.compile(r"([NS])(\d{2})([EW])(\d{3})")

    def __init__(
        self,
        fabdem_dir: str | Path,
        index_path: str | Path = "fabdem_index.parquet",
    ) -> None:
        self.fabdem_dir = Path(fabdem_dir)
        self.index_path = Path(index_path)

        if not self.fabdem_dir.exists():
            raise FileNotFoundError(f"FABDEM directory not found: {fabdem_dir}")

        if FabdemLoader._index_cache is not None:
            self._index = FabdemLoader._index_cache
        else:
            self._index = self._load_or_create_index()
            FabdemLoader._index_cache = self._index

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_create_index(self) -> object:
        """Return spatial index as GeoDataFrame, loading or rebuilding as needed."""
        if HAS_GEOPANDAS and self.index_path.exists():
            try:

                return gpd.read_parquet(self.index_path)
            except Exception as exc:  # pragma: no cover
                import warnings

                warnings.warn(f"Cached index unreadable ({exc}), rebuilding.")
        return self._build_index()

    def _build_index(self) -> object:
        """Scan *fabdem_dir* and return a spatial index (GeoDataFrame or list)."""
        tif_files = list(self.fabdem_dir.glob("**/*.tif"))
        records: list[dict] = []

        for path in tif_files:
            m = self._TILE_PATTERN.search(path.name)
            if m:
                lat_dir, lat_v, lon_dir, lon_v = m.groups()
                lat = int(lat_v) * (1 if lat_dir == "N" else -1)
                lon = int(lon_v) * (1 if lon_dir == "E" else -1)
                minx, miny, maxx, maxy = float(lon), float(lat), float(lon + 1), float(lat + 1)
                # Verify: if the file is a sub-tile crop (fixture), use actual
                # file bounds so the spatial index is geometrically correct.
                try:
                    with rasterio.open(path) as _src:
                        fb = _src.bounds
                    tol = 1e-4  # ~10 m tolerance
                    if (fb.left > minx + tol or fb.bottom > miny + tol
                            or fb.right < maxx - tol or fb.top < maxy - tol):
                        minx, miny, maxx, maxy = fb.left, fb.bottom, fb.right, fb.top
                except Exception:  # pragma: no cover
                    pass
            else:
                try:
                    with rasterio.open(path) as src:
                        minx, miny, maxx, maxy = src.bounds
                except Exception:  # pragma: no cover
                    continue

            records.append(
                {
                    "path": str(path),
                    "minx": minx,
                    "miny": miny,
                    "maxx": maxx,
                    "maxy": maxy,
                }
            )

        if not HAS_GEOPANDAS:
            return records  # plain list fallback


        from shapely.geometry import box as shapely_box

        for r in records:
            r["geometry"] = shapely_box(r["minx"], r["miny"], r["maxx"], r["maxy"])

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326") if records else gpd.GeoDataFrame(
            columns=["path", "geometry"], crs="EPSG:4326"
        )

        try:
            gdf.to_parquet(self.index_path)
        except Exception:  # pragma: no cover
            pass

        return gdf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_window(
        self,
        bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray | None, object, object]:
        """Read FABDEM data for the given bounding box.

        Parameters
        ----------
        bounds : tuple
            ``(minx, miny, maxx, maxy)`` in EPSG:4326 decimal degrees.
        target_shape : tuple of (height, width), optional
            If provided, the output array is resampled to this exact pixel size.
            If ``None``, the native FABDEM resolution is used.

        Returns
        -------
        data : np.ndarray or None
            ``float32`` array of shape ``(height, width)``.  NaN where no data.
            Returns ``None`` if no tiles intersect *bounds*.
        transform : rasterio.Affine or None
            Affine transform of *data*.
        crs : rasterio.CRS or None
            Coordinate reference system (EPSG:4326).
        """
        request_box = box(*bounds)
        candidates = self._candidates(request_box)

        if not candidates:
            return None, None, None

        # Determine output dimensions
        first_path = candidates[0]["path"]
        with rasterio.open(first_path) as src:
            res_x = src.res[0]
            src_crs = src.crs

        minx, miny, maxx, maxy = bounds
        if target_shape:
            height, width = target_shape
        else:
            width = max(1, round((maxx - minx) / res_x))
            height = max(1, round((maxy - miny) / res_x))

        dst_transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
        canvas = np.full((height, width), np.nan, dtype=np.float32)

        for rec in candidates:
            tile_path = Path(rec["path"])
            if not tile_path.exists():  # stale index
                continue
            tile_box = box(rec["minx"], rec["miny"], rec["maxx"], rec["maxy"])
            intersection = tile_box.intersection(request_box)
            if intersection.is_empty:
                continue

            ib = intersection.bounds  # (minx, miny, maxx, maxy)

            d_col0 = round((ib[0] - minx) / (maxx - minx) * width)
            d_col1 = round((ib[2] - minx) / (maxx - minx) * width)
            d_row0 = round((maxy - ib[3]) / (maxy - miny) * height)
            d_row1 = round((maxy - ib[1]) / (maxy - miny) * height)

            d_col0, d_col1 = max(0, d_col0), min(width, d_col1)
            d_row0, d_row1 = max(0, d_row0), min(height, d_row1)
            win_w, win_h = d_col1 - d_col0, d_row1 - d_row0

            if win_w <= 0 or win_h <= 0:
                continue

            try:
                with rasterio.open(tile_path) as src:
                    src_win = from_bounds(*ib, transform=src.transform)
                    data = src.read(
                        1,
                        window=src_win,
                        out_shape=(win_h, win_w),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)

                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan

            except Exception:  # pragma: no cover
                continue

            canvas[d_row0:d_row1, d_col0:d_col1] = data

        return canvas, dst_transform, src_crs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidates(self, request_box: object) -> list[dict]:
        """Return tile records that intersect *request_box*."""
        if HAS_GEOPANDAS and hasattr(self._index, "geometry"):
            gdf = self._index
            hits = gdf[gdf.geometry.intersects(request_box)]
            return hits.to_dict("records")

        # Fallback: plain list scan
        result = []
        rb = request_box.bounds
        for rec in self._index:  # type: ignore[union-attr]
            if rec["maxx"] >= rb[0] and rec["minx"] <= rb[2] and rec["maxy"] >= rb[1] and rec["miny"] <= rb[3]:
                result.append(rec)
        return result
