"""
paleoeurope.utils.tile_index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovers available GIA-corrected and fusion DEM tiles in a directory and
organises them by epoch.

The expected filename convention (produced by :mod:`paleoeurope.gia.correction_pipeline`
and the production Celery workers) is:

    ``<TILE_ID>_t<NN>ka.tif``    — GIA-corrected paleo-DEM  (e.g. ``N62E006_t21ka.tif``)
    ``<TILE_ID>_fusion.tif``      — Modern fused DEM fallback

where ``<TILE_ID>`` is the FABDEM/SRTM 1°×1° tile identifier
``(N|S)DD(E|W)DDD`` (e.g. ``N62E006``).

Bounding box convention
-----------------------
All bounding boxes in this module use the ``(west, south, east, north)`` order
(same as :func:`rasterio.transform.from_bounds`).  This is also aliased as the
type ``BBox = tuple[float, float, float, float]``.

Usage
-----
>>> from paleoeurope.utils.tile_index import TileIndex
>>> idx = TileIndex(tiles_dir='outputs/demo_fusion')
>>> epoch_paths = idx.collect_epoch_paths(
...     bbox=(-2.0, 52.0, 9.0, 58.0),
...     epochs_ka=[0.0, 8.0, 12.0, 21.0],
... )
>>> epoch_paths[21.0]
{'N56E003': PosixPath('.../N56E003_t21ka.tif'), ...}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

__all__ = ["TileIndex", "BBox"]

# Type alias: (west, south, east, north) in decimal degrees.
BBox = tuple[float, float, float, float]

# Compiled regex patterns for tile filenames.
_RE_TILE_ID    = re.compile(r"^(N|S)(\d{2})(E|W)(\d{3})$")
_RE_PALEO_TILE = re.compile(r"^(.+)_t(\d+)ka$")
_RE_FUSION     = re.compile(r"^(.+)_fusion$")


def _parse_sw_corner(tile_id: str) -> Optional[tuple[float, float]]:
    """Return ``(lon_sw, lat_sw)`` for a bare tile ID or ``None`` if not matching."""
    m = _RE_TILE_ID.match(tile_id)
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return float(lon), float(lat)


def _tile_in_bbox(lon_sw: float, lat_sw: float, bbox: BBox) -> bool:
    """Return True if the SW corner of a 1°×1° tile falls inside *bbox*."""
    west, south, east, north = bbox
    return west <= lon_sw < east and south <= lat_sw < north


class TileIndex:
    """Scans a directory and indexes available paleo-DEM and fusion tiles.

    Parameters
    ----------
    tiles_dir : str or Path
        Directory to scan.  Typically the output of the fusion + GIA pipeline
        (e.g. ``outputs/demo_fusion/``).

    Attributes
    ----------
    tiles_dir : Path
    """

    def __init__(self, tiles_dir: str | Path) -> None:
        self.tiles_dir = Path(tiles_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_epoch_paths(
        self,
        bbox: BBox,
        epochs_ka: list[float],
    ) -> dict[float, dict[str, Path]]:
        """Discover available tiles for each epoch within *bbox*.

        Priority per epoch:
        1. GIA-corrected tile ``*_t{N}ka.tif`` (preferred).
        2. Modern fusion tile ``*_fusion.tif``  (fallback if GIA tile missing).

        Parameters
        ----------
        bbox : BBox
            ``(west, south, east, north)`` in decimal degrees.
        epochs_ka : list[float]
            Target epochs in kiloyears BP.

        Returns
        -------
        dict[float, dict[str, Path]]
            ``{epoch_ka: {tile_id: Path, ...}, ...}``
            The inner dict is empty if no tiles were found for an epoch.

        Examples
        --------
        >>> idx = TileIndex('outputs/demo_fusion')
        >>> paths = idx.collect_epoch_paths(bbox=(5.9, 61.9, 7.1, 63.1), epochs_ka=[0.0, 21.0])
        >>> list(paths[21.0].keys())
        ['N62E006']
        """
        paleo_tiles: dict[int, dict[str, Path]] = {}
        fusion_tiles: dict[str, Path] = {}

        for f in self.tiles_dir.glob("*.tif"):
            stem = f.stem

            m_p = _RE_PALEO_TILE.match(stem)
            if m_p:
                bare, ep_int = m_p.group(1), int(m_p.group(2))
                coords = _parse_sw_corner(bare)
                if coords and _tile_in_bbox(*coords, bbox):
                    paleo_tiles.setdefault(ep_int, {})[bare] = f
                continue

            m_f = _RE_FUSION.match(stem)
            if m_f:
                bare   = m_f.group(1)
                coords = _parse_sw_corner(bare)
                if coords and _tile_in_bbox(*coords, bbox):
                    fusion_tiles[bare] = f

        result: dict[float, dict[str, Path]] = {}
        for ep in epochs_ka:
            ep_int = int(round(ep))
            if ep_int in paleo_tiles and paleo_tiles[ep_int]:
                result[ep] = paleo_tiles[ep_int]
            elif fusion_tiles:
                result[ep] = dict(fusion_tiles)   # copy; fallback
            else:
                result[ep] = {}

        return result

    # ------------------------------------------------------------------
    # Module-level convenience function
    # ------------------------------------------------------------------


def collect_epoch_tile_paths(
    tiles_dir: str | Path,
    bbox: BBox,
    epochs_ka: list[float],
) -> dict[float, dict[str, Path]]:
    """Module-level convenience wrapper around :class:`TileIndex`.

    Equivalent to ``TileIndex(tiles_dir).collect_epoch_paths(bbox, epochs_ka)``.
    Kept for backwards-compatibility with the original inline notebook helper.
    """
    return TileIndex(tiles_dir).collect_epoch_paths(bbox, epochs_ka)
