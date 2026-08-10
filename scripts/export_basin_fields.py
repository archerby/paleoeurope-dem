#!/usr/bin/env python3
"""Persist, per epoch, the two arrays ``assign_basins`` computes and throws away.

WHAT THIS IS FOR
================
The 43 baked rasters under ``khvalynian_caspian/topo`` are the right product and
none of them needs re-baking for geometry.  What they do NOT carry is the
PARTITION: which basin owns each cell, and what water surface that cell sits
under.  Both are computed in memory on every render, used to decide what to
paint, and then discarded with the process.

That is the one thing a 4D asset cannot re-derive cheaply, and the one thing it
must not re-implement: "which basin owns this cell" is a seeded priority flood
over a 28-million-cell grid whose answer depends on the seed registry, the three
level rules, the flood ceiling and the connectivity definition.  A second
implementation of it downstream would be a second, unpinned definition of the
same partition, and the two would diverge silently.  So the definition of record
is imported (:func:`paleoeurope.viz.paleocostline_render.assign_basins`), run
once per epoch, and WRITTEN DOWN.

THE THREE OWNER DECISIONS THIS IMPLEMENTS
=========================================
**D-A — the frame is cropped north.**  Every nodata hole in the baked rasters
lies in the far north of the extent.  Cropping north of
:data:`CROP_NORTH_LAT_DEG` removes the nodata question from the product entirely
and shrinks the cos(latitude) unroll error of the 3D asset.  The cost, accepted:
the Fennoscandian forebulge leaves the scene.

WHICH PART OF THAT THIS MODULE MEASURES, AND WHICH PART IT INHERITS.  It
measures where the holes are — :func:`survey_nodata` reports, per epoch, the
count and the southernmost row, and every written epoch asserts the cropped
window holds none.  It does NOT re-measure the rest of the finding the decision
rests on: that no hole is near the water, near the Manych sill, or touching a
coastline or the corridor at any epoch.  That was established when the decision
was taken and is inherited here rather than reproduced, which is why this
paragraph says so instead of restating it as if it came out of this pass.

The bound is a DECISION and the measurement it rests on is a MEASUREMENT, and
this module keeps them apart.  :func:`survey_nodata` measures, over all 43
rasters, the minimum northern bound that leaves zero nodata cells; the export
then applies the chosen bound and records BOTH, plus how much more latitude the
choice discards than the measurement required.  Discarding more data than a
decision needs is allowed; discarding it without saying how much is not.

**D-C — three water sheets clipped by basin_id; the corridor is not drawn.**  So
``corridor="step"`` is the mode of record here: every basin flat at its own
level, the spill corridor MARKED and not given an invented surface.

**D-F — no ramp corridor surface in the 3D asset.**  Follows from D-C.

WHAT IS WRITTEN, AND WHY IT IS THREE FILES AND NOT TWO
======================================================
Per epoch, into :data:`OUT_ROOT`:

``basin_id/basin_id_<epoch>ka.tif``
    ``uint8``.  ``0`` = unowned; ``1..N`` = the basin of that index in the seed
    registry's own order.  The mapping is not implied by the order: it is written
    into the ``basin_code_map`` tag of every file, by name.

``water_level/water_level_<epoch>ka.tif``
    ``float32``, deflate + ``predictor=3``.  The per-cell water SURFACE.  Nodata
    exactly where no basin owns the cell, at the sentinel INHERITED from the
    source raster (:func:`field_nodata_for`) rather than declared here, so the
    two trees cannot drift onto different ones.

``water_state/water_state_<epoch>ka.tif``
    ``uint8`` BITFIELD.  This file is not in the original two-raster spec and it
    is here because leaving it out would have been a STATE SUBSTITUTION.

    ``basin_id`` and ``water_level_m`` together let a consumer evaluate the FLAT
    test ``dem < water_level_m``.  That is not what the engine means by water.
    A cell is water iff it lies in the connected component of ``{dem < level}``
    that contains that basin's seed — the condition that stops a closed pit in
    Anatolia standing below the Caspian's surface being painted as Caspian
    water — and there is a THIRD state besides water and land: a body that holds
    no seed and reaches the edge of what the raster can observe, whose
    connectivity this grid cannot decide either way.  A consumer given only the
    first two files would have to re-run a flood fill to recover any of that,
    which is the second-definition problem this export exists to remove, and the
    flat test it would fall back on differs from the engine's answer by the
    cell count recorded in every file's ``connectivity_*`` tags.  So the three
    states ship as data.  The bits are named in the ``water_state_bits`` tag.

WHAT IS REFUSED, NEVER SUBSTITUTED
==================================
Each of these ends the epoch with a named error recorded in the manifest and NO
raster written.  None of them is worked around:

* **the Pontic level has no row at this epoch.**  Eight of the 43 epochs are
  quarter-ka steps that the Black Sea curve does not carry, and ``basin_level``
  refuses to take a neighbouring row because the two rows bracketing such an
  epoch differ by tens of metres and carry different provenance.  A three-basin
  partition cannot be built without a third level, and an interpolated one would
  be indistinguishable downstream from a measured one.
* **a seed box does not resolve to a wet cell.**  ``resolve_seed`` refuses when
  the lowest cell of a registered box stands at or above its basin's own surface
  at that epoch.  A seed on dry land grows an empty basin and the frame loses a
  sea without failing.  The seed boxes' justifications were written for the
  17-13 ka window and this export runs the full 26-0 ka grid, so this is a live
  possibility and it is checked at every epoch rather than assumed.
* **the Manych saddle this grid resolves disagrees with the measured crest.**
  ``assign_basins`` raises past its own tolerance.  Widening it is a decision to
  be made in writing, and it is not made here.
* **the cropped window still holds nodata.**  D-A rests on a measurement; a
  failed assertion means the measurement was wrong, and the export stops rather
  than writing a product whose nodata story is untrue.

PROVENANCE
==========
Every raster carries the source raster's sha256, its own crop window, the three
levels with the rule and the provenance string that produced each, the three
resolved seeds with the box and the rule that resolved them, the sill verdict for
all three basin pairs, the connectivity states, the exact partition counts, the
flood ceiling, the sha256 of this script and of the renderer module that did the
work, and the versions of the libraries that ran it.  The shipped
``MANYCH_INSET`` crops are the pattern.

No tag template in this module contains a digit.  Every number in every tag is
interpolated from a quantity computed for the artifact being written — including
the pixel size, the crop bound and the epoch — because a figure typed into a
sentence survives the change that falsifies it, and six sentences have already
been retracted in this project for exactly that.

Numbers travel with their units.  ``source_nodata_cells_in_full_raster`` and
``source_nodata_cells_in_crop_window`` say "raster cells" in their own tag names'
company, and the source raster's own ``gia_unknown_ice6g_cells`` tag is copied
through VERBATIM under a name that says it was read from the source rather than
measured here — the two count different things and this file does not reconcile
them.

RUNNING IT
==========
::

    python scripts/export_basin_fields.py --survey        # measure, write no raster
    python scripts/export_basin_fields.py                 # survey + export all 43

A native flood takes long enough per epoch that the whole grid outlives a
backgrounded shell task.  Launch it DETACHED and poll the log, which carries a
wall time per epoch and a total:

    Start-Process -FilePath python -ArgumentList 'scripts/export_basin_fields.py'
                  -RedirectStandardOutput <log> -NoNewWindow

Nothing of size is written to ``D:``.  Everything goes under :data:`OUT_ROOT`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paleoeurope.viz.paleocostline_render import (  # noqa: E402
    BasinField,
    BasinSeed,
    _row_area_km2,
    assign_basins,
    declared_vertical_frame,
)
from scripts.animate_khvalynian_spill import (  # noqa: E402
    SEED_BOXES,
    SEED_RULE,
    BasinLevel,
    ResolvedSeed,
    basin_levels_for,
    dz_at_bosphorus,
    raster_for,
    resolve_seeds,
    sha256_of,
    sill_expectations_for,
)

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def _epochs() -> tuple[float, ...]:
    """The epoch grid, imported and never re-typed.

    A second hand-written list is a second definition that can acquire a gap
    without anybody noticing.
    """
    from scripts import bake_khvalynian as bk

    return tuple(float(e) for e in bk.EPOCHS_KA)


def _topo_dir() -> Path:
    """Where the baked rasters are read from.

    Resolved exactly the way the animation resolves it, so the two cannot end up
    pointing at different trees.
    """
    from scripts.animate_khvalynian_spill import KHV_TOPO_DIR

    return Path(KHV_TOPO_DIR)


#: Everything this script writes.  Deliberately NOT on ``D:``, which had under
#: four gigabytes free when this was written and has already destroyed one file
#: this phase by filling up mid-write.
OUT_ROOT = Path(
    os.environ.get("PALEO_BASIN_FIELDS_ROOT")
    or (Path(os.environ.get("PALEO_RENDER_ROOT") or r"Y:\paleo\renders")
        / "khvalynian_basin_fields")
)

#: Free bytes required under :data:`OUT_ROOT` before a run starts.  One epoch's
#: three rasters are tens of megabytes; the whole grid is a couple of gigabytes.
#: Checked rather than hoped for, because the failure mode of not checking is a
#: truncated file that looks like a written one.
MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# D-A — the crop
# ---------------------------------------------------------------------------

#: THE OWNER'S DECISION, 2026-08-05: crop the frame north at this latitude.
#:
#: It is deliberately NOT the measured minimum.  :func:`survey_nodata` measures
#: what the minimum is; :func:`crop_record` reports the difference between the
#: two on every raster written, so the choice can be revisited against the number
#: instead of against a memory of it.
CROP_NORTH_LAT_DEG = 51.0

#: How far the requested bound may sit off the source raster's own cell grid
#: before the crop is refused.  A crop that has to round is a crop that moves the
#: bound, and a moved bound is a different scene from the one recorded.
CROP_SNAP_TOL_DEG = 1e-9

CROP_RULE = (
    "the northern bound is a DECISION and is applied by taking whole rows of the "
    "SOURCE cell grid: the window starts at the first row whose northern edge "
    "lies at or below the bound and runs to the bottom of the raster, over every "
    "column. No resampling, no interpolation, no reprojection -- the cells of "
    "this file ARE cells of the source file, so an elevation read here is the "
    "elevation the bake wrote. The bound is required to fall on a cell boundary "
    "to within a tolerance recorded in the crop_snap_tol_deg tag; a bound that "
    "would have to be rounded is REFUSED rather than moved, because a moved "
    "bound is a different scene from the one the decision named"
)

def field_nodata_for(source_nodata: Optional[float], levels_m: Sequence[float]) -> float:
    """The sentinel the ``water_level`` raster marks unowned cells with.

    Taken from the SOURCE raster rather than declared here, so the two trees
    cannot drift onto different sentinels and so no number is typed into this
    module that a re-bake could falsify.

    Raises
    ------
    CropBoundError
        The source declares no nodata value, or its sentinel is not safely clear
        of the water surfaces being written.  A sentinel inside the range of the
        data makes a real surface indistinguishable from an absent one.
    """
    if source_nodata is None or not np.isfinite(source_nodata):
        raise CropBoundError(
            "the source raster declares no finite nodata value, so there is no "
            "sentinel to inherit for the unowned cells of the water surface. "
            "One is not invented here: a sentinel chosen downstream can collide "
            "with a sentinel chosen upstream and nothing would say which a given "
            "cell carries."
        )
    sentinel = float(source_nodata)
    finite = [float(v) for v in levels_m if np.isfinite(v)]
    if finite:
        margin = min(abs(sentinel - v) for v in finite)
        span = max(finite) - min(finite)
        if margin <= max(span, 1.0):
            raise CropBoundError(
                f"the source raster's nodata sentinel {_fmt(sentinel)} is only "
                f"{_fmt(margin)} m from the nearest water surface being written "
                f"({_kv({'min_level_m': _fmt(min(finite)), 'max_level_m': _fmt(max(finite))})}). "
                "A sentinel that close to the data cannot be told from a real "
                "surface, and this module will not pick a different one for you."
            )
    return sentinel


#: The corridor mode of record, fixed by owner decision D-C/D-F.  Not a knob:
#: "ramp" would write an interpolated water surface into the corridor, which is
#: precisely the surface the owner decided not to draw.
CORRIDOR_MODE = "step"

#: Bits of the ``water_state`` raster.  ``(name, value, meaning)``.
STATE_BITS: tuple[tuple[str, int, str], ...] = (
    (
        "water",
        1,
        "BasinField.water_mask -- the cell lies in the connected component of "
        "{dem < water_level_m} that contains its basin's own seed. THIS, not the "
        "flat test, is what the engine means by water",
    ),
    (
        "unobservable",
        2,
        "BasinField.unobservable_mask -- the cell lies in a component that holds "
        "no seed and reaches the edge of what this raster can observe, so whether "
        "it is water cannot be decided from this grid at all. Disjoint from the "
        "water bit by construction. Draw it as neither water nor land",
    ),
    (
        "corridor",
        4,
        "BasinField.corridor_mask -- a cell of the spill corridor. In the corridor "
        "mode of record this is a MARK and nothing else: the cell is NOT given an "
        "invented water surface and is NOT painted. It may or may not also carry "
        "the water bit, which is why these are bits and not codes",
    ),
)


class CropBoundError(ValueError):
    """The requested northern bound cannot be taken off the source cell grid."""


class NodataInCropError(AssertionError):
    """The cropped window still holds nodata, so decision D-A rests on a wrong measurement."""


class PartitionError(AssertionError):
    """The per-basin cell counts and the unowned count do not sum to the window."""


class SeedBoxOutsideCropError(ValueError):
    """A registered seed box is not wholly inside the cropped window."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def epoch_tag(epoch_ka: float) -> str:
    """``16.5`` -> ``"16.5"``, spelled the way the bake spells its file names."""
    return f"{float(epoch_ka):g}"


def _fmt(value: float, digits: int = 6) -> str:
    """A float as a plain decimal string, or the word for what it is instead."""
    number = float(value)
    if np.isnan(number):
        return "nan"
    if np.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _kv(pairs: dict) -> str:
    """``{"a": 1, "b": 2}`` -> ``"a=1;b=2"`` -- machine-readable, no prose."""
    return ";".join(f"{k}={v}" for k, v in pairs.items())


def sha256_of_text_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def toolchain() -> dict:
    """What ran this, as values rather than as an assumption about the machine."""
    import rasterio

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "platform": platform.platform(),
    }


def code_provenance() -> dict:
    """The two files that decided every number here, fingerprinted.

    The renderer module is the definition of record for the partition and this
    script is the definition of record for the crop and the tags; a reader who
    wants to know why a cell has the label it has needs both, and a path alone
    identifies a name, not contents.
    """
    from paleoeurope.viz import paleocostline_render as render
    from scripts import animate_khvalynian_spill as anim

    entries = {}
    for label, module in (
        ("export_script", sys.modules[__name__]),
        ("renderer", render),
        ("registry", anim),
    ):
        source = Path(getattr(module, "__file__"))
        entries[label] = {
            "path": str(source),
            "sha256": sha256_of_text_file(source),
            "bytes": source.stat().st_size,
        }
    return entries


# ---------------------------------------------------------------------------
# THE MEASUREMENT D-A RESTS ON
# ---------------------------------------------------------------------------


def survey_nodata(
    epochs: Sequence[float],
    topo_dir: Path | str,
    *,
    strip_rows: int = 512,
    progress: Optional[Any] = None,
) -> dict:
    """Where the nodata actually is, over every epoch, measured not assumed.

    Returns a record carrying, per epoch, the nodata cell count and the
    southernmost row holding one; and across all epochs, the MINIMUM NORTHERN
    BOUND that leaves zero nodata cells.

    Read in strips so a full-resolution pass over the whole grid costs a strip of
    memory rather than a raster of it.  There is no early exit: proving that a
    window holds zero nodata requires looking at all of it.
    """
    import rasterio

    topo_dir = Path(topo_dir)
    per_epoch: dict[str, dict] = {}
    worst_row = -1
    worst_epoch: Optional[float] = None
    grid: Optional[dict] = None

    for epoch in epochs:
        path = raster_for(epoch, topo_dir)
        with rasterio.open(path) as dataset:
            nodata = dataset.nodata
            height, width = dataset.height, dataset.width
            top = float(dataset.bounds.top)
            pixel_deg = abs(float(dataset.transform.e))
            if grid is None:
                grid = {
                    "height": height,
                    "width": width,
                    "top_lat_deg": top,
                    "bottom_lat_deg": float(dataset.bounds.bottom),
                    "west_lon_deg": float(dataset.bounds.left),
                    "east_lon_deg": float(dataset.bounds.right),
                    "pixel_deg": pixel_deg,
                    "pixel_arcsec": pixel_deg * 3600.0,
                    "crs": str(dataset.crs),
                }
            elif (height, width) != (grid["height"], grid["width"]):
                raise CropBoundError(
                    f"{path.name} is {height}x{width} but the epochs surveyed "
                    f"before it are {grid['height']}x{grid['width']}. A crop "
                    "measured on one grid does not describe another."
                )
            count = 0
            last_row = -1
            first_row = -1
            for row_off in range(0, height, strip_rows):
                rows = min(strip_rows, height - row_off)
                block = dataset.read(
                    1, window=rasterio.windows.Window(0, row_off, width, rows)
                )
                bad = ~np.isfinite(block)
                if nodata is not None:
                    bad |= block == nodata
                if not bad.any():
                    continue
                count += int(bad.sum())
                hit = np.flatnonzero(bad.any(axis=1))
                if first_row < 0:
                    first_row = row_off + int(hit[0])
                last_row = row_off + int(hit[-1])
            source_tag = dataset.tags().get("gia_unknown_ice6g_cells")

        per_epoch[epoch_tag(epoch)] = {
            "epoch_ka": float(epoch),
            "raster": str(path),
            "nodata_cells_raster": count,
            "first_nodata_row": first_row,
            "last_nodata_row": last_row,
            "last_nodata_row_south_edge_lat_deg": (
                None
                if last_row < 0
                else float(grid["top_lat_deg"] - (last_row + 1) * grid["pixel_deg"])
            ),
            "source_tag_gia_unknown_ice6g_cells_as_read": source_tag,
        }
        if last_row > worst_row:
            worst_row = last_row
            worst_epoch = float(epoch)
        if progress is not None:
            progress(
                f"survey {epoch_tag(epoch)} ka: {count} nodata raster cell(s), "
                f"southernmost row {last_row}"
            )

    assert grid is not None, "no epoch was surveyed"
    if worst_row < 0:
        minimum_bound = float(grid["top_lat_deg"])
        rule = (
            "no epoch of this grid holds a single nodata raster cell, so nodata "
            "requires no crop at all and the least that can be cropped is "
            "nothing: the bound is the raster's own NORTHERN edge"
        )
    else:
        minimum_bound = float(
            grid["top_lat_deg"] - (worst_row + 1) * grid["pixel_deg"]
        )
        rule = (
            "the southern edge of the SOUTHERNMOST row that holds a nodata cell "
            "in ANY epoch. Every row south of it is finite at every epoch, so a "
            "window whose northern bound is at or below this latitude holds zero "
            "nodata raster cells; a window even one row taller holds at least one"
        )

    return {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topo_dir": str(topo_dir),
        "epochs_surveyed": len(per_epoch),
        "grid": grid,
        "unit_note": (
            "every count in this record is in RASTER CELLS of the grid described "
            "under 'grid'. The source rasters carry a tag named "
            "gia_unknown_ice6g_cells whose number is copied through here VERBATIM, "
            "per epoch, under a key that says it was read rather than measured: it "
            "counts cells of a different grid and this record does not reconcile "
            "the two"
        ),
        "total_nodata_cells_raster": sum(
            e["nodata_cells_raster"] for e in per_epoch.values()
        ),
        "epochs_with_nodata": sorted(
            e["epoch_ka"] for e in per_epoch.values() if e["nodata_cells_raster"]
        ),
        "southernmost_nodata_row": worst_row,
        "southernmost_nodata_row_epoch_ka": worst_epoch,
        "minimum_clean_north_lat_deg": minimum_bound,
        "minimum_clean_north_rule": rule,
        "per_epoch": per_epoch,
    }


def crop_record(survey: dict, north_lat: float = CROP_NORTH_LAT_DEG) -> dict:
    """The chosen bound beside the measured one, with the difference stated.

    This is the block that keeps decision D-A honest.  It says how much MORE
    latitude the chosen bound discards than the measurement required, in degrees,
    in rows, and in the area those rows cover — so the owner can revisit the
    choice against a number instead of against a recollection of one.
    """
    grid = survey["grid"]
    pixel = float(grid["pixel_deg"])
    top = float(grid["top_lat_deg"])
    measured = float(survey["minimum_clean_north_lat_deg"])
    chosen = float(north_lat)

    rows_chosen = (top - chosen) / pixel
    rows_measured = (top - measured) / pixel
    extra_rows = int(round(rows_chosen - rows_measured))
    # Positive means the chosen bound sits SOUTH of the measured minimum, i.e.
    # the choice throws away latitude the nodata question did not require.
    extra_deg = measured - chosen

    # Area of the extra rows, on the raster's own row-area rule, so the figure is
    # the cos(latitude) one rather than a rectangle.
    import rasterio

    transform = rasterio.transform.from_origin(
        float(grid["west_lon_deg"]), top, pixel, pixel
    )
    row_area = _row_area_km2(transform, int(grid["height"]))
    lo = int(round(rows_measured))
    hi = int(round(rows_chosen))
    extra_km2 = float(np.sum(row_area[min(lo, hi):max(lo, hi)]) * int(grid["width"]))
    chosen_is_north_of_clean = chosen > measured

    return {
        "chosen_north_lat_deg": chosen,
        "chosen_north_rule": "OWNER DECISION D-A, 2026-08-05",
        "measured_minimum_clean_north_lat_deg": measured,
        "measured_minimum_rule": survey["minimum_clean_north_rule"],
        "extra_latitude_discarded_deg": float(extra_deg),
        "extra_rows_discarded": extra_rows,
        "extra_area_discarded_km2": extra_km2,
        "chosen_is_north_of_the_clean_bound": bool(chosen_is_north_of_clean),
        "extra_note": (
            "POSITIVE means the chosen bound lies SOUTH of the measured minimum, "
            "i.e. it discards that much latitude the nodata question did not "
            "require -- which is the ordinary case and is a cost of the decision, "
            "recorded on every raster written so the decision can be revisited "
            "against the measurement rather than against a memory of it. NEGATIVE "
            "means the chosen bound lies NORTH of it, i.e. the window still "
            "contains nodata rows, and every epoch that holds one will refuse at "
            "write time. Nothing here changes the bound either way"
        ),
        "rows_kept_of_source": int(grid["height"]) - hi,
        "rows_dropped_of_source": hi,
    }


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def crop_window(dataset, north_lat: float = CROP_NORTH_LAT_DEG):
    """The whole-row window of *dataset* lying at or below *north_lat*.

    Raises
    ------
    CropBoundError
        The bound does not fall on the source cell grid, or it leaves no rows, or
        it is above the raster's own top edge (which would silently keep them all
        while claiming a crop happened).
    """
    from rasterio.windows import Window

    top = float(dataset.bounds.top)
    bottom = float(dataset.bounds.bottom)
    pixel = abs(float(dataset.transform.e))
    bound = float(north_lat)

    if not (bottom < bound < top):
        raise CropBoundError(
            f"the requested northern bound {_fmt(bound)} is not strictly inside "
            f"this raster's latitude range {_fmt(bottom)}..{_fmt(top)}. A bound "
            "at or above the top edge crops nothing while naming a crop, and one "
            "at or below the bottom edge leaves no rows at all."
        )

    rows_f = (top - bound) / pixel
    row_off = int(round(rows_f))
    if abs(rows_f - row_off) > CROP_SNAP_TOL_DEG / pixel:
        landed = top - row_off * pixel
        raise CropBoundError(
            f"the requested northern bound {_fmt(bound)} falls "
            f"{_fmt(abs(rows_f - row_off) * pixel)} deg off this raster's cell "
            f"grid (cell size {_fmt(pixel)} deg); the nearest cell boundary is "
            f"{_fmt(landed)}. The bound is NOT rounded here: a moved bound is a "
            "different scene from the one the decision named. Choose a bound on "
            "the grid, or state the rounded one as the decision."
        )
    return Window(0, row_off, dataset.width, dataset.height - row_off)


def assert_seed_boxes_inside(
    window_bounds: tuple[float, float, float, float],
    boxes: Sequence[Any] = SEED_BOXES,
) -> dict:
    """Every registered seed box lies wholly inside the cropped bounds.

    This is what makes it legitimate to resolve the seeds against the FULL source
    raster and then use the resulting lon/lat on the cropped grid: the crop takes
    whole source cells, so a box entirely inside the window sees exactly the same
    cells in both, and its argmin is therefore the same cell.  Checked, not
    assumed — the whole point of the registry is that a coordinate nobody can
    re-derive can only be trusted.
    """
    west, south, east, north = window_bounds
    outside = [
        b.name
        for b in boxes
        if not (
            west <= b.lon_min and b.lon_max <= east
            and south <= b.lat_min and b.lat_max <= north
        )
    ]
    if outside:
        raise SeedBoxOutsideCropError(
            f"seed box(es) {outside!r} are not wholly inside the cropped window "
            f"{tuple(_fmt(v) for v in window_bounds)}. A box that leaves the "
            "window resolves its argmin over ground the cropped raster does not "
            "carry, so the seed it returns would not be a seed of the array the "
            "flood runs on."
        )
    return {
        "boxes_checked": [b.name for b in boxes],
        "window_bounds_wsen": [float(v) for v in window_bounds],
        "rule": (
            "the crop takes WHOLE SOURCE CELLS, so a seed box lying entirely "
            "inside the window holds exactly the same cells in the source raster "
            "and in the crop, and its deterministic argmin is therefore the same "
            "cell in both. That identity is why the seeds may be resolved against "
            "the source file and used on the cropped grid, and it is checked here "
            "rather than assumed"
        ),
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


@dataclass
class EpochFields:
    """One epoch's result: the paths written, or the reason there are none."""

    epoch_ka: float
    refusal: Optional[str] = None
    record: dict = dataclass_field(default_factory=dict)
    seconds: float = 0.0

    @property
    def written(self) -> bool:
        return self.refusal is None


def _basin_tag_block(
    names: Sequence[str],
    levels: dict[str, BasinLevel],
    seeds: Sequence[ResolvedSeed],
) -> dict[str, str]:
    """Per-basin tags: the code, the level, its rule, its frame, its provenance, its seed."""
    tags: dict[str, str] = {}
    tags["basin_code_map"] = _kv(
        {"0": "unowned", **{str(i + 1): n for i, n in enumerate(names)}}
    )
    tags["basin_code_map_note"] = (
        "the code written in basin_id_*.tif. Code zero means NO basin owns the "
        "cell -- it is above this epoch's flood ceiling, or in a depression that "
        "reaches no seed. It is a CODE, not a nodata sentinel: these rasters "
        "declare no nodata value at all, because the cropped window holds no "
        "cell without an answer"
    )
    tags["basin_levels_m"] = _kv(
        {n: _fmt(levels[n].level_m) for n in names}
    )
    for name in names:
        level = levels[name]
        tags[f"basin_level_m_{name}"] = _fmt(level.level_m)
        tags[f"basin_level_frame_{name}"] = level.frame
        tags[f"basin_level_rule_{name}"] = level.rule
        tags[f"basin_level_provenance_{name}"] = level.provenance
    for seed in seeds:
        tags[f"basin_seed_{seed.basin}"] = _kv(
            {
                "box": seed.box.name,
                "lon": _fmt(seed.lon),
                "lat": _fmt(seed.lat),
                "source_row": seed.row,
                "source_col": seed.col,
                "elevation_m": _fmt(seed.elevation_m),
                "level_m": _fmt(seed.level_m),
            }
        )
        tags[f"basin_seed_box_why_{seed.basin}"] = seed.box.why
    tags["basin_seed_rule"] = SEED_RULE
    return tags


def _sill_tag_block(field: BasinField, expectations: dict) -> dict[str, str]:
    """One tag per basin pair carrying measurement, expectation, gap and verdict."""
    tags: dict[str, str] = {}
    for key in sorted(field.sill_m):
        pair = f"{key[0]}__{key[1]}"
        measured = field.sill_m[key]
        expected = field.sill_expected_m.get(key)
        where = field.sill_lonlat.get(key)
        values = {
            "measured_saddle_m": _fmt(measured),
            "expected_m": "none" if expected is None else _fmt(float(expected)),
            "gap_m": (
                "none"
                if expected is None or not np.isfinite(measured)
                else _fmt(float(measured) - float(expected))
            ),
            "lon": "none" if where is None else _fmt(where[0]),
            "lat": "none" if where is None else _fmt(where[1]),
            "route_cells": len(field.sill_route_rc.get(key, ())),
        }
        tags[f"sill_{pair}"] = _kv(values)
        tags[f"sill_status_{pair}"] = field.sill_expectation_status.get(
            key, "NOT REPORTED BY assign_basins"
        )
        shift = field.sill_smoothing_shift_m.get(key)
        tags[f"sill_smoothing_shift_m_{pair}"] = (
            "NOT MEASURED: this export assigns basins on the raw source cells, so "
            "there is no second, smoothed array to difference against and no "
            "number is written here. Zero is not used as a stand-in"
            if shift is None
            else _fmt(float(shift))
        )
    tags["sill_expectation_pairs"] = _kv(
        {
            f"{a}__{b}": type(v).__name__
            for (a, b), v in sorted(expectations.items())
        }
    )
    return tags


def _partition_counts(field: BasinField, total_cells: int) -> dict:
    """The exact partition, and the guard that it IS one."""
    per_basin = {
        name: int((field.basin_id == i + 1).sum())
        for i, name in enumerate(field.names)
    }
    unowned = int((field.basin_id == 0).sum())
    stray = int(
        ((field.basin_id < 0) | (field.basin_id > len(field.names))).sum()
    )
    accounted = sum(per_basin.values()) + unowned
    if stray or accounted != total_cells:
        raise PartitionError(
            f"the basin labels do not partition the window: "
            f"{sum(per_basin.values())} owned + {unowned} unowned = {accounted} "
            f"against {total_cells} cells in the cropped window, with {stray} "
            "label(s) outside the declared code range "
            f"{_kv({str(i + 1): n for i, n in enumerate(field.names)})}. Either a "
            "cell carries a label no basin claims, or two counts overlap; either "
            "way the code map written into the raster would be a false "
            "description of the array beside it."
        )
    return {
        "per_basin": per_basin,
        "unowned": unowned,
        "total": total_cells,
        "verdict": "EXACT_PARTITION",
    }


def _water_counts(field: BasinField) -> dict:
    per_basin = {
        name: int((field.water_mask & (field.basin_id == i + 1)).sum())
        for i, name in enumerate(field.names)
    }
    return {
        "per_basin": per_basin,
        "total": int(field.water_mask.sum()),
        "unobservable": int(field.unobservable_mask.sum())
        if field.unobservable_mask.size
        else 0,
        "corridor_marked": int(field.corridor_mask.sum()),
    }


def build_tags(
    *,
    product: str,
    epoch_ka: float,
    source_path: Path,
    source_sha256: str,
    source_tags: dict,
    dem_frame: str,
    window,
    window_bounds: tuple[float, float, float, float],
    pixel_deg: float,
    survey: dict,
    crop: dict,
    seed_check: dict,
    field: BasinField,
    levels: dict[str, BasinLevel],
    seeds: Sequence[ResolvedSeed],
    expectations: dict,
    partition: dict,
    water: dict,
    nodata_in_crop: int,
    field_nodata: float,
    tools: dict,
    code: dict,
    seconds: float,
) -> dict[str, str]:
    """The full provenance tag set carried by every raster this module writes.

    NO TEMPLATE HERE CONTAINS A DIGIT.  Every figure is interpolated from a
    quantity computed for the artifact being written.
    """
    epoch_key = epoch_tag(epoch_ka)
    names = list(field.names)
    connectivity = field.connectivity or {}

    tags: dict[str, str] = {
        "product": product,
        "product_note": {
            "basin_id": (
                "which basin owns each cell of the cropped window, as decided by "
                "the seeded priority flood of "
                "paleoeurope.viz.paleocostline_render.assign_basins -- the "
                "drainage divide, not a bounding box. It covers DRY ground as "
                "well as water: ownership is a statement about which seed a cell "
                "can be reached from without crossing a higher barrier, and the "
                "flood stops at the ceiling recorded in the flood_ceiling_m tag"
            ),
            "water_level_m": (
                "the water SURFACE at each owned cell, in the vertical frame the "
                "source raster declares. Nodata exactly where no basin owns the "
                "cell. In the corridor mode of record this array is piecewise "
                "constant -- one value per basin -- because the owner decided the "
                "spill corridor is not drawn; it is written as a per-cell array "
                "rather than three scalars because it is the array the engine "
                "computes, because a ramped run writes a varying surface into the "
                "same file, and because clipping a sheet against it needs it "
                "per-cell anyway"
            ),
            "water_state_bits": (
                "a bitfield naming, per cell, which of the engine's three mask "
                "states hold. Read the water_state_bits tag for the bits. This "
                "file exists because basin_id and water_level_m together support "
                "only the FLAT test dem < water_level_m, and the flat test is not "
                "what the engine means by water -- the difference is counted in "
                "the connectivity_* tags of this same file"
            ),
        }[product],
        "epoch_ka": _fmt(epoch_ka, 2),
        "vertical_frame": dem_frame,
        "vertical_frame_note": (
            "copied from the source raster's own vertical_frame tag and NOT "
            "re-derived. The three basin levels are in this frame; the sill "
            "expectation was checked against it string-for-string by "
            "assign_basins, which refuses a frame mismatch rather than "
            "normalising two spellings"
        ),
        "vertical_datum_contract": source_tags.get(
            "vertical_datum_contract",
            "NOT DECLARED BY THE SOURCE RASTER",
        ),
        # ---- the source -------------------------------------------------
        "source_raster": str(source_path),
        "source_raster_resolved": str(Path(source_path).resolve()),
        "source_raster_sha256": source_sha256,
        "source_raster_epoch_tag_as_read": source_tags.get("epoch_ka", "ABSENT"),
        "source_raster_dem_source_as_read": source_tags.get("dem_source", "ABSENT"),
        "source_tag_gia_unknown_ice6g_cells_as_read": source_tags.get(
            "gia_unknown_ice6g_cells", "ABSENT"
        ),
        "source_tag_gia_unknown_ice6g_cells_note": (
            "copied through VERBATIM from the source raster and NOT reconciled "
            "with anything measured here. Its number counts cells of the ICE-6G "
            "model grid; every count in the tags of this file counts RASTER CELLS "
            "of the grid described by crop_window and pixel_arcsec. Two different "
            "units, both spelled out, neither converted into the other"
        ),
        # ---- the crop, D-A ----------------------------------------------
        "crop_rule": CROP_RULE,
        "crop_snap_tol_deg": _fmt(CROP_SNAP_TOL_DEG, 12),
        "crop_north_lat_deg": _fmt(crop["chosen_north_lat_deg"]),
        "crop_north_decision": crop["chosen_north_rule"],
        "crop_measured_minimum_clean_north_lat_deg": _fmt(
            crop["measured_minimum_clean_north_lat_deg"]
        ),
        "crop_measured_minimum_rule": crop["measured_minimum_rule"],
        "crop_extra_latitude_discarded_deg": _fmt(
            crop["extra_latitude_discarded_deg"]
        ),
        "crop_extra_rows_discarded": str(crop["extra_rows_discarded"]),
        "crop_extra_area_discarded_km2": _fmt(crop["extra_area_discarded_km2"], 3),
        "crop_chosen_is_north_of_the_clean_bound": (
            "true" if crop["chosen_is_north_of_the_clean_bound"] else "false"
        ),
        "crop_extra_note": crop["extra_note"],
        "crop_window": _kv(
            {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            }
        ),
        "crop_bounds_wsen": ",".join(_fmt(v) for v in window_bounds),
        "pixel_deg": _fmt(pixel_deg, 12),
        "pixel_arcsec": _fmt(pixel_deg * 3600.0, 6),
        # ---- nodata, declared honestly ----------------------------------
        "source_nodata_cells_in_full_raster": str(
            survey["per_epoch"][epoch_key]["nodata_cells_raster"]
        ),
        "source_nodata_cells_in_crop_window": str(int(nodata_in_crop)),
        "nodata_declaration": (
            {
                "basin_id": (
                    "this raster declares NO nodata value. Every cell of the "
                    "cropped window carries a label, and the label zero means "
                    "'no basin owns this cell', which is an answer and not an "
                    "absence"
                ),
                "water_state_bits": (
                    "this raster declares NO nodata value. Every cell carries a "
                    "state word, and the word zero means 'none of the three "
                    "states hold here', which is an answer and not an absence"
                ),
                "water_level_m": (
                    "nodata is the value in the nodata_value tag, and it marks "
                    "exactly the cells no basin owns -- the count is in the "
                    "partition_unowned tag and it is the same number, asserted at "
                    "write time. There is no cell where the surface is unknown for "
                    "any other reason: the cropped window holds zero source nodata "
                    "cells, also asserted at write time"
                ),
            }[product]
        ),
        "nodata_value": (
            _fmt(field_nodata, 1) if product == "water_level_m" else "none"
        ),
        "nodata_value_source": (
            "INHERITED from the source raster's own declared nodata value, not "
            "chosen here, and checked at write time to stand clear of every water "
            "surface written beside it"
        ),
        "nodata_unit_note": (
            "the two source_nodata_cells_* figures above count RASTER CELLS of "
            "this file's own grid. They are zero inside the window by decision "
            "D-A, and the first is the size of the problem that decision removes"
        ),
        # ---- the partition ----------------------------------------------
        "corridor_mode_requested": CORRIDOR_MODE,
        "corridor_mode_as_run": field.corridor_mode,
        "corridor_mode_note": (
            "TWO VALUES, on purpose. The requested mode is what this export asks "
            "for at every epoch; the as-run mode is what assign_basins reports "
            "having done. They differ whenever the terrain has no spilling pair "
            "at this epoch, because there is then no corridor at all and the "
            "field says so rather than naming a mode it did not exercise. "
            "Recording only the request would claim a corridor was stepped over "
            "where none existed; recording only the result would lose the fact "
            "that the ramp was never on the table"
        ),
        "corridor_mode_decision": (
            "OWNER DECISIONS D-C and D-F, 2026-08-05: three water sheets clipped "
            "by basin_id, the corridor NOT drawn. The alternative mode writes an "
            "interpolated water surface into the corridor, which is exactly the "
            "surface the owner decided not to draw. Cost accepted and stated: "
            "the spill moment is not visible in the asset, the same way it is not "
            "visible in the shipped step raster"
        ),
        "flood_ceiling_m": _fmt(field.ceiling_m),
        "flood_ceiling_rule": (
            "the ownership flood stops at max(basin level) + the margin recorded "
            "in flood_ceiling_margin_m. Cells above it are unowned and are land at "
            "every basin's level, so nothing that could be water is lost; a saddle "
            "standing above the ceiling is reported as not-a-number and never "
            "guessed"
        ),
        "flood_ceiling_margin_m": _fmt(
            field.ceiling_m - max(field.levels_m)
        ),
        "partition_total_cells": str(partition["total"]),
        "partition_unowned": str(partition["unowned"]),
        "partition_by_basin": _kv(partition["per_basin"]),
        "partition_guard": (
            "ASSERTED AT WRITE TIME: the per-basin cell counts plus the unowned "
            "count equal the total cells of the cropped window exactly, and no "
            "cell carries a label outside the declared code map. A raster whose "
            "code map did not describe the array beside it would be worse than no "
            "code map"
        ),
        "water_cells_by_basin": _kv(water["per_basin"]),
        "water_cells_total": str(water["total"]),
        "water_state_bits": _kv({name: value for name, value, _ in STATE_BITS}),
        "water_state_bit_meanings": " | ".join(
            f"{name}={value}: {meaning}" for name, value, meaning in STATE_BITS
        ),
        "unobservable_cells": str(water["unobservable"]),
        "corridor_cells_marked": str(water["corridor_marked"]),
        "corridor_cells_note": (
            "MARKED, not painted and not given a surface. In the corridor mode of "
            "record these cells are the route the spill would take; the water bit "
            "of water_state_bits says which of them are wet under their own "
            "basin's flat level, and nothing here interpolates between two levels"
        ),
        "spilling": "true" if field.spilling else "false",
        "spilling_rule": (
            "OBSERVED FROM THE TERRAIN by assign_basins, not read off a level "
            "curve: true iff at least one pair of seeded fills merges at this "
            "epoch's own levels"
        ),
        "connected_groups": " | ".join(
            "+".join(group) for group in field.connected_groups
        ),
        # ---- what the flat test would have done --------------------------
        "connectivity_rule": connectivity.get("rule", "NOT REPORTED"),
        "connectivity_states": connectivity.get("states", "NOT REPORTED"),
        "connectivity_flat_mask_cells": str(
            connectivity.get("flat_mask_cells", "NOT REPORTED")
        ),
        "connectivity_connected_cells": str(
            connectivity.get("connected_mask_cells", "NOT REPORTED")
        ),
        "connectivity_not_connected_cells": str(
            connectivity.get("not_connected_cells", "NOT REPORTED")
        ),
        "connectivity_closed_cells": str(
            (connectivity.get("closed") or {}).get("cells", "NOT REPORTED")
        ),
        "connectivity_unobservable_cells": str(
            (connectivity.get("unobservable") or {}).get("cells", "NOT REPORTED")
        ),
        "connectivity_note": (
            "the flat figure is what the test dem < water_level_m alone would "
            "have painted, with no connectivity condition. A consumer that clips "
            "a sheet with the flat test instead of the water bit paints the "
            "difference: closed depressions standing below their owner's surface, "
            "plus bodies whose route to a seed leaves the extent. Both counts are "
            "here so the choice is visible rather than accidental"
        ),
        # ---- how it was computed ------------------------------------------
        "dem_array_assigned": (
            "the RAW cell values of the source GeoTIFF window, read at full "
            "resolution with no resampling, no decimation and no smoothing. This "
            "is deliberately NOT the renderer's smoothed mask array: the product "
            "here is data whose companion geometry is the source raster itself, "
            "so the partition must be a partition of THOSE elevations. The cost, "
            "stated rather than hidden: the saddle a smoothed array resolves is a "
            "different number, and this file therefore carries no smoothing shift "
            "-- see the sill_smoothing_shift_m_* tags, which say 'not measured' "
            "rather than zero"
        ),
        "code_path": (
            "scripts/export_basin_fields.py -> "
            "paleoeurope.viz.paleocostline_render.assign_basins; seeds from "
            "scripts/animate_khvalynian_spill.py resolve_seeds over SEED_BOXES; "
            "levels from basin_levels_for; sill expectations from "
            "sill_expectations_for"
        ),
        "code_export_script_sha256": code["export_script"]["sha256"],
        "code_renderer_sha256": code["renderer"]["sha256"],
        "code_registry_sha256": code["registry"]["sha256"],
        "code_renderer_path": code["renderer"]["path"],
        "seed_box_containment_rule": seed_check["rule"],
        "toolchain": _kv(tools),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": _fmt(seconds, 3),
    }

    tags.update(_basin_tag_block(names, levels, seeds))
    tags.update(_sill_tag_block(field, expectations))
    return tags


def _write_raster(
    path: Path,
    array: np.ndarray,
    *,
    dtype: str,
    nodata: Optional[float],
    transform,
    crs,
    tags: dict[str, str],
    predictor: Optional[int] = None,
) -> dict:
    """Write one product, tiled and deflated, and fingerprint what landed."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "zlevel": 9,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if predictor is not None:
        profile["predictor"] = predictor

    with rasterio.open(path, "w", **profile) as out:
        out.write(array, 1)
        out.update_tags(**tags)

    return {
        "path": str(path),
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "nodata": nodata,
        "tags": len(tags),
    }


def export_epoch(
    epoch_ka: float,
    *,
    topo_dir: Path,
    out_root: Path,
    survey: dict,
    crop: dict,
    north_lat: float = CROP_NORTH_LAT_DEG,
    tools: Optional[dict] = None,
    code: Optional[dict] = None,
) -> EpochFields:
    """Resolve, flood, guard and write one epoch.  Never substitutes anything.

    A content problem — no Pontic row, a dry seed, a saddle that disagrees with
    the measured crest, nodata surviving the crop — comes back as ``refusal`` with
    the original exception's class and message, and NO raster is written for that
    epoch.  The epoch keeps its slot in the manifest.
    """
    import rasterio

    started = time.time()
    result = EpochFields(epoch_ka=float(epoch_ka))
    tools = tools or toolchain()
    code = code or code_provenance()

    try:
        source = raster_for(epoch_ka, topo_dir)
        source_sha = sha256_of(source)
        dem_frame = declared_vertical_frame({"khv": str(source)})
        if not dem_frame:
            raise CropBoundError(
                f"{source.name} declares no vertical_frame tag. A raster that "
                "does not say what its elevations are above cannot have a sill "
                "expectation checked against it, and the levels written beside "
                "it would name no frame either."
            )

        dz_bosphorus = dz_at_bosphorus(epoch_ka)
        levels = basin_levels_for(epoch_ka)
        seeds = resolve_seeds(
            source,
            {name: level.level_m for name, level in levels.items()},
            epoch_ka=epoch_ka,
            raster_sha256=source_sha,
        )

        with rasterio.open(source) as dataset:
            window = crop_window(dataset, north_lat)
            transform = dataset.window_transform(window)
            bounds = rasterio.windows.bounds(window, dataset.transform)
            pixel_deg = abs(float(dataset.transform.e))
            crs = dataset.crs
            source_tags = dict(dataset.tags())
            nodata_value = dataset.nodata
            block = dataset.read(1, window=window).astype("float64")

        seed_check = assert_seed_boxes_inside(bounds)

        field_nodata = field_nodata_for(
            nodata_value, [level.level_m for level in levels.values()]
        )

        bad = ~np.isfinite(block)
        if nodata_value is not None:
            bad |= block == nodata_value
        nodata_in_crop = int(bad.sum())
        # Whatever the sentinel was, it is not an elevation.  Converted before the
        # array reaches the flood so that "no data" cannot pass for very low
        # ground; the refusal below means this can only ever be a no-op, and it is
        # done anyway because a no-op that is asserted is cheaper than one that is
        # argued.
        block[bad] = np.nan
        if nodata_in_crop:
            rows = np.flatnonzero(bad.any(axis=1))
            raise NodataInCropError(
                f"{source.name} holds {nodata_in_crop} nodata raster cell(s) "
                f"inside the cropped window, first at window row {int(rows[0])} "
                f"and last at {int(rows[-1])}. Decision D-A rests on the "
                "measurement that no epoch does; a failed assertion means the "
                "measurement was wrong and the export stops rather than shipping "
                "a product whose nodata story is untrue."
            )

        # Whole-cell identity: the crop kept source cells, so each seed's own
        # source row/col must map onto the window by a pure row shift.  Checked,
        # because it is what licenses resolving seeds on the uncropped file.
        for seed in seeds:
            expected_row = seed.row - int(window.row_off)
            col_f, row_f = ~transform * (seed.lon, seed.lat)
            if int(np.floor(row_f)) != expected_row or int(np.floor(col_f)) != seed.col:
                raise SeedBoxOutsideCropError(
                    f"seed {seed.basin!r} resolved to source cell "
                    f"row={seed.row} col={seed.col} but lands on window cell "
                    f"row={int(np.floor(row_f))} col={int(np.floor(col_f))} "
                    f"against the expected row={expected_row} col={seed.col}. The "
                    "crop did not preserve the cell grid, so a seed measured on "
                    "the source file is not the cell the flood would start from."
                )

        basin_seeds = [
            BasinSeed(
                name=seed.basin, lon=seed.lon, lat=seed.lat, level_m=seed.level_m
            )
            for seed in seeds
        ]
        expectations = sill_expectations_for(epoch_ka, dem_frame, dz_bosphorus)

        field = assign_basins(
            block,
            transform,
            basin_seeds,
            corridor=CORRIDOR_MODE,
            sill_expectations=expectations,
            dem_frame=dem_frame,
            nodata=bad,
            unobservable_policy="mark",
        )

        total_cells = int(block.size)
        partition = _partition_counts(field, total_cells)
        water = _water_counts(field)

        seconds = time.time() - started

        def tags_for(product: str) -> dict[str, str]:
            return build_tags(
                product=product,
                epoch_ka=epoch_ka,
                source_path=source,
                source_sha256=source_sha,
                source_tags=source_tags,
                dem_frame=dem_frame,
                window=window,
                window_bounds=bounds,
                pixel_deg=pixel_deg,
                survey=survey,
                crop=crop,
                seed_check=seed_check,
                field=field,
                levels=levels,
                seeds=seeds,
                expectations=expectations,
                partition=partition,
                water=water,
                nodata_in_crop=nodata_in_crop,
                field_nodata=field_nodata,
                tools=tools,
                code=code,
                seconds=seconds,
            )

        key = epoch_tag(epoch_ka)
        written = {}

        highest = int(field.basin_id.max()) if field.basin_id.size else 0
        if highest > np.iinfo("uint8").max or int(field.basin_id.min()) < 0:
            raise PartitionError(
                f"the basin labels run to {highest} and down to "
                f"{int(field.basin_id.min())}, which does not fit the unsigned "
                "byte the code map is written in. A label that wrapped on the "
                "narrowing would make the code map a false description of the "
                "array beside it."
            )
        written["basin_id"] = _write_raster(
            out_root / "basin_id" / f"basin_id_{key}ka.tif",
            field.basin_id.astype("uint8"),
            dtype="uint8",
            nodata=None,
            transform=transform,
            crs=crs,
            tags=tags_for("basin_id"),
        )

        surface = np.where(
            np.isfinite(field.water_level_m), field.water_level_m, field_nodata
        ).astype("float32")
        unowned_written = int((surface == np.float32(field_nodata)).sum())
        if unowned_written != partition["unowned"]:
            raise PartitionError(
                f"{unowned_written} cell(s) carry the water_level nodata value "
                f"but {partition['unowned']} cell(s) are unowned. The nodata "
                "declaration on that raster would be false: either an owned cell "
                "has no finite surface, or an owned cell's surface happens to "
                "equal the sentinel."
            )
        written["water_level"] = _write_raster(
            out_root / "water_level" / f"water_level_{key}ka.tif",
            surface,
            dtype="float32",
            nodata=field_nodata,
            transform=transform,
            crs=crs,
            tags=tags_for("water_level_m"),
            predictor=3,
        )

        state = np.zeros(field.basin_id.shape, dtype="uint8")
        masks = {
            "water": field.water_mask,
            "unobservable": (
                field.unobservable_mask
                if field.unobservable_mask.shape == state.shape
                else np.zeros(state.shape, dtype=bool)
            ),
            "corridor": field.corridor_mask,
        }
        for name, value, _ in STATE_BITS:
            state[masks[name]] |= np.uint8(value)
        written["water_state"] = _write_raster(
            out_root / "water_state" / f"water_state_{key}ka.tif",
            state,
            dtype="uint8",
            nodata=None,
            transform=transform,
            crs=crs,
            tags=tags_for("water_state_bits"),
        )

        result.record = {
            "epoch_ka": float(epoch_ka),
            "source_raster": str(source),
            "source_raster_sha256": source_sha,
            "vertical_frame": dem_frame,
            "crop_window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
            "crop_bounds_wsen": [float(v) for v in bounds],
            "source_nodata_cells_in_crop_window": nodata_in_crop,
            "water_level_nodata_value": float(field_nodata),
            "levels": {n: v.to_dict() for n, v in levels.items()},
            "seeds": [s.to_dict() for s in seeds],
            "sill": {
                f"{a}__{b}": {
                    "measured_saddle_m": (
                        None if not np.isfinite(field.sill_m[(a, b)])
                        else float(field.sill_m[(a, b)])
                    ),
                    "expected_m": field.sill_expected_m.get((a, b)),
                    "lonlat": field.sill_lonlat.get((a, b)),
                    "status": field.sill_expectation_status.get((a, b)),
                }
                for (a, b) in sorted(field.sill_m)
            },
            "spilling": bool(field.spilling),
            "connected_groups": [list(g) for g in field.connected_groups],
            "corridor_mode_requested": CORRIDOR_MODE,
            "corridor_mode_as_run": field.corridor_mode,
            "flood_ceiling_m": float(field.ceiling_m),
            "partition": partition,
            "water": water,
            "connectivity": field.connectivity,
            "written": written,
            "wall_seconds": seconds,
        }
        result.seconds = seconds

    except Exception as exc:  # noqa: BLE001 - the refusal IS this epoch's content
        result.seconds = time.time() - started
        result.refusal = f"{type(exc).__name__}: {exc}"
        result.record = {
            "epoch_ka": float(epoch_ka),
            "refusal": result.refusal,
            "traceback": traceback.format_exc(limit=8),
            "wall_seconds": result.seconds,
        }
    return result


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """numpy scalars written as their Python equivalents, nothing dropped.

    ``json`` refuses a ``np.int64`` and the connectivity block is full of them.
    Silently stringifying an unknown object would turn a number into prose, so
    anything not a numpy scalar or array raises here instead.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"{type(value).__name__} is not JSON-serialisable and is not silently "
        f"stringified here: {value!r}"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def _log(message: str, log_path: Optional[Path] = None) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class StaleSurveyError(ValueError):
    """A cached survey does not describe the run that wants to reuse it."""


def load_survey(path: Path, epochs: Sequence[float], topo_dir: Path) -> dict:
    """A previously measured survey, reused only if it covers THIS run.

    Re-measuring the nodata of forty-three full-resolution rasters costs many
    minutes, so a restart after a crash should not have to pay for it twice.  But
    a survey is the measurement decision D-A rests on, and reusing one that was
    taken over a different tree or a different epoch set would put a stale number
    on every raster written.  So the reuse is opt-in and every claim it makes is
    checked against the run asking for it.
    """
    if not path.exists():
        raise StaleSurveyError(
            f"no survey to reuse at {path}. Run without the reuse flag to measure "
            "one; nothing is assumed about a file that is not there."
        )
    survey = json.loads(path.read_text(encoding="utf-8"))
    cached_tree = Path(survey.get("topo_dir", ""))
    if cached_tree.resolve() != Path(topo_dir).resolve():
        raise StaleSurveyError(
            f"the survey at {path} was measured over {cached_tree} but this run "
            f"reads {topo_dir}. Two trees are two scenes and this project re-bakes "
            "rasters in place under unchanged names."
        )
    missing = [
        epoch_tag(e) for e in epochs if epoch_tag(e) not in survey.get("per_epoch", {})
    ]
    if missing:
        raise StaleSurveyError(
            f"the survey at {path} covers "
            f"{len(survey.get('per_epoch', {}))} epoch(s) and does not include "
            f"{missing!r}. A minimum clean bound measured without an epoch says "
            "nothing about that epoch's nodata."
        )
    return survey


def assert_free_space(root: Path, needed: int = MIN_FREE_BYTES) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free < needed:
        raise OSError(
            f"{root} has {usage.free} free byte(s), under the "
            f"{needed} required before a run starts. A full disk on this project "
            "has already truncated one file mid-write; the run stops here instead."
        )
    return {"root": str(root), "free_bytes": int(usage.free), "required": int(needed)}


def export_all(
    epochs: Optional[Sequence[float]] = None,
    *,
    topo_dir: Optional[Path | str] = None,
    out_root: Optional[Path | str] = None,
    north_lat: float = CROP_NORTH_LAT_DEG,
    survey: Optional[dict] = None,
    survey_only: bool = False,
    reuse_survey: bool = False,
) -> dict:
    """Survey, then export every epoch.  Returns the manifest it writes."""
    epochs = tuple(_epochs()) if epochs is None else tuple(float(e) for e in epochs)
    topo_dir = Path(topo_dir) if topo_dir is not None else _topo_dir()
    out_root = Path(out_root) if out_root is not None else OUT_ROOT
    log_path = out_root / "export_log.txt"
    started = time.time()

    space = assert_free_space(out_root)
    _log(
        f"out_root {out_root} has {space['free_bytes']} free byte(s)", log_path
    )
    _log(f"topo_dir {topo_dir}; {len(epochs)} epoch(s) requested", log_path)

    survey_path = out_root / "nodata_survey.json"
    if survey is None and reuse_survey:
        survey = load_survey(survey_path, epochs, topo_dir)
        _log(
            f"reusing the survey at {survey_path}, measured "
            f"{survey['measured_utc']}, after checking it covers this run's "
            "epochs and tree",
            log_path,
        )
    if survey is None:
        _log("surveying nodata over every epoch (no early exit)", log_path)
        survey = survey_nodata(
            epochs, topo_dir, progress=lambda m: _log(m, log_path)
        )
        _write_json(survey_path, survey)
    crop = crop_record(survey, north_lat)
    _log(
        "measured minimum clean north bound "
        f"{_fmt(crop['measured_minimum_clean_north_lat_deg'])} deg; chosen "
        f"{_fmt(crop['chosen_north_lat_deg'])} deg",
        log_path,
    )
    if crop["chosen_is_north_of_the_clean_bound"]:
        _log(
            "WARNING: the chosen bound lies NORTH of the clean bound by "
            f"{_fmt(abs(crop['extra_latitude_discarded_deg']))} deg / "
            f"{abs(crop['extra_rows_discarded'])} row(s), so the window still "
            "contains nodata rows and every epoch holding one will refuse at "
            "write time. The bound is not moved to fit",
            log_path,
        )
    else:
        _log(
            "the choice discards "
            f"{_fmt(crop['extra_latitude_discarded_deg'])} deg / "
            f"{crop['extra_rows_discarded']} row(s) / "
            f"{_fmt(crop['extra_area_discarded_km2'], 1)} km2 MORE than the "
            "measurement required",
            log_path,
        )

    manifest: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_root": str(out_root),
        "topo_dir": str(topo_dir),
        "epochs_requested": [float(e) for e in epochs],
        "corridor_mode": CORRIDOR_MODE,
        "crop": crop,
        "nodata_survey": survey,
        "disk": space,
        "toolchain": toolchain(),
        "code": code_provenance(),
        "state_bits": [
            {"name": n, "value": v, "meaning": m} for n, v, m in STATE_BITS
        ],
        "epochs": [],
    }

    if survey_only:
        manifest["survey_only"] = True
        _write_json(out_root / "export_manifest.json", manifest)
        return manifest

    tools = manifest["toolchain"]
    code = manifest["code"]
    written = 0
    refused = 0
    for index, epoch in enumerate(epochs, start=1):
        _log(f"epoch {epoch_tag(epoch)} ka ({index}/{len(epochs)}) starting", log_path)
        result = export_epoch(
            epoch,
            topo_dir=topo_dir,
            out_root=out_root,
            survey=survey,
            crop=crop,
            north_lat=north_lat,
            tools=tools,
            code=code,
        )
        manifest["epochs"].append(result.record)
        if result.written:
            written += 1
            _log(
                f"epoch {epoch_tag(epoch)} ka WRITTEN in "
                f"{_fmt(result.seconds, 1)} s; owned "
                f"{_kv(result.record['partition']['per_basin'])}; water "
                f"{result.record['water']['total']} cell(s); spilling "
                f"{result.record['spilling']}",
                log_path,
            )
        else:
            refused += 1
            _log(
                f"epoch {epoch_tag(epoch)} ka REFUSED after "
                f"{_fmt(result.seconds, 1)} s: {result.refusal}",
                log_path,
            )
        # The manifest is rewritten after every epoch, not only at the end: a run
        # this long that dies at epoch forty would otherwise have produced forty
        # rasters and no record of what any of them are.
        _write_json(out_root / "export_manifest.json", manifest)

    manifest["summary"] = {
        "epochs_written": written,
        "epochs_refused": refused,
        "total_wall_seconds": time.time() - started,
        "refusals": [
            {"epoch_ka": e["epoch_ka"], "refusal": e["refusal"]}
            for e in manifest["epochs"]
            if e.get("refusal")
        ],
    }
    _write_json(out_root / "export_manifest.json", manifest)
    _log(
        f"done: {written} written, {refused} refused, "
        f"{_fmt(manifest['summary']['total_wall_seconds'], 1)} s total",
        log_path,
    )
    return manifest


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survey", action="store_true",
                        help="measure the nodata and stop; write no raster")
    parser.add_argument("--epochs", nargs="*", type=float, default=None,
                        help="epochs in ka; default is every epoch of the bake grid")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--topo-dir", default=None)
    parser.add_argument("--north-lat", type=float, default=CROP_NORTH_LAT_DEG)
    parser.add_argument(
        "--reuse-survey", action="store_true",
        help="reuse nodata_survey.json from the output root instead of "
             "re-measuring, after checking it covers this run's epochs and tree",
    )
    args = parser.parse_args(argv)

    manifest = export_all(
        args.epochs,
        topo_dir=args.topo_dir,
        out_root=args.out_root,
        north_lat=args.north_lat,
        survey_only=args.survey,
        reuse_survey=args.reuse_survey,
    )
    summary = manifest.get("summary")
    if summary is None:
        return 0
    return 0 if summary["epochs_written"] else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
