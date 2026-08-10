"""
paleoeurope.viz.paleocostline_render
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composite per-epoch paleo-topography render.

Pipeline
--------
1. Build a merged DEM mosaic from per-tile GeoTIFFs (down-sampled to
   ``RenderConfig.max_px`` pixels on the longer axis).
2. Load ``Paleocoastlines.shp`` once; filter polygons to the sea-level step
   nearest to the Spratt & Lisiecki (2016) value for the requested epoch.
3. Rasterize land polygons onto the DEM grid to obtain a land/sea binary mask
   (falls back to DEM zero-contour if the shapefile is unavailable or covers
   less than 1 % of the scene).
4. LAND pixels  : hillshade × terrain colourmap by absolute elevation.
5. OCEAN pixels : blue gradient by bathymetric depth (square-root stretch).
6. ICE pixels   : solid white override on top of everything (ICE-7G H_ice > 10 m).
7. Export PNG and return the output ``Path``.

References
----------
Spratt, R. M. & Lisiecki, L. E. (2016). A Late Pleistocene sea-level stack.
    *Climate of the Past*, 12, 1079–1092. https://doi.org/10.5194/cp-12-1079-2016

Paleocoastlines GIS dataset — see docs/data_catalogue.md for citation and DOI.

Examples
--------
>>> from paleoeurope.viz import render_paleocostline_epoch, RenderConfig
>>> cfg = RenderConfig(max_px=1024, dpi=100)
>>> out = render_paleocostline_epoch(
...     epoch_ka=21,
...     tile_paths={"/path/to/N62E006_t21.tif": "/path/to/N62E006_t21.tif"},
...     sea_level_csv=Path("data/sea_level_curves/spratt_lisiecki_2016_simplified.csv"),
...     paleocoastlines_shp=Path("data/paleocostline/Paleocoastlines/Paleocoastlines/Paleocoastlines.shp"),
...     output_dir=Path("outputs"),
...     config=cfg,
... )
>>> print(out)   # PosixPath('outputs/paleo_render_21ka.png')
"""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
import warnings
from array import array
from collections import deque
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Optional, Sequence, Union

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling as RIOResampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.merge import merge as rasterio_merge
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    gaussian_filter1d,
)
from scipy.ndimage import (
    label as ndi_label,
)
from scipy.ndimage import (
    maximum as ndi_maximum,
)
from scipy.ndimage import (
    minimum as ndi_minimum,
)
from scipy.ndimage import (
    minimum_position as ndi_minimum_position,
)
from shapely.geometry import box as shp_box
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# The 2-metre budget, derived once
# ---------------------------------------------------------------------------

#: How far a sill measured on the render grid may sit from an independently
#: measured one before the frame refuses; and, separately, how far a seam blend
#: may move the terrain before it has to say so.
#:
#: **Derivation — a spill-window duration converted to metres, not a tolerance
#: copied from another tolerance.**  ``scripts/bake_khvalynian.py`` records the
#: conversion factor beside ``MANYCH_SILL_PATH_M``, and the Caspian curve's own
#: header states the same thing: moving the Manych crest from the +23.00 m modern
#: sill to the +25.00 m continuous-path crest — a shift of **2.00 m** — "closes the
#: spill window by ~0.3 ka".
#:
#: **What that buys, stated plainly rather than rounded in our favour.**  The
#: Khvalynian sequence is on a **0.25 ka** epoch grid and ``0.3 ka > 0.25 ka``, so
#: 2.00 m of sill error **CAN move the counted spill window by one frame**.  It
#: cannot move it by two: that would need ``0.5 ka``, i.e. ~3.3 m.  So the honest
#: statement of this bound is *"at most one frame at each end of the window"* —
#: **not** "an error that cannot change how many frames a viewer counts as
#: spilling", which is what this comment used to claim and which its own
#: arithmetic contradicts.  An earlier draft carried both sentences at once; the
#: claim was wrong and the arithmetic was right.
#:
#: The alternative was to re-derive the constant down to ``2.00 × 0.25/0.3 =
#: 1.67 m`` so that the stronger claim would hold.  That was **not** taken: the
#: 0.3 ka figure is itself a one-significant-figure statement in the curve header,
#: so 1.67 m would be a false precision derived from a rounded input, and the
#: phase brief's own decision (§4, "Tolerance") is to keep 2.0 m and re-derive it
#: in writing.  This is that writing.
#:
#: What this replaces: ``RenderConfig.seam_blend_max_shift_m`` used to justify
#: itself by citing :func:`assign_basins`' ``sill_tolerance_m``, whose docstring
#: cited it back, and :func:`build_epoch_mosaic` repeated the loop a third time.
#: Three mutually-supporting citations and no physical quantity anywhere in the
#: cycle.  Both call sites now default to this constant and both point here.
#:
#: It is a bound, not a law.  Widening it is allowed and must be done in writing:
#: state the epoch spacing and the ka the new figure buys, the way this comment
#: does.  It governs **one** guard — the sill comparison.  It used to govern the
#: smoothing disclosure as well; see :data:`SMOOTHING_SHIFT_DISCLOSURE_M`.
SILL_TOLERANCE_M = 2.0

#: How far the σ=1.5 mask smoothing may move a measured sill before the frame has
#: to say so (:class:`SmoothingShiftWarning` + the sidecar field).
#:
#: **Why it is a separate constant from :data:`SILL_TOLERANCE_M`, which it happens
#: to equal.**  Until 2026-08-03 the smoothing disclosure was wired straight to
#: ``basin_sill_tolerance_m``, so a caller who widened the *sill* tolerance — a
#: legitimate, in-writing decision about how far the render grid may disagree with
#: an independent measurement — silently also raised the bar on a *different*
#: guard, the one that reports what this module's own Gaussian did to the crest.
#: One knob was silencing two guards, and the second one was the one the caller
#: had said nothing about.  ``tests/test_viz.py`` pins the decoupling.
#:
#: **Derivation.**  Same physical yardstick as :data:`SILL_TOLERANCE_M` and the
#: same arithmetic: 2.00 m of crest movement is ~0.3 ka of spill window on a
#: 0.25 ka grid, i.e. up to one frame.  A smoothing shift that can move the window
#: by a frame is a shift a reader has to be told about.  It is numerically equal to
#: the sill tolerance today **by agreement, not by reference**: change either one
#: without changing the other and nothing here breaks.
SMOOTHING_SHIFT_DISCLOSURE_M = 2.0


# ---------------------------------------------------------------------------
# One colour mapping for a whole sequence
# ---------------------------------------------------------------------------

#: Sun azimuth and elevation for the hillshade, named once.
#:
#: They used to be two literals inside :func:`render_paleocostline_epoch`.  They are
#: constants here because :data:`ANALYTIC_SHADE_RANGE` is derived from ``ALTDEG``
#: and a derivation whose input is a literal buried in a function body cannot be
#: checked.  Changing either value is a rendering change; changing ``ALTDEG``
#: additionally moves the analytic shade floor and therefore every pinned frame.
HILLSHADE_AZDEG = 315.0
HILLSHADE_ALTDEG = 45.0


def analytic_shade_range(altdeg: float = HILLSHADE_ALTDEG) -> tuple[float, float]:
    """The range :func:`matplotlib.colors.LightSource.hillshade` normalises AWAY.

    ``LightSource.shade_normals`` computes ``intensity = n · d`` for the unit
    surface normal ``n`` and the unit light direction ``d``, and then rescales it by
    **that frame's own** ``intensity.min()`` and ``intensity.max()`` — read the
    matplotlib source, it is four lines with a comment admitting the rescale is
    "strictly speaking … incorrect".  Two cells out of the whole array therefore set
    the brightness of every other cell, so identical terrain rendered beside a frame
    with one steeper cell comes out at a different grey.  That is the flicker term
    nobody looks for, because it is not in this repository's code at all.

    The pinned replacement needs no data and no second pass, because the range of
    ``n · d`` is fixed by geometry:

    * ``max = +1`` — attained when the surface normal points straight at the sun.
    * ``min = −cos(altdeg)`` — the infimum over normals with ``n_z > 0``, i.e. a
      vertical face turned directly away.  A DEM normal always has ``n_z > 0``
      (:func:`numpy.gradient` cannot produce an overhang), so nothing can go below
      it.

    **What pinning costs, measured rather than asserted** (``khv_16.5ka.tif`` and
    ``khv_16ka.tif``, ``vert_exag=8``, azimuth 315°, altitude 45°):

    ==============  =====================  ======================  ==============
    grid            observed ``n · d``     analytic range          contrast kept
    ==============  =====================  ======================  ==============
    ``max_px=2000``  −0.333346 … +0.999997  −0.707107 … +1.000000  78.1 %
    ``max_px=7800``  −0.577424 … +1.000000  −0.707107 … +1.000000  92.4 %
    ==============  =====================  ======================  ==============

    At native — the only resolution the two-basin frame is renderable at — the top
    of the range is *already* the analytic maximum to six decimals and the whole
    cost is 7.6 % of shading contrast, paid once, on every frame equally.

    **Honesty about what was and was not observed to flicker.**  Across those two
    adjacent epochs the observed floor agrees to 1e-6 at 2000 px and to every
    printed digit at native, so on THIS pair the hillshade is not the term that
    shimmers — the land percentiles are (see :class:`SequenceLimits`).  That is a
    measurement of two frames, not a property of the mechanism: the floor is set by
    one cell out of 33 696 000, and the ice-dome blend rewrites the array it is
    measured on wherever ``ice_mask`` changes, so it is unpinned rather than stable.

    Returns ``(min, max)``.
    """
    return (-float(np.cos(np.radians(float(altdeg)))), 1.0)


#: The colour the sill-route polyline and its crest marker are drawn in.  Named so
#: the sentences that mention it interpolate it instead of spelling it out.
SILL_ROUTE_COLOUR = "#ff8c1a"

#: Painted where a cell's value lies ABOVE everything the limits pass measured.
OUT_OF_RANGE_HIGH_RGB = (1.0, 0.0, 1.0)

#: Painted where it lies BELOW everything the limits pass measured.
OUT_OF_RANGE_LOW_RGB = (1.0, 0.0, 0.0)

#: Why those two, in numbers rather than in taste.
#:
#: Both were chosen by minimising confusability against **everything this module
#: draws**: ``terrain`` over its used sub-range [0.25, 0.82], ``Blues`` over
#: [0.30, 0.90], ``Blues_r`` over [0.30, 0.92], each multiplied by every hillshade
#: factor the compositor can apply (land ``0.28 … 1.00``, ice ``0.45 … 1.00`` —
#: multiplication only darkens, so a darkened ramp colour equals a reserved colour
#: only if the ramp colour already did), plus the ``#1a1a1a`` background, the
#: ``#ff8c1a`` sill route, and the three legend swatches.  Minimum Euclidean RGB
#: distance, measured 2026-08-03 and pinned by ``tests/test_viz.py``:
#:
#: * magenta ``(1, 0, 1)`` — **0.808**, nearest neighbour the ``terrain`` ramp.
#: * red ``(1, 0, 0)`` — **0.558**, nearest neighbour the ``#ff8c1a`` sill-route
#:   polyline.  That is the weakest separation in the picture and it is stated
#:   rather than left to be discovered; the route is a 1.1 pt line and the reserved
#:   colour is a filled cell, so the two are told apart by shape as well as hue.
#:
#: A single reserved colour would have scored 0.808 on its own.  Two were kept
#: because "deeper than anything the limits saw" and "higher than anything the
#: limits saw" are opposite defects, and a reader must not have to open a JSON file
#: to tell which one the picture is showing.
#: The separations quoted by :data:`OUT_OF_RANGE_COLOUR_NOTE`, named so the sentence
#: and the figure are one object rather than two.  Unlike a cell count these are
#: properties of the module's own palette and not of any frame — but they are still
#: interpolated rather than typed into the prose, because a numeral typed into a
#: sentence is a numeral that can drift from the thing it describes.
OUT_OF_RANGE_MIN_RGB_DISTANCE = {"high": 0.808, "low": 0.558}

OUT_OF_RANGE_COLOUR_NOTE = (
    f"magenta {OUT_OF_RANGE_HIGH_RGB} = the cell's value is ABOVE every value the "
    f"limits pass measured over the sequence; red {OUT_OF_RANGE_LOW_RGB} = BELOW "
    "it. These are NOT the ends of the colour ramp — saturating at the ramp end is "
    "the intended behaviour of a percentile ramp and is counted separately under "
    "saturated_at_ramp_end. A reserved-colour cell means the pinned limits were "
    "computed over data that does not contain this frame, so on a correct run these "
    "counts are zero. Minimum RGB distance to anything else this module draws: "
    f"{OUT_OF_RANGE_MIN_RGB_DISTANCE['high']} for magenta (nearest: the terrain "
    f"ramp), {OUT_OF_RANGE_MIN_RGB_DISTANCE['low']} for red (nearest: the "
    f"{SILL_ROUTE_COLOUR} sill-route polyline)."
)


#: The two tones a body of UNOBSERVABLE connectivity is drawn in, and the period of
#: the stripe between them, in cells.
#:
#: Chosen to be readable as "unresolved" from the PNG alone, without a legend and
#: without the sidecar.  Three properties, each doing work:
#:
#: * a desaturated violet-grey that neither ``Blues`` (water), ``terrain`` (land) nor
#:   ``Blues_r`` (ice) produces anywhere in the sub-ranges this module uses, so it is
#:   not a shade of anything else in the picture;
#: * **no hillshade multiplication.**  Every other surface here is multiplied by the
#:   shade, so an unshaded patch reads as flat and synthetic beside them — which is
#:   what it is: an absence of information, not a terrain;
#: * a regular anti-diagonal stripe.  No DEM produces one, so the eye classifies the
#:   region as an annotation rather than as ground or sea.
#:
#: Deliberately **not** a colour on the depth ramp and **not** the land ramp's dry
#: end: those are the two claims the frame must not make about these cells.
UNOBSERVABLE_RGB = (0.55, 0.50, 0.62)
UNOBSERVABLE_STRIPE_RGB = (0.38, 0.34, 0.45)

#: Stripe period in OUTPUT pixels, i.e. in the saved PNG rather than in the raster.
#:
#: It has to be stated in output pixels because the figure is resampled: the native
#: frame is 7 800 cells wide and is drawn into roughly 1 970, so a period fixed in
#: raster cells is averaged away by the bilinear ``imshow`` and the region arrives at
#: the reader as a FLAT unusual colour.  Measured on the native 16.5 ka frame with an
#: 8-cell period: 264 pixels of the base tone survive and ZERO of the stripe tone —
#: i.e. the half that carries the "this is an annotation, not a terrain" signal was
#: not in the shipped image at all.  The period is therefore scaled by the frame's own
#: downsampling factor at draw time, and the value actually used is reported in
#: ``unobservable_drawn.stripe_period_px`` rather than assumed from this constant.
UNOBSERVABLE_STRIPE_PERIOD_OUTPUT_PX = 8

#: The sentence the figure carries about them.  Filled per frame; no digits here.
UNOBSERVABLE_CAPTION_TEMPLATE = (
    "{components} water bodies ({cells} cells / {area} sq km) reach the "
    "{edges} edge of this raster below a declared level: hatched violet, drawn as "
    "NEITHER water NOR land because this extent cannot see where they connect "
    "(connectivity.unobservable in the sidecar)"
)


def unobservable_stripe_period_px(raster_width: int, cfg: "RenderConfig") -> int:
    """Stripe period in RASTER cells that yields :data:`UNOBSERVABLE_STRIPE_PERIOD_OUTPUT_PX`
    in the saved image.

    ``fig_size_in * dpi`` is the figure's pixel width; ``bbox_inches="tight"`` trims
    it, so this is an estimate of the axes width and the period it returns is an
    estimate too.  Erring toward a WIDER stripe is deliberate: a stripe that is too
    coarse is still legible as a stripe, and one that is too fine averages to a flat
    colour, which is the failure being corrected.
    """
    displayed = max(float(cfg.fig_size_in) * float(cfg.dpi), 1.0)
    downsample = max(float(raster_width) / displayed, 1.0)
    return int(max(2, round(UNOBSERVABLE_STRIPE_PERIOD_OUTPUT_PX * downsample)))


def _unobservable_rgb(mask: np.ndarray, period_px: int) -> np.ndarray:
    """The striped fill for *mask*, as one RGB value per True cell.

    Built on the cell indices rather than on any elevation, because the whole point
    is that no measurement of this region is being displayed.
    """
    rows, cols = np.nonzero(mask)
    period = max(int(period_px), 2)
    stripe = ((rows + cols) % period) < (period // 2)
    out = np.empty((rows.size, 3), dtype=np.float32)
    out[:] = UNOBSERVABLE_RGB
    out[stripe] = UNOBSERVABLE_STRIPE_RGB
    return out


#: Narrowest colour ramp :class:`SequenceLimits` will carry, in metres.
#:
#: :func:`render_paleocostline_epoch` divides the land ramp by ``max(span, 1.0)``.
#: That clamp is on the byte-stable single-basin path and is not being removed, so a
#: pinned ramp narrower than 1 m would be silently widened to 1 m and the picture
#: would not be on the ramp its own sidecar reports.  The object refuses such a span
#: instead — a limit the compositor rewrites is not a limit.
SEQUENCE_RAMP_MIN_SPAN_M = 1.0


class SequenceLimitsEpochError(ValueError):
    """The frame being rendered is not one of the epochs the limits were computed over.

    Pinning a sequence's colour mapping is a claim that every frame in the sequence
    was seen by the pass that produced the numbers.  A caller that renders 13.75 ka
    against limits computed over 17.00 … 16.00 ka is asserting that claim about a
    frame nobody measured, and the picture that comes out is indistinguishable from
    one where it was true.
    """


class SequenceLimitsRasterError(ValueError):
    """The raster being rendered is not the raster the limits were computed from.

    Either its path is absent from :attr:`SequenceLimits.rasters`, or it is present
    and its sha256 no longer matches.  The second case is the one worth having: this
    project re-bakes rasters in place under the same file name (the Pontic W0
    re-bake did exactly that), so a name match is not an identity.
    """


class SequenceLimitsRecipeError(ValueError):
    """The limits were computed under a different recipe than the render will use.

    The percentiles, the vertical exaggeration and the sun position are inputs to
    the numbers, so consuming limits under a config that disagrees with them means
    the sidecar records a recipe the picture was not made with.  A ``RenderConfig``
    field that is silently ignored is the same defect class this module keeps
    deleting elsewhere.
    """


class SequenceLimitsExceededError(ValueError):
    """``clip_policy="raise"`` and the frame holds cells outside the pinned limits.

    Raised **before** the PNG is written, so a sequence cannot ship a frame that
    misrepresents its epoch and then disclose it in a file nobody opens.
    """


@dataclass(frozen=True)
class RasterFingerprint:
    """One input raster, identified by content rather than by name.

    Parameters
    ----------
    path : str
        Resolved absolute path, as a string so it survives JSON.
    sha256 : str
        64 lowercase hex characters of the file's bytes.  There is no opt-out and no
        "not hashed" sentinel: the hash is what makes :class:`SequenceLimits` a
        provenanced object rather than seven floats, and it costs 1.18 s for the
        42 MB production raster against a ~340 s native render.
    size_bytes : int
    epoch_ka : float
        Which epoch of the sequence this raster was read for.
    """

    path: str
    sha256: str
    size_bytes: int
    epoch_ka: float

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError(f"RasterFingerprint.path is {self.path!r}")
        digest = self.sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(
                f"RasterFingerprint.sha256 is {digest!r}. It must be 64 lowercase "
                "hex characters of the file's bytes. There is deliberately no "
                "'unknown' sentinel — a fingerprint that cannot be checked is a "
                "path, and this project already has one of those."
            )
        if int(self.size_bytes) <= 0:
            raise ValueError(
                f"RasterFingerprint.size_bytes is {self.size_bytes!r}; an empty file "
                "cannot have produced a statistic."
            )
        if not np.isfinite(self.epoch_ka):
            raise ValueError(f"RasterFingerprint.epoch_ka is {self.epoch_ka!r}")

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "epoch_ka": float(self.epoch_ka),
        }


@dataclass(frozen=True)
class SequenceEpoch:
    """One frame of a sequence, as the limits pass needs to see it.

    Parameters
    ----------
    epoch_ka : float
        The epoch, spelled exactly as :func:`render_paleocostline_epoch` will be
        called with it.  The membership check is exact float equality — every epoch
        on this project's 0.25 ka grid is a dyadic rational and therefore exact in
        binary, so a tolerance would only hide a typo.
    raster_paths : tuple[str, ...]
        Resolved absolute paths of the DEM GeoTIFFs this frame is built from.  Ice
        side-cars (``<stem>_ice.tif``) are discovered from them, exactly as
        :func:`build_epoch_mosaic` discovers them.
    water_level_m : float
        The scalar water surface the **depth** statistic is taken below, in the
        DEM's own vertical frame.  For a single-basin frame this is
        ``sea_level_override_m``.
    water_level_rule : str
        How that scalar was chosen.  **Mandatory, never blank**, for the same reason
        :class:`SillExpectation` refuses a bare float: a multi-basin frame has no
        single water surface, so whichever one is nominated here is a decision, and
        a decision with no sentence beside it is indistinguishable downstream from
        the only possible answer.
    other_basin_levels_m : tuple[float, ...]
        The remaining declared basin levels, when the frame has more than one.  They
        do **not** enter the ramp — they are used only to widen the envelope the
        out-of-range guard is measured against, because ``depth(cell) =
        level(owner) − dem(cell) ≤ max(levels) − dem(cell)`` holds cell by cell
        without knowing the ownership, which the limits pass deliberately does not
        compute (that is the priority flood, ~65 % of a native frame's wall time).
        Default ``()`` = a single-basin frame.
    """

    epoch_ka: float
    raster_paths: tuple[str, ...]
    water_level_m: float
    water_level_rule: str
    other_basin_levels_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.epoch_ka):
            raise ValueError(f"SequenceEpoch.epoch_ka is {self.epoch_ka!r}")
        if not self.raster_paths:
            raise ValueError(
                f"SequenceEpoch({self.epoch_ka} ka).raster_paths is empty. An epoch "
                "with no raster contributes no statistic and would silently shrink "
                "the sequence the limits claim to cover."
            )
        if not np.isfinite(self.water_level_m):
            raise ValueError(
                f"SequenceEpoch({self.epoch_ka} ka).water_level_m is "
                f"{self.water_level_m!r}. Depth is measured below it; a non-finite "
                "surface makes every depth non-finite."
            )
        if not isinstance(self.water_level_rule, str) or not self.water_level_rule.strip():
            raise ValueError(
                f"SequenceEpoch({self.epoch_ka} ka).water_level_rule is "
                f"{self.water_level_rule!r}. Write down which surface this scalar is "
                "and why it is the one the depth ramp should be built on. A "
                "multi-basin frame has several and this field is the only place the "
                "choice is recorded."
            )
        for level in self.other_basin_levels_m:
            if not np.isfinite(level):
                raise ValueError(
                    f"SequenceEpoch({self.epoch_ka} ka).other_basin_levels_m holds "
                    f"{level!r}; it widens an envelope and a non-finite edge makes "
                    "the envelope unbounded."
                )

    @classmethod
    def from_tile_paths(
        cls,
        epoch_ka: float,
        tile_paths: dict[str, str | Path],
        *,
        water_level_m: float,
        water_level_rule: str,
        other_basin_levels_m: Sequence[float] = (),
    ) -> "SequenceEpoch":
        """Build one from the same ``tile_paths`` dict the renderer takes."""
        return cls(
            epoch_ka=float(epoch_ka),
            raster_paths=tuple(
                sorted(str(Path(p).resolve()) for p in tile_paths.values())
            ),
            water_level_m=float(water_level_m),
            water_level_rule=str(water_level_rule),
            other_basin_levels_m=tuple(float(x) for x in other_basin_levels_m),
        )

    @classmethod
    def from_basin_seeds(
        cls,
        epoch_ka: float,
        tile_paths: dict[str, str | Path],
        seeds: Sequence["BasinSeed"],
        *,
        depth_reference: str,
        water_level_rule: str,
    ) -> "SequenceEpoch":
        """Build one from the ``BasinSeed`` list the frame will be rendered with.

        *depth_reference* names WHICH seed's level the depth ramp is built on.  It is
        mandatory and is not defaulted to "the deepest" or "the first": on the
        Khvalynian frame the two levels are +29.944 m (Caspian) and −18.020 m
        (Pontic), 47.96 m apart, and building the ramp on the wrong one shifts every
        Pontic depth by that much.  The other levels are carried into
        ``other_basin_levels_m`` so the envelope still bounds them.
        """
        by_name = {s.name: s for s in seeds}
        if depth_reference not in by_name:
            raise ValueError(
                f"depth_reference={depth_reference!r} is not one of the seeds "
                f"({sorted(by_name)}). The depth ramp has to be built on a basin "
                "that exists in this frame."
            )
        return cls(
            epoch_ka=float(epoch_ka),
            raster_paths=tuple(
                sorted(str(Path(p).resolve()) for p in tile_paths.values())
            ),
            water_level_m=float(by_name[depth_reference].level_m),
            water_level_rule=str(water_level_rule),
            other_basin_levels_m=tuple(
                float(s.level_m) for s in seeds if s.name != depth_reference
            ),
        )

    def to_dict(self) -> dict:
        return {
            "epoch_ka": float(self.epoch_ka),
            "raster_paths": list(self.raster_paths),
            "water_level_m": float(self.water_level_m),
            "water_level_rule": self.water_level_rule,
            "other_basin_levels_m": [float(x) for x in self.other_basin_levels_m],
        }


@dataclass(frozen=True)
class SequenceLimits:
    """One colour mapping, shared by every frame of a sequence, with its provenance.

    WHAT THIS FIXES
    ---------------
    Every colour limit in :func:`render_paleocostline_epoch` used to be derived from
    the CURRENT frame's own data, so adjacent epochs got different mappings and the
    animation shimmered even where the terrain had not moved.  The full list, all of
    it the same defect, and all of it pinned by this object:

    ==============================  ==================================  ===========
    quantity                        was derived per frame from           field
    ==============================  ==================================  ===========
    land ramp floor                 ``nanpercentile(dem[land], 2)``      ``land_min_m``
    land ramp ceiling               ``nanpercentile(dem[land], 98)``     ``land_max_m``
    land ramp when a frame has no
    land at all                     the literals ``0.0, 800.0``          both above
    bathymetric ramp end            ``percentile(depth[ocean], 95)``     ``depth_max_m``
    the depth COLOURBAR             the same value, printed              ``depth_max_m``
    hillshade contrast stretch      matplotlib's ``intensity.min/max()`` ``shade_min/max``
    ice colour ramp                 ``ice[ice_mask].max()``              ``ice_max_m``
    ice-dome blend weight           the same value                       ``ice_max_m``
    ==============================  ==================================  ===========

    The last row is not a colour limit and is pinned anyway: the dome blend weight
    feeds ``dem_fs``, ``dem_fs`` feeds the hillshade, and the hillshade multiplies
    the LAND colour — so leaving it per-frame would have left the flicker in the
    array the row above it was pinned to remove it from.

    WHAT IT COSTS, MEASURED
    -----------------------
    The mapping between two adjacent Khvalynian frames (``khv_16.5ka.tif`` at its
    own +29.944 m defensible surface, ``khv_16ka.tif`` at +38.099 m), ``max_px=2000``:

    ==================  ==========  ==========  =========
    per-frame quantity   16.5 ka     16.0 ka     moves by
    ==================  ==========  ==========  =========
    land p2              +34.902 m   +43.287 m   8.385 m
    land p98            +2379.065 m +2386.590 m  7.525 m
    depth p95            2146.970 m  2151.377 m  4.407 m
    ==================  ==========  ==========  =========

    Converted to what a viewer sees: **the same elevation is drawn up to 4/255
    differently in the two frames**, worst at +597 m; 21.4 % of the elevation range
    differs by ≥ 1/255; the same depth differs by up to 1.6/255.  At native the land
    percentiles move further still (+37.548 → +46.552 and +2393.161 → +2400.882).
    Two frames, 0.5 ka apart, on terrain that differs only by a smooth GIA field.

    RAMP ENDPOINTS ARE NOT THE ENVELOPE
    -----------------------------------
    Two different pairs of numbers live in here and conflating them would make the
    out-of-range colour useless:

    * ``land_min_m`` / ``land_max_m`` / ``depth_max_m`` are **ramp endpoints** —
      percentiles, where the colour saturates.  Roughly ``100 − p_hi`` % of land
      and ``100 − p_depth`` % of ocean sit outside them in every frame **by
      design**; that is what a p2/p98 ramp is.  (Measured on the pinned 16.5 ka
      Khvalynian frame at ``max_px=2000``: 4.7 % of land cells and 4.8 % of ocean
      cells.  It is not exactly 4 %/5 % because the pinned ramp is the envelope of
      the per-frame ramps, not this frame's own percentile.)  Their counts go in
      the sidecar under ``saturated_at_ramp_end`` and they are NOT painted with a
      reserved colour.
    * ``dem_data_min_m`` / ``dem_data_max_m`` / ``depth_data_max_m`` are the **true
      extrema the limits pass measured** over the whole sequence.  A cell outside
      those is a cell the limits never saw, i.e. the limits were computed over data
      that does not contain this frame.  Those cells ARE painted, counted, and under
      ``clip_policy="raise"`` they stop the run.  **On a correct run these counts
      are exactly 0.**

    Parameters
    ----------
    depth_min_m, depth_max_m : float
        Bathymetric ramp, in metres below the frame's water surface.  ``depth_min_m``
        is 0.0 by construction — :func:`ocean_depth_from_surface` clips at zero — and
        a non-zero floor moves where the palest blue starts.
    land_min_m, land_max_m : float
        Land elevation ramp, metres in the DEM's own frame.
    ice_max_m : float | None
        Ice thickness at which the ice ramp reaches white, metres.  ``None`` means
        the limits pass found no ice side-car anywhere in the sequence; rendering a
        frame that HAS ice against such limits raises, rather than quietly falling
        back to that frame's own maximum.
    shade_min, shade_max : float
        The raw ``n · d`` hillshade intensity mapped to black and to white.  See
        :func:`analytic_shade_range` — the default is geometry, not a measurement,
        and needs no pass over the data.
    dem_data_min_m, dem_data_max_m, depth_data_max_m : float
        The envelope described above.
    depth_percentile, land_percentile_lo, land_percentile_hi : float
        The recipe the ramp endpoints came out of, checked against the consuming
        ``RenderConfig`` so a config field cannot be silently ignored.
    vert_exag, azdeg, altdeg : float
        The hillshade recipe, checked the same way.  Only ``altdeg`` enters the
        analytic shade range; ``vert_exag`` and ``azdeg`` are recorded because they
        DO enter a measured one, and limits carrying measured shade numbers must not
        be consumed under a different sun.
    epochs : tuple[float, ...]
        **Which epochs this was computed over.**  Rendering an epoch that is not in
        here raises :class:`SequenceLimitsEpochError`.
    rasters : tuple[RasterFingerprint, ...]
        **What raster bytes fed it.**  Rendering a raster whose path is absent, or
        whose sha256 has changed, raises :class:`SequenceLimitsRasterError`.
    computed_from : str
        The method, in a sentence.  Never blank.
    epoch_rules : tuple[tuple[float, str], ...]
        Per epoch, the ``SequenceEpoch.water_level_rule`` that produced its depth
        statistic — so a sequence built on a nominated basin level says which one.
    cost_note : str
        What the pass cost, measured on the production raster.
    """

    depth_min_m: float
    depth_max_m: float
    land_min_m: float
    land_max_m: float
    ice_max_m: Optional[float]
    shade_min: float
    shade_max: float
    dem_data_min_m: float
    dem_data_max_m: float
    depth_data_max_m: float
    depth_percentile: float
    land_percentile_lo: float
    land_percentile_hi: float
    vert_exag: float
    azdeg: float
    altdeg: float
    epochs: tuple[float, ...]
    rasters: tuple[RasterFingerprint, ...]
    computed_from: str
    epoch_rules: tuple[tuple[float, str], ...] = ()
    cost_note: str = ""

    def __post_init__(self) -> None:
        scalars = {
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "land_min_m": self.land_min_m,
            "land_max_m": self.land_max_m,
            "shade_min": self.shade_min,
            "shade_max": self.shade_max,
            "dem_data_min_m": self.dem_data_min_m,
            "dem_data_max_m": self.dem_data_max_m,
            "depth_data_max_m": self.depth_data_max_m,
        }
        for name, value in scalars.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"SequenceLimits.{name} is {value!r}. A non-finite limit maps "
                    "every cell to the same colour or to none, and either way the "
                    "sequence stops being comparable frame to frame."
                )
        if self.depth_min_m < 0.0:
            raise ValueError(
                f"SequenceLimits.depth_min_m is {self.depth_min_m!r}. Depth is "
                "measured BELOW the water surface and clipped at zero by "
                "ocean_depth_from_surface, so a negative floor names cells that "
                "cannot exist and silently compresses the ramp instead."
            )
        for lo_name, hi_name in (
            ("depth_min_m", "depth_max_m"),
            ("land_min_m", "land_max_m"),
        ):
            span = float(scalars[hi_name]) - float(scalars[lo_name])
            if span < SEQUENCE_RAMP_MIN_SPAN_M:
                raise ValueError(
                    f"SequenceLimits {lo_name}={scalars[lo_name]!r} .. "
                    f"{hi_name}={scalars[hi_name]!r} spans {span!r} m, below the "
                    f"{SEQUENCE_RAMP_MIN_SPAN_M} m minimum. The compositor divides by "
                    "max(span, 1.0), so a narrower ramp would be silently widened to "
                    "1 m and the picture would not be on the ramp this object "
                    "reports."
                )
        if not (self.shade_min < self.shade_max):
            raise ValueError(
                f"SequenceLimits shade_min={self.shade_min!r} is not below "
                f"shade_max={self.shade_max!r}."
            )
        if self.shade_min < -1.0 or self.shade_max > 1.0:
            raise ValueError(
                f"SequenceLimits shade range [{self.shade_min!r}, "
                f"{self.shade_max!r}] leaves [-1, 1]. It is the dot product of two "
                "unit vectors; a bound outside that range cannot be attained by any "
                "terrain and would darken every frame by a constant nobody chose."
            )
        if self.ice_max_m is not None:
            if not np.isfinite(self.ice_max_m) or self.ice_max_m <= 0.0:
                raise ValueError(
                    f"SequenceLimits.ice_max_m is {self.ice_max_m!r}. Use None to "
                    "say the sequence holds no ice; a zero or negative maximum is a "
                    "divide the ice ramp cannot survive."
                )
        if self.dem_data_min_m > self.dem_data_max_m:
            raise ValueError(
                f"SequenceLimits dem envelope [{self.dem_data_min_m!r}, "
                f"{self.dem_data_max_m!r}] is inverted."
            )
        if self.depth_data_max_m < 0.0:
            raise ValueError(
                f"SequenceLimits.depth_data_max_m is {self.depth_data_max_m!r}; a "
                "negative deepest depth is not a depth."
            )
        if not self.epochs:
            raise ValueError(
                "SequenceLimits.epochs is empty. The whole point of this object is "
                "recording WHICH epochs it was computed over; limits that name none "
                "cannot be checked against the frame that consumes them."
            )
        if len(set(self.epochs)) != len(self.epochs):
            raise ValueError(
                f"SequenceLimits.epochs has repeats: {self.epochs!r}. A repeated "
                "epoch weights one frame twice in a statistic that is supposed to "
                "describe the sequence."
            )
        if not self.rasters:
            raise ValueError(
                "SequenceLimits.rasters is empty. Without the raster hashes the "
                "epoch list is a claim nobody can check."
            )
        if not isinstance(self.computed_from, str) or not self.computed_from.strip():
            raise ValueError(
                "SequenceLimits.computed_from is blank. It is the field that "
                "distinguishes a measured mapping from seven numbers somebody liked."
            )

    # -- membership and consumability ---------------------------------------

    def covers_epoch(self, epoch_ka: float) -> bool:
        """Exact-equality membership.  See :class:`SequenceEpoch` for why exact."""
        return float(epoch_ka) in tuple(float(e) for e in self.epochs)

    def fingerprints_for(self, tile_paths: dict[str, str | Path]) -> list[RasterFingerprint]:
        """The fingerprints matching *tile_paths*, or raise naming what is missing."""
        wanted = [Path(p).resolve() for p in tile_paths.values()]
        known = {Path(fp.path): fp for fp in self.rasters}
        missing = [str(p) for p in wanted if p not in known]
        if missing:
            raise SequenceLimitsRasterError(
                f"{len(missing)} of the {len(wanted)} raster(s) this frame is built "
                f"from are not among the {len(self.rasters)} the sequence limits "
                f"were computed over: {missing}. The limits describe a colour "
                "mapping over terrain they never read, so every statistic in them "
                "is about a different scene. Recompute the limits over the rasters "
                "you are rendering, or render the rasters they were computed over."
            )
        return [known[p] for p in wanted]

    def assert_consumable_by(
        self,
        config: "RenderConfig",
        *,
        epoch_ka: float,
        tile_paths: dict[str, str | Path],
        verify_hashes: bool,
    ) -> dict:
        """Refuse to be consumed under a claim this object does not support.

        Checks, in order: the epoch is one of :attr:`epochs`; every raster is one of
        :attr:`rasters`; optionally that each raster's bytes still hash to what was
        recorded; and that the ramp/hillshade recipe in *config* is the recipe these
        numbers came out of.

        Returns a dict describing what was checked, for the sidecar.  Raises
        :class:`SequenceLimitsEpochError`, :class:`SequenceLimitsRasterError` or
        :class:`SequenceLimitsRecipeError`.
        """
        if not self.covers_epoch(epoch_ka):
            raise SequenceLimitsEpochError(
                f"this frame is {float(epoch_ka)} ka and the sequence limits were "
                f"computed over {list(self.epochs)}. Sharing one colour mapping "
                "across a sequence is a claim that the pass which produced it saw "
                "every frame in that sequence; this frame was not one of them, so "
                "its own extremes are unrepresented and any cell beyond them would "
                "be clipped against a scale nobody measured here. Add the epoch to "
                "the limits pass, or render it with its own per-frame limits and say "
                "so."
            )
        fingerprints = self.fingerprints_for(tile_paths)
        hash_status: str
        if verify_hashes:
            changed = []
            for fp in fingerprints:
                digest, size = _sha256_file(Path(fp.path))
                if digest != fp.sha256:
                    changed.append(
                        f"{fp.path}: limits recorded {fp.sha256} "
                        f"({fp.size_bytes} bytes), the file now hashes {digest} "
                        f"({size} bytes)"
                    )
            if changed:
                raise SequenceLimitsRasterError(
                    "the raster(s) this frame is built from have changed since the "
                    "sequence limits were computed: " + "; ".join(changed) + ". "
                    "This project re-bakes rasters in place under the same file "
                    "name, so a matching path is not a matching scene. Recompute "
                    "the limits."
                )
            hash_status = (
                f"VERIFIED: {len(fingerprints)} raster(s) re-hashed at render time "
                "and matching the sha256 recorded in the limits"
            )
        else:
            hash_status = (
                "NOT VERIFIED: RenderConfig.verify_sequence_raster_hashes is False, "
                "so only the PATHS were matched. This is not the same check — the "
                "Pontic W0 re-bake rewrote 39 frames' inputs in place under "
                "unchanged file names. The sha256 the limits were computed against "
                "is still recorded below; re-hash the files to close it."
            )
        # The last two come from module constants rather than the config, because
        # that is where the sun lives.  They are checked anyway: ``shade_min`` is
        # DERIVED from ``altdeg`` (see analytic_shade_range), so a frame lit from a
        # different altitude than the limits were computed for is drawn on a floor
        # that does not belong to it, and nothing else in the pipeline would notice.
        recipe = [
            ("land_percentile_lo", self.land_percentile_lo, config.land_elev_percentile_lo),
            ("land_percentile_hi", self.land_percentile_hi, config.land_elev_percentile_hi),
            ("vert_exag", self.vert_exag, config.vert_exag),
            ("azdeg", self.azdeg, HILLSHADE_AZDEG),
            ("altdeg", self.altdeg, HILLSHADE_ALTDEG),
        ]
        mismatched = [
            f"{name}: limits computed at {mine!r}, this RenderConfig says {theirs!r}"
            for name, mine, theirs in recipe
            if float(mine) != float(theirs)
        ]
        if mismatched:
            raise SequenceLimitsRecipeError(
                "the sequence limits were computed under a different recipe than "
                "this render will use: " + "; ".join(mismatched) + ". With limits "
                "pinned the RenderConfig percentile fields are not consulted at all, "
                "so leaving them disagreeing would put a recipe in the sidecar that "
                "the picture was not made with — a config field that is silently "
                "ignored is the defect this module keeps deleting. azdeg/altdeg are "
                "module constants (HILLSHADE_AZDEG / HILLSHADE_ALTDEG) rather than "
                "config fields, and a mismatch there means shade_min was derived for "
                "a sun this frame is not lit by. Set them to match, or recompute the "
                "limits."
            )
        return {
            "epoch_in_limits": True,
            "rasters_checked": [fp.to_dict() for fp in fingerprints],
            "hash_status": hash_status,
            "recipe_checked": [
                {"field": name, "limits": float(mine), "config": float(theirs)}
                for name, mine, theirs in recipe
            ],
        }

    def to_dict(self) -> dict:
        return {
            "depth_min_m": float(self.depth_min_m),
            "depth_max_m": float(self.depth_max_m),
            "land_min_m": float(self.land_min_m),
            "land_max_m": float(self.land_max_m),
            "ice_max_m": (None if self.ice_max_m is None else float(self.ice_max_m)),
            "shade_min": float(self.shade_min),
            "shade_max": float(self.shade_max),
            "dem_data_min_m": float(self.dem_data_min_m),
            "dem_data_max_m": float(self.dem_data_max_m),
            "depth_data_max_m": float(self.depth_data_max_m),
            "depth_percentile": float(self.depth_percentile),
            "land_percentile_lo": float(self.land_percentile_lo),
            "land_percentile_hi": float(self.land_percentile_hi),
            "vert_exag": float(self.vert_exag),
            "azdeg": float(self.azdeg),
            "altdeg": float(self.altdeg),
            "epochs": [float(e) for e in self.epochs],
            "rasters": [fp.to_dict() for fp in self.rasters],
            "computed_from": self.computed_from,
            "epoch_rules": [
                {"epoch_ka": float(e), "water_level_rule": r}
                for e, r in self.epoch_rules
            ],
            "cost_note": self.cost_note,
            "ramp_vs_envelope_note": (
                "depth_max_m / land_min_m / land_max_m are RAMP ENDPOINTS "
                "(percentiles); cells beyond them saturate, which is what a "
                "percentile ramp is for, and their counts are reported under "
                "saturated_at_ramp_end. dem_data_* / depth_data_max_m are the TRUE "
                "EXTREMA the limits pass measured over the whole sequence; a cell "
                "beyond THOSE is a cell these limits never saw, is painted with a "
                "reserved colour, and reads zero on a correct run."
            ),
        }


def _sha256_file(path: Path) -> tuple[str, int]:
    """``(hex digest, size in bytes)`` of *path*, read in 1 MiB blocks.

    1.18 s for the 42 MB ``khv_16.5ka.tif`` on this machine, against ~340 s for the
    native render it certifies.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def compute_sequence_limits(
    epochs: Sequence[SequenceEpoch],
    *,
    config: Optional["RenderConfig"] = None,
    depth_percentile: float = 95.0,
    computed_by: str = "",
) -> SequenceLimits:
    """One colour mapping for a whole sequence, from the DEM statistics alone.

    WHY THIS IS NOT A SECOND RENDER PASS
    ------------------------------------
    A native two-basin frame costs ~340 s and ~4.2 GB peak on the machine this was
    written on, which pages under that load, so a limits pass that re-rendered every
    epoch would double a 34-frame sequence from ~3.2 h to ~6.4 h.  It is not needed:
    every quantity the compositor maps is a function of the input DEM, the ice
    side-car and the epoch's water surface, and all three can be read straight off
    the GeoTIFF.

    WHAT IT COSTS, MEASURED
    -----------------------
    Per epoch, on ``khv_16.5ka.tif`` (7800 × 4320 float32, 42 MB on disk, no
    overviews), 2026-08-03:

    ======================================  ==========  =========
    step                                     wall        peak RAM
    ======================================  ==========  =========
    full-resolution read                     1.39 s      135 MB
    statistics (min/max/percentiles)         0.17 s      +
    sha256 of the file bytes                 1.18 s      —
    sum of the parts                         2.74 s      135 MB
    **whole helper, end to end, per epoch**  **2.38 s**  **135 MB**
    ======================================  ==========  =========

    (The end-to-end figure is below the sum of its parts because the parts were
    timed separately on the same file and the second read hit the page cache.  Both
    are given rather than the flattering one.)

    i.e. **0.7 % of one native frame**, and flat in memory — one raster is held at a
    time, so the pass never competes with a render for RAM on a machine that pages
    at 4.2 GB.  A 34-frame sequence costs ~81 s.

    **Decimation was tried and refused, on measurement.**  Reading the same raster
    at 1/2 and 1/4 through ``rasterio``'s ``out_shape`` took **longer** (1.91 s and
    1.70 s against 1.39 s — there are no overviews, so the full file is decoded
    either way and resampling is added on top) and it moved the extremes: the DEM
    maximum read +5566.46 m at full resolution, +5406.72 m at 1/2 and **+5181.12 m
    at 1/4**.  A 385 m error in a limit, bought by making the pass slower.

    WHAT THE NUMBERS ARE, EXACTLY
    -----------------------------
    * ``land_min_m`` = **min over epochs** of that epoch's ``percentile_lo`` of
      ``dem`` over ``dem >= water_level_m``; ``land_max_m`` = **max over epochs** of
      its ``percentile_hi``.  Not the pooled percentile over the concatenated
      sequence: 34 epochs × 33 696 000 float32 is 4.6 GB, which does not fit beside
      a render on this machine.  The envelope-of-per-frame-ramps is also the
      stronger guarantee — no frame's own p2 sits above the pinned floor, so pinning
      can only ever widen a frame's ramp, never crop it.
    * ``depth_max_m`` = max over epochs of ``percentile(water_level − dem, q)`` over
      ``dem < water_level``.
    * ``dem_data_min_m`` / ``dem_data_max_m`` = the true extrema over every valid
      cell of every raster.  Exact, because the read is full-resolution.
    * ``depth_data_max_m`` = max over epochs of ``max(all declared levels) − min(dem
      over wet cells)``.  For a multi-basin epoch this is a genuine **upper bound**
      on the deepest cell — ``depth(cell) = level(owner) − dem(cell) ≤ max(levels) −
      dem(cell)`` holds cell by cell without knowing the ownership.
    * ``shade_min`` / ``shade_max`` = :func:`analytic_shade_range`.  No data is read
      for them at all; they are geometry.

    WHAT IT APPROXIMATES, SAID OUT LOUD
    -----------------------------------
    The land/sea split used here is the plain scalar ``dem >= water_level_m`` on the
    **unsmoothed, unmerged** raster.  The render's mask is not that: it is
    ``dem_threshold_ocean`` on the σ=1.5 ``dem_sm`` of the merged mosaic, with a
    morphological closing and opening, and on a multi-basin frame it is per-basin.
    So the percentile endpoints are approximate.  They are allowed to be, because
    they are only where the colour saturates — and because anything the
    approximation misses lands outside ``dem_data_*`` / ``depth_data_max_m``, where
    the render paints it with a reserved colour and counts it instead of clipping it
    quietly.  That is the whole reason the envelope is a separate pair of numbers.

    One thing that is NOT an approximation: the merge and the seam blend cannot take
    a cell outside ``[dem_data_min_m, dem_data_max_m]``, because bilinear
    resampling and a Gaussian are both convex combinations of their inputs — with
    the caveat that ``build_epoch_mosaic`` substitutes 0.0 for nodata before
    blending, so the claim needs ``dem_data_min_m <= 0 <= dem_data_max_m``, which
    holds on every raster in this project and is asserted rather than assumed by the
    out-of-range counter.

    Parameters
    ----------
    epochs : sequence of SequenceEpoch
        Every frame the sequence will contain.  Rendering an epoch that is not here
        raises at render time.
    config : RenderConfig | None
        Where the land percentiles and ``vert_exag`` are taken from, so the limits
        and the render cannot disagree about the recipe.  Defaults to
        :class:`RenderConfig`'s own defaults.
    depth_percentile : float
        Percentile of depth that reaches the darkest blue.  95.0 reproduces the
        per-frame value this replaces.
    computed_by : str
        Optional free text appended to ``computed_from`` — who ran it, from what
        script.  The method sentence is written whether or not this is given.

    Returns
    -------
    SequenceLimits

    Raises
    ------
    ValueError
        No epochs, a repeated epoch, or a raster that does not exist.
    """
    cfg = config or RenderConfig()
    if not epochs:
        raise ValueError(
            "compute_sequence_limits got no epochs. A shared colour mapping over an "
            "empty sequence is not a mapping; there is nothing for the frames to "
            "share."
        )
    lo_q = float(cfg.land_elev_percentile_lo)
    hi_q = float(cfg.land_elev_percentile_hi)
    q_depth = float(depth_percentile)
    if not (0.0 <= lo_q < hi_q <= 100.0):
        raise ValueError(
            f"land percentiles {lo_q!r} / {hi_q!r} are not an ascending pair inside "
            "[0, 100]."
        )
    if not (0.0 < q_depth <= 100.0):
        raise ValueError(f"depth_percentile {q_depth!r} is outside (0, 100].")

    seen_epochs: list[float] = []
    fingerprints: list[RasterFingerprint] = []
    rules: list[tuple[float, str]] = []

    land_lo = np.inf
    land_hi = -np.inf
    depth_ramp = -np.inf
    dem_lo = np.inf
    dem_hi = -np.inf
    depth_envelope = -np.inf
    ice_hi: Optional[float] = None
    # Counted per RASTER, not per epoch — an epoch may be built from several tiles
    # and only some of them may hold land.  Named for what is counted.
    rasters_with_land = 0
    rasters_with_ocean = 0

    for spec in epochs:
        if float(spec.epoch_ka) in seen_epochs:
            raise ValueError(
                f"epoch {spec.epoch_ka} ka appears twice in the sequence. A repeated "
                "epoch weights one frame twice in every statistic below."
            )
        seen_epochs.append(float(spec.epoch_ka))
        rules.append((float(spec.epoch_ka), spec.water_level_rule))
        all_levels = (float(spec.water_level_m), *(float(x) for x in spec.other_basin_levels_m))
        highest_level = max(all_levels)

        for raw_path in spec.raster_paths:
            path = Path(raw_path)
            if not path.exists():
                raise ValueError(
                    f"epoch {spec.epoch_ka} ka names a raster that does not exist: "
                    f"{path}. build_epoch_mosaic silently DROPS a missing tile, so a "
                    "limits pass that did the same would compute a mapping for a "
                    "scene the render will not draw."
                )
            digest, size = _sha256_file(path)
            fingerprints.append(
                RasterFingerprint(
                    path=str(path.resolve()),
                    sha256=digest,
                    size_bytes=size,
                    epoch_ka=float(spec.epoch_ka),
                )
            )
            with rasterio.open(path) as ds:
                band = ds.read(1).astype(np.float32)
                declared_nodata = ds.nodata
            # Same nodata convention as build_epoch_mosaic: its merge writes -9999.0
            # and then treats everything at or below -9000 as absent.
            valid = np.isfinite(band) & (band > -9_000.0)
            if declared_nodata is not None and np.isfinite(declared_nodata):
                valid &= band != np.float32(declared_nodata)
            if not valid.any():
                raise ValueError(
                    f"{path} has no valid cells at epoch {spec.epoch_ka} ka; it "
                    "cannot contribute a colour limit."
                )
            values = band[valid]
            dem_lo = min(dem_lo, float(values.min()))
            dem_hi = max(dem_hi, float(values.max()))

            is_land = values >= spec.water_level_m
            if is_land.any():
                rasters_with_land += 1
                land_values = values[is_land]
                land_lo = min(land_lo, float(np.percentile(land_values, lo_q)))
                land_hi = max(land_hi, float(np.percentile(land_values, hi_q)))
                del land_values
            is_wet = ~is_land
            if is_wet.any():
                rasters_with_ocean += 1
                wet = values[is_wet]
                depths = np.clip(spec.water_level_m - wet, 0.0, None)
                depth_ramp = max(depth_ramp, float(np.percentile(depths, q_depth)))
                depth_envelope = max(
                    depth_envelope, float(highest_level - float(wet.min()))
                )
                del wet, depths
            del band, valid, values, is_land, is_wet

            ice_path = Path(str(path).replace(".tif", "_ice.tif"))
            if ice_path.exists():
                with rasterio.open(ice_path) as ids:
                    ice_band = np.clip(ids.read(1).astype(np.float32), 0.0, None)
                candidate = float(ice_band.max())
                if candidate > 0.0:
                    ice_hi = candidate if ice_hi is None else max(ice_hi, candidate)
                del ice_band
            gc.collect()

    if rasters_with_land == 0:
        raise ValueError(
            "no raster in this sequence has a single cell at or above its epoch's water "
            "level, so there is no land elevation ramp to pin. Check the water "
            "levels — this is what an F0/F1 frame error looks like from here."
        )
    if rasters_with_ocean == 0:
        raise ValueError(
            "no raster in this sequence has a single cell below its epoch's water level, "
            "so there is no bathymetric ramp to pin. Check the water levels."
        )
    shade_min, shade_max = analytic_shade_range(HILLSHADE_ALTDEG)

    method = (
        "full-resolution read of every input GeoTIFF (NO decimation: measured on "
        "khv_16.5ka.tif, out_shape=1/2 took 1.91 s and 1/4 took 1.70 s against "
        "1.39 s at full resolution, and the DEM maximum fell from +5566.46 m to "
        "+5406.72 m and +5181.12 m respectively). Land ramp = min over the "
        f"sequence's RASTERS (not epochs -- an epoch may be several tiles) of "
        f"percentile {lo_q:g}, and max over them of percentile {hi_q:g}, of dem over "
        "dem >= that epoch's water_level_m -- the envelope of the per-frame ramps, "
        "not the pooled percentile, which would need 4.6 GB for a 34-frame "
        "sequence and would not guarantee that pinning can only widen a frame's own "
        f"ramp. Depth ramp = max over the same rasters of percentile {q_depth:g} of "
        "(water_level_m - dem) over dem < water_level_m. dem_data_* = the true "
        "extrema over every valid cell of every raster, exact. depth_data_max_m = "
        "max over the same rasters of (max of all declared basin levels - min dem "
        "over wet cells), which bounds depth cell by cell without computing basin "
        "ownership. "
        "shade_min/shade_max = analytic_shade_range(altdeg), geometry, no data read. "
        "APPROXIMATION STATED: the land/sea split here is the scalar dem >= "
        "water_level_m on the unsmoothed unmerged raster. It is NOT the render's "
        "split, in three separate ways: the render thresholds the sigma=1.5 MERGED "
        "mosaic, it applies a morphological closing and opening, and on the "
        "multi-basin path its water is additionally HYDROLOGICALLY CONNECTED -- only "
        "components a basin seed reaches (717,981 cells / 121,035 km2 of the native "
        "16.5 ka Khvalynian frame lie below their owner's level and are still not "
        "painted, because nothing connects them to that basin's water). So the "
        "percentile endpoints here are computed over a WIDER wet set than the frame "
        "draws and are approximate -- which is why the true extrema are carried "
        "separately and anything beyond them is painted and counted at render time "
        "rather than clipped. Pinning a ramp over a superset can only widen it; it "
        "cannot clip a cell the frame does draw."
    )
    if computed_by.strip():
        method = f"{method} Run by: {computed_by.strip()}"

    return SequenceLimits(
        depth_min_m=0.0,
        depth_max_m=max(float(depth_ramp), 1.0),
        land_min_m=float(land_lo),
        land_max_m=float(land_hi),
        ice_max_m=ice_hi,
        shade_min=shade_min,
        shade_max=shade_max,
        dem_data_min_m=float(dem_lo),
        dem_data_max_m=float(dem_hi),
        depth_data_max_m=max(float(depth_envelope), 0.0),
        depth_percentile=q_depth,
        land_percentile_lo=lo_q,
        land_percentile_hi=hi_q,
        vert_exag=float(cfg.vert_exag),
        azdeg=HILLSHADE_AZDEG,
        altdeg=HILLSHADE_ALTDEG,
        epochs=tuple(seen_epochs),
        rasters=tuple(fingerprints),
        computed_from=method,
        epoch_rules=tuple(rules),
        cost_note=(
            "2.38 s and 135 MB per epoch measured end to end over the two-epoch "
            "Khvalynian sequence (7800x4320 float32, 42 MB on disk each); the parts "
            "timed separately sum to 2.74 s (1.39 s read + 0.17 s statistics + "
            "1.18 s sha256) and both figures are quoted rather than the flattering "
            "one. That is 0.7 % of the ~340 s a native two-basin render costs, and "
            "it is flat in memory -- one raster is held at a time -- so a 34-frame "
            "sequence costs about 81 s and never competes with the render for RAM. "
            "Decimating the read was tried and is SLOWER and lossier: see "
            "computed_from."
        ),
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

#: What ``RenderConfig.clip_policy`` may be.  ``None`` resolves to ``"mark"``
#: whenever ``sequence_limits`` is set — the default has to be the visible one, and
#: ``"clip"`` (the pre-2026-08 silent behaviour) has to be asked for by name.
CLIP_POLICIES = ("mark", "raise", "clip")


@dataclass
class RenderConfig:
    """Render parameters for :func:`render_paleocostline_epoch`.

    Parameters
    ----------
    max_px : int
        Maximum pixel count along the longer mosaic axis.  Larger mosaics are
        down-sampled at merge time.  Default 2048.
    dpi : int
        Output PNG DPI.  Default 150 (2100 × 2100 px at 14-inch figure size).
    fig_size_in : float
        Square figure side length in inches.  Default 14.
    vert_exag : float
        Vertical exaggeration for hillshade.  Default 8×.
    ice_threshold_m : float
        Ice-sheet pixels thicker than this value (m) receive the white override.
        Default 10 m.
    seam_sigma_px : float
        Gaussian σ for tile-seam blending across the borders WHERE TWO TILES
        ACTUALLY MEET.  Default 5 px.  ``0`` disables the blend outright; a
        negative value is refused.  A single-file input has no seams and is never
        blended at any σ — see :func:`build_epoch_mosaic`.
    seam_blend_max_shift_m : float
        How far the seam blend may move the terrain before it has to say so.
        Defaults to :data:`SILL_TOLERANCE_M` — read that constant for **why 2 m**,
        which is a spill-window duration (2 m ≈ 0.3 ka against a 0.25 ka epoch
        grid) and not a number borrowed from :func:`assign_basins`.  Exceeding it
        emits a :class:`SeamBlendShiftWarning` carrying the measured shift and its
        position.
    legacy_degree_grid_blend : bool
        **Reproduction switch, not a feature.**  ``True`` restores the pre-2026-08
        behaviour of blurring a ±3σ strip around every WHOLE DEGREE line crossing
        the mosaic, tiles or no tiles.  It exists for exactly one purpose: to
        regenerate, byte-for-byte, frames that were baked before that was
        recognised as a defect — the 20 Pontic frames under
        ``renders/black_sea_corrected/topo/`` among them.  It is wrong physics on
        any new render; see :func:`build_epoch_mosaic` for what it costs.
    land_elev_percentile_lo : float
        Lower colour-scale percentile for land elevation (clips deep valleys).
    land_elev_percentile_hi : float
        Upper colour-scale percentile for land elevation (clips alpine peaks).
    measure_smoothing_sill_shift : bool | None
        Whether to re-measure every sill on the **unsmoothed** mosaic as well as on
        the σ=1.5 ``dem_sm`` the mask is actually built from, and report the
        difference (:class:`SmoothingShiftWarning` + a sidecar field).  ``None``
        (default) = measure, but only on the multi-basin path, where a sill exists
        at all; the single-basin ``sea_level_override_m`` path never measures a sill
        and this field is inert for it.  ``False`` skips the second flood and
        records the skip **in the sidecar as an explicit refusal string**, never as
        a zero — the shift is +0.25 m at native and +3.30 m at ``max_px=2000`` on
        the production Caspian↔Pontic raster, so a silent 0.0 would be a lie at
        every resolution.  Costs one extra priority flood (~65 % of a native
        frame's wall time), which is why it can be turned off with disclosure.
    smoothing_shift_disclose_m : float
        How far that measured shift may go before :class:`SmoothingShiftWarning`
        fires.  Defaults to :data:`SMOOTHING_SHIFT_DISCLOSURE_M`, which is this
        guard's **own** threshold.  It used to be wired to
        ``render_paleocostline_epoch``'s ``basin_sill_tolerance_m``, so widening
        the sill tolerance silently raised the bar on the smoothing disclosure as
        well — one knob, two guards, and the caller had said nothing about the
        second one.
    draw_sill_route : bool | None
        Whether to draw the measured minimax spill route as an annotated polyline.
        ``None`` (default) = draw it on the multi-basin path.  The route is the
        path the sill measurement is *about*; when the drawn water does not connect
        the two basins (see :func:`_painted_water_report`) it is the only thing in
        the picture that shows a reader where the verdict comes from.  It is a
        LINE, never painted water — see the ``spill_not_drawn`` disclosure in the
        sidecar for why painting it is refused.
    sequence_limits : SequenceLimits | None
        **One colour mapping shared by every frame of a sequence.**  ``None``
        (default) = every colour limit is derived from THIS frame's own data, which
        is today's behaviour and the reason an animation shimmers even where the
        terrain is identical — see :class:`SequenceLimits` for the full list of
        quantities that does to and for the measured 4/255 it costs between two
        adjacent Khvalynian frames.  When given, the mapping is read from the object
        and the frame's own extremes are used only to COUNT what falls outside it.

        Build it with :func:`compute_sequence_limits`, which reads the DEM
        statistics rather than rendering anything: 2.8 s and 135 MB per epoch
        against ~340 s for a native frame.
    clip_policy : {"mark", "raise", "clip"} | None
        What to do with cells whose value lies outside the envelope
        ``sequence_limits`` records having measured.  Only meaningful with
        ``sequence_limits`` set — passing it without them raises, because a policy
        about limits that do not exist is a policy about nothing.

        * ``None`` (default) → ``"mark"`` when limits are pinned; inert otherwise.
        * ``"mark"`` — paint those cells :data:`OUT_OF_RANGE_HIGH_RGB` /
          :data:`OUT_OF_RANGE_LOW_RGB`, count them, put the counts in the sidecar
          and a line in the figure title.  Visible, never silent.
        * ``"raise"`` — refuse the frame (:class:`SequenceLimitsExceededError`)
          BEFORE the PNG is written.  **This is the production setting for a
          sequence**: it is the only one under which a run cannot quietly
          misrepresent an epoch.
        * ``"clip"`` — clip them to the ramp end without painting, i.e. the
          pre-2026-08 behaviour, kept because reproducing an old frame is a real
          need.  The counts still go in the sidecar; the flag suppresses the paint,
          not the disclosure.
    verify_sequence_raster_hashes : bool | None
        Whether to re-hash each input raster at render time and compare it with the
        sha256 :class:`SequenceLimits` recorded.  ``None`` (default) = verify.
        ``False`` matches PATHS only and records the downgrade in the sidecar as an
        explicit refusal string, never as a pass — this project re-bakes rasters in
        place under unchanged file names, so a matching path is not a matching
        scene.  Costs 1.18 s for the 42 MB production raster, i.e. 0.35 % of a
        native frame.
    """

    max_px: int = 2048
    dpi: int = 150
    fig_size_in: float = 14.0
    vert_exag: float = 8.0
    ice_threshold_m: float = 10.0
    seam_sigma_px: float = 5.0
    seam_blend_max_shift_m: float = SILL_TOLERANCE_M
    legacy_degree_grid_blend: bool = False
    land_elev_percentile_lo: float = 2.0
    land_elev_percentile_hi: float = 98.0
    measure_smoothing_sill_shift: Optional[bool] = None
    smoothing_shift_disclose_m: float = SMOOTHING_SHIFT_DISCLOSURE_M
    draw_sill_route: Optional[bool] = None
    extra_figure_title: str = ""
    ice_model_label: str = "ICE-7G"
    """Name of the GIA/ice model shown in the figure title. The Black Sea bake
    is ICE-6G_C, so it passes 'ICE-6G_C' to keep the annotation truthful."""
    ice_dome_sigma_px: float = 60.0
    """Gaussian σ (pixels) applied to the DEM *before* hillshade on ice-covered
    pixels.  Removes bedrock texture under the glacier so the dome surface
    looks smooth.  σ=60 at 2048 px / 1° tile ≈ 2–3 km smoothing radius at
    30 m FABDEM resolution after downsampling — enough to erase valleys and
    ridges while preserving the large-scale dome curvature.
    Mirrors IceThicknessModel.get_thickness_like(sigma=0.7 on 1°-grid → cubic
    resampling) that produces smooth organic dome shapes in production.
    """
    sequence_limits: Optional[SequenceLimits] = None
    clip_policy: Optional[str] = None
    verify_sequence_raster_hashes: Optional[bool] = None

    def __post_init__(self) -> None:
        """Refuse the two combinations of the new fields that mean nothing.

        Deliberately validates ONLY the fields added in 2026-08.  Every other field
        keeps whatever validation it had (or did not have) so that the 530 existing
        tests construct exactly the configs they always did.
        """
        if self.clip_policy is not None:
            if self.clip_policy not in CLIP_POLICIES:
                raise ValueError(
                    f"RenderConfig.clip_policy={self.clip_policy!r} is not one of "
                    f"{sorted(CLIP_POLICIES)}."
                )
            if self.sequence_limits is None:
                raise ValueError(
                    f"RenderConfig.clip_policy={self.clip_policy!r} was set without "
                    "sequence_limits. A clip policy says what to do with cells that "
                    "fall outside a pinned mapping; with no pinned mapping there is "
                    "nothing to fall outside of, and accepting the flag would let a "
                    "caller believe clip_policy='raise' was protecting a sequence "
                    "that is still deriving every colour limit per frame."
                )
        if self.verify_sequence_raster_hashes is not None and self.sequence_limits is None:
            raise ValueError(
                "RenderConfig.verify_sequence_raster_hashes was set without "
                "sequence_limits. There are no recorded hashes to verify against."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Widest spacing (years) between the two published nodes bracketing a requested
# epoch that still counts as READING the eustatic curve rather than inventing a
# number across a hole in it.  The published Spratt & Lisiecki (2016) stack is on
# a uniform 1 ka grid (NOAA/WDS-Paleo study 19982), so 1000 yr means "the two
# adjacent published samples"; anything wider means the curve does not resolve
# the epoch and :func:`get_sea_level_for_epoch` says so instead of answering.
MAX_CURVE_NODE_GAP_YR = 1_000.0


class SeamBlendShiftWarning(UserWarning):
    """A tile-seam blend moved the terrain further than the caller allowed.

    Smoothing is not neutral on a spillway.  A Gaussian applied across a channel
    narrower than a few working pixels pulls the shoulders into the floor and the
    crest goes UP, which is the direction that turns a spilling basin endorheic.
    Measured on the real Caspian↔Pontic working raster (2026-08-01), the blend
    that used to run across every whole-degree line moved the Manych point sill at
    44.119 E / 45.773 N from +24.00 m to +40.76 m at ``max_px=2000``.

    So the blend reports what it did.  This warning carries the largest shift it
    caused and where, and it fires whenever that exceeds
    ``RenderConfig.seam_blend_max_shift_m``.  It is a warning and not an exception
    because a seam genuinely does need smoothing and 2 m is a bound, not a law —
    but it is never silent.
    """


class SeaLevelCurveResolutionError(ValueError):
    """The eustatic curve does not resolve the requested epoch.

    Raised when the two curve rows bracketing the request are further apart than
    ``max_node_gap_yr``, i.e. when producing a value would mean bridging a hole
    in the curve rather than reading between two adjacent published samples.
    """


def get_sea_level_for_epoch(
    epoch_ka: float,
    csv_path: Path,
    *,
    max_node_gap_yr: float = MAX_CURVE_NODE_GAP_YR,
) -> float:
    """Return the Spratt & Lisiecki (2016) sea-level value for *epoch_ka*.

    The CSV must contain three columns (no header, ``#``-prefixed comment lines
    allowed): **age_BP** (years), **sea_level_m** (m relative to present),
    **uncertainty_m** (m, not used here).

    Interpolated, not snapped
    -------------------------
    Sea level is a continuous quantity sampled on a discrete age grid, so a
    request between two nodes is answered by **linear interpolation in age**
    between the bracketing rows.  On a node the published value is returned
    bit-exact, so nothing that asks for a published age moves.

    This replaces a nearest-neighbour lookup that was *systematically biased*.
    The published stack is on a uniform 1 ka grid, ``pandas.Series.idxmin``
    returns the FIRST row attaining the minimum, and the CSV is ascending in
    age — so an epoch landing exactly halfway between two nodes always resolved
    to the YOUNGER, i.e. SHALLOWER, node.  Four of the thirteen epochs the
    Black-Sea bake ships do exactly that, two of them the hero frames::

          ka    nearest-neighbour      interpolated      error
        16.5      -98.06  (16 ka)         -102.680     +4.620 m
        14.5      -76.64  (14 ka)          -81.605     +4.965 m
         9.5      -15.83  ( 9 ka)          -20.210     +4.380 m
         7.5       -6.11  ( 7 ka)           -7.600     +1.490 m

    and it did so *silently*: the miss is exactly 500 yr — exactly half a step —
    which every guard in the project tests with a strict ``>`` inequality.
    Interpolation removes the tie rather than arbitrating it, so there is no
    half-step boundary condition left to fall through.

    What it refuses to do
    ---------------------
    * **Bridge a hole.**  If the bracketing rows are more than *max_node_gap_yr*
      apart the request is not a lookup, and :class:`SeaLevelCurveResolutionError`
      is raised rather than a plausible-looking number returned.
    * **Clamp in silence.**  Off either end of the curve the endpoint value is
      still returned (callers such as the bake's coverage check turn this into a
      hard stop of their own), but *any* such extrapolation now warns.  The old
      warning fired only above a 1000 yr miss, so a 0-20 ka curve asked for
      21 ka — missing by exactly 1000 yr — clamped without a word.

    This function is for the CONTINUOUS eustatic curve only.  Do not reuse it for
    a step-keyed or stage-keyed series: the discrete coastline layers of
    :func:`filter_paleocostlines_for_epoch` and a closed-basin curve carrying a
    genuine reconnection discontinuity must both stay nearest-neighbour, because
    the value halfway between two of their rows does not exist.

    Parameters
    ----------
    epoch_ka : float
        Target epoch in ka BP (e.g. ``21.0``).
    csv_path : Path
        Path to the eustatic curve CSV (``spratt_lisiecki_2016_full.csv``).
    max_node_gap_yr : float, keyword-only
        Widest bracketing-node spacing (years) still considered a lookup.
        Defaults to :data:`MAX_CURVE_NODE_GAP_YR` (one step of the published
        1 ka grid).  Raise it only if you own a coarser curve and are prepared
        to say so in writing.

    Returns
    -------
    float
        Sea level in metres relative to present (negative = lower than today).

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    SeaLevelCurveResolutionError
        If the curve is empty, or if the bracketing rows are further apart than
        *max_node_gap_yr*.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sea-level CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        comment="#",
        header=None,
        names=["age_bp", "sea_level_m", "uncertainty_m"],
    )
    df = df[["age_bp", "sea_level_m"]].apply(pd.to_numeric, errors="coerce").dropna()
    # np.interp / np.searchsorted are undefined on unsorted x, and a curve file
    # is not required to be sorted.  Sort explicitly; drop duplicate ages so a
    # repeated row cannot produce a zero-width bracket.
    df = df.drop_duplicates(subset="age_bp").sort_values("age_bp")
    ages = df["age_bp"].to_numpy(dtype=float)
    levels = df["sea_level_m"].to_numpy(dtype=float)
    if ages.size == 0:
        raise SeaLevelCurveResolutionError(
            f"Sea-level CSV has no usable (age_bp, sea_level_m) rows: {csv_path}"
        )

    want = float(epoch_ka) * 1_000.0

    # ── Off the end: clamp, but never in silence ────────────────────────────
    if want < ages[0] or want > ages[-1]:
        endpoint_age = ages[0] if want < ages[0] else ages[-1]
        age_diff = abs(endpoint_age - want)
        warnings.warn(
            f"Spratt CSV: closest row is {int(endpoint_age)} BP "
            f"(requested {int(want)} BP, Δ = {age_diff:.0f} yr). "
            f"The curve covers {int(ages[0])}–{int(ages[-1])} BP and does not "
            "reach this epoch — using clamped endpoint value.",
            UserWarning,
            stacklevel=2,
        )
        return float(levels[0] if want < ages[0] else levels[-1])

    # ── On a node: the published value, untouched ───────────────────────────
    hit = np.flatnonzero(ages == want)
    if hit.size:
        return float(levels[hit[0]])

    # ── Between nodes: interpolate, unless that would bridge a hole ─────────
    hi = int(np.searchsorted(ages, want))
    lo = hi - 1
    gap = ages[hi] - ages[lo]
    if gap > max_node_gap_yr:
        raise SeaLevelCurveResolutionError(
            f"eustatic curve {csv_path} does not resolve {epoch_ka} ka "
            f"({int(want)} BP): the bracketing rows are {int(ages[lo])} BP and "
            f"{int(ages[hi])} BP, {gap:.0f} yr apart — wider than the "
            f"{max_node_gap_yr:.0f} yr budget. Interpolating across that hole "
            "would invent a number the curve does not contain. Supply a curve "
            "that samples this epoch, or pass an explicit max_node_gap_yr."
        )
    frac = (want - ages[lo]) / gap
    return float(levels[lo] + frac * (levels[hi] - levels[lo]))


def load_paleocostlines(shp_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load and repair a Paleocoastlines vector file.

    Accepts any OGR-supported format: ``.shp``, ``.fgb`` (FlatGeobuf),
    ``.gpkg``, etc.  The bundled file ``data/paleocostlines/paleocoastlines_north_sea.fgb``
    covers −5°→15°E / 50°→63°N and can be used without any extra downloads.

    Returns a :class:`geopandas.GeoDataFrame` with a numeric ``Sea level``
    column, or ``None`` if loading fails.  Invalid geometries are repaired with
    :func:`shapely.validation.make_valid` and non-polygon results are dropped.

    Parameters
    ----------
    shp_path : Path
        Path to the Paleocoastlines vector file (".shp", ".fgb", or ".gpkg").

    Returns
    -------
    gpd.GeoDataFrame | None
        GeoDataFrame with columns ``geometry`` and ``Sea level`` (float, metres),
        or ``None`` on failure.
    """
    shp_path = Path(shp_path)
    if not shp_path.exists():
        warnings.warn(f"Paleocoastlines shapefile not found: {shp_path}", stacklevel=2)
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf = gpd.read_file(shp_path, engine="pyogrio", on_invalid="ignore")

        gdf = gdf[~gdf.geometry.isna()].copy()

        invalid_mask = ~gdf.geometry.is_valid
        n_repaired = int(invalid_mask.sum())
        if n_repaired:
            gdf.loc[invalid_mask, "geometry"] = (
                gdf.loc[invalid_mask, "geometry"].apply(make_valid)
            )
            gdf = gdf.explode(index_parts=False)
            gdf = gdf[
                gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            ].copy()

        gdf["Sea level"] = pd.to_numeric(gdf["Sea level"], errors="coerce")
        gdf = gdf[gdf["Sea level"].notna()].copy()
        return gdf

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to load Paleocoastlines shapefile: {exc}",
            stacklevel=2,
        )
        return None


def filter_paleocostlines_for_epoch(
    gdf: Optional[gpd.GeoDataFrame],
    target_sl_m: float,
) -> tuple[Optional[gpd.GeoDataFrame], Optional[int]]:
    """Return land polygons for the sea-level step nearest to *target_sl_m*.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame | None
        Full Paleocoastlines GeoDataFrame from :func:`load_paleocostlines`.
    target_sl_m : float
        Target sea level in metres (e.g. ``-120.0`` for LGM).

    Returns
    -------
    land_gdf : gpd.GeoDataFrame | None
        Subset of *gdf* for the matching sea-level step, or ``None`` if *gdf*
        is ``None``.
    matched_level : int | None
        The sea-level value (metres) of the matching step, or ``None``.
    """
    if gdf is None:
        return None, None

    levels = sorted(gdf["Sea level"].dropna().unique())
    matched = int(min(levels, key=lambda x: abs(x - target_sl_m)))
    land = gdf[gdf["Sea level"] == matched].copy()
    return land, matched


def rasterize_land_mask(
    land_gdf: Optional[gpd.GeoDataFrame],
    transform: rasterio.transform.Affine,
    height: int,
    width: int,
    mosaic_crs,
    bounds: BoundingBox,
) -> Optional[np.ndarray]:
    """Rasterize land polygons onto the DEM grid.

    Parameters
    ----------
    land_gdf : gpd.GeoDataFrame | None
        Land polygons (from :func:`filter_paleocostlines_for_epoch`).
    transform : rasterio.transform.Affine
        Affine transform of the output grid.
    height, width : int
        Output raster dimensions (rows × columns).
    mosaic_crs
        CRS of the output grid.  The GeoDataFrame is reprojected if needed.
    bounds : rasterio.coords.BoundingBox
        Geographic extent of the output grid (used for spatial clipping).

    Returns
    -------
    np.ndarray[bool] | None
        Boolean mask where ``True`` = land.  Returns ``None`` if rasterization
        fails or no polygons intersect the scene.
    """
    if land_gdf is None or len(land_gdf) == 0:
        return None

    try:
        clip_box = shp_box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf_proj = land_gdf.to_crs(mosaic_crs)
        clipped = gdf_proj.clip(clip_box)
        if len(clipped) == 0:
            return None

        geoms = [
            (g, 1)
            for g in clipped.geometry
            if g is not None and not g.is_empty
        ]
        if not geoms:
            return None

        mask = rio_rasterize(
            geoms,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)
        return mask

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Land-mask rasterization failed ({exc}); falling back to DEM threshold.",
            stacklevel=2,
        )
        return None


def ocean_depth_from_surface(
    dem: np.ndarray,
    ocean_mask: np.ndarray,
    water_level: float | np.ndarray,
) -> np.ndarray:
    """Water-column depth below the epoch water surface (m, ``>= 0``) on ocean pixels.

    ``depth = clip(water_level - dem, 0, inf)`` inside ``ocean_mask``, else 0.

    ``water_level`` is the epoch/basin surface in the SAME datum as ``dem``
    (present global SL = 0).  Using the epoch surface — not the present 0 m datum —
    is what makes a lowered decoupled basin read as a *shallower* body, so a
    ~100 m drawdown is visible in the shading instead of looking as deep as today.

    It may be a **scalar** (one basin, one surface — the original behaviour, and
    the arithmetic below is unchanged for it) or an **array broadcastable to
    ``dem``** (several basins standing at different levels in one frame, plus the
    ramped spill corridor between them — see :func:`assign_basins`).  Cells whose
    surface is NaN (no basin owns them) are never in ``ocean_mask``, so they
    contribute 0.
    """
    return np.where(ocean_mask, np.clip(water_level - dem, 0.0, None), 0.0)


#: Morphological de-speckle applied by :func:`dem_threshold_ocean`, named rather
#: than written twice as a literal so the sidecar can report the iteration counts
#: that were actually used instead of a number copied by hand beside them.
#: The gaussian sigma, in cells, that ``render_paleocostline_epoch`` applies to the
#: merged mosaic before ANY mask is thresholded from it.  It was a literal inside
#: that function and is named here because the sidecar's ``measured_on`` sentence
#: has to say which array a sill was read off, and a numeral typed into a sentence
#: is a numeral that can drift from the one the code runs.  Changing it changes
#: every frame this module has ever produced, including the 39 byte-stable Pontic
#: ones.
MASK_SMOOTHING_SIGMA_PX = 1.5

OCEAN_CLOSING_ITERATIONS = 2
OCEAN_OPENING_ITERATIONS = 2


def despeckle_ocean_mask(ocean_raw: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    """The morphological de-speckle alone: closing then opening, 2 iterations each.

    Split out of :func:`dem_threshold_ocean` so that a caller who has already
    decided which cells are water — the multi-basin path, whose mask is
    :attr:`BasinField.water_mask` and is **hydrologically connected**, not a bare
    threshold — can apply the SAME de-speckle without re-deriving the threshold a
    second time and getting a different answer.  The single-basin path still goes
    through :func:`dem_threshold_ocean` and is byte-identical to before this split:
    that function's body is now this call, with the threshold in front of it.
    """
    ocean = binary_closing(ocean_raw, iterations=OCEAN_CLOSING_ITERATIONS) & (~nodata)
    return binary_opening(ocean, iterations=OCEAN_OPENING_ITERATIONS).astype(bool)


def dem_threshold_ocean(
    dem_sm: np.ndarray,
    nodata: np.ndarray,
    water_level: float | np.ndarray,
) -> np.ndarray:
    """Boolean ocean mask from the DEM: submerged where ``dem_sm < water_level``.

    ``water_level`` is the epoch/basin surface (present-datum).  Morphological
    closing then opening (2 iterations each) removes single-pixel speckle.  This
    is the basin-correct replacement for the old hard ``dem_sm < 0.0`` fallback,
    which silently ignored the override.

    ``water_level`` may be a **scalar** (one basin at one level — unchanged) or an
    **array broadcastable to ``dem_sm``** (the per-cell water surface produced by
    :func:`assign_basins` when two basins stand at different levels in the same
    frame).  A NaN entry means "no basin owns this cell": ``dem_sm < NaN`` is
    False, so such a cell is land, which is the honest reading — the frame carries
    no level for it.

    **This threshold is FLAT — it has no connectivity condition, and that is
    correct here and wrong for a multi-basin frame.**  With a scalar level (the
    single-basin ocean path, and all 39 Pontic frames) the sea is the world ocean
    and every cell below the surface is genuinely part of it, so no connectivity
    test is wanted.  With a per-basin array it also paints every closed depression
    that happens to lie below its owner's surface, however far inland and however
    hydrologically disconnected — 717 981 cells of it on the native 16.5 ka
    Khvalynian frame.  The multi-basin path therefore does NOT call this function
    for its mask: it takes :attr:`BasinField.water_mask`, which is the seeded
    connected mask, and passes it to :func:`despeckle_ocean_mask` directly.
    """
    return despeckle_ocean_mask((~nodata) & (dem_sm < water_level), nodata)


# ---------------------------------------------------------------------------
# Two basins, two levels, one frame
# ---------------------------------------------------------------------------


class SillResolutionError(ValueError):
    """This frame's DEM disagrees with an independently measured sill elevation.

    ``spilling`` is only as honest as the grid it is observed on, and a spillway
    narrower than the working pixel is filled in by resampling.  A caller who has
    measured the sill independently passes it as ``sill_expectations`` and gets
    this exception instead of a quietly wrong animation.

    TWO DIFFERENT QUANTITIES — the guard compares the SECOND one
    ------------------------------------------------------------
    An earlier version of this docstring quoted the *point-cell* reading and then
    called its residual "inside the 2 m default tolerance below".  Both halves of
    that sentence are true of the point cell and neither is true of the guard, so
    the framing was wrong for the thing it was attached to.  The two quantities,
    both measured on the real Caspian↔Pontic working raster:

    **(a) the POINT CELL** at 44.119 E / 45.773 N — one bilinear sample of the
    downsampled mosaic, which is what ``manych_sill_elevation_m`` re-reads:

    =========================================================  ========  ========
    reading, at ``max_px``                                          2000      4000
    =========================================================  ========  ========
    native GEBCO 15 arcsec, untouched                           +23.00 m  +23.00 m
    bilinear-downsampled to the render mosaic                    +24.00 m  +25.00 m
    ... plus the seam blend as it stood on 2026-08-01            +40.76 m  +30.57 m
    =========================================================  ========  ========

    The last row is **gone as of 2026-08-01** (see :func:`build_epoch_mosaic`); the
    +16.76 m it added at a +35 m Khvalynian highstand turned a SPILLING frame into
    an endorheic one without a word.  The +1.00 / +2.00 m left over is the
    point-cell cost of downsampling, and it is **not monotone in resolution**:
    1000 px → +27, 2000 → +24, 3000 → +25, 4000 → +25, 6000 → +26, native → +23.
    That zig-zag is bilinear *phase* — where the sample grid happens to fall
    relative to the channel — not a cost that decreases as pixels get smaller.  It
    is exactly why **a minimum-``max_px`` precondition is refused as the gate**: at
    2048 px the point cell resolves 1.00 m WORSE than at 2000 px, so any rule of
    the form "≥ N pixels is fine" has a measured blind spot.

    **(b) the CONNECTED-PATH MINIMAX** — ``min{h : both seeds lie in one component
    of {array ≤ h}}``, which is what :func:`_measure_sills` computes off
    :func:`_priority_flood_ownership` and therefore what THIS exception compares.
    A spillway needs a continuous channel, so this is the hydrologically meaningful
    crest.  Measured on ``khv_16.5ka.tif`` against the bake's own
    ``sill_path_f1_m`` = **+27.701 m** (= ``MANYCH_SILL_PATH_M`` +25.00 F0 +
    ``dz_sill_m`` +2.701):

    ========  ===============  =====================  =================
    max_px    raw mosaic       ``dem_sm`` (σ=1.5)     gap vs +27.701
    ========  ===============  =====================  =================
    2000      +32.709          **+36.005**            **+8.30**
    4000      +27.899          **+32.942**            **+5.24**
    7800      +27.751          **+28.001**            **+0.30**
    ========  ===============  =====================  =================

    The guard is armed on ``dem_sm``, the column in bold, because that is the array
    :func:`dem_threshold_ocean` builds the mask from; guarding the raw mosaic would
    certify a surface nobody draws.

    **Consequence, stated as a refusal rather than sold as a setting:** the
    two-basin Caspian↔Pontic frame is renderable only on the native grid
    (``max_px ≥ 7800``).  This class is how that refusal is delivered.  It is
    deliberately NOT implemented as a resolution precondition — see (a) — and this
    module never raises ``max_px`` on the caller's behalf.

    Do not arm it with ``MANYCH_SILL_M`` (+23.00 m).  ``bake_khvalynian`` documents
    that constant as the max of per-column minima, i.e. a **lower bound** on the
    path crest; as an expectation it false-fails by +2.30 m even on the native grid,
    where the terrain and the frame actually agree to 0.30 m.
    """


class MissingSillExpectationError(ValueError):
    """A multi-basin frame was asked for without saying what its sills should be.

    Until 2026-08-03 ``assign_basins`` iterated ``(sill_expectations or {}).items()``,
    so the default ``None`` validated **nothing, at any number of seeds**: the
    entire guard was opt-in, and the one caller most likely to forget it — a batch
    animation loop — got a silently unchecked frame per epoch.

    The check is now mandatory whenever there is more than one basin, and it must
    cover **every** unordered pair, because a mapping that names one pair out of
    three leaves the other two exactly as unchecked as ``None`` did.  A pair that
    genuinely has no independent measurement is declared with
    :class:`UnmeasuredSill` and a written reason, which is recorded in the sidecar
    and shown in the figure — refusing to check is allowed; refusing *silently* is
    not.
    """


class SillExpectationTypeError(TypeError):
    """A sill expectation was given as a bare number instead of a provenanced one.

    A float carries no frame, no source and no place.  Downstream — in a sidecar,
    in a caption, in the next reader's head — it is indistinguishable from a
    measured value, which is the defect class this whole phase exists to remove.
    Pass a :class:`SillExpectation` (or an explicit :class:`UnmeasuredSill`).
    """


class SillFrameMismatchError(ValueError):
    """An expectation's vertical frame is not the frame of the DEM it is checked against.

    Comparing an **F0** constant against an **F1** saddle is a real error that has
    already been made once on this project: it produced the claim that "the guard
    cannot be satisfied at any resolution", which was wrong by exactly the missing
    ``dz_sill_m`` = +2.701 m.  The frames are named in
    ``2026-07-30-vertical-datum-contract.md`` §1; the DEM declares its own in the
    GeoTIFF tag ``vertical_frame`` and the expectation has to match that string.

    Note what the tag on ``khv_16.5ka.tif`` actually says: ``"F0 present-day datum
    (H_modern + dz_solid)"``.  The DEM is F0, the basin LEVELS are F1, and the sill
    expectation is an F0 terrain elevation — which is why comparing +28.001 against
    +27.701 is legitimate.  This class exists so the comparison cannot be made
    without both sides naming the frame out loud.
    """


class UnseededWaterBodyError(ValueError):
    """A water body that contains no basin seed reached a mask that claims to be water.

    THE TEST IS COMPONENT MEMBERSHIP, NOT A COMPARISON OF FLOATS
    ------------------------------------------------------------
    Until 2026-08-03 this fired on a body **whose surface was not bit-equal to any
    declared level**.  That is a test of the corridor ramp, not of the water.  In
    ``corridor="step"`` the surface array is written as ``level_lookup[basin_id]``,
    so every painted cell holds a declared level *by construction*, the
    "undeclared" mask is all-False by construction, and the guard could not fire at
    all.  Step is the mode the acceptance artifact is rendered in, so the refusal
    this phase exists to install (brief §5, RAISE-not-compute item 5) was
    structurally unreachable on the shipped frame: at ``max_px=7800`` with a Caspian
    and a Pontic seed the **Aegean + Marmara** and the **Levantine** were still
    painted, each wearing the *Pontic lake's* −18.02 m surface, because the priority
    flood labels those cells "pontic" — a surface neither of them has.

    A declared LEVEL is not a declared WATER BODY.  This frame states a surface for
    the water that reaches a seed; a component that reaches no seed is a body the
    frame carries no level for, whatever float ended up in its cells.  So the
    quantity guarded is *a connected component of the mask that contains no basin
    seed*, found by labelling the mask and reading the seeds' own labels.  It is the
    same test in ``"ramp"`` and in ``"step"``.

    2026-08-04: IT IS NOW AN INVARIANT ASSERTION, AND IT STILL RAISES
    ----------------------------------------------------------------
    The refusal was right; the MASK it audited was wrong.  ``assign_basins`` used to
    build ``water_mask = isfinite(dem) & (dem < water_level)`` — a flat threshold
    with no connectivity condition — and then refuse every seedless component of it.
    On the real three-basin Khvalynian frame that is 179 components (``"step"``) and
    the frame could not be rendered at all.  But a pit in Anatolia that happens to
    lie below the Caspian's +29.94 m is not Caspian water and never was; the
    flat-surface model has no business filling it, and the honest correction is to
    stop calling it water rather than to loosen the refusal.  ``water_mask`` is now
    **hydrologically connected** (:func:`_seeded_water_components`): a cell is water
    only if it is reachable from a basin seed through cells that are each below
    their own declared surface.

    Under that definition a seedless component is not water at all, so it never has
    to be refused — and this exception becomes **unfirable by construction on
    ``assign_basins``' own mask**.  It is deliberately NOT deleted, NOT downgraded to
    a warning and NOT given a size threshold or a ``ceiling_margin``: it is now the
    post-condition that proves the construction did what it says.  If it ever fires
    from :func:`assign_basins` again, the connectivity pass is broken and the frame
    must not ship.  ``tests/test_viz.py`` exercises it against a deliberately
    malformed mask so that "cannot fire" stays a proof and does not decay into
    "never tested".

    Where it can still legitimately fire is the RENDERER's mask, which is the
    connected mask **after** the morphological closing/opening — a closing that
    bridges two specks can create a body upstream never had.  That is counted, not
    raised, under ``painted_water.unseeded_bodies_in_the_drawn_mask``.

    What the connected mask costs, stated rather than left to be discovered
    ----------------------------------------------------------------------
    Cells that used to be painted are now dry.  That is the point, and the size of
    it is written into the data, not only into this docstring: every
    ``.basins.json`` carries a ``connectivity`` block giving the flat-mask cell
    count, the connected count, the difference per basin, and the largest dropped
    components with the surface each was being painted at.  The way to paint one of
    them honestly is unchanged and is never a guess: **seed it** with a provenanced
    level of its own, or render an extent that does not contain it.
    """


#: What :func:`assign_basins` may do with a body whose connectivity this raster
#: cannot decide.  ``"drop"`` is deliberately absent — see
#: :class:`UnobservableConnectivityError`.
UNOBSERVABLE_POLICIES = ("mark", "refuse")


class UnobservableConnectivityError(ValueError):
    """A water body whose connectivity this raster cannot decide, refused on request.

    THE THIRD STATE
    ---------------
    ``{dem < level_b}`` splits into components, and a component either holds basin
    *b*'s seed or it does not.  The rule "water iff it holds a seed" is right for
    topography and wrong AT THE RASTER EDGE, where a component's route to the seed
    may simply leave the picture.  The Khvalynian scene stops at 36 N and the
    Aegean's link to the Levantine basin runs south of that line, so on the shipped
    frame the Aegean+Marmara and the Levantine hold no seed — not because a barrier
    was observed between them and the Pontic lake, but because this extent cannot
    see the connection either way.

    Three states, and the third must not collapse into either of the first two:

    * **CONNECTED** — reaches its seed inside the extent.  Water.  Painted.
    * **DISCONNECTED / CLOSED** — a depression whose rim is wholly inside the extent
      and stands above the level.  Not water.  Not painted.
    * **UNOBSERVABLE** — a component touching the edge of the observed grid below a
      declared level.  Not decidable here.

    Painting an unobservable body asserts a connection nobody observed, and hands it
    a surface nobody measured for it; dropping it asserts a barrier nobody observed,
    and draws open sea as dry land.  Both substitute a state for an unknown.  The
    default, ``unobservable_policy="mark"``, does neither: the body is drawn in a
    reserved hatched colour that reads as unresolved, named in the legend and the
    caption, and counted — cells, km², the edges it touches and the level it sits
    under — in ``basins.json``.  This exception is the other honest option, for a
    caller whose frame is ABOUT the unobservable region and for whom a disclosure is
    not enough.

    There is deliberately **no "drop" policy**.  Dropping is the behaviour this
    class exists to remove, and offering it as a setting is how it comes back.
    """


class SmoothingShiftWarning(UserWarning):
    """The σ=1.5 mask smoothing moved the measured sill.

    :func:`render_paleocostline_epoch` thresholds ``dem_sm``, not the mosaic, so the
    sill the frame resolves is the sill *of the smoothed surface*.  Measured on
    ``khv_16.5ka.tif`` (connected-path minimax, raw mosaic vs ``dem_sm``):

    ========  ============  ===========  ==========
    max_px    raw mosaic    ``dem_sm``   shift
    ========  ============  ===========  ==========
    2000      +32.709       +36.005      **+3.296**
    4000      +27.899       +32.942      **+5.043**
    7800      +27.751       +28.001      **+0.250**
    ========  ============  ===========  ==========

    The σ is kept: it feeds ``dem_threshold_ocean`` on *every* path, and removing it
    changes all 39 Pontic frames and destroys the byte-baseline they were just
    re-baked to.  So the shift is **measured and reported** instead.

    What must never happen here is a modelled correction.  A Gaussian lowers an
    isolated tent crest by ``σ·slope·sqrt(2/π)`` and raises a narrow channel floor
    by something else entirely; subtracting either from a measured sill would
    replace a number that came off the terrain with one that came off an assumed
    crest shape.  The shift is a fact about this raster at this resolution, and it
    is emitted as a warning and written into the sidecar — never applied.
    """


@dataclass(frozen=True)
class SillExpectation:
    """A sill elevation somebody measured, with everything needed to check it.

    A bare float is refused (:class:`SillExpectationTypeError`) because the fields
    below are exactly what distinguishes a measurement from a guess, and none of
    them can be reconstructed later from the number alone.

    Parameters
    ----------
    value_m : float
        The expected saddle elevation, in the SAME vertical frame as the DEM the
        guard runs on — i.e. no conversion happens here.  For the Khvalynian frame
        this is the bake's ``sill_path_f1_m`` = ``MANYCH_SILL_PATH_M`` +
        ``dz_sill_m`` (+27.701 m at 16.5 ka), **not** ``MANYCH_SILL_M`` (+23.00 m),
        which is documented as a lower bound on the path crest and false-fails by
        +2.30 m on the native grid.
    frame : str
        The vertical frame, spelled exactly as the DEM's own GeoTIFF tag
        ``vertical_frame`` spells it.  It is compared string-for-string against the
        DEM's declared frame and a mismatch raises
        :class:`SillFrameMismatchError`.  Not normalised, not fuzzy-matched: two
        spellings of "the same" frame are two claims, and reconciling them is the
        caller's job, in writing.
    source : str
        Where the number came from — a file path, a CSV row, a citation, or the
        literal string ``"OWNER DECISION <date>"``.  Never blank and never
        "obvious".
    measured_by : str
        How it was obtained: the method, not the person.  E.g. ``"connected-path
        minimax (priority flood) on the raw 15-arcsec mosaic"``.  This is the field
        that stops a point-cell reading being compared against a path minimax —
        the two differ by up to 8 m on this project's own raster.
    lon, lat : float
        Where it was measured.  A sill elevation without a position cannot be
        re-derived, only trusted.
    """

    value_m: float
    frame: str
    source: str
    measured_by: str
    lon: float
    lat: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value_m):
            raise ValueError(
                f"SillExpectation.value_m is {self.value_m!r}. A non-finite "
                "expectation cannot fail a comparison, so it would certify the "
                "frame by accident. Use UnmeasuredSill(reason=...) to say the "
                "sill is not known."
            )
        for name in ("frame", "source", "measured_by"):
            text = getattr(self, name)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"SillExpectation.{name} is {text!r}. It is one of the three "
                    "fields that distinguish a measurement from a guess and it "
                    "has no defensible default; write down what it actually is."
                )
        if not (np.isfinite(self.lon) and np.isfinite(self.lat)):
            raise ValueError(
                f"SillExpectation lon/lat = {self.lon!r}/{self.lat!r}. A sill "
                "elevation without a position cannot be re-derived, only trusted."
            )


@dataclass(frozen=True)
class UnmeasuredSill:
    """A declared refusal to check one basin pair's sill, with the reason recorded.

    This is not an escape hatch for the guard — it carries **no number**, so it can
    never be mistaken downstream for a measurement, and it is written into the
    sidecar (``sill[].expectation_status``) and shown in the figure title, so a
    frame checked this way says so on its face.  It exists because there are pairs
    for which "not measured" is the true answer: two territories that never touch
    at this epoch's flood ceiling, a third basin added to stop the ramp inventing a
    surface, or a synthetic fixture whose saddle is deliberately out of reach.

    Silence is what is forbidden, not abstention.
    """

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "UnmeasuredSill(reason=...) needs a written reason. Declining to "
                "check a sill is allowed; declining without saying why is the "
                "silent no-op this guard was made mandatory to remove."
            )


class CorridorNotRampableError(NotImplementedError):
    """More than one pair of basins is spilling into another at this epoch.

    The linear water-surface ramp implemented by :func:`assign_basins` is defined
    for exactly **one** donor→receiver corridor.  With several simultaneous
    corridors the surface in each depends on how the discharges divide, which is a
    hydraulic question this module does not answer and will not guess at.  Pass
    ``corridor="step"`` to get every basin at its own flat level with the
    corridors *marked* instead.
    """


@dataclass(frozen=True)
class BasinSeed:
    """One named basin, located by a point that is under its own water.

    Parameters
    ----------
    name : str
        Basin name, e.g. ``"caspian"``.  Must be unique within one call and is
        what appears in the figure, the sidecar JSON and the sill keys.
    lon, lat : float
        Seed coordinates **in the CRS of the DEM being assigned** (EPSG:4326 for
        every raster in this project).  The cell containing this point is the
        growth origin of the basin.
    level_m : float
        This basin's water surface for the epoch, in the SAME vertical frame as
        the DEM — i.e. F1 for the paleo-DEM, which for an F2 literature level
        means ``level + Δz_solid(x_marker, t)`` has ALREADY been applied by the
        caller.  See ``2026-07-30-vertical-datum-contract.md`` §1 and §5a-bis;
        this function does no frame conversion and cannot detect a wrong one.
    """

    name: str
    lon: float
    lat: float
    level_m: float


@dataclass(frozen=True)
class BasinField:
    """The per-cell result of :func:`assign_basins`.

    Attributes
    ----------
    basin_id : np.ndarray[int32]
        ``0`` = no basin owns this cell (nodata, above the flood ceiling, or in a
        depression that reaches neither seed).  ``i+1`` = owned by ``names[i]``.
    names, levels_m : tuple
        The seeds, in input order.
    bottleneck_m : np.ndarray[float64]
        For each owned cell, the lowest elevation a path from its seed can be held
        under — the minimax ("bottleneck") elevation.  ``inf`` where unowned.
        This is what makes the sill measurable: the saddle between two basins is
        the lowest ``max(bottleneck_a, bottleneck_b)`` over the cells where their
        territories touch.
    water_level_m : np.ndarray[float64]
        The water SURFACE at each cell: the owning basin's level, except inside a
        ramped spill corridor where it is interpolated between donor and receiver.
        NaN where no basin owns the cell.
    water_mask : np.ndarray[bool]
        The water, **hydrologically connected**: the union of the 4-connected
        components of ``dem < water_level_m`` that contain a basin seed.  This is
        the two-basin generalisation of the single scalar threshold, and the
        connectivity condition is the half of it that a scalar ocean does not need
        — an ocean cell below the surface really is part of the ocean, whereas a
        closed depression below a *lake's* surface is a pit, not that lake.  How
        much the flat test ``dem < water_level_m`` over-painted is in
        :attr:`connectivity`; the definition, and what "connected" means in each
        corridor mode, is in :func:`_seeded_water_components`.
    unobservable_mask : np.ndarray[bool]
        Cells of the components that hold no seed **and reach the edge of what this
        raster can observe** — an array side, or a nodata cell.  Disjoint from
        :attr:`water_mask` by construction.  These are not water and they are not
        land: this grid cannot tell, because the route that would decide it runs
        outside the extent.  The renderer draws them in a reserved hatched colour;
        see :class:`UnobservableConnectivityError` for why neither painting them nor
        dropping them is honest, and :attr:`connectivity` for the count, the area,
        the edges touched and the level each body sits under.
    corridor_mask : np.ndarray[bool]
        Cells of the spill corridor.  In ``"ramp"`` mode these are the corridor
        cells that are actually wet **and connected to a seed** (a stranded pocket
        below the ramp is dropped here as well as from ``water_mask``, because the
        renderer restores this mask verbatim into the drawn ocean); in ``"step"``
        mode they are the cells the spill would flow through, marked but NOT given
        a made-up surface, and NOT painted.
    spilling : bool
        **Observed from the terrain**, not read off a curve: True iff at least one
        pair of seeded fills merges at the epoch's own levels.
    connected_groups : tuple[tuple[str, ...], ...]
        Basins grouped by hydrological connection at this epoch.  Two separate
        basins give ``(("caspian",), ("pontic",))``; once they spill,
        ``(("caspian", "pontic"),)``.
    sill_m, sill_lonlat : dict
        Per unordered basin pair (key = the two names, sorted): the measured
        saddle elevation and where it is.  **NaN / None when the saddle lies above
        the flood ceiling** — the algorithm never invents one it did not reach.
    corridor_mode : {"none", "ramp", "step"}
    ceiling_m : float
        The elevation the ownership flood was capped at.
    sill_expected_m : dict
        Per pair: the ``SillExpectation.value_m`` this frame was checked against, or
        ``None`` where the pair was declared :class:`UnmeasuredSill`.
    sill_expectation_status : dict
        Per pair: a sentence saying how the pair was disposed of — the source and
        method of the expectation it agreed with, or ``"UNMEASURED: <reason>"``.
        This is what makes an unchecked pair visible in the sidecar instead of
        being indistinguishable from a checked one.
    sill_smoothing_shift_m : dict
        Per pair: ``sill_on_dem_sm − sill_on_the_unsmoothed_mosaic``, or ``None``
        when the second measurement was not taken.  Never 0.0 as a stand-in for
        "not measured": the real shift is +0.25 m at native and +3.30 m at
        ``max_px=2000`` on the production raster.
    sill_route_rc : dict
        Per pair: the seed-to-seed **minimax route** through that pair's saddle, as
        a tuple of ``(row, col)``.  Empty where the saddle was never reached.  This
        is the line the caption draws; it is not water and is never painted as
        water.
    connectivity : dict
        What the flat threshold ``dem < water_level_m`` would have painted and what
        the connected mask actually paints: cell counts, the difference per basin,
        the component counts, and the largest dropped bodies with the surface each
        was being painted at.  Written straight into ``basins.json`` as its own
        block.  See :func:`_connectivity_report`.
    Note that there is no ``unseeded_water`` attribute.  There used to be: painted
    bodies holding no seed were *disclosed* here when the float in their cells
    happened to equal a declared level, and refused otherwise.  That split is gone.
    :attr:`water_mask` is now built from the seeded components outwards, so it has
    no seedless body **by construction** rather than by refusal, and the bodies the
    flat threshold would have painted are counted in :attr:`connectivity` — where
    they are a measurement of what was NOT drawn, not a list of things that were.
    :class:`UnseededWaterBodyError` is retained as the post-condition that proves
    the construction.  What the DRAWN mask has after the renderer's morphology is
    counted separately, on that mask, under
    ``painted_water.unseeded_bodies_in_the_drawn_mask``.
    """

    basin_id: np.ndarray
    names: tuple[str, ...]
    levels_m: tuple[float, ...]
    bottleneck_m: np.ndarray
    water_level_m: np.ndarray
    water_mask: np.ndarray
    corridor_mask: np.ndarray
    spilling: bool
    connected_groups: tuple[tuple[str, ...], ...]
    unobservable_mask: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool)
    )
    sill_m: dict = dataclass_field(default_factory=dict)
    sill_lonlat: dict = dataclass_field(default_factory=dict)
    corridor_mode: str = "none"
    ceiling_m: float = float("nan")
    sill_expected_m: dict = dataclass_field(default_factory=dict)
    sill_expectation_status: dict = dataclass_field(default_factory=dict)
    sill_smoothing_shift_m: dict = dataclass_field(default_factory=dict)
    sill_route_rc: dict = dataclass_field(default_factory=dict)
    connectivity: dict = dataclass_field(default_factory=dict)


def _seed_rowcol(
    transform: rasterio.transform.Affine,
    lon: float,
    lat: float,
    shape: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Cell containing (*lon*, *lat*); raises if that is off the raster."""
    height, width = shape
    col_f, row_f = ~transform * (float(lon), float(lat))
    row, col = int(np.floor(row_f)), int(np.floor(col_f))
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"basin seed {name!r} at {lon:.4f} E / {lat:.4f} N falls outside the "
            f"raster (it maps to row {row}, col {col} of a {height}x{width} grid). "
            "A seed off the frame would silently produce an empty basin."
        )
    return row, col


def _priority_flood_ownership(
    dem: np.ndarray,
    seed_rc: Sequence[tuple[int, int]],
    ceiling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grow every seed by ascending bottleneck elevation.

    Returns ``(owner, bottleneck, parent)``.

    This is the classic priority-flood / flooding-watershed: the frontier cell that
    can be reached while staying LOWEST is always claimed next, so a cell ends up
    owned by the basin it drains to — the water divide, not a bounding box.  The
    value recorded per cell is the minimax elevation along the path that claimed
    it, which is exactly what a saddle measurement needs.

    ``parent`` is the flat index of the cell each cell was claimed FROM (``-1`` for
    the seeds themselves).  It used to be thrown away as soon as it had been used,
    which meant the module measured a spill route and then could not draw it.  It
    costs one ``int32`` per cell (135 MB on the 33.7 M-cell native grid) and it is
    what :func:`_minimax_route` walks: because ``bottleneck`` is non-decreasing
    along a parent chain by construction, the chain from any cell back to its seed
    is a path whose maximum elevation is exactly that cell's bottleneck value — the
    minimax path, not an approximation of one.

    Deterministic: heap entries are ``(bottleneck, flat_index, label)`` and a cell
    is pushed at most once, so ``flat_index`` breaks every tie the same way on
    every run and on every machine.  4-connectivity, matching
    :func:`scipy.ndimage.label`'s default structure so the ownership flood and the
    spill-connectivity test agree about what "connected" means.

    Cells above *ceiling*, and NaN cells, are never claimed.
    """
    height, width = dem.shape
    flat = np.ascontiguousarray(dem, dtype=np.float64).ravel()
    # Python lists are ~4x faster than numpy scalar indexing in this inner loop,
    # which runs once per cell of a 2000-px mosaic.  ``parent`` is an array.array
    # rather than a list for the opposite reason: it holds large ints, which are
    # not interned, so a list of them costs ~36 B/cell (1.2 GB natively) against
    # array.array's 4.
    elevations = flat.tolist()
    passable = (np.isfinite(flat) & (flat <= ceiling)).tolist()
    owner = [0] * flat.size
    bottleneck = [float("inf")] * flat.size
    parent = array("i", [-1]) * flat.size

    heap: list[tuple[float, int, int]] = []
    for label, (row, col) in enumerate(seed_rc, start=1):
        idx = row * width + col
        owner[idx] = label
        bottleneck[idx] = elevations[idx]
        heapq.heappush(heap, (elevations[idx], idx, label))

    push, pop = heapq.heappush, heapq.heappop
    last_col = width - 1
    last_row = height - 1
    while heap:
        here, idx, label = pop(heap)
        row = idx // width
        col = idx - row * width
        for nbr in (
            idx - width if row else -1,
            idx + width if row < last_row else -1,
            idx - 1 if col else -1,
            idx + 1 if col < last_col else -1,
        ):
            if nbr < 0 or owner[nbr] or not passable[nbr]:
                continue
            owner[nbr] = label
            deeper = elevations[nbr]
            reach = here if here > deeper else deeper
            bottleneck[nbr] = reach
            parent[nbr] = idx
            push(heap, (reach, nbr, label))

    return (
        np.asarray(owner, dtype=np.int32).reshape(height, width),
        np.asarray(bottleneck, dtype=np.float64).reshape(height, width),
        np.frombuffer(parent, dtype=np.int32).reshape(height, width).copy(),
    )


def _minimax_route(
    parent: np.ndarray,
    crest_rc: tuple[int, int],
    partner_rc: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """The seed-to-seed minimax path through a saddle, as a cell chain.

    *crest_rc* and *partner_rc* are the two 4-adjacent cells that
    :func:`_measure_sills` found straddling the basin border at the winning saddle.
    Each one's parent chain runs back to its own seed, so concatenating one chain
    reversed with the other gives an unbroken 4-connected route from seed to seed
    whose highest cell IS the reported sill.

    This is the object the caption needs.  The drawn water does not connect the two
    basins even when the terrain does (the corridor is refused as paint — see
    :class:`UnseededWaterBodyError` and the ``spill_not_drawn`` sidecar field), so
    the route is how a reader who only ever sees the PNG can tell which centimetres
    the verdict is about.
    """
    height, width = parent.shape
    flat_parent = parent.ravel()

    def chain(rc: tuple[int, int]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        idx = rc[0] * width + rc[1]
        seen = 0
        limit = flat_parent.size + 1
        while idx >= 0:
            out.append((idx // width, idx - (idx // width) * width))
            idx = int(flat_parent[idx])
            seen += 1
            if seen > limit:                     # pragma: no cover - impossible
                raise RuntimeError(
                    "the priority flood's parent pointers contain a cycle; a "
                    "route drawn from them would be a fiction"
                )
        return out

    return tuple(chain(crest_rc)[::-1] + chain(partner_rc))


def _measure_sills(
    basin_id: np.ndarray,
    bottleneck: np.ndarray,
    n_basins: int,
) -> dict[tuple[int, int], tuple[float, tuple[int, int], tuple[int, int]]]:
    """Saddle elevation + location for every pair of basin territories that touch.

    Returns ``{(label_lo, label_hi): (elevation, crest_rc, partner_rc)}``.

    The saddle between two basins is ``min`` over all 4-adjacent cell pairs
    straddling their common border of ``max(bottleneck_a, bottleneck_b)``, and its
    location is the higher of that winning pair — i.e. the actual crest cell.
    ``partner_rc`` is the OTHER cell of that winning pair, on the far side of the
    border; it is kept because the two together are the join between the two
    parent chains that make up the seed-to-seed minimax route
    (:func:`_minimax_route`), and throwing it away is what previously made the
    measured route undrawable.

    Pairs whose territories never touch (because the flood ceiling cut the ridge
    off) are simply absent, and the caller reports them as unknown.
    """
    best: dict[tuple[int, int], tuple[float, tuple[int, int], tuple[int, int]]] = {}
    height, width = basin_id.shape
    rows, cols = np.indices((height, width))

    for axis in (0, 1):
        if axis == 0:
            lab_a, lab_b = basin_id[:-1, :], basin_id[1:, :]
            bot_a, bot_b = bottleneck[:-1, :], bottleneck[1:, :]
            row_a, row_b = rows[:-1, :], rows[1:, :]
            col_a, col_b = cols[:-1, :], cols[1:, :]
        else:
            lab_a, lab_b = basin_id[:, :-1], basin_id[:, 1:]
            bot_a, bot_b = bottleneck[:, :-1], bottleneck[:, 1:]
            row_a, row_b = rows[:, :-1], rows[:, 1:]
            col_a, col_b = cols[:, :-1], cols[:, 1:]

        touching = (lab_a > 0) & (lab_b > 0) & (lab_a != lab_b)
        if not touching.any():
            continue
        crest = np.maximum(bot_a, bot_b)[touching]
        take_b = (bot_b >= bot_a)[touching]
        crest_row = np.where(take_b, row_b[touching], row_a[touching])
        crest_col = np.where(take_b, col_b[touching], col_a[touching])
        other_row = np.where(take_b, row_a[touching], row_b[touching])
        other_col = np.where(take_b, col_a[touching], col_b[touching])
        pair_lo = np.minimum(lab_a, lab_b)[touching]
        pair_hi = np.maximum(lab_a, lab_b)[touching]
        key = pair_lo.astype(np.int64) * (n_basins + 1) + pair_hi

        for encoded in np.unique(key):
            sel = key == encoded
            winner = int(np.argmin(crest[sel]))
            value = float(crest[sel][winner])
            where = (int(crest_row[sel][winner]), int(crest_col[sel][winner]))
            partner = (int(other_row[sel][winner]), int(other_col[sel][winner]))
            pair = (int(encoded // (n_basins + 1)), int(encoded % (n_basins + 1)))
            if pair not in best or value < best[pair][0]:
                best[pair] = (value, where, partner)

    return best


def _unseeded_water_bodies(
    water_mask: np.ndarray,
    water_level: np.ndarray,
    seed_rc: Sequence[tuple[int, int]],
    transform: rasterio.transform.Affine,
) -> tuple[int, list[int], list[tuple[int, float, float, float, float]], np.ndarray]:
    """Connected components of *water_mask* that contain no seed.

    Returns ``(n_components, seed_labels, unseeded, labels)`` where *seed_labels* is
    the component label under each seed **in input order** (``0`` = that seed is not
    in the mask at all), *unseeded* holds
    ``(cells, surface_min, surface_max, lon, lat)`` per seedless component, largest
    first, and *labels* is the labelling itself.

    ``labels`` is returned rather than recomputed by the caller because
    :func:`_seeded_water_components` needs exactly this array to BUILD the connected
    mask from, and labelling the native 33.7 M-cell grid twice to answer one question
    twice is both slow and an invitation for the two answers to differ.

    **4-CONNECTIVITY**, i.e. :func:`scipy.ndimage.label`'s default structure, which
    is the neighbourhood the whole module uses: :func:`_priority_flood_ownership`
    grows 4-connected, :func:`_measure_sills` scans 4-adjacent border pairs,
    :func:`_geodesic_hops` counts 4-connected hops and :func:`_minimax_route` walks a
    4-connected chain.  It is not a stylistic choice.  Under 8-connectivity two water
    cells touching only at a corner count as one body, so water leaks diagonally
    between two land cells through an opening of zero width — which would join
    basins across a divide the priority flood calls closed, and the painted mask and
    the spill verdict could then disagree about whether a channel exists.

    The test is **membership**, not a comparison of the surface floats against the
    declared levels.  The previous version asked "is this body's surface one of the
    numbers some basin declares", which in ``corridor="step"`` is true of every
    painted cell by construction — ``water_level`` is written as
    ``level_lookup[basin_id]`` — so the question had one possible answer and the
    guard built on it could not fire.  A component that reaches no seed is a water
    body this frame states no surface for, even when the float in its cells is a
    number some *other* body's basin declares.  See :class:`UnseededWaterBodyError`.

    The representative ``(lon, lat)`` is the cell of the component where the surface
    is lowest, taken in one C pass: a Python loop doing ``argmax(labels == i)`` is
    O(cells × components) and costs ~20 s on the native grid.
    """
    labels, n_components = ndi_label(water_mask)
    seed_labels = [int(labels[rc]) for rc in seed_rc]
    if n_components == 0:
        return 0, seed_labels, [], labels

    seeded = {v for v in seed_labels if v > 0}
    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
    index = np.array(
        [i for i in range(1, n_components + 1) if i not in seeded], dtype=np.int64
    )
    if index.size == 0:
        return int(n_components), seed_labels, [], labels

    surf_lo = np.atleast_1d(np.asarray(ndi_minimum(water_level, labels, index=index)))
    surf_hi = np.atleast_1d(np.asarray(ndi_maximum(water_level, labels, index=index)))
    where = ndi_minimum_position(water_level, labels, index=index)

    unseeded: list[tuple[int, float, float, float, float]] = []
    for offset, label_id in enumerate(index):
        cells = int(sizes[label_id])
        row, col = int(where[offset][0]), int(where[offset][1])
        lon, lat = transform * (col + 0.5, row + 0.5)
        unseeded.append(
            (cells, float(surf_lo[offset]), float(surf_hi[offset]),
             float(lon), float(lat))
        )
    unseeded.sort(key=lambda e: -e[0])
    return int(n_components), seed_labels, unseeded, labels


def _seeded_water_components(
    below_surface: np.ndarray,
    water_level: np.ndarray,
    seed_rc: Sequence[tuple[int, int]],
    transform: rasterio.transform.Affine,
    nodata: np.ndarray,
    basin_id: np.ndarray,
    names: Sequence[str],
) -> dict:
    """Sort every component of the flat mask into CONNECTED, UNOBSERVABLE or CLOSED.

    *below_surface* is the flat test ``isfinite(dem) & (dem < water_level)``, cell by
    cell against that cell's own declared surface.  Its components fall into three
    states and **the third is not a synonym for either of the first two**:

    ``CONNECTED``
        The component holds a basin seed.  It is that basin's water, and it is
        painted.

    ``UNOBSERVABLE``
        The component holds no seed **and reaches the edge of what this raster can
        observe** — an array side, or a nodata cell (:func:`_boundary_component_labels`).
        Its route to a seed, if it has one, runs outside the extent, so this raster
        cannot decide the question.  On the production Khvalynian frame that is the
        Aegean+Marmara and the Levantine: the scene stops at 36 N and the Aegean's
        link to the Levantine basin runs south of it.  These are open sea whose
        connectivity is not observable here.

    ``CLOSED``
        The component holds no seed and its rim lies wholly inside the extent, above
        the surface it carries.  It is a closed depression that happens to lie below
        whichever basin's level the ownership flood labelled its ground with — a pit
        in Anatolia below the Caspian's surface is not Caspian water — and a
        flat-surface model has no business filling it.  Not water, not painted.

    THE STATE THAT WAS BEING SUBSTITUTED
    ------------------------------------
    Until this revision the rule was "seeded component or nothing", so UNOBSERVABLE
    was collapsed into CLOSED and open sea was drawn as dry land in every shipped
    frame.  Dropping such a body asserts a barrier nobody observed, exactly as
    painting it would assert a connection nobody observed; both are substitutions of
    a state for an unknown, which is the same defect as substituting a number for a
    measurement.  What this function does instead is **keep the three states apart**
    and hand all three back, with the cells, the area, the edges touched and the
    surface each body carries, so the caller can disclose the third rather than
    quietly resolve it.  ``assign_basins`` marks it; ``unobservable_policy="refuse"``
    refuses the frame outright; nothing in this module drops it.

    Returns a dict with ``connected_mask``, ``unobservable_mask``, ``seed_labels``,
    ``flat_components``, ``labels`` (the labelling of *below_surface*, returned so a
    caller can measure a named component exactly instead of estimating it), and the
    per-body descriptions ``unobservable`` and ``closed``, largest first.

    WHAT "CONNECTED" MEANS IN EACH CORRIDOR MODE
    --------------------------------------------
    Connectivity is evaluated **on the final per-cell surface**, i.e. on
    ``below_surface`` as the caller built it, not on any one basin's level.  That
    single rule specialises correctly to both modes:

    * ``corridor="step"`` — the surface is piecewise constant,
      ``level_lookup[basin_id]``, so ``below_surface`` is each basin's own flat
      threshold glued along the territory borders, and the component containing seed
      *b* is exactly the component of ``{dem < level_b}`` containing *b* restricted
      to where that level is the declared one.  Where two basins' waters do touch
      along their border they are one component holding two seeds, and both are
      kept — which is the honest reading: a single water body with a level
      discontinuity across it is what the step mode is *for*.  The corridor itself
      needs no special handling: ``corridor_mask`` is ``candidates``, carved out of
      ``components(levels[donor])`` — the donor-seed component of a flood at the
      donor's own level — so it is already seed-connected by construction, and in
      step mode it is MARKED, never painted, so it never enters this mask at all.
    * ``corridor="ramp"`` — the surface varies cell by cell inside the corridor, so
      there is no single level whose threshold could be labelled instead.  A
      corridor cell is water iff the bed is below the interpolated profile there AND
      it is reachable from a seed through cells that are each below their own
      surface.  Because the ramp is linear and monotone in along-channel graph
      distance, the wetted corridor is an unbroken chain from the donor's water to
      the receiver's water wherever the bed stays under the profile; where the bed
      rises above it the chain breaks, and anything beyond the break that still lies
      under the profile is a **stranded pocket** — which is now simply not water.
      That is the ``_pocket_dem`` fixture, and at native it is the Aegean+Marmara.

    4-connectivity throughout, for the reason given in
    :func:`_unseeded_water_bodies`, which does the labelling.
    """
    n_components, seed_labels, _unseeded, labels = _unseeded_water_bodies(
        below_surface, water_level, seed_rc, transform
    )
    seeded = {v for v in seed_labels if v > 0}
    keep = np.zeros(n_components + 1, dtype=bool)
    for value in seeded:
        keep[value] = True
    connected_mask = keep[labels]

    edges = _boundary_component_labels(labels, below_surface, nodata)
    on_boundary: set[int] = set()
    for touched in edges.values():
        on_boundary |= touched

    unobservable_labels = sorted(
        lab for lab in range(1, n_components + 1)
        if lab not in seeded and lab in on_boundary
    )
    closed_labels = sorted(
        lab for lab in range(1, n_components + 1)
        if lab not in seeded and lab not in on_boundary
    )

    unobs = np.zeros(n_components + 1, dtype=bool)
    for lab in unobservable_labels:
        unobs[lab] = True
    unobservable_mask = unobs[labels]

    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
    area = _component_area_km2(labels, n_components, transform)

    def describe(chosen: Sequence[int]) -> list[dict]:
        """One entry per body, largest first, measured on THIS call's own arrays."""
        if not chosen:
            return []
        index = np.asarray(sorted(chosen, key=lambda lab: -int(sizes[lab])),
                           dtype=np.int64)
        surf_lo = np.atleast_1d(np.asarray(ndi_minimum(water_level, labels, index=index)))
        surf_hi = np.atleast_1d(np.asarray(ndi_maximum(water_level, labels, index=index)))
        where = ndi_minimum_position(water_level, labels, index=index)
        out = []
        for offset, lab in enumerate(index):
            row, col = int(where[offset][0]), int(where[offset][1])
            lon, lat = transform * (col + 0.5, row + 0.5)
            owner = int(basin_id[row, col])
            out.append(
                {
                    "cells": int(sizes[lab]),
                    "km2": float(area[lab]),
                    "surface_min_m": float(surf_lo[offset]),
                    "surface_max_m": float(surf_hi[offset]),
                    "painted_at_basin": names[owner - 1] if owner else None,
                    "touched_edges": sorted(
                        name for name in OBSERVATION_BOUNDARIES
                        if int(lab) in edges[name]
                    ),
                    "lon": float(lon),
                    "lat": float(lat),
                }
            )
        return out

    return {
        "connected_mask": connected_mask,
        "unobservable_mask": unobservable_mask,
        "seed_labels": seed_labels,
        "flat_components": int(n_components),
        "labels": labels,
        "connected_cells": int(connected_mask.sum()),
        "connected_components": len(seeded),
        "connected_km2": float(sum(float(area[lab]) for lab in seeded)),
        "unobservable": describe(unobservable_labels),
        "closed": describe(closed_labels),
        "edges": edges,
    }


#: The pixel neighbourhood every connectivity question in this module is asked in.
#:
#: Named rather than spelled ``4`` in a sentence because it ships inside
#: ``basins.json`` twice — once as the number ``connectivity`` and once inside the
#: prose of ``connectivity_note`` — and those two must be one object.  A note whose
#: digits are typed by hand is a note that can contradict the field beside it.
_CONNECTIVITY = 4

#: WGS84 mean meridional degree, km.  Interpolated into :data:`_AREA_RULE` rather
#: than typed into it, for the same reason.
_DEG_KM = 111.32

#: How a cell count on a lon/lat grid is turned into an area, written once so the
#: sentence that ships in the sidecar names the same formula the code runs.
#:
#: ``area(row) = |a| * |e| * _DEG_KM^2 * cos(lat_row)`` km^2, with ``a`` and ``e``
#: the affine's degree-per-pixel terms and ``lat_row`` the latitude of that row's
#: cell centres — i.e. a per-ROW cosine, not one cosine taken at the scene's mean
#: latitude.  The mean-latitude shortcut is wrong by a cosine ratio that grows with
#: the frame's latitude span, which on this project's extents is larger than most of
#: the differences these areas are quoted to illustrate.  No projection is involved
#: and none is claimed.
_AREA_RULE = (
    f"area(row) = |transform.a| * |transform.e| * ({_DEG_KM} squared) * cos(lat_row) "
    f"sq km, summed per row over the cells of the mask; {_DEG_KM} km/deg is the "
    "WGS84 mean degree, the cosine is taken PER ROW and not at the scene's mean "
    "latitude, and no map projection is involved"
)


def _row_area_km2(transform: rasterio.transform.Affine, height: int) -> np.ndarray:
    """Cell area in km^2 for each row of a north-up lon/lat grid.  See :data:`_AREA_RULE`."""
    rows = np.arange(height, dtype=np.float64) + 0.5
    lat = transform.f + transform.e * rows
    return (
        abs(transform.a) * abs(transform.e) * (_DEG_KM ** 2) * np.cos(np.radians(lat))
    )


def _component_area_km2(
    labels: np.ndarray, n_components: int, transform: rasterio.transform.Affine
) -> np.ndarray:
    """Area in km^2 of every labelled component, indexed by label.  See :data:`_AREA_RULE`.

    Accumulated one raster ROW at a time because the per-cell area depends only on
    the row, so a row's contribution is ``bincount(row) * area(row)``.  The
    alternative — broadcasting the per-row area to a full float64 array and taking
    one weighted ``bincount`` — allocates a second copy of the whole grid (270 MB at
    native), on a machine that already pages during this call.
    """
    out = np.zeros(n_components + 1, dtype=np.float64)
    row_area = _row_area_km2(transform, labels.shape[0])
    for row in range(labels.shape[0]):
        line = labels[row]
        line = line[line > 0]
        if line.size:
            out += np.bincount(line, minlength=n_components + 1) * row_area[row]
    out[0] = 0.0
    return out


#: The four sides of the raster plus the ragged inner edge of its valid data.  A
#: water body that reaches ANY of them has a connection this raster cannot follow.
OBSERVATION_BOUNDARIES = ("north", "south", "west", "east", "nodata")


def _boundary_component_labels(
    labels: np.ndarray, mask: np.ndarray, nodata: np.ndarray
) -> dict[str, set[int]]:
    """Which labelled components touch which edge of what this raster can observe.

    The observation window is the set of cells that carry an elevation.  It ends at
    two kinds of edge and both are returned separately, because they mean different
    things to a reader:

    * the four **array sides** — the scene was cropped there, and whatever the water
      does beyond the crop is outside the picture by choice;
    * **nodata** — the mosaic has no elevation for that cell, so the water's fate
      there is unknown even though it is inside the frame.

    Either way the answer to "does this body reach a seed?" is *not decidable from
    this raster*, which is a third state and not a synonym for "no".  See
    :func:`_seeded_water_components`.
    """
    height, width = labels.shape
    hits: dict[str, set[int]] = {}
    for name, window in (
        ("north", (slice(0, 1), slice(None))),
        ("south", (slice(height - 1, height), slice(None))),
        ("west", (slice(None), slice(0, 1))),
        ("east", (slice(None), slice(width - 1, width))),
    ):
        hits[name] = {int(v) for v in np.unique(labels[window]) if v > 0}
    adjacent = np.zeros(labels.shape, dtype=bool)
    adjacent[:-1, :] |= nodata[1:, :]
    adjacent[1:, :] |= nodata[:-1, :]
    adjacent[:, :-1] |= nodata[:, 1:]
    adjacent[:, 1:] |= nodata[:, :-1]
    hits["nodata"] = {
        int(v) for v in np.unique(labels[mask & adjacent]) if v > 0
    }
    return hits


def _connectivity_report(
    *,
    flat: np.ndarray,
    states: dict,
    basin_id: np.ndarray,
    names: Sequence[str],
    transform: rasterio.transform.Affine,
    corridor_before: int,
    corridor_after: int,
    corridor_mode: str,
    unobservable_policy: str,
) -> dict:
    """The three states of the flat mask, counted and kept apart.

    This is the quantification the connected mask owes the reader, and it belongs in
    the data rather than only in a commit message: a frame that silently paints
    fewer cells than the previous revision is exactly as opaque as one that silently
    paints more.  Every number here is measured on this call's own arrays.

    THE THREE STATES ARE REPORTED SEPARATELY, ON PURPOSE
    ----------------------------------------------------
    ``connected`` is water and is painted.  ``closed`` is a depression whose rim
    stands inside the extent and above the surface it carries — not water.
    ``unobservable`` is a body that reaches the edge of what this raster can see, so
    whether it is water cannot be decided here at all.  An earlier revision summed
    the last two into one ``dropped`` figure and drew both as land, which made open
    sea clipped by the frame indistinguishable from a pit in Anatolia.  The sum is
    still reported (``not_connected_cells``) so the three visibly add up, but it is
    no longer the only thing reported.

    ``cells_by_basin`` keys on the OWNER of each cell — the basin whose declared
    level the flat mask was about to paint it at — with ``unowned`` for cells no
    basin owns (they carry a NaN surface, so they were never in the flat mask
    either, and the count is 0 by construction; it is reported rather than omitted
    so the counts visibly sum).

    ``bodies[].painted_at_basin`` is read off ``basin_id`` at the component's own
    representative cell, which is the lowest-surface cell
    :func:`_unseeded_water_bodies` located — not inferred by matching the surface
    float against the declared levels, because two basins may share a level.  Its
    ``km2`` is the exact per-row sum over that component's own cells.
    """
    height, width = flat.shape
    row_area = _row_area_km2(transform, height)
    connected = states["connected_mask"]
    unobservable = states["unobservable_mask"]
    closed_mask = flat & ~connected & ~unobservable

    def km2(mask: np.ndarray) -> float:
        return float(np.dot(mask.sum(axis=1, dtype=np.int64), row_area))

    def by_basin(mask: np.ndarray) -> tuple[dict, dict]:
        cells: dict[str, int] = {}
        areas: dict[str, float] = {}
        for i, name in enumerate(names):
            owned = mask & (basin_id == i + 1)
            cells[name] = int(owned.sum())
            areas[name] = km2(owned)
        unowned = mask & (basin_id == 0)
        cells["unowned"] = int(unowned.sum())
        areas["unowned"] = km2(unowned)
        return cells, areas

    def by_key(bodies: Sequence[dict], key) -> dict:
        out: dict[str, dict] = {}
        for body in bodies:
            entry = out.setdefault(
                key(body), {"components": 0, "cells": 0, "km2": 0.0}
            )
            entry["components"] += 1
            entry["cells"] += int(body["cells"])
            entry["km2"] += float(body["km2"])
        return out

    unobservable_bodies = states["unobservable"]
    closed_bodies = states["closed"]
    closed_cells, closed_km2 = by_basin(closed_mask)
    unobs_cells, unobs_km2 = by_basin(unobservable)

    flat_cells = int(flat.sum())
    connected_cells = int(connected.sum())
    return {
        "rule": (
            f"a cell is water iff it lies in the {_CONNECTIVITY}-connected component "
            "of {dem < water_level_m} that contains a basin seed; water_level_m is "
            "each cell's OWN declared surface (its owner basin's level, or the "
            "interpolated profile inside a ramped corridor). A component that holds "
            "no seed but REACHES THE EDGE OF THE OBSERVED GRID is not decided by "
            "this rule at all — see the unobservable block"
        ),
        "connectivity": _CONNECTIVITY,
        "connectivity_note": (
            f"{_CONNECTIVITY}-connected, i.e. scipy.ndimage.label's default "
            "structure and the same neighbourhood _priority_flood_ownership, "
            "_measure_sills, _geodesic_hops and _minimax_route use. Under "
            f"{2 * _CONNECTIVITY}-connectivity two cells touching only at a corner "
            "would count as one "
            "body, so water would leak diagonally between two land cells through an "
            "opening of zero width and could join basins across a divide the "
            f"{_CONNECTIVITY}-connected flood calls closed"
        ),
        "states": (
            "CONNECTED = holds a basin seed -> water, painted. UNOBSERVABLE = holds "
            "no seed and touches the edge of what this raster can observe (an array "
            "side, or a nodata cell), so its route to a seed may run outside the "
            "extent and this grid cannot decide -> NOT painted as basin water and "
            "NOT drawn as land; see unobservable.policy. CLOSED = holds no seed and "
            "its rim lies wholly inside the extent above the surface it carries -> a "
            "depression, not water, not painted"
        ),
        "flat_mask_cells": flat_cells,
        "flat_components": int(states["flat_components"]),
        "connected_mask_cells": connected_cells,
        "not_connected_cells": flat_cells - connected_cells,
        "not_connected_fraction_of_flat": (
            0.0 if flat_cells == 0 else (flat_cells - connected_cells) / flat_cells
        ),
        "connected": {
            "components": int(states["connected_components"]),
            "cells": connected_cells,
            "km2": float(states["connected_km2"]),
        },
        "unobservable": {
            "policy": unobservable_policy,
            "components": len(unobservable_bodies),
            "cells": int(unobservable.sum()),
            "km2": km2(unobservable),
            "cells_by_basin": unobs_cells,
            "km2_by_basin": unobs_km2,
            "by_edge": by_key(
                unobservable_bodies, lambda b: "+".join(b["touched_edges"]) or "none"
            ),
            "by_level_m": by_key(
                unobservable_bodies,
                lambda b: f"{b['surface_min_m']:+.3f}..{b['surface_max_m']:+.3f}",
            ),
            "boundaries_checked": list(OBSERVATION_BOUNDARIES),
            "bodies": unobservable_bodies,
            "note": (
                "components of the flat mask that hold no basin seed AND reach the "
                "edge of the observed grid. Their connectivity is NOT decided by "
                "this raster: a body whose route to a seed leaves the extent is not "
                "hydrologically disconnected, it is unmeasured here. touched_edges "
                "names which boundary each one reaches and surface_min/max_m the "
                "declared level it sits under. They are neither painted as that "
                "basin's water (which would assert a connection nobody observed) "
                "nor drawn as land (which would assert a barrier nobody observed): "
                "under policy='mark' the renderer paints them in a reserved hatched "
                "colour that reads as unresolved and names them in the legend and "
                "the caption, and under policy='refuse' assign_basins raises "
                "UnobservableConnectivityError instead of producing a frame. The "
                "only way to turn one of them into water is the same as for any "
                "other body: seed it with a provenanced level of its own, or render "
                "an extent that contains its connection"
            ),
        },
        "closed": {
            "components": len(closed_bodies),
            "cells": int(closed_mask.sum()),
            "km2": km2(closed_mask),
            "cells_by_basin": closed_cells,
            "km2_by_basin": closed_km2,
            "bodies": closed_bodies[:10],
            "bodies_note": (
                "the largest of them, largest first; components is the full count. "
                "surface_min/max_m is the surface each body WAS being painted at by "
                "the flat test, i.e. the level of whichever basin the ownership "
                "flood labelled its ground with, not a level anybody measured for it"
            ),
            "note": (
                "components of the flat mask that hold no seed and touch no "
                "observation boundary: their rim is inside the extent and stands "
                "above the surface they carry, so the terrain in THIS raster says "
                "they are closed. This is the class the flat test was inventing "
                "water for, and it is correctly not painted"
            ),
        },
        "area_rule": _AREA_RULE,
        # The constant area_rule quotes, as a VALUE, so the sentence can be checked
        # against a field instead of against a memory.
        "area_rule_deg_km": _DEG_KM,
        # The corridor mask is trimmed to the connected water in RAMP mode only,
        # because the renderer restores that mask verbatim into the drawn ocean
        # there.  In "step" and "none" nothing is trimmed and the two counts are
        # equal — reported rather than omitted so the reader can see that.
        "corridor_mask_cells_before": int(corridor_before),
        "corridor_mask_cells_after": int(corridor_after),
        "corridor_mask_cells_dropped": int(corridor_before - corridor_after),
        "corridor_trim_rule": (
            "corridor_mask is intersected with the connected water mask when "
            f"corridor_mode == 'ramp' (this frame: {corridor_mode!r}). A ramped "
            "corridor cell that the connectivity pass dropped is a STRANDED POCKET "
            "— under the interpolated profile, but cut off from both the donor's "
            "and the receiver's water by bed standing above it — and leaving it in "
            "corridor_mask would let the renderer's ramp-mode OR put it straight "
            "back into the painted ocean. In 'step' and 'none' the mask is a MARK, "
            "never ORed in, and is not trimmed"
        ),
        "note": (
            "flat_mask_cells is what the FLAT test dem < water_level_m would have "
            "painted, with no connectivity condition at all. connected / "
            "unobservable / closed partition its components, and their cells sum to "
            "it; not_connected_cells is unobservable.cells + closed.cells, kept so "
            "the partition is visibly complete. Read the closed block for what the "
            "flat model was inventing and the unobservable block for what this "
            "extent cannot answer — collapsing the two is what drew open sea as land"
        ),
    }


def _corridor_cost(
    *,
    corridor_mask: np.ndarray,
    water_mask: np.ndarray,
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    donor: Optional[int],
    receiver: Optional[int],
    names: Sequence[str],
    levels: Sequence[float],
) -> dict:
    """What painting the corridor would draw, measured on THIS frame.

    Every figure the step-mode ``spill_not_drawn`` disclosure quotes comes from
    here.  It used to quote three constants baked into the template — a cell count,
    an area and a piece count measured once, on a different frame, with a different
    Caspian level — and the same ``basins.json`` carried a ``corridor.cells`` that
    contradicted them.  Two revisions of the module even disagreed with each other
    (1 099 766 cells / 747 pieces in one comment, 1 713 697 / 118 in the shipped
    sentence).  A sentence a reader cannot audit against the array beside it is worse
    than no sentence, so the sentence now has no numerals of its own.

    ``bed_max_m`` is measured rather than asserted for the same reason: the corridor
    is carved out of ``components(levels[donor])``, i.e. ``dem < donor level``, so it
    is bounded above by the donor's own surface — a fact an earlier revision stated
    as "+203.71 m", which that construction makes impossible.  The bound and the
    measurement both ship, so the claim is checkable.
    """
    cells = int(corridor_mask.sum())
    if not cells:
        return {
            "cells": 0,
            "km2": 0.0,
            "components": 0,
            "cells_not_in_the_connected_water": 0,
            "bed_min_m": None,
            "bed_max_m": None,
            "donor": None,
            "donor_level_m": None,
            "receiver": None,
            "receiver_level_m": None,
            "note": (
                "no corridor cell on this frame, so painting it would draw nothing"
            ),
        }
    row_area = _row_area_km2(transform, corridor_mask.shape[0])
    _labels, pieces = ndi_label(corridor_mask)
    del _labels
    bed = dem[corridor_mask]
    return {
        "cells": cells,
        "km2": float(np.dot(corridor_mask.sum(axis=1, dtype=np.int64), row_area)),
        "components": int(pieces),
        "cells_not_in_the_connected_water": int(
            (corridor_mask & ~water_mask).sum()
        ),
        "bed_min_m": float(np.nanmin(bed)),
        "bed_max_m": float(np.nanmax(bed)),
        "donor": None if donor is None else names[donor],
        "donor_level_m": None if donor is None else float(levels[donor]),
        "receiver": None if receiver is None else names[receiver],
        "receiver_level_m": None if receiver is None else float(levels[receiver]),
        "note": (
            "what corridor_mask would add to the drawn water if it were painted, "
            "measured on this frame. bed_max_m is bounded above by donor_level_m "
            "because the corridor is carved out of a flood AT that level; both are "
            "reported so the bound can be checked rather than believed. "
            "cells_not_in_the_connected_water is how many of those cells are not in "
            "the frame's own water mask, i.e. stand above the level they would have "
            "to be under or reach no seeded body"
        ),
    }


def _unobservable_refusal(
    bodies: Sequence[dict], names: Sequence[str], levels: Sequence[float]
) -> str:
    """The message :class:`UnobservableConnectivityError` carries, built from the numbers."""
    worst = "; ".join(
        f"{b['cells']:,} cells / {b['km2']:,.1f} sq km under "
        f"{b['surface_min_m']:+.2f}…{b['surface_max_m']:+.2f} m touching the "
        f"{'+'.join(b['touched_edges'])} boundary, near "
        f"{b['lon']:.4f} E / {b['lat']:.4f} N"
        for b in bodies[:5]
    )
    return (
        f"{len(bodies)} water bodies totalling "
        f"{sum(int(b['cells']) for b in bodies):,} cells / "
        f"{sum(float(b['km2']) for b in bodies):,.1f} sq km reach the edge of what "
        "this raster can observe below a declared level (the seeded levels are "
        + ", ".join(f"{n} {v:+.2f} m" for n, v in zip(names, levels))
        + f"). Largest: {worst}. Each one holds no basin seed, but its route to a "
        "seed — if it has one — runs OUTSIDE this extent, so its connectivity is "
        "not decided by this grid. It is not a closed depression and must not be "
        "drawn as one. You asked for this to be a refusal "
        "(unobservable_policy='refuse'). The ways out, none of which is a "
        "tolerance: render an extent that contains the connection; seed the body "
        "with a provenanced level of its own; or accept the default "
        "unobservable_policy='mark', which draws it as unresolved and records its "
        "cells, area, touched edges and level in basins.json."
    )


def _unseeded_water_refusal(
    unseeded: Sequence[tuple[int, float, float, float, float]],
    names: Sequence[str],
    levels: Sequence[float],
    *,
    mask_description: str,
) -> str:
    """The message :class:`UnseededWaterBodyError` carries, built from the numbers."""
    worst = "; ".join(
        f"{cells:,} cells at {lo:+.2f}…{hi:+.2f} m near {lon:.4f} E / {lat:.4f} N"
        for cells, lo, hi, lon, lat in unseeded[:5]
    )
    return (
        f"{len(unseeded)} water bodies totalling "
        f"{sum(e[0] for e in unseeded):,} cells were painted without containing any "
        f"basin seed (the seeded levels are "
        + ", ".join(f"{n} {v:+.2f} m" for n, v in zip(names, levels))
        + f"). Largest: {worst}. These are connected components of {mask_description} "
        "that no seed falls in, so this frame states no water surface for them: the "
        "float in their cells belongs to whichever basin the priority flood labelled "
        "the ground with, not to a body anybody measured. The test is component "
        "MEMBERSHIP and it is the same test in corridor='ramp' and corridor='step' — "
        "comparing the surface against the declared levels instead was a no-op in "
        "step mode, where every painted cell holds a declared level by construction. "
        "Do not widen a tolerance and do not switch corridor mode: seed each body "
        "with a provenanced level of its own, or render an extent that does not "
        "contain it. NOTE (2026-08-04): assign_basins' own water_mask is now built "
        "connected — only components a seed reaches are water — so this message "
        "reaching you FROM assign_basins means the connectivity pass in "
        "_seeded_water_components is broken, not that a level is missing. From the "
        "renderer it means the morphological closing bridged two specks into a body "
        "that did not exist upstream."
    )


def _geodesic_hops(region: np.ndarray, source: np.ndarray) -> np.ndarray:
    """4-connected hop count from *source* to every cell of *region*, staying inside it.

    Cells of *region* adjacent to *source* are at distance 1.  Unreachable cells
    are ``inf``.  This is a graph distance in PIXELS, not a metric length and not a
    hydraulic path — it exists only to parameterise the corridor ramp from one end
    to the other, and inside a channel it is monotone along the channel, which is
    all the ramp needs.
    """
    height, width = region.shape
    dist = np.full(region.size, np.inf)
    inside = region.ravel()

    adjacent = np.zeros_like(region)
    adjacent[:-1, :] |= source[1:, :]
    adjacent[1:, :] |= source[:-1, :]
    adjacent[:, :-1] |= source[:, 1:]
    adjacent[:, 1:] |= source[:, :-1]
    start = np.flatnonzero((region & adjacent).ravel())
    dist[start] = 1.0

    queue = deque(int(i) for i in start)
    last_row, last_col = height - 1, width - 1
    while queue:
        idx = queue.popleft()
        here = dist[idx]
        row = idx // width
        col = idx - row * width
        for nbr in (
            idx - width if row else -1,
            idx + width if row < last_row else -1,
            idx - 1 if col else -1,
            idx + 1 if col < last_col else -1,
        ):
            if nbr < 0 or not inside[nbr] or dist[nbr] <= here + 1.0:
                continue
            dist[nbr] = here + 1.0
            queue.append(nbr)

    return dist.reshape(height, width)


def assign_basins(
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    seeds: Sequence[BasinSeed],
    *,
    ceiling_margin_m: float = 100.0,
    corridor: str = "ramp",
    sill_expectations: Optional[dict] = None,
    sill_tolerance_m: float = SILL_TOLERANCE_M,
    dem_frame: Optional[str] = None,
    unsmoothed_dem: Optional[np.ndarray] = None,
    smoothing_shift_max_m: float = SMOOTHING_SHIFT_DISCLOSURE_M,
    nodata: Optional[np.ndarray] = None,
    unobservable_policy: str = "mark",
) -> BasinField:
    """Assign every cell to the basin it drains to, and threshold each at its own level.

    This is the multi-basin generalisation of a single scalar sea level.  The hero
    frame of the Khvalynian reconstruction — the Caspian near +35 m and the Pontic
    near −30 m **in one image**, with water crossing the Manych — is inexpressible
    with one number, and this is the object that expresses it.

    How membership is decided
    -------------------------
    By a seeded priority flood over the DEM (:func:`_priority_flood_ownership`),
    not by a bounding box: the frontier cell reachable at the LOWEST elevation is
    always claimed next, so a cell ends up in the basin whose seed it can reach
    without crossing a higher barrier — the drainage divide.  A consequence worth
    having is that the **saddle between two basins falls out of the same flood**,
    so the sill elevation reported here is *measured off the terrain* rather than
    quoted from a document.

    How the spill is decided
    ------------------------
    Also from the terrain.  Basin **A spills into B iff there is a path from A's
    seed to B's seed on which every cell lies below A's own water surface, and B is
    the first basin that path reaches.**  Two halves, both load-bearing:

    * **A donor-level flood, not a pairwise max.**  The test is
      ``dem < level_A``, never ``dem < max(level_A, level_B)``.  The pairwise max
      floods every basin lying between the two to the donor's level as well, which
      invents connections: a Caspian at +35 m "reaches" the Mediterranean straight
      through a Pontic basin 65 m lower, whatever the Bosphorus sill does.
    * **First basin reached.**  Water crossing into a third basin's territory joins
      that basin's water body at *that* basin's level.  Whether it goes further is
      the third basin's own donor test, so a qualifying path may cross only the
      donor's territory, the receiver's, and ground no basin drains.  Transitive
      reach is rebuilt afterwards by union-find — A→C and C→D put A, C and D in one
      :attr:`~BasinField.connected_groups` entry without ever asserting A→D.

    The verdict is a property of the terrain and the levels, not of the order
    *seeds* is given in.  It is the animation's whole subject, so it is a
    first-class output (:attr:`BasinField.spilling`,
    :attr:`BasinField.connected_groups`) rather than an assertion carried in from a
    level curve.

    Which cells are water — the connectivity condition, and its THIRD state
    ----------------------------------------------------------------------
    **A cell is basin b's water iff it lies in the connected component of
    ``{dem < level_b}`` that contains b's own seed**, and
    :attr:`BasinField.water_mask` is the union of that over every basin.  The
    condition is not decoration: a flat ``dem < level`` test paints every closed
    depression that happens to lie below its assigned owner's surface, however far
    inland and however hydrologically disconnected — a pit in Anatolia below the
    Caspian's surface is not Caspian water, it is a pit, and a flat-surface model has
    no business filling it.

    That rule is right for topography and **wrong at the raster edge**, and the
    correction is the point of this section.  A component whose route to a seed runs
    OUTSIDE the extent holds no seed either, and it is not hydrologically
    disconnected — its connectivity is simply not observable from this raster.
    Dropping it draws open sea as dry land.  So there are three states, and the third
    collapses into neither of the other two:

    * **CONNECTED** — holds a seed.  Water.  In :attr:`BasinField.water_mask`.
    * **CLOSED** — holds no seed, and its rim lies wholly inside the extent above the
      surface it carries.  A depression.  Not water, not painted.
    * **UNOBSERVABLE** — holds no seed and touches the edge of the observed grid (an
      array side, or a nodata cell).  Not decidable here.  It goes into
      :attr:`BasinField.unobservable_mask`, is drawn distinctly rather than as either
      water or land, and is counted in :attr:`connectivity` with its cells, its area,
      the edges it touches and the level it sits under.  Pass
      ``unobservable_policy="refuse"`` to raise
      :class:`UnobservableConnectivityError` instead of producing such a frame.

    Connectivity is evaluated on the **final** per-cell surface, so the same rule
    covers both corridor modes; :func:`_seeded_water_components` works the two cases
    out in full.  In ``"step"`` the surface is piecewise constant and the rule
    reduces to the sentence above, and ``corridor_mask`` (the raw ``candidates``,
    carved out of a flood at the donor's own level) is seed-connected already and is
    MARKED, not painted.  In ``"ramp"`` the surface varies per cell inside the
    corridor, so a corridor cell is water iff the bed is below the interpolated
    profile *and* the cell is reachable from a seed through cells that are each
    below their own surface — a pocket beyond a stretch of bed that rises above the
    profile is stranded, and stranded is not wet.  It is removed from
    ``corridor_mask`` as well as from ``water_mask``, because the renderer restores
    ``corridor_mask`` verbatim in ramp mode.

    4-connectivity throughout, matching :func:`_priority_flood_ownership`,
    :func:`_measure_sills`, :func:`_geodesic_hops` and :func:`_minimax_route`.  Under
    8-connectivity water would pass through a corner between two land cells — an
    opening of zero width — and could join two basins across a divide the flood
    calls closed, so the mask and the spill verdict could disagree about whether a
    channel exists.

    How much this drops, and against what, is measured per call and returned in
    :attr:`BasinField.connectivity`; the renderer writes that block into
    ``basins.json``.  A frame that quietly paints fewer cells than the previous
    revision is exactly as opaque as one that quietly paints more.

    The corridor between two levels
    -------------------------------
    Between a donor basin at +35 m and a receiver at −30 m the water surface is a
    **gradient, not a step, and not a 60 m wall of blue**.  With ``corridor="ramp"``
    (default) each corridor cell gets

    ``L = level_donor + (level_receiver − level_donor) · d_donor / (d_donor + d_receiver)``

    where the two ``d`` are 4-connected hop distances measured *inside the corridor*
    from the donor's water and from the receiver's water.  The profile is linear in
    along-channel graph distance, monotone, and exact at both ends.  It also fixes
    the failure mode a naive "flood everything below the donor level" produces —
    filling the whole receiving basin to the donor's surface — because far from the
    channel ``d_donor ≫ d_receiver``, so ``L`` collapses onto the receiver's level
    and those cells stay dry.

    **What this ramp is not.**  It is not hydraulics.  There is no discharge, no
    channel roughness, no backwater: the lateral extent of the flow is set here by
    where the terrain lies below a linearly-interpolated surface, and a real
    spillway's width is set by how much water is moving.  A linear water-surface
    profile is the standard first-order assumption for a uniform channel and it is
    defensible for a figure; it is not a flood model.  If several corridors are
    open at once the interpolation is undefined and this function
    :class:`raises <CorridorNotRampableError>` — pass ``corridor="step"`` to get
    every basin flat at its own level with the corridor cells MARKED in
    :attr:`BasinField.corridor_mask` instead of drawn as an invented surface.

    Parameters
    ----------
    dem : np.ndarray
        Elevation grid, metres, present-day datum; NaN = nodata.  Pass the SAME
        array the land/sea threshold will run on (the renderer passes its smoothed
        ``dem_sm``), so ownership, sill and mask cannot disagree.
    transform : rasterio.transform.Affine
        Affine transform of *dem*, used only to locate the seeds and to report the
        sill positions.
    seeds : sequence of BasinSeed
        One per basin.  Levels must already be in the DEM's vertical frame.
    ceiling_margin_m : float
        The ownership flood stops at ``max(level) + ceiling_margin_m``.  Cells above
        it are unowned (and land at every basin's level, so nothing is lost) — the
        cap is what keeps the flood proportional to the interesting part of the
        scene.  Raise it if you need a saddle that stands far above every level;
        a saddle above the ceiling is reported as NaN, never guessed.
    corridor : {"ramp", "step"}
    sill_expectations : mapping {(name, name): SillExpectation | UnmeasuredSill}
        **Mandatory once there is more than one basin, and it must name every
        unordered pair.**  Until 2026-08-03 this defaulted to ``None`` and the
        validation loop ran ``(sill_expectations or {}).items()``, so the whole
        guard was a no-op at any number of seeds — a frame could be, and was,
        shipped unchecked.  With one basin there is no pair and none is required.

        Each value is a :class:`SillExpectation` — a bare float raises
        :class:`SillExpectationTypeError`, because a number with no frame, no
        source and no place is indistinguishable downstream from a measured one.
        Its ``frame`` is compared string-for-string against *dem_frame*
        (:class:`SillFrameMismatchError` on a mismatch), and its ``value_m``
        against what THIS grid resolves, raising :class:`SillResolutionError`
        beyond *sill_tolerance_m*.  A saddle the flood never reached also raises:
        "I could not see it" must not read as "it agrees".

        A pair with no independent measurement is declared
        :class:`UnmeasuredSill`, whose written reason is carried into the sidecar
        and the figure.  Key order does not matter — but naming the SAME pair under
        BOTH orderings raises, because two entries are two claims and picking one of
        them by dict iteration order is exactly the substitution this guard exists
        to prevent.

        **The structure of this mapping is validated at any number of basins.**
        Only the numeric comparison needs a pair; a bare float, a key naming a basin
        that is not in the frame, or a duplicated pair is malformed with one seed
        just as it is with three, and a one-basin frame used to swallow all of them
        without a word.
    sill_tolerance_m : float
        Defaults to :data:`SILL_TOLERANCE_M`; read that constant for the derivation
        (a spill-window duration, 2 m ≈ 0.3 ka on a 0.25 ka epoch grid).  Widening
        it is a decision to be made in writing, not a way to get a frame out.
    dem_frame : str | None
        The vertical frame *dem* is in, spelled the way the raster spells it — the
        renderer passes the GeoTIFF's own ``vertical_frame`` tag.  Required as soon
        as any :class:`SillExpectation` is supplied: without it the frame check
        cannot happen, and a check that cannot happen must not look like a check
        that passed.
    unsmoothed_dem : np.ndarray | None
        The same terrain BEFORE the σ=1.5 smoothing that produced *dem*.  When
        given, every reachable sill is measured on it too and the difference is
        reported in :attr:`BasinField.sill_smoothing_shift_m`, with a
        :class:`SmoothingShiftWarning` past *smoothing_shift_max_m*.  It is never
        subtracted from anything — see that warning for why a modelled
        σ-correction is refused.  Costs a second priority flood.
    smoothing_shift_max_m : float
        Threshold for that warning; defaults to
        :data:`SMOOTHING_SHIFT_DISCLOSURE_M`, which is this guard's own constant.
        It is deliberately NOT *sill_tolerance_m*: widening the sill tolerance is a
        statement about how far the render grid may disagree with an independent
        measurement, and it must not silently also raise the bar on what this
        module's own Gaussian is allowed to do unannounced.
    nodata : np.ndarray | None
        Where *dem* has no elevation.  Defaults to ``~isfinite(dem)``, which is the
        same set the mask test uses.  It is a parameter because the observation
        boundary — the edge past which connectivity cannot be decided — is the edge
        of the VALID data, not only the edge of the array, and a caller that knows
        its nodata by a sentinel rather than by NaN must be able to say so.
    unobservable_policy : {"mark", "refuse"}
        What to do with a component that holds no seed and reaches the observation
        boundary.  ``"mark"`` (default) puts it in
        :attr:`BasinField.unobservable_mask` and counts it in :attr:`connectivity`,
        for the renderer to draw as unresolved; ``"refuse"`` raises
        :class:`UnobservableConnectivityError`.  There is no ``"drop"`` and no
        ``"paint"``: dropping asserts a barrier nobody observed and painting asserts
        a connection nobody observed.

    Raises
    ------
    ValueError
        No seeds; duplicate basin names; a seed off the raster; a seed on nodata;
        an expectation naming a basin that is not in *seeds*; the same pair named
        under both orderings; or a seed standing ABOVE its own level — that last one
        would quietly produce an empty basin, which is precisely the class of silent
        wrongness this project refuses.
    MissingSillExpectationError
        More than one basin and no expectations, or expectations that do not cover
        every pair.
    SillExpectationTypeError
        An expectation given as a bare number.
    SillFrameMismatchError
        An expectation's frame is not the DEM's frame, or no DEM frame was declared.
    SillResolutionError
        A sill this grid resolves disagrees with a supplied measurement.
    UnseededWaterBodyError
        A **post-condition**, not a refusal path: ``water_mask`` is built from the
        seeded components outwards, so no component of it can be seedless and this
        cannot fire from a correct build.  It is kept, un-loosened and without a
        size threshold, so that "cannot fire" is checked on the array rather than
        argued from the code.  It also fires if a seed is somehow not in the mask.
    UnobservableConnectivityError
        ``unobservable_policy="refuse"`` and at least one component reaches the
        observation boundary below a declared level.
    CorridorNotRampableError
        ``corridor="ramp"`` with more than one spilling pair.

    WHAT IS REFUSED HERE AND WHAT IS ONLY COUNTED
    ---------------------------------------------
    Design-brief §5 ("RAISE-not-compute") item 5 asked for a **refusal** when a water
    body inside the extent holds no seed, so that 545,978 Aegean and 83,001 Levantine
    cells could not ship at an invented surface.  What that refusal was protecting
    against — a body drawn as water at a level nobody measured for it — is gone: no
    seedless body is painted as basin water on any path.  But for a while the item
    was satisfied by a SILENT DROP plus a sidecar count, which traded one substitution
    for another, and that is now split apart.  Plainly, as of this revision:

    **Refused (raises, no frame produced):**

    * a seed that does not lie in the frame's own water mask, or a seedless component
      inside ``water_mask`` itself — :class:`UnseededWaterBodyError`, the
      post-condition;
    * a component that reaches the observation boundary, *when the caller asks for it
      to be refused* — :class:`UnobservableConnectivityError`,
      ``unobservable_policy="refuse"``.

    **Counted and DRAWN AS UNRESOLVED, not refused and not dropped:**

    * every component that reaches the observation boundary under the default
      ``unobservable_policy="mark"``.  It is in ``connectivity.unobservable`` with its
      cells, km², the edges it touches and the level it sits under, it is in
      :attr:`BasinField.unobservable_mask`, and the renderer paints it in a reserved
      hatched colour with a legend entry and a caption line, so a reader of the PNG
      alone is told.  This is neither the original refusal nor a silent drop.

    **Counted only, and correctly so:**

    * closed depressions — components whose rim is inside the extent and above the
      surface they carry.  They are not water under any reading of the terrain in
      this raster, so there is nothing to refuse; ``connectivity.closed`` says how
      many and how large.

    Warns
    -----
    SmoothingShiftWarning
        The smoothing moved a sill further than *smoothing_shift_max_m*.
    """
    if corridor not in ("ramp", "step"):
        raise ValueError(f"corridor={corridor!r} is not one of 'ramp', 'step'")
    if unobservable_policy not in UNOBSERVABLE_POLICIES:
        raise ValueError(
            f"unobservable_policy={unobservable_policy!r} is not one of "
            f"{UNOBSERVABLE_POLICIES!r}. In particular there is no 'drop': a body "
            "whose connection to a seed leaves the extent is not observed to be "
            "disconnected, and drawing it as land asserts a barrier nobody measured "
            "— which is the defect this parameter exists to remove. See "
            "UnobservableConnectivityError."
        )
    seeds = list(seeds)
    if not seeds:
        raise ValueError("assign_basins needs at least one BasinSeed")
    names = [s.name for s in seeds]
    if len(set(names)) != len(names):
        raise ValueError(f"basin names must be unique, got {names!r}")

    dem = np.asarray(dem, dtype=np.float64)
    height, width = dem.shape
    seed_rc: list[tuple[int, int]] = []
    for seed in seeds:
        row, col = _seed_rowcol(transform, seed.lon, seed.lat, (height, width), seed.name)
        value = dem[row, col]
        if not np.isfinite(value):
            raise ValueError(
                f"basin seed {seed.name!r} at {seed.lon:.4f} E / {seed.lat:.4f} N "
                "lands on a nodata cell of the DEM."
            )
        if value >= seed.level_m:
            raise ValueError(
                f"basin seed {seed.name!r} sits at {value:+.2f} m, which is above "
                f"its own level {seed.level_m:+.2f} m. A seed must be under its "
                "basin's water; on dry land it grows an empty basin and the frame "
                "silently loses a sea."
            )
        seed_rc.append((row, col))
    if len(set(seed_rc)) != len(seed_rc):
        raise ValueError(
            "two basin seeds fall in the same DEM cell; they cannot be told apart"
        )

    levels = [float(s.level_m) for s in seeds]

    # ── The sill guard, before anything expensive runs ───────────────────────
    # This block used to be four lines at the bottom of the sill measurement,
    # iterating ``(sill_expectations or {}).items()`` — so the default None
    # validated nothing however many basins were in the frame.  It is now a
    # precondition: structural first (cheap, and a wrong call should not have to
    # pay for a 70-second flood to be told so), then the numeric comparison once
    # the terrain has been measured.
    pair_keys = [
        tuple(sorted((names[i], names[j])))
        for i in range(len(seeds))
        for j in range(i + 1, len(seeds))
    ]
    checks: dict[tuple[str, str], Union[SillExpectation, UnmeasuredSill]] = {}
    if pair_keys and not sill_expectations:
        raise MissingSillExpectationError(
            f"this frame has {len(seeds)} basins "
            f"({', '.join(names)}) and therefore "
            f"{len(pair_keys)} sill(s) — "
            + ", ".join(f"{a}<->{b}" for a, b in pair_keys)
            + " — but sill_expectations is "
            f"{sill_expectations!r}. The spill verdict this call returns is a "
            "property of the grid it is observed on, and a spillway narrower "
            "than the working pixel is filled in by resampling: on the "
            "production Caspian<->Pontic raster the same saddle reads +36.01 m "
            "at max_px=2000 and +28.00 m at max_px=7800, which is the "
            "difference between 'endorheic' and 'spilling'. State each sill as "
            "a SillExpectation(value_m, frame, source, measured_by, lon, lat), "
            "or declare it UnmeasuredSill(reason='...') and have the refusal "
            "recorded in the sidecar."
        )

    # THE STRUCTURE OF WHAT WAS SUPPLIED IS VALIDATED WHATEVER THE SEED COUNT.
    # This loop used to sit inside ``if pair_keys:``, so a ONE-SEED call swallowed
    # every expectation given to it — a bare float, a frame-mismatched one, a key
    # naming basins that are not in the frame — without a word.  Only the
    # COMPARISON needs a pair; a malformed expectation is malformed at any number of
    # basins, and silence about it is the same silent no-op the guard was made
    # mandatory to remove.
    for raw_pair, given in (sill_expectations or {}).items():
        try:
            shaped = not isinstance(raw_pair, str) and len(tuple(raw_pair)) == 2
        except TypeError:
            shaped = False
        if not shaped:
            raise ValueError(
                f"sill_expectations key {raw_pair!r} is not a pair of basin names. "
                "Keys are two-element tuples, e.g. ('caspian', 'pontic'); a string "
                "would be silently split into its characters by tuple(sorted(...))."
            )
        if not isinstance(given, (SillExpectation, UnmeasuredSill)):
            raise SillExpectationTypeError(
                f"sill_expectations[{tuple(raw_pair)!r}] is "
                f"{type(given).__name__} {given!r}. A bare number carries no "
                "vertical frame, no source and no place, so nothing "
                "downstream can tell it from a measured value — which is the "
                "one defect this guard exists to remove. Pass "
                "SillExpectation(value_m=..., frame=<the raster's own "
                "vertical_frame tag>, source=..., measured_by=..., lon=..., "
                "lat=...), or UnmeasuredSill(reason='...')."
            )
        key = tuple(sorted(raw_pair))
        if key not in pair_keys:
            raise ValueError(
                f"sill_expectations key {tuple(raw_pair)!r} is not a basin pair "
                f"of this call; the basins are {tuple(names)!r}."
                + (
                    " This frame has ONE basin, so it has no pairs at all and "
                    "nothing here can be compared against a saddle."
                    if len(seeds) == 1 else ""
                )
            )
        # THE SAME PAIR UNDER BOTH ORDERINGS IS TWO SUPPLIED MEASUREMENTS, AND
        # ``checks[key] = given`` KEPT WHICHEVER dict ITERATION ORDER HAPPENED TO
        # REACH LAST.  One of the two was discarded and never mentioned — a value
        # decided by iteration order passing as a measurement, which is precisely
        # the defect class this guard exists to remove.  Reproduced with
        # ('east','west') -> -999.0 and ('west','east') -> +5.0: the call returned
        # normally having compared exactly one of them.
        if key in checks:
            raise ValueError(
                f"sill_expectations names the pair {key[0]}<->{key[1]} twice, under "
                "both orderings. Two entries are two claims about the same saddle "
                "and this function will not pick one by dict iteration order: "
                f"{checks[key]!r} and {given!r}. Reconcile them into a single entry "
                "in writing."
            )
        checks[key] = given

    if pair_keys:
        missing = [k for k in pair_keys if k not in checks]
        if missing:
            raise MissingSillExpectationError(
                "sill_expectations covers "
                f"{len(checks)} of this frame's {len(pair_keys)} basin pairs; "
                + ", ".join(f"{a}<->{b}" for a, b in missing)
                + " is/are unaccounted for. A partial mapping leaves those pairs "
                "exactly as unchecked as sill_expectations=None did. Give each one "
                "a SillExpectation or an UnmeasuredSill(reason='...')."
            )

    declared_frames = {
        e.frame for e in checks.values() if isinstance(e, SillExpectation)
    }
    if declared_frames and dem_frame is None:
        raise SillFrameMismatchError(
            "sill_expectations declare the vertical frame(s) "
            f"{sorted(declared_frames)!r} but assign_basins was given no "
            "dem_frame, so the frames cannot be compared. A frame check that "
            "cannot run must not look like one that passed: comparing an F0 "
            "constant against an F1 saddle is a mistake already made once on "
            "this project, and it was wrong by exactly the dz_sill_m term "
            "(+2.701 m at 16.5 ka). Pass the DEM's own vertical_frame tag. If "
            "this call came from render_paleocostline_epoch, the input "
            "GeoTIFF(s) carry no 'vertical_frame' tag — a raster that does not "
            "say what its elevations are above cannot have them checked."
        )
    for key, expectation in checks.items():
        if isinstance(expectation, SillExpectation) and (
            expectation.frame.strip() != str(dem_frame).strip()
        ):
            raise SillFrameMismatchError(
                f"the {key[0]}<->{key[1]} expectation is in frame "
                f"{expectation.frame!r} but this DEM declares "
                f"{dem_frame!r}. The two numbers are elevations above "
                "different surfaces and their difference is not a "
                "disagreement about terrain. Convert AT THE THRESHOLD (see "
                "2026-07-30-vertical-datum-contract.md §1 and §8), then say so "
                "in SillExpectation.frame; do not normalise the two spellings "
                "here."
            )

    ceiling = float(max(levels) + ceiling_margin_m)
    basin_id, bottleneck, flood_parent = _priority_flood_ownership(dem, seed_rc, ceiling)

    # ── Sills, measured ──────────────────────────────────────────────────────
    touching = _measure_sills(basin_id, bottleneck, len(seeds))
    sill_m: dict[tuple[str, str], float] = {}
    sill_lonlat: dict[tuple[str, str], tuple[float, float] | None] = {}
    sill_route_rc: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            key = tuple(sorted((names[i], names[j])))
            found = touching.get((i + 1, j + 1))
            if found is None:
                sill_m[key] = float("nan")
                sill_lonlat[key] = None
                sill_route_rc[key] = ()
            else:
                value, (row, col), partner = found
                sill_m[key] = value
                lon, lat = transform * (col + 0.5, row + 0.5)
                sill_lonlat[key] = (float(lon), float(lat))
                sill_route_rc[key] = _minimax_route(flood_parent, (row, col), partner)
    del flood_parent

    # ── Sills, checked against what somebody measured ────────────────────────
    sill_expected_m: dict[tuple[str, str], Optional[float]] = {}
    sill_status: dict[tuple[str, str], str] = {}
    for key, expectation in checks.items():
        measured = sill_m[key]
        if isinstance(expectation, UnmeasuredSill):
            sill_expected_m[key] = None
            sill_status[key] = f"UNMEASURED: {expectation.reason}"
            continue
        expected = float(expectation.value_m)
        sill_expected_m[key] = expected
        if not np.isfinite(measured):
            raise SillResolutionError(
                f"a sill of {expected:+.2f} m ({expectation.source}) was supplied "
                f"for {key[0]}<->{key[1]}, but their territories never met: the "
                f"saddle stands above this call's flood ceiling of "
                f"{ceiling:+.2f} m. Raise ceiling_margin_m rather than reading "
                "'not measured' as 'agrees'."
            )
        gap = measured - expected
        if abs(gap) > sill_tolerance_m:
            raise SillResolutionError(
                f"this grid resolves the {key[0]}<->{key[1]} sill at "
                f"{measured:+.2f} m, but it was measured at {expected:+.2f} m "
                f"({expectation.source}; {expectation.measured_by}; "
                f"{expectation.lon:.4f} E / {expectation.lat:.4f} N; frame "
                f"{expectation.frame}) — a gap of {gap:+.2f} m, wider than the "
                f"{sill_tolerance_m:.2f} m tolerance. A spillway narrower than the "
                "working pixel is filled in by resampling and the sill rises, so a "
                "frame built on this grid can report a basin as endorheic when the "
                "terrain spills. THIS IS A REFUSAL, NOT A SETTING: on the "
                "production Caspian<->Pontic raster the connected-path saddle on "
                "dem_sm reads +36.01 m at max_px=2000 (gap +8.30), +32.94 at 4000 "
                "(gap +5.24) and +28.00 at 7800 (gap +0.30), so that frame is "
                "renderable only on the native grid. Nothing here raises max_px "
                "for you, and no minimum-max_px precondition is enforced, because "
                "resolution is NOT monotone in sill fidelity — the point-cell "
                "reading goes +27/+24/+25/+25/+26/+23 m at 1000/2000/3000/4000/"
                "6000/native px, which is bilinear phase, not a resolution cost. "
                "Render on a grid that resolves the channel, or widen "
                "sill_tolerance_m in writing."
            )
        sill_status[key] = (
            f"CHECKED: |{measured:+.2f} - {expected:+.2f}| = {abs(gap):.2f} m "
            f"<= {sill_tolerance_m:.2f} m; expectation from {expectation.source} "
            f"({expectation.measured_by}) at {expectation.lon:.4f} E / "
            f"{expectation.lat:.4f} N in frame {expectation.frame}"
        )

    # ── What the smoothing did to those sills ────────────────────────────────
    sill_shift: dict[tuple[str, str], Optional[float]] = dict.fromkeys(sill_m, None)
    if unsmoothed_dem is not None:
        raw = np.asarray(unsmoothed_dem, dtype=np.float64)
        if raw.shape != dem.shape:
            raise ValueError(
                f"unsmoothed_dem has shape {raw.shape} but the DEM being assigned "
                f"is {dem.shape}; they must be the same grid or the difference is "
                "not a smoothing shift."
            )
        raw_id, raw_bottleneck, _raw_parent = _priority_flood_ownership(
            raw, seed_rc, ceiling
        )
        del _raw_parent
        raw_touching = _measure_sills(raw_id, raw_bottleneck, len(seeds))
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                key = tuple(sorted((names[i], names[j])))
                found = raw_touching.get((i + 1, j + 1))
                if found is None or not np.isfinite(sill_m[key]):
                    continue
                shift = float(sill_m[key] - found[0])
                sill_shift[key] = shift
                if abs(shift) > smoothing_shift_max_m:
                    warnings.warn(
                        f"the mask smoothing moved the {key[0]}<->{key[1]} sill "
                        f"from {found[0]:+.2f} m on the unsmoothed grid to "
                        f"{sill_m[key]:+.2f} m on the array the mask is built "
                        f"from — {shift:+.2f} m, more than the "
                        f"{smoothing_shift_max_m:.2f} m allowed by "
                        "smoothing_shift_max_m. The shift is REPORTED, never "
                        "subtracted: a modelled sigma-correction would replace a "
                        "number taken off the terrain with one taken off an "
                        "assumed crest shape. Measured on khv_16.5ka.tif it is "
                        "+3.30 m at max_px=2000, +5.04 m at 4000 and +0.25 m at "
                        "7800 — i.e. it is a resolution symptom, and the answer is "
                        "a grid that resolves the channel.",
                        SmoothingShiftWarning,
                        stacklevel=2,
                    )
        del raw, raw_id, raw_bottleneck

    # ── Spill, observed ──────────────────────────────────────────────────────
    # Basin A spills into basin B iff there is a path from A's seed to B's seed on
    # which EVERY cell lies below A's OWN water surface — a flood at the DONOR's
    # level — and B is the FIRST basin that path reaches.
    #
    # This replaces a pairwise ``dem < max(level_i, level_j)`` test that
    # manufactured connections the terrain does not have.  Flooding to the pairwise
    # max floods every basin in BETWEEN to the donor's level as well, so a Caspian
    # at +35 m "reached" the Mediterranean straight through a Pontic basin standing
    # 65 m lower — regardless of what the Bosphorus sill does, because the Pontic
    # had been filled to +35 m by fiat.  The three-basin hero frame is exactly that
    # configuration: it reported two spilling pairs and ``corridor="ramp"`` refused
    # to draw it.
    #
    # Water that crosses into a third basin's territory joins THAT basin's water
    # body, standing at THAT basin's level; whether it travels any further is the
    # third basin's own donor test, asked separately in this same loop.  So a path
    # may cross only the donor's territory, the receiver's, and ground no basin
    # drains (the corridor crest itself).  Transitive reach is then rebuilt by the
    # union-find below, which is where it belongs: A→C and C→D put A, C and D in one
    # group without ever claiming A→D on C's behalf.
    #
    # The result cannot depend on the order *seeds* was given in: every ordered pair
    # is asked, and the answer for (i, j) is a property of the terrain, the two
    # levels and the third-party territories — none of which is an ordering.
    component_cache: dict[float, np.ndarray] = {}

    def components(level: float) -> np.ndarray:
        cached = component_cache.get(level)
        if cached is None:
            cached, _ = ndi_label(np.isfinite(dem) & (dem < level))
            component_cache[level] = cached
        return cached

    def donor_reach(i: int) -> np.ndarray:
        """Everything a flood at basin *i*'s own level wets, starting from its seed."""
        labelled = components(levels[i])
        # The seed is under its own water (validated above), so its label is > 0.
        return labelled == labelled[seed_rc[i]]

    spills: list[tuple[int, int]] = []           # directed (donor, receiver)
    for i in range(len(seeds)):
        reach = donor_reach(i)
        for j in range(len(seeds)):
            if i == j or not reach[seed_rc[j]]:
                continue
            third_party = np.zeros(dem.shape, dtype=bool)
            for k in range(len(seeds)):
                if k != i and k != j:
                    third_party |= basin_id == k + 1
            if third_party.any():
                direct, _ = ndi_label(reach & ~third_party)
                home = direct[seed_rc[i]]
                if home == 0 or direct[seed_rc[j]] != home:
                    continue                     # only reachable THROUGH a third basin
            spills.append((i, j))

    merged_pairs: list[tuple[int, int]] = sorted(
        {(min(i, j), max(i, j)) for i, j in spills}
    )

    parent = list(range(len(seeds)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merged_pairs:
        parent[find(i)] = find(j)
    grouped: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        grouped.setdefault(find(i), []).append(name)
    connected_groups = tuple(sorted(tuple(sorted(g)) for g in grouped.values()))

    # ── Water surface ────────────────────────────────────────────────────────
    level_lookup = np.concatenate(([np.nan], np.asarray(levels, dtype=np.float64)))
    water_level = level_lookup[basin_id]
    corridor_mask = np.zeros(dem.shape, dtype=bool)
    corridor_mode = "none"
    corridor_donor: Optional[int] = None
    corridor_receiver: Optional[int] = None

    if merged_pairs:
        if corridor == "ramp" and len(merged_pairs) > 1:
            spelled = ", ".join(f"{names[i]}<->{names[j]}" for i, j in merged_pairs)
            raise CorridorNotRampableError(
                f"{len(merged_pairs)} basin pairs are connected at this epoch "
                f"({spelled}). The linear ramp is defined for ONE donor->receiver "
                "corridor; with several the surface in each depends on how the "
                "discharge divides, which this module does not model. Pass "
                "corridor='step' to render every basin flat at its own level with "
                "the corridors marked."
            )
        i, j = merged_pairs[0]
        donor, receiver = (i, j) if levels[i] >= levels[j] else (j, i)
        if levels[donor] > levels[receiver]:
            spill_labels = components(levels[donor])
            extent = spill_labels == spill_labels[seed_rc[donor]]
            donor_water = extent & (basin_id == donor + 1)
            receiver_labels = components(levels[receiver])
            receiver_water = receiver_labels == receiver_labels[seed_rc[receiver]]
            candidates = extent & ~donor_water & ~receiver_water
            # A basin that is neither donor nor receiver keeps its OWN flat surface.
            # ``extent`` is a flood at the DONOR's level and reaches straight across
            # a third basin's territory; interpolating a corridor level there would
            # repaint that basin with a number belonging to two other seas — in the
            # hero frame, lifting a Mediterranean at −45 m onto the Caspian→Pontic
            # ramp.  Whether the third basin is itself connected is its own donor
            # test, already asked above.
            for other in range(len(seeds)):
                if other != donor and other != receiver:
                    candidates &= basin_id != other + 1

            if candidates.any():
                corridor_donor, corridor_receiver = donor, receiver
                if corridor == "step":
                    corridor_mode = "step"
                    corridor_mask = candidates
                else:
                    corridor_mode = "ramp"
                    d_donor = _geodesic_hops(candidates, donor_water)
                    d_receiver = _geodesic_hops(candidates, receiver_water)
                    reach_d = np.isfinite(d_donor)
                    reach_r = np.isfinite(d_receiver)
                    both = reach_d & reach_r
                    frac = np.full(dem.shape, np.nan)
                    frac[both] = d_donor[both] / (d_donor[both] + d_receiver[both])
                    # A pocket that only the donor can reach is a BAY of the donor
                    # lake and stands at the donor's level; one only the receiver
                    # can reach never saw the spill at all.
                    frac[reach_d & ~reach_r] = 0.0
                    frac[~reach_d & reach_r] = 1.0
                    ramp = levels[donor] + (levels[receiver] - levels[donor]) * frac
                    ramped = candidates & np.isfinite(frac)
                    water_level = np.where(ramped, ramp, water_level)
                    corridor_mask = ramped & (dem < water_level)

    # ── Water MASK, hydrologically connected ─────────────────────────────────
    # ``flat`` is the test this module shipped until 2026-08-04 — every cell below
    # the surface IT carries, with no connectivity condition — and it is kept here
    # only as the thing the connected mask is MEASURED AGAINST.  It is not the mask.
    # See _seeded_water_components for the definition and for what "connected"
    # means in each corridor mode.
    flat = np.isfinite(dem) & (dem < water_level)
    dem_nodata = ~np.isfinite(dem) if nodata is None else np.asarray(nodata, dtype=bool)
    if dem_nodata.shape != dem.shape:
        raise ValueError(
            f"nodata has shape {dem_nodata.shape} but the DEM being assigned is "
            f"{dem.shape}; the observation boundary must be measured on the same "
            "grid as the water."
        )
    states = _seeded_water_components(
        flat, water_level, seed_rc, transform, dem_nodata, basin_id, names
    )
    water_mask = states["connected_mask"]
    unobservable_mask = states["unobservable_mask"]
    seed_labels = states["seed_labels"]
    flat_labels = states["labels"]

    # ── The third state, refused only if the caller asked for it ─────────────
    if states["unobservable"] and unobservable_policy == "refuse":
        raise UnobservableConnectivityError(
            _unobservable_refusal(states["unobservable"], names, levels)
        )

    # A seed that is not itself in the mask would make the whole construction
    # vacuous (nothing would be kept for that basin).  It cannot happen — the seed
    # was validated to stand below its own level, its cell is owned by itself so it
    # carries its own level, and no seed is ever inside the ramped corridor — but
    # "cannot happen" is what the flat mask's 179 refused components were too.
    dry = [names[i] for i, value in enumerate(seed_labels) if value == 0]
    if dry:
        raise UnseededWaterBodyError(
            f"basin seed(s) {', '.join(dry)} do not lie in this frame's own water "
            "mask, so the connected-component construction has no origin for them "
            "and every cell of those basins would be dropped. A seed is validated "
            "to stand below its own level before the flood runs, so this is a bug "
            "in the water-surface assembly, not a missing level."
        )

    # A ramped corridor cell that the connectivity pass dropped is a STRANDED
    # POCKET: below the interpolated profile, but cut off from both the donor's and
    # the receiver's water by a stretch of bed standing above it.  It must leave
    # ``corridor_mask`` as well, because the renderer restores that mask verbatim
    # into the drawn ocean in ramp mode ("every one of its cells is wet by
    # construction") — an OR that would put every stranded pocket straight back.
    corridor_before = int(corridor_mask.sum())
    if corridor_mode == "ramp":
        corridor_mask = corridor_mask & water_mask

    connectivity = _connectivity_report(
        flat=flat,
        states=states,
        basin_id=basin_id,
        names=names,
        transform=transform,
        corridor_before=corridor_before,
        corridor_after=int(corridor_mask.sum()),
        corridor_mode=corridor_mode,
        unobservable_policy=unobservable_policy,
    )
    # What painting the corridor WOULD cost, measured on THIS frame's own corridor
    # mask.  The step-mode disclosure quotes these figures, and it quoted them from
    # a constant embedded in a template until 2026-08-04 — a constant measured on a
    # different frame, which the same JSON object contradicted.
    connectivity["corridor_if_painted"] = _corridor_cost(
        corridor_mask=corridor_mask,
        water_mask=water_mask,
        dem=dem,
        transform=transform,
        donor=corridor_donor,
        receiver=corridor_receiver,
        names=names,
        levels=levels,
    )
    del flat, flat_labels, states

    # ── POST-CONDITION: every water body holds a seed ────────────────────────
    # This used to be a REFUSAL on the flat mask and it fired 179 times on the
    # production frame, which is why no three-basin frame rendered.  The refusal was
    # right and the mask was wrong; now the mask is built connected, so this can no
    # longer fire from here — and it is re-checked, on the real array, rather than
    # argued from the construction.  It is NOT loosened, NOT size-thresholded and
    # NOT given a margin: if it fires, _seeded_water_components is broken and the
    # frame must not ship.
    _n_kept, _kept_seed_labels, unseeded, _kept_labels = _unseeded_water_bodies(
        water_mask, water_level, seed_rc, transform
    )
    del _kept_labels
    if unseeded:
        raise UnseededWaterBodyError(
            _unseeded_water_refusal(
                unseeded, names, levels,
                mask_description=(
                    "assign_basins' own water_mask (the SEEDED CONNECTED components "
                    "of dem < water_level_m, before the renderer's morphological "
                    "de-speckle)"
                ),
            )
        )

    return BasinField(
        basin_id=basin_id,
        names=tuple(names),
        levels_m=tuple(levels),
        bottleneck_m=bottleneck,
        water_level_m=water_level,
        water_mask=water_mask,
        corridor_mask=corridor_mask,
        spilling=bool(merged_pairs),
        connected_groups=connected_groups,
        unobservable_mask=unobservable_mask,
        sill_m=sill_m,
        sill_lonlat=sill_lonlat,
        corridor_mode=corridor_mode,
        ceiling_m=ceiling,
        sill_expected_m=sill_expected_m,
        sill_expectation_status=sill_status,
        sill_smoothing_shift_m=sill_shift,
        sill_route_rc=sill_route_rc,
        connectivity=connectivity,
    )


def interior_tile_seams(
    tile_bounds: Sequence[BoundingBox],
    mosaic_bounds: BoundingBox,
    tolerance: float,
) -> tuple[list[float], list[float]]:
    """Where the tiles of a mosaic actually meet: (seam longitudes, seam latitudes).

    A seam is an edge shared by two tiles, i.e. a tile edge that falls in the
    *interior* of the merged extent.  A tile edge coinciding with the outside of
    the mosaic is not a seam — there is nothing on the far side of it to blend to.
    A one-file mosaic therefore has no seams at all, whatever its extent.

    This replaces "every whole degree line crossing the mosaic", which was a proxy
    for the tile grid that is wrong in both directions: it blurs interior terrain
    when the tiles are bigger than a degree (or when there is only one), and it
    misses a genuine seam that does not land on a whole degree.

    *tolerance* is the distance (in CRS units) within which an edge counts as
    lying ON the mosaic boundary rather than inside it; pass one output pixel.
    """
    lons: set[float] = set()
    lats: set[float] = set()
    for b in tile_bounds:
        for x in (float(b.left), float(b.right)):
            if mosaic_bounds.left + tolerance < x < mosaic_bounds.right - tolerance:
                lons.add(round(x, 9))
        for y in (float(b.bottom), float(b.top)):
            if mosaic_bounds.bottom + tolerance < y < mosaic_bounds.top - tolerance:
                lats.add(round(y, 9))
    return sorted(lons), sorted(lats)


def seam_blend_shift_m(
    raw: np.ndarray,
    blended: np.ndarray,
    transform: rasterio.transform.Affine,
) -> tuple[float, tuple[float, float]]:
    """Largest elevation the blend moved, and where — ``(metres, (lon, lat))``.

    A smoothing step is not neutral on a spillway: a Gaussian applied across a
    channel only a few working pixels wide pulls the shoulders into the floor and
    the crest goes UP, which is the direction that turns a spilling basin
    endorheic.  So the blend is measured against the surface it was handed, and
    the number is available rather than inferred later from a wrong picture.
    """
    both = np.isfinite(raw) & np.isfinite(blended)
    shift = np.where(both, np.abs(blended - raw), 0.0)
    if not shift.size:
        return 0.0, (float("nan"), float("nan"))
    row, col = np.unravel_index(int(np.argmax(shift)), shift.shape)
    lon, lat = transform * (int(col) + 0.5, int(row) + 0.5)
    return float(shift[row, col]), (float(lon), float(lat))


def declared_vertical_frame(tile_paths: dict[str, str | Path]) -> Optional[str]:
    """The vertical frame the input rasters declare, verbatim, or ``None``.

    Read from the GeoTIFF tag ``vertical_frame``.  ``khv_16.5ka.tif`` carries the
    literal string ``"F0 present-day datum (H_modern + dz_solid)"``; it is returned
    unchanged and compared unchanged, because normalising two spellings of a datum
    into one is precisely the step at which an F0/F1 confusion stops being visible.

    ``None`` means **every** raster says nothing, which is not the same as F0 and is
    not treated as agreement: :func:`assign_basins` refuses a
    :class:`SillExpectation` it cannot frame-check.

    SILENCE FROM *SOME* TILES IS NOT AGREEMENT EITHER
    --------------------------------------------------
    Until 2026-08-03 the loop skipped an untagged tile and skipped a tile whose path
    does not exist, then returned the one tag it did find — so a mosaic in which one
    tile of four declares ``"F0 present-day datum (H_modern + dz_solid)"`` reported
    that frame *for the whole merged surface*, and a typo'd path was worth exactly as
    much as agreement.  That is the same defect as returning ``None`` for a tagged
    raster, with the sign reversed, and it contradicted this docstring's own promise.

    So: a mixed mosaic **raises**.  A frame is a property of the merged surface, and
    a surface assembled from one tile that names its datum and three that do not is
    in no stated frame at all.  Tag the other tiles, or drop them.

    An unreadable path counts as an undeclared tile for the same reason: the merge
    (:func:`build_epoch_mosaic`) silently drops non-existent inputs, so a mosaic
    built from a dict with a bad path is a *different* mosaic from the one the caller
    described, and certifying its frame would certify something nobody asked for.
    When NOTHING declares a frame the answer is still ``None`` — there is no claim to
    contradict, and the caller is refused downstream instead.

    Raises
    ------
    ValueError
        The tiles of one mosaic declare DIFFERENT frames (merging them would produce
        a surface that is in neither), or SOME declare a frame while others are
        silent or missing.
    """
    frames: dict[str, str] = {}
    silent: list[str] = []
    missing: list[str] = []
    for path in tile_paths.values():
        candidate = Path(path)
        if not candidate.exists():
            missing.append(str(candidate))
            continue
        with rasterio.open(candidate) as ds:
            tag = ds.tags().get("vertical_frame")
        if tag:
            frames[str(candidate)] = str(tag)
        else:
            silent.append(str(candidate))
    distinct = set(frames.values())
    if len(distinct) > 1:
        spelled = "; ".join(f"{k} -> {v!r}" for k, v in sorted(frames.items()))
        raise ValueError(
            "the tiles of this mosaic declare different vertical_frame tags "
            f"({spelled}). Merging them gives a surface that is in neither frame, "
            "and every elevation compared against it afterwards is wrong by the "
            "difference. Re-bake the tiles into one frame."
        )
    if distinct and (silent or missing):
        spelled = ", ".join(f"{k} -> {v!r}" for k, v in sorted(frames.items()))
        undeclared = "; ".join(
            part for part in (
                ("declares nothing: " + ", ".join(sorted(silent))) if silent else "",
                ("does not exist: " + ", ".join(sorted(missing))) if missing else "",
            ) if part
        )
        raise ValueError(
            f"{len(frames)} of this mosaic's {len(frames) + len(silent) + len(missing)} "
            f"tiles declare a vertical frame ({spelled}) and the rest do not "
            f"({undeclared}). Returning the tagged tile's frame for the whole merged "
            "surface would treat SILENCE AS AGREEMENT: the merge is one array, its "
            "frame is a property of every cell in it, and an untagged tile states "
            "nothing about the datum of the cells it contributes. A path that does "
            "not exist is worse than silent — build_epoch_mosaic drops it, so the "
            "mosaic that gets built is not the one described here. Tag every tile, "
            "or pass only the tiles you can vouch for."
        )
    return distinct.pop() if distinct else None


def build_epoch_mosaic(
    tile_paths: dict[str, str | Path],
    config: Optional[RenderConfig] = None,
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, BoundingBox, object, float]:
    """Merge per-tile GeoTIFFs into a single down-sampled mosaic array.

    Seams — and only seams
    ----------------------
    Where two tiles meet, the merge leaves a step, and a 1-D Gaussian across the
    shared edge hides it.  The seams are taken from the **tiles' own bounds**
    (:func:`interior_tile_seams`).  Two consequences that were not true before:

    * a **single-file input is never blended**, at any σ.  It has no seams, so
      there is nothing to hide, and blurring it is pure interior damage.  Both
      bakes in this repo pass exactly one file.
    * a genuine seam that does not land on a whole degree is still blended.

    What that was costing.  Until 2026-08 the strips ran along every whole degree
    line crossing the mosaic, tiles or no tiles.  Measured on the real
    Caspian↔Pontic working raster (a single 15-arcsec file, 25.5–58 E / 36–54 N):

    ==========================  ==========  ==========  =====================
    setting                     Manych raw  → blended   cells moved >1 cm
    ==========================  ==========  ==========  =====================
    ``max_px=2000``               +24.00 m    +40.76 m  75 % of the mosaic
    ``max_px=4000``               +25.00 m    +30.57 m  44 % of the mosaic
    ==========================  ==========  ==========  =====================

    A +16.76 m rise at the Manych point sill turns a Caspian spilling over
    +23.00 m at a +35 m highstand into an endorheic lake, silently, in the
    direction that deletes the subject of the animation.

    Never silently
    --------------
    Whatever blending does happen is measured against the unblended merge
    (:func:`seam_blend_shift_m`) and reported as a :class:`SeamBlendShiftWarning`
    as soon as it exceeds ``config.seam_blend_max_shift_m``, which defaults to
    :data:`SILL_TOLERANCE_M` — see that constant for the derivation, which is a
    spill-window duration and not a number borrowed from :func:`assign_basins`
    (that citation used to run in a circle between the two).  A smoothing step that
    can move a crest must say by how much and where.

    Parameters
    ----------
    tile_paths : dict[str, str | Path]
        Dict whose *values* are paths to paleo-DEM GeoTIFFs.  Ice-sheet side-car
        files are discovered automatically as ``<tile>_ice.tif``.
    config : RenderConfig | None
        Render configuration.  Defaults to :class:`RenderConfig` with all
        defaults if ``None``.  ``seam_sigma_px=0`` disables the blend outright;
        ``legacy_degree_grid_blend=True`` restores the whole-degree behaviour
        described above, for reproducing frames baked before it was fixed.

    Returns
    -------
    dem : np.ndarray[float32], shape (H, W)
        Merged DEM in metres (``np.nan`` for nodata).
    ice : np.ndarray[float32], shape (H, W)
        Ice-thickness array in metres (zeros where no sidecar exists).
    transform : rasterio.transform.Affine
        Affine transform of the mosaic.
    bounds : rasterio.coords.BoundingBox
        Geographic extent of the mosaic.
    crs
        CRS of the mosaic.
    resolution : float
        Pixel size in degrees.

    Raises
    ------
    FileNotFoundError
        If no valid tile files are found.
    ValueError
        If ``config.seam_sigma_px`` is negative.

    Warns
    -----
    SeamBlendShiftWarning
        The blend moved the terrain further than ``config.seam_blend_max_shift_m``.
    """
    cfg = config or RenderConfig()
    sig = float(cfg.seam_sigma_px)
    if sig < 0.0:
        raise ValueError(
            f"seam_sigma_px={sig!r} is negative. A Gaussian σ is a width, not a "
            "direction; pass 0 to switch the seam blend off."
        )

    valid = [Path(f) for f in tile_paths.values() if Path(f).exists()]
    if not valid:
        raise FileNotFoundError("No valid tiles found in tile_paths.")

    dem_ds = [rasterio.open(f) for f in valid]
    try:
        crs = dem_ds[0].crs
        tile_bounds = [d.bounds for d in dem_ds]
        west = min(d.bounds.left for d in dem_ds)
        east = max(d.bounds.right for d in dem_ds)
        south = min(d.bounds.bottom for d in dem_ds)
        north = max(d.bounds.top for d in dem_ds)
        bounds = BoundingBox(west, south, east, north)

        nat_res = dem_ds[0].res[0]
        n_px = max((east - west) / nat_res, (north - south) / nat_res)
        step = max(1.0, n_px / cfg.max_px)
        tgt_res = nat_res * step

        mosaic, xform = rasterio_merge(
            dem_ds, res=tgt_res, resampling=RIOResampling.bilinear, nodata=-9999.0
        )
    finally:
        for d in dem_ds:
            d.close()

    mosaic = mosaic[0].astype(np.float32)
    mosaic[mosaic <= -9_000.0] = np.nan
    H, W = mosaic.shape

    # ── ICE sidecars ────────────────────────────────────────────────────────
    ice_paths = [
        Path(str(f).replace(".tif", "_ice.tif"))
        for f in valid
        if Path(str(f).replace(".tif", "_ice.tif")).exists()
    ]
    if ice_paths:
        ice_ds = [rasterio.open(f) for f in ice_paths]
        try:
            ice_m, _ = rasterio_merge(
                ice_ds, res=tgt_res, resampling=RIOResampling.bilinear, nodata=0.0
            )
        finally:
            for d in ice_ds:
                d.close()
        ice = np.clip(ice_m[0].astype(np.float32), 0.0, None)
    else:
        ice = np.zeros((H, W), dtype=np.float32)

    # ── Tile-seam Gaussian blending ─────────────────────────────────────────
    # Only across edges two tiles actually share.  ``legacy_degree_grid_blend``
    # restores the whole-degree grid this replaced — see the docstring for the
    # +16.76 m it moved the Manych sill by, and use it only to reproduce a frame
    # baked before the fix.
    if cfg.legacy_degree_grid_blend:
        seam_lons = [float(v) for v in range(int(np.ceil(west)), int(east) + 1)]
        seam_lats = [float(v) for v in range(int(np.ceil(south)), int(north) + 1)]
    else:
        seam_lons, seam_lats = interior_tile_seams(tile_bounds, bounds, tgt_res)

    if sig > 0.0 and (seam_lons or seam_lats):
        hw = int(np.ceil(sig * 3))
        dem_s = mosaic.copy()
        nd = np.isnan(dem_s)
        dem_s[nd] = 0.0

        for lat in seam_lats:
            r = int(round((north - lat) / tgt_res))
            r0, r1 = max(0, r - hw), min(H, r + hw + 1)
            if r0 < r1:
                dem_s[r0:r1, :] = gaussian_filter1d(dem_s[r0:r1, :], sigma=sig, axis=0)

        for lon in seam_lons:
            c = int(round((lon - west) / tgt_res))
            c0, c1 = max(0, c - hw), min(W, c + hw + 1)
            if c0 < c1:
                dem_s[:, c0:c1] = gaussian_filter1d(dem_s[:, c0:c1], sigma=sig, axis=1)

        blended = np.where(nd, np.nan, dem_s)

        # Measured, then said out loud.  A blend that can move a sill must never
        # do it in silence — the alternative is discovering it in a frame that
        # reports a spilling basin as endorheic.
        shift, (shift_lon, shift_lat) = seam_blend_shift_m(mosaic, blended, xform)
        if shift > cfg.seam_blend_max_shift_m:
            warnings.warn(
                f"tile-seam blend (σ={sig:g} px, "
                f"{len(seam_lons)} lon / {len(seam_lats)} lat seam"
                f"{'' if len(seam_lons) + len(seam_lats) == 1 else 's'}) moved the "
                f"terrain by up to {shift:.2f} m at "
                f"{shift_lon:.4f} E / {shift_lat:.4f} N, more than the "
                f"{cfg.seam_blend_max_shift_m:.2f} m allowed by "
                "seam_blend_max_shift_m. Smoothing across a channel narrower than a "
                "few working pixels RAISES its crest, which is the direction that "
                "turns a spilling basin endorheic. Check any sill within "
                f"{hw} px of a seam before trusting this frame's spill verdict.",
                SeamBlendShiftWarning,
                stacklevel=2,
            )

        mosaic = blended
        del dem_s
        gc.collect()

    return mosaic, ice, xform, bounds, crs, tgt_res


def render_paleocostline_epoch(
    epoch_ka: float,
    tile_paths: dict[str, str | Path],
    sea_level_csv: Path,
    paleocoastlines_shp: Path,
    output_dir: Path,
    config: Optional[RenderConfig] = None,
    *,
    data_source_label: str = "fusion",
    preloaded_coast_gdf: Optional[gpd.GeoDataFrame] = None,
    sea_level_override_m: Optional[float] = None,
    force_dem_mask: bool = False,
    basin_seeds: Optional[Sequence[BasinSeed]] = None,
    basin_ceiling_margin_m: float = 100.0,
    basin_corridor: str = "ramp",
    basin_sill_expectations: Optional[dict] = None,
    basin_sill_tolerance_m: float = SILL_TOLERANCE_M,
    basin_unobservable_policy: str = "mark",
) -> Path:
    """Render a composite paleo-topography image for one epoch.

    This is the top-level convenience function.  For batch rendering multiple
    epochs, call :func:`load_paleocostlines` once and pass the result via
    *preloaded_coast_gdf* to avoid re-reading the shapefile on each epoch.

    Parameters
    ----------
    epoch_ka : float
        Epoch to render in ka BP (e.g. ``21.0``).
    tile_paths : dict[str, str | Path]
        Dict mapping tile IDs to file paths of paleo-DEM GeoTIFFs for this
        epoch.  Ice-sheet sidecars (``<stem>_ice.tif``) are auto-detected.
    sea_level_csv : Path
        Path to ``spratt_lisiecki_2016_simplified.csv``.
    paleocoastlines_shp : Path
        Path to ``Paleocoastlines.shp``.
    output_dir : Path
        Directory where the output PNG will be written.
    config : RenderConfig | None
        Render parameters.  Uses defaults if ``None``.
    data_source_label : str
        Short label embedded in the figure title (e.g. ``"fusion"``).
    preloaded_coast_gdf : gpd.GeoDataFrame | None
        Pre-loaded Paleocoastlines GeoDataFrame (from a previous call to
        :func:`load_paleocostlines`).  If ``None`` the shapefile is loaded
        inside this call.
    sea_level_override_m : float | None
        Per-basin sea-level override in metres vs present global sea level
        (negative = below today).  When ``None`` (default) the global
        Spratt & Lisiecki (2016) value is used — identical, backward-compatible
        behaviour.  When given, this value (NOT the Spratt value) selects the
        coastline step via :func:`filter_paleocostlines_for_epoch`, and the
        override is recorded in the figure title / ``mask_src`` label so the
        frame is self-documenting.

        This exists because a sill-dammed basin (e.g. the Black Sea, an
        isolated lake until ~9 ka) must NOT inherit the global eustatic curve
        inside the decoupled basin — it has its own cited level curve.  Pass
        that basin's level for *epoch_ka* here.
    force_dem_mask : bool
        When True, skip the vector Paleocoastlines and derive land/sea directly
        from the paleo-DEM against ``selection_sl`` (``dem < selection_sl``).  For a
        decoupled basin this makes the shoreline follow the cited level continuously
        instead of snapping to the coarse global-eustatic vector steps (which barely
        move across a -30 to -115 m sweep and mix in the wrong open-ocean basin).
    basin_seeds : sequence of BasinSeed | None
        **Opt-in multi-basin path.**  ``None`` (default) leaves every line of the
        single-level behaviour above exactly as it was.  When given, each cell is
        thresholded against ITS OWN basin's level instead of one scalar, so two
        basins standing at different levels — the Caspian near +35 m and the Pontic
        near −30 m — can share one frame, with a ramped water surface across the
        spillway between them.  See :func:`assign_basins` for how membership, the
        sill and the spill are derived from the terrain, and for what the corridor
        ramp is and is not.

        Levels must already be in the paleo-DEM's vertical frame (F1).  This is the
        DEM-driven path by construction: the vector Paleocoastlines carry one global
        step per polygon and cannot express two surfaces, so they are not consulted.
        Mutually exclusive with *sea_level_override_m*.

        A machine-readable sidecar ``<png stem>.basins.json`` is written next to the
        PNG recording the levels, the measured sill(s), and whether the basins were
        observed to be spilling — the animation needs that per frame, and a figure
        is not a data format.
    basin_ceiling_margin_m : float
        Passed through to :func:`assign_basins`.
    basin_corridor : {"ramp", "step"}
        Passed through to :func:`assign_basins`.
    basin_sill_expectations, basin_sill_tolerance_m
        Passed through to :func:`assign_basins`, where the expectations are
        **mandatory** as soon as *basin_seeds* holds more than one basin.  The
        mosaic this function builds is down-sampled and gaussian-smoothed, and a
        channel narrower than the resulting pixel is filled in — see
        :class:`SillResolutionError` for the measured cost (+8.30 m at
        ``max_px=2000`` on the Caspian↔Pontic raster, which is the whole spill
        verdict) and :func:`build_epoch_mosaic` for the seam blend that used to add
        another +16.76 m on top.

        The vertical frame each expectation is checked against is read from the
        input GeoTIFF's own ``vertical_frame`` tag by
        :func:`declared_vertical_frame`; a raster that does not declare one cannot
        have a :class:`SillExpectation` checked against it, and says so.
    basin_unobservable_policy : {"mark", "refuse"}
        Passed through to :func:`assign_basins`.  ``"mark"`` (default) draws every
        water body whose connectivity this extent cannot decide in a reserved
        hatched colour, names it in the legend and the caption, and records its
        cells, area, touched edges and level in ``basins.json``; ``"refuse"`` raises
        :class:`UnobservableConnectivityError` instead of producing a frame.  There
        is no policy that drops such a body — dropping it draws open sea as land.

    Returns
    -------
    Path
        Absolute path of the saved PNG file.

    Raises
    ------
    FileNotFoundError
        If no valid tiles are found for this epoch.
    ValueError
        If both *basin_seeds* and *sea_level_override_m* are given.
    SequenceLimitsEpochError, SequenceLimitsRasterError, SequenceLimitsRecipeError
        ``config.sequence_limits`` is set and does not describe this frame: the
        epoch is not one it was computed over, a raster is absent from it or has
        been re-baked under the same name, the percentile / ``vert_exag`` recipe
        disagrees, or the frame has ice the limits pass never saw.  All four are
        checked BEFORE the mosaic is composited.
    SequenceLimitsExceededError
        ``config.clip_policy == "raise"`` and the frame holds cells outside the
        envelope the limits recorded measuring.  Raised before the PNG is written.

    Notes
    -----
    With ``config.sequence_limits`` set, a second sidecar
    ``<png stem>.palette.json`` is written beside the PNG carrying the mapping, its
    provenance, the out-of-range counts and the ramp saturation counts.  Without it
    nothing changes: every colour limit is derived from this frame's own data, as it
    always was.
    """
    cfg = config or RenderConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sea level & coastline ────────────────────────────────────────────────
    # The global Spratt & Lisiecki value is always computed (it is the default
    # and remains the basis for the global-ocean datum).  A per-basin override,
    # when supplied, REPLACES it for the coastline-step selection only — this is
    # how a sill-dammed basin (e.g. the Black Sea, isolated until ~9 ka) is
    # driven by its own cited level curve instead of the global eustatic curve.
    if basin_seeds is not None and sea_level_override_m is not None:
        raise ValueError(
            "render_paleocostline_epoch got BOTH basin_seeds and "
            "sea_level_override_m. They are two different answers to 'what is the "
            "water surface here' and there is no defensible way to combine them: "
            "put every basin's level in its own BasinSeed."
        )

    spratt_sl = get_sea_level_for_epoch(epoch_ka, sea_level_csv)

    if sea_level_override_m is not None:
        selection_sl = float(sea_level_override_m)
        sl_source_label = f"basin-override {selection_sl:+.0f} m"
    else:
        selection_sl = spratt_sl
        sl_source_label = None  # default: label derived from the matched step

    if basin_seeds is None:
        coast_gdf = (
            preloaded_coast_gdf
            if preloaded_coast_gdf is not None
            else load_paleocostlines(paleocoastlines_shp)
        )
        land_gdf, coast_level = filter_paleocostlines_for_epoch(coast_gdf, selection_sl)
    else:
        # The vector Paleocoastlines carry ONE global sea-level step per polygon.
        # A frame with two basins at two levels has no such step, so the vector
        # path is not "unavailable" here — it is inapplicable, and saying so is
        # cheaper than loading a shapefile whose answer would be discarded.
        land_gdf, coast_level = None, None

    # ── DEM mosaic ───────────────────────────────────────────────────────────
    # The frame is read from the rasters BEFORE they are merged and smoothed, so a
    # sill expectation is checked against what the data says it is in, not against
    # what the caller remembers.
    dem_vertical_frame = declared_vertical_frame(tile_paths)
    dem, ice, xform, bounds, crs, _res = build_epoch_mosaic(tile_paths, cfg)

    # ── Compositing ─────────────────────────────────────────────────────────
    H, W = dem.shape
    nodata = np.isnan(dem)

    dem_sm = gaussian_filter(
        np.where(nodata, 0.0, dem), sigma=MASK_SMOOTHING_SIGMA_PX
    )
    dem_sm = np.where(nodata, np.nan, dem_sm)
    dem_f = np.where(nodata, 0.0, dem)
    dem_fs = np.where(nodata, 0.0, dem_sm)

    # Ice mask from sidecar
    ice_mask = (~nodata) & (ice > cfg.ice_threshold_m)

    # ── The sequence colour mapping, if one was pinned ────────────────────────
    # Checked here, before anything is computed from it, so a frame the limits
    # never saw costs the caller a refusal rather than 340 s and a wrong picture.
    limits = cfg.sequence_limits
    limits_audit: Optional[dict] = None
    clip_policy = (cfg.clip_policy or "mark") if limits is not None else None
    if limits is not None:
        limits_audit = limits.assert_consumable_by(
            cfg,
            epoch_ka=epoch_ka,
            tile_paths=tile_paths,
            verify_hashes=cfg.verify_sequence_raster_hashes is not False,
        )
        if ice_mask.any() and limits.ice_max_m is None:
            raise SequenceLimitsRecipeError(
                f"this frame has {int(ice_mask.sum()):,} ice cells (thickest "
                f"{float(ice[ice_mask].max()):.2f} m) and the sequence limits record "
                "ice_max_m=None, i.e. the limits pass found no ice side-car anywhere "
                "in the sequence. Falling back to this frame's own maximum is exactly "
                "the per-frame normalisation the limits exist to remove, and it would "
                "do it silently on the one array — the dome blend — that also feeds "
                "the hillshade and therefore the LAND colour. Recompute the limits "
                "over the rasters whose _ice.tif side-cars exist."
            )

    # The reference thickness the ice ramp AND the dome blend are both normalised
    # by.  One variable because it was always one expression written twice; pinning
    # it in one place is what stops the dome weight — and through it dem_fs, the
    # hillshade and every land pixel — from being re-derived per frame.
    if ice_mask.any():
        ice_reference_m = (
            max(float(ice[ice_mask].max()), 1.0)
            if limits is None
            else float(limits.ice_max_m)
        )
    else:
        ice_reference_m = 1.0

    # ── Ice dome smoothing ────────────────────────────────────────────────────
    # Production IceThicknessModel smooths the ice surface before reprojection:
    # gaussian_filter(sigma=0.7 on 1°-grid) → cubic bicubic reproject.
    # At tile resolution the equivalent is a large-sigma smooth on the DEM that
    # replaces the jagged bedrock texture under the glacier with a smooth dome.
    # This is the most visible part of the pipeline — without it the ice surface
    # shows fjord/mountain topography instead of a clean dome shape.
    if ice_mask.any():
        # Heavily smooth the ice surface to remove bedrock texture
        dem_ice_input = np.where(nodata, 0.0, dem)
        dem_ice_dome = gaussian_filter(dem_ice_input, sigma=cfg.ice_dome_sigma_px)
        # Blend: full dome smoothing where ice is thick, taper at margins
        ice_h_norm = np.clip(ice / ice_reference_m, 0.0, 1.0)
        dem_dome_blend = np.where(
            ice_mask,
            dem_ice_dome * ice_h_norm + dem_f * (1.0 - ice_h_norm),
            dem_f,
        )
        # Use blended dome surface for hillshade only on ice pixels
        dem_fs = np.where(ice_mask, dem_dome_blend, dem_fs)
        del dem_ice_input, dem_ice_dome, dem_dome_blend, ice_h_norm

    # Land / sea mask. force_dem_mask -> DEM-driven at selection_sl (basin path);
    # otherwise vector first, DEM fallback (also at selection_sl, not a hard 0 m).
    vector_land = None if force_dem_mask else rasterize_land_mask(
        land_gdf, xform, H, W, crs, bounds
    )

    basin_field: Optional[BasinField] = None
    morphology: dict = {}
    # Cells whose connectivity this raster cannot decide.  Empty on every path but
    # the multi-basin one, and empty there too unless a body reaches the observation
    # boundary — so the single-basin composite is assembled exactly as before.
    unobservable_mask = np.zeros((H, W), dtype=bool)
    if basin_seeds is not None:
        # Ownership, sill and mask are all derived from the SAME array the
        # single-basin path thresholds (dem_sm), so they cannot disagree.
        basin_field = assign_basins(
            dem_sm,
            xform,
            basin_seeds,
            ceiling_margin_m=basin_ceiling_margin_m,
            corridor=basin_corridor,
            sill_expectations=basin_sill_expectations,
            sill_tolerance_m=basin_sill_tolerance_m,
            dem_frame=dem_vertical_frame,
            unsmoothed_dem=(
                None if cfg.measure_smoothing_sill_shift is False
                else np.where(nodata, np.nan, dem)
            ),
            # NOT basin_sill_tolerance_m.  Widening the sill tolerance is a
            # statement about the grid vs an independent measurement; it must not
            # silently also raise the bar on what this module's own sigma=1.5 is
            # allowed to do to the crest without saying so.  One knob, one guard.
            smoothing_shift_max_m=cfg.smoothing_shift_disclose_m,
            nodata=nodata,
            unobservable_policy=basin_unobservable_policy,
        )
        # THE MASK COMES FROM THE FIELD, IT IS NOT RE-DERIVED HERE.  Both lines
        # used to threshold ``dem_sm < water_level_m`` again — the FLAT test, with
        # no connectivity condition — so the renderer painted every closed
        # depression lying below its owner's level whatever assign_basins had
        # decided about them.  ``basin_field.water_mask`` is the hydrologically
        # connected mask (only components a basin seed reaches), and it is the one
        # array the ownership, the sill and the paint all now come off.
        ocean_raw = basin_field.water_mask
        ocean_despeckled = despeckle_ocean_mask(ocean_raw, nodata)
        ocean_mask = ocean_despeckled
        if basin_field.corridor_mode == "ramp":
            # The de-speckle above is a single-pixel-noise filter; a RAMPED corridor
            # is the SUBJECT of this frame, every one of its cells is wet by
            # construction (``corridor_mask = ramped & (dem < water_level)``), and a
            # narrow one must not be opened away — so it is restored verbatim.
            ocean_mask = ocean_despeckled | (basin_field.corridor_mask & (~nodata))
        # ... but a STEP corridor is NOT wet.  In step mode ``corridor_mask`` is the
        # raw ``candidates`` set — the cells a spill would flow THROUGH — and every
        # cell in it keeps its owner's flat level, which most of them stand above.
        # ORing it into the ocean mask paints water where the DEM says there is none
        # and makes the two basins look JOINED, which is the one thing this frame
        # must not fake.  What that would cost is measured per frame by
        # ``_corridor_cost`` and quoted by the disclosure; no figure for it is
        # written down here, because a comment and a sentence carrying the same
        # remembered constant is how the two came to disagree (1 099 766 cells /
        # 747 pieces here against 1 713 697 / 118 in the shipped string).  The
        # corridor is marked (see the route polyline and the sidecar), never painted.
        #
        # THE THIRD STATE IS NOT LAND.  ``unobservable_mask`` holds the components
        # that reach no seed but DO reach the edge of the observed grid — on the
        # production frame the Aegean+Marmara and the Levantine, whose link to the
        # Mediterranean runs south of the scene's 36 N edge.  Leaving them out of
        # ``ocean_mask`` and in ``land_mask`` draws open sea as dry land, which is
        # what shipped until this revision.  They are excluded from BOTH and painted
        # in a reserved hatched colour instead; ``land_mask`` is the only mask on
        # this path that changes, and only on the multi-basin path.
        unobservable_mask = basin_field.unobservable_mask
        if unobservable_mask.shape != (H, W):        # pragma: no cover
            raise ValueError(
                "BasinField.unobservable_mask has shape "
                f"{unobservable_mask.shape} but the frame is {(H, W)}; the "
                "connectivity states were measured on a different grid from the "
                "one being drawn."
            )
        # The de-speckle can close a hole that lands inside an unobservable body.
        # Painted water wins there — it is the mask the depth ramp and the legend
        # are about — so the reserved colour never overlaps the blue, and the count
        # in the legend is of what is actually drawn as unresolved.
        unobservable_mask = unobservable_mask & (~ocean_mask) & (~nodata)
        land_mask = (~nodata) & (~ocean_mask) & (~unobservable_mask)
        morphology = {
            "closing_iterations": OCEAN_CLOSING_ITERATIONS,
            "opening_iterations": OCEAN_OPENING_ITERATIONS,
            "cells_added": int((ocean_despeckled & ~ocean_raw).sum()),
            "cells_removed": int((ocean_raw & ~ocean_despeckled).sum()),
            "cells_drawn": int(ocean_mask.sum()),
            "note": (
                # NO NUMERALS.  This note used to argue that the de-speckle is small
                # by quoting "2 157 added and 1 905 removed out of 8 059 790 drawn
                # cells on the native 16.5 ka step frame" — three constants measured
                # on a TWO-SEED frame, shipped unchanged on three-basin frames, and
                # contradicted by cells_added / cells_removed / cells_drawn sitting
                # in this very dict.  The fields ARE the argument; the sentence says
                # what they mean and which of them to compare.
                "cells_added/removed are the DE-SPECKLE ALONE — the closing then "
                "opening of despeckle_ocean_mask, measured against the mask it was "
                "given, which is BasinField.water_mask (the seeded connected "
                "components of dem_sm < water_level_m, NOT the flat threshold; see "
                "the connectivity block) — with any corridor restoration excluded. "
                "The closing fills small holes, so cells_added ARE cells drawn as "
                "water while standing above their own surface, or outside every "
                "seeded body; that is the pre-existing de-speckle and it bounds "
                "painted_water.corridor_cells_painted_above_their_surface from "
                "above. How small it is ON THIS FRAME is cells_added against "
                "cells_drawn, both in this dict; do not carry a ratio over from "
                "another frame. The number that must be zero is "
                "painted_water.corridor_overpaint_from_the_or."
            ),
        }
        # What the corridor OR itself added, kept separate from the de-speckle so
        # the step-overpaint has a number of its own that must be exactly zero.
        corridor_audit = {
            "corridor_cells_restored_by_the_or": int(
                (ocean_mask & ~ocean_despeckled).sum()
            ),
            "corridor_overpaint_from_the_or": int(
                (ocean_mask & ~ocean_despeckled & ~ocean_raw).sum()
            ),
        }
        del ocean_raw, ocean_despeckled
        mask_src = "DEM contour, per-basin: " + " / ".join(
            f"{name} {level:+.2f} m"
            for name, level in zip(basin_field.names, basin_field.levels_m)
        )
    elif vector_land is not None:
        land_mask = (~nodata) & vector_land
        ocean_mask = (~nodata) & (~vector_land)
        mask_src = f"Paleocoastlines {coast_level:+d} m"
        if sl_source_label is not None:
            mask_src += f" [{sl_source_label}]"
    else:
        ocean_mask = dem_threshold_ocean(dem_sm, nodata, selection_sl)
        land_mask = (~nodata) & (~ocean_mask)
        mask_src = (
            f"DEM contour {selection_sl:+.0f} m"
            if force_dem_mask
            else f"DEM threshold {selection_sl:+.0f} m (vector unavailable)"
        )
        if sl_source_label is not None:
            mask_src += f" [{sl_source_label}]"

    # Hillshade
    cos_lat = np.cos(np.radians((bounds.bottom + bounds.top) / 2.0))
    dx_m = abs(xform.a) * 111_320.0 * cos_lat
    dy_m = abs(xform.e) * 111_320.0
    ls = LightSource(azdeg=HILLSHADE_AZDEG, altdeg=HILLSHADE_ALTDEG)
    shade_low = shade_high = None
    if limits is None:
        # matplotlib rescales the raw intensity by THIS frame's own min and max —
        # see analytic_shade_range() for the four lines that do it.  Unchanged, so
        # the single-basin path stays byte-stable.
        hs = ls.hillshade(dem_fs, vert_exag=cfg.vert_exag, dx=dx_m, dy=dy_m)
    else:
        raw_shade = _raw_hillshade_intensity(dem_fs, ls, cfg.vert_exag, dx_m, dy_m)
        shade_low = raw_shade < limits.shade_min
        shade_high = raw_shade > limits.shade_max
        hs = np.clip(
            (raw_shade - limits.shade_min) / (limits.shade_max - limits.shade_min),
            0.0,
            1.0,
        )
        del raw_shade
    hs3 = np.stack([hs] * 3, axis=-1)

    # Land colour
    # Exclude ice-covered pixels from the elevation normalization so that ice
    # elevations (e.g. 3000-5400 m at LGM) don't compress the land colour range.
    land_non_ice = land_mask & (~ice_mask)
    if limits is not None:
        lv_min, lv_max = float(limits.land_min_m), float(limits.land_max_m)
    elif land_non_ice.any():
        lv_min = float(np.nanpercentile(dem[land_non_ice], cfg.land_elev_percentile_lo))
        lv_max = float(np.nanpercentile(dem[land_non_ice], cfg.land_elev_percentile_hi))
    elif land_mask.any():
        lv_min = float(np.nanpercentile(dem[land_mask], cfg.land_elev_percentile_lo))
        lv_max = float(np.nanpercentile(dem[land_mask], cfg.land_elev_percentile_hi))
    else:
        lv_min, lv_max = 0.0, 800.0
    l_norm = np.clip((dem_f - lv_min) / max(lv_max - lv_min, 1.0), 0.0, 1.0)
    l_base = plt.cm.terrain(l_norm * 0.57 + 0.25)[..., :3]
    land_rgb = np.clip(l_base * (hs3 * 0.72 + 0.28), 0.0, 1.0)

    # Ocean colour — sqrt-stretched depth BELOW THE EPOCH SURFACE (selection_sl),
    # so a lowered decoupled basin reads as shallower, not as deep as today.
    water_surface = (
        selection_sl if basin_field is None else basin_field.water_level_m
    )
    depth = ocean_depth_from_surface(dem_f, ocean_mask, water_surface)
    if limits is not None:
        depth_floor_m = float(limits.depth_min_m)
        max_depth = float(limits.depth_max_m)
        depth_norm = np.clip(
            np.sqrt(
                np.clip(depth - depth_floor_m, 0.0, None)
                / max(max_depth - depth_floor_m, 1.0)
            ),
            0.0,
            1.0,
        )
    else:
        depth_floor_m = 0.0
        if ocean_mask.any():
            max_depth = max(float(np.percentile(depth[ocean_mask], 95)), 1.0)
        else:
            max_depth = 1.0
        depth_norm = np.clip(np.sqrt(depth / max_depth), 0.0, 1.0)
    ocean_rgb = plt.cm.Blues(depth_norm * 0.60 + 0.30)[..., :3]

    # Ice colour — snow/blue-ice palette driven by hillshade from smoothed dome
    # Matches the visual convention used in paleo-DEM publications:
    # thin ice → steel-blue, thick ice → white, with subtle hillshade shading.
    if ice_mask.any():
        # Ice hillshade already baked into hs3 (computed from smoothed dem_fs)
        # Normalise ice thickness: 0 (threshold) → 1 (max) → maps to colour range.
        # ``ice_reference_m`` is this frame's own maximum when nothing is pinned and
        # the sequence-wide maximum when a SequenceLimits is.
        ice_depth_norm = np.clip(ice / ice_reference_m, 0.0, 1.0)
        # plt.cm.Blues_r: low value = dark blue, high = white
        # We map 0→0.3 (thin ice = steel blue) … 1→0.92 (thick ice ≈ white)
        ice_cmap_val = ice_depth_norm * 0.62 + 0.30
        ice_base_rgb = plt.cm.Blues_r(ice_cmap_val)[..., :3]
        # Blend with hillshade (weaker exaggeration on ice → 0.5 ambient + 0.5 shaded)
        ice_rgb = np.clip(ice_base_rgb * (hs3 * 0.55 + 0.45), 0.0, 1.0)
    else:
        ice_rgb = None

    # Composite RGBA
    composite = np.full((H, W, 4), [0.06, 0.06, 0.06, 1.0], dtype=np.float32)
    composite[ocean_mask, :3] = ocean_rgb[ocean_mask]
    composite[land_mask, :3] = land_rgb[land_mask]
    # THE THIRD STATE, DRAWN AS A THIRD STATE.  Painted after land and before ice so
    # it cannot be mistaken for either, in a colour no ramp in this module produces
    # and carrying a regular diagonal stripe no terrain does — the picture has to say
    # "unresolved" to a reader who never opens the sidecar.  Empty except on the
    # multi-basin path, so nothing below runs on the byte-stable single-basin frames.
    unobservable_stripe_px = unobservable_stripe_period_px(W, cfg)
    if unobservable_mask.any():
        composite[unobservable_mask, :3] = _unobservable_rgb(
            unobservable_mask, unobservable_stripe_px
        )
    if ice_mask.any() and ice_rgb is not None:
        composite[ice_mask, :3] = ice_rgb[ice_mask]

    # ── What the pinned mapping could not represent ───────────────────────────
    palette_report: Optional[dict] = None
    if limits is not None:
        palette_report = _sequence_limit_clipping(
            limits=limits,
            dem_f=dem_f,
            depth=depth,
            ice=ice,
            nodata=nodata,
            land_mask=land_mask,
            ocean_mask=ocean_mask,
            ice_mask=ice_mask,
            shade_low=shade_low,
            shade_high=shade_high,
            ramp_land_min_m=lv_min,
            ramp_land_max_m=lv_max,
            ramp_depth_max_m=max_depth,
        )
        palette_report["clip_policy"] = clip_policy
        palette_report["limits_audit"] = limits_audit
        out_high = palette_report.pop("_mask_high")
        out_low = palette_report.pop("_mask_low")
        if clip_policy == "raise" and palette_report["out_of_range"]["total"]:
            raise SequenceLimitsExceededError(
                _sequence_limits_exceeded_message(epoch_ka, limits, palette_report)
            )
        if clip_policy == "mark":
            # LOW first, then HIGH, so a cell that is somehow both reads as high —
            # and the two counts are reported separately either way, so the overlap
            # is visible in the numbers rather than decided by paint order alone.
            composite[out_low, :3] = OUT_OF_RANGE_LOW_RGB
            composite[out_high, :3] = OUT_OF_RANGE_HIGH_RGB
        del out_high, out_low

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(cfg.fig_size_in, cfg.fig_size_in), dpi=cfg.dpi
    )
    ax.imshow(
        composite,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="equal",
        interpolation="bilinear",
    )
    del composite

    if ocean_mask.any():
        sm = plt.cm.ScalarMappable(
            cmap="Blues",
            norm=mcolors.Normalize(vmin=0.0, vmax=max_depth),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
        depth_label = (
            "Depth below basin surface (m)"
            if (sea_level_override_m is not None or basin_field is not None)
            else "Depth below sea level (m)"
        )
        cb.set_label(depth_label, fontsize=11)

    n_valid = max(int((~nodata).sum()), 1)
    pct_land = land_mask.sum() / n_valid * 100
    pct_sea = ocean_mask.sum() / n_valid * 100
    pct_ice = ice_mask.sum() / n_valid * 100
    unobservable_cells = int(unobservable_mask.sum())
    pct_unobservable = unobservable_cells / n_valid * 100

    legend_items = [
        Line2D([0], [0], color="#2a6db5", lw=6,
               label=f"{water_legend_word(sea_level_override_m, basin_field)}"
                     f"  ({mask_src})"),
        Line2D([0], [0], color="#6b8f3a", lw=6, label="Land  (hillshade + terrain)"),
    ]
    if basin_field is not None:
        legend_items.append(
            Line2D([0], [0], color="#2a6db5", lw=6,
                   label=_basin_legend_label(basin_field))
        )
        # The third state gets a swatch of its own, with the count measured on the
        # mask that was actually painted.  Only when there is something to point at:
        # a legend entry naming a colour that appears nowhere in the image is the
        # same defect as a false number in a caption.
        if unobservable_cells:
            unobservable_block = basin_field.connectivity.get("unobservable", {})
            legend_items.append(
                Line2D(
                    [0], [0], color=UNOBSERVABLE_RGB, lw=6,
                    label=(
                        "CONNECTIVITY NOT OBSERVABLE from this extent  "
                        f"({unobservable_block.get('components', 0):,} bodies, "
                        f"{unobservable_cells:,} cells, "
                        f"{unobservable_block.get('km2', 0.0):,.0f} km²)\n"
                        "neither water nor land — the route that would decide it "
                        "leaves the frame"
                    ),
                )
            )
        route_handle = _draw_sill_routes(ax, basin_field, xform, cfg)
        if route_handle is not None:
            legend_items.append(route_handle)
    if ice_mask.any():
        legend_items.append(
            Line2D([0], [0], color="#c6e2f7", lw=6,
                   label=f"Ice sheet  (ICE-6G sice > {cfg.ice_threshold_m:.0f} m, "
                         f"dome-smoothed σ={cfg.ice_dome_sigma_px:.0f} px)")
        )
    if palette_report is not None:
        legend_items.append(
            Line2D([0], [0], color="#7f7f7f", lw=6,
                   label=_sequence_limits_legend_label(limits, palette_report))
        )
        # Only under "mark".  Under "clip" nothing was painted, and a legend swatch
        # naming a colour that appears nowhere in the image is the same defect as a
        # false number in a caption.
        if clip_policy == "mark":
            if palette_report["out_of_range"]["high"]:
                legend_items.append(
                    Line2D([0], [0], color=OUT_OF_RANGE_HIGH_RGB, lw=6,
                           label=f"ABOVE every value the limits measured  "
                                 f"({palette_report['out_of_range']['high']:,} cells)")
                )
            if palette_report["out_of_range"]["low"]:
                legend_items.append(
                    Line2D([0], [0], color=OUT_OF_RANGE_LOW_RGB, lw=6,
                           label=f"BELOW every value the limits measured  "
                                 f"({palette_report['out_of_range']['low']:,} cells)")
                )

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=9,
        framealpha=0.75,
        facecolor="#1a1a1a",
        labelcolor="white",
    )

    title_lines = [
        f"Paleo Render  –  {epoch_ka} ka BP  [{data_source_label.upper()}]",
        f"Coastline: {mask_src}  |  GIA + {cfg.ice_model_label} corrected DEM",
        (
            f"land={pct_land:.0f}%  sea={pct_sea:.0f}%  ice={pct_ice:.0f}%  |  "
            f"{bounds.left:.1f}–{bounds.right:.1f}°E  "
            f"{bounds.bottom:.1f}–{bounds.top:.1f}°N"
            # Appended, not inserted: the single-basin path never has an
            # unobservable cell, so this line stays character-for-character what it
            # was on the 39 byte-stable Pontic frames.
            + (
                f"  |  connectivity unobservable={pct_unobservable:.1f}%"
                if unobservable_cells else ""
            )
        ),
    ]
    if basin_field is not None:
        title_lines.append(_basin_title_line(basin_field))
        # The third state, on the face of the picture.  A reader of the PNG alone
        # must not be able to read the hatched region as either sea or land, and
        # must not have to open the sidecar to learn that it is neither.
        if unobservable_cells:
            block = basin_field.connectivity.get("unobservable", {})
            title_lines.append(
                UNOBSERVABLE_CAPTION_TEMPLATE.format(
                    components=f"{block.get('components', 0):,}",
                    cells=f"{unobservable_cells:,}",
                    area=f"{block.get('km2', 0.0):,.0f}",
                    edges="/".join(sorted(block.get("by_edge", {}))) or "observed",
                )
            )
        painted_water = _painted_water_report(
            ocean_mask, basin_field, xform, list(basin_seeds)
        )
        painted_water.update(corridor_audit)
        # The disagreement, stated on the face of the picture and not only in the
        # sidecar: a reader looking at two separate blue shapes under the word
        # SPILLING has to be told which of the two is the claim.
        if basin_field.spilling and not painted_water["basins_joined"]:
            title_lines.append(
                f"Spill verdict is measured on the terrain; the DRAWN water is "
                f"{painted_water['components']} separate bodies and the basins are "
                f"NOT joined in this image — "
                f"{painted_water['gap_cells']:,} of "
                f"{sum(painted_water['route_cells_by_pair'].values()):,} route "
                f"cells are dry because the BED THERE STANDS ABOVE the surface "
                f"they would have to be under (see spill_not_drawn in the sidecar)"
            )
    if palette_report is not None:
        title_lines.append(_sequence_limits_title_line(limits, palette_report))
    if cfg.extra_figure_title:
        title_lines.append(cfg.extra_figure_title)
    ax.set_title("\n".join(title_lines), fontsize=10, pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    out_path = output_dir / f"paleo_render_{epoch_ka}ka.png"
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    if basin_field is not None:
        _write_basin_sidecar(
            out_path.parent / f"{out_path.stem}.basins.json",
            epoch_ka,
            basin_field,
            nodata,
            grid={
                "max_px": int(cfg.max_px),
                "height": int(H),
                "width": int(W),
                "pixel_deg": float(abs(xform.a)),
                "mask_smoothing_sigma_px": float(MASK_SMOOTHING_SIGMA_PX),
                "west": float(bounds.left),
                "south": float(bounds.bottom),
                "east": float(bounds.right),
                "north": float(bounds.top),
                "note": (
                    "max_px is what was ASKED for; height/width/pixel_deg are what "
                    "the merge produced. A sill elevation without the grid it came "
                    "off cannot be re-derived, only trusted."
                ),
            },
            painted_water=painted_water,
            morphology=morphology,
            sill_tolerance_m=basin_sill_tolerance_m,
            dem_vertical_frame=dem_vertical_frame,
            smoothing_measured=cfg.measure_smoothing_sill_shift is not False,
            # The route note is measured on the SAME array the route was drawn from
            # and the sill was read off — dem_sm, not the raw mosaic — so the
            # elevations it quotes are the ones elevation_m came from.
            dem=dem_sm,
            transform=xform,
            unobservable_drawn_cells=unobservable_cells,
            stripe_period_px=unobservable_stripe_px,
        )

    if palette_report is not None:
        palette_report["epoch_ka"] = float(epoch_ka)
        palette_report["grid"] = {
            "max_px": int(cfg.max_px),
            "height": int(H),
            "width": int(W),
            "pixel_deg": float(abs(xform.a)),
        }
        palette_report["limits"] = limits.to_dict()
        palette_report["reserved_colours"] = {
            "high_rgb": list(OUT_OF_RANGE_HIGH_RGB),
            "low_rgb": list(OUT_OF_RANGE_LOW_RGB),
            "note": OUT_OF_RANGE_COLOUR_NOTE,
        }
        palette_report["figure_size_note"] = SEQUENCE_LIMITS_FIGURE_SIZE_NOTE
        (out_path.parent / f"{out_path.stem}.palette.json").write_text(
            json.dumps(palette_report, indent=2), encoding="utf-8"
        )

    gc.collect()
    return out_path.resolve()


def _raw_hillshade_intensity(
    elevation: np.ndarray,
    light: LightSource,
    vert_exag: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """``n · d`` — matplotlib's hillshade intensity BEFORE its per-frame stretch.

    ``LightSource.hillshade`` builds the unit normal field and calls
    ``shade_normals``, whose last act is to rescale the dot product by *this array's
    own* minimum and maximum.  That rescale is the whole flicker term (see
    :func:`analytic_shade_range`), so the pinned path needs the value it is applied
    to, and matplotlib does not expose it.

    Verified equal to matplotlib's own pre-stretch intensity: rescaling this
    function's output by its own min/max reproduces ``LightSource.hillshade`` to
    **3.3e-16** on a random 97 × 131 surface (2026-08-03, pinned by
    ``tests/test_viz.py``).

    It is also **cheaper than matplotlib's route**, which is why the pinned path is
    not a memory regression on the native frame: ``LightSource.hillshade``
    materialises ``np.empty(shape + (3,))`` with no dtype, i.e. **float64** — 809 MB
    at 7800 × 4320 — while the closed form below never leaves 2-D:

    ``n = (−e_dx, −e_dy, 1) / ‖·‖`` so ``n · d = (−e_dx·d₀ − e_dy·d₁ + d₂) /
    sqrt(e_dx² + e_dy² + 1)``.
    """
    # matplotlib's sign convention: row 0 is the TOP of the image, so dy is
    # implicitly negative.  Copied deliberately rather than re-derived — the whole
    # point is to produce the same number it does.
    e_dy, e_dx = np.gradient(vert_exag * elevation, -dy, dx)
    d0, d1, d2 = light.direction
    denominator = np.sqrt(e_dx * e_dx + e_dy * e_dy + 1.0)
    intensity = (-e_dx * d0 - e_dy * d1 + d2) / denominator
    del e_dx, e_dy, denominator
    return intensity


def _sequence_limit_clipping(
    *,
    limits: SequenceLimits,
    dem_f: np.ndarray,
    depth: np.ndarray,
    ice: np.ndarray,
    nodata: np.ndarray,
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    ice_mask: np.ndarray,
    shade_low: Optional[np.ndarray],
    shade_high: Optional[np.ndarray],
    ramp_land_min_m: float,
    ramp_land_max_m: float,
    ramp_depth_max_m: float,
) -> dict:
    """Count what the pinned mapping could not represent, split two ways.

    The two ways are not the same thing and the split is the point:

    ``saturated_at_ramp_end``
        Cells beyond a **percentile** ramp endpoint.  ~4 % of land and ~5 % of ocean
        sit there in every frame *by construction* — that is what a p2/p98 ramp is
        for.  They are reported and NOT painted; painting them would magenta out a
        twentieth of the sea on a correct run and the signal would be worthless.

    ``out_of_range``
        Cells beyond the **true extrema the limits pass measured over the sequence**.
        A cell there is a cell the limits never saw, which means the limits were
        computed over data that does not contain this frame.  These are painted with
        the reserved colours, counted, and under ``clip_policy="raise"`` they stop
        the run.  **On a correct run every one of these counts is 0** — that is what
        makes them worth reading.

    Returns the report plus two private boolean masks (``_mask_high`` /
    ``_mask_low``) for the compositor to paint; the caller pops them before the dict
    is written to JSON.
    """
    valid = ~nodata
    painted_land = land_mask & (~ice_mask)

    # --- beyond the envelope: painted, and zero on a correct run --------------
    land_high = painted_land & (dem_f > limits.dem_data_max_m)
    land_low = painted_land & (dem_f < limits.dem_data_min_m)
    depth_high = ocean_mask & (depth > limits.depth_data_max_m)
    if ice_mask.any() and limits.ice_max_m is not None:
        ice_high = ice_mask & (ice > limits.ice_max_m)
    else:
        ice_high = np.zeros_like(valid)
    shaded = (land_mask | ice_mask) & valid
    shade_hi = (shade_high & shaded) if shade_high is not None else np.zeros_like(valid)
    shade_lo = (shade_low & shaded) if shade_low is not None else np.zeros_like(valid)

    mask_high = land_high | depth_high | ice_high | shade_hi
    mask_low = land_low | shade_lo
    mask_low = mask_low & (~mask_high)

    # --- at the ramp end: reported, never painted ----------------------------
    sat_land_high = painted_land & (dem_f > ramp_land_max_m)
    sat_land_low = painted_land & (dem_f < ramp_land_min_m)
    sat_depth = ocean_mask & (depth > ramp_depth_max_m)

    n_valid = max(int(valid.sum()), 1)
    total_oor = int(mask_high.sum()) + int(mask_low.sum())
    return {
        "out_of_range": {
            "high": int(mask_high.sum()),
            "low": int(mask_low.sum()),
            "total": total_oor,
            "fraction_of_valid_cells": total_oor / n_valid,
            "by_quantity": {
                "land_elevation_above_dem_data_max_m": int(land_high.sum()),
                "land_elevation_below_dem_data_min_m": int(land_low.sum()),
                "depth_above_depth_data_max_m": int(depth_high.sum()),
                "ice_thickness_above_ice_max_m": int(ice_high.sum()),
                "shade_above_shade_max": int(shade_hi.sum()),
                "shade_below_shade_min": int(shade_lo.sum()),
            },
            "note": (
                "cells whose value lies outside the TRUE EXTREMA the limits pass "
                "measured over the sequence -- not outside the ramp, which is a "
                "percentile and is supposed to saturate. A non-zero count here means "
                "the pinned limits were computed over data that does not contain "
                "this frame, so on a correct run every one of these is zero. They "
                "are "
                "painted with the reserved colours under clip_policy='mark' and they "
                "abort the frame under clip_policy='raise'. HOW THE THREE COUNTS "
                "RELATE: 'high' and 'low' are DISJOINT cell counts and total = high "
                "+ low -- a cell that is out of range in both directions at once is "
                "counted under 'high' only, and painted magenta. The by_quantity "
                "entries are NOT disjoint (one cell can be both land-below and "
                "shade-above), so they can sum to more than total; they answer "
                "'which limit did this frame breach', not 'how many cells'."
            ),
        },
        "saturated_at_ramp_end": {
            "land_above_land_max_m": int(sat_land_high.sum()),
            "land_below_land_min_m": int(sat_land_low.sum()),
            "depth_above_depth_max_m": int(sat_depth.sum()),
            "note": (
                "cells sitting at the end of a PERCENTILE ramp. Roughly the "
                "complement of land_percentile_hi (as a percentage) of the land, and "
                "the complement of depth_percentile of the ocean, land here in every "
                "frame by construction; this is what the "
                "ramp is for and they are deliberately NOT painted with a reserved "
                "colour. The count is here so a frame whose saturation jumps against "
                "its neighbours can be spotted without re-rendering it."
            ),
        },
        "valid_cells": int(valid.sum()),
        "ramp_actually_used": {
            "land_min_m": float(ramp_land_min_m),
            "land_max_m": float(ramp_land_max_m),
            "depth_max_m": float(ramp_depth_max_m),
            "note": (
                "the endpoints this frame was actually drawn on. With limits pinned "
                "they equal the limits' own; they are re-stated from the compositor "
                "so the sidecar reports what ran rather than what was configured."
            ),
        },
        "_mask_high": mask_high,
        "_mask_low": mask_low,
    }


def _sequence_limits_exceeded_message(
    epoch_ka: float, limits: SequenceLimits, report: dict
) -> str:
    """The refusal, with every number a caller needs to widen the limits on purpose."""
    per_quantity = ", ".join(
        f"{name}={count:,}"
        for name, count in sorted(report["out_of_range"]["by_quantity"].items())
        if count
    )
    return (
        f"epoch {epoch_ka} ka holds "
        f"{report['out_of_range']['total']:,} cells "
        f"({report['out_of_range']['fraction_of_valid_cells'] * 100:.4f} % of "
        f"{report['valid_cells']:,} valid) outside the extrema the sequence limits "
        f"recorded measuring: {per_quantity}. The pinned envelope is dem "
        f"[{limits.dem_data_min_m:+.3f}, {limits.dem_data_max_m:+.3f}] m, depth "
        f"<= {limits.depth_data_max_m:.3f} m, ice <= {limits.ice_max_m!r} m, shade "
        f"[{limits.shade_min:.6f}, {limits.shade_max:.6f}], computed over epochs "
        f"{list(limits.epochs)}. clip_policy='raise' refuses rather than clipping, "
        "because a sequence that clips one epoch against limits taken from other "
        "epochs misrepresents that epoch in a way no later reader can detect from "
        "the PNG. Recompute the limits over the epochs you are rendering, or set "
        "clip_policy='mark' to ship the frame with those cells painted and counted."
    )


#: Where the per-frame numbers go, and why not in the title.
#:
#: ``render_paleocostline_epoch`` saves with ``bbox_inches="tight"``, so the figure's
#: pixel size is a function of the widest line of text in it.  The title is ABOVE the
#: axes and every character of it widens the saved PNG; the legend sits at
#: ``loc="lower right"`` INSIDE the axes and does not.  Measured 2026-08-03 on a
#: 6 in / 50 dpi figure: a legend label growing from 12 to 13 characters left the
#: output at 286 × 295 px, unchanged, while the title alone drove 286 → 775 px.
#:
#: That matters here because the out-of-range counts differ frame to frame by
#: construction, and E-ANIM's acceptance requires all 34 frames to come out the same
#: pixel size.  Putting a varying-length count in the title would have made a
#: colour-flicker fix into a geometry-flicker bug: the first draft of this line did
#: exactly that and produced 1015 px for one epoch and 1014 px for the next.
#:
#: So: the TITLE line is fixed for a given ``SequenceLimits`` + ``clip_policy`` and
#: carries no count; the counts go in the LEGEND and in ``.palette.json``.  The
#: residual — that the *presence* of an out-of-range legend entry is itself a
#: frame-to-frame difference — is real, is far smaller, and is recorded in the
#: sidecar under ``figure_size_note`` for E-ANIM, whose own acceptance already
#: carries the fixed-bbox requirement.  Changing ``bbox_inches`` here is NOT done:
#: it would break ``test_pontic_single_basin_path_is_byte_stable``, and frame
#: geometry is E-ANIM's file, not this one.
SEQUENCE_LIMITS_FIGURE_SIZE_NOTE = (
    "This frame was saved with bbox_inches='tight', so its pixel size is a function "
    "of the widest line of text in the figure. The pinned-limits TITLE line is "
    "deliberately fixed for a given SequenceLimits + clip_policy and carries no "
    "per-frame count; the counts live in the legend (inside the axes, measured not "
    "to move the tight bbox for label-length changes of this order) and in this "
    "file. What remains is that a frame WITH out-of-range cells gains a legend entry "
    "a clean frame does not have. For an animation, render with a FIXED bbox rather "
    "than relying on that being small: E-ANIM's acceptance already requires all "
    "frames identical in pixel size."
)


def _sequence_limits_legend_label(limits: SequenceLimits, report: dict) -> str:
    """The legend entry that says the picture is NOT on its own frame's scale."""
    return (
        f"Colour scale PINNED across {len(limits.epochs)} epochs "
        f"({min(limits.epochs):g}–{max(limits.epochs):g} ka)\n"
        f"land {limits.land_min_m:+.0f}…{limits.land_max_m:+.0f} m  •  "
        f"depth 0…{limits.depth_max_m:.0f} m  •  shade "
        f"{limits.shade_min:.3f}…{limits.shade_max:.3f}  •  NOT this frame's own"
    )


def _sequence_limits_title_line(limits: SequenceLimits, report: dict) -> str:
    """One caption line, FIXED for a given limits object and clip policy.

    A caption is data, so it says which mapping the picture is on, over how many
    epochs, under which policy, and where the numbers are.  It deliberately carries
    **no per-frame count** — see :data:`SEQUENCE_LIMITS_FIGURE_SIZE_NOTE` for the
    measurement that decided that.  What it must never do is let a reader conclude
    the frame is on its own scale, or that nobody checked.
    """
    return (
        f"Colour limits PINNED over {len(limits.epochs)} epochs "
        f"({min(limits.epochs):g}–{max(limits.epochs):g} ka; "
        f"p{limits.land_percentile_lo:g}/p{limits.land_percentile_hi:g} land, "
        f"p{limits.depth_percentile:g} depth, analytic shade range) — NOT this "
        f"frame's own; clip_policy={report['clip_policy']!r}; every cell outside the "
        f"measured envelope is counted in the .palette.json sidecar"
    )


#: The disclosure that has to survive from the sidecar into the caption, so a
#: reader who sees only the PNG cannot conclude either that the verdict is wrong or
#: that the basins visibly merged.
#:
#: **Keyed by corridor mode, because the true sentence is not the same in both.**
#: The single string this replaces said the channel "runs below the interpolated
#: water surface for part of its length, and those cells are left dry rather than
#: painted".  Both halves were wrong. A cell BELOW the water surface is exactly the
#: cell that gets PAINTED — ``dem_sm < water_level`` is the elevation half of the
#: test, and nothing in it makes a cell dry — so the stated cause was the negation
#: of the real one: those cells are dry because the BED STANDS AT OR ABOVE the
#: surface, not below it.  And "the interpolated water surface" describes something
#: ``corridor="step"`` does not do: step gives every cell its own basin's flat
#: declared level and interpolates nothing.  Step is the mode the acceptance
#: artifact ships in, so the shipped ``.basins.json`` carried a sentence that was
#: wrong about the sign AND about the algorithm.
#:
#: **The painting rule gained its second half, and the sentence says so.**
#: ``dem_sm < water_level`` alone paints every closed depression lying below
#: whichever basin's level the ownership flood labelled its ground with; a cell is
#: now painted only if it is ALSO reachable from a basin seed through cells for
#: which the same is true — and a body that reaches the edge of the observed grid is
#: neither painted nor drawn as land, because this raster cannot decide it.
#:
#: **EVERY NUMERAL IN THESE SENTENCES IS INTERPOLATED FROM THE FRAME BEING
#: WRITTEN.**  The templates below contain no digits at all.  The step template used
#: to embed "1,713,697 cells / 271,632 km² in 118 disjoint pieces" and a "+29.94 m"
#: donor bound; those were measured once, on another frame, with another Caspian
#: level, and the same ``basins.json`` carried a ``corridor.cells`` that contradicted
#: them — two revisions of this module even disagreed with each other (1 099 766
#: cells / 747 pieces against 1 713 697 / 118).  A sentence a reader cannot audit
#: against the array beside it is worse than no sentence.
_SPILL_DISCLOSURE_COMMON = (
    "The spill verdict is measured on the TERRAIN, not read off the blue in this "
    "image: it is the connected-path minimax between the basin seeds — the lowest "
    "crest a continuous channel can be held under — compared against the donor "
    "basin's own declared surface. The blue answers a different question, cell by "
    "cell: a cell is painted only where the bed lies BELOW the water surface that "
    "cell carries AND that cell can be reached from a basin seed through cells for "
    "which the same is true. The second half is not decoration — without it the "
    "flat test paints every closed depression that happens to lie below the level "
    "of whichever basin the ownership flood labelled its ground with, a pit in "
    "Anatolia included; how many cells that is on this frame is in the connectivity "
    "block, which also names the largest of them and the surface each was being "
    "painted at. A component that reaches no seed but DOES reach the edge of the "
    "observed grid is a third case and is neither painted nor drawn as land: this "
    "raster cannot see where it goes, and connectivity.unobservable says how many "
    "such bodies there are, how large, which boundary each touches and which level "
    "it sits under. "
)
_SPILL_DISCLOSURE_TAIL = (
    "So two separate blue shapes under the word 'spilling' are not a contradiction "
    "and not a near-miss: the crest is passable at the donor's level, and the "
    "channel floor beyond it still stands above the level the receiving side is "
    "drawn at. The verdict is about a crest; the paint is about every cell. Neither "
    "one is a correction of the other, and the basins did NOT visibly merge here — "
    "if they had, painted_water.basins_joined would say true. The orange polyline is "
    "ONE minimax path through that crest (it is not the only one — see "
    "sills[].route_note); how many of its cells are drawn dry is "
    "painted_water.gap_cells, out of painted_water.route_cells_by_pair."
)

#: Per corridor mode.  Rendered by :func:`spill_not_drawn_disclosure`, which fills
#: every ``{}`` from the frame being written.  No template here contains a digit.
SPILL_NOT_DRAWN_DISCLOSURE_TEMPLATE = {
    "step": (
        "{common}"
        "This frame is corridor='step': every cell is drawn at its OWN basin's "
        "flat declared level and nothing is interpolated anywhere. Along the spill "
        "route the bed stands AT OR ABOVE the level of the basin that owns it for "
        "part of its length, so those cells are left dry — they are dry BECAUSE the "
        "ground there is higher than the water surface, which is the opposite of "
        "what an earlier version of this sentence said. Painting the corridor "
        "anyway was costed and refused: ON THIS FRAME it draws {corridor_cells} "
        "cells / {corridor_area} sq km in {corridor_components} disjoint pieces, of "
        "which {corridor_not_water} either stand above the level their own basin is "
        "drawn at or reach no basin's seeded water. The bed over that set runs "
        "{bed_min} to {bed_max} m and is bounded above by the donor {donor}'s own "
        "{donor_level} m surface, because the corridor is carved out of a flood AT "
        "that level; both figures are measured here so the bound can be checked. "
        "And it made the two basins look joined. "
        "{tail}"
    ),
    "ramp": (
        "{common}"
        "This frame is corridor='ramp': inside the one donor->receiver corridor the "
        "surface is interpolated linearly in along-channel graph distance between "
        "the two declared levels (a first-order water-surface profile, not "
        "hydraulics), and everywhere else each cell carries its own basin's flat "
        "declared level. Along the spill route the bed stands AT OR ABOVE that "
        "surface for part of its length, so those cells are left dry — they are dry "
        "BECAUSE the ground there is higher than the water surface, which is the "
        "opposite of what an earlier version of this sentence said. The drawn "
        "corridor is {corridor_cells} cells / {corridor_area} sq km in "
        "{corridor_components} pieces on this frame. "
        "{tail}"
    ),
    "none": (
        "{common}"
        "This frame has no corridor: no pair of basins is connected at these "
        "levels, or the connection wets no cell that is not already one basin's own "
        "water. Every cell carries its own basin's flat declared level, nothing is "
        "interpolated, and the separate blue shapes are separate because the terrain "
        "between them stands above both levels. Where a route was measured at all, "
        "the orange polyline is ONE minimax path through the crest (it is not the "
        "only one — see sills[].route_note); how many of its cells are drawn dry is "
        "painted_water.gap_cells."
    ),
}


def _number(value, spec: str = ",") -> str:
    """A measured value for a sentence, or the word for its absence.

    ``None`` becomes ``"not measured"`` rather than ``0`` or an empty string: a
    quantity that was not measured and a quantity that measured zero are different
    claims, and this module has already retracted one sentence for confusing them.
    """
    if value is None:
        return "not measured"
    return format(value, spec)


def spill_not_drawn_disclosure(corridor_mode: str, cost: Optional[dict] = None) -> str:
    """The disclosure sentence that is TRUE of *corridor_mode*, ON THIS FRAME.

    *cost* is ``connectivity["corridor_if_painted"]`` — what painting the corridor
    would draw, measured by :func:`_corridor_cost` on the frame being written.  Every
    numeral in the returned sentence comes from it; the templates carry none.

    Raises :class:`KeyError` on an unknown mode rather than falling back to a generic
    one: a disclosure that is not about the algorithm that ran is not a disclosure.
    Raises :class:`ValueError` when a mode whose sentence quotes corridor figures is
    asked for without them, because a template with an unfilled hole would ship a
    literal brace where a measurement belongs.
    """
    template = SPILL_NOT_DRAWN_DISCLOSURE_TEMPLATE[corridor_mode]
    if corridor_mode == "none":
        return template.format(common=_SPILL_DISCLOSURE_COMMON)
    if cost is None:
        raise ValueError(
            f"the corridor_mode={corridor_mode!r} disclosure quotes what the "
            "corridor would cost, and no measurement of it was supplied. Pass "
            "BasinField.connectivity['corridor_if_painted']. The figures used to be "
            "constants embedded in this template, measured on one frame and shipped "
            "unchanged on every other one; that is the defect this signature "
            "removes, so there is no default."
        )
    return template.format(
        common=_SPILL_DISCLOSURE_COMMON,
        tail=_SPILL_DISCLOSURE_TAIL,
        corridor_cells=_number(cost["cells"]),
        corridor_area=_number(cost["km2"], ",.0f"),
        corridor_components=_number(cost["components"]),
        corridor_not_water=_number(cost["cells_not_in_the_connected_water"]),
        bed_min=_number(cost["bed_min_m"], "+.2f"),
        bed_max=_number(cost["bed_max_m"], "+.2f"),
        donor=cost["donor"],
        donor_level=_number(cost["donor_level_m"], "+.2f"),
    )


def water_legend_word(
    sea_level_override_m: Optional[float],
    basin_field: Optional[BasinField],
) -> str:
    """What to call the blue in the legend.

    The ternary this replaces read ``"Lake" if sea_level_override_m is not None
    else "Ocean"``.  ``sea_level_override_m`` is ``None`` on the multi-basin path —
    the two are mutually exclusive arguments — so the Khvalynian hero frame, which
    is a Caspian and a Pontic lake standing at two different levels with no ocean
    anywhere in the extent, was captioned **"Ocean"**.  The label now keys on the
    thing that actually decides what the water is: whether a
    :class:`BasinField` produced it.

    The single-basin branch was written as ``"Basin water" if len(names) > 1 else
    "Basin water"`` — an unfinished edit whose two arms were identical, so the
    one-seed case had no label of its own and no test exercised it.  It is finished
    here: **one** seeded basin standing at **one** declared level is the same object
    the ``sea_level_override_m`` path already calls a Lake, expressed with a seed
    instead of a scalar, so it gets the same word.  More than one basin is not
    called a lake, because "Basin water" is the only label true of a Caspian and a
    Pontic standing at different levels in one frame.
    """
    if basin_field is not None:
        return "Basin water" if len(basin_field.names) > 1 else "Lake"
    return "Lake" if sea_level_override_m is not None else "Ocean"


def _draw_sill_routes(
    ax,
    field: BasinField,
    transform: rasterio.transform.Affine,
    cfg: RenderConfig,
) -> Optional[Line2D]:
    """Draw ONE minimax spill route per pair as an annotated polyline.

    Not water.  A LINE, in a colour no water body uses, with the crest cell marked
    and labelled with the elevation the verdict rests on.  This is the disclosure
    half of the corridor decision: the frame refuses to paint 1.1 M cells of
    corridor (see the step-mode comment in :func:`render_paleocostline_epoch`), so
    the route is the only thing that shows a reader *where* the two basins connect
    and lets them see, in the same picture, that the drawn water does not.

    WHAT THE LINE MAY AND MAY NOT CLAIM
    -----------------------------------
    The drawn polyline **is** a valid minimax path — verified: it is 4-connected,
    its endpoints are the two seed cells, and ``max(dem_sm)`` along it equals the
    reported sill to the last float64 bit.  What it is **not** is *the* spillway.
    A minimax path is an equivalence class, usually a large one: every path whose
    highest cell is that same crest qualifies equally, and which member gets drawn
    is decided by the flat-index tie-break inside :func:`_priority_flood_ownership`'s
    heap.  Measured on ``khv_16.5ka.tif`` at ``max_px=7800`` with the two production
    seeds (2026-08-03, re-derivable by
    ``test_the_drawn_route_is_a_valid_minimax_path_and_not_a_spillway``): the drawn
    route is 10 095 cells long, 4-connected, endpoints exactly the two seed cells,
    and ``max(dem_sm)`` along it is +28.000944137573242 m — the reported sill, to
    the last float64 bit.  It also reaches **47.4854 N, 1.750° ≈ 195 km north of its
    own crest at 45.7354 N**, and **1 193 of its cells (11.8 %) lie below −1 000 m**,
    down to **−2 509.77 m** in the abyssal Black Sea.  Hydrologically absurd,
    mathematically correct, and completely undetermined by the data.

    (Against the *modern GEBCO* crest constant 45.7062 N the same excursion measures
    198 km.  The two crest latitudes are 0.029° apart because they are measured on
    two different rasters — ``khv_16.5ka.tif`` at σ=1.5 and the modern COG — and the
    figure quoted above is the one taken on the raster the route was drawn from.)

    It used to be legended "Measured spill route (minimax path through the sill)".
    "Measured" and the definite article together assert a determinate spillway that
    the data does not fix, which is the picture making a claim the array cannot
    support.  So the legend, the annotation and the sidecar now say **one** path and
    say that it is not unique.  The line is kept rather than dropped because the
    quantity it is *about* — the crest elevation and the crest CELL — is determinate,
    and without the line a reader who sees only the PNG has nothing at all
    connecting the word "spilling" to a place on the map.

    The other two options were weighed and not taken.  Constraining the route to
    corridor cells would draw a different curve, equally arbitrary within the
    corridor, and would draw nothing at all in ``corridor="none"``; dropping the
    line would remove the only in-picture link between the verdict and the terrain,
    which the ``spill_not_drawn`` disclosure explicitly points at.

    Returns a legend handle, or ``None`` when nothing was drawn.
    """
    if cfg.draw_sill_route is False:
        return None
    drawn = False
    for pair, route in sorted(field.sill_route_rc.items()):
        if not route:
            continue
        rows = np.fromiter((rc[0] for rc in route), dtype=np.int64, count=len(route))
        cols = np.fromiter((rc[1] for rc in route), dtype=np.int64, count=len(route))
        lon, lat = transform * (cols + 0.5, rows + 0.5)
        ax.plot(lon, lat, color=SILL_ROUTE_COLOUR, lw=1.1, alpha=0.95, zorder=5)
        drawn = True
        crest = field.sill_lonlat.get(pair)
        value = field.sill_m.get(pair, float("nan"))
        if crest is not None and np.isfinite(value):
            ax.plot(
                [crest[0]], [crest[1]], marker="v", markersize=7,
                markerfacecolor=SILL_ROUTE_COLOUR, markeredgecolor="black",
                markeredgewidth=0.6, linestyle="none", zorder=6,
            )
            ax.annotate(
                f"{pair[0]}↔{pair[1]} sill {value:+.2f} m\n"
                f"{crest[0]:.3f} E / {crest[1]:.3f} N\n"
                "(crest determinate; the path drawn to it is not)",
                xy=crest,
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=7,
                color="white",
                zorder=7,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#1a1a1a",
                      "edgecolor": SILL_ROUTE_COLOUR, "alpha": 0.85},
            )
    if not drawn:
        return None
    return Line2D(
        [0], [0], color=SILL_ROUTE_COLOUR, lw=2,
        label=SILL_ROUTE_LEGEND_LABEL,
    )


#: What the polyline may say about itself.  Pinned by ``tests/test_viz.py``, which
#: fails if it claims a determinate spillway.  See :func:`_draw_sill_routes` for the
#: measurement behind every clause: the crest elevation and the crest cell are
#: determinate, the path to them is one member of an equivalence class picked by a
#: flat-index tie-break, and on the production raster that member runs hundreds of
#: km off the Manych–Don–Azov line and through the abyssal Black Sea.
SILL_ROUTE_LEGEND_LABEL = (
    "One minimax path through the sill — NOT unique, NOT a mapped spillway\n"
    "(any path with the same crest qualifies; the crest is what is measured)  •  NOT water"
)

#: The deep-water threshold the route note reports an excursion against, in metres.
#: Named so the sentence and the comparison are one object.
_ROUTE_DEEP_M = -1000.0


def sill_route_stats(
    route: Sequence[tuple[int, int]],
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    crest_lonlat: Optional[tuple[float, float]],
) -> dict:
    """Everything :func:`sill_route_note` says, as fields rather than as prose.

    The note and this dict are shipped side by side in ``sills[]`` so that every
    numeral in the sentence can be checked against a field of the same JSON object.
    That is the whole property the retired ``SILL_ROUTE_NOTE`` constant lacked: it
    quoted "the 10,095-cell route ... down to -2,509.77 m" beside a ``route_cells``
    field that disagreed with it on every frame but the one it was measured on.
    """
    stats = {
        "cells": len(route),
        "deep_threshold_m": float(_ROUTE_DEEP_M),
        "north_lat": None,
        "south_lat": None,
        "crest_lat": None if crest_lonlat is None else float(crest_lonlat[1]),
        "north_of_crest_deg": None,
        "north_of_crest_km": None,
        "cells_below_deep_threshold": None,
        "pct_below_deep_threshold": None,
        "min_elevation_m": None,
        "deg_km": float(_DEG_KM),
    }
    if not route:
        return stats

    rows = np.fromiter((rc[0] for rc in route), dtype=np.int64, count=len(route))
    cols = np.fromiter((rc[1] for rc in route), dtype=np.int64, count=len(route))
    _lon, lat = transform * (cols + 0.5, rows + 0.5)
    lat = np.asarray(lat, dtype=np.float64)
    elevation = np.asarray(dem, dtype=np.float64)[rows, cols]
    deep = elevation < _ROUTE_DEEP_M
    stats["north_lat"] = float(lat.max())
    stats["south_lat"] = float(lat.min())
    stats["cells_below_deep_threshold"] = int(deep.sum())
    stats["pct_below_deep_threshold"] = float(100.0 * float(deep.mean()))
    stats["min_elevation_m"] = float(np.nanmin(elevation))
    if crest_lonlat is not None:
        north = float(lat.max()) - float(crest_lonlat[1])
        stats["north_of_crest_deg"] = north
        stats["north_of_crest_km"] = abs(north) * _DEG_KM
    return stats


def sill_route_note(stats: dict) -> str:
    """What the drawn polyline may say about itself, from THIS route's measurements.

    Same statement as :data:`SILL_ROUTE_LEGEND_LABEL`, with the evidence.  Every
    figure comes from *stats* — :func:`sill_route_stats` on the route being written
    — and the same dict ships beside the sentence as ``sills[].route_stats``, so a
    reader can check each numeral against a field instead of against a memory.

    It used to be a module constant reading "the 10,095-cell route reaches
    47.4854 N, 1.750 deg (about 195 km) north of its own crest at 45.7354 N, and
    1,193 of its cells (11.8 %) lie below -1,000 m, down to -2,509.77 m".  Those
    figures were measured once, on ``khv_16.5ka.tif`` at ``max_px=7800`` with two
    particular seeds, and shipped verbatim in EVERY frame's ``route_note`` — beside
    a ``route_cells`` field, in the same JSON object, that contradicted them on
    every frame but that one.

    The claim the sentence makes is structural and does not depend on the numbers:
    a minimax path is an equivalence class, every path sharing the crest qualifies,
    and which member is drawn is decided by the flat-index tie-break inside
    :func:`_priority_flood_ownership`'s heap.  The numbers are what let a reader see
    how far from a mapped spillway that arbitrariness actually goes on the frame in
    front of them.
    """
    head = (
        "ONE minimax path from seed to seed — one member of an equivalence class, "
        "and not a determinate spillway. It is a genuine minimax path — "
        f"{_CONNECTIVITY}-connected, endpoints at the two seeds, max(dem_sm) along "
        "it equal to elevation_m to the last float64 bit — but every path sharing "
        "that crest is equally valid and the one drawn is whichever the priority "
        "flood's flat-index tie-break produced. It is NOT constrained to a mapped "
        "spillway. "
    )
    tail = (
        "Every figure here is measured on THIS route and repeated as a field in "
        "route_stats beside this sentence. What the data determines is elevation_m "
        "and (lon, lat) — the crest — not the line drawn to it."
    )
    if not stats["cells"]:
        return head + "No route was measured for this pair on this frame. " + tail

    measured = (
        f"On this frame it is {stats['cells']:,} cells long, its northernmost point "
        f"is {stats['north_lat']:.4f} N and its southernmost "
        f"{stats['south_lat']:.4f} N"
    )
    if stats["crest_lat"] is not None:
        measured += (
            f", i.e. {stats['north_of_crest_deg']:+.3f} deg (about "
            f"{stats['north_of_crest_km']:,.0f} km) north of its own crest at "
            f"{stats['crest_lat']:.4f} N"
        )
    measured += (
        f"; {stats['cells_below_deep_threshold']:,} of its cells "
        f"({stats['pct_below_deep_threshold']:.1f} %) lie below "
        f"{stats['deep_threshold_m']:+,.0f} m, down to "
        f"{stats['min_elevation_m']:+,.2f} m. "
    )
    return head + measured + tail


def _painted_water_report(
    ocean_mask: np.ndarray,
    field: BasinField,
    transform: rasterio.transform.Affine,
    seeds: Sequence[BasinSeed],
) -> dict:
    """What the frame actually drew, counted rather than described.

    ``basins_joined`` answers the question a reader asks of the picture — "did the
    two seas visibly become one?" — and it is deliberately reported next to
    ``spilling``, which answers a different question about the terrain.  On
    ``khv_16.5ka.tif`` at ``max_px=7800`` with ``corridor="step"`` the honest
    answers are ``spilling=True`` and ``basins_joined=False``, and the sidecar says
    both.

    ONE MASK, NAMED
    ---------------
    Every count here is measured on the DRAWN ``ocean_mask`` — the array the PNG is
    painted from, after :func:`despeckle_ocean_mask`'s closing/opening and after any
    corridor restoration.  ``unseeded_pools`` used to be the exception: it was
    copied out of ``BasinField``, i.e. measured on ``assign_basins``' PRE-de-speckle
    ``water_mask``, and sat in this dict beside counts of a different array.  The
    result on the production frame was a sidecar reporting 179 "drawn water bodies"
    next to ``components = 99``, with 131 of the 179 not in the drawn mask at all
    and 10 of them single cells the opening had already removed.  Two masks, one
    dict, no label.  The field is now ``unseeded_bodies_in_the_drawn_mask``, it is
    computed here from ``ocean_mask``, and its name says which array it counts.

    ``assign_basins``' own mask has no seedless body **by construction** — it is the
    union of the seeded connected components (:func:`_seeded_water_components`), and
    :class:`UnseededWaterBodyError` is retained there as the post-condition that says
    so.  Anything reported here is therefore a body the RENDERER's morphology
    created — a closing that bridged two specks, an opening that severed a one-cell
    isthmus — and it is worth seeing precisely because it did not exist upstream.

    What this dict does NOT count is the water the flat threshold would have painted
    and the connected mask does not.  Those cells are absent from ``ocean_mask``, so
    counting them here would be counting a different array again; they are in the
    sidecar's ``connectivity`` block, measured by :func:`_connectivity_report`.
    """
    seed_rc = [
        _seed_rowcol(transform, s.lon, s.lat, ocean_mask.shape, s.name) for s in seeds
    ]
    # ONE labelling of ONE mask feeds every count below, which is the whole point of
    # this function's rewrite: components, seed membership and the seedless bodies
    # are three readings of the same array and must not be able to disagree.
    n_components, seed_labels, unseeded_drawn, _labels = _unseeded_water_bodies(
        ocean_mask, field.water_level_m, seed_rc, transform
    )
    del _labels
    # ``0`` = that seed is not in the drawn water at all.  It used to be DROPPED
    # before counting, so a frame in which only ONE of two seeds landed in painted
    # water reported basins_joined=true: the surviving set had one element.  This is
    # the exact field a reader consults to check the disclosure, so it now requires
    # every seed to be wet AND all of them in the same component.
    dry_seeds = [s.name for s, v in zip(seeds, seed_labels) if v == 0]
    joined = (
        len(seed_labels) > 1
        and not dry_seeds
        and len(set(seed_labels)) == 1
    )

    gap_cells: dict[str, int] = {}
    route_cells: dict[str, int] = {}
    for pair, route in sorted(field.sill_route_rc.items()):
        key = f"{pair[0]}|{pair[1]}"
        route_cells[key] = len(route)
        if not route:
            gap_cells[key] = 0
            continue
        rows = np.fromiter((rc[0] for rc in route), dtype=np.int64, count=len(route))
        cols = np.fromiter((rc[1] for rc in route), dtype=np.int64, count=len(route))
        gap_cells[key] = int((~ocean_mask[rows, cols]).sum())

    return {
        "cells": int(ocean_mask.sum()),
        "components": int(n_components),
        "basins_joined": bool(joined),
        # Which drawn component each seed landed in, 0 = none.  Without this a
        # reader cannot tell "not joined because they are two bodies" from "not
        # joined because one seed is on dry paint".
        "seed_components": {s.name: int(v) for s, v in zip(seeds, seed_labels)},
        "seeds_not_in_drawn_water": dry_seeds,
        # In step mode this is SMALL, and it stopped being large on 2026-08-04 for a
        # reason worth stating: it was 607 188 when the mask was the flat
        # ``dem < water_level_m`` — corridor cells that stood below their own
        # owner's level and were drawn on their own merits.  Nearly all of them were
        # Pontic-owned closed depressions disconnected from the Pontic lake (the
        # Aegean+Marmara among them), so the connected mask drops them and 277 are
        # left on the native 16.5 ka step frame — every one of those from the
        # morphological CLOSING.  ``candidates`` is carved out of the donor-level
        # flood MINUS the donor's own water and the receiver's, so on this frame it
        # is disjoint from the connected water mask; that is measured here, not
        # assumed, because a receiver-owned corridor cell can in principle reach a
        # seed through the donor's water.
        "corridor_cells_painted": int((ocean_mask & field.corridor_mask).sum()),
        # The step-corridor overpaint, counted so it cannot come back silently.
        # ``water_mask`` is the seeded connected components of
        # ``dem < water_level_m`` — a cell's own declared surface, plus the
        # requirement that a seed reaches it — so this counts cells drawn as water
        # that either stand ABOVE the level they would have to be under or belong to
        # no seeded body.  ORing the step corridor in would draw 1 713 697 cells /
        # 271 632 km² in 118 disjoint pieces (the bed on that set is bounded above by
        # the donor's own +29.944 m level — measured maximum +29.9439 m — not by any
        # larger figure; an earlier revision of this comment claimed +203.71 m, which
        # the construction of ``candidates`` makes impossible) on khv_16.5ka.tif at
        # max_px=7800 with corridor="step".  It is 277, and every one of those comes
        # from the morphological CLOSING inside despeckle_ocean_mask, which fills
        # 1-px holes and has always done so; it is bounded by
        # morphology.cells_added (2 157).  The quantity that isolates THIS fix, and
        # that must be exactly zero, is ``corridor_overpaint_from_the_or``, added by
        # the renderer.
        "corridor_cells_painted_above_their_surface": int(
            (ocean_mask & field.corridor_mask & ~field.water_mask).sum()
        ),
        "gap_cells": int(sum(gap_cells.values())),
        "gap_cells_by_pair": gap_cells,
        "route_cells_by_pair": route_cells,
        "unseeded_bodies_in_the_drawn_mask": [
            {
                "cells": cells,
                # null, not NaN: the drawn mask can hold cells the morphology added
                # outside every basin's territory, which carry NO surface at all.
                # "no basin owns this" and "the surface is 0" are different claims,
                # and NaN is not valid JSON in the first place.
                "surface_min_m": (None if not np.isfinite(lo) else lo),
                "surface_max_m": (None if not np.isfinite(hi) else hi),
                "lon": lon,
                "lat": lat,
            }
            for cells, lo, hi, lon, lat in unseeded_drawn
        ],
        "note": (
            "EVERY count in this dict is measured on the DRAWN ocean_mask — the "
            "array the PNG is painted from, i.e. BasinField.water_mask after "
            "despeckle_ocean_mask's closing/opening and after any corridor "
            f"restoration. components = its {_CONNECTIVITY}-connected components. "
            "basins_joined = EVERY basin seed falls in painted water AND all of them "
            "in the SAME component, i.e. the picture shows a single water body; a "
            "seed on dry paint makes it false and is named in "
            "seeds_not_in_drawn_water (dropping such seeds before counting used to "
            "report joined=true when only one of two seeds was wet). It is "
            "independent of 'spilling', which is measured on the terrain. gap_cells "
            "= cells of the drawn minimax route that were NOT painted. "
            "corridor_cells_painted = corridor cells that ended up in the drawn mask "
            "(in step mode most of them are there on their own merits, standing "
            "below their owner's declared level and connected to its water). "
            "corridor_overpaint_from_the_or = corridor cells the corridor "
            "restoration ADDED to the mask while they stand above their own surface: "
            "it is the step-overpaint defect and it must be zero. "
            "unseeded_bodies_in_the_drawn_mask = components of THIS mask containing "
            "no basin seed. assign_basins' own water_mask has none BY CONSTRUCTION "
            "(it is built from the seeded components outwards; see the connectivity "
            "block and UnseededWaterBodyError), so an entry here is one the "
            "renderer's morphological CLOSING created by bridging two specks, and it "
            "did not exist upstream. The field is named for its mask because it used "
            "to be copied from the other one and reported that other mask's body "
            "count beside this mask's component count. What the flat threshold would "
            "have painted and is not painting is NOT here — it is in the "
            "connectivity block, because it is a count of cells absent from this "
            "mask; the cells drawn as CONNECTIVITY-UNOBSERVABLE are absent from it "
            "too, and are in connectivity.unobservable."
        ),
    }


def _basin_title_line(field: BasinField) -> str:
    """One-line summary of the multi-basin state, for the figure title."""
    state = "SPILLING" if field.spilling else "endorheic"
    groups = " | ".join("+".join(g) for g in field.connected_groups)
    sills = ", ".join(
        f"{'~'.join(pair)} sill "
        + ("above flood ceiling" if not np.isfinite(value) else f"{value:+.2f} m")
        for pair, value in sorted(field.sill_m.items())
    )
    unmeasured = [
        pair for pair, status in sorted(field.sill_expectation_status.items())
        if status.startswith("UNMEASURED")
    ]
    tail = ""
    if unmeasured:
        tail = "  |  UNVERIFIED sill: " + ", ".join(
            "~".join(p) for p in unmeasured
        )
    return (
        f"Basins: {state} ({groups})  |  {sills}  |  "
        f"corridor: {field.corridor_mode}{tail}"
    )


def _basin_legend_label(field: BasinField) -> str:
    corridor = {
        "ramp": "corridor: linear water-surface ramp (not hydraulics)",
        "step": "corridor: MARKED, not surfaced (step)",
        "none": "no corridor",
    }[field.corridor_mode]
    return f"{'spilling' if field.spilling else 'basins separate'}  ({corridor})"


def _smoothing_shift_note(
    *,
    shift: Optional[float],
    measured_here: bool,
    smoothing_measured: bool,
) -> str:
    """Why ``smoothing_shift_m`` is what it is, FOR THIS PAIR.

    This note used to be chosen from one frame-wide flag, ``smoothing_measured``,
    while the value it annotates is per pair and filled in only where the *raw*
    flood also reached that saddle (``sill_shift`` starts as
    ``dict.fromkeys(sill_m, None)``).  So a pair the unsmoothed flood never reached
    carried ``null`` under a sentence reading "measured, NEVER subtracted" — a
    disclosure that described a measurement that had not been taken.  There are four
    distinct reasons the field can read ``null`` and they are now told apart.
    """
    if shift is not None:
        return (
            "sill(dem_sm) - sill(unsmoothed mosaic), measured on THIS pair, NEVER "
            "subtracted from anything"
        )
    if not smoothing_measured:
        return (
            # No figure. This branch used to say "the real figure is +0.25 m at
            # max_px=7800 and +3.30 m at max_px=2000 on khv_16.5ka.tif" — a
            # measurement of two other frames, shipped as the explanation of a null
            # on THIS one. The point it was making survives without it: null is not
            # zero.
            "NOT MEASURED: RenderConfig.measure_smoothing_sill_shift is False, so no "
            "second flood was run for any pair. This is not 0.0, and the real shift "
            "on this grid is unknown because nobody measured it here. Re-render with "
            "measure_smoothing_sill_shift left at its default to get the number for "
            "this frame"
        )
    if not measured_here:
        return (
            "NOT MEASURED for this pair: this grid never reached the pair's saddle at "
            "all (elevation_m is null — it stands above the flood ceiling), so there "
            "is no smoothed value to difference against anything. Not 0.0"
        )
    return (
        "NOT MEASURED for this pair: the second flood ran, but on the UNSMOOTHED "
        "mosaic these two territories never met, so there is no unsmoothed saddle to "
        "subtract. That the smoothed grid reaches a saddle the raw one does not is "
        "itself a fact about the smoothing, and it is stated here rather than "
        "reported as 0.0"
    )


def _write_basin_sidecar(
    path: Path,
    epoch_ka: float,
    field: BasinField,
    nodata: np.ndarray,
    *,
    grid: dict,
    painted_water: dict,
    morphology: dict,
    sill_tolerance_m: float,
    dem_vertical_frame: Optional[str],
    smoothing_measured: bool,
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    unobservable_drawn_cells: int,
    stripe_period_px: int,
) -> None:
    """Record the multi-basin state next to the PNG, machine-readably.

    A figure is not a data format.  The animation needs, per frame, which basins
    existed, at what level, whether they were connected, and where the sill was —
    so it is written out rather than left to be re-derived or read off a picture.

    It also records **what the frame was measured on** and **what it did not draw**.
    A sill elevation without the grid it came off cannot be re-derived, only
    trusted; and ``spilling=true`` beside two visibly separate blue shapes is
    exactly the disagreement this file exists to state rather than hide.

    EVERY SENTENCE IN HERE IS BUILT FOR THIS FRAME
    ----------------------------------------------
    Prose shipped in a sidecar is data.  This project has retracted three shipped
    sentences — a sign inverted, a "+203.71 m" impossible on the cell set the same
    sentence named, and two notes whose cell counts the same JSON object
    contradicted — and all three had the same cause: a figure measured once, on one
    frame, embedded in a template and shipped unchanged on every other frame.  So
    the templates that feed this payload contain **no numerals at all**; every
    number in every note is interpolated here, from the frame being written.
    ``tests/test_viz.py`` enforces both halves — the structural rule on the source
    and the traceability of every numeral in a written sidecar back to a field of
    the same JSON object.
    """
    valid = int((~nodata).sum())
    # Measured once per pair, then shipped BOTH as fields and as the sentence built
    # from them, so the two cannot disagree.
    route_stats = {
        pair: sill_route_stats(
            field.sill_route_rc.get(pair, ()),
            dem,
            transform,
            field.sill_lonlat.get(pair),
        )
        for pair in field.sill_m
    }
    payload = {
        "epoch_ka": float(epoch_ka),
        "spilling": bool(field.spilling),
        "connected_groups": [list(g) for g in field.connected_groups],
        "flood_ceiling_m": float(field.ceiling_m),
        "grid": grid,
        "basins": [
            {
                "name": name,
                "level_m": float(level),
                "cells_owned": int((field.basin_id == i + 1).sum()),
                "cells_water": int(
                    (field.water_mask & (field.basin_id == i + 1)).sum()
                ),
            }
            for i, (name, level) in enumerate(zip(field.names, field.levels_m))
        ],
        "sills": [
            {
                "pair": list(pair),
                "elevation_m": (None if not np.isfinite(value) else float(value)),
                "lon": (
                    None
                    if field.sill_lonlat.get(pair) is None
                    else field.sill_lonlat[pair][0]
                ),
                "lat": (
                    None
                    if field.sill_lonlat.get(pair) is None
                    else field.sill_lonlat[pair][1]
                ),
                "expected_m": field.sill_expected_m.get(pair),
                # measured - expected, as a FIELD. expectation_status states it in
                # prose, and a number that appears only inside a sentence is a
                # number nobody can check against the object it is shipped in.
                "gap_m": (
                    None
                    if (
                        field.sill_expected_m.get(pair) is None
                        or not np.isfinite(value)
                    )
                    else float(value) - float(field.sill_expected_m[pair])
                ),
                "tolerance_m": float(sill_tolerance_m),
                "expectation_status": field.sill_expectation_status.get(
                    pair, "NOT CHECKED: single-basin frame has no pair"
                ),
                "measured_on": (
                    "dem_sm — the gaussian array dem_threshold_ocean builds the "
                    f"mask from, sigma={grid['mask_smoothing_sigma_px']} cells — at "
                    f"max_px={grid['max_px']}, "
                    f"{grid['height']}x{grid['width']} cells of "
                    f"{grid['pixel_deg']:.10g} deg, vertical_frame="
                    f"{dem_vertical_frame!r}"
                ),
                "smoothing_shift_m": field.sill_smoothing_shift_m.get(pair),
                "smoothing_shift_note": _smoothing_shift_note(
                    shift=field.sill_smoothing_shift_m.get(pair),
                    measured_here=np.isfinite(value),
                    smoothing_measured=smoothing_measured,
                ),
                "route_cells": len(field.sill_route_rc.get(pair, ())),
                "route_stats": route_stats[pair],
                "route_note": sill_route_note(route_stats[pair]),
                "note": (
                    "measured from this frame's DEM by seeded priority flood, as "
                    "the connected-path minimax, NOT a single-cell reading of the "
                    "sill point — those are different quantities and this project's "
                    "own raster separates them by metres; null = the saddle stands "
                    "above the flood ceiling and was NOT guessed"
                ),
            }
            for pair, value in sorted(field.sill_m.items())
        ],
        "corridor": {
            "mode": field.corridor_mode,
            "cells": int(field.corridor_mask.sum()),
            "painted": field.corridor_mode == "ramp",
            "note": (
                "ramp = water surface interpolated linearly in along-channel graph "
                "distance between the donor and receiver levels; it is a "
                "first-order water-surface profile, NOT a hydraulic model (no "
                "discharge, no roughness, no backwater), and its cells are wet by "
                "construction so they are drawn. step = every basin flat at its "
                "own level and the corridor only MARKED — those cells are NOT "
                "painted, because most of them stand above the level they would "
                "have to be painted at."
            ),
        },
        # How much "water" the FLAT threshold was inventing, measured on THIS
        # frame's own arrays.  It sits beside painted_water rather than inside it
        # because the two count different masks: painted_water counts the DRAWN
        # mask, this counts the difference between the drawn mask's input and the
        # flat test that used to be its input.  A frame that quietly paints fewer
        # cells than the previous revision is as opaque as one that quietly paints
        # more, so the difference ships in the data.
        "connectivity": field.connectivity,
        "painted_water": painted_water,
        "morphology": morphology,
        # Selected by the mode that actually ran, and FILLED from this frame.  A
        # single string cannot be true of both modes: "step" interpolates nothing,
        # "ramp" does.  And a corridor cost measured on another frame cannot be true
        # of this one, which is why the figures come from
        # connectivity["corridor_if_painted"] rather than from the template.  See
        # SPILL_NOT_DRAWN_DISCLOSURE_TEMPLATE.
        "spill_not_drawn": spill_not_drawn_disclosure(
            field.corridor_mode, field.connectivity.get("corridor_if_painted")
        ),
        "spill_not_drawn_mode": field.corridor_mode,
        # What the PICTURE did with the third state, beside what the connectivity
        # pass found: the two can differ by the de-speckle, which may close a hole
        # inside an unobservable body and hand those cells to the painted water.
        "unobservable_drawn": {
            "cells": int(unobservable_drawn_cells),
            "cells_classified": int(
                field.connectivity.get("unobservable", {}).get("cells", 0)
            ),
            "rgb": list(UNOBSERVABLE_RGB),
            "stripe_rgb": list(UNOBSERVABLE_STRIPE_RGB),
            "stripe_period_px": int(stripe_period_px),
            "stripe_period_output_px": int(UNOBSERVABLE_STRIPE_PERIOD_OUTPUT_PX),
            "note": (
                "cells the frame drew in the reserved striped colour that reads as "
                "unresolved — neither the water ramp nor the land ramp, and "
                "deliberately not hillshaded, because there is no terrain claim to "
                "shade. stripe_period_px is in RASTER cells and is scaled from "
                "stripe_period_output_px by this frame's own downsampling, because a "
                "period fixed in raster cells averages away under the figure's "
                "resampling and the region then arrives as a flat colour. "
                "cells_classified is what connectivity.unobservable found; "
                "cells is what survived being drawn, which is smaller when the "
                "morphological closing handed some of those cells to the painted "
                "water. Both are reported so the difference is visible rather than "
                "reconciled silently"
            ),
        },
        "valid_cells": valid,
        "vertical_frame": (
            "basin levels are consumed AS GIVEN, in the DEM's own frame. No frame "
            "conversion happens here; the DEM declares "
            f"{dem_vertical_frame!r} in its vertical_frame tag. The contract that "
            "governs the conversion is named in vertical_frame_contract"
        ),
        # A document reference is not a measurement, and a dated filename inside
        # prose is indistinguishable from one to anything that reads the sentence.
        # It is a field.
        "vertical_frame_contract": (
            "Paleo-Europe_3D/docs/superpowers/research/"
            "2026-07-30-vertical-datum-contract.md"
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
