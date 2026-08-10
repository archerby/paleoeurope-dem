#!/usr/bin/env python
"""
bake_black_sea.py
=================

Build a GIA-corrected paleo-DEM of the **Black Sea proper** per epoch from the
GMRT bathymetry/topography tile + ICE-6G_C bedrock orography, then render each
epoch's topo frame TWICE:

  * **WRONG**  — using the global eustatic (Spratt & Lisiecki 2016) sea level.
  * **CORRECT** — using the cited Pontic / Neoeuxinian basin level curve.

WHY
---
The Black Sea was a **sill-dammed lake**, decoupled from the global ocean until
reconnection at ~9 ka.  Inside that closed basin the land/sea boundary must be
driven by the basin's OWN level curve, NOT the global eustatic curve.  Whenever
the two curves disagree (e.g. 16 ka: global ~-98 m vs Pontic ~-30 m;
10 ka: global ~-25 m vs Pontic ~-80 m) the corrected coastline differs
dramatically.  This script visualises that scientific correction.

Because the whole scene is Black-Sea-proper water (one decoupled basin), a
single basin level is valid for the entire frame.

THE THRESHOLD IS NOT THE CURVE VALUE (vertical-datum contract, 2026-07-30)
-------------------------------------------------------------------------
``2026-06-21-black-sea-level-curve.csv`` is **mixed-frame**.  Rows older than each
column's reconnection date (gradual > 9.0 ka, catastrophic > 7.15 ka) are **F2 —
RSL as reported**: the *present-day measured depth* of a marker that stood at the
water surface at time t.  An F2 number is not the elevation of a palaeo water
surface in the present-day frame, so it cannot be compared with a GIA-corrected
``H_paleo`` as-is.  The conversion is

    T(t) = level(t) + Δz_solid(x_marker, t)          (F2 -> F1)

and it is applied **at the threshold, never in the curve** — the level column is
the cited literature value, and rewriting it would break
:func:`validate_basin_curve`'s F2 semantics (contract §8).  See
:func:`pontic_threshold`, which also states, per epoch and per scenario, WHICH of
the three rules produced the number.

Rows at or after the reconnection date are **F4 — global-mean sea level**, i.e.
an *intended ocean* threshold.  For those the contract's §2 derivation gives the
**exact** companion of this DEM: ``T(t) = -offset(t)``.  Exact, not approximate:
whatever local sea-surface residual is baked into ``Δz_engine`` cancels against
the same scalar.  The deliberately-WRONG comparison frames are the one place a
eustatic curve survives, and that is the whole point of them — see ``main()``.

TWO GUARDS ON READING THE CURVE (added 2026-08-03)
--------------------------------------------------
Both were latent substitutions, both are now refusals, and neither changes a number
this bake has ever produced — all 13 shipped epochs land on exact curve rows and on
exact ICE-6G epochs.

1. **The nearest-age lookup is bounded.**  :func:`basin_level` used to take whatever
   row minimised ``|age_ka - epoch|``, however far away.  At 16.75 ka the 17.0 and
   16.5 ka rows are equidistant and read -58.3 m and -30.0 m, so the threshold was
   28.3 m of DataFrame row order — and the two rows do not even carry the same kind
   of provenance.  Past :data:`_CURVE_AGE_TOL_KA` it now raises
   :class:`CurveAgeGapError`.
2. **Off-grid epochs no longer hit a raw ``FileNotFoundError``.**  ICE-6G ships
   0.5 ka steps to 21 ka, so 14.7 and 7.15 ka — rows this curve genuinely carries —
   used to die inside :func:`_open_bs_window`.  They are now blended linearly in
   time between the two bracketing ICE-6G epochs by :func:`_ice6g_bracket` (moved
   here from ``bake_khvalynian``, one implementation for both bakes) and the result
   is TAGGED ``ice6g_interpolated=True``.  An untagged interpolation is a
   substituted number.  Cross-check: 14.7 ka resolves to **-17.278 m**, which is the
   contract §5a-bis per-row value -17.28 m measured independently.

``unc_low`` / ``unc_high`` IS ONE PAIR PER ROW — NOT A PER-SCENARIO BAND
------------------------------------------------------------------------
Both curves carry ``unc_low`` / ``unc_high`` beside their level columns.  **They are
carried ONCE PER ROW and are read identically whichever scenario column is asked
for.**  Calling the result "the band around the *maximal* scenario" is a claim the
data does not make, and on the Caspian curve it is measurably false: at 16.0 ka the
pair is [+22.0, +50.0] while ``defensible`` reads +35.0 and ``maximal`` +48.0 — one
pair bracketing BOTH schools, i.e. an inter-school **scenario envelope**, not a
per-scenario uncertainty.  Measured on all 71 Caspian rows x all six level columns:
no level falls outside the pair, and 63 distinct pairs serve 71 rows.

So the object is named :class:`LevelEnvelope`, read by :func:`basin_level_envelope`,
and every result carries a measured ``scope`` naming which level columns the pair
actually brackets AT THAT ROW.  That scope is a measurement and not a constant: on
the **Pontic** curve the same pair fails to contain ``catastrophic`` at 9.0 / 8.5 /
8.0 / 7.5 ka, so there it is not an envelope over both columns and the field says so.

The edges are read off the SAME row as the level, converted with the SAME
``conversion_m`` and the same rule as the central (contract §8, edge for edge), and
:class:`MissingEnvelopeError` refuses one that is absent, NaN, inverted, DEGENERATE
(zero width — a "band" asserting certainty), or does not contain the central it was
checked against.  Nothing substitutes the central, nothing borrows the neighbouring
row (widths run 75 m to 5 m on this curve), and a synthesized low-scenario COLUMN is
forbidden outright.

Here the envelope is carried beside the threshold and nothing is thresholded on it, so
this module CATCHES the refusal and records it in ``envelope_status``.  The Khvalynian
bake does not catch, because there the envelope drives a spill verdict.  It fires on
the shipped data: the catastrophic column reads -100 m at 9.0 / 8.5 / 8.0 / 7.5 ka
against a pair written for the gradual column, and 9.0 and 7.5 ka are baked epochs.

PALEO-DEM
---------
paleo_dem(t) = H_modern + Δz_solid(t)   (present-datum solid-earth displacement)
Δz_solid(t)  = [Topo(t) - Topo(0)] - sea_surface_offset(t)   (ICE-6G_C Topo)
sea_surface_offset(t) = median ΔTopo over FAR-FIELD ICE-FREE TROPICAL CONTINENTAL
    LAND, computed from the GLOBAL Topo field
    (Orog is clamped to 0 over water -> GIA vanishes under the sea; bare ΔTopo
     double-counts the epoch sea surface. See paleoeurope.gia.ice6g_topo.)
ice_height = 0 everywhere (no ice sheet over the Black Sea).

THE DATUM IS GLOBAL, THE DISPLACEMENT IS LOCAL
----------------------------------------------
``gia_delta_subset`` hands the **global** ``Topo`` field to
:func:`~paleoeurope.gia.ice6g_topo.implied_sea_surface_offset` and only the
**regional** subset to :func:`~paleoeurope.gia.ice6g_topo.solid_displacement`.
Getting that backwards is precisely the bug that was retired on 2026-07-30: the
datum used to be the deep-ocean median, i.e. it was taken from cells that are
themselves displaced (the ocean floor stands ~+29 to +32 m higher at the LGM with
the water load absent), which zeroed a real signal and pushed it onto the
continents sign-flipped.  A regional cut cannot define a global far-field datum at
all and now raises rather than improvising.  Pinned by
``tests/test_bake_black_sea.py::
test_datum_is_taken_from_the_global_field_and_displacement_from_the_subset``.

THE ICE-SURFACE GUARD, AND THIS BAKE'S NaN POLICY
-------------------------------------------------
Where ``sftgif > 0`` ICE-6G ``Topo`` is the **ice surface, not bedrock**, and these
files carry no bedrock-under-ice field at all.  ``gia_delta_subset`` therefore
passes ``sftgif`` for BOTH endpoints through to ``solid_displacement``, which
returns **NaN** over ice.  NaN means *unknown*, end to end: there is no code path
here that turns it into a number.

``build_paleo_dem(epoch, gia_nan_policy=...)`` decides what to do with such a
hole, with the same two options and the same names as
:data:`paleoeurope.gia.correction_pipeline.GIA_NAN_POLICIES`, and deliberately
**no** ``"zero"``:

  * ``"raise"`` (default) — abort with :class:`UnknownGiaInBakeError`.  Correct for
    every bbox this script is pointed at today: the Black-Sea window is ice-free at
    all 13 shipped epochs (measured, not assumed — see
    ``test_black_sea_window_is_ice_free_at_every_shipped_epoch``), as is the Caspian
    bbox, so an unknown cell means a bug or the wrong epoch, not a fact about the
    world.
  * ``"nodata"`` — for a future bbox that genuinely straddles a palaeo-ice margin.
    Unknown cells are written as the raster's declared nodata
    (:data:`BS_NODATA_M`), never as 0 m of correction.

Masking runs BEFORE smoothing, via
:func:`~paleoeurope.gia.ice6g_topo.smooth_preserving_ice_mask` with the documented
:data:`~paleoeurope.gia.ice6g_topo.SMOOTHING_RIM_CELLS` rim; ``apply_gia_delta`` is
then called with ``sigma_blur=0.0`` so its own plain, NaN-propagating Gaussian does
not smooth a second time.  For an ice-free window this is bit-for-bit the historical
computation (same σ, same ``mode="reflect"`` boundary, same float64 promotion).

OUTPUT
------
  $PALEO_OUT_ROOT/black_sea_corrected/topo/correct/{gradual,catastrophic}/*.png
  $PALEO_OUT_ROOT/black_sea_corrected/topo/wrong/*.png
  (QC gallery over all three: scripts/make_bs_qc_gallery.py -> topo/bs_qc_gallery.html)

ROOTS
-----
Inputs and outputs are located through two environment variables so the same
script runs on the Windows workstation and on the Linux compute box (findus,
data under /mnt/tank):

  * ``PALEO_DATA_ROOT`` — the Paleo-Europe_3D data tree (rasters, curves, shapes).
    Default: the sibling ``Paleo-Europe_3D`` checkout next to this repo.
  * ``PALEO_OUT_ROOT``  — where render trees are written.
    Default: this repo's parent directory (the ``renders/`` root).

Both defaults are derived from this file's location, so nothing is bound to a
particular drive letter.

Run (repo NOT pip-installed):
    PYTHONPATH=$(pwd) python scripts/bake_black_sea.py
    PALEO_DATA_ROOT=/mnt/tank/Paleo-Europe_3D PALEO_OUT_ROOT=/mnt/tank/renders \\
        PYTHONPATH=$(pwd) python scripts/bake_black_sea.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

# Make the package importable when run directly (repo is not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

# Force the headless Agg backend BEFORE importing paleocostline_render — that
# module renders via matplotlib.pyplot and does not set a backend itself, so this
# guard is required for the render calls to work in a headless/background context.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from paleoeurope.gia.deformation import apply_gia_delta
from paleoeurope.gia.ice6g_topo import (
    SMOOTHING_RIM_CELLS,
    ice_surface_mask,
    implied_sea_surface_offset,
    smooth_preserving_ice_mask,
    solid_displacement,
)
from paleoeurope.viz.paleocostline_render import (
    RenderConfig,
    get_sea_level_for_epoch,
    load_paleocostlines,
    render_paleocostline_epoch,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# Roots are resolved ONCE at import.  Defaults are derived from this file's
# location (see module docstring) so no drive letter is baked in; override with
# PALEO_DATA_ROOT / PALEO_OUT_ROOT when the trees live elsewhere (e.g. findus:
# /mnt/tank).
_DEFAULT_DATA_ROOT = _REPO_ROOT.parents[1] / "Paleo-Europe_3D"
_DEFAULT_OUT_ROOT = _REPO_ROOT.parent

PALEO_DATA_ROOT = Path(
    os.environ.get("PALEO_DATA_ROOT") or _DEFAULT_DATA_ROOT
).resolve()
PALEO_OUT_ROOT = Path(os.environ.get("PALEO_OUT_ROOT") or _DEFAULT_OUT_ROOT).resolve()

_RAW_DIR = PALEO_DATA_ROOT / "data" / "raw"

BS_DEM = _RAW_DIR / "GEBCO" / "bs_gmrt_bsproper.tif"
ICE6G_DIR = _RAW_DIR / "ICE6G"
PONTIC_CSV = (
    PALEO_DATA_ROOT
    / "docs"
    / "superpowers"
    / "research"
    / "2026-06-21-black-sea-level-curve.csv"
)
# THE global-eustatic (F4) curve.  This is the PUBLISHED series and nothing else:
# ``spratt_lisiecki_2016_full.csv`` is a bit-exact transcription of the
# ``SeaLev_shortPC1`` + ``SeaLev_shortPC1_err_sig`` columns of the NOAA/WDS-Paleo
# distribution ``spratt2016.txt`` (study 19982), i.e. the scaled first principal
# component of the seven 0-430 ka reconstructions of
#   Spratt, R. M. & Lisiecki, L. E. (2016). A Late Pleistocene sea level stack.
#   Climate of the Past 12, 1079-1092.  doi:10.5194/cp-12-1079-2016
# 0-30 ka at 1 ka steps.  Pinned row-by-row against the NOAA file by
# tests/test_bake_black_sea.py::test_spratt_csv_is_bit_exact_with_the_noaa_shortpc1_column.
#
# WHAT THIS REPLACED, AND WHY IT MATTERS.  Two files named
# ``spratt_lisiecki_2016_simplified.csv`` ship in this project (0-20 ka at 1 ka
# under PALEO_DATA_ROOT, 0-26 ka at 0.5 ka in this repo).  NEITHER is Spratt &
# Lisiecki: both are hand-drawn round-number curves whose values appear nowhere
# in the published stack (16 ka: -105 m in the file vs -98.06 m published;
# 10 ka: -50 m vs -24.59 m).  22 ka was not covered at all and got CLAMPED to the
# 20 ka row.  This module's own docstring above quotes -98 m and -25 m — the
# PUBLISHED numbers — so the data now matches the intent that was written down
# all along.
#
# THE PRESENT DATUM IS NOT ZERO, AND IT IS NOW SUBTRACTED — decision taken by the
# owner 2026-07-31, recorded here rather than inferred from the arithmetic.  The
# published stack reads **+8.49 m at 0 ka** (1 sigma +/-5.23 m, so 0 m is well
# inside its own uncertainty); the retired hand-drawn file passed through 0.00 m by
# construction.  :func:`spratt_present_datum_m` still reports the PUBLISHED value;
# :func:`eustatic_level_m` is the 0-referenced series, and it is what every frame
# thresholded on this curve uses.
#
#   WHY SUBTRACT.  ``offset(t)`` is 0 at 0 ka by construction and the paleo-DEM is
#   a PRESENT-DATUM surface (``H_paleo(x, 0) == H_modern(x)`` exactly), so an F4
#   threshold laid on it must also be 0 at t = 0.  Left as published, the 0 ka and
#   2 ka comparison frames render with sea level standing ABOVE the modern coast
#   (+8.49 m and +4.49 m), which reads as a bug to every viewer and is not what the
#   authors' Holocene scaling residual means.
#
#   WHAT IT COSTS, STATED PLAINLY.  This is a FRAMING decision, not a correction of
#   the data: Spratt & Lisiecki scaled the stack against both Holocene and LGM
#   estimates, so neither end is privileged and the residual is not an error to be
#   removed.  Subtracting 8.49 m moves the 22 ka LGM frame from -125.82 m to
#   **-134.31 m** (and the stack's own deepest node, 24 ka, from -130.00 to
#   -138.49 m) — i.e. it buys agreement at t = 0 by pushing the glacial end 8.5 m
#   deeper than the published curve.  ``2026-07-30-vertical-datum-contract.md`` §5b
#   declined to make this call in passing; it is made here, on purpose, in one
#   place, and it is pinned by
#   ``test_eustatic_series_is_zero_referenced_to_present`` /
#   ``test_zero_referencing_moves_the_lgm_frame_to_the_documented_value``.
SPRATT_CSV = _RAW_DIR / "sea_level_curves" / "spratt_lisiecki_2016_full.csv"

# Widest acceptable distance (years) between a requested epoch and the nearest row
# of the eustatic curve.  ``get_sea_level_for_epoch`` is NEAREST-NEIGHBOUR, so this
# IS the resolution of the lookup; 500 yr is half a step of the published 1 ka
# series, i.e. "the epoch is genuinely bracketed".  Anything wider is either a
# truncated curve (a clamp) or a curve too coarse to answer the question.
SPRATT_MAX_NODE_GAP_YR = 500.0
PALEOCOASTLINES_SHP = (
    _RAW_DIR
    / "paleocostline"
    / "Paleocoastlines"
    / "Paleocoastlines"
    / "Paleocoastlines.shp"
)

OUT_BASE = PALEO_OUT_ROOT / "black_sea_corrected" / "topo"
# Intermediate per-epoch GeoTIFFs are a rebuildable cache; they stay inside the
# repo's own outputs/ tree (already repo-relative, never drive-absolute).
PALEO_TIF_DIR = _REPO_ROOT / "outputs" / "_bs_paleo"

# DATUM-SANITY window (m vs present global SL) per closed basin, consulted by
# _check_datum_sanity().  Adding a basin = adding a row here; the logic is generic.
#   pontic  — the cited Neoeuxinian/Pontic curve never rises above present SL.
#   caspian — deep Khvalynian lowstands reach ~-100 m; the upper bound admits the
#             defensible +35 m highstand (HESS 2024) and the UNDATED +48/+50 m cap.
#
# This window catches datum blow-ups and unit errors ONLY.  It is NOT a sign
# guard: the Caspian legitimately takes both signs, so every rung of the
# Khvalynian ladder (+48/+50, +34/+36, +20/+22, +14/+15, +4/+6, 0/-2, -5/-6,
# -10/-12, -16/-17) has a mirror twin that also lands inside -160..+60.  Sign
# flips are detected at CURVE level — see validate_basin_curve().
_DATUM_SANITY_RANGE_M: dict[str, tuple[float, float]] = {
    "pontic": (-150.0, 1.0),
    "caspian": (-160.0, 60.0),
}

# Present-day (0 ka) surface of each basin as (anchor_m, tolerance_m).  These are
# measured, not modelled:
#   pontic  — connected to the world ocean, so 0 m by definition of the datum.
#   caspian — the modern Caspian Sea surface sits ~28 m BELOW global sea level;
#             the tolerance absorbs its metre-scale historical drift.
_PRESENT_DAY_ANCHOR_M: dict[str, tuple[float, float]] = {
    "pontic": (0.0, 2.0),
    "caspian": (-28.0, 6.0),
}

# How close a curve row must sit to 0 ka to count as the present-day anchor.
_ANCHOR_AGE_TOL_KA = 0.25

# How close a curve row must sit to a REQUESTED EPOCH before its level may be read
# as that epoch's level (ka).  This is the guard on the nearest-age lookup, and it
# is deliberately tight enough that "nearest" means "the row for this epoch":
#
#   * both shipped curves write ``age_ka`` to two decimals, i.e. 0.01 ka is their
#     own age resolution, so this tolerance absorbs the representation and nothing
#     else;
#   * the tightest ADJACENT row spacing is 0.05 ka in the Caspian curve and 0.10 ka
#     in the Pontic one, so 0.01 ka can never reach a second row and the lookup
#     cannot be ambiguous by a hair.
#
# WHAT IT COSTS AND WHY THAT IS THE POINT: 16.75 ka is 0.25 ka from BOTH the 17.0
# and the 16.5 ka Pontic rows, whose levels are -58.3 m and -30.0 m.  Without a
# tolerance ``idxmin`` silently returns the first of the two in DataFrame order —
# 28.3 m of threshold decided by row order — and the two rows do not even carry the
# same provenance (a seismic-wedge lowstand locus vs the Bosphorus sill marker), so
# there is no defensible way to pick one.  With the tolerance that epoch RAISES.
_CURVE_AGE_TOL_KA = 0.01

# The Manych spillway sill that the Khvalynian Caspian had to overtop to drain
# west into the Pontic basin (m vs present global SL).
#
# PROVENANCE: measured 2026-07-28 from GEBCO 2024 (15 arcsec grid) at the
# saddle point 44.119 E / 45.773 N — the lowest point along the Kuma-Manych
# depression between the Caspian and Pontic basins.  Not a literature quote:
# this is the number our own DEM gives, so a curve is checked against the same
# topography the render uses.
#
# This is the THRESHOLD Check A tests against, not merely a number quoted in the
# message.  A Khvalynian *spillway* curve that peaks below it never spills, and
# is therefore not describing the event this render is about.
MANYCH_SILL_M = 23.0

# Check B — physical continuity.  Maximum |dlevel| / |dage| (m per ka) allowed
# between age-sorted adjacent samples of a closed-basin curve.  A closed-basin
# water surface cannot jump arbitrarily; a sign error confined to a BLOCK of
# rows leaves a large discontinuity at each edge of that block, and that is what
# this catches.
#
# The constants are per basin because ONE constant provably cannot do both jobs:
#
#   pontic  — MEASURED on the shipped curve
#             (docs/superpowers/research/2026-06-21-black-sea-level-curve.csv),
#             both scenarios, 2026-07-28:
#               gradual      max 100.000 m/ka  (9.4 -> 9.5 ka, -30 -> -40 m:
#                                               the reconnection at the sill)
#               catastrophic max 226.857 m/ka  (7.15 -> 7.5 ka, -20.6 -> -100 m:
#                                               the Ryan-Pitman catastrophic
#                                               reconnection; note the 7.15 ka
#                                               anchor also sits on a 14C axis,
#                                               which inflates this rate)
#             Limit 340 m/ka = 1.50x the measured worst case.  The meltwater
#             rise into 16.5 ka (56.8 m/ka) clears it by 6x.
#
#   caspian — no real curve exists yet, so the limit is anchored to the fastest
#             DOCUMENTED Caspian event rather than to the ladder's own spacing:
#             kurbanov2020's Early Khvalynian rise, "~-150 to >+40 m a.s.l."
#             across the 17-13 ka window = ~47.5 m/ka (~62.5 m/ka on the
#             Bezrodnykh +100 m variant).  Limit 100 m/ka = 2.1x that (1.6x on
#             the Bezrodnykh variant), and LETHAL to the arslanov2016 subset
#             flip AT 0.5-KA SPACING: negating the +48/+50 cap next to the
#             +34/+36 Talginka rung is an 82-86 m step across 0.5 ka =
#             164-172 m/ka, i.e. 1.64-1.72x ABOVE the limit.
#
# WHY NOT ONE CONSTANT: the Pontic curve genuinely contains a 226.857 m/ka
# event, while the attack we must kill is only 164 m/ka.  Any single constant
# loose enough to admit the real curve is also loose enough to admit the typo.
# Stating that plainly rather than picking a number that satisfies neither.
#
# THESE VALUES MUST STAY IDENTICAL to _MAX_LEVEL_RATE_M_PER_KA in
# Paleo-Europe_3D/scripts/export_bs_levels.py.  They diverged once already
# (engine 340/120 vs viewer 300/100), which meant a curve the viewer accepted
# the engine rejected.  Reconciled 2026-07-28 by taking the safer margin on
# each side: the Pontic limit is the more permissive of the two (a false
# REJECTION would break the one basin that already works), the Caspian limit is
# the tighter and better-sourced of the two (it cites a real published rise
# rate instead of the fixture's own spacing).
#
# RE-MEASURE THE CASPIAN VALUE when _build_caspian_curve.py lands: if the real
# curve contains a genuine Manych drawdown above 100 m/ka the constant must go
# up, and the arslanov2016 regression test must then be re-checked for lethality.
_MAX_LEVEL_RATE_M_PER_KA: dict[str, float] = {
    "pontic": 340.0,
    "caspian": 100.0,
}

# On the ICE-6G 0.5-ka grid (only ages with an I6_C.VM5a_10min.<age>.nc file),
# chosen to sample the cited curve's load-bearing features: 22 LGM min (-120),
# 16.5 spill (-30), 12/11/10 lowstand middle, 9.5/9 reconnection, Holocene refill.
EPOCHS_KA = [22, 18, 16.5, 14.5, 12, 11, 10, 9.5, 9, 7.5, 5, 2, 0]

# Reconnection scenarios exposed by the cited curve (carry both, assert neither).
SCENARIOS = ("gradual", "catastrophic")

# Region buffer (deg) for the GIA subset around the BS bbox 27-42E / 40-48N.
LAT_MIN, LAT_MAX = 40.0, 48.0
LON_MIN, LON_MAX = 27.0, 42.0
BUF = 1.0

logger = logging.getLogger(__name__)

# ICE-6G icemask variable name.  Where it is > 0 the shipped ``Topo`` is the ICE
# SURFACE and there is no bedrock field to fall back on, so Δz is UNKNOWN there.
ICE6G_ICEMASK_VAR = "sftgif"

# Gaussian σ (in ICE-6G cells) applied to Δz before interpolation onto the tile.
# Historically this was passed to apply_gia_delta as ``sigma_blur``; it is now
# applied here, through smooth_preserving_ice_mask, so that masking precedes
# smoothing.  The VALUE must not change: 2.0 is what every shipped bs_*.tif used.
GIA_SIGMA_BLUR = 2.0

# Nodata written into the per-epoch paleo-DEM GeoTIFF.  Named so the raster header
# and the value substituted for an unknown cell cannot drift apart.
BS_NODATA_M = -9999.0

# What build_paleo_dem() does when the GIA correction is UNKNOWN over part of the
# window (ICE-6G 'Topo' is the ice surface there and these files ship no
# bedrock-under-ice field).
#
#   "raise"  — abort.  Correct for every bbox this script is pointed at today: the
#              Black-Sea window is ice-free at all 13 shipped epochs (measured; see
#              tests/test_bake_black_sea.py::
#              test_black_sea_window_is_ice_free_at_every_shipped_epoch) and so is
#              the Caspian bbox, so an unknown cell means a bug or the wrong epoch,
#              not a fact about the world.
#   "nodata" — the legitimate case for a FUTURE bbox that genuinely straddles a
#              palaeo-ice margin (Europe / Doggerland at 21 ka).  Unknown cells are
#              written as BS_NODATA_M and logged, never as 0 m of correction.
#
# There is deliberately NO "zero": a zero Δz reads as "this ground did not move"
# over exactly the ground that moved most.
#
# THESE MUST STAY IDENTICAL to paleoeurope.gia.correction_pipeline.GIA_NAN_POLICIES.
# They are duplicated rather than imported so that a bake script does not drag in
# the whole tile pipeline (Ice6gLoader, ICE-7G, rasterio windowed reads) for one
# tuple and one exception class; the drift guard is
# tests/test_bake_black_sea.py::test_bake_nan_policies_match_the_correction_pipeline.
BAKE_GIA_NAN_POLICIES: tuple[str, ...] = ("raise", "nodata")
BAKE_GIA_NAN_POLICY = "raise"


class UnknownGiaInBakeError(ValueError):
    """The bake was asked to write a raster whose GIA correction is unknown.

    The bake counterpart of
    :class:`paleoeurope.gia.correction_pipeline.UnknownGiaCorrectionError`, raised
    when ``gia_nan_policy="raise"`` (the default) and part of the window is
    glaciated at the epoch or today.  ICE-6G ``Topo`` is the **ice surface** there
    and the shipped 10 arcmin files carry no bedrock-under-ice field at all, so
    adding that cell to a modern DEM is meaningless and zero-filling it is worse.

    Resolve it by restricting the bbox/epoch to ice-free ground (what the Black-Sea
    and Caspian windows already are), by sourcing Δz from ICE-7G's
    bedrock-deformation field, or by passing ``gia_nan_policy="nodata"`` to record
    the hole honestly as nodata.  It will not be filled with 0 m.
    """


class SprattCoverageError(ValueError):
    """The eustatic curve cannot answer for an epoch the bake is about to render.

    ``get_sea_level_for_epoch`` is a NEAREST-NEIGHBOUR lookup that never fails: off
    the end of a curve it returns the end row, and its own truncation warning only
    fires above a 1000 yr miss.  A 0-20 ka curve asked for 21 ka therefore misses
    by exactly 1000 yr and comes back CLAMPED, SILENTLY — which is how the bake
    ran for months against a curve that stopped 2 ka short of its own oldest
    epoch.  This check runs before the first frame and converts that silence into
    a stop.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spratt_ages_bp(csv_path: Path | None = None) -> np.ndarray:
    """Ages (years BP) of the eustatic curve, ascending.  Raises if absent."""
    csv_path = Path(csv_path) if csv_path is not None else SPRATT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Sea-level CSV not found: {csv_path}")
    df = pd.read_csv(
        csv_path, comment="#", header=None,
        names=["age_bp", "sea_level_m", "uncertainty_m"],
    )
    return np.sort(df["age_bp"].to_numpy(dtype=float))


def spratt_present_datum_m(csv_path: Path | None = None) -> float:
    """Value the eustatic curve takes at 0 ka **as published**, in metres.

    Zero for a curve referenced to present; **+8.49 m** for the published Spratt &
    Lisiecki ``shortPC1`` stack, whose Holocene scaling residual is inside its own
    1 sigma (+/-5.23 m at 0 ka).  This function deliberately reports the RAW
    published value and never the re-referenced one, so that the size of the
    framing decision stays visible; :func:`eustatic_level_m` is what subtracts it.
    """
    csv_path = Path(csv_path) if csv_path is not None else SPRATT_CSV
    return float(get_sea_level_for_epoch(0.0, csv_path))


def eustatic_level_m(epoch: float, csv_path: Path | None = None) -> float:
    """0-referenced global-mean (F4) sea level for *epoch*, in metres.

    ``published(epoch) - published(0 ka)``.  This is THE eustatic number every
    frame in this script thresholds on; nothing else may call
    :func:`~paleoeurope.viz.paleocostline_render.get_sea_level_for_epoch` for a
    threshold, because that returns the un-referenced published value and would
    put sea level 8.49 m above the modern coast at t = 0.

    Rationale, cost and provenance of the subtraction: see the ``SPRATT_CSV``
    comment block.  Lookup is NEAREST-NEIGHBOUR on a 1 ka grid, so
    :func:`check_spratt_coverage` must have passed first or the answer may be a
    silent clamp.
    """
    csv_path = Path(csv_path) if csv_path is not None else SPRATT_CSV
    published = float(get_sea_level_for_epoch(float(epoch), csv_path))
    return published - spratt_present_datum_m(csv_path)


def check_spratt_coverage(
    epochs_ka,
    csv_path: Path | None = None,
    max_node_gap_yr: float = SPRATT_MAX_NODE_GAP_YR,
) -> None:
    """Raise unless every epoch is genuinely bracketed by the eustatic curve.

    Two distinct failures, one exception:

    * **clamping** — the epoch lies outside the curve's own age range, so the
      nearest-neighbour lookup returns an endpoint that has nothing to do with it;
    * **resolution** — the epoch is inside the range but the nearest row is more
      than *max_node_gap_yr* away, so the "lookup" is really an unstated
      extrapolation across a gap.

    Raises
    ------
    SprattCoverageError
        Listing every offending epoch, the nearest row, and the miss in years.
    """
    ages = _spratt_ages_bp(csv_path)
    lo, hi = float(ages[0]), float(ages[-1])
    problems: list[str] = []
    for epoch in epochs_ka:
        want = float(epoch) * 1000.0
        nearest = float(ages[int(np.abs(ages - want).argmin())])
        gap = abs(nearest - want)
        if want < lo or want > hi:
            problems.append(
                f"{epoch} ka ({want:.0f} BP) is OUTSIDE the curve "
                f"[{lo:.0f}, {hi:.0f}] BP — the lookup would clamp to "
                f"{nearest:.0f} BP (miss {gap:.0f} yr)"
            )
        elif gap > max_node_gap_yr:
            problems.append(
                f"{epoch} ka ({want:.0f} BP) has no row within "
                f"{max_node_gap_yr:.0f} yr — nearest is {nearest:.0f} BP "
                f"(miss {gap:.0f} yr)"
            )
    if problems:
        raise SprattCoverageError(
            "eustatic curve "
            f"{Path(csv_path) if csv_path is not None else SPRATT_CSV} "
            "cannot answer for every requested epoch:\n  "
            + "\n  ".join(problems)
        )


def _epoch_filename(epoch: float) -> str:
    """ICE-6G filename token for an epoch.

    Integers use the bare int (``16`` -> ``...16.nc``); half-ka use one decimal
    (``0.5`` -> ``...0.5.nc``).  ``0`` -> ``...0.nc``.
    """
    if float(epoch).is_integer():
        return f"I6_C.VM5a_10min.{int(epoch)}.nc"
    return f"I6_C.VM5a_10min.{epoch}.nc"


# ---------------------------------------------------------------------------
# The shipped ICE-6G age grid, and linear-in-time interpolation between it
# ---------------------------------------------------------------------------
#
# MOVED HERE 2026-08-03 from ``bake_khvalynian``, which is where it was written and
# where it is still re-exported from.  It moved because ``pontic_threshold`` needs
# the same mechanism and this module cannot import that one (it is the other way
# round), and because two copies of a time-interpolation rule is two rules that can
# disagree about what "linear in time" means.  The behaviour is unchanged: an epoch
# that IS a shipped ICE-6G age returns ``(age, age, 0.0)`` and no interpolation code
# runs, so every on-grid number in either bake is bit-identical to before.


@lru_cache(maxsize=8)
def _ice6g_ages_in(directory: Path) -> tuple[float, ...]:
    """Ages (ka) with an ICE-6G file in *directory*, ascending.  Cached per DIR.

    Keyed on the directory rather than cached once, because :data:`ICE6G_DIR` is a
    module attribute that tests (and a future compute box) legitimately repoint: a
    single-slot cache would answer a question about one distribution with the
    contents of another, which is the same class of error as reading a level off the
    wrong row.
    """
    ages: list[float] = []
    for path in Path(directory).glob("I6_C.VM5a_10min.*.nc"):
        token = path.name[len("I6_C.VM5a_10min.") : -len(".nc")]
        try:
            ages.append(float(token))
        except ValueError:  # pragma: no cover - a stray file, not an age
            continue
    if not ages:
        raise FileNotFoundError(f"no ICE-6G files under {directory}")
    return tuple(sorted(ages))


def ice6g_ages_ka() -> tuple[float, ...]:
    """Ages (ka) for which an ICE-6G file exists, ascending.

    Read from the directory, never assumed: the distribution steps 0.5 ka to 21 ka
    and 1 ka beyond, and hard-coding that would silently mis-bracket if a file were
    added or missing.
    """
    return _ice6g_ages_in(ICE6G_DIR)


def _ice6g_bracket(epoch: float) -> tuple[float, float, float]:
    """``(lo, hi, w)`` such that ``f(epoch) = (1-w)*f(lo) + w*f(hi)``.

    ``lo == hi`` and ``w == 0.0`` when *epoch* is itself a shipped ICE-6G age, so an
    on-grid epoch is computed EXACTLY and no interpolation code runs.  Off-grid, the
    blend is linear in time between the two exactly-computed neighbours.

    Raises
    ------
    ValueError
        Outside the shipped range.  Extrapolating a GIA field past the end of the
        model is not a small error; it is an invented epoch.
    """
    ages = np.asarray(ice6g_ages_ka(), dtype=float)
    want = float(epoch)
    exact = np.isclose(ages, want, rtol=0.0, atol=1e-9)
    if exact.any():
        age = float(ages[exact][0])
        return age, age, 0.0
    if want < ages[0] or want > ages[-1]:
        raise ValueError(
            f"{want:g} ka is outside the shipped ICE-6G range "
            f"[{ages[0]:g}, {ages[-1]:g}] ka — the GIA field would have to be "
            "extrapolated, which invents an epoch rather than approximating one"
        )
    hi_i = int(np.searchsorted(ages, want))
    lo, hi = float(ages[hi_i - 1]), float(ages[hi_i])
    return lo, hi, (want - lo) / (hi - lo)


def ice6g_is_interpolated(epoch: float) -> bool:
    """Is this epoch's GIA field a blend of two shipped ICE-6G epochs?

    ``True`` is not a small caveat, which is why it is carried as its own field on
    :class:`BasinThreshold` rather than left implicit in a note: an UNTAGGED
    interpolation is indistinguishable downstream from a number the model actually
    produced.  ICE-6G ships 0.5 ka steps to 21 ka, so every quarter-ka epoch of the
    Khvalynian window — and 14.7 and 7.15 ka on the Pontic curve — is one of these.
    """
    lo, hi, _w = _ice6g_bracket(epoch)
    return lo != hi


def _ice6g_time_note(epoch: float) -> str:
    """Human/tag-readable statement of how *epoch* got its GIA field."""
    lo, hi, w = _ice6g_bracket(epoch)
    if lo == hi:
        return f"exact ICE-6G epoch {lo:g} ka (no time interpolation)"
    return (
        f"linear in time between ICE-6G {lo:g} and {hi:g} ka, w={w:.4f} "
        "(each endpoint computed exactly; the blend is the only approximation)"
    )


def _check_datum_sanity(level: float, basin: str, where: str) -> float:
    """Reject *level* if it is outside ``_DATUM_SANITY_RANGE_M[basin]``.

    This is a DATUM-SANITY bound, nothing more.  It catches datum blow-ups and
    unit errors — a level of -3000 m (metres read as decimetres, or an
    ellipsoid/geoid mix-up) or +900 m — and it catches an out-of-window value in
    a basin that only ever takes one sign.

    **This cannot detect a sign flip in a two-signed basin.**  The Caspian is
    two-signed: the Khvalynian ladder runs from +48/+50 m down to -16/-17 m, so
    negating any rung lands inside the same symmetric window and is accepted.  A
    magnitude window can never see that.  Curve-level validation does — see
    :func:`validate_basin_curve`.

    *where* is a short human label (``"14.0 ka"``, ``"row 7"``) used in the
    message.  Raises ``ValueError`` rather than asserting — ``assert`` is
    stripped under ``python -O`` and this guard protects a scientific invariant.
    """
    if basin not in _DATUM_SANITY_RANGE_M:
        raise ValueError(
            f"unknown basin {basin!r}; known basins={sorted(_DATUM_SANITY_RANGE_M)}"
        )
    lo, hi = _DATUM_SANITY_RANGE_M[basin]
    if not lo <= level <= hi:
        raise ValueError(
            f"{basin} level {level:+.1f} m at {where} is outside the datum-sanity "
            f"window {lo:+.1f}..{hi:+.1f} m — datum / unit blow-up (curve doc §3). "
            "This bound does NOT check the sign of the curve; "
            "validate_basin_curve() does that."
        )
    return level


def validate_basin_curve(
    df: pd.DataFrame,
    basin: str,
    scenario: str = "gradual",
    require_spillway_highstand: bool = True,
) -> None:
    """Validate a whole closed-basin level curve; call ONCE at load time.

    This is the real sign-flip guard.  A per-scalar magnitude window cannot see a
    sign flip in a basin that legitimately takes both signs (see
    :func:`_check_datum_sanity`), so the discriminating information has to come
    from the curve as a whole.  Checks, in the order they run:

    1. **Mirror test** — if a near-present value fits the known present-day
       anchor BETTER when negated, the curve is flipped.  Basin-agnostic: it
       needs no hand-maintained ladder, only the anchor.
    2. **Anchor check** — every near-present value must match the measured
       present-day surface (Pontic 0 m ±2, Caspian -28 m ±6).
    3. **Check A, spillway highstand** (Caspian only, opt-out via
       *require_spillway_highstand*) — a curve describing the Khvalynian
       transgression must PEAK AT OR ABOVE the measured Manych sill,
       ``MANYCH_SILL_M`` = +23.00 m (GEBCO 2024, 44.119 E / 45.773 N).  Below
       that the basin never spills west into the Pontic, so the curve is not
       describing the event this render is about.  The threshold and the message
       are the same number on purpose.
    4. **Check B, physical continuity** — ``|dlevel| / |dage|`` between
       age-sorted ADJACENT samples must stay under
       ``_MAX_LEVEL_RATE_M_PER_KA[basin]``.  A closed-basin surface cannot jump
       arbitrarily, so a sign error confined to a BLOCK of rows shows up as a
       large discontinuity at each edge of the block.  This is the only check
       that reads the whole curve pairwise, and it is what catches the
       arslanov2016 typo (only the +48/+50 rows negated).
    5. **Datum sanity, every row** — blow-ups and unit errors anywhere.

    Sign diagnosis runs BEFORE datum sanity on purpose: a flipped curve usually
    trips the datum window as well, and "out of range" would be a misleading
    diagnosis for what is actually a sign error.

    KNOWN LIMIT — what each check actually reads, and what stays invisible:

    * Mirror and anchor read ONLY the rows within ``_ANCHOR_AGE_TOL_KA`` of
      0 ka (all of them, not just the nearest).  They say nothing whatever about
      any older row.
    * Check A reads ONLY ``max(curve)`` — a single extremum.  It cannot see a
      flip that leaves some rung above +23 m standing.
    * Check B reads every ADJACENT age-sorted pair, so it sees block edges, but
      it is blind to any error that is *smooth*: a whole-curve negation (no edge
      at all — caught by mirror/anchor/datum instead), a uniform datum shift, a
      unit error, and — importantly — a sign flip of a block whose values are
      small enough that the induced step stays under the per-basin limit.
      Concretely, for the Caspian limit of 120 m/ka at the ladder's 0.5-ka
      spacing, negating a rung of |level| < ~30 m in a locally flat neighbourhood
      is INVISIBLE to Check B; it is covered only if it also drags ``max`` below
      the sill (Check A) or moves the anchor.  We do not claim that class.
    * Duplicate ages (``dage == 0``) yield no defined rate and are SKIPPED by
      Check B.  Duplicates inside the anchor window are still fully checked by
      mirror/anchor; duplicates elsewhere are not covered.
    * Where the anchor is exactly 0 m (the Pontic basin, connected to the world
      ocean) the mirror test is structurally blind — negating 0 gives -0.0, so
      both fits are 0 m.  A flipped Pontic curve is rejected by the datum window
      instead, which works because that basin IS single-signed and its window is
      asymmetric (-150..+1 m).  It is the COMBINATION that covers both basin
      types; no single check suffices.

    Raises ``ValueError`` on the first violation; returns ``None`` when the curve
    is usable.  ``_build_caspian_curve.py`` is required to pass this.
    """
    if basin not in _DATUM_SANITY_RANGE_M:
        raise ValueError(
            f"unknown basin {basin!r}; known basins={sorted(_DATUM_SANITY_RANGE_M)}"
        )
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )

    curve = pd.to_numeric(df[col], errors="coerce")
    ages = pd.to_numeric(df["age_ka"], errors="coerce")
    ok = curve.notna() & ages.notna()
    if not ok.any():
        raise ValueError(f"{basin}/{scenario} curve has no usable (age, level) rows")
    curve, ages = curve[ok], ages[ok]

    # --- present-day anchor rows -------------------------------------------
    # EVERY row inside the tolerance is examined, not just min(|age|): a
    # duplicated near-present row carrying a flipped value used to hide behind
    # the untouched one.
    anchor_m, anchor_tol = _PRESENT_DAY_ANCHOR_M[basin]
    age_gap = ages.abs()
    in_window = age_gap <= _ANCHOR_AGE_TOL_KA
    if not bool(in_window.any()):
        idx_near = age_gap.idxmin()
        raise ValueError(
            f"{basin}/{scenario} curve has no 0 ka anchor row (nearest age is "
            f"{float(ages.loc[idx_near]):.2f} ka) — cannot validate the sign of "
            "the curve without the present-day anchor"
        )
    anchor_idx = list(ages.index[in_window])

    # --- 1. mirror test (basin-agnostic), over ALL anchor-window rows -------
    for idx0 in anchor_idx:
        at0 = float(curve.loc[idx0])
        fit, flipped_fit = abs(at0 - anchor_m), abs(-at0 - anchor_m)
        if fit > flipped_fit:
            raise ValueError(
                f"{basin}/{scenario} curve at {float(ages.loc[idx0]):g} ka (inside "
                f"the 0 ka anchor window) is {at0:+.1f} m but the measured "
                f"present-day anchor is {anchor_m:+.1f} m; the curve fits the "
                f"anchor better when negated (|{at0:+.1f} - {anchor_m:+.1f}| = "
                f"{fit:.1f} m vs |{-at0:+.1f} - {anchor_m:+.1f}| = "
                f"{flipped_fit:.1f} m) — this is a SIGN FLIP, not a highstand"
            )

    # --- 2. anchor check, over ALL anchor-window rows -----------------------
    for idx0 in anchor_idx:
        at0 = float(curve.loc[idx0])
        fit = abs(at0 - anchor_m)
        if fit > anchor_tol:
            raise ValueError(
                f"{basin}/{scenario} curve at {float(ages.loc[idx0]):g} ka (inside "
                f"the 0 ka anchor window) is {at0:+.1f} m, off the measured "
                f"present-day anchor {anchor_m:+.1f} m by {fit:.1f} m "
                f"(tolerance {anchor_tol:.1f} m) — wrong datum or wrong basin"
            )

    # --- 3. Check A: the highstand must clear the Manych sill (Caspian) -----
    if basin == "caspian" and require_spillway_highstand:
        peak = float(curve.max())
        if not peak >= MANYCH_SILL_M:
            raise ValueError(
                f"caspian/{scenario} curve highstand is only {peak:+.1f} m, below "
                f"the measured Manych sill {MANYCH_SILL_M:+.2f} m (GEBCO 2024, "
                "44.119 E / 45.773 N): a Khvalynian transgression must clear the "
                "sill to spill west into the Pontic basin, so this curve is "
                "flipped, squashed, or is not describing a spilling interval — "
                "pass require_spillway_highstand=False for a deliberately "
                "non-spilling Caspian interval"
            )

    # --- 4. Check B: physical continuity between adjacent samples -----------
    # Non-finite rows are filtered HERE ONLY, so that +inf still reaches the
    # datum window below instead of being silently dropped from the curve.
    a_all = ages.to_numpy(dtype=float)
    l_all = curve.to_numpy(dtype=float)
    finite = np.isfinite(a_all) & np.isfinite(l_all)
    a_fin, l_fin = a_all[finite], l_all[finite]
    order = np.argsort(a_fin, kind="stable")
    a_sorted, l_sorted = a_fin[order], l_fin[order]
    max_rate = _MAX_LEVEL_RATE_M_PER_KA[basin]
    for i in range(len(a_sorted) - 1):
        d_age = float(a_sorted[i + 1] - a_sorted[i])
        if d_age <= 0.0:
            continue  # duplicate ages: no defined rate (see KNOWN LIMIT)
        d_level = float(l_sorted[i + 1] - l_sorted[i])
        rate = abs(d_level) / d_age
        if rate > max_rate:
            raise ValueError(
                f"{basin}/{scenario} curve jumps {d_level:+.1f} m between "
                f"{float(a_sorted[i]):g} ka ({float(l_sorted[i]):+.1f} m) and "
                f"{float(a_sorted[i + 1]):g} ka ({float(l_sorted[i + 1]):+.1f} m) "
                f"— {rate:.1f} m/ka, over the {basin} continuity limit of "
                f"{max_rate:.1f} m/ka.  A closed-basin surface cannot move that "
                "fast; a discontinuity this large at a block edge is the "
                "signature of a SIGN FLIP confined to part of the curve (the "
                "arslanov2016 '-48 to -50' typo for the +48/+50 rungs)"
            )

    # --- 5. datum sanity, every row ----------------------------------------
    for age, level in zip(ages, curve):
        _check_datum_sanity(float(level), basin, f"{float(age):g} ka")


class CurveAgeGapError(ValueError):
    """The curve carries no row at this epoch, and a neighbour's will not be taken.

    The nearest-age lookup used to be unconditional: whatever row minimised
    ``|age_ka - epoch|`` supplied the level, however far away it was and however
    the tie was broken.  On the shipped Pontic curve that is worth **28.3 m** at
    16.75 ka (the 17.0 ka row's -58.3 m and the 16.5 ka row's -30.0 m are
    equidistant, so ``idxmin`` returns whichever comes first in the DataFrame), and
    those two rows are not even the same KIND of number — one is a seismic-wedge
    lowstand compiled from five studies, the other is the modern Bosphorus sill
    depth used as an overflow proxy.  Blending or picking between them is inventing
    a level, so this raises instead.

    It is a refusal, not a failure: an epoch the curve does not cover has no level,
    and the honest frame for it is a refused frame (an animation keeps its slot and
    hatches it) rather than a plausible-looking neighbouring number.
    """


def _curve_row_index(
    df: pd.DataFrame, epoch: float, *, age_tol_ka: float = _CURVE_AGE_TOL_KA
) -> Any:
    """Index of THE curve row for *epoch*, or raise :class:`CurveAgeGapError`.

    One lookup for the level, the envelope and the marker, so those three can never
    be read off different rows of the same curve.

    Raises
    ------
    CurveAgeGapError
        The nearest row is further than *age_tol_ka* away, or two rows are equally
        near and disagree on ANY of the three (a duplicated age).  Both messages name
        the candidate rows and the spread between their values, because that spread
        is the size of the error the old lookup was making silently.
    """
    ages = pd.to_numeric(df["age_ka"], errors="coerce").astype(float)
    want = float(epoch)
    gaps = (ages - want).abs()
    if not bool(gaps.notna().any()):
        raise CurveAgeGapError(
            f"the curve has no usable age_ka values at all, so {want:g} ka cannot "
            "be located in it"
        )
    best = float(gaps.min())
    near = list(gaps.index[gaps <= float(age_tol_ka)])
    if not near:
        idx_near = gaps.idxmin()
        raise CurveAgeGapError(
            f"the curve has no row at {want:g} ka: the nearest is "
            f"{float(ages.loc[idx_near]):g} ka, {best:.3f} ka away, over the "
            f"{float(age_tol_ka):g} ka tolerance.  A neighbouring row's level is NOT "
            "substituted — on this curve the two rows bracketing a quarter-ka epoch "
            "can differ by tens of metres and carry different provenance, so "
            "'nearest' would be a number decided by DataFrame row order.  Ask for "
            "an epoch the curve carries, or widen age_tol_ka at the call site and "
            "write down why."
        )
    if len(near) > 1:
        # EVERYTHING THIS INDEX IS USED TO READ, or the scan certifies nothing.
        # Levels and envelope edges both drive a verdict, so a duplicate that agrees
        # on the central line and differs on its envelope is still ambiguous — and
        # so is one that agrees on both and differs on WHICH MARKER APPLIES, which
        # until now passed and left the choice to DataFrame row order.  A marker
        # decides ``conversion_m``, i.e. the whole F2 -> F1 term; substituting the
        # wrong one is the systematically one-signed error (up to 5.8 m, contract
        # §5a-bis) that :class:`UnconvertibleThresholdError` exists to refuse.
        # Marker columns are compared as STRINGS: two of the three are coordinates
        # and one is a source token, and a coordinate that differs in the last
        # decimal is a different marker.
        numeric = [
            col
            for col in df.columns
            if str(col).startswith("level_m_global_") or col in ENVELOPE_COLUMNS
        ]
        values: dict[str, list] = {
            col: sorted(
                {float(v) for v in pd.to_numeric(df.loc[near, col], errors="coerce")}
            )
            for col in numeric
        }
        for col in _MARKER_COLUMNS:
            if col in df.columns:
                values[col] = sorted({str(v) for v in df.loc[near, col]})
        disagree = {c: v for c, v in values.items() if len(v) > 1}
        if disagree:
            raise CurveAgeGapError(
                f"{len(near)} curve rows sit within {float(age_tol_ka):g} ka of "
                f"{want:g} ka and they disagree: {disagree}.  Which one applies "
                "would be decided by DataFrame row order, which is not a "
                "measurement — de-duplicate the curve."
            )
    return gaps.idxmin()


def basin_level(
    df: pd.DataFrame,
    epoch: float,
    scenario: str = "gradual",
    basin: str = "pontic",
    *,
    age_tol_ka: float = _CURVE_AGE_TOL_KA,
) -> float:
    """Closed-basin level (m vs present global SL) at *epoch*, from ITS OWN row.

    *scenario* selects the reconnection column (``gradual`` or ``catastrophic``).
    *basin* selects the datum-sanity window applied to the returned scalar.  That
    window is a blow-up guard only — the sign of the curve is validated once, at
    load time, by :func:`validate_basin_curve`.

    The lookup is still nearest-age, but it is now BOUNDED by *age_tol_ka*
    (:data:`_CURVE_AGE_TOL_KA`, 0.01 ka by default) and raises
    :class:`CurveAgeGapError` past it.  It used to be unbounded, which meant an
    off-grid epoch quietly received a neighbouring row's number; see that class for
    the 28.3 m this costs at 16.75 ka.  Every epoch either bake actually bakes lands
    on an exact row, so no shipped value changes.
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    if basin not in _DATUM_SANITY_RANGE_M:
        raise ValueError(
            f"unknown basin {basin!r}; known basins={sorted(_DATUM_SANITY_RANGE_M)}"
        )
    idx = _curve_row_index(df, epoch, age_tol_ka=age_tol_ka)
    level = float(df.loc[idx, col])
    return _check_datum_sanity(level, basin, f"{float(epoch):g} ka")


def pontic_level(
    df: pd.DataFrame,
    epoch: float,
    scenario: str = "gradual",
    *,
    age_tol_ka: float = _CURVE_AGE_TOL_KA,
) -> float:
    """Nearest-age **Pontic** basin level (m vs present global SL) for *epoch*.

    Thin wrapper over :func:`basin_level` pinned to the ``pontic`` range, kept for
    the Black-Sea callers.  Same values and same row selection as before for every
    epoch the curve carries; an epoch it does not carry now raises instead of
    borrowing a neighbour (:class:`CurveAgeGapError`).
    """
    return basin_level(df, epoch, scenario, basin="pontic", age_tol_ka=age_tol_ka)


# ---------------------------------------------------------------------------
# The curve's per-row level ENVELOPE — read, never widened, never invented
# ---------------------------------------------------------------------------

#: The two columns both shipped curves carry beside their level columns.  They are
#: ABSOLUTE levels in the same frame and the same units as the central column, not a
#: half-width: ``unc_low <= level <= unc_high`` holds on all 71 Caspian rows for all
#: six level columns (pinned by ``test_unc_band_is_absolute_not_a_width``).
#:
#: THEY ARE CARRIED ONCE PER **ROW**, NOT PER SCENARIO.  There is one pair in the
#: file and every scenario column is read against the same two numbers, so a result
#: that stamped a scenario on the pair would be asserting an uncertainty the curve
#: does not carry.  What the pair IS at a given row is a measurement — see
#: :func:`envelope_scope`.
ENVELOPE_COLUMNS: tuple[str, str] = ("unc_low", "unc_high")

#: What produced an envelope, reported per row exactly like :data:`THRESHOLD_RULES`.
#:   ``envelope_as_carried_in_the_curve`` — the two columns, converted with the SAME
#:                                          conversion term and rule as the central.
#:   ``not_applicable_F4``                — the row's threshold is ``-offset(t)``,
#:                                          not its level, so converting the LEVEL's
#:                                          envelope would put a bar on a quantity
#:                                          nothing uses.
#:   ``refused``                          — :class:`MissingEnvelopeError`; see
#:                                          ``envelope_status``.
ENVELOPE_RULES: tuple[str, ...] = (
    "envelope_as_carried_in_the_curve",
    "not_applicable_F4",
    "refused",
)

#: What is and is not MACHINE-READABLE about where an edge came from.
#:
#: The previous literal here read ``"UNKNOWN:not_recorded_in_the_curve"``, and that
#: overstated the gap on the very rows the headline number rests on.  Neither curve
#: has an ``unc_source`` or ``unc_interp_flag`` COLUMN — that part is true, and it is
#: what this constant now says.  But the ``note`` column DOES record the edges'
#: sources in prose on the rows where it matters.  Measured on the Caspian curve, 5
#: of 71 rows name ``unc_low`` / ``unc_high`` outright (20.0, 18.0, 17.4, 15.0,
#: 14.0 ka) and 7 discuss the pair; the 15.0 ka note records that HESS2024 carries
#: the reliably-dated level as a band "+22 to +35 m a.s.l. reliably dated to
#: 18(17)-13 kyr BP", that +35 is its TOP EDGE, and that ``unc_low`` carries the
#: +22 m bottom edge from Makshaev & Tkach; the 14.0 ka note records a carried
#: conflict (``unc_low`` = +11 from Kurbanov et al. 2023 via HESS2024).
#:
#: Asserting "not recorded" where the curve records it is a FALSE PROVENANCE CLAIM,
#: the same class of error as substituting a number.  So the claim made here is the
#: one that survives measurement: the provenance is not machine-readable, and
#: whether this particular row's note records it is stated per row by
#: :func:`envelope_note_provenance`.
#:
#: Adding real ``unc_source`` / ``unc_interp_flag`` columns to the curve builder is
#: still the fix, and it is still out of scope (that repo is read-only here).
ENVELOPE_PROVENANCE_NO_COLUMN = (
    "NOT_MACHINE_READABLE:the curve has no unc_source/unc_interp_flag column"
)


class MissingEnvelopeError(ValueError):
    """This row's level envelope cannot be used, and will not be reconstructed.

    Raised for all six ways the ``unc_low`` / ``unc_high`` pair can fail to be a
    usable envelope, because every repair is a substitution that is
    indistinguishable downstream from a measured one:

    * a curve without the :data:`ENVELOPE_COLUMNS` at all;
    * a NaN / non-numeric edge;
    * an INVERTED pair (``unc_low > unc_high``);
    * a DEGENERATE pair (``unc_low == unc_high``) — zero width.  That is not an
      envelope, it is a claim of certainty wearing an envelope's clothes: it makes
      the low, central and high verdicts identical by construction and every
      robustness flag built on it unconditionally ``True``, so a reader sees three
      agreeing verdicts where the file supplied one number.  Neither shipped curve
      has such a row (measured), which is exactly why refusing costs nothing today
      and catches a future edit;
    * a pair that does not CONTAIN the central it was checked against — which really
      happens: the Pontic catastrophic column reads -100.0 m at 9.0 / 8.5 / 8.0 /
      7.5 ka against pairs of [-40, -30] ... [-27, -17], i.e. the pair was written
      for the GRADUAL column and the catastrophic level sits 60-83 m below it.  Two
      of those four (9.0 and 7.5 ka) are baked epochs;
    * a converted edge outside :data:`_THRESHOLD_SANITY_RANGE_M` (a datum/unit
      blow-up on the edge rather than on the central).

    The tempting repairs are all forbidden and each has a measured cost.
    Substituting the central collapses the triple back to the boolean this phase
    exists to replace.  Borrowing the neighbouring row's pair is not a small error:
    the Caspian envelope's WIDTH varies from 28 m to 2 m across the curve, and the
    Pontic's from 75 m to 5 m.  Synthesizing a low-scenario COLUMN in the curve
    instead is refused twice over — ``max(unc_low)`` = +22.0 m is below
    :data:`MANYCH_SILL_M` = +23.0 m, so :func:`validate_basin_curve`'s Check A
    rejects the file outright, and the vertical-datum contract's §8 requires the
    frame conversion to happen at the THRESHOLD, not in the curve.
    """


def level_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Every ``level_m_global_*`` column of *df*, in file order.

    The set the envelope's scope is measured AGAINST, read from the file rather than
    listed here — a curve that grows a seventh scenario must widen the measurement,
    not silently fall outside it.
    """
    return tuple(str(c) for c in df.columns if str(c).startswith("level_m_global_"))


def envelope_scope(
    df: pd.DataFrame, idx: Any, low_m: float, high_m: float
) -> tuple[tuple[str, ...], str]:
    """Which level columns ``[low_m, high_m]`` actually brackets at row *idx*.

    Returns ``(bracketed_columns, scope_sentence)``.  BOTH are measurements: nothing
    here knows in advance that the Caspian pair is an inter-school envelope over all
    six columns, or that the Pontic pair fails to contain ``catastrophic`` at four
    rows.  It reads the row and says what it finds.

    The sentence is what ships in the data.  It exists because a reader of
    ``khvalynian_sill_test.csv`` who sees ``scenario=maximal`` beside
    ``envelope_lo_f2_m=+17.1`` will otherwise read +17.1 as the maximal column's own
    lower bound, and it is not: it is the bottom of a pair that also brackets
    ``defensible``, ``late_rise``, ``early_rise`` and both aliases.
    """
    columns = level_columns(df)
    bracketed = tuple(
        col
        for col in columns
        if np.isfinite(v := float(pd.to_numeric(df.loc[idx, col], errors="coerce")))
        and low_m - 1e-9 <= v <= high_m + 1e-9
    )
    short = tuple(c[len("level_m_global_") :] for c in bracketed)
    if len(bracketed) == len(columns) and len(columns) > 1:
        sentence = (
            f"INTER-SCHOOL SCENARIO ENVELOPE: one (unc_low, unc_high) pair carried "
            f"ONCE PER CURVE ROW and read identically for every scenario; measured "
            f"here to bracket ALL {len(columns)} level columns of this curve "
            f"({'|'.join(short)}).  It is NOT the uncertainty of any single "
            f"scenario column, and an edge of it must not be read as that column's "
            f"own bound."
        )
    else:
        sentence = (
            f"ROW-LEVEL PAIR, NOT AN ENVELOPE OVER THIS CURVE'S SCENARIOS: the one "
            f"(unc_low, unc_high) pair this row carries brackets "
            f"{len(bracketed)} of {len(columns)} level columns "
            f"({'|'.join(short) if short else 'none'}); the rest fall outside it."
        )
    return bracketed, sentence


def envelope_note_provenance(df: pd.DataFrame, idx: Any) -> tuple[bool, str]:
    """Does THIS row's ``note`` record where its envelope edges came from?

    Returns ``(recorded_in_note, statement)``.  Measured by looking for the edge
    column names in the row's own free text, not assumed either way — the point of
    this function is that the old blanket ``"UNKNOWN:not_recorded_in_the_curve"``
    asserted an absence that the 15.0 and 14.0 ka notes contradict.

    It deliberately does NOT parse a source out of the prose.  Prose is not a
    machine-readable provenance and pretending otherwise would substitute a citation;
    what ships is "the note names this edge — go read it" versus "the note is silent
    about this edge", plus the standing fact that no ``unc_source`` column exists.
    """
    note = "" if "note" not in df.columns else str(df.loc[idx, "note"])
    named = tuple(c for c in ENVELOPE_COLUMNS if c in note)
    if named:
        return True, (
            f"{ENVELOPE_PROVENANCE_NO_COLUMN}; but THIS ROW's note names "
            f"{','.join(named)} in prose and states where the edge comes from — read "
            f"the note column, and do not report the provenance as absent"
        )
    return False, (
        f"{ENVELOPE_PROVENANCE_NO_COLUMN}; and this row's note does not name either "
        f"edge, so for THIS row the edge sources are genuinely unrecorded.  Other "
        f"rows of the same curve do record theirs in prose"
    )


@dataclass(frozen=True)
class LevelEnvelope:
    """One curve ROW's ``unc_low`` / ``unc_high`` pair, as ABSOLUTE levels.

    ``low_m`` and ``high_m`` are the curve's own two columns; nothing here is
    symmetrised, re-centred or widened.

    THERE IS NO ``scenario`` FIELD, and its absence is the fix.  The pair is a
    property of the ROW: the same two numbers serve every level column, so stamping
    a scenario on them asserted a per-scenario uncertainty the file does not carry.
    What is per-scenario is only the containment CHECK — hence
    :attr:`containment_checked_against`, which names the column whose central value
    was verified to lie inside the pair, and :attr:`central_m`, that column's value.

    :attr:`scope` and :attr:`bracketed_columns` say, per row and by measurement, how
    many of the curve's level columns the pair really covers.
    """

    epoch_ka: float
    #: The level column whose central was containment-checked.  NOT ownership.
    containment_checked_against: str
    basin: str
    central_m: float
    low_m: float
    high_m: float
    width_m: float
    rule: str
    provenance: str
    #: Measured: the level columns this pair brackets at this row.
    bracketed_columns: tuple[str, ...]
    #: Measured: every level column the curve carries.
    all_level_columns: tuple[str, ...]
    #: The sentence that ships in the data.  See :func:`envelope_scope`.
    scope: str
    #: Measured: does this row's ``note`` name an edge column?
    provenance_in_note: bool


def basin_level_envelope(
    df: pd.DataFrame,
    epoch: float,
    scenario: str = "gradual",
    basin: str = "pontic",
    *,
    age_tol_ka: float = _CURVE_AGE_TOL_KA,
) -> LevelEnvelope:
    """The ROW-level envelope containing ``basin_level(df, epoch, scenario, basin)``.

    Read straight out of :data:`ENVELOPE_COLUMNS` of the SAME row the level comes
    from (:func:`_curve_row_index`), so a level and its envelope can never be taken
    off different rows.

    *scenario* selects the column whose central value is containment-checked and
    reported as :attr:`LevelEnvelope.containment_checked_against`.  It does **not**
    select an envelope: there is one pair per row and it is the same pair for every
    scenario.

    Raises
    ------
    CurveAgeGapError
        The curve carries no row at this epoch.
    MissingEnvelopeError
        The pair is absent, NaN, inverted, degenerate, or does not contain the
        central it was checked against.  See that class for why none is repaired.
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    where = f"{float(epoch):g} ka / {scenario}"
    missing = [c for c in ENVELOPE_COLUMNS if c not in df.columns]
    if missing:
        raise MissingEnvelopeError(
            f"curve is missing {missing} at {where} — this curve carries no "
            "uncertainty envelope at all, and one is not manufactured from the "
            "central line or from a neighbouring row"
        )
    idx = _curve_row_index(df, epoch, age_tol_ka=age_tol_ka)
    central = float(df.loc[idx, col])
    lo_raw, hi_raw = (
        pd.to_numeric(df.loc[idx, ENVELOPE_COLUMNS[0]], errors="coerce"),
        pd.to_numeric(df.loc[idx, ENVELOPE_COLUMNS[1]], errors="coerce"),
    )
    lo, hi = float(lo_raw), float(hi_raw)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise MissingEnvelopeError(
            f"envelope edge is not a number at {where} "
            f"({ENVELOPE_COLUMNS[0]}={df.loc[idx, ENVELOPE_COLUMNS[0]]!r}, "
            f"{ENVELOPE_COLUMNS[1]}={df.loc[idx, ENVELOPE_COLUMNS[1]]!r}) — a "
            "missing edge is not the central value and is not the neighbouring "
            "row's edge"
        )
    if lo > hi:
        raise MissingEnvelopeError(
            f"envelope is inverted at {where}: {ENVELOPE_COLUMNS[0]}={lo:.1f} m is "
            f"above {ENVELOPE_COLUMNS[1]}={hi:.1f} m.  Swapping them would silently "
            "repair a data error whose direction is unknown"
        )
    if lo == hi:
        raise MissingEnvelopeError(
            f"envelope is degenerate at {where}: {ENVELOPE_COLUMNS[0]} == "
            f"{ENVELOPE_COLUMNS[1]} == {lo:.1f} m, width 0.0 m.  A zero-width "
            "envelope is not an uncertainty, it is a claim of CERTAINTY: it forces "
            "the low, central and high verdicts to agree by construction and makes "
            "every robustness flag built on it unconditionally True, so a reader "
            "would see three independent-looking verdicts produced by one number.  "
            "It is refused rather than reported with width_m=0.0"
        )
    if not lo <= central <= hi:
        raise MissingEnvelopeError(
            f"envelope does not contain the central it was checked against "
            f"({col} = {central:.1f}, envelope [{lo:.1f}, {hi:.1f}])"
        )
    bracketed, scope = envelope_scope(df, idx, lo, hi)
    in_note, provenance = envelope_note_provenance(df, idx)
    return LevelEnvelope(
        epoch_ka=float(epoch),
        containment_checked_against=col,
        basin=basin,
        central_m=central,
        low_m=lo,
        high_m=hi,
        width_m=hi - lo,
        rule=ENVELOPE_RULES[0],
        provenance=provenance,
        bracketed_columns=bracketed,
        all_level_columns=level_columns(df),
        scope=scope,
        provenance_in_note=in_note,
    )


def _convert_envelope(
    envelope: LevelEnvelope, conversion_m: float, where: str
) -> tuple[float, float]:
    """Put both envelope edges in the DEM's frame with the SAME term as the central.

    Contract §8 ("convert at the threshold, not in the curve") applies edge for edge:
    an edge converted with a different ``conversion_m`` — a per-edge Δz for the
    edge's own site, say — would make the envelope's WIDTH partly a GIA artefact.
    The §6 update measures ``Δz(sill) - Δz(marker)`` on the band median at
    +0.03..+0.18 m, i.e. below the 0.73 m of headroom the lower edge has at 16.0 ka
    but not negligible against it; that is recorded in
    :data:`ENVELOPE_PROVENANCE_NO_COLUMN`'s discussion and deliberately not modelled.

    Raises :class:`MissingEnvelopeError` if a converted edge leaves
    :data:`_THRESHOLD_SANITY_RANGE_M` — a blow-up on an edge is still a blow-up, and
    the envelope is then unusable rather than clipped.
    """
    lo_t = envelope.low_m + float(conversion_m)
    hi_t = envelope.high_m + float(conversion_m)
    sane_lo, sane_hi = _THRESHOLD_SANITY_RANGE_M
    for name, value in ((ENVELOPE_COLUMNS[0], lo_t), (ENVELOPE_COLUMNS[1], hi_t)):
        if not np.isfinite(value) or not sane_lo <= value <= sane_hi:
            raise MissingEnvelopeError(
                f"converted envelope edge {name}={value:+.2f} m at {where} is "
                f"outside the threshold sanity window "
                f"{sane_lo:+.1f}..{sane_hi:+.1f} m — datum or unit blow-up on the "
                "edge; the envelope is not clipped to fit"
            )
    return lo_t, hi_t


# ---------------------------------------------------------------------------
# F2 -> F1: the threshold conversion  T(t) = level(t) + Δz_solid(x_marker, t)
# ---------------------------------------------------------------------------

#: The three rules that may produce a closed-basin threshold, most exact first.
#: A row that matches NONE of them raises :class:`UnconvertibleThresholdError`;
#: there is deliberately no fourth, "use the nearest marker we happen to have"
#: rule, because that substitution is what the 2026-07-30 audit measured as
#: systematically one-signed (see :func:`pontic_threshold`).
#:
#:   ``marker``        the ROW's own located marker.  Exact.
#:   ``locus``         no marker, but a DOCUMENTED region the level must have come
#:                     from; median Δz over it, with its p5-p95 spread carried into
#:                     the frame's own error bar.
#:   ``ocean_offset``  the row is F4 (the basin has reconnected and IS the ocean),
#:                     so the exact companion of this DEM is ``-offset(t)``.
THRESHOLD_RULES: tuple[str, ...] = ("marker", "locus", "ocean_offset")

#: Datum-sanity window for a CONVERTED threshold, in metres.  Wider and different
#: from ``_DATUM_SANITY_RANGE_M['pontic']`` (-150..+1) on purpose: that window
#: bounds an F2 *level*, which never rises above present sea level, while a
#: converted threshold legitimately can — ``-offset(t)`` is +2.3 m at 5 ka
#: (ICE-6G's far-field mid-Holocene highstand) and the F2 rows gain +10..+19 m.
#: Blow-up / unit guard only.
_THRESHOLD_SANITY_RANGE_M: tuple[float, float] = (-200.0, 60.0)


class UnconvertibleThresholdError(ValueError):
    """This row's F2 level cannot be put into the DEM's frame, and will not be faked.

    Raised when the curve row has **no located marker** and **no documented
    locus** — i.e. neither :data:`THRESHOLD_RULES` ``"marker"`` nor ``"locus"``
    applies and the row is not F4.

    As of 2026-08-01 **no row of the shipped curve reaches it.**  It used to fire
    for the 10.0 / 9.5 ka pair and the catastrophic column at 9.0 / 7.5 ka, whose
    marker_source named Giosan's Danube-delta separation surface and Lericolais's
    -100 m dune/terrace.  Both papers were then acquired and read in full, and
    neither prints a coordinate — so ``marker_lon``/``marker_lat`` stay ``UNKNOWN``
    and those six frames convert by rule ``"locus"`` over the two boxes
    georeferenced from the papers' own printed graticules (see :data:`_LOCI`).
    The class stays, and stays reachable, because the next curve row with a new
    provenance must still hit it rather than borrow somebody else's marker.

    The tempting repair — reuse the Bosphorus cell — is exactly the substitution
    the 2026-07-30 marker audit rejected: that cell sits low in the basin's Δz
    field, so it is systematically ONE-SIGNED and OVER-FLOODS every row it is
    borrowed for (~5.5 m at 22/18 ka, 3.9 m at 10.9 ka).  Resolve it by giving the
    row a marker (``marker_lon``/``marker_lat`` in the curve CSV) or by citing a
    locus and adding it to :data:`_LOCI` / :data:`_MARKER_SOURCE_LOCI`.
    """


@dataclass(frozen=True)
class _Locus:
    """A documented region a levelless F2 row could have been measured in."""

    name: str
    lon_deg: tuple[float, float]
    lat_deg: tuple[float, float]
    modern_depth_m: tuple[float, float]
    why: str


#: Loci that the curve CSV's own header block documents, verbatim.  Depth bands are
#: read from ICE-6G ``Topo`` at 0 ka on the SAME 10-arcmin grid the Δz field lives
#: on, so the band and the sample never come from different rasters.  Bounds are
#: INCLUSIVE; that is the definition of record for these numbers.
_LOCI: dict[str, _Locus] = {
    "nw_outer_shelf_seismic_wedge": _Locus(
        name="nw_outer_shelf_seismic_wedge",
        lon_deg=(28.0, 33.0),
        lat_deg=(43.0, 46.0),
        modern_depth_m=(-130.0, -90.0),
        why=(
            "the NW outer shelf, where the LGM lowstand wedge / forced-regression "
            "tract the -118..-120 m rows are compiled from actually sits "
            "(Aksu2002b; Kaplin_Selivanov2004; Lericolais2009; Popescu2004; "
            "Ryan1997 - a five-study seismic compilation, no single site)"
        ),
    ),
    "basin_shelf_band": _Locus(
        name="basin_shelf_band",
        lon_deg=(27.0, 42.0),
        lat_deg=(40.0, 48.0),
        modern_depth_m=(-130.0, -20.0),
        why=(
            "the whole Black-Sea-proper shelf band, i.e. the depths across which "
            "the shoreline actually moves; the honest locus for a row that is "
            "interpolated on a DERIVED falling limb and names no site at all"
        ),
    ),
    "nw_outer_shelf_lericolais_dune_terrace": _Locus(
        name="nw_outer_shelf_lericolais_dune_terrace",
        lon_deg=(29.5, 31.0),
        lat_deg=(43.5, 44.5),
        modern_depth_m=(-115.0, -90.0),
        why=(
            "the Romanian outer shelf at the head of the Danube (Viteaz) canyon, "
            "where Lericolais; Bulois; Gillet & Guichard 2009 (Global Planet. "
            "Change 66:65-75) map the -100 m dune field and the wave-cut terrace "
            "that the -100 m catastrophic-column rows ARE.  THE PAPER PRINTS NO "
            "COORDINATE - established by reading it in full, not assumed: it says "
            "only that 'A Differential GPS system was deployed for accurate "
            "(about 1 m) positioning', and no lon/lat appears in text, table or "
            "caption.  This box is georeferenced from the printed graticule of its "
            "Fig. 2.  IT IS DELIBERATELY WIDER THAN THE PUBLISHED MOSAIC: the "
            "pseudo-3D mosaic is 20x5 km at 30d00'-30d15' E / 43d53'-43d54' N, "
            "SMALLER THAN ONE 10-arcmin ICE-6G cell, and holds exactly 1 usable "
            "cell - below locus_solid_displacement's minimum of 5, so a median and "
            "a p5-p95 bar over the mosaic alone would be a fiction.  The widening "
            "follows the terrace segment the paper itself describes ('-98 m to "
            "-112 m around the head of the Danube (Viteaz) canyon', '-97 m north "
            "of it'), and it does NOT move the answer: the widened median tracks "
            "the single cell under the mosaic (30.1667 E / 43.9167 N) to "
            "-0.114 / -0.110 / -0.103 / -0.080 m at 10.0 / 9.5 / 9.0 / 7.5 ka, "
            "n=7 cells at every epoch"
        ),
    ),
    "danube_delta_separation_surface": _Locus(
        name="danube_delta_separation_surface",
        lon_deg=(28.8, 30.0),
        lat_deg=(44.6, 45.4),
        modern_depth_m=(-45.0, 5.0),
        why=(
            "the Danube delta plain and its adjacent inner shelf - the extent of "
            "the borehole network and of the 25-35 mbsl lacustrine/marine "
            "separation surface contoured in Giosan; Filip & Constantinescu 2009 "
            "(Quat. Sci. Rev. 28:1-6) Fig. 2b, which is what the -30 m gradual "
            "rows are.  THE PAPER PRINTS NO COORDINATE - established by reading it "
            "in full: its Table I is radiocarbon ages, and the new drill core is a "
            "filled dot on Figs. 1-2.  This box is georeferenced by linear "
            "interpolation off that figure's printed 29d/30d and 45d/44.5d "
            "graticule ticks, so its precision is about +/-0.05 deg (~5 km) and no "
            "better; an area georeferenced this way is a LOCUS and is never "
            "written into marker_lon/marker_lat.  CHECK: the same interpolation "
            "puts the CV/958 drill core at ~29.75 E / 45.0 N - the St. George / "
            "Sulina coast the text independently places it on.  The depth band "
            "spans the shoreline (-45 m to +5 m) because the separation surface is "
            "buried under the subaerial delta as well as under the inner shelf"
        ),
    ),
}

#: ``marker_source`` prefix -> locus.  Keyed on the prefix (not the epoch) so the
#: mapping is a statement about the KIND of provenance, and a re-run of
#: ``_build_bs_curve.py`` that renumbers rows cannot silently repoint it.
#: Anything not listed here has no documented locus and RAISES.
#:
#: The last four entries close the 10.5-7.5 ka window (2026-08-01).  Note that the
#: window is **three** provenances, not one: the papers disagree about the
#: CONTROVERSY (-30 m Giosan vs -100 m Lericolais) but two of its rows carry
#: NEITHER number, so charging them to either paper would be a fabrication —
#:
#:   * ``bracket_midpoint_held_flat`` (10.5 / 10.0 ka, BOTH columns) reads -80 m,
#:     which is neither -30 nor -100: it is the 10.9 ka central value clamped flat
#:     by ``interp_dict``, and that -80 is itself the midpoint of BLVK9814's
#:     measured -55 mbsl ceiling and the -100 m lowstand, i.e. from no source.  A
#:     midpoint of two markers ~45 m apart in depth has no marker →
#:     ``basin_shelf_band``.  This is WHY both 10.0 ka columns get the SAME
#:     threshold; splitting them onto Giosan vs Lericolais would manufacture a
#:     ~3.9 m difference out of an attribution neither column's -80 m supports.
#:   * ``gradual_tail_ramp`` (9.5 ka gradual) is -40 m, a hand-set ``grad_tail``
#:     value in ``_build_bs_curve.py`` ramping to the 9.4 ka reconnection.  No
#:     source measures it → ``basin_shelf_band``.
#:   * the two paper loci below are the rows that DO carry a cited number.
_MARKER_SOURCE_LOCI: tuple[tuple[str, str], ...] = (
    ("UNKNOWN:seismic_lowstand_wedge", "nw_outer_shelf_seismic_wedge"),
    ("UNKNOWN:interpolated_on_the_DERIVED_falling_limb", "basin_shelf_band"),
    ("UNKNOWN:DERIVED_anchor_source_gives_no_absolute_level", "basin_shelf_band"),
    ("UNKNOWN:lericolais2009_nw_outer_shelf", "nw_outer_shelf_lericolais_dune_terrace"),
    ("UNKNOWN:giosan2009_danube_delta", "danube_delta_separation_surface"),
    ("UNKNOWN:bracket_midpoint_held_flat", "basin_shelf_band"),
    ("UNKNOWN:gradual_tail_ramp", "basin_shelf_band"),
)

#: Curve columns the conversion needs.  Added to the CSV 2026-07-30; a curve
#: without them cannot be converted at all and must say so rather than guess.
_MARKER_COLUMNS = ("marker_lon", "marker_lat", "marker_source")

#: Literal strings the curve uses where a marker does not exist.  ``NOT_APPLICABLE``
#: is spelled out because pandas coerces ``"NA"`` to NaN.
_MARKER_UNKNOWN = "UNKNOWN"
_MARKER_NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class BasinThreshold:
    """One frame's land/sea threshold, with the rule that produced it.

    ``threshold_m`` is what goes into ``sea_level_override_m``.  Everything else
    exists so that a caller — a log line, a raster tag, a reader — can see WHICH of
    :data:`THRESHOLD_RULES` applied and how wide the resulting bar is.
    """

    epoch_ka: float
    scenario: str
    basin: str
    level_m: float
    frame: str
    rule: str
    threshold_m: float
    conversion_m: float | None
    bracket_m: tuple[float, float] | None
    n_cells: int | None
    marker_source: str
    provenance: str
    # ── the ROW's level envelope, put in the same frame by the same term ──────
    # ``None`` is a statement, and which statement it is lives in
    # ``envelope_status``: an F4 row has no meaningful envelope on its threshold, a
    # refused row has an unusable one.  The two must not be conflated, which is why
    # there is a status string beside the four numbers rather than a lone ``None``.
    #
    # These are NOT per-scenario bounds even though this object carries a
    # ``scenario``.  ``envelope_scope`` says so in words, on every row that has one.
    envelope_lo_m: float | None = None
    envelope_hi_m: float | None = None
    threshold_envelope_lo_m: float | None = None
    threshold_envelope_hi_m: float | None = None
    envelope_status: str = "UNKNOWN:the envelope was not evaluated by this caller"
    envelope_rule: str = "UNKNOWN:the envelope was not evaluated by this caller"
    envelope_provenance: str = ENVELOPE_PROVENANCE_NO_COLUMN
    envelope_scope: str = "UNKNOWN:the envelope was not evaluated by this caller"
    # ── was this epoch's GIA field blended between two shipped ICE-6G epochs? ──
    # An untagged interpolation is a substituted number, so it is a field and not a
    # footnote.  ``False`` for every epoch either bake bakes today.
    ice6g_interpolated: bool = False
    ice6g_time_note: str = ""


def _window_solid_displacement_at_shipped_epoch(
    epoch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Δz, modern Topo, lats, lons) at an epoch ICE-6G actually ships."""
    win = _open_bs_window(epoch)
    dz = solid_displacement(
        win.topo_t,
        win.topo_0,
        win.offset,
        sftgif_t=win.sftgif_t,
        sftgif_0=win.sftgif_0,
    )
    return dz, win.topo_0, win.lats, win.lons


def _window_solid_displacement(epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Δz, modern Topo, lats, lons) over the BS window, RAW 10-arcmin cells.

    Deliberately **unsmoothed**, unlike the field that reaches the raster
    (``GIA_SIGMA_BLUR`` = 2.0 cells).  A marker is a point measurement: blurring a
    single-cell probe over ~5 neighbouring cells would mix in ground the cited
    marker does not describe.  Raw cells are also the protocol the 2026-07-30
    contract and the curve CSV's header table were measured under, so the numbers
    here are directly comparable with both (16.5 ka: offset 77.906 m,
    Δz@Bosphorus 11.979 m).  The residual inconsistency — a σ=2 raster compared
    against an unsmoothed threshold — is sub-metre over this window and is stated
    rather than hidden.

    OFF-GRID EPOCHS (added 2026-08-03).  ICE-6G ships 0.5 ka steps to 21 ka, so
    14.7 ka and 7.15 ka — rows the Pontic curve genuinely carries — and every
    quarter-ka epoch of the Khvalynian window used to reach this function and get a
    raw ``FileNotFoundError`` out of :func:`_open_bs_window`.  They are now blended
    linearly in time between the two bracketing ICE-6G epochs, each endpoint
    computed exactly, by the same :func:`_ice6g_bracket` the Khvalynian bake uses.
    An epoch ICE-6G ships takes ``lo == hi`` and is bit-identical to before; an
    epoch it does not ship is TAGGED ``ice6g_interpolated=True`` on the
    :class:`BasinThreshold` it produces, because an untagged interpolation is a
    substituted number.
    """
    lo, hi, w = _ice6g_bracket(epoch)
    dz, topo_0, lats, lons = _window_solid_displacement_at_shipped_epoch(lo)
    if lo != hi:
        dz_hi, _t0, _la, _lo = _window_solid_displacement_at_shipped_epoch(hi)
        dz = (1.0 - w) * dz + w * dz_hi
    return dz, topo_0, lats, lons


def _cell_index(lats: np.ndarray, lons: np.ndarray, lon: float, lat: float) -> tuple[int, int]:
    """Nearest grid cell to (*lon*, *lat*), refusing a point outside the window."""
    if not (float(np.min(lats)) <= lat <= float(np.max(lats))):
        raise UnconvertibleThresholdError(
            f"marker latitude {lat:.4f} N is outside the ICE-6G window "
            f"{float(np.min(lats)):.4f}..{float(np.max(lats)):.4f} N — the Δz there "
            "was never read, and the nearest edge cell is a different place"
        )
    lon_wrapped = ((np.asarray(lons, dtype=float) - lon + 180.0) % 360.0) - 180.0
    if float(np.abs(lon_wrapped).min()) > 1.0:
        raise UnconvertibleThresholdError(
            f"marker longitude {lon:.4f} E is outside the ICE-6G window "
            f"{float(np.min(lons)):.4f}..{float(np.max(lons)):.4f} E"
        )
    return int(np.abs(np.asarray(lats, dtype=float) - lat).argmin()), int(
        np.abs(lon_wrapped).argmin()
    )


def marker_solid_displacement(
    epoch: float, lon: float, lat: float
) -> tuple[float, tuple[float, float]]:
    """``(Δz at the marker cell, 3x3 cell-block min/max)`` in metres.

    The block is the honest local spread of a point probe on a 10-arcmin grid —
    NOT a literature uncertainty on the marker's depth, which this module has no
    access to.
    """
    dz, _topo0, lats, lons = _window_solid_displacement(epoch)
    i, j = _cell_index(lats, lons, lon, lat)
    value = float(dz[i, j])
    if not np.isfinite(value):
        raise UnconvertibleThresholdError(
            f"Δz at the marker cell {lon:.4f}E/{lat:.4f}N is not finite at "
            f"{epoch:g} ka — ICE-6G 'Topo' is the ice surface there and these files "
            "carry no bedrock-under-ice field, so the conversion term is UNKNOWN"
        )
    block = dz[
        max(i - 1, 0) : i + 2,
        max(j - 1, 0) : j + 2,
    ]
    finite = block[np.isfinite(block)]
    return value, (float(finite.min()), float(finite.max()))


def locus_solid_displacement(
    epoch: float, locus_name: str
) -> tuple[float, tuple[float, float], int]:
    """``(median Δz, (p5, p95), n_cells)`` over a documented locus, in metres."""
    if locus_name not in _LOCI:
        raise ValueError(f"unknown locus {locus_name!r}; known={sorted(_LOCI)}")
    locus = _LOCI[locus_name]
    dz, topo_0, lats, lons = _window_solid_displacement(epoch)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0
    lo_d, hi_d = locus.modern_depth_m
    mask = (
        (lon_wrapped >= locus.lon_deg[0])
        & (lon_wrapped <= locus.lon_deg[1])
        & (lat2d >= locus.lat_deg[0])
        & (lat2d <= locus.lat_deg[1])
        & (topo_0 >= lo_d)
        & (topo_0 <= hi_d)
        & np.isfinite(dz)
    )
    values = dz[mask]
    if values.size < 5:
        raise UnconvertibleThresholdError(
            f"locus {locus_name!r} contains only {values.size} usable ICE-6G cells "
            f"at {epoch:g} ka — too few for a median and a p5-p95 bar, so there is "
            "no defensible conversion term for this row"
        )
    return (
        float(np.median(values)),
        (float(np.percentile(values, 5)), float(np.percentile(values, 95))),
        int(values.size),
    )


def _marker_source_for_scenario(raw: str, scenario: str) -> str:
    """Resolve a possibly per-column ``marker_source`` down to one token.

    ``MIXED:gradual=...;catastrophic=...`` exists because at 9.0 / 7.5 ka the
    gradual column has already reconnected (F4) while the catastrophic column is
    still a deep F2 lake.  A per-ROW answer would be wrong for one of them.
    """
    token = str(raw).strip()
    if not token.startswith("MIXED:"):
        return token
    parts = {}
    for chunk in token[len("MIXED:") :].split(";"):
        key, sep, value = chunk.partition("=")
        if sep:
            parts[key.strip()] = value.strip()
    if scenario not in parts:
        raise UnconvertibleThresholdError(
            f"marker_source {raw!r} is MIXED but names no entry for scenario "
            f"{scenario!r} (has {sorted(parts)}) — the frame of this column is "
            "undetermined and must not be guessed"
        )
    return parts[scenario]


def _parse_marker_position(row, scenario: str) -> tuple[float, float] | None:
    """``(lon, lat)`` if this row carries a located marker, else ``None``.

    The curve stores the literal strings ``UNKNOWN`` / ``NOT_APPLICABLE`` where no
    marker exists; those are branched on, never coerced.  A row with only one of
    the two coordinates is a data error, not a half-known marker, and raises.
    """
    raw_lon, raw_lat = str(row["marker_lon"]).strip(), str(row["marker_lat"]).strip()
    parsed = []
    for raw in (raw_lon, raw_lat):
        if raw in (_MARKER_UNKNOWN, _MARKER_NOT_APPLICABLE) or raw.startswith(
            (_MARKER_UNKNOWN + ":", _MARKER_NOT_APPLICABLE + ":")
        ):
            parsed.append(None)
            continue
        try:
            parsed.append(float(raw))
        except ValueError:
            parsed.append(None)
    lon, lat = parsed
    if (lon is None) != (lat is None):
        raise UnconvertibleThresholdError(
            f"curve row carries marker_lon={raw_lon!r} and marker_lat={raw_lat!r} — "
            "half a coordinate pair is a data error, not a located marker"
        )
    return None if lon is None else (lon, lat)


def _locus_for_marker_source(token: str) -> str | None:
    for prefix, locus_name in _MARKER_SOURCE_LOCI:
        if token.startswith(prefix):
            return locus_name
    return None


def _check_threshold_sanity(value: float, where: str) -> float:
    lo, hi = _THRESHOLD_SANITY_RANGE_M
    if not np.isfinite(value):
        raise UnconvertibleThresholdError(
            f"converted threshold at {where} is not finite — the conversion term is "
            "unknown and must not be written as a number"
        )
    if not lo <= value <= hi:
        raise ValueError(
            f"converted threshold {value:+.2f} m at {where} is outside the "
            f"sanity window {lo:+.1f}..{hi:+.1f} m — datum or unit blow-up"
        )
    return float(value)


def pontic_threshold(
    df: pd.DataFrame, epoch: float, scenario: str = "gradual"
) -> BasinThreshold:
    """Land/sea threshold for one Pontic frame, in the paleo-DEM's own frame.

    This is the code the 2026-07-30 vertical-datum contract asks for and that had,
    until now, existed only as prose.  The bake used to pass the raw CSV level
    straight through, which is the contract's §8 row *"re-bake the DEM without
    §5a's threshold change -> hero frames over-corrected ~12 m"*.

    THE CONVERSION.  For an **F2** row (RSL as reported — the present-day measured
    depth of a marker that stood at the water surface at time t)::

        T(t) = level(t) + Δz_solid(x_marker, t)

    applied HERE, at the threshold, and never in the curve: the level column is the
    cited literature value, and rewriting it would silently break
    :func:`validate_basin_curve`'s F2 anchor/mirror semantics (contract §8).

    For an **F4** row (the basin has reconnected and *is* the ocean, so the level is
    a global mean, not a local water surface) the CSV value is not used as the
    threshold at all: the exact companion of this DEM is ``-offset(t)``
    (:func:`sea_surface_offset`, contract §2).

    WHICH RULE APPLIED, AND WHY IT MATTERS.  Only 2 of the curve's 57 rows carry a
    located marker; 35 are the literal string ``UNKNOWN``.  Three rules, in
    decreasing exactness — ``marker``, ``locus``, ``ocean_offset`` — and every
    result reports its own in ``rule`` and in ``provenance``.  A row that matches
    none of them raises :class:`UnconvertibleThresholdError` rather than borrowing
    somebody else's marker.  That refusal is the point: the audit measured the
    obvious fallback (the Bosphorus cell) as systematically ONE-SIGNED — it sits
    low in the basin's Δz field, so it OVER-FLOODS every row it is borrowed for, by
    ~5.5 m at 22/18 ka and ~3.9 m at 10.9 ka.  Re-introducing it silently would
    undo the entire phase this function exists to close.

    THE ENVELOPE IS CARRIED, NOT THRESHOLDED ON.  Both curve columns of
    :data:`ENVELOPE_COLUMNS` are converted with the SAME ``conversion_m`` and the
    same rule as the central line and reported as ``threshold_envelope_lo_m`` /
    ``threshold_envelope_hi_m`` (contract §8: convert at the threshold, not in the
    curve — edge for edge).  Nothing here thresholds on them, which is exactly why
    this is the ONE site in the phase that CATCHES :class:`MissingEnvelopeError`
    instead of propagating it: an unusable envelope is recorded in
    ``envelope_status`` and the frame is still produced.  ``caspian_threshold`` does
    not catch, because there the envelope drives a verdict.

    The envelope's SCOPE is measured per row and reported in ``envelope_scope``.  On
    this curve it is not an envelope over both scenarios at 9.0 / 8.5 / 8.0 / 7.5 ka
    — the pair was written for ``gradual`` and does not contain ``catastrophic`` —
    and at those rows the field says exactly that instead of implying a shared bound.

    Raises
    ------
    UnconvertibleThresholdError
        No marker and no documented locus.  **No row of the shipped curve is in
        that state as of 2026-08-01** — the last six that were (10.0 and 9.5 ka in
        both columns, and the catastrophic column at 9.0 and 7.5 ka) now resolve
        by ``locus``, because the two papers behind them were read and their
        figures' graticules yielded documented AREAS, not coordinates.
    CurveAgeGapError
        The curve carries no row at this epoch.  Off-grid epochs used to take a
        neighbouring row's level silently — 28.3 m at 16.75 ka — and now refuse.
    ValueError
        Unknown scenario, a curve without the marker columns, or a converted
        threshold outside :data:`_THRESHOLD_SANITY_RANGE_M`.
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    missing = [c for c in _MARKER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"curve is missing {missing} — without the marker columns an F2 level "
            "cannot be converted into the DEM's frame at all.  Regenerate the CSV "
            "with _build_bs_curve.py (they were added 2026-07-30); do NOT fall back "
            "on a basin-wide constant."
        )

    idx = _curve_row_index(df, epoch)
    row = df.loc[idx]
    level = pontic_level(df, epoch, scenario)
    where = f"{float(epoch):g} ka / {scenario}"
    token = _marker_source_for_scenario(row["marker_source"], scenario)
    interpolated = ice6g_is_interpolated(epoch)
    time_note = _ice6g_time_note(epoch)

    # --- F4: the basin IS the ocean here; use the DEM's exact companion --------
    if token.startswith(_MARKER_NOT_APPLICABLE):
        offset = sea_surface_offset(epoch)
        # ``+ 0.0`` normalises the -0.0 that ``-offset`` gives at 0 ka.
        threshold = _check_threshold_sanity(-offset + 0.0, where)
        return BasinThreshold(
            epoch_ka=float(epoch),
            scenario=scenario,
            basin="pontic",
            level_m=level,
            frame="F4",
            rule="ocean_offset",
            threshold_m=threshold,
            conversion_m=None,
            bracket_m=None,
            n_cells=None,
            marker_source=token,
            provenance=(
                f"rule=ocean_offset frame=F4 {where}: reconnected basin, "
                f"T = -offset(t) = {threshold:+.3f} m (exact, contract sec.2).  The CSV "
                f"level {level:+.2f} m is a GLOBAL MEAN and is NOT used as the "
                f"threshold.  envelope: {ENVELOPE_RULES[1]}.  {time_note}"
            ),
            envelope_lo_m=None,
            envelope_hi_m=None,
            threshold_envelope_lo_m=None,
            threshold_envelope_hi_m=None,
            envelope_status=(
                "NOT_APPLICABLE: this row's threshold is -offset(t), not its level, "
                "so an envelope converted from the level column would be a bar on a "
                "number nothing here uses"
            ),
            envelope_rule=ENVELOPE_RULES[1],
            envelope_provenance=ENVELOPE_PROVENANCE_NO_COLUMN,
            envelope_scope=(
                "NOT_APPLICABLE: F4 row — the threshold is -offset(t), so the level "
                "column's envelope brackets nothing this frame uses"
            ),
            ice6g_interpolated=interpolated,
            ice6g_time_note=time_note,
        )

    # --- the envelope, for the F2 rules below ----------------------------------
    # THE ONE DELIBERATE CATCH IN THIS PHASE.  ``basin_level_envelope`` raises and
    # ``caspian_threshold`` lets it raise, because there the envelope DRIVES a
    # verdict.  Here it does not: the Pontic threshold is the central line, the
    # envelope is carried beside it for a reader, and refusing the whole frame over
    # an unusable envelope would delete 39 correct frames plus the 13
    # deliberately-wrong comparison frames over a number none of them thresholds on.
    # So the refusal is RECORDED instead of raised — with the same message the
    # exception carries, so the two cannot drift — and the four numeric fields stay
    # ``None``.  This really fires: the catastrophic column at 9.0 / 8.5 / 8.0 /
    # 7.5 ka carries -100 m against a pair written for the gradual column, and 9.0
    # and 7.5 ka are baked epochs.
    envelope: LevelEnvelope | None
    try:
        envelope = basin_level_envelope(df, epoch, scenario, basin="pontic")
        envelope_status = "OK"
        envelope_rule = ENVELOPE_RULES[0]
        envelope_scope_text = envelope.scope
        envelope_provenance = envelope.provenance
    except MissingEnvelopeError as exc:
        envelope = None
        envelope_status = f"REFUSED: {exc}"
        envelope_rule = ENVELOPE_RULES[2]
        envelope_scope_text = f"REFUSED: no usable envelope at this row — {exc}"
        envelope_provenance = ENVELOPE_PROVENANCE_NO_COLUMN

    def _envelope_fields(conversion: float) -> dict[str, object]:
        """The four envelope numbers converted with *conversion*, or four ``None``."""
        if envelope is None:
            return dict(
                envelope_lo_m=None,
                envelope_hi_m=None,
                threshold_envelope_lo_m=None,
                threshold_envelope_hi_m=None,
                envelope_status=envelope_status,
                envelope_rule=envelope_rule,
                envelope_provenance=envelope_provenance,
                envelope_scope=envelope_scope_text,
            )
        try:
            lo_t, hi_t = _convert_envelope(envelope, conversion, where)
        except MissingEnvelopeError as exc:
            return dict(
                envelope_lo_m=None,
                envelope_hi_m=None,
                threshold_envelope_lo_m=None,
                threshold_envelope_hi_m=None,
                envelope_status=f"REFUSED: {exc}",
                envelope_rule=ENVELOPE_RULES[2],
                envelope_provenance=envelope_provenance,
                envelope_scope=f"REFUSED: the envelope did not convert — {exc}",
            )
        return dict(
            envelope_lo_m=envelope.low_m,
            envelope_hi_m=envelope.high_m,
            threshold_envelope_lo_m=lo_t,
            threshold_envelope_hi_m=hi_t,
            envelope_status=envelope_status,
            envelope_rule=envelope_rule,
            envelope_provenance=envelope_provenance,
            envelope_scope=envelope_scope_text,
        )

    def _envelope_note(fields: Mapping[str, object]) -> str:
        if fields["threshold_envelope_lo_m"] is None:
            return f"envelope: {fields['envelope_status']}"
        return (
            f"envelope [{fields['envelope_lo_m']:+.2f}.."
            f"{fields['envelope_hi_m']:+.2f} m] -> "
            f"[{fields['threshold_envelope_lo_m']:+.3f}.."
            f"{fields['threshold_envelope_hi_m']:+.3f} m] "
            f"(same conversion term as the central; {fields['envelope_scope']}; "
            f"{fields['envelope_provenance']})"
        )

    # --- F2 rule 1: the row's own located marker.  Exact. ---------------------
    position = _parse_marker_position(row, scenario)
    if position is not None:
        lon, lat = position
        dz, block = marker_solid_displacement(epoch, lon, lat)
        threshold = _check_threshold_sanity(level + dz, where)
        fields = _envelope_fields(float(dz))
        return BasinThreshold(
            epoch_ka=float(epoch),
            scenario=scenario,
            basin="pontic",
            level_m=level,
            frame="F2",
            rule="marker",
            threshold_m=threshold,
            conversion_m=float(dz),
            bracket_m=(level + block[0], level + block[1]),
            n_cells=None,
            marker_source=token,
            provenance=(
                f"rule=marker frame=F2 {where}: T = level({level:+.2f} m) + "
                f"dz({dz:+.3f} m) @ {lon:.4f}E/{lat:.4f}N = {threshold:+.3f} m "
                f"[3x3 cell spread {level + block[0]:+.3f}..{level + block[1]:+.3f} m]"
                f" ({token}); {_envelope_note(fields)}; {time_note}"
            ),
            ice6g_interpolated=interpolated,
            ice6g_time_note=time_note,
            **fields,
        )

    # --- F2 rule 2: no marker, but a DOCUMENTED locus + its p5-p95 bar --------
    locus_name = _locus_for_marker_source(token)
    if locus_name is not None:
        median, (p5, p95), n_cells = locus_solid_displacement(epoch, locus_name)
        threshold = _check_threshold_sanity(level + median, where)
        fields = _envelope_fields(float(median))
        return BasinThreshold(
            epoch_ka=float(epoch),
            scenario=scenario,
            basin="pontic",
            level_m=level,
            frame="F2",
            rule="locus",
            threshold_m=threshold,
            conversion_m=float(median),
            bracket_m=(level + p5, level + p95),
            n_cells=n_cells,
            marker_source=token,
            provenance=(
                f"rule=locus frame=F2 {where}: T = level({level:+.2f} m) + "
                f"dz({median:+.3f} m, median over locus {locus_name!r}, "
                f"{n_cells} cells) = {threshold:+.3f} m "
                f"[p5-p95 {level + p5:+.3f}..{level + p95:+.3f} m] - "
                f"{_LOCI[locus_name].why}; {_envelope_note(fields)}; {time_note}"
            ),
            ice6g_interpolated=interpolated,
            ice6g_time_note=time_note,
            **fields,
        )

    # --- F2 rule 3: neither.  Refuse. ----------------------------------------
    raise UnconvertibleThresholdError(
        f"{where}: the curve row has NO located marker (marker_source={token!r}) and "
        f"no documented locus in _MARKER_SOURCE_LOCI, so the F2 level "
        f"{level:+.2f} m cannot be put into the paleo-DEM's frame.  It is NOT "
        "converted with the Bosphorus cell or any other row's marker: that "
        "substitution is systematically one-signed (it sits low in the basin's Δz "
        "field and OVER-FLOODS, by ~5.5 m at 22/18 ka), which is the defect this "
        "conversion exists to remove.  Fix it by adding marker_lon/marker_lat to "
        "the curve row, or by citing a locus and registering it in _LOCI / "
        "_MARKER_SOURCE_LOCI."
    )


def _icemask_subset(ds_sub) -> np.ndarray | None:
    """``sftgif`` (percent) from an already-subset ICE-6G dataset, or ``None``.

    ``None`` means the file carries no icemask at all, which is *no ice
    information* — not *no ice*.  Callers must not conflate the two; see
    :func:`gia_delta_subset`.
    """
    if ICE6G_ICEMASK_VAR not in ds_sub.variables:
        return None
    return ds_sub[ICE6G_ICEMASK_VAR].values.astype(np.float64)


@dataclass(frozen=True)
class _BsWindow:
    """Raw ICE-6G fields for one epoch: global datum + the regional subset."""

    epoch_ka: float
    file_t: Path
    file_0: Path
    offset: float
    topo_t: np.ndarray
    topo_0: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    sftgif_t: np.ndarray | None
    sftgif_0: np.ndarray | None


def _open_bs_window(epoch: float) -> _BsWindow:
    """Read the epoch/0-ka ICE-6G pair once: GLOBAL datum, REGIONAL fields.

    Shared by :func:`gia_delta_subset` (which turns it into Δz for the bake) and by
    :func:`pontic_threshold` (which samples Δz at a marker or over a locus).  One
    reader means the threshold cannot silently drift onto a different datum,
    window or grid from the raster it is compared against.
    """
    f_t = ICE6G_DIR / _epoch_filename(epoch)
    f_0 = ICE6G_DIR / _epoch_filename(0)
    lat_lo, lat_hi = LAT_MIN - BUF, LAT_MAX + BUF
    lon_lo, lon_hi = LON_MIN - BUF, LON_MAX + BUF

    with xr.open_dataset(f_t) as ds_t, xr.open_dataset(f_0) as ds_0:
        # Sea-surface datum: GLOBAL field, deliberately not the subset below.
        offset = implied_sea_surface_offset(
            ds_t["Topo"].values.astype(np.float64),
            ds_0["Topo"].values.astype(np.float64),
        )
        sub_t = ds_t.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        sub_0 = ds_0.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        return _BsWindow(
            epoch_ka=float(epoch),
            file_t=f_t,
            file_0=f_0,
            offset=float(offset),
            topo_t=sub_t["Topo"].values.astype(np.float64),
            topo_0=sub_0["Topo"].values.astype(np.float64),
            lats=sub_t["lat"].values.astype(np.float64),
            lons=sub_t["lon"].values.astype(np.float64),
            sftgif_t=_icemask_subset(sub_t),
            sftgif_0=_icemask_subset(sub_0),
        )


def sea_surface_offset(epoch: float) -> float:
    """``offset(t)`` in metres — the sea-surface datum this DEM is built on.

    Median ``ΔTopo`` over far-field ice-free tropical continental land, from the
    **global** ICE-6G field (see :func:`gia_delta_subset` for why the global field
    and not a regional cut).  ``0.0`` at 0 ka by construction.

    Its negation ``-offset(t)`` is the **exact** land/sea companion of
    ``H_paleo`` for any cell ICE-6G treats as ocean — contract §2:

        ``submerged  <=>  H_paleo < -offset(t)``

    Exact, with no uniformity assumption: whatever local sea-surface residual is
    baked into ``Δz_engine`` cancels against the same scalar on the other side.

    Linear in time between shipped ICE-6G epochs (:func:`_ice6g_bracket`), so the
    F4 rows the Pontic curve carries off the ICE-6G grid — 14.7 and 7.15 ka — resolve
    instead of raising ``FileNotFoundError``.  Every epoch on the grid is read
    exactly, as before.
    """
    lo, hi, w = _ice6g_bracket(epoch)
    if lo == hi:
        return _open_bs_window(lo).offset
    return (1.0 - w) * _open_bs_window(lo).offset + w * _open_bs_window(hi).offset


def gia_delta_subset(
    epoch: float, *, require_ice_information: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dz_solid, model_lats, model_lons) for the BS region at *epoch*.

    ``dz_solid`` = present-datum solid-earth displacement (Topo-based), subset to
    the BS bbox +/- ``BUF``, **NaN wherever ICE-6G ``Topo`` is the ice surface
    rather than the solid surface**.

    Two different grids are used on purpose and it is not interchangeable:

    * the **GLOBAL** ``Topo`` field defines the sea-surface datum (median ΔTopo over
      far-field ice-free tropical continental land — a global far-field quantity
      that a regional cut cannot define and that
      :func:`~paleoeurope.gia.ice6g_topo.implied_sea_surface_offset` now refuses to
      improvise);
    * the **REGIONAL** subset, with its own ``sftgif`` for both endpoints, defines
      the local displacement.

    This is the only code path in the project that runs GIA on the real ICE-6G
    distribution, so the ice-surface guard has to be armed here or it is armed
    nowhere.  The Black-Sea window happens to be ice-free at every shipped epoch,
    which is exactly why the missing guard was invisible.

    Parameters
    ----------
    require_ice_information : bool
        When the ICE-6G file carries no ``sftgif`` at all, refuse (default) rather
        than fall back on ``solid_displacement``'s
        :class:`~paleoeurope.gia.ice6g_topo.IceSurfaceUnmaskedWarning`.  A warning
        printed on every epoch of every run is not a guard.  Pass ``False`` only
        for a field you have separately established to be ice-free; the returned
        array then holds ICE-SURFACE elevations over any glaciated cell.

    Raises
    ------
    ValueError
        If ``sftgif`` is absent from both endpoints and *require_ice_information*
        is true, or if the subset contains a non-finite ``Δz`` that the icemask does
        not explain (a stray ``_FillValue`` in ``Topo`` must not masquerade as an
        ice hole).
    """
    win = _open_bs_window(epoch)
    f_t, f_0 = win.file_t, win.file_0
    lat_lo, lat_hi = LAT_MIN - BUF, LAT_MAX + BUF
    lon_lo, lon_hi = LON_MIN - BUF, LON_MAX + BUF
    offset = win.offset
    topo_t, topo_0 = win.topo_t, win.topo_0
    model_lats, model_lons = win.lats, win.lons
    sftgif_t, sftgif_0 = win.sftgif_t, win.sftgif_0

    if sftgif_t is None and sftgif_0 is None and require_ice_information:
        raise ValueError(
            f"neither {f_t.name} nor {f_0.name} carries a '{ICE6G_ICEMASK_VAR}' "
            "field, so it is UNKNOWABLE from these files whether the "
            f"{lat_lo:g}..{lat_hi:g}N / {lon_lo:g}..{lon_hi:g}E window is glaciated. "
            "Where it is, ICE-6G 'Topo' is the ICE SURFACE (up to +3506 m), not "
            "bedrock, and there is no bedrock-under-ice field to fall back on.  "
            "Supply files with the icemask, or pass "
            "require_ice_information=False if you have separately established that "
            "this field is ice-free."
        )

    # dilate_cells=0 is correct here: the rim is applied by
    # smooth_preserving_ice_mask() in build_paleo_dem(), which is what smooths.
    dz = solid_displacement(
        topo_t,
        topo_0,
        offset,
        sftgif_t=sftgif_t,
        sftgif_0=sftgif_0,
    ).astype(np.float32)  # (M, N)

    # NaN must mean "ice surface, no bedrock in these files".  A _FillValue (1e+20)
    # or NaN in Topo would otherwise be indistinguishable from an ice hole, and a
    # data problem would be silently absorbed into the ice policy below.
    unknown = ~np.isfinite(dz)
    if unknown.any():
        ice = ice_surface_mask(sftgif_t, sftgif_0)
        unexplained = unknown if ice is None else (unknown & ~ice)
        if unexplained.any():
            raise ValueError(
                f"{int(unexplained.sum())} cells of the {epoch:g} ka Black-Sea "
                "window have a non-finite Δz that the icemask does not explain — "
                "most likely a _FillValue (1e+20) or NaN in 'Topo'.  That is a data "
                "problem, not an ice hole; it must not be absorbed into the "
                "gia_nan_policy path."
            )

    return dz, model_lats, model_lons


def _require_model_grid_brackets_tile(
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    tile_lats: np.ndarray,
    tile_lons: np.ndarray,
) -> None:
    """Refuse a tile that reaches outside the GIA model window.

    :func:`~paleoeurope.gia.deformation.apply_gia_delta` interpolates with
    ``fill_value=0.0``, so any tile pixel outside the model grid would silently be
    baked with **0 m of GIA** — a zero-filled unknown wearing the same clothes as a
    measured zero.  The Black-Sea tile is comfortably inside the bbox +/- ``BUF``
    window (27.00-42.01 E / 39.99-48.00 N against 26-43 E / 39-49 N), so this never
    fires today; it exists so that repointing the bbox cannot reintroduce the
    silent fill.
    """
    for name, model, tile in (
        ("lat", model_lats, tile_lats),
        ("lon", model_lons, tile_lons),
    ):
        m_lo, m_hi = float(np.min(model)), float(np.max(model))
        t_lo, t_hi = float(np.min(tile)), float(np.max(tile))
        if t_lo < m_lo or t_hi > m_hi:
            raise ValueError(
                f"the DEM tile spans {name} {t_lo:.4f}..{t_hi:.4f} but the ICE-6G "
                f"model window only covers {m_lo:.4f}..{m_hi:.4f}: pixels outside "
                "it would be extrapolated by apply_gia_delta as 0 m of GIA, i.e. a "
                "zero-filled unknown.  Widen LAT_MIN/LAT_MAX/LON_MIN/LON_MAX/BUF "
                "to cover the tile."
            )


def build_paleo_dem(
    epoch: float,
    *,
    gia_nan_policy: str = BAKE_GIA_NAN_POLICY,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Bake the GIA-corrected paleo-DEM GeoTIFF for *epoch*; return its path.

    Parameters
    ----------
    gia_nan_policy : {'raise', 'nodata'}
        What to do when the GIA correction is unknown over part of the window
        because ICE-6G ``Topo`` is the ice surface there.  See
        :data:`BAKE_GIA_NAN_POLICIES`.  There is no ``'zero'``.
    provenance : mapping, optional
        Free-form GDAL metadata tags written into the raster header.  ``main()``
        uses it to record, in the raster itself, every threshold this epoch is
        rendered at, which of :data:`THRESHOLD_RULES` produced it and how wide its
        bar is — so a .tif that outlives this run still says what frame it belongs
        to.  Keys/values are stringified; nothing here is read back by the bake.

    Raises
    ------
    UnknownGiaInBakeError
        With ``gia_nan_policy='raise'`` (the default) when any cell of the window is
        glaciated at *epoch* or today.
    ValueError
        For an unknown *gia_nan_policy*, or if the DEM tile is not bracketed by the
        ICE-6G model window.
    """
    if gia_nan_policy not in BAKE_GIA_NAN_POLICIES:
        raise ValueError(
            f"gia_nan_policy={gia_nan_policy!r} is not one of "
            f"{BAKE_GIA_NAN_POLICIES}.  There is deliberately no 'zero' policy: "
            "0 m of correction is a legitimate MEASURED value over stable far-field "
            "ground, so writing it for an unknown cell makes the two "
            "indistinguishable."
        )

    with rasterio.open(BS_DEM) as ds:
        modern_dem = ds.read(1).astype(np.float32)
        transform = ds.transform
        crs = ds.crs
        width, height = ds.width, ds.height

    # Column centres (ascending) and row centres (DESCENDING north->south).
    cols = np.arange(width)
    rows = np.arange(height)
    tile_lons = (transform * (cols + 0.5, np.zeros_like(cols) + 0.5))[0]
    tile_lats = (transform * (np.zeros_like(rows) + 0.5, rows + 0.5))[1]

    dz, model_lats, model_lons = gia_delta_subset(epoch)
    _require_model_grid_brackets_tile(model_lats, model_lons, tile_lats, tile_lons)

    # ── Unknown Δz policy ──────────────────────────────────────────────────────
    # NaN here means "ICE-6G Topo is the ice surface, there is no solid surface in
    # these files", not "no correction".  It must never be added to the DEM as a
    # number, and it must never reach the raster as a bare NaN under a header that
    # declares nodata=BS_NODATA_M.
    gia_unknown = ~np.isfinite(dz)
    n_unknown = int(np.count_nonzero(gia_unknown))
    if n_unknown:
        detail = (
            f"epoch {epoch:g} ka: the GIA correction is unknown over {n_unknown} of "
            f"{gia_unknown.size} ICE-6G cells "
            f"({100.0 * n_unknown / gia_unknown.size:.1f}%) of the "
            f"{LAT_MIN - BUF:g}..{LAT_MAX + BUF:g}N / {LON_MIN - BUF:g}.."
            f"{LON_MAX + BUF:g}E window, because ICE-6G 'Topo' is the ICE SURFACE "
            "there and these files carry no bedrock-under-ice field."
        )
        if gia_nan_policy == "raise":
            raise UnknownGiaInBakeError(
                detail
                + "  Restrict this bbox/epoch to ice-free ground, take Δz from "
                "ICE-7G's bedrock-deformation field, or pass "
                "gia_nan_policy='nodata' to record the hole as nodata.  It will "
                "not be filled with 0 m."
            )
        logger.warning(
            "%s  gia_nan_policy='nodata': written as nodata=%r, NOT as 0 m of "
            "correction.",
            detail,
            BS_NODATA_M,
        )

    # ── Mask BEFORE smoothing ─────────────────────────────────────────────────
    # solid_displacement() already NaN-ed the ice cells (inside gia_delta_subset);
    # this is the ice-aware smooth, with the documented SMOOTHING_RIM_CELLS rim so
    # the margin cells resting on one-sided support are discarded rather than
    # published.  mode="reflect" is NOT this helper's default ("nearest"): it is
    # chosen to reproduce exactly the boundary handling apply_gia_delta applied
    # historically, so an ice-free window is bit-for-bit unchanged.  A plain
    # gaussian_filter here would smear the ~+2000 m ice-margin step outward (>1 m
    # out to 6-7 cells, measured) or propagate NaN over the whole kernel support.
    dz_smooth = smooth_preserving_ice_mask(
        dz.astype(np.float64),
        sigma=GIA_SIGMA_BLUR,
        dilate_cells=SMOOTHING_RIM_CELLS,
        mode="reflect",
    )
    ice_height = np.zeros_like(dz_smooth)

    paleo_dem, _ice = apply_gia_delta(
        modern_dem,
        model_lats,
        model_lons,
        dz_smooth,
        ice_height,
        tile_lats,
        tile_lons,
        # Already smoothed above, ice-mask-preserving.  apply_gia_delta's own
        # gaussian_filter is a plain one and would both smooth twice and propagate
        # NaN across the kernel support; 0.0 makes it an exact identity.
        sigma_blur=0.0,
    )

    # Unknown Δz interpolates to NaN on the tile grid.  Write the DECLARED nodata,
    # which is an honest "unknown", and never a number that reads as elevation.
    #
    # ONLY cells the GIA hole created: the shipped GMRT tile carries NaN of its own
    # (outside the Black-Sea-proper cut), and those NaNs have always been written
    # through as NaN.  Rewriting them to BS_NODATA_M would change what
    # render_paleocostline_epoch(force_dem_mask=True) draws — NaN never compares
    # below the sea level, -9999 always does — so the modern DEM's own holes are
    # left exactly as they were.  Same discrimination as
    # correction_pipeline's ``gia_unknown & ~nodata_mask``.
    if n_unknown:
        unknown_tile = ~np.isfinite(paleo_dem) & np.isfinite(modern_dem)
        if unknown_tile.any():
            paleo_dem = np.where(unknown_tile, np.float32(BS_NODATA_M), paleo_dem)

    PALEO_TIF_DIR.mkdir(parents=True, exist_ok=True)
    out_tif = PALEO_TIF_DIR / f"bs_{epoch}ka.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": BS_NODATA_M,
        "compress": "deflate",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(paleo_dem.astype(np.float32), 1)
        dst.update_tags(
            epoch_ka=f"{float(epoch):g}",
            gia_model="ICE-6G_C VM5a 10min",
            gia_datum="implied_sea_surface_offset (far-field ice-free tropical land)",
            gia_sea_surface_offset_m=f"{sea_surface_offset(epoch):.6f}",
            gia_nan_policy=gia_nan_policy,
            vertical_frame="F0 present-day datum (H_modern + dz_solid)",
            vertical_datum_contract=(
                "Paleo-Europe_3D/docs/superpowers/research/"
                "2026-07-30-vertical-datum-contract.md"
            ),
            **{str(k): str(v) for k, v in (provenance or {}).items()},
        )

    return out_tif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def preflight_thresholds(
    pontic_df: pd.DataFrame,
    epochs=None,
    scenarios=SCENARIOS,
) -> tuple[dict[tuple[float, str], BasinThreshold], dict[tuple[float, str], str]]:
    """Resolve every frame's threshold BEFORE a single raster is written.

    Returns ``(resolved, unconvertible)``.  Doing this up front is the same
    discipline as :func:`check_spratt_coverage`: a frame that cannot be put into
    the DEM's vertical frame must be discovered before the run half-fills an
    output tree, not on epoch 7 of 13.

    :class:`CurveAgeGapError` is collected here alongside
    :class:`UnconvertibleThresholdError` (2026-08-03) because both are the same kind
    of answer — *this frame has no defensible threshold* — and a caller sweeping a
    grid finer than the curve (an animation) needs the refusal as DATA, one row per
    frame, rather than as a traceback that stops the sweep at the first gap.  The
    stored string is prefixed with the class name so the two are never confused: a
    missing curve row and an unconvertible level want different fixes.  All 13
    shipped epochs land on exact rows, so this changes nothing about the bake:
    26/26 resolve, 0 unconvertible.
    """
    epochs = EPOCHS_KA if epochs is None else epochs
    resolved: dict[tuple[float, str], BasinThreshold] = {}
    unconvertible: dict[tuple[float, str], str] = {}
    for epoch in epochs:
        for scenario in scenarios:
            try:
                resolved[(epoch, scenario)] = pontic_threshold(
                    pontic_df, epoch, scenario
                )
            except (UnconvertibleThresholdError, CurveAgeGapError) as exc:
                unconvertible[(epoch, scenario)] = f"{type(exc).__name__}: {exc}"
    return resolved, unconvertible


def main(*, skip_unconvertible: bool = False) -> None:
    """Bake the 13 epochs.

    Parameters
    ----------
    skip_unconvertible : bool
        What to do about a frame whose F2 level cannot be converted into the
        paleo-DEM's vertical frame (no marker, no documented locus — see
        :class:`UnconvertibleThresholdError`).  ``False`` (default) aborts the whole
        run before anything is written.  ``True`` renders the frames that CAN be
        converted and simply does not produce the others — no substituted number,
        no placeholder raster.  The opt-out has to be written down at the call
        site, exactly like ``require_ice_information`` and
        ``require_spillway_highstand``: silence must not buy a weaker guarantee.
    """
    pontic_df = pd.read_csv(PONTIC_CSV, comment="#")
    # Validate the curve ONCE, before any epoch lookup: sign flips are only
    # detectable on the curve as a whole, never on an individual scalar.
    for scenario in SCENARIOS:
        validate_basin_curve(pontic_df, "pontic", scenario)

    # Same idea for the eustatic curve, and for the same reason: a truncated
    # curve does not fail, it CLAMPS.  Stop before the first frame instead.
    check_spratt_coverage(EPOCHS_KA)
    datum = spratt_present_datum_m()
    logger.info(
        "eustatic curve: %s (%.0f-%.0f ka), published datum %+.2f m at 0 ka; every "
        "frame thresholds on the 0-REFERENCED series (eustatic_level_m), so 0 ka "
        "renders at 0.00 m and the 22 ka frame at %+.2f m.  See the SPRATT_CSV "
        "comment for what that framing choice costs at the glacial end.",
        SPRATT_CSV.name,
        _spratt_ages_bp()[0] / 1000.0,
        _spratt_ages_bp()[-1] / 1000.0,
        datum,
        eustatic_level_m(22),
    )

    # ── Threshold pre-flight ──────────────────────────────────────────────────
    # T(t) = level(t) + Δz_solid(x_marker, t) for F2 rows, -offset(t) for F4 rows,
    # and a refusal for a row with neither marker nor locus.  Resolve them ALL now.
    thresholds, unconvertible = preflight_thresholds(pontic_df)
    for _key, threshold in sorted(thresholds.items(), key=lambda kv: -kv[0][0]):
        logger.info("threshold %s", threshold.provenance)
    for (epoch, scenario), why in sorted(unconvertible.items(), key=lambda kv: -kv[0][0]):
        logger.error("UNCONVERTIBLE %g ka / %s: %s", epoch, scenario, why)
    if unconvertible and not skip_unconvertible:
        raise UnconvertibleThresholdError(
            f"{len(unconvertible)} of {len(EPOCHS_KA) * len(SCENARIOS)} frames "
            "cannot be put into the paleo-DEM's vertical frame: "
            + ", ".join(f"{e:g} ka/{s}" for (e, s) in sorted(unconvertible))
            + ".  Nothing has been written.  Give those curve rows a marker, "
            "register a locus for them, or call main(skip_unconvertible=True) to "
            "render only the frames that ARE convertible — that option omits them "
            "entirely and never substitutes a number for one."
        )

    cfg = RenderConfig(max_px=2000, ice_model_label="ICE-6G_C")

    # Output dirs: one CORRECT dir per scenario + one WRONG (global) dir.
    out_wrong = OUT_BASE / "wrong"
    out_wrong.mkdir(parents=True, exist_ok=True)
    out_correct = {s: OUT_BASE / "correct" / s for s in SCENARIOS}
    for d in out_correct.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load the Paleocoastlines vector ONCE and reuse. force_dem_mask ignores the
    # vector, but passing it as preloaded_coast_gdf avoids re-loading the big
    # shapefile on every render call. Best-effort: the DEM-driven path does not
    # need it, so do not abort if it is absent.
    coast_gdf = load_paleocostlines(PALEOCOASTLINES_SHP)

    print()
    header = (
        f"{'epoch':>6} | {'eustatic':>9} | {'grad_T':>8} {'rule':>13} | "
        f"{'cat_T':>8} {'rule':>13}"
    )
    print(header)
    print("-" * len(header))

    for epoch in EPOCHS_KA:
        tags = {"eustatic_wrong_frame_m": f"{eustatic_level_m(epoch):.3f}"}
        for scenario in SCENARIOS:
            key = (epoch, scenario)
            if key in thresholds:
                tags[f"threshold_{scenario}"] = thresholds[key].provenance
            else:
                tags[f"threshold_{scenario}"] = (
                    "UNCONVERTIBLE (no marker, no documented locus) — "
                    "no frame was rendered for this scenario at this epoch"
                )
        paleo_tif = build_paleo_dem(epoch, provenance=tags)

        # CORRECT — one frame per scenario, DEM-driven at the CONVERTED threshold.
        cells = {}
        for scenario in SCENARIOS:
            threshold = thresholds.get((epoch, scenario))
            if threshold is None:
                cells[scenario] = f"{'--':>8} {'skipped':>13}"
                continue
            cells[scenario] = f"{threshold.threshold_m:>+8.2f} {threshold.rule:>13}"
            render_paleocostline_epoch(
                epoch,
                {"bs": str(paleo_tif)},
                sea_level_csv=SPRATT_CSV,
                paleocoastlines_shp=PALEOCOASTLINES_SHP,
                output_dir=out_correct[scenario],
                config=cfg,
                data_source_label=f"Pontic-{scenario}-{threshold.rule}",
                sea_level_override_m=threshold.threshold_m,
                force_dem_mask=True,
                preloaded_coast_gdf=coast_gdf,
            )

        # WRONG — the deliberately-wrong comparison frame.
        #
        # THIS FRAME STAYS EUSTATIC ON PURPOSE.  Two documents disagree about it and
        # the disagreement is resolved here so the next reader does not "fix" it:
        #
        #   * The re-bake verifier says these frames should switch to -offset(t),
        #     because §5b of `2026-07-30-vertical-datum-contract.md` measures them as
        #     10-37 m worse once the DEM is corrected.  That measurement is right.
        #   * The contract audit says the opposite FOR THIS SCRIPT: these frames are
        #     deliberately wrong.  They exist to show what a naive global-eustatic
        #     treatment does inside a decoupled basin, which is the entire scientific
        #     point of the render.  Switching them to -offset(t) would make them
        #     CORRECT and delete the comparison.
        #
        # Ruling (owner, 2026-07-31): keep the eustatic curve here — being wrong is
        # the job — but make it the GENUINE published Spratt & Lisiecki series
        # rather than the hand-drawn sketch that used to ship, 0-referenced to
        # present so that the 0/2 ka frames do not render sea level above the modern
        # coast.  Being wrong about the PHYSICS is the demonstration; being wrong
        # about the DATA was just a bug.
        #
        # The rule that decides this: a eustatic value survives only where a frame is
        # MEANT to be wrong.  Anywhere a frame's threshold is an INTENDED ocean
        # surface it must be -offset(t), which contract §2 proves exact — and that is
        # what `pontic_threshold`'s "ocean_offset" rule now does for every
        # post-reconnection (F4) row of the CORRECT frames.
        render_paleocostline_epoch(
            epoch,
            {"bs": str(paleo_tif)},
            sea_level_csv=SPRATT_CSV,
            paleocoastlines_shp=PALEOCOASTLINES_SHP,
            output_dir=out_wrong,
            config=cfg,
            data_source_label="global-eustatic (deliberately wrong)",
            sea_level_override_m=eustatic_level_m(epoch),
            force_dem_mask=True,
            preloaded_coast_gdf=coast_gdf,
        )

        print(
            f"{epoch:>6} | {eustatic_level_m(epoch):>+9.2f} | "
            f"{cells['gradual']} | {cells['catastrophic']}"
        )

    print()
    print(f"Correct (gradual):      {out_correct['gradual']}")
    print(f"Correct (catastrophic): {out_correct['catastrophic']}")
    print(f"Wrong (global):         {out_wrong}")
    if unconvertible:
        print()
        print(
            f"OMITTED {len(unconvertible)} frame(s) with no convertible threshold "
            "(no marker, no documented locus): "
            + ", ".join(f"{e:g} ka/{s}" for (e, s) in sorted(unconvertible))
        )


if __name__ == "__main__":
    main()
