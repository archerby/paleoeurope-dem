#!/usr/bin/env python
"""
bake_khvalynian.py
==================

Bake the **Khvalynian (Caspian) side** of the Manych spill scene: a GIA-corrected
paleo-DEM per epoch over **Caspian + Manych + Pontic in one raster**, plus a
first-class **sill test** that answers, epoch by epoch, the only question this
render exists to ask — *did the Caspian clear the Manych sill and spill west?*

WHAT THIS IS NOT
----------------
It is **not** a renderer.  ``bake_black_sea.py`` renders because its whole scene is
one decoupled basin, so a single scalar threshold is valid across the frame.  This
extent contains **two basins standing at different levels** (that is the point of
the hero shot), so a single ``sea_level_override_m`` over 25.5-58 E would flood the
Pontic to the Caspian's surface.  Compositing the two masks is a separate step and
is deliberately not done here — see :func:`main`'s docstring.  What ships from this
module is the shared paleo-DEM and the sill verdicts.

THE THRESHOLD IS NOT THE CURVE VALUE (vertical-datum contract, 2026-07-30)
-------------------------------------------------------------------------
``2026-07-28-caspian-khvalynian-level-curve.csv`` carries **F2** levels — the
present-day surveyed elevation of a dated deposit that stood at the water surface
at time *t* (contract §1/§6).  Same frame as the Pontic pre-reconnection rows, so
"two basins, one frame" is met; and, exactly like them, an F2 level cannot be laid
on a GIA-corrected ``H_paleo`` as-is.  The conversion is

    T(t) = level(t) + Δz_solid(x_marker, t)                       (F2 -> F1)

applied **at the threshold, never in the curve** (contract §8: rewriting the level
column breaks ``validate_basin_curve``'s F2 anchor/mirror semantics).

**The Caspian curve has no marker columns at all**, so of
:data:`~scripts.bake_black_sea.THRESHOLD_RULES` only ``locus`` can apply, and it is
the only rule this module implements.  There is no ``ocean_offset`` rule either:
the Caspian never reconnects to the world ocean — it is endorheic at every row of
the curve, present day included (-28 m) — so no row of it is ever F4.

WHY A BASIN-BAND LOCUS IS DEFENSIBLE HERE AND WAS NOT FOR THE PONTIC
--------------------------------------------------------------------
Substituting one cell's Δz for an unlocated Pontic marker is worth **up to 5.8 m**
and is systematically one-signed (contract §5a-bis), because the Pontic markers span
the Bosphorus, the NW shelf and the deep-basin seismic wedge, across which Δz varies
by ~7 m.  The Caspian markers and the Manych sill sit inside one shallow,
slowly-varying Δz field.

**Measured here, on this module's own locus (:data:`CASPIAN_SHORELINE_BAND`), not
inherited from any document** — ``Δz(sill) − Δz(locus median)`` on the ICE-6G
10-arcmin grid, no smoothing:

    18.0 ka  -0.292      16.0 ka  -0.270      14.0 ka  +0.177
    17.5 ka  -0.287      15.5 ka  -0.205      13.5 ka  +0.167
    17.0 ka  -0.229      15.0 ka  -0.171      13.0 ka  +0.155
    16.5 ka  -0.243      14.5 ka  -0.146      12.0 ka  +0.148

i.e. **-0.29 .. +0.18 m** over 18-12 ka.  Note this is NOT the "+0.03 to +0.18 m"
the contract's §6 update reports: that figure was measured on a different band, it
is one-signed, and mine is not.  Both are ≤ 0.3 m, so the *conclusion* — a band
locus is good enough for the Caspian — survives; the *bracket* is re-measured and
is the one carried in :class:`SillTest`.  :func:`sill_marker_differential` is the
function of record and the tests pin the numbers above.

THE SILL IS THE EVENT
---------------------
Spill iff the Caspian's water surface stands above the Manych sill's crest **in the
same frame**.  Both sides are put in F1 and compared there:

    T(t)            = level_F2(t) + Δz(locus, t)          <- the basin surface
    sill_F1(t)      = MANYCH_SILL_M + Δz(x_sill, t)       <- the crest
    margin(t)       = T(t) - sill_F1(t)

Contract §6 shows this is algebraically the F2 shortcut ``level_F2 > 23.00 +
[Δz(sill) − Δz(marker)]``, i.e. "compare against the MODERN sill", up to the
differential above.  Both are reported.  The frame-MIXED test the CSV's
``paleo_sill_m_asl`` column invites — an F2 level against an F1 sill — is also
reported, labelled wrong, and used for nothing: it discards ~2.7 m of spill margin.

``MANYCH_SILL_M`` = **+23.00 m** and its SIGN, established here rather than quoted:
:func:`manych_sill_elevation_m` reads the working COG at 44.119 E / 45.773 N and
gets **+23** (int16 GEBCO metres; the 5x5 block around it runs +22..+42, the saddle
being the low corner).  Positive — the crest stands *above* present sea level, which
is why an endorheic Caspian at -28 m does not spill today.  Pinned by
``test_manych_sill_elevation_is_read_from_the_raster_and_is_positive``.

The continuous-path controlling col of the same raster is **+25.00 m** (the curve's
own header; a spillway needs a connected channel, and +23.00 is the max of
per-column minima, i.e. a LOWER BOUND).  Both verdicts are carried; neither is
asserted.

NO DEEP LGM LOWSTAND
--------------------
The Caspian stood at **+5 .. +15 m a.s.l. through 26-19 ka** — 33-43 m *above*
modern, not below it.  Yanina's -100 m Atelian lowstand is MIS 4 / early MIS 3
(~70-40 ka) and is a mis-dating when placed at the LGM.  :func:`check_no_lgm_lowstand`
makes that executable: it raises if any 26-19 ka level leaves
:data:`LGM_PLATEAU_WINDOW_M`.  A lowstand that did not happen must not be animated.

EPOCH GRID
----------
:data:`EPOCHS_KA` is **0.25 ka across 17-13 ka** — the transgression, the spill and
its closure — and coarser outside.  The shipped 13-epoch Pontic grid puts the whole
event between two frames.  Every one of the 43 epochs lands on an EXACT row of the
Caspian curve (pinned by ``test_every_baked_epoch_has_an_exact_curve_row``), so the
nearest-age lookup in ``basin_level`` never interpolates a level.

ICE-6G ships 0.5 ka steps to 21 ka and 1 ka beyond, so the eight quarter-ka epochs
(16.75, 16.25, ... 13.25) have no file.  Their Δz and offset are **linear in time
between the two bracketing ICE-6G epochs**, each endpoint computed exactly; see
:func:`_ice6g_bracket`, and the ``gia_time_interpolation`` raster tag records it per
epoch.  Where that approximation is worst is stated rather than hidden: MWP-1A steps
``offset`` from 66.374 m at 14.5 ka to 45.997 m at 14.0 ka, so 14.25 ka — **the epoch
at which the spill window closes** — is a midpoint across a 20 m jump.  The curve's
own header flags 14.0 ka for the same reason.  Do not read a precise closing date.

NaN POLICY: THIS WINDOW *DOES* STRADDLE A PALAEO-ICE MARGIN
-----------------------------------------------------------
``bake_black_sea``'s docstring says the Caspian bbox is ice-free at every epoch.
That was true of the narrow Caspian bbox; it is **false for this scene**, which
reaches 54 N / 25.5 E and so clips the SE margin of the Fennoscandian ice sheet.
Measured on the shipped files over this window +/- BUF: **101 glaciated ICE-6G cells
at 18.0 and 17.5 ka, 33 at 22 ka, 71 at 26 ka, and zero from 17.0 ka onward**
(``ice_cells_in_window``; pinned by
``test_the_window_straddles_the_ice_margin_before_17_ka``).

So the default here is ``gia_nan_policy="nodata"``, not ``"raise"`` — the legitimate
case ``bake_black_sea`` documents for "a FUTURE bbox that genuinely straddles a
palaeo-ice margin".  Those cells are written as :data:`KHV_NODATA_M`, never as 0 m of
correction, and the ``SMOOTHING_RIM_CELLS`` rim widens the hole rather than resting a
published value on one-sided support.  There is still deliberately no ``"zero"``.

WHAT THIS BAKE ACTUALLY DECIDES (measured on the shipped curve, 2026-08-01)
---------------------------------------------------------------------------
The F1-vs-F1 test at the +23.00 m point sill, contiguous on the 0.25 ka grid:

    defensible   spilling_west  **16.5  -> 14.25 ka**   (10 frames)
    maximal      spilling_west  **16.75 -> 14.25 ka**   (11 frames)

Three things this grid buys that the shipped 13-epoch grid cannot see, none of them
cosmetic:

* **The closing frame exists.**  On a 0.5 ka grid the defensible window reads
  16.5 -> 14.5 ka; 14.25 ka — where it actually closes — is not sampled at all.
* **The +23.00 / +25.00 sill ambiguity costs exactly one frame.**  At the
  continuous-path crest the 14.25 ka defensible frame demotes to ``at_sill``
  (margin +2.17 m against +23.00, -0.24 m against +25.00).  Every other frame is
  unchanged.  That is the sill uncertainty as a picture rather than as a caveat.
* **The frame-MIXED reading loses the same frame**, independently: comparing the F2
  level against the F1 ``paleo_sill_m_asl`` closes the defensible window at 14.5 ka.
  Contract §6 predicted that; it is reproduced here from the raw fields.

One verdict on the whole grid is flagged **not robust** — 14.0 ka, margin -1.18 m,
inside the locus bracket in both scenarios.  That is the same epoch the curve's own
header flags as an ICE-6G MWP-1A artefact.  Two independent reasons not to quote a
closing date to better than a few hundred years; ``verdict_is_robust`` carries it.

AND WHAT IT DOES *NOT* DECIDE: THE ENVELOPE (added 2026-08-03, corrected 2026-08-03)
------------------------------------------------------------------------------------
Everything above is a property of the **adopted central curve**.  ``verdict_is_robust``
does not soften that: it moves the ICE-6G/GIA locus while holding the LEVEL FIXED, so
it covers one axis of uncertainty and is silent about the other.  The curve carries
the other axis in two columns, ``unc_low`` / ``unc_high``, and until now nothing read
them.

**WHAT THOSE TWO COLUMNS ARE, AND THE CLAIM THAT WAS WITHDRAWN.**  They are carried
**once per curve ROW** and are read identically whichever scenario column is asked
for — one pair, six level columns.  Measured on all 71 rows: no level column at any
row falls outside the pair, and 63 distinct pairs serve 71 rows.  At 16.00 ka the
pair is [+22.0, +50.0] while ``defensible`` reads +35.0 and ``maximal`` +48.0, so it
brackets BOTH schools at once.  It is therefore an **inter-school scenario
envelope**, not a per-scenario uncertainty band, and the first version of this code
said otherwise: it took ``scenario`` into ``LevelBand.scenario`` and reported fields
called ``band_*`` inside a per-scenario report, so a reader of
``khvalynian_sill_test.csv`` seeing ``scenario=maximal, level_f2_m=+33.4,
band_lo_f2_m=+17.1`` read +17.1 as the maximal column's own lower bound.  It is not.
Every row of the report now carries ``envelope_scope``, which says what the pair is
and how many level columns it brackets, measured at that row.

A direct consequence, and it is also in the data: the whole envelope triple —
``envelope_lo_f2_m``, ``envelope_hi_f2_m``, ``margin_envelope_lo_m``,
``margin_envelope_hi_m``, ``verdict_envelope_lo``, ``verdict_envelope_hi``,
``envelope_verdict``, ``verdict_is_envelope_robust``, ``verdict_is_jointly_robust`` —
is **identical between ``defensible`` and ``maximal`` at all 43 epochs** (verified;
``envelope_is_scenario_invariant`` carries it per row).  It repeats in a per-scenario
report because the report has one row per scenario, not because the scenario changed
anything.

Read, converted at the threshold with the SAME term as the central, and put through
the same sill test, the two edges say:

* the envelope's **LOWER edge is endorheic at every one of the 43 epochs**, in both
  scenarios.  Not "rarely spills" — never.  At 16.5 ka it is 5.66 m short;
* the envelope's **UPPER edge spills 18.0 -> 13.75 ka** — but see the circularity
  paragraph below before citing either end of that;
* the lower edge comes closest at **16.0 ka, 0.73 m short** — well inside the +/-2 m
  the curve's own header puts on the palaeo-sill.  Push the locus to its p95 on top
  of that and five epochs (16.00 / 15.75 / 15.50 / 15.25 / 15.00) stop reading
  ``endorheic`` and read ``at_sill``, missing by 0.130 / 0.169 / 0.176 / 0.125 /
  0.071 m.  ``verdict_is_jointly_robust`` is that flag; those five are its only
  False rows, and they do not overlap ``verdict_is_robust``'s single one (14.0 ka).

THE UPPER EDGE IS NEAR-CIRCULAR WHERE IT SITS ON THE SILL
----------------------------------------------------------
``verdict_envelope_hi`` is **not independent evidence of a spill at every epoch**, and
which epochs those are is established by :func:`envelope_hi_circularity` from the
curve, not from a hardcoded list of ages.  At 17.70 / 17.50 / 17.40 ka the column
``level_m_global_early_rise`` reads exactly **+25.0 m** and ``unc_high`` equals it;
``early_rise`` is the Semikolennykh et al. 2022 Manych-spillway OSL reading, in which
water stands ON the spillway floor BY CONSTRUCTION — the 17.7 ka note says so in the
curve itself ("THE VALUE IS THE SILL, NOT A LEVEL: +25.0 m is MANYCH_SILL_PATH_M").
At 18.00 ka ``unc_high`` is +25.0 for the same reason without matching a column: the
note records that the old +13.5 m ceiling was withdrawn because a hiatus "cannot
exclude a short excursion inside itself, so it is not a ceiling at 18 ka and unc_high
now carries +25".

So at those rows the envelope's upper edge was derived from the proposition *"water
was at the sill"*, and asking whether it clears the sill is close to circular.  Of the
43 baked epochs exactly **18.0 and 17.5 ka** are affected — which are precisely the
two epochs where ``verdict_envelope_hi`` spills OUTSIDE the 17-13 ka window, i.e. the
two frames where the upper edge is the only thing that reaches the crest.  Those rows
carry ``envelope_hi_is_circular=True`` plus the measured reason, the caption refuses
to cite them, and :func:`spill_window_ka` will not report an upper-edge window
without being told what to do about them.

PROVENANCE OF THE EDGES — WHAT IS AND IS NOT MACHINE-READABLE
--------------------------------------------------------------
The earlier literal ``"UNKNOWN:not_recorded_in_the_curve"`` **overstated the gap, on
the very rows the headline rests on**, and it has been withdrawn.  There is no
``unc_source`` and no ``unc_interp_flag`` COLUMN — that much is true and is what
:data:`~scripts.bake_black_sea.ENVELOPE_PROVENANCE_NO_COLUMN` now says.  But the
``note`` column DOES record the edges' sources in prose: the 15.0 ka note records
that HESS2024 carries the reliably-dated level as a band "+22 to +35 m a.s.l.
reliably dated to 18(17)-13 kyr BP", that +35 is its TOP EDGE and that ``unc_low``
carries the +22 m bottom edge from Makshaev & Tkach; the 14.0 ka note records a
carried conflict.  Measured: 5 of 71 rows name an edge column outright.  Asserting
"not recorded" where the curve records it is a false provenance claim, the same class
of error as substituting a number, so each row now reports whether ITS note records
its edges (``envelope_provenance_in_note``).

The caption still says "the envelope **as carried in the curve**" and still may not
say "the published band" — but now because the pair is not a published per-scenario
band at all, not because its sources are unrecorded.

The envelope is NOT a third scenario.  Synthesizing a low-scenario level column is
refused twice over: ``max(unc_low)`` = +22.0 m is below ``MANYCH_SILL_M`` = +23.0 m,
so ``validate_basin_curve``'s Check A rejects such a file outright, and the
vertical-datum contract's §8 requires the conversion to happen at the threshold and
never in the curve.

TWO GUARDS ON READING A CURVE, SHARED WITH THE PONTIC BAKE
-----------------------------------------------------------
Both live in ``bake_black_sea`` and both are refusals, and neither changes a number
this grid produces (every one of the 43 epochs lands on an exact Caspian row):

* :func:`~scripts.bake_black_sea.basin_level`'s nearest-age lookup is now BOUNDED —
  past ``_CURVE_AGE_TOL_KA`` it raises ``CurveAgeGapError`` instead of handing back
  a neighbouring row's number (28.3 m at 16.75 ka on the Pontic curve, the pick
  decided by DataFrame row order);
* :func:`_ice6g_bracket` MOVED to ``bake_black_sea`` — it was written here, and
  ``pontic_threshold`` needs the same mechanism for the 14.7 and 7.15 ka rows, which
  used to die on a raw ``FileNotFoundError``.  It is re-exported under its old name
  here.  Every off-grid result is TAGGED ``ice6g_interpolated=True``.

THE RASTER IS THE PRODUCT, SO IT HAS TO BE READABLE ALONE (added 2026-08-05)
-----------------------------------------------------------------------------
The PNGs were validation; what ships downstream is the per-epoch raster.  A mesher
holding one ``khv_<epoch>ka.tif`` and nothing else could not reconstruct the epoch
from it, for two reasons that were both defects of the TAGS and not of the pixels.

**1. The nodata declaration was in the wrong unit and said so nowhere.**
``gia_unknown_ice6g_cells`` reads 101 at 18 ka; the file holds **305 600** nodata
cells at 18 ka.  Both are true and they count different things — ICE-6G 10-arcmin
MODEL cells whose ``Topo`` is the ice surface, against 15-arcsec RASTER cells
carrying the sentinel — and no tag named either unit, so the obvious reading
understated the hole by three orders of magnitude.  Both counts now ship under names
that carry their unit (``gia_unknown_ice6g_model_cells``, ``nodata_raster_cells``),
with the hole's bounding box, its contact with each border, and the consequence:
the hole reaches the north and west edges (1 760 of 7 800 cells on row 0, 180 of
4 320 on column 0 at 18 ka), so **any fill of it is extrapolation, not
interpolation**.  The raster count is MEASURED per epoch off the file — it cannot be
derived from the model grid, because the ice-aware smoothing rim widens the hole to
~3 026 raster cells per unknown model cell against a nominal 1 600.

**2. The water surfaces were not machine-readable.**  The Caspian's existed only
inside a 1 299-character prose ``threshold_*`` tag, the Mediterranean was implicit as
the negation of ``gia_sea_surface_offset_m`` with nothing saying so, and the Pontic
was absent.  Three JSON tags now carry one water sheet each — value, unit, frame and
the RULE that produced it (:func:`water_level_tags`).  The Pontic is **not computable
at 8 of the 43 epochs**: its curve is on a 0.5 ka grid and this bake samples 0.25 ka
across 17-13 ka.  Those eight carry an explicit ``null`` plus that epoch's own
measured reason, never an interpolant — at 16.75 ka the bracketing rows are converted
by DIFFERENT rules (``locus`` at 17.0, ``marker`` at 16.5), so an interpolant would
belong to neither estimator, and the two levels are 28.30 m apart with both rows
exactly 0.25 ka away, i.e. a tie DataFrame row order would break.

:func:`retag_paleo_dem` adds this set to rasters that are ALREADY BAKED, in place.
It writes no pixel and proves it: the band is sha256-ed before and after and a
difference raises :class:`PixelDataChangedError`.  It is idempotent, so a re-run is
free, and it snapshots each raster's pre-pass tag state to a sidecar exactly once.

OUTPUT
------
  $PALEO_OUT_ROOT/khvalynian_caspian/topo/khv_<epoch>ka.tif   (paleo-DEM, tagged)
  $PALEO_OUT_ROOT/khvalynian_caspian/khvalynian_sill_test.csv (THE sill report)
  $PALEO_OUT_ROOT/khvalynian_caspian/khvalynian_mesher_retag.csv (the tag pass)
  $PALEO_OUT_ROOT/khvalynian_caspian/_retag/khv_<epoch>ka.tags-before.json

The report is written by any run that writes rasters, and by a report-only run given
``--write``.  Its 22 original columns are unchanged; everything the envelope work
added is additive and named ``envelope_*``, and everything the mesher pass added is
additive and listed in the ``mesher_tags`` tag.

ROOTS
-----
``PALEO_DATA_ROOT`` / ``PALEO_OUT_ROOT``, resolved by ``bake_black_sea`` and reused
verbatim so the two bakes cannot end up pointing at different trees.  The working
raster is located through ``KHVALYNIAN_DEM_PATH`` (default: the verified COG on Y:).

Run (repo NOT pip-installed).  THE WRITE GATE IS ON THE *QUESTION*, NOT ON THE BAKE
(re-scoped 2026-08-03): a real bake always refreshes the report beside the rasters it
just wrote, because a stale report next to fresh rasters is the failure this gate
exists to prevent; ``--sill-report-only`` — the flag whose whole purpose is "compute
this in seconds and show me" — writes nothing unless ``--write`` says so.
    PYTHONPATH=$(pwd) python scripts/bake_khvalynian.py --sill-report-only
    PYTHONPATH=$(pwd) python scripts/bake_khvalynian.py --sill-report-only --write
    PYTHONPATH=$(pwd) python scripts/bake_khvalynian.py --epochs 16.5

The additive tag pass over rasters that already exist.  It reads every band twice —
~2.5 min per epoch, ~1.8 h over the 43-epoch grid — so it OUTLIVES a backgrounded
shell task and must be launched detached.  ``PALEO_OUT_ROOT`` is not optional here:
unset, it resolves to the repo's sibling directory and the pass will happily retag
whatever partial tree it finds there instead of the product tree.
    PALEO_OUT_ROOT=... python scripts/bake_khvalynian.py --retag-dry-run --epochs 18
    PALEO_OUT_ROOT=... python scripts/bake_khvalynian.py --retag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import Window

# Make the package importable when run directly (repo is not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paleoeurope.gia.deformation import apply_gia_delta  # noqa: E402
from paleoeurope.gia.ice6g_topo import (  # noqa: E402
    SMOOTHING_RIM_CELLS,
    ice_surface_mask,
    implied_sea_surface_offset,
    smooth_preserving_ice_mask,
    solid_displacement,
)

# The Pontic bake owns the threshold machinery, the datum guards, the NaN policy and
# the curve validator.  They are IMPORTED, not re-implemented: two copies of a
# scientific guard is two guards that can disagree, and the constants here (the
# Manych sill, the nodata value, the σ, the policy names) are the same physical
# facts.  Nothing in that module is edited by this one.
from scripts import bake_black_sea as _bs  # noqa: E402
from scripts.bake_black_sea import (  # noqa: E402
    _CURVE_AGE_TOL_KA,
    _MARKER_COLUMNS,
    BAKE_GIA_NAN_POLICIES,
    BS_NODATA_M,
    ENVELOPE_COLUMNS,
    ENVELOPE_PROVENANCE_NO_COLUMN,
    ENVELOPE_RULES,
    GIA_SIGMA_BLUR,
    ICE6G_DIR,
    ICE6G_ICEMASK_VAR,
    MANYCH_SILL_M,
    PALEO_DATA_ROOT,
    PALEO_OUT_ROOT,
    PONTIC_CSV,
    SCENARIOS as PONTIC_SCENARIOS,
    BasinThreshold,
    CurveAgeGapError,
    LevelEnvelope,
    MissingEnvelopeError,
    UnconvertibleThresholdError,
    UnknownGiaInBakeError,
    _check_threshold_sanity,
    _convert_envelope,
    _curve_row_index,
    _epoch_filename,
    _ice6g_bracket,
    _ice6g_time_note,
    _parse_marker_position,
    _require_model_grid_brackets_tile,
    basin_level,
    basin_level_envelope,
    envelope_note_provenance,
    envelope_scope,
    ice6g_ages_ka,
    ice6g_is_interpolated,
    level_columns,
    pontic_level,
    pontic_threshold,
    validate_basin_curve,
)

#: RE-EXPORTED DELIBERATELY — the objects, not their names, so that removing one
#: from the import list above is a NameError here rather than a silent API change.
#:
#: ``_ice6g_bracket`` / ``ice6g_ages_ka`` were WRITTEN in this module and now live
#: in ``bake_black_sea``, because ``pontic_threshold`` needs the same mechanism and
#: the import can only go one way.  They keep their names in this namespace so every
#: caller, raster tag and test that says ``bake_khvalynian._ice6g_bracket`` still
#: means the ONE implementation.  Behaviour is unchanged: an epoch ICE-6G ships
#: returns ``(age, age, 0.0)`` and no interpolation code runs.
#:
#: The envelope types are re-exported for the same reason — a reader of the
#: Khvalynian bake should not have to know which module raises when the ``unc_low`` /
#: ``unc_high`` pair is unusable.
_REEXPORTED = (
    ice6g_ages_ka,
    _ice6g_bracket,
    ice6g_is_interpolated,
    MissingEnvelopeError,
    CurveAgeGapError,
    LevelEnvelope,
    basin_level_envelope,
    level_columns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: The working raster for the Caspian scene: a 33 MB tiled COG, 7800x4320 at 15
#: arcsec, EPSG:4326, 25.5-58 E / 36-54 N — Pontic + Manych + the whole Caspian
#: including the southern deep.  Verified bit-identical to the GEBCO 2024 source at
#: 14 probe points.  Overridable so the bake runs on the Linux compute box.
KHV_DEM = Path(
    os.environ.get("KHVALYNIAN_DEM_PATH")
    or r"Y:\paleo\data\gebco_2024_khvalynian_pontic_n54.0_s36.0_w25.5_e58.0_cog.tif"
)

CASPIAN_CSV = (
    PALEO_DATA_ROOT
    / "docs"
    / "superpowers"
    / "research"
    / "2026-07-28-caspian-khvalynian-level-curve.csv"
)

OUT_BASE = PALEO_OUT_ROOT / "khvalynian_caspian"
OUT_TOPO = OUT_BASE / "topo"
SILL_REPORT_CSV = OUT_BASE / "khvalynian_sill_test.csv"

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------

#: Scene bbox = the working raster's own extent.  Stated in CODE rather than read
#: from the file so that :func:`_require_model_grid_brackets_tile` has something to
#: check the raster AGAINST, and so a swapped raster is caught instead of silently
#: redefining the scene.
#:
#: TWO working rasters now exist and the second is not a replacement: the 58 E / 54 N
#: tile is what the shipped 43-epoch Khvalynian set was baked on, and the 60 E / 62 N
#: tile is the drainage_implied window.  Reading the bbox off whichever file happens
#: to be configured would make both of those scenes unreproducible from the code and
#: would hand the guard its own input to check itself against.  So the pairing is a
#: TABLE: a raster this table does not name is REFUSED, by filename, before anything
#: opens it.
#:
#: 60 E is not a design choice.  It is the eastern edge of the GEBCO 2024 tile this
#: project holds (``gebco_2024_n75.0_s25.0_w-15.0_e60.0.tif``); the window cannot be
#: widened past it from this source, whatever the science would prefer.
KHV_SCENES: dict[str, tuple[float, float, float, float]] = {
    # filename: (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    "gebco_2024_khvalynian_pontic_n54.0_s36.0_w25.5_e58.0_cog.tif": (36.0, 54.0, 25.5, 58.0),
    "gebco_2024_khvalynian_pontic_n62.0_s36.0_w25.5_e60.0_cog.tif": (36.0, 62.0, 25.5, 60.0),
}


def _scene_for(dem_path: Path) -> tuple[float, float, float, float]:
    """The declared scene bbox for *dem_path*, or a refusal.

    Refusing an unnamed raster is the point.  The alternative -- defaulting to the
    historical 58 E / 54 N box -- would bake a wider tile's northern and eastern
    margin with 0 m of GIA, which ``apply_gia_delta``'s ``fill_value=0.0`` produces
    silently and which is indistinguishable from a measured zero over stable
    far-field ground.  :func:`_require_model_grid_brackets_tile` catches exactly
    that case downstream; this catches it before the first byte is read.
    """
    name = Path(dem_path).name
    try:
        return KHV_SCENES[name]
    except KeyError:
        raise SystemExit(
            "REFUSING an unknown working raster: %r.\n"
            "The scene bbox lives in KHV_SCENES, not in the file, so that a swapped "
            "raster cannot silently redefine the scene. Known rasters:\n  %s\n"
            "If this really is a new scene, add it to KHV_SCENES with its bbox "
            "written out, and re-read the note there about 60 E being the GEBCO "
            "tile's own eastern edge."
            % (name, "\n  ".join(sorted(KHV_SCENES)))
        ) from None


LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = _scene_for(KHV_DEM)


def _require_tile_is_the_declared_scene(src) -> None:
    """Refuse a raster whose extent is not the bbox :data:`KHV_SCENES` declares.

    The comment above KHV_SCENES has always said "a swapped raster is caught
    instead of silently redefining the scene", and until this function existed
    that was only half true.  ``_require_model_grid_brackets_tile`` catches a tile
    that reaches OUTSIDE the GIA window; a tile that is SMALLER than the declared
    box sails through it, because a smaller tile is bracketed even more
    comfortably.  The failure that leaves is quiet: every downstream consumer --
    the crop of record, the sill probe's window, the nodata census, the basin
    fields' ``row_off`` -- computes its rows and columns from LAT_MIN..LAT_MAX,
    so a short raster yields a frame that is internally consistent, correctly
    tagged, and about the wrong ground.

    Tolerance is half a pixel: the bbox is stated in whole degrees and the grid is
    15 arcsec, so anything the same scene can legitimately differ by is far below
    it, and anything bigger is a different raster.

    SCOPE, stated rather than left to be discovered.  This checks a raster that
    CLAIMS to be a declared scene -- i.e. whose filename is a key of
    :data:`KHV_SCENES` -- against the bbox declared for that key.  A raster with
    any other name is passed through untouched, because it is not making the
    claim: ``build_paleo_dem`` is called with small synthetic tiles by the unit
    tests, and demanding that a fixture covering 43-45 E / 45-47 N be the whole
    Caspian scene is a category error, not a guard.  The swapped-raster case this
    exists for is a file replaced UNDER a declared name, and that is exactly what
    is checked.  An undeclared raster is still bounded by
    :func:`_require_model_grid_brackets_tile`, and ``KHV_DEM`` itself cannot be
    undeclared -- :func:`_scene_for` refuses it at import.

    It also reads the bbox from the table entry rather than from the module-level
    ``LAT_MIN``/``LON_MIN``, so that a caller baking a second scene explicitly
    cannot be checked against the first one's box.
    """
    declared_bbox = KHV_SCENES.get(Path(getattr(src, "name", "")).name)
    if declared_bbox is None:
        return
    lat_min, lat_max, lon_min, lon_max = declared_bbox
    half_px = max(abs(src.res[0]), abs(src.res[1])) / 2.0
    declared = {"W": lon_min, "S": lat_min, "E": lon_max, "N": lat_max}
    actual = {"W": src.bounds.left, "S": src.bounds.bottom,
              "E": src.bounds.right, "N": src.bounds.top}
    off = {k: actual[k] - declared[k] for k in declared
           if abs(actual[k] - declared[k]) > half_px}
    if off:
        raise ValueError(
            "the DEM tile is not the scene KHV_SCENES declares for it. Declared "
            "W=%g S=%g E=%g N=%g; the raster reads W=%g S=%g E=%g N=%g; edges off "
            "by more than half a pixel (%.6g deg): %s. Either the file was swapped "
            "under its name, or KHV_SCENES is wrong about it -- do not proceed on "
            "the assumption that one of them is close enough."
            % (lon_min, lat_min, lon_max, lat_max,
               actual["W"], actual["S"], actual["E"], actual["N"], half_px,
               ", ".join("%s%+.6g" % (k, v) for k, v in sorted(off.items())))
        )

#: Region buffer (deg) for the ICE-6G subset around the scene bbox.
BUF = 1.0

#: Nodata written into the per-epoch paleo-DEM GeoTIFF.  Bound to the Pontic bake's
#: value so the two trees cannot drift onto different sentinels.
KHV_NODATA_M = BS_NODATA_M

#: Default NaN policy.  "nodata", not "raise" — this window is NOT ice-free before
#: 17 ka (see the module docstring and :func:`ice_cells_in_window`).  There is still
#: no "zero": 0 m of correction is a legitimate measured value over stable far-field
#: ground, so writing it for an unknown cell makes the two indistinguishable.
KHV_GIA_NAN_POLICY = "nodata"

#: Rows read per windowed pass over the working raster.  The COG's internal blocks
#: are 512x512, so a multiple of 512 keeps strip boundaries on block boundaries.
#: The full 7800x4320 float32 tile would be ~135 MB resident; a strip is ~16 MB.
STRIP_ROWS = 512

# ---------------------------------------------------------------------------
# The event: the Manych sill
# ---------------------------------------------------------------------------

#: The saddle point of the Kuma-Manych depression, measured 2026-07-28 from GEBCO
#: 2024 (15 arcsec).  :func:`manych_sill_elevation_m` re-reads the working raster
#: here, so the constant and the topography the bake uses cannot drift apart.
MANYCH_SILL_LON, MANYCH_SILL_LAT = 44.119, 45.773

#: Continuous-path controlling col of the same raster: the minimax over any
#: connected Caspian -> Don/Azov route, identical under 4- and 8-connectivity and
#: disconnected at +24 m (curve CSV header).  A spillway needs a CONTINUOUS channel,
#: so this is the hydrologically meaningful crest; ``MANYCH_SILL_M`` = +23.00 m is
#: the max of per-column minima, i.e. a LOWER BOUND on it.  Both verdicts are
#: carried and neither is asserted — +25.00 closes the spill window by ~0.3 ka.
MANYCH_SILL_PATH_M = 25.00

#: Half-width (m) of the ``at_sill`` band around a zero margin.  Matches the +/-0.5 m
#: tolerance the curve's own header uses for its spill-state columns, so the two
#: readings are directly diffable.
AT_SILL_TOL_M = 0.5

#: Verdict vocabulary, coarsest-to-finest ordering irrelevant — these are the three
#: states the curve CSV already uses, reproduced verbatim so a diff is possible.
SPILL_STATES = ("spilling_west", "at_sill", "endorheic")

# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------

#: The Caspian curve's contested axis is the MAGNITUDE of the highstand, not the
#: timing of a reconnection, so its scenarios are not the Pontic's.
#:   defensible — the reliably-DATED central line, peaking at +35 m
#:                (arslanov2016 Table 1 No 1, Ergeni, 16.14 +/- 0.68 cal ka BP).
#:   maximal    — the classic +48 m maximum, which no study has ever dated.
SCENARIOS = ("defensible", "maximal")

#: 26-19 ka level window (m a.s.l.).  Established: the Caspian sat at +5 to +15 m
#: through the LGM, i.e. 33-43 m ABOVE modern.  This bound exists to catch a
#: re-imported Atelian -100 m lowstand (MIS 4 / early MIS 3, ~70-40 ka — a
#: mis-dating when placed at the LGM) and an absurd highstand, in one check.
LGM_PLATEAU_WINDOW_M = (0.0, 20.0)
LGM_PLATEAU_KA = (19.0, 26.0)


def _build_epoch_grid() -> tuple[float, ...]:
    """0.25 ka across 17-13 ka, coarser outside.  See the module docstring.

    Built rather than written out so the fine window is one edit, and so the
    "coarser outside" claim is structural instead of a hand-typed list that can
    quietly acquire a gap.
    """
    coarse_glacial = [26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0]
    lead_in = [17.5]
    fine = [round(17.0 - 0.25 * k, 2) for k in range(17)]  # 17.00 .. 13.00
    tail = [12.5, 12.0, 11.5, 11.0, 10.5, 10.0]
    coarse_holocene = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    return tuple(coarse_glacial + lead_in + fine + tail + coarse_holocene)


EPOCHS_KA: tuple[float, ...] = _build_epoch_grid()

#: The interval :data:`EPOCHS_KA` samples at 0.25 ka.  Exported so a test can assert
#: the spacing instead of re-deriving it.
FINE_WINDOW_KA = (13.0, 17.0)
FINE_STEP_KA = 0.25


# ---------------------------------------------------------------------------
# The locus: what stands in for the missing marker column
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaspianLocus:
    """The region an unlocated Caspian F2 level could have been measured in.

    Deliberately NOT ``bake_black_sea._Locus``.  That type's band field is named
    ``modern_depth_m`` because every Pontic locus is below sea level; the Caspian
    shoreline band is mostly ABOVE it (-30 .. +50 m), and storing +50 m in a field
    called "depth" would mislabel the number.  Same idea, honest field name.
    """

    name: str
    lon_deg: tuple[float, float]
    lat_deg: tuple[float, float]
    modern_elev_m: tuple[float, float]
    why: str


#: THE locus.  Bounds are INCLUSIVE and this is the definition of record.
#:
#: The geographic box is the Caspian lowland the curve's own dated anchors sit in —
#: Ergeni, Raygorod, Cherniy Yar, Leninsk (Lower Volga) and Mangyshlak (east
#: Caspian).  **Their individual coordinates are deliberately NOT hard-coded**: no
#: source consulted for this curve prints them, and an invented lon/lat is
#: indistinguishable downstream from a measured one — the exact defect class the
#: Pontic marker phase exists to remove.  The box is a statement about the region,
#: which is a thing the sources do establish.
#:
#: The elevation band is read from ICE-6G ``Topo`` at 0 ka on the SAME 10-arcmin grid
#: the Δz field lives on, so band and sample never come from different rasters.  It
#: spans the curve's own level range (-28 m modern surface to +48 m maximal
#: highstand) with a margin: that is exactly the ground across which the Khvalynian
#: shoreline moves, i.e. where a dated shoreline deposit can be.
CASPIAN_SHORELINE_BAND = CaspianLocus(
    name="caspian_khvalynian_shoreline_band",
    lon_deg=(44.0, 54.0),
    lat_deg=(42.0, 50.0),
    modern_elev_m=(-30.0, 50.0),
    why=(
        "the Caspian lowland shoreline band, i.e. the ground the Khvalynian "
        "shoreline actually crosses between the modern -28 m surface and the +48 m "
        "maximal highstand; the honest locus for a curve that carries no marker "
        "columns at all (arslanov2016 Ergeni; kurbanov2020 Raygorod/Leninsk; "
        "kurbanov2023 Lower Volga; yanina2018 Mangyshlak - region attested, "
        "coordinates not published, and not invented here)"
    ),
)

#: Minimum locus cells for a median and a p5-p95 bar to mean anything.
_MIN_LOCUS_CELLS = 5


# ---------------------------------------------------------------------------
# ICE-6G: the shipped age grid, and linear-in-time interpolation between it
# ---------------------------------------------------------------------------


def _time_interpolation_note(epoch: float) -> str:
    """Human/tag-readable statement of how *epoch* got its GIA field.

    Kept as this module's name for :func:`~scripts.bake_black_sea._ice6g_time_note`
    (the raster tag ``gia_time_interpolation`` and :class:`SillTest` both read it),
    so moving the implementation next to :func:`_ice6g_bracket` did not rename a
    string that ends up inside published GeoTIFF tags.
    """
    return _ice6g_time_note(epoch)


@dataclass(frozen=True)
class _KhvWindow:
    """Raw ICE-6G fields for one SHIPPED epoch: global datum + regional subset."""

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


def _icemask_subset(ds_sub) -> np.ndarray | None:
    """``sftgif`` (percent) from an already-subset dataset, or ``None``.

    ``None`` means *no ice information*, not *no ice*.
    """
    if ICE6G_ICEMASK_VAR not in ds_sub.variables:
        return None
    return ds_sub[ICE6G_ICEMASK_VAR].values.astype(np.float64)


@lru_cache(maxsize=8)
def _open_khv_window(epoch: float) -> _KhvWindow:
    """Read one SHIPPED ICE-6G epoch: GLOBAL datum, REGIONAL fields.

    THE DATUM IS GLOBAL, THE DISPLACEMENT IS LOCAL.  ``implied_sea_surface_offset``
    is handed the **global** ``Topo`` field — the median ΔTopo over far-field
    ice-free tropical continental land is a global quantity that a regional cut
    cannot define, and taking it from a regional (or deep-ocean) cut is precisely
    the bug retired on 2026-07-30.  Only the regional subset reaches
    ``solid_displacement``.

    Cached: every threshold, every sill test and every raster for an epoch reads the
    same window, so they cannot drift onto different datums or grids.
    """
    f_t = ICE6G_DIR / _epoch_filename(epoch)
    f_0 = ICE6G_DIR / _epoch_filename(0)
    lat_lo, lat_hi = LAT_MIN - BUF, LAT_MAX + BUF
    lon_lo, lon_hi = LON_MIN - BUF, LON_MAX + BUF

    with xr.open_dataset(f_t) as ds_t, xr.open_dataset(f_0) as ds_0:
        offset = implied_sea_surface_offset(
            ds_t["Topo"].values.astype(np.float64),
            ds_0["Topo"].values.astype(np.float64),
        )
        sub_t = ds_t.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        sub_0 = ds_0.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
        return _KhvWindow(
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


def _raw_solid_displacement(epoch: float) -> np.ndarray:
    """Δz over the scene window at a SHIPPED epoch, RAW 10-arcmin cells, NaN over ice."""
    win = _open_khv_window(epoch)
    return solid_displacement(
        win.topo_t,
        win.topo_0,
        win.offset,
        sftgif_t=win.sftgif_t,
        sftgif_0=win.sftgif_0,
    )


def sea_surface_offset(epoch: float) -> float:
    """``offset(t)`` in metres — the sea-surface datum this DEM is built on.

    Median ``ΔTopo`` over far-field ice-free tropical continental land, from the
    **global** ICE-6G field.  ``0.0`` at 0 ka by construction.  Linear in time
    between shipped epochs (:func:`_ice6g_bracket`).

    Its negation is the exact land/sea companion of ``H_paleo`` for a cell ICE-6G
    treats as OCEAN (contract §2) — which is why it appears in this module only as
    provenance.  Neither the Caspian nor the Pontic is an ocean at any epoch this
    bake covers, so no threshold here is ``-offset(t)``.
    """
    lo, hi, w = _ice6g_bracket(epoch)
    if lo == hi:
        return _open_khv_window(lo).offset
    return (1.0 - w) * _open_khv_window(lo).offset + w * _open_khv_window(hi).offset


def solid_displacement_field(epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Δz, lats, lons)`` over the scene window, RAW cells, NaN over ice.

    Deliberately **unsmoothed**, unlike the field that reaches the raster
    (``GIA_SIGMA_BLUR`` = 2.0 cells).  A locus statistic is a set of point probes;
    blurring each over ~5 neighbouring cells would mix in ground the locus does not
    describe, and the contract's own measurements were taken on raw cells, so the
    numbers here are directly comparable with it.  The residual inconsistency — a
    σ=2 raster compared against an unsmoothed threshold — is sub-metre over this
    window and is stated rather than hidden.
    """
    lo, hi, w = _ice6g_bracket(epoch)
    win = _open_khv_window(lo)
    dz = _raw_solid_displacement(lo)
    if lo != hi:
        dz = (1.0 - w) * dz + w * _raw_solid_displacement(hi)
    return dz, win.lats, win.lons


def ice_cells_in_window(epoch: float) -> int:
    """How many ICE-6G cells of the scene window are an ICE SURFACE, not bedrock.

    Non-zero means ``Topo`` there is the ice surface and these files carry no
    bedrock-under-ice field, so Δz is UNKNOWN — not zero.  Measured, never assumed;
    this is the function that establishes the module's ``"nodata"`` default.
    """
    lo, hi, _w = _ice6g_bracket(epoch)
    total = 0
    for age in {lo, hi}:
        win = _open_khv_window(age)
        mask = ice_surface_mask(win.sftgif_t, win.sftgif_0)
        total = max(total, 0 if mask is None else int(np.count_nonzero(mask)))
    return total


# ---------------------------------------------------------------------------
# Sampling: the sill cell and the locus band
# ---------------------------------------------------------------------------


def _cell_index(lats: np.ndarray, lons: np.ndarray, lon: float, lat: float) -> tuple[int, int]:
    """Nearest ICE-6G cell to (*lon*, *lat*), refusing a point outside the window."""
    if not (float(np.min(lats)) <= lat <= float(np.max(lats))):
        raise UnconvertibleThresholdError(
            f"latitude {lat:.4f} N is outside the ICE-6G window "
            f"{float(np.min(lats)):.4f}..{float(np.max(lats)):.4f} N — the Δz there "
            "was never read, and the nearest edge cell is a different place"
        )
    lon_wrapped = ((np.asarray(lons, dtype=float) - lon + 180.0) % 360.0) - 180.0
    if float(np.abs(lon_wrapped).min()) > 1.0:
        raise UnconvertibleThresholdError(
            f"longitude {lon:.4f} E is outside the ICE-6G window "
            f"{float(np.min(lons)):.4f}..{float(np.max(lons)):.4f} E"
        )
    return int(np.abs(np.asarray(lats, dtype=float) - lat).argmin()), int(
        np.abs(lon_wrapped).argmin()
    )


def sill_solid_displacement(epoch: float) -> tuple[float, tuple[float, float]]:
    """``(Δz at the Manych sill cell, 3x3 cell-block min/max)`` in metres.

    The block is the honest local spread of a point probe on a 10-arcmin grid — NOT
    a literature uncertainty on the sill's elevation, which this module has no
    access to.
    """
    dz, lats, lons = solid_displacement_field(epoch)
    i, j = _cell_index(lats, lons, MANYCH_SILL_LON, MANYCH_SILL_LAT)
    value = float(dz[i, j])
    if not np.isfinite(value):
        raise UnconvertibleThresholdError(
            f"Δz at the Manych sill cell is not finite at {epoch:g} ka — ICE-6G "
            "'Topo' is the ice surface there and these files carry no "
            "bedrock-under-ice field, so the sill's palaeo-crest is UNKNOWN"
        )
    block = dz[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
    finite = block[np.isfinite(block)]
    return value, (float(finite.min()), float(finite.max()))


def locus_solid_displacement(
    epoch: float, locus: CaspianLocus = CASPIAN_SHORELINE_BAND
) -> tuple[float, tuple[float, float], int]:
    """``(median Δz, (p5, p95), n_cells)`` over the locus band, in metres."""
    dz, lats, lons = solid_displacement_field(epoch)
    topo_0 = _open_khv_window(_ice6g_bracket(epoch)[0]).topo_0
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0
    lo_e, hi_e = locus.modern_elev_m
    mask = (
        (lon_wrapped >= locus.lon_deg[0])
        & (lon_wrapped <= locus.lon_deg[1])
        & (lat2d >= locus.lat_deg[0])
        & (lat2d <= locus.lat_deg[1])
        & (topo_0 >= lo_e)
        & (topo_0 <= hi_e)
        & np.isfinite(dz)
    )
    values = dz[mask]
    if values.size < _MIN_LOCUS_CELLS:
        raise UnconvertibleThresholdError(
            f"locus {locus.name!r} contains only {values.size} usable ICE-6G cells "
            f"at {epoch:g} ka — too few for a median and a p5-p95 bar, so there is "
            "no defensible conversion term for this epoch"
        )
    return (
        float(np.median(values)),
        (float(np.percentile(values, 5)), float(np.percentile(values, 95))),
        int(values.size),
    )


def sill_marker_differential(epoch: float) -> tuple[float, tuple[float, float]]:
    """``Δz(sill) − Δz(locus)``, and the bracket the locus spread puts on it.

    THIS is the quantity that decides whether a basin-band locus may stand in for
    the marker columns the Caspian curve does not have.  Contract §6 argues it is
    negligible for this basin; that argument is **re-measured here**, per epoch, and
    the number carried — see the module docstring for what this module's own locus
    gives (-0.29 .. +0.18 m over 18-12 ka), and note it is NOT the one-signed
    "+0.03 .. +0.18 m" the contract reports for a different band.

    The bracket is ``Δz(sill) − p95`` .. ``Δz(sill) − p5``: how far the differential
    could move if the true marker sat at either edge of the locus rather than at its
    median.  It is wider than the differential itself, which is the point — a
    verdict that flips inside this bracket is not a verdict.
    """
    dz_sill, _block = sill_solid_displacement(epoch)
    median, (p5, p95), _n = locus_solid_displacement(epoch)
    return dz_sill - median, (dz_sill - p95, dz_sill - p5)


# ---------------------------------------------------------------------------
# The threshold: F2 -> F1
# ---------------------------------------------------------------------------


def check_no_lgm_lowstand(
    df: pd.DataFrame, scenarios: Sequence[str] = SCENARIOS
) -> None:
    """Raise if any 26-19 ka level leaves :data:`LGM_PLATEAU_WINDOW_M`.

    Established, and made executable here rather than left in prose: the Caspian
    stood at **+5 to +15 m a.s.l. through 26-19 ka**, i.e. 33-43 m ABOVE modern.
    Yanina's -100 m Atelian lowstand belongs to MIS 4 / early MIS 3 (~70-40 ka) and
    is a mis-dating when placed at the LGM; kurbanov2020's -150 m appears once, in an
    unsourced final sentence of its Conclusions.  A curve that re-imports either
    would animate a lowstand that did not happen, and this bake refuses to.

    The bound is two-sided on purpose: it also catches a Khvalynian rung
    accidentally dragged back into the plateau.
    """
    lo_ka, hi_ka = LGM_PLATEAU_KA
    lo_m, hi_m = LGM_PLATEAU_WINDOW_M
    ages = pd.to_numeric(df["age_ka"], errors="coerce")
    plateau = (ages >= lo_ka) & (ages <= hi_ka)
    for scenario in scenarios:
        col = f"level_m_global_{scenario}"
        if col not in df.columns:
            raise ValueError(
                f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
            )
        levels = pd.to_numeric(df[col], errors="coerce")
        for idx in df.index[plateau]:
            level = float(levels.loc[idx])
            if not np.isfinite(level) or not lo_m <= level <= hi_m:
                raise ValueError(
                    f"caspian/{scenario} level {level:+.1f} m at "
                    f"{float(ages.loc[idx]):g} ka is outside the established "
                    f"{lo_ka:g}-{hi_ka:g} ka plateau window "
                    f"{lo_m:+.1f}..{hi_m:+.1f} m a.s.l.  The Caspian stood at +5 to "
                    "+15 m through the LGM — 33-43 m ABOVE modern.  A deep value "
                    "here is the Atelian (-100 m, MIS 4 / early MIS 3, ~70-40 ka) "
                    "mis-dated onto the LGM; do not animate a lowstand that did not "
                    "happen."
                )


#: THE crest an envelope edge can have been "set to", and it is the continuous-path
#: controlling col, not the point sill.  Neither constant is changed here; this one
#: is used only to answer "was this edge SET TO the sill?", which is a question about
#: how a number was made and not a frame comparison.
#:
#: WHY NOT ``MANYCH_SILL_M`` (+23.00 m) AS WELL — measured, not assumed.  A basin
#: standing on its spillway floor stands at the CONTINUOUS-PATH col: a spillway needs
#: a connected channel, and +23.00 is the max of per-column minima, i.e. a LOWER
#: BOUND on that col that no water surface is pinned to.  The curve's own 17.7 ka
#: note makes the identification explicitly and makes it against +25: *"THE VALUE IS
#: THE SILL, NOT A LEVEL: +25.0 m is MANYCH_SILL_PATH_M, the continuous-path
#: controlling col measured from GEBCO 2024."*  Including +23.00 was tried and it
#: produces a FALSE POSITIVE: at 18.5 ka ``unc_high`` = +22.5 m is within
#: :data:`AT_SILL_TOL_M` of it, but that row is ``interp_flag=interpolated`` and
#: +22.5 is simply the midpoint of the +20.0 / +25.0 ramp between 19.0 and 18.0 ka —
#: a waypoint that passes near the lower bound, not an edge derived from a spillway.
#: (18.5 ka is not a baked epoch either, so this changes no shipped verdict; the
#: criterion still has to be right on the curve rather than only on the grid.)
_CREST_CONSTANT_M: tuple[str, float] = ("MANYCH_SILL_PATH_M", MANYCH_SILL_PATH_M)


def envelope_hi_circularity(df: pd.DataFrame, epoch: float) -> tuple[bool, str]:
    """Is this row's ``unc_high`` derived from "water was at the sill"?

    Returns ``(is_circular, reason)``.  **Established by MEASUREMENT on the curve, not
    by a hardcoded list of ages** — the ages this fires at are an output of the
    reading, so a curve edit moves them and nothing here has to be remembered.

    Two measurements, and the edge is circular if the second holds:

    1. does ``unc_high`` coincide with a LEVEL COLUMN's value at this row?  Measured
       across every ``level_m_global_*`` column the file carries.  On the shipped
       Caspian curve only ``early_rise`` ever does, at 20.0 / 17.7 / 17.5 / 17.4 ka.
    2. does ``unc_high`` stand AT the measured Manych spillway crest
       (:data:`_CREST_CONSTANT_M`), within this module's own :data:`AT_SILL_TOL_M`?
       That is the test for "the edge IS the sill", and it is what makes the
       comparison circular: an upper edge placed on the crest cannot come out of a
       crest test as anything but ``at_sill`` or a hair either side.

    Why (2) and not (1) alone: at 20.0 ka ``unc_high`` equals ``early_rise`` = +15.0 m,
    which is 10 m BELOW the crest.  Comparing +15.0 m against the sill is perfectly
    informative — it says ``endorheic`` — so that row is not flagged.  Why (1) is
    measured and reported anyway: it names the mechanism.  ``early_rise`` is the
    Semikolennykh et al. 2022 Manych-spillway OSL reading, in which the level is at
    the sill BY CONSTRUCTION, and the curve's own 17.7 ka note says so outright —
    *"THE VALUE IS THE SILL, NOT A LEVEL: +25.0 m is MANYCH_SILL_PATH_M"*.  At 18.0 ka
    (2) holds without (1): the note records that ``unc_high`` was raised to +25 because
    a hiatus "cannot exclude a short excursion inside itself".  Same presupposition,
    same circularity, and a rule keyed only on column equality would have missed it.

    A REVIEWER'S FRAMING THAT IS WRONG, RECORDED SO IT IS NOT RE-LITIGATED: it was
    claimed that ``unc_high`` at these rows *"IS ``MANYCH_SILL_PATH_M``, the F0 modern
    GEBCO crest constant"* — i.e. that this module's constant had been substituted
    into the data.  It had not.  What the curve records, in its own 17.7 ka note, is
    that the +25.0 m is the level Semikolennykh's spillway OSL reading REQUIRES
    ("Water in the strait constrains the level to be AT OR ABOVE the contemporaneous
    sill; it gives no upper bound and it is not a measured Caspian surface"), and
    that the number used for that sill is the GEBCO continuous-path col.  So the edge
    is not a pasted constant, it is a number DERIVED from the proposition the sill
    test asks about — which is why the flag below is about derivation and not about
    equality with our constant.  The defect is real; the framing was not.
    """
    idx = _curve_row_index(df, epoch)
    hi = float(pd.to_numeric(df.loc[idx, ENVELOPE_COLUMNS[1]], errors="coerce"))
    where = f"{float(epoch):g} ka"
    from_columns = tuple(
        str(col)[len("level_m_global_") :]
        for col in level_columns(df)
        if np.isclose(
            float(pd.to_numeric(df.loc[idx, col], errors="coerce")), hi, atol=1e-9
        )
    )
    crest_name, crest_value = _CREST_CONSTANT_M
    if abs(hi - crest_value) > AT_SILL_TOL_M:
        return False, (
            f"NOT CIRCULAR at {where}: unc_high = {hi:+.2f} m is not at the measured "
            f"Manych spillway crest ({crest_name} = {crest_value:+.2f} m, "
            f"{abs(hi - crest_value):.2f} m away, over the {AT_SILL_TOL_M:g} m "
            f"at_sill tolerance), so testing it against the crest carries information"
            + (
                f" (it does coincide with level_m_global_{'/'.join(from_columns)}, "
                f"which is recorded and is not by itself circular)"
                if from_columns
                else ""
            )
        )
    origin = (
        f"it is the value of level_m_global_{'/'.join(from_columns)} at this row"
        if from_columns
        else "it matches no level column at this row; the curve's note records it as "
        "the ceiling that a spillway reading cannot be excluded below"
    )
    return True, (
        f"CIRCULAR at {where}: unc_high = {hi:+.2f} m stands AT the measured Manych "
        f"spillway crest ({crest_name} = {crest_value:+.2f} m, |diff| "
        f"{abs(hi - crest_value):.2f} m <= AT_SILL_TOL_M {AT_SILL_TOL_M:g} m), and "
        f"{origin}.  The Manych-spillway scenario (level_m_global_early_rise, "
        f"Semikolennykh et al. 2022 OSL) puts water ON the spillway floor BY "
        f"CONSTRUCTION, so this edge was derived from the proposition the sill test "
        f"asks about.  verdict_envelope_hi here carries NO INDEPENDENT EVIDENCE of a "
        f"spill and must not be cited as any."
    )


def caspian_threshold(
    df: pd.DataFrame, epoch: float, scenario: str = "defensible"
) -> BasinThreshold:
    """Land/sea threshold for one Caspian frame, in the paleo-DEM's own frame.

    ``T(t) = level_F2(t) + Δz_solid(locus, t)``, applied HERE and never in the curve.

    Of :data:`~scripts.bake_black_sea.THRESHOLD_RULES` only two can ever apply to
    this basin:

    * ``marker`` — if a future rebuild of the curve adds ``marker_lon`` /
      ``marker_lat``, they are preferred and reported as such.  Nothing is invented
      to get there: today the columns are absent and this branch does not run.
    * ``locus``  — the shipped case.  Median Δz over
      :data:`CASPIAN_SHORELINE_BAND`, with its p5-p95 spread carried into the
      frame's own bar.

    ``ocean_offset`` cannot apply: the Caspian is endorheic at every row of the
    curve including 0 ka (-28 m), so no row of it is F4 and ``-offset(t)`` is never
    its threshold.

    THE ENVELOPE IS NOT CAUGHT HERE.
    :func:`~scripts.bake_black_sea.basin_level_envelope` raises
    :class:`~scripts.bake_black_sea.MissingEnvelopeError` and this function lets it
    propagate, unlike ``pontic_threshold`` which records it and carries on.  The
    asymmetry is deliberate and it is not stylistic: on the Pontic side the envelope
    is carried beside a threshold nothing computes from it, while here both edges
    reach :class:`SillTest` and produce ``verdict_envelope_lo`` /
    ``verdict_envelope_hi``.  A frame whose envelope is unusable would still get a
    spill verdict triple, and two thirds of that triple would be fiction.

    THE ENVELOPE IS NOT THIS SCENARIO'S BAND.  ``unc_low`` / ``unc_high`` are carried
    once per curve ROW and bracket all six level columns of this file, so the two
    edges below — and every verdict derived from them — are IDENTICAL for
    ``defensible`` and ``maximal``.  ``envelope_scope`` says that in the data, on
    every row, because a per-scenario report is exactly where that gets misread.

    Raises
    ------
    ValueError
        Unknown scenario, or a converted threshold outside the sanity window.
    CurveAgeGapError
        The curve carries no row at this epoch (every baked epoch does).
    UnconvertibleThresholdError
        The locus is unusable at this epoch (too few finite cells).
    MissingEnvelopeError
        The row's envelope is absent, NaN, inverted, degenerate, or does not contain
        the central it was checked against.  NOT caught here — see above.
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    level = basin_level(df, epoch, scenario, basin="caspian")
    envelope = basin_level_envelope(df, epoch, scenario, basin="caspian")
    where = f"{float(epoch):g} ka / {scenario}"
    interpolated = ice6g_is_interpolated(epoch)
    time_note = _time_interpolation_note(epoch)

    # --- rule 1: a located marker, if the curve ever grows one ----------------
    if all(c in df.columns for c in _MARKER_COLUMNS):
        idx = _curve_row_index(df, epoch)
        position = _parse_marker_position(df.loc[idx], scenario)
        if position is not None:
            lon, lat = position
            dz, lats, lons = solid_displacement_field(epoch)
            i, j = _cell_index(lats, lons, lon, lat)
            conversion = float(dz[i, j])
            block = dz[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
            finite = block[np.isfinite(block)]
            threshold = _check_threshold_sanity(level + conversion, where)
            env_lo, env_hi = _convert_envelope(envelope, conversion, where)
            return BasinThreshold(
                epoch_ka=float(epoch),
                scenario=scenario,
                basin="caspian",
                level_m=level,
                frame="F2",
                rule="marker",
                threshold_m=threshold,
                conversion_m=conversion,
                bracket_m=(level + float(finite.min()), level + float(finite.max())),
                n_cells=None,
                marker_source=str(df.loc[idx, "marker_source"]),
                provenance=(
                    f"rule=marker frame=F2 {where}: T = level({level:+.2f} m) + "
                    f"dz({conversion:+.3f} m) @ {lon:.4f}E/{lat:.4f}N = "
                    f"{threshold:+.3f} m; envelope "
                    f"[{envelope.low_m:+.2f}..{envelope.high_m:+.2f} m] -> "
                    f"[{env_lo:+.3f}..{env_hi:+.3f} m] ({envelope.scope}) "
                    f"({envelope.provenance}); {time_note}"
                ),
                envelope_lo_m=envelope.low_m,
                envelope_hi_m=envelope.high_m,
                threshold_envelope_lo_m=env_lo,
                threshold_envelope_hi_m=env_hi,
                envelope_status="OK",
                envelope_rule=ENVELOPE_RULES[0],
                envelope_provenance=envelope.provenance,
                envelope_scope=envelope.scope,
                ice6g_interpolated=interpolated,
                ice6g_time_note=time_note,
            )

    # --- rule 2: the documented basin band + its p5-p95 bar -------------------
    median, (p5, p95), n_cells = locus_solid_displacement(epoch)
    threshold = _check_threshold_sanity(level + median, where)
    env_lo, env_hi = _convert_envelope(envelope, float(median), where)
    return BasinThreshold(
        epoch_ka=float(epoch),
        scenario=scenario,
        basin="caspian",
        level_m=level,
        frame="F2",
        rule="locus",
        threshold_m=threshold,
        conversion_m=float(median),
        bracket_m=(level + p5, level + p95),
        n_cells=n_cells,
        marker_source="ABSENT:the caspian curve carries no marker columns",
        provenance=(
            f"rule=locus frame=F2 {where}: T = level({level:+.2f} m) + "
            f"dz({median:+.3f} m, median over locus "
            f"{CASPIAN_SHORELINE_BAND.name!r}, {n_cells} cells) = "
            f"{threshold:+.3f} m [p5-p95 {level + p5:+.3f}..{level + p95:+.3f} m] - "
            f"{CASPIAN_SHORELINE_BAND.why}; envelope "
            f"[{envelope.low_m:+.2f}..{envelope.high_m:+.2f} m] -> "
            f"[{env_lo:+.3f}..{env_hi:+.3f} m] ({envelope.scope}) "
            f"({envelope.provenance}); {time_note}"
        ),
        envelope_lo_m=envelope.low_m,
        envelope_hi_m=envelope.high_m,
        threshold_envelope_lo_m=env_lo,
        threshold_envelope_hi_m=env_hi,
        envelope_status="OK",
        envelope_rule=ENVELOPE_RULES[0],
        envelope_provenance=envelope.provenance,
        envelope_scope=envelope.scope,
        ice6g_interpolated=interpolated,
        ice6g_time_note=time_note,
    )


# ---------------------------------------------------------------------------
# The sill test — a first-class output, not a by-product
# ---------------------------------------------------------------------------


def _verdict(margin: float, tol: float = AT_SILL_TOL_M) -> str:
    if margin > tol:
        return "spilling_west"
    if margin < -tol:
        return "endorheic"
    return "at_sill"


@dataclass(frozen=True)
class SillTest:
    """Did the Caspian clear the Manych sill at this epoch, in this scenario?

    Every field is either a measurement or an explicit frame statement.  The
    PRIMARY answer is :attr:`verdict` / :attr:`margin_m`, which compares two F1
    quantities.  The others exist so the reading can be audited, not averaged.
    """

    epoch_ka: float
    scenario: str
    #: F2 level as the curve reports it (present-day elevation of a dated deposit).
    level_f2_m: float
    #: F1 water surface: ``level_f2_m + dz_locus_m``.  Same number as the frame's
    #: render threshold — one quantity, not two.
    surface_f1_m: float
    dz_locus_m: float
    dz_sill_m: float
    #: ``Δz(sill) − Δz(locus)`` and the bracket the locus spread puts on it.
    differential_m: float
    differential_bracket_m: tuple[float, float]
    #: F1 crest at the measured +23.00 m point sill, and at the +25.00 m
    #: continuous-path controlling col.
    sill_f1_m: float
    sill_path_f1_m: float
    #: PRIMARY: surface_f1 − sill_f1, and its verdict.
    margin_m: float
    verdict: str
    #: Same test at the continuous-path crest.  Carried, not asserted.
    margin_path_m: float
    verdict_path: str
    #: Does the verdict survive the locus bracket?  False = the answer is inside the
    #: uncertainty and must not be quoted as a date.
    #:
    #: SCOPE, restated 2026-08-03 because the envelope fields below could be
    #: mistaken for part of it: this flag covers the ICE-6G/GIA locus spread ONLY —
    #: the level is held FIXED at the central line and only ``dz`` is moved to its p5
    #: and p95.  It says nothing whatever about the level's own uncertainty, which is
    #: what ``verdict_is_envelope_robust`` covers.  The computation is unchanged.
    verdict_is_robust: bool
    #: The F2 shortcut of contract §6: level_f2 vs the MODERN +23.00 m sill.
    #: Algebraically the primary test up to ``differential_m``.
    margin_modern_sill_m: float
    verdict_modern_sill: str
    #: DELIBERATELY WRONG, kept only so the two readings can be diffed: an F2 level
    #: against an F1 sill, which is what reading ``paleo_sill_m_asl`` off the CSV
    #: does.  It discards ~2.7 m of spill margin.  Used for nothing.
    margin_mixed_frame_m: float
    verdict_mixed_frame: str
    # ── THE TRIPLE: the same test at the ROW ENVELOPE's two edges ─────────────
    # WHY THIS EXISTS.  Until 2026-08-03 the spill was a property of the ADOPTED
    # CENTRAL curve alone, and the only robustness flag on it (verdict_is_robust)
    # moves the GIA locus while holding the level fixed.  The curve's own
    # unc_low/unc_high pair was never read, and it is not a detail: its LOWER edge is
    # endorheic at every one of the 17 epochs of 17-13 ka, in both scenarios and on
    # the whole 43-epoch grid.  So the honest caption is not "the Khvalynian spilled
    # into the Pontic" but "at the central estimate it spilled; at the lower edge of
    # the envelope the curve carries, it did not" — and the second half is what these
    # fields make checkable.
    #
    # WHAT THE EDGES ARE, RESTATED HERE BECAUSE THIS IS A PER-SCENARIO ROW.  They are
    # NOT this scenario's uncertainty.  The curve carries ONE (unc_low, unc_high)
    # pair per ROW; it brackets all six of its level columns, so it spans defensible
    # AND maximal at once.  Every field in this block is therefore IDENTICAL between
    # the two scenarios at all 43 epochs (verified), and ``envelope_scope`` /
    # ``envelope_is_scenario_invariant`` say so in the data rather than leaving a
    # reader of one row to infer that +17.1 m is "maximal's lower bound".
    #
    # Both edges are converted with the SAME dz_locus_m and the same rule as the
    # central (contract sec.8, edge for edge), so the envelope's width here is the
    # curve's width and not a GIA artefact.
    envelope_lo_f2_m: float
    envelope_hi_f2_m: float
    envelope_width_m: float
    surface_envelope_lo_f1_m: float
    surface_envelope_hi_f1_m: float
    margin_envelope_lo_m: float
    margin_envelope_hi_m: float
    verdict_envelope_lo: str
    verdict_envelope_hi: str
    #: The triple as one token: the shared verdict when all three agree, else
    #: ``"<lo>..<hi>"`` — which IS the caption, in one field.
    envelope_verdict: str
    #: Do the envelope's two edges agree with the central?  False = the spill state
    #: is a property of the adopted curve, not of the evidence as the curve carries
    #: it.
    verdict_is_envelope_robust: bool
    #: THE JOINT CORNER: the envelope's LOWER edge pushed by the locus's p95, i.e.
    #: the two uncertainty sources moved together, both in the direction most
    #: favourable to spilling within the pessimistic reading.
    margin_joint_corner_m: float
    #: Does ``verdict_envelope_lo`` survive that corner?  ONE-SIDED ON PURPOSE, and
    #: the side is the one the caption makes a claim about: "the lower edge never
    #: cleared the Manych".  Measured, this is False at exactly five epochs — 16.00 /
    #: 15.75 / 15.50 / 15.25 / 15.00 ka, corner margins -0.130 / -0.169 / -0.176 /
    #: -0.125 / -0.071 m — where the lower edge degrades from ``endorheic`` to
    #: ``at_sill``.  The upper edge has its own knife edge at 13.75-13.00 ka; it is
    #: reported in ``provenance`` rather than as a second boolean, because no caption
    #: rests on it.
    verdict_is_jointly_robust: bool
    # ── WHAT THE ENVELOPE IS, IN THE DATA, ON EVERY ROW ───────────────────────
    #: The measured scope sentence: an inter-school scenario envelope carried once
    #: per curve ROW, or (on a curve where the pair does not bracket every column)
    #: exactly which columns it does bracket.
    envelope_scope: str
    #: Measured: is the whole triple above the same for every scenario at this epoch?
    #: True on this curve at every row, and that is a FACT ABOUT THE FILE, not a
    #: coincidence — there is one pair per row.
    envelope_is_scenario_invariant: bool
    #: Measured: how many of the curve's level columns the envelope brackets here,
    #: and how many there are.
    envelope_brackets_n_columns: int
    envelope_n_level_columns: int
    # ── IS THE UPPER EDGE INDEPENDENT EVIDENCE HERE? ──────────────────────────
    #: True where ``unc_high`` stands AT a measured Manych crest, i.e. the edge was
    #: derived from "water was at the sill" and testing it against the sill is close
    #: to circular.  Measured per row by :func:`envelope_hi_circularity`; on the
    #: baked grid it is True at 18.0 and 17.5 ka, which are exactly the two epochs
    #: where ``verdict_envelope_hi`` spills outside the 17-13 ka window.
    envelope_hi_is_circular: bool
    #: The reason, in full, including the numbers it was measured from.
    envelope_hi_circularity: str
    envelope_rule: str
    envelope_provenance: str
    #: Measured: does THIS row's ``note`` record where its edges came from?  The
    #: earlier blanket "not recorded in the curve" was false on the 15.0 and 14.0 ka
    #: rows, which name ``unc_low`` and its source outright.
    envelope_provenance_in_note: bool
    gia_time_interpolation: str
    provenance: str


class MissingLocusBracketError(ValueError):
    """The locus bracket is absent, so no robustness flag can be computed here.

    ``verdict_is_robust`` and ``verdict_is_jointly_robust`` are both built on
    ``BasinThreshold.bracket_m``.  The pre-existing expression for them ends in the
    fallback ``... if threshold.bracket_m else (surface, surface)``, and with no
    bracket that fallback collapses the p5/p95 offsets to 0.0: ``verdict_is_robust``
    becomes unconditionally ``True`` and ``verdict_is_jointly_robust`` degenerates
    into a copy of ``verdict_envelope_lo``.  Two flags whose entire job is to say
    "this answer is inside the uncertainty" would then say "certain", silently.

    Every rule ``caspian_threshold`` can return sets ``bracket_m`` — ``marker`` from
    the 3x3 cell block, ``locus`` from the p5-p95 of the band — so no shipped row
    reaches this.  It is raised rather than left as a fallback precisely because it
    is unreachable today: the fallback's failure mode is invisible, and an
    unreachable refusal is cheaper than a plausible-looking ``True``.

    The fallback expression itself is deliberately NOT edited.  ``verdict_is_robust``
    is pinned byte-for-byte by ``test_verdict_is_robust_is_unchanged``, so the guard
    goes in front of it and the arithmetic stays exactly as it was.
    """


def _envelope_verdict(low: str, central: str, high: str) -> str:
    """The triple as one token: the shared verdict, or ``"<low>..<high>"``.

    A single string rather than a boolean because the interesting case is precisely
    the one a boolean throws away — at 16.5 ka defensible the envelope runs
    ``endorheic..spilling_west``, and that IS the finding.
    """
    if low == central == high:
        return low
    return f"{low}..{high}"


def sill_test(df: pd.DataFrame, epoch: float, scenario: str = "defensible") -> SillTest:
    """Run the sill test for one epoch and scenario.  See :class:`SillTest`."""
    threshold = caspian_threshold(df, epoch, scenario)
    dz_sill, _block = sill_solid_displacement(epoch)
    differential, diff_bracket = sill_marker_differential(epoch)

    surface = threshold.threshold_m
    sill_f1 = MANYCH_SILL_M + dz_sill
    sill_path_f1 = MANYCH_SILL_PATH_M + dz_sill
    margin = surface - sill_f1
    margin_path = surface - sill_path_f1

    # THE GUARD IN FRONT OF THE FALLBACK, not instead of it.  See
    # MissingLocusBracketError: with no bracket both robustness flags degenerate to
    # "certain" without saying so.  The two lines below are unchanged and pinned.
    if not threshold.bracket_m:
        raise MissingLocusBracketError(
            f"{float(epoch):g} ka / {scenario}: the threshold carries no locus "
            f"bracket (rule={threshold.rule!r}, n_cells={threshold.n_cells!r}), so "
            "verdict_is_robust and verdict_is_jointly_robust would both be computed "
            "against a ZERO-WIDTH GIA spread and come out unconditionally True.  A "
            "robustness flag that cannot fail is not a flag; the frame is refused "
            "instead.  No rule caspian_threshold implements can produce this — "
            "marker brings a 3x3 block, locus brings p5-p95 — so reaching it means a "
            "new rule was added without a bracket."
        )
    lo_b, hi_b = threshold.bracket_m if threshold.bracket_m else (surface, surface)
    robust = _verdict(lo_b - sill_f1) == _verdict(hi_b - sill_f1) == _verdict(margin)

    margin_modern = threshold.level_m - MANYCH_SILL_M
    margin_mixed = threshold.level_m - sill_f1

    # ── the ROW ENVELOPE's two edges, through the SAME test ──────────────────
    # caspian_threshold has already converted both edges with the same
    # conversion_m and the same rule as the central and refused the row outright if
    # the envelope was unusable, so these four are never None here and nothing below
    # needs a fallback.  The locus half-widths are re-derived from the central's own
    # bracket so the corner cannot be built on a different p5/p95 than the one
    # verdict_is_robust used.
    surface_low = float(threshold.threshold_envelope_lo_m)
    surface_high = float(threshold.threshold_envelope_hi_m)
    margin_low = surface_low - sill_f1
    margin_high = surface_high - sill_f1
    verdict_low = _verdict(margin_low)
    verdict_high = _verdict(margin_high)
    locus_p95_offset = hi_b - surface  # = p95 - median, >= 0
    margin_joint = margin_low + locus_p95_offset
    jointly_robust = _verdict(margin_low + (lo_b - surface)) == _verdict(
        margin_joint
    ) == verdict_low

    # ── what the envelope IS, measured at this row and shipped beside the numbers ─
    circular, circular_why = envelope_hi_circularity(df, epoch)
    idx = _curve_row_index(df, epoch)
    all_columns = level_columns(df)
    bracketed, scope = envelope_scope(
        df, idx, float(threshold.envelope_lo_m), float(threshold.envelope_hi_m)
    )
    # Scenario-invariance is a property of the FILE, so it is read off the file: one
    # (unc_low, unc_high) pair per row means the pair cannot vary with the scenario,
    # and the check that makes it a real measurement is that the pair brackets every
    # scenario column, i.e. no scenario is left outside the envelope it is reported
    # against.  On a curve where that fails (the Pontic one, at four rows) this is
    # False and the scope sentence names the columns that fall outside.
    scenario_invariant = len(bracketed) == len(all_columns) and len(all_columns) > 1
    _in_note, _prov = envelope_note_provenance(df, idx)

    return SillTest(
        epoch_ka=float(epoch),
        scenario=scenario,
        level_f2_m=threshold.level_m,
        surface_f1_m=surface,
        dz_locus_m=float(threshold.conversion_m),
        dz_sill_m=dz_sill,
        differential_m=differential,
        differential_bracket_m=diff_bracket,
        sill_f1_m=sill_f1,
        sill_path_f1_m=sill_path_f1,
        margin_m=margin,
        verdict=_verdict(margin),
        margin_path_m=margin_path,
        verdict_path=_verdict(margin_path),
        verdict_is_robust=bool(robust),
        margin_modern_sill_m=margin_modern,
        verdict_modern_sill=_verdict(margin_modern),
        margin_mixed_frame_m=margin_mixed,
        verdict_mixed_frame=_verdict(margin_mixed),
        envelope_lo_f2_m=float(threshold.envelope_lo_m),
        envelope_hi_f2_m=float(threshold.envelope_hi_m),
        envelope_width_m=float(threshold.envelope_hi_m)
        - float(threshold.envelope_lo_m),
        surface_envelope_lo_f1_m=surface_low,
        surface_envelope_hi_f1_m=surface_high,
        margin_envelope_lo_m=margin_low,
        margin_envelope_hi_m=margin_high,
        verdict_envelope_lo=verdict_low,
        verdict_envelope_hi=verdict_high,
        envelope_verdict=_envelope_verdict(verdict_low, _verdict(margin), verdict_high),
        verdict_is_envelope_robust=bool(
            verdict_low == _verdict(margin) == verdict_high
        ),
        margin_joint_corner_m=margin_joint,
        verdict_is_jointly_robust=bool(jointly_robust),
        envelope_scope=scope,
        envelope_is_scenario_invariant=bool(scenario_invariant),
        envelope_brackets_n_columns=len(bracketed),
        envelope_n_level_columns=len(all_columns),
        envelope_hi_is_circular=bool(circular),
        envelope_hi_circularity=circular_why,
        envelope_rule=threshold.envelope_rule,
        envelope_provenance=threshold.envelope_provenance,
        envelope_provenance_in_note=bool(_in_note),
        gia_time_interpolation=_time_interpolation_note(epoch),
        provenance=(
            f"SILL TEST {float(epoch):g} ka / {scenario}: surface_F1 "
            f"{surface:+.2f} m [{lo_b:+.2f}..{hi_b:+.2f}] vs crest_F1 "
            f"{sill_f1:+.2f} m (= {MANYCH_SILL_M:+.2f} modern + "
            f"{dz_sill:+.3f} GIA) -> margin {margin:+.2f} m -> {_verdict(margin)}"
            f"{'' if robust else ' (NOT ROBUST across the locus bracket)'}; "
            f"continuous-path crest {sill_path_f1:+.2f} m -> "
            f"{_verdict(margin_path)}; differential dz(sill)-dz(locus) "
            f"{differential:+.3f} m [{diff_bracket[0]:+.3f}..{diff_bracket[1]:+.3f}]"
            f"; ENVELOPE {threshold.envelope_rule} "
            f"[{float(threshold.envelope_lo_m):+.2f}.."
            f"{float(threshold.envelope_hi_m):+.2f} m F2] -> surface "
            f"[{surface_low:+.2f}..{surface_high:+.2f} m F1] -> margins "
            f"[{margin_low:+.2f}..{margin_high:+.2f} m] -> "
            f"{_envelope_verdict(verdict_low, _verdict(margin), verdict_high)}"
            f"{'' if jointly_robust else f' (the LOWER edge is not robust: at the locus p95 corner it reads {margin_joint:+.3f} m -> {_verdict(margin_joint)})'}"
            f"; the upper edge at the locus p5 corner reads "
            f"{margin_high + (lo_b - surface):+.3f} m -> "
            f"{_verdict(margin_high + (lo_b - surface))}"
            f"; SCOPE {scope}"
            f"; UPPER-EDGE CIRCULARITY {circular_why}"
            f"; envelope provenance {threshold.envelope_provenance}"
        ),
    )


def sill_report(
    df: pd.DataFrame,
    epochs: Sequence[float] = EPOCHS_KA,
    scenarios: Sequence[str] = SCENARIOS,
) -> pd.DataFrame:
    """The whole sill test, every epoch x scenario, as a table.

    This is THE output of this module.  It needs no raster, costs one ICE-6G read
    per bracketing epoch, and answers the question the render exists to ask.
    """
    rows = []
    for epoch in epochs:
        for scenario in scenarios:
            record = asdict(sill_test(df, epoch, scenario))
            lo, hi = record.pop("differential_bracket_m")
            record["differential_bracket_lo_m"] = lo
            record["differential_bracket_hi_m"] = hi
            rows.append(record)
    return pd.DataFrame(rows)


def spill_window_ka(
    report: pd.DataFrame,
    scenario: str,
    verdict_column: str = "verdict",
    *,
    drop_circular: bool = False,
) -> tuple[float, float] | None:
    """``(opens_ka, closes_ka)`` — oldest and youngest ``spilling_west`` epoch.

    ``None`` when the scenario never spills.  Ages, so *opens* is the LARGER number.
    Reported off :attr:`SillTest.verdict` by default, i.e. the F1-vs-F1 test at the
    adopted central line.

    *verdict_column* selects which of the triple is being asked about:
    ``"verdict_envelope_lo"`` and ``"verdict_envelope_hi"`` give the window at the
    row envelope's two edges.  Measured on the shipped curve,
    ``verdict_envelope_lo`` returns ``None`` for both scenarios over the whole
    43-epoch grid — the lower edge of the envelope as the curve carries it never
    clears the Manych at any epoch this bake covers.  That is the second half of the
    caption, and it is a query rather than a constant so it cannot drift from the
    report.

    *drop_circular* excludes rows flagged ``envelope_hi_is_circular``.  It exists
    because those rows' upper edge was DERIVED from "water was at the sill" (see
    :func:`envelope_hi_circularity`), so an upper-edge window that includes them is
    partly an artefact of the question.  It is an explicit argument rather than a
    default because dropping rows silently would be its own substitution: the caller
    has to say which window it is asking for, and both windows are reported.
    """
    known = [c for c in report.columns if str(c).startswith("verdict")]
    if verdict_column not in report.columns:
        raise ValueError(
            f"unknown verdict column {verdict_column!r}; the report carries {known}"
        )
    # A column that is not a verdict would compare equal to nothing and this would
    # return None — i.e. it would answer "never spills" to a question that was
    # never asked.  That is the failure mode this guard exists for.
    values = set(report[verdict_column].astype(str))
    if not values <= set(SPILL_STATES):
        raise ValueError(
            f"unknown verdict column {verdict_column!r}: it holds "
            f"{sorted(values)[:4]}, which is not the {SPILL_STATES} vocabulary, so a "
            "'never spills' answer off it would be an artefact.  Known verdict "
            f"columns: {known}"
        )
    rows = report
    if drop_circular:
        if "envelope_hi_is_circular" not in report.columns:
            raise ValueError(
                "drop_circular=True needs the 'envelope_hi_is_circular' column, and "
                "this report has none — it is a pre-2026-08-03 report and cannot "
                "answer which of its upper-edge spills are independent evidence"
            )
        rows = rows[~rows["envelope_hi_is_circular"].astype(bool)]
    hit = rows[
        (rows["scenario"] == scenario) & (rows[verdict_column] == "spilling_west")
    ]
    if hit.empty:
        return None
    return float(hit["epoch_ka"].max()), float(hit["epoch_ka"].min())


#: The uncertainty the curve's own header attaches to ``paleo_sill_m_asl``:
#: "+23.00 m + local GIA anomaly ... Uncertainty ~+/-2 m".  Quoted here because the
#: caption compares the lower bound's shortfall against it, and a caption that
#: compares a number against an unnamed bar is not auditable.
MANYCH_SILL_GIA_BAR_M = 2.0


def sill_caption(
    report: pd.DataFrame,
    window_ka: tuple[float, float] = FINE_WINDOW_KA,
) -> str:
    """The one sentence this report licenses, with BOTH envelope edges in it.

    Computed from the report rather than written down, so it cannot drift from the
    numbers: the central windows come from :func:`spill_window_ka`, the lower-edge
    claim from the ``verdict_envelope_lo`` column, the closest approach from the
    smallest ``|margin_envelope_lo_m|`` inside *window_ka*, and the upper-edge
    window from ``verdict_envelope_hi`` with the circular rows dropped.

    FOUR WORDINGS THAT ARE REFUSED, and what each would misstate:

    * *"the published band"* — there is no published per-scenario band here to
      quote.  ``unc_low`` / ``unc_high`` are carried ONCE PER CURVE ROW and bracket
      every level column of the file, so they are an inter-school scenario
      ENVELOPE.  The caption names it that way and says **"as carried in the
      curve"**.
    * *"the band around the maximal scenario"* (or the defensible one) — the same
      error stated per row.  The edges are identical for both scenarios; the caption
      says so instead of letting a per-scenario report imply otherwise.
    * *"the upper edge shows the spill starting by 18 ka"* — at the rows flagged
      ``envelope_hi_is_circular`` the upper edge IS the Manych crest, carried from a
      spillway reading in which water stands on the sill by construction.  The
      caption reports the upper-edge window with those rows dropped and then names
      them and refuses them explicitly, rather than quietly excluding them.
    * *"a comfortable margin"* — the lower edge misses by as little as 0.73 m, well
      inside the sill's own +/-2 m GIA bar, so the shortfall is quoted with its epoch
      and compared against that bar in the same breath.

    Raises
    ------
    ValueError
        The report does not cover *window_ka*, lacks a field the sentence asserts
        from, or its ``verdict_envelope_lo`` column contains a spill.  Each would
        make the sentence a claim about epochs that were not tested, or a false one.
    """
    for column in (
        "verdict",
        "verdict_envelope_lo",
        "verdict_envelope_hi",
        "margin_envelope_lo_m",
        "envelope_lo_f2_m",
        "envelope_hi_f2_m",
        "envelope_hi_is_circular",
        "envelope_brackets_n_columns",
        "envelope_n_level_columns",
        "epoch_ka",
        "scenario",
    ):
        if column not in report.columns:
            raise ValueError(
                f"report is missing {column!r} — this caption states both edges of "
                "the envelope, names what the envelope is, and refuses the circular "
                "rows; it cannot be produced from a report that does not carry them"
            )
    lo_ka, hi_ka = float(min(window_ka)), float(max(window_ka))
    inside = report[(report["epoch_ka"] >= lo_ka) & (report["epoch_ka"] <= hi_ka)]
    if inside.empty:
        raise ValueError(
            f"the report covers no epoch of {hi_ka:g}-{lo_ka:g} ka, so nothing can "
            "be said about the envelope's lower edge over that interval"
        )
    spilling_low = sorted(
        set(inside.loc[inside["verdict_envelope_lo"] == "spilling_west", "epoch_ka"])
    )
    if spilling_low:
        raise ValueError(
            "the envelope's lower edge spills at "
            f"{spilling_low} — the caption below asserts it never does, and it is "
            "not printed against a report that contradicts it"
        )
    windows = {}
    for scenario in sorted(set(report["scenario"])):
        windows[scenario] = spill_window_ka(report, scenario)
    central = ", ".join(
        f"{s} {w[0]:g}->{w[1]:g} ka" if w else f"{s} never clears the sill"
        for s, w in sorted(windows.items())
    )

    # ── WHOSE lower edge is closest?  Measured, not assumed. ──────────────────
    # The old line took idxmin over ALL rows and then printed an epoch with no
    # scenario beside it, immediately after quoting per-scenario windows — a reader
    # could only read the epoch as belonging to whichever scenario was named last.
    # It is legitimate to print one epoch here ONLY if the lower edge is the same
    # for every scenario, which is a property of this curve and therefore a thing to
    # CHECK rather than to rely on.
    by_epoch = inside.groupby("epoch_ka")["margin_envelope_lo_m"]
    spread = float((by_epoch.max() - by_epoch.min()).abs().max())
    invariant = spread <= 1e-9
    if invariant:
        margins = by_epoch.min()
        closest_epoch = float(margins.abs().idxmin())
        shortfall = abs(float(margins.loc[closest_epoch]))
        whose = (
            f"the same for every scenario in this report "
            f"({'/'.join(sorted(set(report['scenario'])))}); "
            f"max spread across scenarios {spread:.2e} m"
        )
    else:
        row = inside.loc[inside["margin_envelope_lo_m"].abs().idxmin()]
        closest_epoch = float(row["epoch_ka"])
        shortfall = abs(float(row["margin_envelope_lo_m"]))
        whose = (
            f"on the {row['scenario']} scenario — the lower edge is NOT the same "
            f"across scenarios in this report (spread up to {spread:.3f} m)"
        )
    edge = inside[inside["epoch_ka"] == closest_epoch].iloc[0]

    # The epochs the report ACTUALLY covers inside the window, not the window's
    # nominal ends: a report of two epochs must not license a sentence about
    # seventeen.  With the full grid these are 17 and 13 ka and the count is 17.
    covered = sorted({float(e) for e in inside["epoch_ka"]})
    # "inside the bar" is a comparison, so it is made rather than asserted; a
    # shortfall wider than the bar would make the reassuring wording false.
    against_bar = (
        f"inside the sill's own +/-{MANYCH_SILL_GIA_BAR_M:g} m GIA bar"
        if shortfall < MANYCH_SILL_GIA_BAR_M
        else f"OUTSIDE the sill's own +/-{MANYCH_SILL_GIA_BAR_M:g} m GIA bar"
    )

    # ── the upper edge, with the circular rows named and refused ──────────────
    circular = sorted(
        {
            float(e)
            for e in report.loc[
                report["envelope_hi_is_circular"].astype(bool), "epoch_ka"
            ]
        }
    )
    upper = {
        s: spill_window_ka(report, s, "verdict_envelope_hi", drop_circular=True)
        for s in sorted(set(report["scenario"]))
    }
    upper_text = ", ".join(
        f"{s} {w[0]:g}->{w[1]:g} ka" if w else f"{s} never clears the sill"
        for s, w in sorted(upper.items())
    )
    if circular:
        refusal = (
            f"  The upper edge ALSO reads spilling_west at "
            f"{'/'.join(f'{e:g}' for e in circular)} ka and that is NOT independent "
            f"evidence and is not cited as any: at those rows unc_high stands on the "
            f"Manych crest itself, carried from the Semikolennykh et al. 2022 "
            f"Manych-spillway OSL reading in which the water is AT the sill by "
            f"construction, so testing it against the sill is circular."
        )
    else:
        refusal = (
            "  No row of this report has an upper edge standing on the Manych crest, "
            "so none of the upper-edge spills above is circular."
        )

    n_bracketed = int(edge["envelope_brackets_n_columns"])
    n_columns = int(edge["envelope_n_level_columns"])

    # PROVENANCE, STATED AS WHAT WAS MEASURED.  The literal this used to print,
    # "UNKNOWN:not_recorded_in_the_curve", asserted an absence the curve contradicts
    # on the rows the headline number rests on: the 15.0 ka note records HESS2024's
    # "+22 to +35 m a.s.l." band, names +35 as its top edge and says unc_low carries
    # the +22 m bottom edge from Makshaev & Tkach; the 14.0 ka note records a carried
    # conflict.  So the sentence says the two true things — no machine-readable
    # column, and how many rows of THIS report record their edges in prose — and
    # leaves the citation where it is instead of manufacturing one.
    if "envelope_provenance_in_note" in report.columns:
        noted = sorted(
            {
                float(e)
                for e in report.loc[
                    report["envelope_provenance_in_note"].astype(bool), "epoch_ka"
                ]
            }
        )
        n_rows = len(set(report["epoch_ka"]))
        provenance_text = (
            f"{ENVELOPE_PROVENANCE_NO_COLUMN}; but measured, {len(noted)} of "
            f"{n_rows} epochs in this report record their edges' sources in prose in "
            f"their own note column"
            + (f" ({'/'.join(f'{e:g}' for e in noted)} ka)" if noted else "")
            + ", so 'not recorded in the curve' would be false - read the note"
        )
    else:
        provenance_text = ENVELOPE_PROVENANCE_NO_COLUMN
    return (
        f"at the central estimate the Caspian spilled west into the Pontic "
        f"({central}).  The curve's uncertainty is NOT a per-scenario band: "
        f"unc_low/unc_high is ONE pair carried per curve ROW, measured here to "
        f"bracket {n_bracketed} of {n_columns} level columns at once - an "
        f"inter-school scenario envelope as carried in the curve, {whose}.  At its "
        f"LOWER edge ({float(edge['envelope_lo_f2_m']):+.1f} m F2 at "
        f"{closest_epoch:g} ka) it never cleared the Manych at any of the "
        f"{len(covered)} epochs of {max(covered):g}-{min(covered):g} ka - but by as "
        f"little as {shortfall:.2f} m at {closest_epoch:g} ka, {against_bar}.  At "
        f"its UPPER edge ({float(edge['envelope_hi_f2_m']):+.1f} m F2 at the same "
        f"epoch) it clears the crest over {upper_text}, counting only rows where "
        f"that edge is independent of the sill.{refusal}  Envelope provenance: "
        f"{provenance_text}."
    )


# ---------------------------------------------------------------------------
# The working raster
# ---------------------------------------------------------------------------


def manych_sill_elevation_m(dem_path: Path | None = None) -> float:
    """Modern elevation (m) at the Manych saddle, read from the working raster.

    A windowed 1x1 read — the raster is 7800x4320 and is never loaded whole.  The
    value it returns is the provenance of :data:`MANYCH_SILL_M`, and its SIGN is the
    whole event: **positive** means the crest stands above present sea level, which
    is why an endorheic Caspian at -28 m does not spill today.  A negative reading
    would mean the sill is a submarine channel and this render has no subject.
    """
    path = Path(dem_path) if dem_path is not None else KHV_DEM
    with rasterio.open(path) as ds:
        row, col = ds.index(MANYCH_SILL_LON, MANYCH_SILL_LAT)
        value = ds.read(1, window=Window(col, row, 1, 1)).astype(np.float64)
        if ds.nodata is not None and np.isclose(value, ds.nodata).all():
            raise ValueError(
                f"the working raster has NODATA at the Manych saddle "
                f"{MANYCH_SILL_LON:.3f}E/{MANYCH_SILL_LAT:.3f}N — the sill's modern "
                "crest cannot be established from it"
            )
    return float(value[0, 0])


def _tile_coordinates(ds) -> tuple[np.ndarray, np.ndarray]:
    """Column-centre lons (ascending) and row-centre lats (descending) of *ds*."""
    cols = np.arange(ds.width)
    rows = np.arange(ds.height)
    tile_lons = (ds.transform * (cols + 0.5, np.zeros_like(cols) + 0.5))[0]
    tile_lats = (ds.transform * (np.zeros_like(rows) + 0.5, rows + 0.5))[1]
    return np.asarray(tile_lats, dtype=float), np.asarray(tile_lons, dtype=float)


# ===========================================================================
# THE MESHER TAG SET — what a 4D reader needs and the bake used to throw away
# ===========================================================================
#
# The rasters are the product; the PNGs were validation.  A mesher that has only a
# ``khv_<epoch>ka.tif`` must be able to reconstruct the epoch from it, and until now
# it could not, for two reasons that are both defects of the TAGS and not of the
# pixels:
#
# 1. THE NODATA DECLARATION WAS IN THE WRONG UNIT AND SAID SO NOWHERE.
#    ``gia_unknown_ice6g_cells`` reads 101 at 18 ka.  The file holds 305 600 nodata
#    cells at 18 ka.  Both numbers are true and they count DIFFERENT THINGS — 101
#    ICE-6G_C 10-arcmin MODEL cells whose ``Topo`` is the ice surface, versus 305 600
#    15-arcsec RASTER cells carrying the sentinel — and no tag named either unit.  A
#    reader taking the first for the second is wrong by three orders of magnitude.
#    :func:`nodata_tags` names both units IN THE TAG NAMES, carries the true raster
#    count MEASURED FROM THE FILE (never derived from the model grid: the ice-aware
#    smoothing rim widens the hole, so the nominal 1600:1 area ratio does not hold),
#    and adds the hole's bounding box and its contact with the raster edge — because
#    a hole that reaches the edge cannot be filled by interpolation at all.
#
# 2. THE WATER SURFACES WERE NOT MACHINE-READABLE.  The Caspian's F1 surface existed
#    only inside a 1 299-character prose ``threshold_*`` tag; the Mediterranean was
#    implicit as the negation of ``gia_sea_surface_offset_m`` with nothing saying so;
#    the Pontic was ABSENT.  :func:`water_level_tags` writes one JSON tag per basin
#    carrying the value, its unit, its frame, and the RULE that produced it.
#
# NOTHING HERE COMPUTES AN ELEVATION.  This is an additive tag pass: the geometry is
# finished and every function below is either a measurement of pixels that already
# exist or a re-read of a curve.  :func:`retag_paleo_dem` sha256s the pixel array
# before and after and refuses to return if they differ.


#: Schema token stamped into every raster this pass touches.  Bump it only when the
#: MEANING of a tag changes; a reader keys on it to know which contract it is reading.
MESHER_TAG_SCHEMA = "khvalynian-mesher-tags/1"

#: The code path of record, carried in the data as the provenance rules require.
MESHER_TAG_CODE_PATH = "scripts/bake_khvalynian.py::mesher_tags"

#: The three water sheets this scene contains.  ``mediterranean`` is the world ocean
#: reaching the frame's SW corner through the Aegean; it is named separately from
#: ``pontic`` because after reconnection the two share a surface and before it they
#: do not, and a mesher must be able to tell which epoch it is looking at.
BASIN_SHEETS = ("caspian", "mediterranean", "pontic")

#: Sidecars of the pre-pass tag state, one JSON per raster.  Written ONCE (a re-run
#: must not overwrite the true original with a copy of its own output), so the
#: shipped tag set stays recoverable after an in-place pass.
RETAG_SIDECAR_DIR = OUT_BASE / "_retag"

#: Per-epoch manifest of the additive pass: pixel sha256 before and after, the
#: measured counts, and whether the tags moved.  This is the artefact that makes the
#: "no pixel was touched" claim checkable by somebody who did not run the pass.
RETAG_MANIFEST_CSV = OUT_BASE / "khvalynian_mesher_retag.csv"


class PixelDataChangedError(RuntimeError):
    """The pixel array is not what it was before the tags were written.

    This is the whole safety of the additive pass, so it is an exception and not a
    log line.  ``rasterio.open(..., "r+")`` + ``update_tags`` is expected to rewrite
    only the TIFF directory, but "expected" is not "verified": the pass hashes the
    band before and after and raises this if the two differ.  Recovery is to restore
    the raster from its source of record and re-bake it, NOT to re-run the pass.
    """


def _ascii_tag(key: str, value: str) -> str:
    """Return *value*, refusing anything GDAL's metadata channel would corrupt.

    Measured, not assumed: a U+2014 em dash written through ``update_tags`` reads
    back as U+FFFD.  The shipped ``threshold_*`` tags already carry one such
    replacement character, mid-sentence, where an em dash was intended.  A tag is
    data, and silently mangled data is the defect class this whole phase exists to
    remove, so every value this module writes is ASCII or it is not written.
    """
    if not value.isascii():
        bad = sorted({c for c in value if not c.isascii()})
        raise ValueError(
            f"tag {key!r} carries non-ASCII {bad!r}: GDAL's GTiff metadata channel "
            "round-trips it as U+FFFD (measured), which would ship a corrupted "
            "character inside published provenance.  Write it in ASCII."
        )
    return value


def _json_tag(payload: Mapping[str, object]) -> str:
    """Deterministic, ASCII-safe JSON for a tag value.

    ``sort_keys`` and ``ensure_ascii`` are both load-bearing.  The first makes the
    additive pass idempotent: a re-run must produce a byte-identical string or
    "unchanged" cannot be asserted.  The second is how non-ASCII text is carried
    WITHOUT being mangled and WITHOUT being altered — a borrowed sentence (an
    upstream exception message, say: ``CurveAgeGapError``'s own text contains an em
    dash) becomes a ``\\uXXXX`` escape on the way in and the exact original character
    on the way out of ``json.loads``.  Escaping is lossless; letting GDAL see the raw
    character is not, and substituting a hyphen for the dash would be editing
    somebody else's quoted words.
    """
    return json.dumps(dict(payload), sort_keys=True, ensure_ascii=True)


# ---------------------------------------------------------------------------
# 1. The nodata declaration, in units that say what they count
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodataCensus:
    """What a raster's nodata actually is, measured on that raster's own cells.

    Every field is a count or an extent OF THIS FILE.  Nothing here is derived from
    the ICE-6G model grid — that derivation is exactly the mistake the shipped
    ``gia_unknown_ice6g_cells`` tag invites, and it is wrong by a factor of ~30
    because the ice-aware smoothing rim (``SMOOTHING_RIM_CELLS``) widens the hole
    well beyond the nominal one-model-cell-to-1600-raster-cells footprint.

    ``row_min`` .. ``lat_max`` are ``None`` when the file has NO nodata cell.  They
    are not zeros: a zero row index is a real place (the top row of the frame) and a
    reader cannot tell an absent hole from a hole in the corner if the absence is
    spelled ``0``.  Same reasoning as :data:`KHV_GIA_NAN_POLICY` having no ``"zero"``.
    """

    height: int
    width: int
    #: Cell size in the raster's own CRS units, from its transform.  Carried because
    #: the units note has to name the raster's resolution and MUST NOT hardcode it:
    #: "15 arcsec" is true of the shipped frames and would be a lie the moment one is
    #: resampled, which is precisely how the tag this pass exists to fix went wrong.
    cell_lon: float
    cell_lat: float
    nodata_value: float
    cells: int
    row_min: int | None
    row_max: int | None
    col_min: int | None
    col_max: int | None
    lon_min: float | None
    lon_max: float | None
    lat_min: float | None
    lat_max: float | None
    edge_north: int
    edge_south: int
    edge_west: int
    edge_east: int

    @property
    def cells_total(self) -> int:
        return int(self.height) * int(self.width)

    @property
    def fraction(self) -> float:
        return float(self.cells) / float(self.cells_total)

    @property
    def touches_edge(self) -> bool | None:
        """``None`` = there is no hole, so the question does not apply."""
        if self.cells == 0:
            return None
        return bool(
            self.edge_north or self.edge_south or self.edge_west or self.edge_east
        )

    @property
    def edges_touched(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, count in (
                ("north", self.edge_north),
                ("south", self.edge_south),
                ("west", self.edge_west),
                ("east", self.edge_east),
            )
            if count
        )


def nodata_census_from_mask(mask: np.ndarray, transform, nodata_value: float) -> NodataCensus:
    """Census of a boolean nodata *mask* on the grid described by *transform*.

    Split out from :func:`nodata_census` so the arithmetic — the bounding box, the
    cell-EDGE longitudes, the four border counts — is testable on a 4x4 array
    instead of on 33.7 million cells.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"nodata mask must be 2-D, got shape {mask.shape}")
    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    cell_lon, cell_lat = abs(float(transform.a)), abs(float(transform.e))
    cells = int(np.count_nonzero(mask))
    if cells == 0:
        return NodataCensus(
            height=height,
            width=width,
            cell_lon=cell_lon,
            cell_lat=cell_lat,
            nodata_value=float(nodata_value),
            cells=0,
            row_min=None,
            row_max=None,
            col_min=None,
            col_max=None,
            lon_min=None,
            lon_max=None,
            lat_min=None,
            lat_max=None,
            edge_north=0,
            edge_south=0,
            edge_west=0,
            edge_east=0,
        )
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    row_min, row_max = int(rows[0]), int(rows[-1])
    col_min, col_max = int(cols[0]), int(cols[-1])
    # Cell EDGES, not centres: (col, row) indexes the upper-left corner of a cell, so
    # the box that CONTAINS every nodata cell runs from (col_min, row_min) to
    # (col_max + 1, row_max + 1).  A centre-based box would understate the hole by
    # half a cell on each side, which is the sort of half-truth this pass exists to
    # stop shipping.
    lon_min, lat_max = transform * (col_min, row_min)
    lon_max, lat_min = transform * (col_max + 1, row_max + 1)
    return NodataCensus(
        height=height,
        width=width,
        cell_lon=cell_lon,
        cell_lat=cell_lat,
        nodata_value=float(nodata_value),
        cells=cells,
        row_min=row_min,
        row_max=row_max,
        col_min=col_min,
        col_max=col_max,
        lon_min=float(lon_min),
        lon_max=float(lon_max),
        lat_min=float(lat_min),
        lat_max=float(lat_max),
        edge_north=int(np.count_nonzero(mask[0, :])),
        edge_south=int(np.count_nonzero(mask[-1, :])),
        edge_west=int(np.count_nonzero(mask[:, 0])),
        edge_east=int(np.count_nonzero(mask[:, -1])),
    )


def nodata_census(path: Path) -> tuple[NodataCensus, str]:
    """``(census, sha256 of the band-1 array)`` for a raster on disk.

    One read of the whole band.  The hash is taken on the same array the census is
    taken on, so the pair is self-consistent by construction and the caller cannot
    accidentally certify a census against a different read.
    """
    array, transform, nodata, _tags = read_band(Path(path))
    mask = array == np.asarray(nodata, dtype=array.dtype)
    return (
        nodata_census_from_mask(mask, transform, nodata),
        hashlib.sha256(array.tobytes()).hexdigest(),
    )


def gia_unknown_model_cells(epoch: float) -> tuple[int, int]:
    """``(unknown, total)`` ICE-6G MODEL cells over the scene window at *epoch*.

    Recomputed from the distribution rather than read back out of the raster tag, so
    the pass can CHECK the shipped ``gia_unknown_ice6g_cells`` instead of copying it
    forward.  This is the same quantity :func:`build_paleo_dem` counts, on the same
    interpolated field, so an interpolated epoch is counted on the field it actually
    used and not on either endpoint.
    """
    dz, _lats, _lons = solid_displacement_field(epoch)
    return int(np.count_nonzero(~np.isfinite(dz))), int(dz.size)


def _model_cell_deg(epoch: float) -> tuple[float, float]:
    """``(d_lat, d_lon)`` of the ICE-6G grid, read off its own coordinate vectors."""
    win = _open_khv_window(_ice6g_bracket(epoch)[0])
    return (
        float(np.abs(np.diff(win.lats)).mean()),
        float(np.abs(np.diff(win.lons)).mean()),
    )


def _model_cell_arcmin(epoch: float) -> float:
    """The ICE-6G latitude cell size in arcmin.  MEASURED, never written down."""
    return _model_cell_deg(epoch)[0] * 60.0


def _model_grid_note(epoch: float) -> str:
    """One line naming the ICE-6G grid the model-cell count is taken on."""
    win = _open_khv_window(_ice6g_bracket(epoch)[0])
    d_lat, d_lon = _model_cell_deg(epoch)
    return (
        f"ICE-6G_C VM5a, cell {d_lat * 60.0:.1f}x{d_lon * 60.0:.1f} arcmin, window "
        f"lat {float(win.lats.min()):.6f}..{float(win.lats.max()):.6f} N / lon "
        f"{float(win.lons.min()):.6f}..{float(win.lons.max()):.6f} E "
        f"(scene bbox + BUF {BUF:g} deg), {win.topo_t.shape[0]}x{win.topo_t.shape[1]} "
        f"= {win.topo_t.size} cells"
    )


def nodata_tags(
    epoch: float,
    census: NodataCensus,
    *,
    model_unknown: int,
    model_total: int,
    model_grid: str,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
) -> dict[str, str]:
    """The honest nodata declaration: two units, both named, both counted.

    Every sentence below is a TEMPLATE — it contains no digit of its own and
    interpolates only quantities measured for the raster being written.  That rule
    exists because six sentences have already been retracted from this project for
    carrying a figure that was impossible on the set the sentence itself named.
    """
    if census.cells:
        bbox_rowcol = (
            f"row_min={census.row_min} row_max={census.row_max} "
            f"col_min={census.col_min} col_max={census.col_max} "
            "(0-based, inclusive, band 1)"
        )
        bbox_lonlat = (
            f"lon_min={census.lon_min:.6f} lon_max={census.lon_max:.6f} "
            f"lat_min={census.lat_min:.6f} lat_max={census.lat_max:.6f} "
            "(cell EDGES, in the raster's own CRS and units)"
        )
        touches = "true" if census.touches_edge else "false"
    else:
        bbox_rowcol = "NONE:this file carries no nodata cell"
        bbox_lonlat = "NONE:this file carries no nodata cell"
        touches = "NOT_APPLICABLE:this file carries no nodata cell"

    # BOTH RESOLUTIONS ARE MEASURED, NEITHER IS WRITTEN DOWN.  "10-arcmin" and
    # "15-arcsec" happen to be true of the shipped pair of grids; as literals they
    # would survive a resample and start lying, which is the exact failure this whole
    # tag pass exists to repair.  The model side comes off the ICE-6G coordinate
    # vectors, the raster side off this file's own transform.
    model_arcmin = _model_cell_arcmin(epoch)
    raster_arcsec = float(census.cell_lat) * 3600.0
    nominal = _nominal_cells_per_model_cell(epoch, census)
    if model_unknown:
        measured_ratio = census.cells / float(model_unknown)
        ratio_clause = (
            f"the measured ratio is {measured_ratio:.1f} raster cells per unknown "
            f"model cell against a nominal {nominal:.0f} from the two cell sizes, "
            f"because the ice-aware smoothing rim (SMOOTHING_RIM_CELLS="
            f"{SMOOTHING_RIM_CELLS}) widens the hole rather than resting a published "
            "value on one-sided support"
        )
    else:
        ratio_clause = (
            "no model cell is unknown at this epoch, so no ratio between the two "
            "counts is defined"
        )

    if census.cells == 0:
        fill_policy = (
            "NOT_APPLICABLE:this file carries no nodata cell, so there is nothing to "
            "fill and no extrapolation question"
        )
    elif census.touches_edge:
        edges = "/".join(census.edges_touched)
        counts = ", ".join(
            f"{name} border {getattr(census, 'edge_' + name)} of "
            f"{census.width if name in ('north', 'south') else census.height}"
            for name in census.edges_touched
        )
        fill_policy = (
            f"EXTRAPOLATION:the hole reaches the {edges} raster edge ({counts}), so "
            "any fill of it is extrapolation beyond the data rather than "
            "interpolation between it.  This bake fills nothing: the cells carry the "
            f"declared nodata and gia_nan_policy={gia_nan_policy}"
        )
    else:
        fill_policy = (
            "INTERIOR:the hole touches no raster edge, so it is enclosed by data on "
            "every side.  This bake still fills nothing: the cells carry the declared "
            f"nodata and gia_nan_policy={gia_nan_policy}"
        )

    tags = {
        "nodata_value_m": f"{census.nodata_value:.1f}",
        "nodata_raster_cells": str(census.cells),
        "nodata_raster_cells_total": str(census.cells_total),
        "nodata_raster_cells_fraction": f"{census.fraction:.9f}",
        "nodata_raster_grid": (
            f"{census.height} rows x {census.width} cols, cell "
            f"{census.cell_lat * 3600.0:.3f} x {census.cell_lon * 3600.0:.3f} arcsec, "
            "band 1, in the raster's own CRS"
        ),
        "nodata_bbox_rowcol": bbox_rowcol,
        "nodata_bbox_lonlat": bbox_lonlat,
        "nodata_touches_raster_edge": touches,
        "nodata_edge_cells": (
            f"north={census.edge_north} south={census.edge_south} "
            f"west={census.edge_west} east={census.edge_east} "
            "(nodata cells lying ON each border row/column)"
        ),
        "nodata_fill_policy": fill_policy,
        "gia_unknown_ice6g_model_cells": str(model_unknown),
        "gia_unknown_ice6g_model_cells_total": str(model_total),
        "gia_unknown_ice6g_model_grid": model_grid,
        "nodata_units_note": (
            "TWO DIFFERENT UNITS, BOTH NAMED.  gia_unknown_ice6g_model_cells="
            f"{model_unknown} of {model_total} counts ICE-6G MODEL cells of "
            f"{model_arcmin:.3g} arcmin whose Topo is the ICE SURFACE, so dz there is "
            f"UNKNOWN.  nodata_raster_cells={census.cells} of {census.cells_total} "
            f"counts RASTER cells of this file of {raster_arcsec:.3g} arcsec carrying "
            f"the declared sentinel, MEASURED FROM THIS FILE.  {ratio_clause}.  "
            "The legacy tag "
            "gia_unknown_ice6g_cells carries the MODEL-cell number and its name says "
            "no unit; read gia_unknown_ice6g_model_cells for that quantity and "
            "nodata_raster_cells for the raster one, and never one for the other"
        ),
        "gia_unknown_ice6g_cells_unit": (
            f"ICE-6G MODEL cells of {model_arcmin:.3g} arcmin (NOT raster cells).  "
            "Superseded in name only by gia_unknown_ice6g_model_cells, which carries "
            "the identical number with its unit in the tag name; the raster-cell "
            "count of nodata is nodata_raster_cells"
        ),
    }
    return {k: _ascii_tag(k, v) for k, v in tags.items()}


def _nominal_cells_per_model_cell(epoch: float, census: NodataCensus) -> float:
    """Raster cells per model cell from the two grids' own cell sizes.

    NOMINAL only, and it is reported beside the measured ratio precisely so a reader
    can see the two disagree: on the shipped frames the nominal figure is ~1600 and
    the measured one ~3026, because the ice-aware smoothing rim widens the hole.
    Deriving the raster-cell count from this number is the mistake the pass exists to
    stop, so it is published as a foil and never used as an estimate.
    """
    d_lat, d_lon = _model_cell_deg(epoch)
    return (d_lat / float(census.cell_lat)) * (d_lon / float(census.cell_lon))


# ---------------------------------------------------------------------------
# 2. Three machine-readable water surfaces, each with the rule that made it
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _pontic_curve() -> pd.DataFrame:
    """The Pontic curve, read once.  ``comment='#'`` is mandatory (contract §8)."""
    return pd.read_csv(PONTIC_CSV, comment="#")


@lru_cache(maxsize=1)
def _caspian_curve() -> pd.DataFrame:
    """The validated Caspian curve, read once.

    Private and cached so that tagging 43 rasters does not re-validate the same file
    43 times.  :func:`load_curve` stays the public, uncached entry point — a caller
    that wants the validation to run again is entitled to it.
    """
    return load_curve()


def _pontic_gap_reason(df: pd.DataFrame, epoch: float, scenario: str, gap: str) -> str:
    """Why the Pontic level is NULL here, and why it must not be interpolated.

    A template: it carries no digit of its own.  The bracketing ages, their rules,
    their levels and the width of the nearest-row tie are all MEASURED off the curve
    handed in, for the epoch being written.

    The refusal is not squeamishness.  Interpolating across this gap crosses a
    CONVERSION-RULE boundary — at 16.75 ka the row below is converted by ``locus``
    and the row above by ``marker``, which are different estimators of a different
    quantity — and falling back on the nearest row is a coin flip decided by
    DataFrame order across a tie tens of metres wide.
    """
    ages = np.sort(np.asarray(df["age_ka"], dtype=float))
    below = ages[ages < float(epoch)]
    above = ages[ages > float(epoch)]
    # "The nearest row" is NOT always a single row, and on this grid it usually is
    # not: a quarter-ka epoch sits exactly half way between two half-ka rows.  Naming
    # one of them would be the very coin flip this refusal exists to refuse -- and
    # the upstream message quoted in ``gap`` names the other one, because it breaks
    # the tie by DataFrame order.  So the tie is reported as a tie.
    distances = np.abs(ages - float(epoch))
    nearest_d = float(distances.min())
    nearest = [float(a) for a in ages[distances <= nearest_d + 1e-12]]
    if len(nearest) == 1:
        nearest_clause = (
            f"nearest row {nearest[0]:g} ka, {nearest_d:.2f} ka away"
        )
    else:
        nearest_clause = (
            f"{len(nearest)} rows tie for nearest at {nearest_d:.2f} ka "
            f"({'/'.join(f'{a:g}' for a in nearest)} ka), so there is no such thing "
            "as THE nearest row here"
        )
    head = (
        f"REFUSED:{gap}  The Pontic curve carries no row at {float(epoch):g} ka "
        f"(its grid is coarser than this bake's; {nearest_clause}, tolerance "
        f"{_CURVE_AGE_TOL_KA:g} ka)."
    )
    if below.size == 0 or above.size == 0:
        return (
            head
            + "  The epoch is outside the curve's own age range on one side, so it "
            "is not even bracketed.  No value is written and none is guessed."
        )
    age_lo, age_hi = float(below[-1]), float(above[0])

    def _rule_and_level(age: float) -> tuple[str, float | None, float | None]:
        try:
            level = float(pontic_level(df, age, scenario))
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            return f"UNRESOLVED:{type(exc).__name__}", None, None
        try:
            th = pontic_threshold(df, age, scenario)
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            return f"UNRESOLVED:{type(exc).__name__}", level, None
        return str(th.rule), level, float(th.threshold_m)

    rule_lo, level_lo, thr_lo = _rule_and_level(age_lo)
    rule_hi, level_hi, thr_hi = _rule_and_level(age_hi)

    if rule_lo != rule_hi:
        rule_clause = (
            f"the two bracketing rows are converted by DIFFERENT rules ({rule_lo} at "
            f"{age_lo:g} ka, {rule_hi} at {age_hi:g} ka), so any interpolant between "
            "them would cross a rule boundary and belong to neither estimator"
        )
    else:
        rule_clause = (
            f"both bracketing rows are converted by rule {rule_lo}, so no rule "
            "boundary is crossed here and this epoch's refusal rests on the plainer "
            "fact that the curve carries no row at it"
        )
    if level_lo is None or level_hi is None:
        tie_clause = (
            "and the width of the choice a nearest-row fallback would make could not "
            "be measured, because at least one bracketing row has no readable level"
        )
    else:
        tie_clause = (
            f"and a nearest-row fallback would be choosing between levels "
            f"{level_lo:+.2f} m at {age_lo:g} ka and {level_hi:+.2f} m at "
            f"{age_hi:g} ka, {abs(level_hi - level_lo):.2f} m apart"
        )
        if thr_lo is not None and thr_hi is not None:
            tie_clause += (
                f" (as water surfaces in this raster's frame, {thr_lo:+.3f} m and "
                f"{thr_hi:+.3f} m, {abs(thr_hi - thr_lo):.3f} m apart)"
            )
    return (
        f"{head}  NOT INTERPOLATED: {rule_clause}, {tie_clause}.  "
        "No value is written and none is guessed."
    )


def _level_record(
    scenario: str,
    value: float | None,
    rule: str | None,
    *,
    status: str,
    bracket: tuple[float, float] | None = None,
    refusal: str | None = None,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "water_level_m": None if value is None else float(value),
        "status": status,
        "rule": rule,
        "bracket_m": None if bracket is None else [float(bracket[0]), float(bracket[1])],
        "refusal": refusal,
    }


def _sheet_payload(basin: str, epoch: float, levels: list[dict[str, object]]) -> dict:
    return {
        "schema": MESHER_TAG_SCHEMA,
        "basin": basin,
        "epoch_ka": float(epoch),
        "unit": "m",
        "frame": (
            "F1 palaeo water surface expressed against the PRESENT-DAY geoid "
            "(vertical-datum contract sec.1)"
        ),
        "comparable_with": (
            "this raster's own elevations, which are F0 = H_modern + dz_solid on the "
            "same present-day geoid; a cell is submerged in this basin iff "
            "H_paleo < water_level_m"
        ),
        "written_by": MESHER_TAG_CODE_PATH,
        "levels": levels,
    }


def water_level_tags(
    epoch: float,
    *,
    caspian_curve: pd.DataFrame | None = None,
    pontic_curve: pd.DataFrame | None = None,
) -> dict[str, str]:
    """One JSON tag per basin: the water surface, its unit, its frame, its rule.

    THE THREE SHEETS, AND WHY THREE.

    * **caspian** — endorheic at every row of its curve, present day included, so its
      surface is never ``-offset(t)``.  ``T = level_F2 + dz_solid(locus)``, converted
      at the threshold and never in the curve (contract §8).  Two scenarios, because
      the curve's contested axis is the MAGNITUDE of the highstand and the two
      schools do not agree on it; both are carried, neither is preferred here.
    * **mediterranean** — the world ocean reaching this frame through the Aegean.
      Its surface is ``-offset(t)`` EXACTLY (contract §2): whatever local sea-surface
      residual is baked into ``dz_engine`` cancels against the same scalar in the
      threshold.  This is the number that already shipped as the negation of
      ``gia_sea_surface_offset_m`` with no tag anywhere saying that is what it was.
      One value, no scenario axis: a global mean has no schools.
    * **pontic** — a lake before reconnection and the ocean after, so its rule
      CHANGES with epoch and the tag says which one applied.  At 8 of this bake's 43
      epochs it is NOT COMPUTABLE and the record carries an explicit null plus the
      measured reason; see :func:`_pontic_gap_reason`.

    Nothing is substituted for a missing value anywhere below.  A basin that cannot
    be resolved gets ``water_level_m: null`` and a ``refusal`` string; it never gets
    a neighbouring epoch's number, an interpolant, or another basin's surface.
    """
    epoch = float(epoch)
    tags: dict[str, str] = {}

    # ── caspian ──────────────────────────────────────────────────────────────
    caspian_levels: list[dict[str, object]] = []
    try:
        df_c = _caspian_curve() if caspian_curve is None else caspian_curve
    except Exception as exc:  # noqa: BLE001 - disclosed, never substituted
        df_c = None
        caspian_unreadable = f"{type(exc).__name__}: {exc}"
    else:
        caspian_unreadable = ""
    for scenario in SCENARIOS:
        if df_c is None:
            caspian_levels.append(
                _level_record(
                    scenario,
                    None,
                    None,
                    status="REFUSED",
                    refusal=(
                        "the Caspian curve could not be read, so no level exists to "
                        f"convert: {caspian_unreadable}"
                    ),
                )
            )
            continue
        try:
            th = caspian_threshold(df_c, epoch, scenario)
        except Exception as exc:  # noqa: BLE001 - disclosed, never substituted
            caspian_levels.append(
                _level_record(
                    scenario,
                    None,
                    None,
                    status="REFUSED",
                    refusal=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        caspian_levels.append(
            _level_record(
                scenario,
                th.threshold_m,
                (
                    f"rule={th.rule} frame={th.frame}: water_level = level_F2("
                    f"{th.level_m:+.2f} m) + dz_solid({th.conversion_m:+.3f} m"
                    + (
                        ""
                        if th.n_cells is None
                        else f", median over locus "
                        f"{CASPIAN_SHORELINE_BAND.name!r}, {th.n_cells} ICE-6G cells"
                    )
                    + ").  Converted at the threshold, never in the curve "
                    "(vertical-datum contract sec.8).  The full prose provenance is "
                    f"the raster tag threshold_{scenario}"
                ),
                status="OK",
                bracket=th.bracket_m,
            )
        )
    tags["water_level_caspian_json"] = _json_tag(
        _sheet_payload("caspian", epoch, caspian_levels)
    )

    # ── mediterranean (the world ocean) ──────────────────────────────────────
    offset = sea_surface_offset(epoch)
    mediterranean = _sheet_payload(
        "mediterranean",
        epoch,
        [
            _level_record(
                "all",
                -offset + 0.0,  # + 0.0 normalises the -0.0 that -offset gives at 0 ka
                (
                    "rule=ocean_offset frame=F3->F1, EXACT (vertical-datum contract "
                    "sec.2): water_level = -offset(t), offset = "
                    "implied_sea_surface_offset = median dTopo over far-field ice-free "
                    "tropical continental land, taken from the GLOBAL ICE-6G Topo "
                    "field.  It needs no uniformity assumption: whatever local "
                    "sea-surface residual is baked into dz_solid cancels against the "
                    "same scalar here.  Identical to the negation of this raster's "
                    "gia_sea_surface_offset_m tag, which is what that tag has always "
                    "meant and never said"
                ),
                status="OK",
            )
        ],
    )
    mediterranean["scenario_axis"] = (
        "none: a global-mean sea surface has no competing schools, so this sheet "
        "carries one level and the scenario is spelled 'all'"
    )
    tags["water_level_mediterranean_json"] = _json_tag(mediterranean)

    # ── pontic ───────────────────────────────────────────────────────────────
    pontic_levels: list[dict[str, object]] = []
    try:
        df_p = _pontic_curve() if pontic_curve is None else pontic_curve
    except Exception as exc:  # noqa: BLE001 - disclosed, never substituted
        df_p = None
        pontic_unreadable = f"{type(exc).__name__}: {exc}"
    else:
        pontic_unreadable = ""
    for scenario in PONTIC_SCENARIOS:
        if df_p is None:
            pontic_levels.append(
                _level_record(
                    scenario,
                    None,
                    None,
                    status="REFUSED",
                    refusal=(
                        "the Pontic curve could not be read, so no level exists to "
                        f"convert: {pontic_unreadable}"
                    ),
                )
            )
            continue
        try:
            th = pontic_threshold(df_p, epoch, scenario)
        except _bs.CurveAgeGapError as exc:
            pontic_levels.append(
                _level_record(
                    scenario,
                    None,
                    None,
                    status="REFUSED",
                    refusal=_pontic_gap_reason(df_p, epoch, scenario, str(exc)),
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - disclosed, never substituted
            pontic_levels.append(
                _level_record(
                    scenario,
                    None,
                    None,
                    status="REFUSED",
                    refusal=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if th.rule == "ocean_offset":
            rule = (
                f"rule={th.rule} frame={th.frame}, EXACT (vertical-datum contract "
                "sec.2): the basin has RECONNECTED and IS the ocean at this epoch, so "
                "water_level = -offset(t) and the curve's own level "
                f"({th.level_m:+.2f} m) is a GLOBAL MEAN that is NOT used as the "
                "surface.  Same number as the mediterranean sheet: after "
                "reconnection the two share a surface"
            )
        else:
            rule = (
                f"rule={th.rule} frame={th.frame}: water_level = level_F2("
                f"{th.level_m:+.2f} m) + dz_solid({th.conversion_m:+.3f} m"
                + ("" if th.n_cells is None else f", over {th.n_cells} ICE-6G cells")
                + f"), marker_source={th.marker_source}.  Converted at the threshold, "
                "never in the curve (vertical-datum contract sec.8)"
            )
        pontic_levels.append(
            _level_record(
                scenario, th.threshold_m, rule, status="OK", bracket=th.bracket_m
            )
        )
    pontic = _sheet_payload("pontic", epoch, pontic_levels)
    pontic["rule_changes_with_epoch"] = (
        "yes: a closed lake before reconnection (F2 level + dz_solid) and the ocean "
        "after it (-offset(t)).  Read the per-scenario 'rule' field; do not assume "
        "one rule holds across the sequence"
    )
    tags["water_level_pontic_json"] = _json_tag(pontic)

    tags["water_level_basins"] = ",".join(BASIN_SHEETS)
    tags["water_level_index"] = _ascii_tag(
        "water_level_index",
        "one JSON tag per water sheet, each carrying the value, its unit, its frame "
        "and the RULE that produced it: "
        + ", ".join(f"water_level_{b}_json" for b in BASIN_SHEETS)
        + ".  A null water_level_m is always accompanied by a 'refusal' string "
        "stating what could not be established; it is never a zero and never a "
        "neighbouring epoch's value",
    )
    return {k: _ascii_tag(k, v) for k, v in tags.items()}


# ---------------------------------------------------------------------------
# 3. The composed tag set, and the in-place additive pass over baked rasters
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def file_sha256(path: str) -> str:
    """sha256 of a file, streamed.  Cached: the source COG is one file for 43 epochs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dem_source_sha256(dem_path: Path) -> str:
    try:
        return file_sha256(str(Path(dem_path)))
    except OSError as exc:
        return f"UNAVAILABLE:{type(exc).__name__} reading {dem_path}"


def mesher_tags(
    epoch: float,
    census: NodataCensus,
    *,
    pixel_sha256: str,
    dem_path: Path,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
    caspian_curve: pd.DataFrame | None = None,
    pontic_curve: pd.DataFrame | None = None,
) -> dict[str, str]:
    """The whole additive set for one raster.  Pure in its inputs, hence idempotent.

    Given the same pixels, the same curves and the same epoch this returns a
    byte-identical dict, which is what makes the in-place pass re-runnable: a second
    run compares equal and writes the same bytes rather than accumulating drift.
    """
    model_unknown, model_total = gia_unknown_model_cells(epoch)
    tags: dict[str, str] = {
        "mesher_tag_schema": MESHER_TAG_SCHEMA,
        "mesher_tags_written_by": (
            f"{MESHER_TAG_CODE_PATH} (additive pass; writes no pixel, and asserts "
            "that by sha256 of the band before and after)"
        ),
        "pixel_sha256": pixel_sha256,
        "pixel_sha256_of": (
            f"band 1 as float32, C order, row-major, {census.height} rows x "
            f"{census.width} cols, exactly the bytes rasterio read(1) returns"
        ),
        "dem_source_sha256": _dem_source_sha256(dem_path),
    }
    tags.update(
        nodata_tags(
            epoch,
            census,
            model_unknown=model_unknown,
            model_total=model_total,
            model_grid=_model_grid_note(epoch),
            gia_nan_policy=gia_nan_policy,
        )
    )
    tags.update(
        water_level_tags(
            epoch, caspian_curve=caspian_curve, pontic_curve=pontic_curve
        )
    )
    tags["mesher_tags"] = ",".join(sorted([*tags, "mesher_tags"]))
    return {k: _ascii_tag(k, v) for k, v in tags.items()}


@dataclass(frozen=True)
class RetagResult:
    """One raster's trip through the additive pass."""

    path: str
    epoch_ka: float
    pixel_sha256_before: str
    pixel_sha256_after: str
    nodata_raster_cells: int
    gia_unknown_ice6g_model_cells: int
    legacy_tag_agrees: bool
    tags_written: int
    tags_changed: int
    already_current: bool
    pontic_levels_resolved: int
    pontic_levels_refused: int

    @property
    def pixels_unchanged(self) -> bool:
        return self.pixel_sha256_before == self.pixel_sha256_after


#: Read attempts per band before the pass gives up on a raster.  The product tree
#: lives on an SMB share and a 43-epoch pass reads ~11 GB through it twice; a first
#: run died on its first file with ``TIFFFillTile: got 0 bytes, expected 101230``,
#: and the same file read cleanly on its own seconds later.  A transport hiccup must
#: not cost 1.8 h of work.
_BAND_READ_ATTEMPTS = 4


def read_band(path: Path, *, attempts: int = _BAND_READ_ATTEMPTS):
    """``(band, transform, nodata, tags)``, retrying a TRANSPORT failure only.

    THIS RETRY CANNOT HIDE A CORRUPT FILE, and that is by construction rather than by
    hope.  It re-attempts only when the read RAISES; a read that succeeds is taken as
    it came, and :func:`retag_paleo_dem` then compares the band's sha256 before and
    after the tag write.  Two successful reads that disagree therefore raise
    :class:`PixelDataChangedError` instead of being silently reconciled.  A raster
    that fails every attempt is reported with its epoch, never skipped.
    """
    last: Exception | None = None
    for attempt in range(1, int(attempts) + 1):
        try:
            with rasterio.open(path) as ds:
                if ds.nodata is None:
                    raise ValueError(
                        f"{path} declares no nodata value, so 'how many nodata cells "
                        "does it hold' has no answer.  A sentinel that is not "
                        "declared is a number that reads as an elevation."
                    )
                return (
                    np.ascontiguousarray(ds.read(1)),
                    ds.transform,
                    float(ds.nodata),
                    dict(ds.tags()),
                )
        except rasterio.errors.RasterioIOError as exc:
            last = exc
            logger.warning(
                "%s: band read failed on attempt %d/%d (%s).  Retrying; this is a "
                "transport retry and it does not relax the pixel-identity check.",
                Path(path).name,
                attempt,
                attempts,
                exc,
            )
            time.sleep(2.0 * attempt)
    raise RuntimeError(
        f"{path}: the band could not be read in {attempts} attempts.  The last error "
        f"was {type(last).__name__}: {last}.  This raster is NOT retagged and NOT "
        "skipped silently -- establish whether the file or the transport is at fault "
        "before re-running."
    ) from last


def _epoch_of_raster(path: Path, tags: Mapping[str, str]) -> float:
    """The epoch this raster IS, taken from its own tag.

    From the tag and not from the filename: the filename is a label a rename can
    change, the tag was written by the run that computed the pixels.  A raster with
    no ``epoch_ka`` tag is refused rather than parsed out of its name.
    """
    raw = tags.get("epoch_ka")
    if raw is None:
        raise ValueError(
            f"{path} carries no 'epoch_ka' tag, so which epoch it is cannot be "
            "established from the file.  Parsing it out of the filename would make a "
            "rename able to change a raster's age; pass epoch= explicitly if you know "
            "it from elsewhere."
        )
    return float(raw)


def retag_paleo_dem(
    path: Path,
    *,
    epoch: float | None = None,
    caspian_curve: pd.DataFrame | None = None,
    pontic_curve: pd.DataFrame | None = None,
    dry_run: bool = False,
    sidecar_dir: Path | None = None,
) -> RetagResult:
    """Add the mesher tag set to an ALREADY-BAKED raster, in place.

    THE SAFETY OF THIS PASS IS ONE ASSERTION.  The band is read and hashed BEFORE the
    tags are written and read and hashed AGAIN afterwards; if the two hashes differ
    the function raises :class:`PixelDataChangedError`.  Nothing else in this module
    can change an elevation, and this is how that is proved rather than asserted.

    RE-RUNNABLE AND IDEMPOTENT.  :func:`mesher_tags` is a pure function of the pixels,
    the curves and the epoch, so a second run computes the same strings; the result
    reports ``already_current=True`` and ``tags_changed=0``.  The pre-pass tag state
    is snapshotted to a sidecar the FIRST time only — a re-run must not overwrite the
    genuine original with a copy of this pass's own output.

    THE LEGACY TAG IS CHECKED, NOT COPIED.  ``gia_unknown_ice6g_cells`` is recomputed
    from the ICE-6G distribution and compared with what the raster carries;
    ``legacy_tag_agrees`` records the answer per epoch instead of assuming it.
    """
    path = Path(path)
    array, transform, nodata, before_tags = read_band(path)
    epoch = _epoch_of_raster(path, before_tags) if epoch is None else float(epoch)
    sha_before = hashlib.sha256(array.tobytes()).hexdigest()
    census = nodata_census_from_mask(
        array == np.asarray(nodata, dtype=array.dtype), transform, nodata
    )
    del array

    dem_path = Path(before_tags.get("dem_source") or KHV_DEM)
    new_tags = mesher_tags(
        epoch,
        census,
        pixel_sha256=sha_before,
        dem_path=dem_path,
        gia_nan_policy=before_tags.get("gia_nan_policy", KHV_GIA_NAN_POLICY),
        caspian_curve=caspian_curve,
        pontic_curve=pontic_curve,
    )
    changed = {k: v for k, v in new_tags.items() if before_tags.get(k) != v}

    model_unknown = int(new_tags["gia_unknown_ice6g_model_cells"])
    legacy_raw = before_tags.get("gia_unknown_ice6g_cells")
    legacy_agrees = legacy_raw is not None and int(legacy_raw) == model_unknown
    if legacy_raw is not None and not legacy_agrees:
        logger.warning(
            "%s: the shipped gia_unknown_ice6g_cells tag reads %s but the ICE-6G "
            "distribution gives %d unknown MODEL cells at %g ka.  The recomputed "
            "value is what gia_unknown_ice6g_model_cells carries; the legacy tag is "
            "left exactly as it was.",
            path.name,
            legacy_raw,
            model_unknown,
            epoch,
        )

    pontic = json.loads(new_tags["water_level_pontic_json"])["levels"]
    resolved = sum(1 for r in pontic if r["water_level_m"] is not None)

    sidecar_dir = Path(sidecar_dir) if sidecar_dir is not None else RETAG_SIDECAR_DIR
    if not dry_run:
        sidecar = sidecar_dir / f"{path.stem}.tags-before.json"
        if not sidecar.exists():
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                json.dumps(
                    {
                        "raster": str(path),
                        "epoch_ka": epoch,
                        "pixel_sha256": sha_before,
                        "snapshot_of": (
                            "the GDAL tag set as it stood BEFORE "
                            f"{MESHER_TAG_SCHEMA} was applied; written once, never "
                            "overwritten by a re-run"
                        ),
                        "tags": before_tags,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                encoding="ascii",
            )
        if changed:
            with rasterio.open(path, "r+") as ds:
                ds.update_tags(**new_tags)

    sha_after = hashlib.sha256(read_band(path)[0].tobytes()).hexdigest()
    if sha_after != sha_before:
        raise PixelDataChangedError(
            f"{path}: the band-1 sha256 moved from {sha_before} to {sha_after} while "
            "only GDAL metadata was written.  This pass is additive by contract and "
            "must not touch a single elevation.  Do not re-run it against this file; "
            "restore the raster and re-bake it."
        )

    return RetagResult(
        path=str(path),
        epoch_ka=epoch,
        pixel_sha256_before=sha_before,
        pixel_sha256_after=sha_after,
        nodata_raster_cells=census.cells,
        gia_unknown_ice6g_model_cells=model_unknown,
        legacy_tag_agrees=bool(legacy_agrees),
        tags_written=len(new_tags),
        tags_changed=len(changed),
        already_current=not changed,
        pontic_levels_resolved=resolved,
        pontic_levels_refused=len(pontic) - resolved,
    )


def retag_all(
    topo_dir: Path | None = None,
    epochs: Sequence[float] | None = None,
    *,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    sidecar_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the additive pass over every baked raster; return (and write) a manifest.

    ~2.5 min per epoch on the shipped 7800x4320 frames, dominated by the two full
    band reads the sha256 assertion requires.  Over 43 epochs that is ~1.8 h, which
    is longer than a backgrounded shell task survives — **launch it detached**.

    A failure on one epoch aborts: a manifest that silently omits the raster that
    broke is worse than no manifest, and the pass is safe to re-run from the start
    because it is idempotent.
    """
    topo_dir = Path(topo_dir) if topo_dir is not None else OUT_TOPO
    if epochs is None:
        paths = sorted(
            topo_dir.glob("khv_*ka.tif"), key=lambda p: -float(p.stem[4:-2])
        )
    else:
        paths = [topo_dir / f"khv_{float(e):g}ka.tif" for e in epochs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raster(s) named for this pass do not exist: "
            f"{[str(p) for p in missing[:5]]}"
        )

    rows = []
    for n, p in enumerate(paths, start=1):
        result = retag_paleo_dem(
            p, dry_run=dry_run, sidecar_dir=sidecar_dir
        )
        rows.append(asdict(result) | {"pixels_unchanged": result.pixels_unchanged})
        logger.info(
            "[%d/%d] %s  %g ka  nodata=%d cells  model_unknown=%d  legacy_agrees=%s  "
            "pontic %d/%d resolved  tags_changed=%d  pixels_unchanged=%s",
            n,
            len(paths),
            p.name,
            result.epoch_ka,
            result.nodata_raster_cells,
            result.gia_unknown_ice6g_model_cells,
            result.legacy_tag_agrees,
            result.pontic_levels_resolved,
            result.pontic_levels_resolved + result.pontic_levels_refused,
            result.tags_changed,
            result.pixels_unchanged,
        )

    manifest = pd.DataFrame(rows)
    if not dry_run:
        target = (
            Path(manifest_path) if manifest_path is not None else RETAG_MANIFEST_CSV
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(target, index=False)
        logger.info("Retag manifest: %s", target)
    return manifest


def build_paleo_dem(
    epoch: float,
    *,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
    provenance: Mapping[str, str] | None = None,
    dem_path: Path | None = None,
    out_dir: Path | None = None,
    strip_rows: int = STRIP_ROWS,
    caspian_curve: pd.DataFrame | None = None,
    pontic_curve: pd.DataFrame | None = None,
) -> Path:
    """Bake the GIA-corrected paleo-DEM GeoTIFF for *epoch*; return its path.

    ``H_paleo = H_modern + Δz_solid(t)``, written strip by strip: the working raster
    is 7800x4320 and is read through windows, never whole.  Strip size does not
    change the result — pinned by
    ``test_striped_bake_is_identical_to_a_single_strip_bake``.

    Parameters
    ----------
    gia_nan_policy : {'raise', 'nodata'}
        What to do when Δz is unknown because ICE-6G ``Topo`` is the ice surface.
        Default ``"nodata"`` for this scene, which genuinely straddles the
        Fennoscandian margin before 17 ka (module docstring).  There is no
        ``'zero'``.
    provenance : mapping, optional
        Extra GDAL tags.  :func:`main` uses it to record every threshold and every
        sill verdict this epoch carries, so a .tif that outlives the run still says
        what frame it belongs to and what it decided.
    """
    if gia_nan_policy not in BAKE_GIA_NAN_POLICIES:
        raise ValueError(
            f"gia_nan_policy={gia_nan_policy!r} is not one of "
            f"{BAKE_GIA_NAN_POLICIES}.  There is deliberately no 'zero' policy: "
            "0 m of correction is a legitimate MEASURED value over stable far-field "
            "ground, so writing it for an unknown cell makes the two "
            "indistinguishable."
        )

    dem_path = Path(dem_path) if dem_path is not None else KHV_DEM
    out_dir = Path(out_dir) if out_dir is not None else OUT_TOPO

    dz, model_lats, model_lons = solid_displacement_field(epoch)

    # ── Unknown Δz policy ────────────────────────────────────────────────────
    gia_unknown = ~np.isfinite(dz)
    n_unknown = int(np.count_nonzero(gia_unknown))
    if n_unknown:
        detail = (
            f"epoch {epoch:g} ka: the GIA correction is unknown over {n_unknown} of "
            f"{gia_unknown.size} ICE-6G cells "
            f"({100.0 * n_unknown / gia_unknown.size:.1f}%) of the "
            f"{LAT_MIN - BUF:g}..{LAT_MAX + BUF:g}N / {LON_MIN - BUF:g}.."
            f"{LON_MAX + BUF:g}E window, because ICE-6G 'Topo' is the ICE SURFACE "
            "there (the SE margin of the Fennoscandian sheet clips this scene's NW "
            "corner before 17 ka) and these files carry no bedrock-under-ice field."
        )
        if gia_nan_policy == "raise":
            raise UnknownGiaInBakeError(
                detail
                + "  Restrict this bbox/epoch to ice-free ground, take Δz from "
                "ICE-7G's bedrock-deformation field, or keep the default "
                "gia_nan_policy='nodata' to record the hole as nodata.  It will "
                "not be filled with 0 m."
            )
        logger.warning(
            "%s  gia_nan_policy='nodata': written as nodata=%r, NOT as 0 m of "
            "correction.",
            detail,
            KHV_NODATA_M,
        )

    # ── Mask BEFORE smoothing ────────────────────────────────────────────────
    # solid_displacement() already NaN-ed the ice cells; this is the ice-aware
    # smooth, with the documented rim so margin cells resting on one-sided support
    # are discarded rather than published.  mode="reflect" reproduces exactly the
    # boundary handling apply_gia_delta applies historically, so an ice-free window
    # is bit-for-bit the plain computation.
    dz_smooth = smooth_preserving_ice_mask(
        dz.astype(np.float64),
        sigma=GIA_SIGMA_BLUR,
        dilate_cells=SMOOTHING_RIM_CELLS,
        mode="reflect",
    )
    ice_height = np.zeros_like(dz_smooth)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"khv_{float(epoch):g}ka.tif"

    with rasterio.open(dem_path) as src:
        tile_lats, tile_lons = _tile_coordinates(src)
        _require_tile_is_the_declared_scene(src)
        _require_model_grid_brackets_tile(model_lats, model_lons, tile_lats, tile_lons)
        src_nodata = src.nodata
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": KHV_NODATA_M,
            "compress": "deflate",
            # PREDICTOR=3 is the floating-point predictor.  LOSSLESS — it reorders
            # bytes before deflate, it does not quantise — and measured on the real
            # 7800x4320 frame it takes the file from 107.9 MB to 41.9 MB and is
            # FASTER to write.  Over the 43-epoch grid that is 4.6 GB -> 1.8 GB,
            # which matters: PALEO_OUT_ROOT's volume had 20 GB free when this was
            # written.  Bit-exactness is pinned by
            # test_striped_bake_is_identical_to_a_single_strip_bake, which reads
            # both files back and compares arrays.
            "predictor": 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
        # THE MESHER TAGS ARE MEASURED ON THE PIXELS AS THEY ARE WRITTEN, not
        # inferred from the model grid afterwards.  The running hash consumes the
        # strips in write order, which is row-major from row 0 over the full width,
        # so it equals the hash of the whole band as ``read(1)`` returns it — pinned
        # by ``test_the_pixel_sha256_tag_is_the_hash_of_the_band_as_read_back``.
        pixel_digest = hashlib.sha256()
        nodata_mask = np.zeros((src.height, src.width), dtype=bool)

        with rasterio.open(out_tif, "w", **profile) as dst:
            for row0 in range(0, src.height, int(strip_rows)):
                nrows = min(int(strip_rows), src.height - row0)
                window = Window(0, row0, src.width, nrows)
                modern = src.read(1, window=window).astype(np.float32)

                # The modern DEM's own holes stay holes.  They are written as the
                # declared nodata, not as an elevation, and never confused with a
                # GIA hole: the two have different causes and the tag says which
                # policy governed the GIA one.
                src_hole = (
                    np.zeros(modern.shape, dtype=bool)
                    if src_nodata is None
                    else (modern == np.float32(src_nodata))
                )

                paleo, _ice = apply_gia_delta(
                    modern,
                    model_lats,
                    model_lons,
                    dz_smooth,
                    ice_height,
                    tile_lats[row0 : row0 + nrows],
                    tile_lons,
                    # Already smoothed above, ice-mask-preserving.  apply_gia_delta's
                    # own gaussian_filter is a plain one: it would smooth twice and
                    # propagate NaN across the kernel support.  0.0 makes it an exact
                    # identity.
                    sigma_blur=0.0,
                )
                # Unknown Δz interpolates to NaN on the tile grid.  Write the DECLARED
                # nodata — an honest "unknown" — never a number that reads as
                # elevation, and never 0 m of correction.
                paleo = np.where(
                    src_hole | ~np.isfinite(paleo), np.float32(KHV_NODATA_M), paleo
                )
                paleo = np.ascontiguousarray(paleo.astype(np.float32))
                dst.write(paleo, 1, window=window)
                pixel_digest.update(paleo.tobytes())
                nodata_mask[row0 : row0 + nrows, :] = paleo == np.float32(KHV_NODATA_M)

            census = nodata_census_from_mask(
                nodata_mask, src.transform, float(KHV_NODATA_M)
            )
            mesher = mesher_tags(
                epoch,
                census,
                pixel_sha256=pixel_digest.hexdigest(),
                dem_path=dem_path,
                gia_nan_policy=gia_nan_policy,
                caspian_curve=caspian_curve,
                pontic_curve=pontic_curve,
            )
            dst.update_tags(**mesher)
            dst.update_tags(
                epoch_ka=f"{float(epoch):g}",
                basin="caspian+manych+pontic (one raster, two basin levels)",
                dem_source=str(dem_path),
                gia_model="ICE-6G_C VM5a 10min",
                gia_datum=(
                    "implied_sea_surface_offset (far-field ice-free tropical land)"
                ),
                gia_sea_surface_offset_m=f"{sea_surface_offset(epoch):.6f}",
                gia_nan_policy=gia_nan_policy,
                gia_unknown_ice6g_cells=str(n_unknown),
                gia_time_interpolation=_time_interpolation_note(epoch),
                vertical_frame="F0 present-day datum (H_modern + dz_solid)",
                vertical_datum_contract=(
                    "Paleo-Europe_3D/docs/superpowers/research/"
                    "2026-07-30-vertical-datum-contract.md"
                ),
                manych_sill_modern_m=f"{MANYCH_SILL_M:.2f}",
                manych_sill_path_m=f"{MANYCH_SILL_PATH_M:.2f}",
                manych_sill_lonlat=f"{MANYCH_SILL_LON:.4f}E/{MANYCH_SILL_LAT:.4f}N",
                **{str(k): str(v) for k, v in (provenance or {}).items()},
            )

    return out_tif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_curve(csv_path: Path | None = None) -> pd.DataFrame:
    """Read and FULLY validate the Caspian curve.  Call once, before any epoch.

    ``comment="#"`` is mandatory: the file leads with a long header block and a bare
    ``read_csv`` takes its first line as the header.  Validation is
    ``validate_basin_curve`` (mirror test, present-day anchor, Manych highstand,
    physical continuity, datum sanity) plus this module's
    :func:`check_no_lgm_lowstand`.  A sign flip is only detectable on the curve as a
    whole, never on an individual scalar — which is exactly the failure mode the
    U+00FE/U+0002 PDF encoding produced once already in this project.
    """
    path = Path(csv_path) if csv_path is not None else CASPIAN_CSV
    df = pd.read_csv(path, comment="#")
    for scenario in SCENARIOS:
        validate_basin_curve(df, "caspian", scenario)
    check_no_lgm_lowstand(df, SCENARIOS)
    return df


def _print_report(report: pd.DataFrame) -> None:
    """Print the report.  The envelope columns are labelled as an ENVELOPE.

    ``circ`` marks a row whose UPPER edge stands on the Manych crest, i.e. where
    ``verdict_envelope_hi`` carries no independent evidence of a spill.  It is a
    column and not a footnote because the two rows it marks (18.0 / 17.5 ka) are the
    only two where the upper edge spills outside the 17-13 ka window.
    """
    header = (
        f"{'epoch':>6} | {'scenario':>11} | {'level_F2':>9} {'surface_F1':>10} | "
        f"{'crest_F1':>8} {'margin':>8} | {'verdict':>13} {'robust':>6} | "
        f"{'+25m path':>13} | {'envelope lo..hi margins':>23} "
        f"{'envelope verdict':>29} {'circ':>5}"
    )
    print()
    print(header)
    print("-" * len(header))
    for _i, r in report.iterrows():
        print(
            f"{r.epoch_ka:>6g} | {r.scenario:>11} | {r.level_f2_m:>+9.2f} "
            f"{r.surface_f1_m:>+10.2f} | {r.sill_f1_m:>+8.2f} {r.margin_m:>+8.2f} | "
            f"{r.verdict:>13} {str(bool(r.verdict_is_robust)):>6} | "
            f"{r.verdict_path:>13} | "
            f"{r.margin_envelope_lo_m:>+11.2f} {r.margin_envelope_hi_m:>+11.2f} "
            f"{r.envelope_verdict:>29} "
            f"{('CIRC' if bool(r.envelope_hi_is_circular) else '-'):>5}"
        )


def main(
    epochs: Sequence[float] | None = None,
    scenarios: Sequence[str] = SCENARIOS,
    *,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
    write_rasters: bool = True,
    write_report: bool | None = None,
    csv_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Bake the Khvalynian side, print the sill report, return it.

    ASKING IS READ-ONLY, BAKING IS NOT (gate re-scoped 2026-08-03).  ``write_report``
    defaults to ``None``, which means **follow ``write_rasters``**:

    * a REAL BAKE (``write_rasters=True``) always refreshes the report.  A run that
      writes rasters and leaves the previous report beside them is the worst of the
      three states — the renderer reads its sill expectation out of that CSV, so
      fresh rasters against a stale report is a silent frame/verdict mismatch.  The
      first version of this gate did exactly that: ``--epochs 16.5`` wrote rasters
      and left the old report in place;
    * a READ-ONLY QUERY (``--sill-report-only``) writes nothing unless ``--write``
      says so.  That flag's whole purpose is "compute this in seconds and show me",
      and before the gate existed it silently overwrote
      ``$PALEO_OUT_ROOT/khvalynian_caspian/khvalynian_sill_test.csv`` as a side
      effect of asking a question.

    ``write_report=True`` / ``False`` still force the issue either way.  The report
    is returned and printed in every case.

    NO FRAME IS RENDERED HERE, and that is a scientific decision rather than an
    omission.  This extent holds **two basins standing at different levels** — that
    is the hero shot's whole premise — so there is no single scalar
    ``sea_level_override_m`` that is correct across it: the Caspian threshold would
    flood the Pontic to the Caspian's surface and vice versa.  Producing the
    two-mask composite is a separate step that needs a basin partition (a connected
    component seeded inside each basin, cut at the Manych sill), and inventing one
    in passing here would put a wrong coastline in front of a reader.  What this
    function writes is the shared paleo-DEM and the sill verdicts; both are exactly
    what that composite consumes.
    """
    epochs = list(EPOCHS_KA if epochs is None else epochs)
    # The gate, resolved once and in one place: None = "follow the bake".
    write_report = bool(write_rasters) if write_report is None else bool(write_report)
    df = load_curve(csv_path)

    sill_modern = manych_sill_elevation_m()
    if sill_modern <= 0.0:
        raise ValueError(
            f"the working raster reads {sill_modern:+.2f} m at the Manych saddle: a "
            "non-positive crest would mean the sill is a submarine channel and the "
            "spill has no threshold.  Check the raster and the sign convention "
            "before going further."
        )
    logger.info(
        "Manych saddle %.3fE/%.3fN reads %+.2f m from %s; MANYCH_SILL_M = %+.2f m, "
        "continuous-path controlling col %+.2f m.",
        MANYCH_SILL_LON,
        MANYCH_SILL_LAT,
        sill_modern,
        KHV_DEM.name,
        MANYCH_SILL_M,
        MANYCH_SILL_PATH_M,
    )

    report = sill_report(df, epochs, scenarios)

    out_base = Path(out_dir) if out_dir is not None else OUT_BASE
    report_path = out_base / SILL_REPORT_CSV.name
    # BEFORE the rasters, deliberately.  If the raster loop dies half way, the
    # rasters that did get written are described by a report that matches them; the
    # other order would leave fresh rasters beside the previous report, which is
    # exactly the mismatch this gate exists to prevent.
    if write_report:
        out_base.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)

    if write_rasters:
        out_base.mkdir(parents=True, exist_ok=True)
        for epoch in epochs:
            tags: dict[str, str] = {}
            for scenario in scenarios:
                threshold = caspian_threshold(df, epoch, scenario)
                test = sill_test(df, epoch, scenario)
                tags[f"threshold_{scenario}"] = threshold.provenance
                tags[f"sill_test_{scenario}"] = test.provenance
                tags[f"sill_verdict_{scenario}"] = test.verdict
            build_paleo_dem(
                epoch,
                gia_nan_policy=gia_nan_policy,
                provenance=tags,
                out_dir=out_base / "topo",
            )

    _print_report(report)
    print()
    for scenario in scenarios:
        window = spill_window_ka(report, scenario)
        low = spill_window_ka(report, scenario, verdict_column="verdict_envelope_lo")
        high = spill_window_ka(report, scenario, verdict_column="verdict_envelope_hi")
        high_clean = spill_window_ka(
            report, scenario, verdict_column="verdict_envelope_hi", drop_circular=True
        )
        if window is None:
            print(f"{scenario:>11}: never clears the sill on this grid")
        else:
            print(
                f"{scenario:>11}: spilling_west {window[0]:g} -> {window[1]:g} ka "
                "(F1 vs F1, +23.00 m point sill)"
            )
        def fmt(w: tuple[float, float] | None) -> str:
            return (
                "never clears the sill" if w is None else f"{w[0]:g} -> {w[1]:g} ka"
            )

        print(
            f"{'':>11}  ENVELOPE (one pair per curve ROW, all scenarios) "
            f"lo edge: {fmt(low)} | hi edge: {fmt(high)} "
            f"| hi edge, circular rows dropped: {fmt(high_clean)}"
        )
    circular_epochs = sorted(
        {float(e) for e in report.loc[report["envelope_hi_is_circular"], "epoch_ka"]}
    )
    if circular_epochs:
        print(
            f"{'':>11}  upper edge NOT independent evidence at "
            f"{'/'.join(f'{e:g}' for e in circular_epochs)} ka "
            "(unc_high stands on the Manych crest; see envelope_hi_circularity)"
        )
    print()
    try:
        print(sill_caption(report))
    except ValueError as exc:
        print(f"NO CAPTION: {exc}")
    print()
    if write_report:
        print(f"Sill report: {report_path}")
    else:
        print(
            f"Sill report NOT written (read-only query; pass --write to save it): "
            f"{report_path}"
        )
    if write_rasters:
        print(f"Paleo-DEMs:  {out_base / 'topo'}")
    return report


def _cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument(
        "--epochs",
        type=float,
        nargs="+",
        default=None,
        help="epochs (ka) to bake; default is the full 0.25-ka-over-17-13 grid",
    )
    parser.add_argument(
        "--sill-report-only",
        action="store_true",
        help=(
            "compute and PRINT the sill report, write no rasters (seconds, not "
            "hours).  Writes nothing unless --write is also given"
        ),
    )
    parser.add_argument(
        "--nan-policy",
        choices=BAKE_GIA_NAN_POLICIES,
        default=KHV_GIA_NAN_POLICY,
        help="what to do where the GIA correction is unknown (no 'zero' exists)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "force the sill report CSV to be written to $PALEO_OUT_ROOT.  Only "
            "needed with --sill-report-only: a run that writes RASTERS already "
            "refreshes the report, because the renderer reads its sill expectation "
            "out of that CSV and fresh rasters beside a stale report is a silent "
            "mismatch.  A read-only query writes nothing without this flag"
        ),
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help=(
            "ADDITIVE TAG PASS over rasters that are ALREADY BAKED: add the mesher "
            "tag set (honest nodata declaration in named units, three "
            "machine-readable basin water levels) in place.  Writes NO pixel and "
            "proves it — the band is sha256-ed before and after and a difference "
            "raises.  Idempotent, so it is safe to re-run.  ~2.5 min per epoch, "
            "~1.8 h for the full grid: LAUNCH IT DETACHED"
        ),
    )
    parser.add_argument(
        "--retag-dry-run",
        action="store_true",
        help=(
            "compute the mesher tags and report what WOULD change, writing neither "
            "tags nor manifest nor sidecar.  Still reads every band twice"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.retag or args.retag_dry_run:
        manifest = retag_all(epochs=args.epochs, dry_run=args.retag_dry_run)
        changed = int(manifest["tags_changed"].astype(bool).sum())
        print()
        print(
            f"Retagged {len(manifest)} raster(s); {changed} needed a tag write, "
            f"{len(manifest) - changed} were already current."
        )
        print(
            f"Pixel arrays unchanged: {int(manifest['pixels_unchanged'].sum())}/"
            f"{len(manifest)} (sha256 of band 1, before vs after)."
        )
        disagree = manifest.loc[~manifest["legacy_tag_agrees"], "epoch_ka"].tolist()
        if disagree:
            print(
                "gia_unknown_ice6g_cells disagrees with the recomputed model-cell "
                f"count at {len(disagree)} epoch(s): "
                f"{'/'.join(f'{e:g}' for e in disagree)} ka.  The legacy tag was left "
                "as it was; gia_unknown_ice6g_model_cells carries the recomputed one."
            )
        refused = int(manifest["pontic_levels_refused"].sum())
        print(
            f"Pontic water levels: {int(manifest['pontic_levels_resolved'].sum())} "
            f"resolved, {refused} refused with an explicit null and a measured reason."
        )
        if not args.retag_dry_run:
            print(f"Manifest: {RETAG_MANIFEST_CSV}")
        return
    main(
        epochs=args.epochs,
        gia_nan_policy=args.nan_policy,
        write_rasters=not args.sill_report_only,
        # None = follow write_rasters.  --write forces it on and never off, so the
        # flag can add a write and can never suppress the one a real bake owes.
        write_report=True if args.write else None,
    )


if __name__ == "__main__":
    _cli()
