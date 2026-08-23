#!/usr/bin/env python
"""The Khvalynian spill, 17.00 -> 13.00 ka, on TWO epoch grids.

WHAT THIS BUILDS
================
One frame per epoch of ``bake_khvalynian.EPOCHS_KA`` restricted to 13-17 ka,
rendered from ``$PALEO_OUT_ROOT/khvalynian_caspian/topo/khv_<epoch>ka.tif`` with
**three** declared basins, ONE colour mapping shared by every frame, and a manifest
from which any single frame can be audited without re-running anything.

TWO GRIDS, BOTH SHIPPED, NEITHER PRESENTED AS THE OTHER
=======================================================
The owner asked for 0.25 ka over 17-13 ka.  That is 17 epochs and EIGHT of them
cannot produce a Pontic level at all: the Pontic curve is on a 0.5 ka grid in this
window and carries no row at 16.75 / 16.25 / 15.75 / 15.25 / 14.75 / 14.25 / 13.75 /
13.25 ka, so ``bake_black_sea.basin_level`` raises ``CurveAgeGapError`` rather than
take a neighbour's value.  At 16.75 ka the two candidates are -58.3 m and -30.0 m,
28.3 m apart, and "nearest" would be decided by DataFrame row order.  That guard is
correct, it is NOT widened here, and the missing rows are NOT interpolated: closing
that gap is curve science, not a call-site tolerance.

So both grids ship, each saying in its own manifest what the other is:

===========================  =======  ==============================================
grid                         slots    what it is
===========================  =======  ==============================================
``step_ka=0.5``  (9 epochs)  9        **the deliverable.** Every frame is real
                                      imagery; nothing is a placeholder.
``step_ka=0.25`` (17 slots)  17       **the honest picture of what the data
                                      supports.** The same 9 rendered frames, byte
                                      for byte, plus the 8 unresolvable epochs kept
                                      as hatched refusal frames in their own slots.
===========================  =======  ==============================================

They are built from ONE render pass, not two.  :func:`write_subsequence` re-encodes
the 17-slot run's own canvas frames, verifying each by sha256 against what that run
recorded, so "the imagery is the same" is a fact of construction and a hash rather
than a claim.  ``manifest.grid`` on each names its step, its slots, the epochs it
does not carry, and the other deliverable by path.

Deliverables under ``$PALEO_ANIM_ROOT`` (default
``Y:/paleo/renders/khvalynian_spill_animation``), one subdirectory per grid:

``frames/paleo_render_<epoch>ka.png``   the renderer's own frames + its two sidecars
``canvas/frame_<nn>_<epoch>ka.png``     every frame padded to ONE pixel size
``khvalynian_spill.gif``                the animation, ONE 256-entry colour table
``khvalynian_spill.webp``               the SAME frames, LOSSLESS and 24-bit -- the
                                        escape from the GIF's 256-colour ceiling,
                                        proved by decoding it back frame by frame
``sequence_manifest.json``              per frame: levels, rules, sills, verdict, hashes
``sequence_limits.json``                the pinned mapping + which epoch produced each extremum
``master_palette.json``                 the 256 colours every frame is mapped through,
                                        what they cost, and WHICH QUANTIZER and why
``run.log``                             append-only progress, so a died run is auditable

A sub-sequence built with ``--subsequence-of`` additionally COPIES the canvas PNGs
it re-encodes into its own ``canvas/`` (9 frames = 22.2 MB, measured), so the
deliverable directory can be moved on its own without losing its source images.

THE FIVE THINGS THAT HAVE ALREADY BITTEN THIS PROJECT, AND WHERE EACH IS HANDLED
===============================================================================
1. **Per-frame colour limits, in the RENDERER.**  Adjacent epochs derived their own
   land/depth/ice/hillshade limits, so the animation shimmered where the terrain had
   not moved (measured by P-PALETTE: land p2 +34.902 -> +43.287 m between 16.5 and
   16.0 ka, i.e. the same elevation drawn up to 4/255 differently).  Handled by
   :func:`sequence_limits_for` -> ``RenderConfig.sequence_limits``, with
   ``clip_policy="raise"``: a frame holding a cell outside the measured envelope
   refuses instead of being clipped into agreement.
1b. **Per-frame colour tables, in the ENCODER -- the same defect one layer down.**
   Pinning eight colour limits in the renderer buys nothing if the GIF writer then
   re-quantizes each frame on its own, and until 2026-08-04 it did.  Measured by
   parsing the shipped ``khvalynian_spill.gif``'s own image descriptors: **16 of its
   17 frames carried a LOCAL 256-entry colour table** (only frame 0 used the global
   one), and adjacent local tables shared as few as **103 of 256** entries -- so the
   same RGB triple was written to a different index, and a different RGB triple to
   the same index, from one epoch to the next.  Two lines of
   ``PIL.GifImagePlugin`` produce it and both are reachable from
   ``imageio.v3.imwrite(..., extension=".gif")``, which passes no palette:
   ``_normalize_mode`` does ``im.convert("P", palette=Image.Palette.ADAPTIVE)`` per
   frame, and ``_write_multiple_frames`` then sets
   ``encoderinfo["include_color_table"] = True`` for every delta frame *because*
   ``palette`` is ``None``.  Handled by :func:`build_master_palette` (ONE
   quantization over the whole sequence), :func:`map_frame_through` (every frame
   mapped through that one palette, never re-derived) and a write that hands PIL the
   palette explicitly -- and then by :func:`assert_one_global_palette`, which parses
   the written file's blocks back out and refuses a local colour table.  The write is
   no more trusted than the frame delay is.
2. **``imageio`` duration is MILLISECONDS.**  On the installed 2.37.2,
   ``duration=0.25`` writes a frame delay PIL reads back as ``None, 0, 0`` --
   silently, with no warning and a valid GIF.  :func:`write_gif` writes the value
   and then :func:`read_frame_durations_ms` READS IT BACK with PIL and
   :func:`assert_frame_durations_ms` asserts it.  The write is never trusted.
3. **Frames of differing pixel size.**  ``render_paleocostline_epoch`` saves with
   ``bbox_inches="tight"`` (see ``SEQUENCE_LIMITS_FIGURE_SIZE_NOTE`` in the
   renderer), so its output size is a function of the widest line of text in the
   figure.  That file is not writable from here, so the frames are normalised
   HERE: :func:`pad_to_canvas` centres every frame on one canvas filled with the
   figure's own background, and :func:`assert_uniform_size` refuses a sequence
   that is not uniform after it.  The per-frame padding is recorded in the
   manifest, so "nothing moved" is a number and not a hope.
4. **A refused epoch dropped or interpolated.**  Dropping compresses time;
   interpolating invents a level.  :func:`refusal_frame` emits a visibly hatched
   placeholder that keeps the epoch's slot and its full duration and carries
   ``{"threshold": null, "refusal": "<class>: <message>"}``.

WHAT THE MANIFEST HAS TO CARRY, AND WHAT IT WAS MISSING UNTIL 2026-08-04
========================================================================
The manifest exists so that any single frame can be audited without re-running
anything.  Three things it recorded nothing about, all found by reading the shipped
artifacts rather than the code:

* **Neither level curve was identified.**  Every raster was fingerprinted; the two
  CSVs that two of the three water surfaces are READ FROM were never named, pathed
  or hashed.  :func:`level_curve_provenance` now puts path + sha256 for both in
  ``manifest.level_curves``, plus a third entry stating that the Mediterranean has
  no curve and why.
* **Every raster path pointed into a discarded scratchpad.**  The run read a
  byte-verified local mirror under ``%TEMP%\\claude\\...\\<session-uuid>\\...``, and
  recorded only that, so seventeen fingerprints hung off seventeen paths that died
  with the session.  :func:`resolve_raster_source` puts the canonical
  :data:`CANONICAL_TOPO_DIR` path back **by sha256**, and says ``UNRESOLVED`` rather
  than guessing from a file name.
* **No sill verdict was recorded anywhere.**  The expectation was there and the
  measured saddle was there when a frame rendered; whether they agreed had to be
  inferred, in practice from the absence of a ``SillExpectationError`` -- which is
  equally consistent with the guard never having run, and on the 2026-08-03
  deliverable that is exactly what happened.  :func:`_sill_verdict` now writes
  ``expected_m``, ``measured_saddle_m``, ``gap_m``, ``tolerance_m`` and a ``verdict``
  as VALUES on every pair of every frame, and ``NOT_MEASURED`` says out loud that an
  unrun guard is not a passed guard.

HOW BIG THE CLAIM IS, AND WHERE IT IS WRITTEN DOWN
==================================================
A ramp-mode frame draws an interpolated surface across the corridor, and the only
question a reader can ask of it is: **how much ground does that surface call water
that a flat surface calls land?**  Until 2026-08-04 nothing in the manifest, the
sidecars or the caption answered it.  It is now measured for every frame at write
time by :func:`ramp_new_wet_ground`, from that frame's own ``.basins.json``:

``frames[].ramp_new_wet_ground``      per frame, in cells, square kilometres and
                                      disjoint pieces, with the share of that
                                      frame's painted water
``manifest.ramp_claim``               the same per epoch, never summed across them
``frames[].disclosure_addendum.text`` the same numbers in the sentence DRAWN on the
                                      composed canvas, built from the same object

It is deliberately NOT named for the corridor, because
``painted_water.corridor_cells_restored_by_the_or`` already looks like it and is
about two orders of magnitude smaller -- that field counts only the cells the
de-speckle's opening removed and the corridor OR restored.  Both travel together in
every record, with the ratio between them computed, so neither can be quoted for the
other.  The claim is checked rather than argued: ``bed_min_m`` against
``receiver_level_m`` on the same frame says whether every corridor cell stands above
the receiver's flat surface, and a frame where it does not reports the count as an
upper bound and says why.

THE THIRD BASIN
===============
The frame 25.5-58 E / 36-54 N contains water that is neither Caspian nor Pontic:
the Aegean+Marmara and the Levantine.  Until 2026-08-03 they were painted at the
Pontic lake's surface because the priority flood labels that ground "pontic"; the
renderer now raises :class:`~paleoeurope.viz.paleocostline_render.UnseededWaterBodyError`
on them.  They are declared here as one basin, ``mediterranean``, at

    level(t) = -sea_surface_offset(t)          (frame F3)

which is DERIVED, not chosen.  The Mediterranean is connected to the world ocean,
and the vertical-datum contract's section 2 proves that for a cell ICE-6G treats as ocean
the exact land/sea companion of ``H_paleo`` is the scalar ``-offset(t)`` -- with no
assumption that the sea-surface change is spatially uniform, because whatever local
residual is baked into ``dz_engine`` cancels against the same scalar in the
threshold.  ``bake_khvalynian.sea_surface_offset`` is the function that returns it.

Cross-checked at manifest write time, every run, against a table that actually
prints three decimals -- ``2026-07-31-manych-sill-history.md``, measured there from
the same ICE-6G files with the same far-field-debiased estimator the bake uses:

    ka      offset(t) measured   reference     difference
    18.00      81.958069 m       81.958 m      +0.000069 m
    17.00      79.466736 m       79.467 m      -0.000264 m
    16.50      77.905823 m       77.906 m      -0.000177 m
    15.00      69.791992 m       69.792 m      -0.000008 m
    14.50      66.374115 m       66.374 m      +0.000115 m
    14.00      45.997406 m       45.997 m      +0.000406 m

Every difference is inside 0.0005 m, i.e. rounding in the printed figure.  **No
disagreement** -- and five of the six epochs are epochs this animation renders, so
the check lands on the sequence's own frames: the Mediterranean surface of each is
``-offset(t)``, and ``frames[].basin_levels.mediterranean.level_m`` should be the
negative of the measured column above, to the bit.

This replaced a block that claimed to compare against "the five figures the contract
prints".  The contract carries **no 21 ka row at all** (so its 87.332 had no source
in either repository), its offset column prints **two** decimals rather than three,
and its own 2026-07-31 amendment says *"do not quote a number out of this table"* --
see :data:`OFFSET_REFERENCE`.

WHAT THIS MODULE REFUSES, AND WHY EACH REFUSAL IS THE ANSWER
============================================================
* **A hand-written seed.**  Every ``BasinSeed`` comes from :data:`SEED_BOXES`
  through :func:`resolve_seed`, which is a deterministic argmin over a NAMED box on
  the epoch's own raster and records the box, the cell, the elevation, the raster
  sha256 and the epoch in the manifest.  See :class:`SeedBox` for why this is
  load-bearing for the PICTURE even though B1 measured the spill VERDICT to be
  seed-independent.
* **An invented crest.**  Of the three basin pairs, one has an independent
  measurement (``caspian<->pontic``, :func:`manych_expectation`) and two are
  declared :class:`~paleoeurope.viz.paleocostline_render.UnmeasuredSill` with the
  reason written out -- including, for the Bosphorus, the crest this session DID
  measure and the reason it cannot be used as a guard on this grid.  See
  :data:`BOSPHORUS_CREST_F0_M`.
* **A substituted level.**  The Pontic curve carries no row at the eight
  quarter-ka epochs of this window (16.75 / 16.25 / 15.75 / 15.25 / 14.75 / 14.25 /
  13.75 / 13.25 ka) and ``basin_level`` refuses to take a neighbour's -- at 16.75
  the two candidates are -58.3 m and -30.0 m, 28.3 m apart, and "nearest" would be
  decided by DataFrame row order.  Those eight epochs become REFUSAL FRAMES on the
  0.25 ka grid.  They are not interpolated and they are not dropped -- and the
  0.5 ka deliverable does not quietly become "the 0.25 ka animation" by leaving them
  out: it is named for its own step and its manifest names all eight.

THE THIRD STATE: WHAT THE MASK NOW REFUSES TO DECIDE
====================================================
Two revisions ago every three-basin frame of this sequence refused outright on
``UnseededWaterBodyError``.  One revision ago it rendered, because ``assign_basins``
kept the components a seed falls in and DROPPED the rest -- and that is the sentence
being retired here, because dropping a body is not a neutral act.  A component with
no seed is one of two quite different things, and the retired rule collapsed them:

* its rim may lie wholly inside the extent, above the surface it carries.  Then this
  raster DOES observe a barrier, "not water" is a measurement, and drawing it as land
  states what the terrain says.  That is **CLOSED**.
* or it may reach the edge of what this raster can observe -- an array side, or a
  nodata cell.  Then the route that would decide it runs outside the frame, and this
  grid cannot say whether the body is connected to anything.  That is
  **UNOBSERVABLE**, and drawing it as land asserts a barrier nobody observed, exactly
  as painting it would assert a connection nobody observed.

``assign_basins`` now returns all three states and this module carries all three into
the manifest (``frames[].spill.connectivity``, ``frames[].spill.unobservable_drawn``,
and the per-sequence roll-up in ``connectivity_states``).  UNOBSERVABLE cells leave
BOTH ``ocean_mask`` and ``land_mask`` and are drawn in a reserved hatched violet,
unhillshaded, with a legend swatch and a caption line -- so a reader of the PNG alone
cannot read them as either sea or land.

WHAT THAT IS ON THE PRODUCTION FRAME
------------------------------------
Read out of ``spill_0.25ka_17_slots/frames/paleo_render_16.5ka.basins.json``
(``khv_16.5ka.tif`` at ``max_px=7800``, ``corridor="step"``, three seeds, at the
surfaces caspian +29.94439 m / pontic -18.02037 m / mediterranean -77.90582 m):

    state          components        cells             km2   drawn as
    CONNECTED               3    8,436,006    1,310,646.85   this basin's water
    UNOBSERVABLE            2       64,117       11,012.08   hatched violet
    CLOSED                179      140,597       22,378.66   land
    ------------------------------------------------------------------------------
    flat mask             184    8,640,720

The three rows ADD to the flat mask in both columns -- that is the identity
``connectivity_states.partition_is_exact_on_every_frame`` asserts on every frame of
both grids, and it is why "there are three states here" is arithmetic rather than a
claim.

The UNOBSERVABLE bodies, each with the edge it reaches and the surface it sits under:

       51,880 cells      9,006.02 km2  south        34.5146 E / 36.6104 N  under -77.90582 m (mediterranean)
       12,237 cells      2,006.06 km2  west         26.6729 E / 40.6021 N  under -77.90582 m (mediterranean)

The largest CLOSED bodies, for contrast -- none of them touches an observation
boundary, which is precisely why they stay land:

       35,027 cells      5,611.46 km2  rim inside the extent  57.3521 E / 42.3729 N  under +29.94439 m (caspian)
       32,096 cells      5,229.57 km2  rim inside the extent  28.1687 E / 40.9688 N  under -77.90582 m (mediterranean)
       24,282 cells      3,830.65 km2  rim inside the extent  54.0271 E / 43.4104 N  under +29.94439 m (caspian)

WHAT CHANGED IN THE PICTURE, AND WHAT DID NOT
---------------------------------------------
Against ``_RETIRED_PRE-UNOBSERVABLE_2026-08-04_spill_0.25ka_17_slots/frames/paleo_render_16.5ka.basins.json``, which is the
same raster at the same resolution with the same seeds and the same levels and is
kept beside this run for exactly this comparison:

* the flat mask is the same (8,640,720 cells) and the connected mask
  is the same (8,436,006 cells);
* that run's single ``dropped_cells`` field held 204,714 cells /
  33,390.74 km2, ALL of them painted as dry land.  This run splits that
  same set into 64,117 UNOBSERVABLE and 140,597 CLOSED --
  they sum to the retired total, which is what makes this a reclassification and not
  a re-measurement;
* so 64,117 cells / 11,012.08 km2 stopped being drawn as land.
  Everything else about the water is untouched: ``painted_water.cells`` is
  8,431,654 in both runs.

``unobservable_drawn`` records what the figure actually painted in the reserved
colour, 64,117 cells against 64,117 classified,
so the difference the morphological closing makes is visible rather than reconciled
silently, together with the stripe period the frame used (30
raster cells here).  The period is scaled by the frame's own downsampling: fixed in
raster cells it averages away under the figure's resampling and the region arrives at
the reader as a flat unusual colour, which is the half of the signal that says
"annotation, not terrain".

Neither number was worked around.  ``UnseededWaterBodyError`` was not loosened, no
size threshold and no ``ceiling_margin`` was added, and no level was invented for any
of these bodies.  There is deliberately no ``"drop"`` spelling of
``unobservable_policy``: the two arms are ``"mark"`` (what this sequence ships) and
``"refuse"`` (no frame at all), because dropping is the behaviour being removed and
offering it as a setting is how it comes back.

WHAT REFUSES NOW, AND WHY IT IS SHIPPED AS A MARK RATHER THAN A REFUSAL
=======================================================================
With the mask corrected the native 16.5 ka frame reaches the colour-limits check and
refuses there instead, on ``SequenceLimitsExceededError``:
``shade_above_shade_max=1`` -- **one cell of 33,696,000**, and the limits it breaches
were computed over that same epoch, so no recomputation can satisfy it.

That is not a data excursion, and :func:`measure_analytic_shade_excursion` is what
says so rather than an argument here.  ``SequenceLimits.shade_max`` is +1.0 from
``analytic_shade_range``: it is GEOMETRY -- the largest value ``n . d`` can take for a
unit surface normal and a unit light direction -- and at native resolution the
terrain ATTAINS it.  Measured on ``khv_16.5ka.tif`` at ``max_px=7800``,
``vert_exag=8``, azimuth 315 deg, altitude 45 deg, the largest raw intensity is
**1.000000029629851** at 37.081250 E / 40.956250 N, i.e. **2.96e-08 above +1.0, which
is 0.25 of one float32 epsilon** (``np.finfo(np.float32).eps`` = 1.1920929e-07).
Below the range nothing is close: the smallest raw intensity is -0.577424 against an
analytic floor of -0.707107, so ``shade_below_shade_min`` is 0.

A bound that a theorem says cannot be exceeded, exceeded by a quarter of a float32
ulp, is arithmetic.  So this module renders with ``clip_policy="mark"`` -- the cell
is painted with the reserved out-of-range colour and COUNTED in the frame's own
``.palette.json`` -- and then applies its own per-quantity check,
:func:`assert_only_the_analytic_shade_ceiling_was_marked`, which is strictly harder
than ``clip_policy="raise"`` on everything that is actually measured:

* ``land_elevation_*``, ``depth_above_*`` and ``ice_thickness_above_*`` must be
  **zero** on every frame.  Those limits ARE measured over the sequence's rasters, so
  a breach is real information about the frame and is refused, exactly as ``"raise"``
  would have refused it.
* ``shade_above_shade_max`` may be non-zero, and every epoch it is non-zero on has its
  excursion MEASURED on its own raster in the same run and written into the manifest.
  An excursion larger than one float32 epsilon refuses the sequence.

The cost of the difference is small and it is NOT a per-frame constant.  Measured on
all nine rendered epochs of this sequence, **4 cells in total** -- one each at 16.5,
16.0, 14.0 and 13.5 ka, none at 17.0, 15.5, 15.0, 14.5 or 13.0 ka.  The pattern is
what a rounding tail looks like: every epoch's largest raw intensity sits at the
ceiling to seven decimals, and the four that cross it do so by 2.96e-08 (16.5 ka) or
8.56e-09 (16.0 / 14.0 / 13.5 ka) -- 0.249 and 0.072 of a float32 epsilon.  The five
that do not cross stop at 0.9999999875 or 0.9999999453.  Every frame's floor
clearance is 0.129683 or 0.129682, i.e. eleven million float32 epsilons clear of
``shade_min``.  The per-epoch counts are in
``manifest.frames[].clipping.out_of_range.by_quantity``, the per-epoch excursions in
``manifest.frames[].analytic_shade_ceiling``, and the total in
``manifest.analytic_shade_ceiling``; do not read a rate off this paragraph.

**And the reserved colour does not become a visible pixel.**  The mask is applied to
the composite at DEM resolution, but ``imshow`` then draws the mosaic into an axes a
few times smaller, so one marked cell is a small fraction of one screen pixel's
colour and survives no rounding.  Measured on this run's own
``frames/paleo_render_16.5ka.png`` (2964 x 1273 px): the reserved RGB
(255, 0, 255) occupies exactly 539 pixels, and every one of them is inside
the single rectangle rows 1165-1175, columns 1355-1403 -- which is 11 x 49 =
539, i.e. a SOLID rectangle with nothing left over.  That rectangle is the
LEGEND swatch, beside the words "ABOVE every value the limits measured".  Zero pixels
anywhere inside the map.  So the mark reaches a reader through the legend line and the
sidecar count, and not through the picture.  That is worth knowing before anyone
concludes from a clean-looking frame that nothing was marked.

Do not read the pixel geometry above as a constant.  It is a property of THIS frame:
``bbox_inches="tight"`` sizes the figure from the widest line of text in it, so a
frame carrying one more caption line is a different size and the swatch is at
different rows.  What is invariant is the arithmetic -- count equals the area of one
rectangle -- and that is what a reader should re-run, on whichever frame they have.

USAGE
=====
    # 1. the render pass: 17 slots, 9 rendered + 8 refusals kept in place
    python scripts/animate_khvalynian_spill.py \\
        --out-root Y:/paleo/renders/khvalynian_spill_animation/spill_0.25ka_17_slots

    # 2. the deliverable, re-encoded from that run's own verified canvas frames
    python scripts/animate_khvalynian_spill.py \\
        --subsequence-of Y:/paleo/renders/khvalynian_spill_animation/spill_0.25ka_17_slots \\
        --grid-step-ka 0.5 \\
        --out-root Y:/paleo/renders/khvalynian_spill_animation/spill_0.5ka_9_epochs

    # 3. the same pass with a second, tighter panel composed into every canvas.
    #    --inset takes the NAME of a registered extent; INSET_EXTENTS is the list, and
    #    each entry carries its own seed boxes, its own per-bound reason and its own
    #    sill guard on its own grid. MANYCH_CORRIDOR_TIGHT (36.4-46 E / 44.6-47.6 N)
    #    is the corridor-as-the-frame extent: it draws the same corridor as the wider
    #    MANYCH_INSET -- 15,274 cells against 15,286 on the 16.5 ka probe, 0.08 %
    #    apart, at the same saddle and the same crest cell -- at 1.3646x the linear
    #    scale. What fixed each of its four bounds is in its own ``why``.
    python scripts/animate_khvalynian_spill.py --corridor ramp \\
        --inset manych_corridor_crest_kerch_col_and_the_nearest_all_epoch_water \\
        --out-root Y:/paleo/renders/khvalynian_spill_animation/spill_tight

    # 4. THE CLOSE-UP. The same nine epochs, the same corridor, the same rasters --
    #    and the corridor is the WHOLE PAGE. --panels inset drops the whole-scene
    #    panel from the canvas (it is still rendered and hashed under frames/full,
    #    and every verdict in the manifest is still that panel's; it is not shrunk
    #    into a locator, because a resampled panel would put colours no render
    #    produced into a canvas that is quantized through ONE master palette).
    #    --fig-size-in is the only lever this module has on the legend that sits
    #    inside the map: the legend is fixed in POINTS and the map in INCHES, so a
    #    bigger figure shrinks the one against the other AND adds pixels. Measured
    #    on the 16.5 ka corridor panel, 14.0 in -> 22.0 in:
    #        map 1794x562 -> 2940x920 px, 186.88 -> 306.25 px/deg of longitude
    #        legend 21.67 -> 8.11 % of the map, 27.68 -> 17.86 % of its water
    #        sill callout 58.08 % under the legend -> 0.00 %, and the measured
    #        crest cell goes from INSIDE the legend box to 97 rows above it
    python scripts/animate_khvalynian_spill.py --grid-step-ka 0.5 --corridor ramp \\
        --inset manych_corridor_crest_kerch_col_and_the_nearest_all_epoch_water \\
        --panels inset --fig-size-in 22 \\
        --out-root Y:/paleo/renders/khvalynian_spill_animation/spill_closeup

    python scripts/animate_khvalynian_spill.py --plan-only     # no render, no GIF
    python scripts/animate_khvalynian_spill.py --max-px 2000   # a probe, NOT a deliverable

``--max-px`` below 7800 is accepted only with ``--i-know-this-grid-cannot-resolve-
the-sill`` because the Caspian<->Pontic saddle reads +36.01 m at 2000 px against
+28.00 m at native, which is the difference between "endorheic" and "spilling".
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paleoeurope.viz.paleocostline_render import (  # noqa: E402
    BasinSeed,
    RenderConfig,
    SequenceEpoch,
    SequenceLimits,
    SillExpectation,
    UnmeasuredSill,
    compute_sequence_limits,
    declared_vertical_frame,
    render_paleocostline_epoch,
)

# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------

#: TWO SPELLINGS OF ONE FILE, AND WHICH ONE THE ARTIFACTS CARRY.
#:
#: ``Path.resolve()`` on Windows expands a mapped network drive to the UNC share it
#: points at, so ``Y:\paleo\renders\...`` comes back
#: ``\\192.168.88.250\Storage\paleo\renders\...``.  The 2026-08-03 manifests
#: therefore mixed the two: ``raster.path_as_read`` was drive-lettered (it is the
#: string the run was handed), while ``seeds[].raster_path``, ``canvas_frame``,
#: ``palette.master.built_over`` and ``level_curves[].path`` were UNC (they went
#: through ``resolve()``), and nothing said the two were the same file.
#:
#: **The verdict taken here: this is not a defect, and neither spelling is "the"
#: path.**  The UNC form names the real location -- the host and share the bytes
#: live on -- and survives being read on a machine that has no ``Y:``.  The drive
#: letter is a per-machine mapping, and it is what every other document of this
#: project writes down and what a human types.  A manifest that carried only one of
#: them would be either unreadable off this machine or unresolvable on it.
#:
#: So BOTH are recorded.  :func:`path_record` writes ``{as_written, resolved,
#: differ}`` wherever a reader will chase a file, and ``manifest["paths"]`` carries
#: the drive map measured at write time -- every mapped drive letter that appears in
#: the run's own paths, with the UNC target ``resolve()`` gave for it -- so any
#: single-spelling value elsewhere in the file can be converted either way.
PATH_SPELLING_DECISION = (
    "Path.resolve() on Windows expands a mapped drive to its UNC target, so "
    "Y:\\paleo\\... and \\\\192.168.88.250\\Storage\\paleo\\... are the SAME file. "
    "That is not a defect and neither spelling is privileged: the UNC form names "
    "the real location and survives being read on a machine with no Y: mapping; "
    "the drive letter is the per-machine mapping this project writes down "
    "everywhere else and the one a human types. Both are therefore recorded. "
    "'as_written' is the spelling the run was handed, 'resolved' is what "
    "Path.resolve() returned for it in the same call, and 'drive_map' below lists "
    "every mapped drive letter appearing in this run's paths with the UNC target "
    "measured for it at write time -- so a value elsewhere in this file that "
    "carries only one spelling can be converted with no guessing"
)


def path_record(path: Any) -> Optional[dict]:
    """``{as_written, resolved, differ}`` for one path, or ``None`` for ``None``.

    Both spellings, measured in the same call, for the reason in
    :data:`PATH_SPELLING_DECISION`.  ``resolved`` falls back to ``as_written`` when
    the filesystem cannot resolve it (a dead mirror path, a share that is offline),
    and ``differ`` then reads ``False`` for a reason the record states rather than
    for agreement -- so ``resolve_error`` is present exactly when that happened.
    """
    if path is None:
        return None
    as_written = Path(path)
    try:
        resolved = as_written.resolve()
        error = None
    except Exception as exc:  # noqa: BLE001 - an unresolvable path is data
        resolved = as_written
        error = f"{type(exc).__name__}: {exc}"
    record = {
        "as_written": str(as_written),
        "resolved": str(resolved),
        "differ": str(as_written) != str(resolved),
    }
    if error is not None:
        record["resolve_error"] = error
    return record


def drive_map(paths: Iterable[Any]) -> list[dict]:
    """Every mapped drive letter appearing in *paths*, with its UNC target.

    Measured at write time by resolving one path per drive letter -- never a table
    typed out here, because a drive mapping is machine state and a stale one written
    into a manifest is exactly the class of sentence this phase exists to remove.
    """
    seen: dict[str, dict] = {}
    for path in paths:
        if not path:
            continue
        text = str(path)
        if len(text) < 2 or text[1] != ":":
            continue
        letter = text[0].upper() + ":"
        if letter in seen:
            continue
        record = path_record(text) or {}
        resolved = str(record.get("resolved", ""))
        tail = text[2:].lstrip("\\/")
        target: Optional[str] = None
        if resolved.startswith("\\\\") and tail and resolved.endswith(tail):
            target = resolved[: len(resolved) - len(tail)].rstrip("\\")
        seen[letter] = {
            "drive": letter,
            "unc_target": target,
            "expands_to_a_unc_share": resolved.startswith("\\\\"),
            "measured_on": {"as_written": text, "resolved": resolved},
            "measured_by": (
                "pathlib.Path.resolve() at manifest write time, then the common "
                "tail of the two spellings removed -- so the target is derived from "
                "this run's own path, never a mapping table typed into the source"
            ),
        }
    return [seen[key] for key in sorted(seen)]


#: The tree of record: the root ``bake_khvalynian`` writes its rasters under and
#: the root this module writes its deliverables under.  ``Y:`` is a mapped drive on
#: this machine; :data:`PATH_SPELLING_DECISION` records what that means for a path
#: written into a manifest.
CANONICAL_RENDER_ROOT = Path(
    os.environ.get("PALEO_RENDER_ROOT") or r"Y:\paleo\renders"
)

#: Where the 17 epoch rasters are KEPT.  This is the store ``bake_khvalynian``
#: writes and the only path a reader can be sent to, so it is a separate constant
#: from :data:`KHV_TOPO_DIR`.
#:
#: WHY THE TWO ARE SEPARATE, WITH THE MEASUREMENT THAT FORCED IT
#: ------------------------------------------------------------
#: The 2026-08-03 run read a byte-verified local mirror because the SMB share
#: measured 0.4-3.4 MB/s against ~3 GB of reads, and the manifest then recorded
#: seventeen paths of the form
#: ``%TEMP%\\claude\\<project>\\<session-uuid>\\scratchpad\\sanim\\mirror_root\\...``
#: -- a per-session agent scratchpad, i.e. a directory designed to be discarded.
#: Every raster fingerprint in that manifest was therefore attached to a path that
#: was dead on arrival.  Reading from a mirror is legitimate and nothing about the
#: DATA changes; recording only the mirror is not.  :func:`resolve_raster_source`
#: puts the canonical path back, and it does it BY SHA256 rather than by name.
#: Deliberately NOT ``.resolve()``d: on Windows that expands the mapped drive to
#: its UNC target, and the path a reader is given should be the one the project
#: writes down everywhere else (the mirror sidecar, the bake, this brief).  The
#: comparison in :func:`resolve_raster_source` resolves both sides, so a run that
#: reads through either spelling still matches, and :func:`path_record` writes both.
CANONICAL_TOPO_DIR = Path(
    os.environ.get("PALEO_SOURCE_TOPO_DIR")
    or (CANONICAL_RENDER_ROOT / "khvalynian_caspian" / "topo")
)

#: Root the bake wrote its rasters under, when a run overrides it.  ``None`` when
#: ``PALEO_OUT_ROOT`` is unset, which is the ordinary case and the one the default
#: below turns into the tree of record rather than into a guess.
PALEO_OUT_ROOT: Optional[Path] = (
    Path(os.environ["PALEO_OUT_ROOT"]).resolve()
    if os.environ.get("PALEO_OUT_ROOT")
    else None
)

#: Where the 17 epoch rasters are READ from on this run.
#:
#: THE DEFAULT IS THE TREE OF RECORD, AND IT WAS NOT UNTIL 2026-08-04
#: -----------------------------------------------------------------
#: This used to read ``$PALEO_OUT_ROOT/khvalynian_caspian/topo`` with
#: ``PALEO_OUT_ROOT`` **defaulting to the repository's parent directory**, i.e.
#: ``D:\pasha\RESEARCH\renders\khvalynian_caspian\topo``.  That directory holds
#: **2 of the 17 rasters** (``khv_16.5ka.tif`` and ``khv_16ka.tif``), so a run with
#: no environment set planned a two-frame sequence and called it the 13-17 ka
#: window -- and the two dataset-gated acceptance tests of this module failed for
#: that reason and no other.  The tree of record is :data:`CANONICAL_TOPO_DIR`,
#: which holds all 17 baked with the current code.  ``PALEO_OUT_ROOT`` still wins
#: when it is set, because that is how a mirror run is pointed somewhere else; what
#: changed is that the FALLBACK is now the store the rasters actually live in
#: instead of a directory that merely resembles it.
KHV_TOPO_DIR = (
    (PALEO_OUT_ROOT / "khvalynian_caspian" / "topo")
    if PALEO_OUT_ROOT is not None
    else CANONICAL_TOPO_DIR
)

#: The sidecar a byte-verified mirror leaves behind, searched for beside the
#: rasters and in the animation's own output root.  It carries, per file, the
#: SOURCE path, the MIRROR path and the sha256 that was computed on both and
#: agreed -- which is exactly what is needed to put a canonical path back on a
#: fingerprint without trusting a file name.
MIRROR_PROVENANCE_NAME = "raster_mirror_provenance.json"

#: Everything this script writes.  Deliberately NOT on D:, which had 710 MB free
#: when this was written; one native frame plus its GIF does not fit there.
#:
#: **The sentence above used to be false of the code under it.**  Until 2026-08-04
#: the default was ``PALEO_OUT_ROOT / "khvalynian_spill_animation"`` with
#: ``PALEO_OUT_ROOT`` defaulting to the repository's parent -- i.e. a run with no
#: environment set wrote its frames, its GIF and its WebP to
#: ``D:\pasha\RESEARCH\renders\khvalynian_spill_animation``, the disk this comment
#: says it deliberately avoids.  The default is now derived from
#: :data:`CANONICAL_RENDER_ROOT` and ``PALEO_ANIM_ROOT`` still overrides it.
#:
#: Not ``.resolve()``d, for the reason in :data:`PATH_SPELLING_DECISION`: both
#: spellings are recorded, and the one written first is the one a human types.
ANIM_ROOT = Path(
    os.environ.get("PALEO_ANIM_ROOT")
    or (CANONICAL_RENDER_ROOT / "khvalynian_spill_animation")
)

#: The window, taken from the bake's own grid rather than typed out, so the two
#: cannot drift.  ``bake_khvalynian`` samples 13-17 ka at 0.25 ka; this is exactly
#: those 17 epochs.
FINE_WINDOW_KA = (13.0, 17.0)

#: The finest grid the bake wrote rasters for inside :data:`FINE_WINDOW_KA`.  The
#: owner's request is on this step.
FINE_STEP_KA = 0.25

#: The step every epoch of which has a Pontic curve row, so every frame is imagery.
#: NOT a rounding of the request and NOT a repair of the curve -- see
#: :data:`PONTIC_GAP_RULE`.  It is a SELECTION out of the same baked grid.
HALF_KA_STEP = 0.5

#: Why eight of the seventeen quarter-ka epochs carry no frame, written once and
#: shipped in both manifests.  Every number in it is read out of the two curve CSVs
#: :func:`level_curve_provenance` hashes, at the epochs it names.
PONTIC_GAP_RULE = (
    "the Pontic level curve is on a 0.5 ka grid across this window and carries no "
    "row at 16.75 / 16.25 / 15.75 / 15.25 / 14.75 / 14.25 / 13.75 / 13.25 ka, so "
    "bake_black_sea.basin_level raises CurveAgeGapError instead of taking a "
    "neighbour's value. At 16.75 ka the two candidates are -58.3 m and -30.0 m, "
    "28.3 m apart, and 'nearest' would be decided by DataFrame row order. The guard "
    "is NOT widened and the rows are NOT interpolated: 28.3 m of Pontic surface is "
    "curve science, not a call-site tolerance, and an interpolated row would be "
    "indistinguishable downstream from a dated one. The Caspian curve DOES carry "
    "all seventeen; the Mediterranean needs none, being -offset(t) from ICE-6G_C. "
    "So the gap is one basin's, and it is the receiving basin's"
)

#: Frame delay, MILLISECONDS.  See :func:`write_gif` for why the unit is shouted.
#: 250 ms is 25 centiseconds, and the GIF format stores centiseconds, so this value
#: survives the round trip exactly -- a delay like 205 ms would not.
FRAME_DURATION_MS = 250

#: The grid the two-basin frame is renderable on.  Not a default that can be
#: lowered quietly: at 2000 px the Caspian<->Pontic saddle on ``dem_sm`` reads
#: +36.005 m against the independently measured +27.701 m, an +8.30 m gap, and at
#: 4000 px +32.942 m (+5.24).  At 7800 px the gap is +0.30 m.
NATIVE_MAX_PX = 7800

#: The colour the frames are padded to a common canvas with.  It is the figure
#: background ``render_paleocostline_epoch`` saves on -- matplotlib's default white,
#: since its ``savefig`` passes no ``facecolor`` -- so the padding is invisible
#: rather than a grey border that reads as part of the picture.
CANVAS_PAD_RGB = (255, 255, 255)

#: Used only when NOT ONE frame rendered, so there is no measured size to pad to.
#: ``fig_size_in`` x ``dpi`` is the figure before ``bbox_inches="tight"`` trims it,
#: i.e. an upper bound on every frame this config can produce.
FALLBACK_CANVAS_PX = (
    int(round(RenderConfig().fig_size_in * RenderConfig().dpi)),
    int(round(RenderConfig().fig_size_in * RenderConfig().dpi)),
)


def fallback_canvas_for(fig_size_in: float, dpi: float) -> tuple[int, int]:
    """The un-trimmed figure box of THIS config, in pixels.

    :data:`FALLBACK_CANVAS_PX` is the same quantity for the renderer's own defaults
    and is kept because it is what a caller who passes no figure gets.  A run that
    raises the figure and keeps the default fallback would, on a sequence where not
    one frame rendered, declare a canvas SMALLER than the figure it asked for -- an
    upper bound that is not one.
    """
    return (
        int(round(float(fig_size_in) * float(dpi))),
        int(round(float(fig_size_in) * float(dpi))),
    )


#: THE FIGURE THE CLOSE-UP BUILD IS DRAWN INTO, and why it is not the renderer's
#: own 14.0 in / 150 dpi.
#:
#: ``render_paleocostline_epoch`` is off limits here, so the legend cannot be moved,
#: resized or dropped.  It does not have to be: the legend's box is sized in POINTS
#: and the map axes is sized in INCHES x DPI, so the fraction of the map the legend
#: covers goes as 1/fig_size_in and the pixels the map is drawn with go as
#: fig_size_in x dpi.  ONE knob therefore does both jobs the owner asked for --
#: fewer degrees per pixel AND a legend that stops eating the south-east corner.
#:
#: Measured on the 16.5 ka corridor panel, on the panel's own PNG (see
#: :func:`measure_legend_occlusion`, which is the same measurement the manifest
#: carries per frame):
#:
#:     fig_size_in   map axes px     px/deg lon   legend, % of map   water under it
#:        14.0       1794 x  562       186.88          21.67 %          27.68 %
#:        22.0       2940 x  920       306.25           8.11 %          17.86 %
#:
#: And what that is FOR, on the same two frames: at 14.0 in the legend box swallows
#: 58.08 % of the sill callout AND contains the measured crest cell itself
#: (44.110417 E / 45.735417 N lands at row 349 of a 562-row map whose legend starts
#: at row 311).  At 22.0 in the callout is 0.00 % covered and the crest sits 97 rows
#: -- 0.3163 deg -- above the legend's top edge.  The legend still covers water at
#: the bottom-right corner and always will, because that corner is where the Caspian
#: is: 17.86 % of this panel's water against 27.68 %.  That residue is DISCLOSED per
#: frame rather than declared solved.
#:
#: dpi is NOT raised with it.  Raising dpi multiplies text pixels and map pixels
#: alike, so it buys resolution and NO relief at all on the occlusion; raising
#: fig_size_in is the only one of the two that changes the ratio.  22.0 is chosen
#: rather than something larger because every disclosure in the frame -- the title
#: block, the legend entries, the sill callout -- is fixed in points, so a bigger
#: figure makes them smaller RELATIVE to the map, and a disclosure nobody can read
#: is a disclosure that is not made.  The occlusion is re-measured on every frame
#: this module writes and disclosed in the caption whether it improved or not.
CLOSEUP_FIGURE_SIZE_IN = 22.0
CLOSEUP_FIGURE_DPI = 150

#: Why a figure size is in this module's manifest at all.  Shipped, digit-free of
#: replacement fields, so it is prose and not a template.
FIGURE_LEVER_RULE = (
    "the legend, the title block and the sill callout are drawn INSIDE the map axes "
    "and sized in POINTS by render_paleocostline_epoch, which this module may not "
    "edit. The map axes is sized in inches x dpi. So the fraction of the map its own "
    "legend covers goes as one over fig_size_in and is independent of dpi, and "
    "fig_size_in is the ONLY lever this module has on it. Raising it makes the map "
    "bigger in pixels AND smaller in legend; it also makes every fixed-point "
    "disclosure smaller relative to the map, which is the cost. What it actually "
    "bought on each frame is measured on that frame's own PNG and recorded TWICE, "
    "in frames[].inset.legend_occlusion and in the sentence drawn on that frame's "
    "own caption band, which is built from the same object -- never assumed from "
    "the ratio."
)


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class MissingEpochRasterError(FileNotFoundError):
    """An epoch of the window has no baked raster.

    A neighbouring epoch's raster is NOT substituted.  The DEM carries the GIA
    field for its own epoch; borrowing the 16.5 ka raster for 16.25 ka would move
    every elevation in the frame by the 0.25 ka of solid displacement and label the
    result 16.25 ka.  Bake the epoch or let it refuse.
    """


class IncompleteRasterTreeError(FileNotFoundError):
    """The raster directory does not hold every epoch the sequence asked for.

    A partial tree is the one failure this module used to absorb quietly.  A
    missing raster became a per-epoch refusal frame, which is the right answer when
    ONE epoch was never baked -- and the wrong answer when the directory is simply
    the wrong directory, because the run then produced a short sequence, wrote a
    manifest describing it, and named it for the whole 13-17 ka window.  Measured:
    with the pre-2026-08-04 default of
    ``D:\\pasha\\RESEARCH\\renders\\khvalynian_caspian\\topo`` the tree held **2 of
    the 17** rasters and the run planned two frames without a word about it.

    So the DIRECTORY is checked before any epoch is planned, and the message
    carries the count found, the count needed, both spellings of the directory, and
    the epochs that are missing by name.
    """


class SeedBoxEmptyError(ValueError):
    """A named seed box holds no finite cell on this raster.

    The box is a statement about geography; if the raster has no data there, the
    statement cannot be resolved into a cell and no other cell will do.
    """


class SeedAboveItsLevelError(ValueError):
    """The argmin of a basin's own box is not under that basin's water.

    ``assign_basins`` refuses this too, and it is worth refusing twice: a seed on
    dry land grows an empty basin, so the frame silently loses a sea instead of
    failing.  Four of twelve plausible hand-written seeds tried on this project
    were caught by exactly this check.
    """


class BasinLevelUnavailableError(ValueError):
    """One of the three basins has no defensible level at this epoch.

    Carried rather than raised out of the run: the epoch becomes a refusal frame
    keeping its slot, because dropping it compresses time and interpolating it
    invents a level.

    ``resolved`` holds the levels that DID come out before the refusal.  At the
    eight quarter-ka epochs of this window the Caspian and the Mediterranean both
    resolve and only the Pontic does not, and throwing that away would make a frame
    say "no levels" when it has two of three -- a refusal that overstates itself is
    as wrong as one that understates.  It reaches the placeholder frame, its sidecar
    and the manifest.
    """

    def __init__(self, *args, resolved: Optional[dict] = None) -> None:
        super().__init__(*args)
        self.resolved: dict = dict(resolved or {})


class GifDurationError(AssertionError):
    """The delay written into the animation is not the delay asked for.

    ``imageio`` takes the duration in MILLISECONDS.  On the installed 2.37.2 a
    seconds-shaped value like ``0.25`` is written as a zero frame delay, silently,
    producing a valid GIF that plays as fast as the decoder can go.  Four of the
    five GIF builders in the sibling ``Paleo-Europe_3D/scripts`` tree are wrong in
    exactly this way today.  So the value is read back out of the file with PIL and
    compared, and the write is never trusted.

    **The writer changed on 2026-08-04 and this guard did not stop applying.**
    :func:`write_gif` no longer goes through ``imageio`` -- it hands PIL the frames
    and the palette directly, because that is the only path that keeps a single
    colour table -- and PIL has the SAME failure, measured on this machine through
    the new writer: ``duration=0.25`` comes back ``[None, None, None]`` (no delay
    at all, not even a zero), ``duration=250`` comes back ``[250, 250, 250]``, and
    ``duration=205`` comes back ``[200, 200, 200]`` because the format stores
    centiseconds.  Same guard, same reason, different library underneath it.
    """


class PaletteError(AssertionError):
    """The animation does not carry the ONE palette its frames were mapped through.

    Raised by :func:`assert_one_global_palette` after the file has been written and
    PARSED BACK, never from the writer's own return value.  The defect it exists to
    catch is not hypothetical: the 2026-08-03 deliverable shipped 16 local colour
    tables in 17 image descriptors, which reinstated a per-frame colour mapping on
    top of the pinned one -- the exact defect ``RenderConfig.sequence_limits``
    exists to remove, one layer downstream, invisible to every check that looked at
    the frames instead of at the bytes.

    It is also raised at BUILD time when the master palette cannot be made
    well-formed -- see :func:`build_master_palette` for why a repeated colour-table
    entry is a correctness problem and not a cosmetic one.
    """


class FrameSizeError(AssertionError):
    """The frames of a sequence are not all the same pixel size.

    ``render_paleocostline_epoch`` saves with ``bbox_inches="tight"``, so its output
    size follows the widest line of text in the figure; an animation built from
    those directly jitters.  :func:`pad_to_canvas` is the fix and this is its
    assertion.
    """


class ClipPolicyBreachError(AssertionError):
    """A frame was marked out-of-range for something that is actually MEASURED.

    This module renders with ``clip_policy="mark"`` for one reason only: the
    hillshade ceiling ``SequenceLimits.shade_max`` is +1.0 from
    :func:`~paleoeurope.viz.paleocostline_render.analytic_shade_range`, which is a
    geometric supremum the terrain ATTAINS at native resolution, so a single cell
    rounds through it in float32 and ``clip_policy="raise"`` aborts a frame against
    limits computed from that same frame.

    Every OTHER quantity in the out-of-range report is an extremum measured over the
    sequence's own rasters, so a non-zero count there is real information about the
    frame and must stop the run.  :func:`assert_only_the_analytic_shade_ceiling_was_marked`
    is that check, and it is strictly stronger than ``"raise"`` would have been on
    those quantities, because it also runs on frames ``"raise"`` would never have
    reached.
    """


class ShadeExcursionError(AssertionError):
    """The hillshade rose above its analytic ceiling by more than arithmetic can.

    ``n . d`` for a unit normal and a unit light vector cannot exceed +1: that is a
    theorem, not a measurement, so every cell above it is floating-point.  What is
    NOT arithmetic is an excursion larger than the precision of the arithmetic.
    :data:`SHADE_EXCURSION_BUDGET` is that bound, and this is what happens when a
    frame exceeds it -- the number is measured on that frame's own raster in the same
    run, never assumed from another epoch.
    """


class SubsequenceSourceError(ValueError):
    """A sub-sequence cannot be built from the run it was pointed at.

    :func:`write_subsequence` re-encodes frames another run already produced instead
    of rendering them again, so the ONE thing it must never do is re-encode a frame
    it cannot prove is the frame that run measured.  It refuses a missing manifest, a
    missing slot, a slot that never rendered, and -- the case this class exists for --
    a canvas PNG whose sha256 is not the sha256 the source run recorded for it.
    """


# ---------------------------------------------------------------------------
# THE SEED REGISTRY  (RAISE-not-compute item 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedBox:
    """The named region a basin's seed is the deterministic argmin of.

    WHY A BOX AND NOT A COORDINATE
    ------------------------------
    Before this registry the repository held **zero** provenanced ``BasinSeed``
    coordinates: every seed used in every measurement was invented by whoever
    needed one, and a lon/lat without the grid it was derived on can only be
    trusted, never re-derived.  A box plus "the lowest cell in it" is a rule -- it
    can be re-run on any raster and it produces the same cell or an explicit
    refusal.

    WHY THIS IS LOAD-BEARING EVEN THOUGH THE VERDICT IS SEED-INDEPENDENT
    -------------------------------------------------------------------
    B1 swept 1 953 seed pairs (31 Caspian x 63 Pontic, every tile-argmin below
    -200 m) over ``khv_16.5ka.tif`` at ``max_px`` 2000 / 4000 / 7800 and found the
    Caspian<->Pontic saddle invariant **to the last float64 bit**:
    ``+28.000944137573242 m`` at 44.110417 E / 45.735417 N, spread exactly 0.00.
    So the SPILL VERDICT is safe against a bad seed.

    **The PICTURE is not.**  Moving the Pontic seed to another cell of the same
    connected component changed the painted corridor by 2 383 005 cells -- a factor
    of 4.68.  Ownership is a minimax divide, and which side of a divide a given
    plateau falls on depends on where the flood started, so the blue in the image
    and the corridor mask in the sidecar both move with the seed even where the
    number in the caption does not.  A registry is therefore not paperwork for a
    quantity nobody uses; it is what makes the drawn water reproducible.

    Parameters
    ----------
    basin : str
        The basin name that reaches ``BasinSeed.name``, the sill keys, the figure
        and the sidecar.
    name : str
        The box's own name, recorded beside the resolved cell.
    lon_min, lat_min, lon_max, lat_max : float
        INCLUSIVE bounds, EPSG:4326, degrees.  This is the definition of record.
    why : str
        What the box is a statement about.  Never blank: the box is a geographic
        claim and a claim with no sentence beside it is indistinguishable from a
        rectangle chosen to make an argmin come out somewhere.
    """

    basin: str
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    why: str

    def __post_init__(self) -> None:
        if not (self.lon_min < self.lon_max and self.lat_min < self.lat_max):
            raise ValueError(
                f"SeedBox {self.name!r} bounds are not ascending: "
                f"lon {self.lon_min}..{self.lon_max}, lat {self.lat_min}..{self.lat_max}"
            )
        for field_name in ("basin", "name", "why"):
            text = getattr(self, field_name)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"SeedBox.{field_name} is {text!r} and has no default")

    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)``."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def to_dict(self) -> dict:
        return {
            "basin": self.basin,
            "name": self.name,
            "bounds_wsen": list(self.bounds()),
            "why": self.why,
        }


#: THE REGISTRY.  One box per basin of this scene, each a region the sources
#: establish rather than a rectangle drawn to reach a chosen cell.
#:
#: Every box was resolved on ``khv_16.5ka.tif`` at native resolution while it was
#: written.  Two of the three had hand-written counterparts in a previous phase and
#: the comparison is stated exactly, because "it reproduces them" would have been
#: half true:
#:
#: * ``tests/test_viz.py::_KHV_CASPIAN_SEED`` = 50.74375 / 38.95208 -- the rule
#:   returns **the same cell**: row 3611, col 6058, centre 50.743750 / 38.952083,
#:   -1048.861 m.
#: * ``_KHV_PONTIC_SEED`` = 35.28542 / 42.20208 -- the rule returns a **different
#:   and deeper cell**: row 2832, col 2347 (centre 35.281250 / 42.197917) at
#:   -2645.555 m, against row 2831, col 2348 at -2578.521 m.  One cell away
#:   diagonally, 67.03 m deeper.  Both lie in the same connected component, which is
#:   why B1's sweep found the SADDLE identical either way -- and it is exactly the
#:   size of difference that moves the PAINTED CORRIDOR, which B1 measured and found
#:   is not invariant.
#:
#: So this is the first derivation either coordinate has had, and for the Pontic it
#: is also a correction.
SEED_BOXES: tuple[SeedBox, ...] = (
    SeedBox(
        basin="caspian",
        name="caspian_south_basin",
        lon_min=49.0,
        lat_min=36.5,
        lon_max=54.0,
        lat_max=40.5,
        why=(
            "the South Caspian basin -- the deepest closed water in the scene "
            "(-1048.9 m on khv_16.5ka.tif) and the only part of the Caspian below "
            "-1000 m inside this extent. Every Khvalynian surface in 17-13 ka "
            "stands more than a kilometre above it, so the box cannot resolve to a "
            "cell that is dry at any epoch of the window"
        ),
    ),
    SeedBox(
        basin="pontic",
        name="pontic_abyssal_plain",
        lon_min=30.0,
        lat_min=42.0,
        lon_max=38.0,
        lat_max=44.5,
        why=(
            "the central Black Sea abyssal plain -- the deepest part of the Pontic "
            "basin (-2645.6 m on khv_16.5ka.tif). Bounded north of 42 N and south "
            "of 44.5 N to exclude the Sea of Azov, and east of 30 E to exclude the "
            "Marmara, both of which are separate water bodies at these levels and "
            "would make the seed's basin a different claim"
        ),
    ),
    SeedBox(
        basin="mediterranean",
        name="aegean_rhodes_basin",
        lon_min=25.5,
        lat_min=36.0,
        lon_max=29.0,
        lat_max=38.0,
        why=(
            "the Rhodes Basin of the south Aegean -- the deepest Mediterranean "
            "water inside this extent (-4450.6 m on khv_16.5ka.tif). The "
            "Mediterranean here is the world ocean, continuing past the frame's "
            "west and south edges; the box names the deepest piece of it that the "
            "frame actually contains rather than a point in a sea the raster cuts "
            "off"
        ),
    ),
)


@dataclass(frozen=True)
class ResolvedSeed:
    """One seed, and everything needed to re-derive it.

    A ``BasinSeed`` carries a lon/lat and a level and nothing else, which is enough
    for ``assign_basins`` and not enough for anybody reading the output six months
    later.  This is the object the manifest records.
    """

    basin: str
    box: SeedBox
    epoch_ka: float
    raster_path: str
    raster_sha256: str
    row: int
    col: int
    lon: float
    lat: float
    elevation_m: float
    level_m: float
    rule: str

    def to_basin_seed(self) -> BasinSeed:
        return BasinSeed(
            name=self.basin, lon=self.lon, lat=self.lat, level_m=self.level_m
        )

    def to_dict(self) -> dict:
        return {
            "basin": self.basin,
            "box": self.box.to_dict(),
            "epoch_ka": float(self.epoch_ka),
            "raster_path": self.raster_path,
            "raster_sha256": self.raster_sha256,
            "row": int(self.row),
            "col": int(self.col),
            "lon": float(self.lon),
            "lat": float(self.lat),
            "elevation_m": float(self.elevation_m),
            "level_m": float(self.level_m),
            "rule": self.rule,
        }


#: The rule, in one sentence, recorded on every resolved seed.
SEED_RULE = (
    "argmin of the raster's own elevation over the named box, read at FULL "
    "resolution through a rasterio window (no decimation, no resampling: a "
    "decimated read moved the extremes of this project's rasters by up to 385 m), "
    "ties broken by the first occurrence in C order, which is deterministic for a "
    "given raster"
)


def sha256_of(path: Path | str) -> str:
    """The file's sha256, in 1 MiB blocks.

    No ``"unknown"`` sentinel and no opt-out: a fingerprint that cannot be checked
    is a path, and this project re-bakes rasters in place under unchanged names, so
    a matching path is not a matching scene.  Costs ~1.2 s for the 42 MB production
    raster.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_seed(
    box: SeedBox,
    raster_path: Path | str,
    *,
    level_m: float,
    epoch_ka: float,
    raster_sha256: Optional[str] = None,
) -> ResolvedSeed:
    """Deterministic argmin of *box* on *raster_path*, refused if it is not wet.

    Raises
    ------
    SeedBoxEmptyError
        The box holds no finite cell on this raster.
    SeedAboveItsLevelError
        The lowest cell of the box still stands at or above the basin's own water
        surface at this epoch.  This is the refusal that catches a box naming the
        wrong sea, a level in the wrong vertical frame, and an epoch at which the
        basin this box describes does not exist.
    """
    import rasterio
    from rasterio.windows import from_bounds

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise MissingEpochRasterError(
            f"seed box {box.name!r} cannot be resolved: {raster_path} does not exist"
        )
    with rasterio.open(raster_path) as dataset:
        window = from_bounds(*box.bounds(), dataset.transform)
        block = dataset.read(1, window=window).astype("float64")
        nodata = dataset.nodata
        if nodata is not None:
            block = np.where(block == nodata, np.nan, block)
        row_off = int(round(window.row_off))
        col_off = int(round(window.col_off))
        finite = np.isfinite(block)
        if not finite.any():
            raise SeedBoxEmptyError(
                f"seed box {box.name!r} ({box.bounds()}) holds no finite cell on "
                f"{raster_path.name}. The box is a statement about geography and "
                "this raster does not carry that ground; no other cell substitutes "
                "for it."
            )
        flat = np.where(finite, block, np.inf)
        index = int(np.argmin(flat))
        local_row, local_col = np.unravel_index(index, block.shape)
        row = row_off + int(local_row)
        col = col_off + int(local_col)
        elevation = float(block[local_row, local_col])
        lon, lat = dataset.transform * (col + 0.5, row + 0.5)

    if not (elevation < float(level_m)):
        raise SeedAboveItsLevelError(
            f"the lowest cell of seed box {box.name!r} on {raster_path.name} is "
            f"{elevation:+.3f} m, which is not below basin {box.basin!r}'s level of "
            f"{float(level_m):+.3f} m at {float(epoch_ka):g} ka. A seed on dry land "
            "grows an empty basin and the frame loses a sea without failing. Either "
            "the box names a body this epoch does not have, or the level is in the "
            "wrong vertical frame."
        )

    return ResolvedSeed(
        basin=box.basin,
        box=box,
        epoch_ka=float(epoch_ka),
        raster_path=str(raster_path.resolve()),
        raster_sha256=raster_sha256 or sha256_of(raster_path),
        row=row,
        col=col,
        lon=float(lon),
        lat=float(lat),
        elevation_m=elevation,
        level_m=float(level_m),
        rule=SEED_RULE,
    )


def resolve_seeds(
    raster_path: Path | str,
    levels_m: dict[str, float],
    *,
    epoch_ka: float,
    raster_sha256: Optional[str] = None,
    boxes: Sequence[SeedBox] = SEED_BOXES,
) -> tuple[ResolvedSeed, ...]:
    """Every registered box resolved against one epoch's raster."""
    missing = [b.basin for b in boxes if b.basin not in levels_m]
    if missing:
        raise BasinLevelUnavailableError(
            f"no level supplied for {missing!r} at {float(epoch_ka):g} ka; the "
            "registry has a box for each and a box cannot be checked against a "
            "level that is not there."
        )
    digest = raster_sha256 or sha256_of(raster_path)
    return tuple(
        resolve_seed(
            box,
            raster_path,
            level_m=levels_m[box.basin],
            epoch_ka=epoch_ka,
            raster_sha256=digest,
        )
        for box in boxes
    )


# ---------------------------------------------------------------------------
# THE INSET  --  a second, tighter extent on the same terrain
# ---------------------------------------------------------------------------


class InsetExtentError(ValueError):
    """The inset extent does not contain something it is required to contain."""


class InsetCropError(ValueError):
    """The cropped raster is not a faithful window of the raster it names."""


@dataclass(frozen=True)
class InsetExtent:
    """A named sub-extent of the scene, its seed boxes, and why its bounds are there.

    WHY THIS IS A REGISTRY ENTRY AND NOT A LAYOUT CHOICE
    ---------------------------------------------------
    An inset is a second grid.  Everything :class:`SeedBox` exists to stop applies to
    it twice over: its seeds are argmins over a different array, its saddle is a
    minimax over a different graph, and its water is a flood over a different domain.
    A rectangle typed to make a picture look right, with seeds typed inside it, would
    reproduce -- on the very frame this phase exists to make legible -- the defect
    that RAISE-not-compute item 6 was closed to remove: a coordinate that was typed
    is indistinguishable downstream from one that was measured.

    So the bounds are DECLARED (a framing decision must be made by somebody) but they
    are declared with a reason, and every one of the four is then CHECKED against a
    measurement: the crest must be inside, each basin's box must resolve to a cell
    that is under that basin's own water at the epoch being drawn, and -- once the
    flood has run -- the whole minimax route between the two seeds must lie inside as
    well.  A bound that fails any of those refuses the frame; none of them is widened
    for it.

    Parameters
    ----------
    name : str
        The extent's own name, recorded on every artifact derived from it.
    lon_min, lat_min, lon_max, lat_max : float
        INCLUSIVE bounds, EPSG:4326, degrees.  Snapped to the source raster's own
        cell grid at crop time, so the inset holds SOURCE CELLS, not resampled ones.
    why : str
        What the bounds are a statement about -- one clause per bound.  Never blank.
    seed_boxes : tuple[SeedBox, ...]
        The registry for THIS extent.  Its basins are the only basins the inset can
        declare, and each box must lie inside the bounds.
    must_contain : tuple[tuple[str, float, float], ...]
        ``(name, lon, lat)`` points the extent is REQUIRED to contain, checked before
        anything is rendered.  The Manych crest is one; it is the cell the whole
        verdict is about, and an inset that does not contain it would draw a corridor
        whose controlling col is somewhere else.
    """

    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    why: str
    seed_boxes: tuple[SeedBox, ...]
    must_contain: tuple[tuple[str, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not (self.lon_min < self.lon_max and self.lat_min < self.lat_max):
            raise ValueError(
                f"InsetExtent {self.name!r} bounds are not ascending: "
                f"lon {self.lon_min}..{self.lon_max}, lat {self.lat_min}..{self.lat_max}"
            )
        for field_name in ("name", "why"):
            text = getattr(self, field_name)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"InsetExtent.{field_name} is {text!r}")
        if not self.seed_boxes:
            raise ValueError(
                f"InsetExtent {self.name!r} registers no seed box. assign_basins "
                "cannot declare a basin it has no seed for, and a sub-extent whose "
                "seeds are inherited from the full scene would be seeded outside "
                "itself -- _seed_rowcol would clamp or raise, and either way the "
                "cell drawn would not be the cell named."
            )
        outside = [
            b.name for b in self.seed_boxes
            if not (
                self.lon_min <= b.lon_min and b.lon_max <= self.lon_max
                and self.lat_min <= b.lat_min and b.lat_max <= self.lat_max
            )
        ]
        if outside:
            raise InsetExtentError(
                f"seed box(es) {outside!r} are not wholly inside InsetExtent "
                f"{self.name!r} {self.bounds()}. A box that leaves the extent "
                "resolves its argmin over ground the inset raster does not carry, so "
                "the seed it returns cannot be the seed the frame is drawn from."
            )
        for label, lon, lat in self.must_contain:
            if not self.contains(lon, lat):
                raise InsetExtentError(
                    f"InsetExtent {self.name!r} {self.bounds()} does not contain "
                    f"{label} at {lon:.6f} E / {lat:.6f} N."
                )

    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)``."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.lon_min <= float(lon) <= self.lon_max
            and self.lat_min <= float(lat) <= self.lat_max
        )

    def basins(self) -> tuple[str, ...]:
        return tuple(b.basin for b in self.seed_boxes)

    def pairs(self) -> tuple[tuple[str, str], ...]:
        names = sorted(self.basins())
        return tuple(
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bounds_wsen": list(self.bounds()),
            "span_deg": [
                round(self.lon_max - self.lon_min, 10),
                round(self.lat_max - self.lat_min, 10),
            ],
            "why": self.why,
            "basins": list(self.basins()),
            "seed_boxes": [b.to_dict() for b in self.seed_boxes],
            "must_contain": [
                {"name": n, "lon": float(lo), "lat": float(la)}
                for n, lo, la in self.must_contain
            ],
        }


#: The Manych crest as MEASURED on the production raster at native resolution -- the
#: cell ``assign_basins`` returns for the ``caspian<->pontic`` saddle, reproduced by
#: B1's 1 953-seed sweep to the last float64 bit and again, this session, on the
#: inset window alone.  It is the one point the inset is REQUIRED to contain, and it
#: is not typed here as a location: it is the coordinate the sill measurement itself
#: reports, and the render re-measures it on every frame.
MANYCH_CREST_MEASURED_LON = 44.110416666666666
MANYCH_CREST_MEASURED_LAT = 45.735416666666666

#: THE INSET OF RECORD.  Every bound below was chosen by a measurement stated in
#: ``why`` and re-checked at render time; none is a framing preference.
#:
#: Measured on ``khv_16.5ka.tif`` at native resolution while this was written, on the
#: window this extent cuts (3 144 x 1 104 cells, source window col 2400 / row 1488):
#:
#: * the ``caspian<->pontic`` saddle reads **+28.000944138 m at 44.110417 E /
#:   45.735417 N** -- the same value and the same cell as the full 25.5-58 E scene at
#:   ``max_px=7800`` (+28.000944137573242 m), i.e. this grid resolves the channel;
#: * the minimax route between the two inset seeds spans 35.8188-47.8188 E /
#:   43.5896-47.4854 N, inside the extent with margins of 0.319 / 0.390 / 0.781 /
#:   0.315 deg (W/S/E/N);
#: * both seed boxes resolve to a cell under their own basin's surface at ALL NINE
#:   epochs of the 0.5 ka window, with margins of +85.8..+104.9 m (Caspian) and
#:   +2128.1..+2150.7 m (Pontic).
MANYCH_INSET = InsetExtent(
    name="manych_corridor_and_both_receiving_seas",
    lon_min=35.5,
    lat_min=43.2,
    lon_max=48.6,
    lat_max=47.8,
    why=(
        "EAST 48.6 E: far enough into the Caspian to hold a cell under the Caspian "
        "surface at every epoch of 17-13 ka. The registered box resolves to -66.94 m "
        "at 47.7604 E / 44.7938 N on khv_16.5ka.tif, which is 85.79 m below the "
        "LOWEST Caspian surface of the window (+18.85 m at 17.0 ka), so the box "
        "cannot resolve dry. | WEST 35.5 E and SOUTH 43.2 N: far enough into the "
        "Black Sea to hold a cell under the Pontic surface at every epoch. The "
        "registered box resolves to -2168.60 m at 36.9896 E / 43.8979 N on "
        "khv_16.5ka.tif, which is 2128.26 m below the LOWEST Pontic surface of the "
        "window (-40.35 m at 17.0 ka) and 2150.58 m below the surface of its OWN "
        "epoch (-18.02 m at 16.5 ka). Until 2026-08-04 this clause carried the "
        "second figure against the first surface -- a margin measured on one epoch "
        "wearing another epoch's name, which is the pairing this file exists to make "
        "impossible. Measured epoch by epoch, each raster's own seed against that "
        "epoch's own surface, the margin runs +2128.12 m (17.0 ka) to +2150.71 m "
        "(15.0 ka) -- the range the module docstring above records -- so the box "
        "cannot resolve dry anywhere in the window. The "
        "west bound also keeps the Kerch narrows (~36.4-36.6 E) inside, because that "
        "is where the full-scene route crosses from the Azov lowland to the Black "
        "Sea; cutting it would leave the inset to find its own western connection "
        "over the Kuban lowland and draw a channel the full frame does not. | NORTH "
        "47.8 N: the corridor is NOT confined to the Kuma-Manych latitudes. Measured "
        "per 0.5 deg of longitude, the trough floor between 45.2 and 46.6 N rises to "
        "+38.20 m at 40.99 E -- above the Caspian's own +29.94 m surface at 16.5 ka "
        "-- so the continuous path at those longitudes runs north of that band, "
        "through the lower Don. The measured route tops out at 47.4854 N. | SOUTH "
        "43.2 N is also the bound that keeps the measured route inside: the route's "
        "SOUTHERNMOST CELL sits at 43.5896 N (the value assert_route_inside_the_inset "
        "records as route_lat_span[0] on the 16.5 ka frame), clearing the edge by "
        "0.3896 deg -- 93.5 cells of this raster's 1/240 deg grid. Until 2026-08-04 "
        "this clause reported that clearance as half a cell, 0.0021 deg: that figure "
        "is the grid's own cell size halved and is not a margin at all, and the "
        "clearance the route actually has is 187 times it. | The extent is "
        "13.1 x 4.6 deg against the scene's 32.5 x 18.0. BOTH PANELS ARE MAGNIFIED "
        "2.48x, IN BOTH AXES: each is drawn by one imshow with aspect='equal' inside "
        "the same square figure box, and both extents are wider than that box, so "
        "both axes are width-limited and the linear scale is the WIDTH ratio "
        "32.5/13.1 in longitude and in latitude alike -- re-measured on the "
        "renderer's own figure setup, which gives the inset axes 2.48x the full "
        "axes' pixels per degree in each direction. 18.0/4.6 = 3.91 is the ratio of "
        "the two LATITUDE SPANS, i.e. how much less latitude the inset shows; it is "
        "not a magnification of anything drawn, and the sentence shipped until "
        "2026-08-04 reported it as the latitude magnification. The inset panel is "
        "therefore SHORTER than the full panel, not stretched. It is not tighter "
        "because it CANNOT be: assign_basins refuses a "
        "seed that is not under its own water, a basin with no seed inside the "
        "extent cannot be declared at all, and the nearest cell that is under the "
        "Pontic surface at every epoch of this window is in the Black Sea proper."
    ),
    seed_boxes=(
        SeedBox(
            basin="caspian",
            name="caspian_mangyshlak_shelf_inset",
            lon_min=47.2,
            lat_min=44.0,
            lon_max=48.55,
            lat_max=46.0,
            why=(
                "the mid-Caspian slope east of the Volga-delta shallows and north of "
                "the Mangyshlak sill -- the deepest Caspian water this extent "
                "contains (-66.94 m on khv_16.5ka.tif at 47.7604 E / 44.7938 N). It "
                "is NOT the South Caspian box of SEED_BOXES: that box lies at 49-54 "
                "E / 36.5-40.5 N, outside this extent entirely, which is exactly why "
                "an inset needs a registry of its own rather than an inherited "
                "coordinate. Every Khvalynian surface of 17-13 ka stands at least "
                "85.8 m above this cell, so the box cannot resolve to dry ground at "
                "any epoch of the window"
            ),
        ),
        SeedBox(
            basin="pontic",
            name="pontic_northeast_abyssal_inset",
            lon_min=36.6,
            lat_min=43.65,
            lon_max=38.6,
            lat_max=44.6,
            why=(
                "the north-eastern corner of the Black Sea abyssal plain, the "
                "deepest Pontic water this extent contains (-2168.60 m on "
                "khv_16.5ka.tif at 36.9896 E / 43.8979 N). Deliberately in the deep "
                "basin and NOT in the incised Kerch channel, which also holds cells "
                "below every Pontic surface of the window (-62.88 m at 36.4979 E / "
                "45.5979 N) but is a narrow trench whose connection to the open "
                "basin at a given level is the very thing the frame is measuring: "
                "seeding there would declare 'pontic' to be whatever pocket that "
                "trench happens to be part of"
            ),
        ),
    ),
    must_contain=(
        ("the measured Manych crest", MANYCH_CREST_MEASURED_LON,
         MANYCH_CREST_MEASURED_LAT),
    ),
)


# ---------------------------------------------------------------------------
# THE GATEWAY THE SILL GUARD CANNOT SEE
# ---------------------------------------------------------------------------
#
# A tighter extent has one failure mode the ``caspian<->pontic`` expectation does
# NOT catch, and it has to be measured separately or it ships silently.
#
# The guarded quantity is the CONTROLLING COL of the whole route -- the Manych
# crest, +28.000944 m at 16.5 ka.  Everything downstream of it is below it by
# construction, so the guard is indifferent to WHICH low crossing the water takes
# out of the palaeo-Azov/Don lowland into the Black Sea.  There are two, and they
# are 2.51 m apart:
#
#   the KERCH line   +10.030412 m at 36.610417 E / 45.331250 N
#   the TAMAN line   +12.536081 m at 37.243750 E / 45.043750 N
#
# Measured on ``khv_16.5ka.tif`` at native resolution, on ``dem_sm`` -- the same
# sigma=1.5 array ``assign_basins`` measures its saddles on -- as a connected-path
# minimax between a cell in the palaeo-Azov lowland (37.202083 E / 46.197917 N,
# -2.0207 m) and the eastern Black Sea (37.602083 E / 44.197917 N, -1918.3033 m),
# first over the whole window and then with the domain cut at 36.7 E so the Kerch
# line is unavailable.  Both readings reproduce on a 672 x 672-cell window and on
# the 34-40.5 E / 43.5-47.5 N one, to the last float64 bit.
#
# So an extent whose west bound falls between the two hands the spill to a crossing
# 2.505670 m higher and 0.63 deg further east, draws a different outlet, and the
# sill verdict stays ``WITHIN_TOLERANCE`` throughout -- measured: every west bound
# from 35.5 E to 37.2 E gives the same +28.000944 m at the same crest cell.  The
# expectation is a statement about the controlling col and it is doing its job; it
# simply is not a statement about the picture, and this pair of constants is what
# lets a bound be placed against the picture with a number.
AZOV_PONTIC_COL_KERCH_M = 10.030411720275879
AZOV_PONTIC_COL_KERCH_LON = 36.610416666666666
AZOV_PONTIC_COL_KERCH_LAT = 45.331250000000004

AZOV_PONTIC_COL_TAMAN_M = 12.536081314086914
AZOV_PONTIC_COL_TAMAN_LON = 37.24375
AZOV_PONTIC_COL_TAMAN_LAT = 45.04375

#: How both cols above were obtained, recorded beside them for the same reason
#: :data:`SEED_RULE` is recorded beside every seed.
AZOV_PONTIC_COL_RULE = (
    "connected-path minimax (priority flood, 4-connected) on dem_sm -- the "
    "sigma=1.5 gaussian of the native khv_16.5ka.tif mosaic, i.e. the SAME array "
    "assign_basins measures its saddles on -- between one cell in the palaeo-Azov "
    "lowland and one in the eastern Black Sea deep. The KERCH value is the col over "
    "the whole window; the TAMAN value is the col when the domain is cut west of "
    "the Kerch line, which is what an extent whose west bound falls east of it "
    "does. Neither is read off a single cell and neither is quoted from a document"
)


#: THE TIGHT EXTENT: the corridor as the frame rather than as an inset in a scene.
#:
#: WHAT THIS IS FOR, AND WHY IT IS A SECOND REGISTRY ENTRY RATHER THAN NEW BOUNDS
#: ------------------------------------------------------------------------------
#: :data:`MANYCH_INSET` is not edited.  It shipped, its frames were rendered from
#: it and its manifests name it; an extent that changed under its own name would
#: make every one of those artifacts unreproducible.  This is a different extent
#: with a different name, and the two travel together in :data:`INSET_EXTENTS`.
#:
#: Every bound below is one step back from a cliff this session MEASURED at 16.5 ka
#: on the canonical raster, on each candidate's own crop, own gaussian and own
#: flood, and every one of the nine epochs of the 0.5 ka window was then re-checked
#: on its own raster.  The four cliffs, and the value on each side of them:
#:
#:     bound   last value that holds   first value that fails   what fails there
#:     WEST      37.2 E (+28.000944)     37.3 E                  the two territories
#:                                                               stop touching at all
#:     SOUTH     44.8 N (all nine)       44.9 N                  the Pontic seed is
#:                                                               DRY at 17.0/13.0 ka
#:     EAST      45.4 E (+28.000944)     45.2 E (+40.135)        the Caspian's water
#:                                                               loses the crest
#:     NORTH     47.2 N (+28.000944)     47.1 N (+89.971)        the lower Don is cut
#:
#: The registered bounds are 36.4 / 44.6 / 46.0 / 47.6, i.e. NOT the cliff values --
#: see the ``why`` for the clearance each one keeps and for what the caution costs.
MANYCH_CORRIDOR_TIGHT = InsetExtent(
    name="manych_corridor_crest_kerch_col_and_the_nearest_all_epoch_water",
    lon_min=36.4,
    lat_min=44.6,
    lon_max=46.0,
    lat_max=47.6,
    why=(
        "WEST 36.4 E: the gateway, and it is the one bound the sill guard cannot "
        "set. There are exactly two crossings from the palaeo-Azov/Don lowland into "
        "the Black Sea and both lie far below the Manych crest, so the "
        "caspian<->pontic expectation is indifferent between them: measured on "
        "khv_16.5ka.tif at native resolution, EVERY west bound from 35.5 E to 37.2 E "
        "gives the same +28.000944 m at the same crest cell, and only at 37.3 E do "
        "the two territories stop touching below the flood ceiling (+51.359 m). What "
        "does change is which outlet the picture shows: the Kerch line is "
        "+10.030412 m at 36.610417 E / 45.331250 N and the Taman line, the col that "
        "survives cutting the domain at 36.7 E, is +12.536081 m at 37.243750 E / "
        "45.043750 N -- 2.505670 m higher and 0.633333 deg further east (see "
        "AZOV_PONTIC_COL_RULE for how both were measured). 36.4 E keeps the Kerch col "
        "inside by 0.210417 deg, 50.5 cells of this raster's 1/240 deg grid, and the "
        "col is registered in must_contain so an extent that loses it refuses instead "
        "of quietly redrawing the outlet. | SOUTH 44.6 N: the Pontic seed, and this "
        "is the bound that is NOT a preference. The registered box resolves to "
        "-1219.323 m at 37.260417 E / 44.622917 N on khv_16.5ka.tif, and epoch by "
        "epoch -- each raster's own argmin against that epoch's own surface -- the "
        "margin runs +1178.852 m (17.0 ka) to +1201.318 m (15.5 ka). One step "
        "tighter is a REFUSAL and not a squeeze: at south 44.9 N the deepest Pontic "
        "water this extent would hold is -26.944 m at 37.097917 E / 44.910417 N on "
        "khv_17ka.tif, which stands ABOVE the 17.0 ka Pontic surface of -40.346 m and "
        "above 13.0 ka's -34.802 m, so resolve_seed raises SeedAboveItsLevelError at "
        "two of the nine epochs -- an extent that works at 16.5 ka and fails at 17.0 "
        "and 13.0. 44.8 N does still resolve (-245.944 m at 36.952083 E / 44.810417 "
        "N, +205.597 m under the 17.0 ka surface); 44.6 N is where the north-eastern "
        "Black Sea slope still carries kilometre-deep water. North of 44.956250 N "
        "there is no Pontic water at all inside this extent's own longitudes at "
        "16.5 ka, and the last cell before that limit (36.535417 E / 44.956250 N, "
        "-18.109 m) is one that only the 16.5 ka surface covers. "
        "| EAST 46.0 E: the Caspian seed AND the Caspian's own connection to the "
        "crest. The registered box resolves to -19.728 m at 45.722917 E / 46.093750 N "
        "on khv_16.5ka.tif, with a per-epoch margin of +38.249 m (13.0 ka) to "
        "+57.685 m (16.0 ka) against each epoch's own Caspian surface. The bound is "
        "not set by that margin, though: at east 45.2 E the caspian<->pontic saddle "
        "reads +40.135 m instead of +28.000944 -- a gap of +12.43 m against a 2.00 m "
        "tolerance -- because the deepest Caspian water the extent then holds "
        "(-10.61 m at 45.118750 E / 46.689583 N) reaches the crest only over a ridge "
        "whose way round the extent has just cut off. 45.4 E still reads +28.000944 m "
        "at the registered crest cell. 46.0 E clears that cliff by 0.6 deg. | NORTH "
        "47.6 N: the corridor is NOT confined to the Kuma-Manych latitudes, and this "
        "extent measures the cliff the shipped inset argued from a trough profile. At "
        "north 47.1 N the saddle reads +89.971 m -- the lower Don is outside and the "
        "flood has to cross the divide somewhere else entirely -- and at 47.2 N it "
        "reads +28.000944 m again. 47.6 N clears that cliff by 0.4-0.5 deg and also "
        "contains this extent's own measured minimax route, whose northernmost cell "
        "is 47.485417 N, by 0.114583 deg. | ROUTE CLEARANCE IS NOT A MARGIN ON THIS "
        "EXTENT, and it is written down here so nobody reads one into it. This "
        "extent's own route touches the west, south and east edges to within "
        "0.006250 / 0.002083 / 0.014583 deg. That is not the extent being too tight: "
        "every cell of the route is at or below the controlling col by construction, "
        "so which cells the parent chain walks between the seeds is decided by the "
        "priority flood's flat-index tie-break, and the renderer's own sidecar says "
        "so in route_note -- one member of an equivalence class, not a determinate "
        "spillway. The four bounds above are each fixed by a measurement that does "
        "NOT move when the tie-break does. | THE FRAME. 9.6 x 3.0 deg against the "
        "scene's 32.5 x 18.0 and MANYCH_INSET's 13.1 x 4.6, i.e. 28.80 of 60.26 "
        "square degrees, 47.79 % of that inset's area. Both panels are drawn by one "
        "imshow with aspect='equal' inside the same square figure box and both "
        "extents are wider than that box, so both axes are width-limited and the "
        "linear scale is the WIDTH ratio in longitude and in latitude alike: "
        "3.3854x the full panel against MANYCH_INSET's 2.4809x, 1.3646x more, "
        "re-measured on the renderer's own figure setup rather than argued from the "
        "spans. | WHAT THE CAUTION COSTS, so the choice can be re-litigated from "
        "numbers rather than from taste. The tightest extent every one of whose "
        "bounds still holds at ALL NINE epochs is 36.6 / 44.8 / 45.4 / 47.2, "
        "8.8 x 2.4 deg, and it was measured at all nine: same saddle, same crest "
        "cell, same gaps. It is 3.6932x, i.e. 9.09 % more linear scale than this "
        "one. It is not registered because it puts the Kerch col 0.010417 deg -- 2.5 "
        "cells -- inside its west edge, the lower-Don cliff 0.1 deg from its north "
        "edge and the Caspian-severing cliff 0.2 deg from its east edge, and a bound "
        "two cells from the thing it exists to contain is a bound that will be "
        "crossed by the next raster rather than by an argument."
    ),
    seed_boxes=(
        SeedBox(
            basin="caspian",
            name="caspian_depression_floor_inside_the_tight_extent",
            lon_min=45.30,
            lat_min=45.45,
            lon_max=45.95,
            lat_max=46.35,
            why=(
                "the floor of the Caspian Depression at the extent's north-east "
                "corner, well west of the Volga's course -- the deepest cell of the "
                "Caspian's OWN water body that this extent contains (-19.728 m on "
                "khv_16.5ka.tif at 45.722917 E / 46.093750 N). It is neither of the "
                "Caspian boxes this module already carries: SEED_BOXES' South "
                "Caspian box is at 49-54 E / 36.5-40.5 N and MANYCH_INSET's "
                "mid-Caspian slope box at 47.2-48.55 E, both of them east of this "
                "extent's own east bound entirely, which is why a third extent needs "
                "a third registry rather than an inherited coordinate. The cell is "
                "under the Caspian surface at every epoch of 17-13 ka with at least "
                "+38.249 m to spare, so the box cannot resolve dry anywhere in the "
                "window"
            ),
        ),
        SeedBox(
            basin="pontic",
            name="pontic_northeast_slope_inside_the_tight_extent",
            lon_min=37.10,
            lat_min=44.62,
            lon_max=37.60,
            lat_max=44.85,
            why=(
                "the north-eastern continental slope of the Black Sea, off the "
                "Caucasian coast -- the deepest Pontic water this extent contains "
                "(-1219.323 m on khv_16.5ka.tif at 37.260417 E / 44.622917 N). "
                "Deliberately on the open slope and NOT in the Kerch narrows, which "
                "this extent also contains and which also hold cells below every "
                "Pontic surface of the window (-62.88 m at 36.4979 E / 45.5979 N, "
                "the figure MANYCH_INSET's own Pontic box records): that trench is "
                "the gateway whose connection to the open basin at a given level is "
                "the very thing the frame is measuring, and seeding in it would "
                "declare 'pontic' to be whatever pocket the gateway happens to be "
                "part of. The Sea of Azov is not an option at any epoch of this "
                "window -- inside 34.5-39.5 E / 45.0-47.5 N the Pontic water body "
                "has ZERO cells at 16.5 ka -- which is why the nearest usable Pontic "
                "water is in the Black Sea proper and why this box sits where it does"
            ),
        ),
    ),
    must_contain=(
        ("the measured Manych crest", MANYCH_CREST_MEASURED_LON,
         MANYCH_CREST_MEASURED_LAT),
        ("the measured Kerch col, the palaeo-Azov lowland's outlet to the Black Sea",
         AZOV_PONTIC_COL_KERCH_LON, AZOV_PONTIC_COL_KERCH_LAT),
    ),
)


#: EVERY EXTENT THIS MODULE OFFERS, in one place, so the CLI, the manifest and the
#: tests all read the same list and a new extent cannot be half-registered.
INSET_EXTENTS: tuple[InsetExtent, ...] = (MANYCH_INSET, MANYCH_CORRIDOR_TIGHT)


def inset_extent_by_name(name: str) -> InsetExtent:
    """The registered extent called *name*, or a refusal naming the ones there are."""
    for extent in INSET_EXTENTS:
        if extent.name == name:
            return extent
    raise InsetExtentError(
        f"no InsetExtent named {name!r} is registered. The registry holds "
        f"{[e.name for e in INSET_EXTENTS]!r}; an extent that is not in it has no "
        "seed boxes, no per-bound reason and no sill expectation of its own, and a "
        "rectangle passed in by name would have none of the three."
    )


#: How the inset raster is cut, recorded on every crop.
INSET_CROP_RULE = (
    "a rasterio window read on the SOURCE cell grid, offsets and size rounded to "
    "whole cells, written back at the source resolution with the source's own "
    "nodata, CRS and GeoTIFF tags -- including vertical_frame, without which "
    "assign_basins refuses to frame-check a SillExpectation and the inset would be "
    "renderable only unguarded. NO resampling and NO reprojection: the crop is "
    "re-read after writing and compared cell by cell against the source window, and "
    "a single differing cell refuses. That check is the whole reason the inset may "
    "consume the full scene's pinned SequenceLimits -- the two pictures are drawn "
    "from the same numbers because they are drawn from the same cells."
)


def crop_to_extent(
    src: Path | str,
    extent: InsetExtent,
    dst: Path | str,
    *,
    src_sha256: Optional[str] = None,
) -> dict:
    """Cut *extent* out of *src* on the source cell grid.  Returns the crop record.

    Raises
    ------
    InsetExtentError
        The extent is not wholly inside the source raster.
    InsetCropError
        The written file does not re-read as the source window, cell for cell.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as dataset:
        bounds = dataset.bounds
        if not (
            bounds.left <= extent.lon_min and extent.lon_max <= bounds.right
            and bounds.bottom <= extent.lat_min and extent.lat_max <= bounds.top
        ):
            raise InsetExtentError(
                f"InsetExtent {extent.name!r} {extent.bounds()} is not inside "
                f"{src.name}'s own bounds {tuple(bounds)}. A window that leaves the "
                "raster is padded with nodata by rasterio, and an inset with a "
                "nodata margin would draw an observation boundary the terrain does "
                "not have."
            )
        raw = from_bounds(*extent.bounds(), dataset.transform)
        window = Window(
            int(round(raw.col_off)), int(round(raw.row_off)),
            int(round(raw.width)), int(round(raw.height)),
        )
        block = dataset.read(1, window=window)
        profile = dataset.profile.copy()
        transform = dataset.window_transform(window)
        tags = dict(dataset.tags())
        cut_bounds = rasterio.windows.bounds(window, dataset.transform)

    profile.update(
        height=int(window.height), width=int(window.width), transform=transform
    )
    tags.update(
        {
            "inset_extent_name": extent.name,
            "inset_extent_bounds_wsen": ",".join(f"{v:.10g}" for v in extent.bounds()),
            "inset_source_raster": str(src.resolve()),
            "inset_source_sha256": src_sha256 or sha256_of(src),
            "inset_source_window": (
                f"col_off={int(window.col_off)},row_off={int(window.row_off)},"
                f"width={int(window.width)},height={int(window.height)}"
            ),
            "inset_crop_rule": INSET_CROP_RULE,
        }
    )
    with rasterio.open(dst, "w", **profile) as out:
        out.write(block, 1)
        out.update_tags(**tags)

    with rasterio.open(dst) as check:
        written = check.read(1)
    if written.shape != block.shape or not np.array_equal(written, block):
        differing = (
            "a different shape"
            if written.shape != block.shape
            else f"{int((written != block).sum()):,} differing cell(s)"
        )
        raise InsetCropError(
            f"{dst.name} re-reads with {differing} against the window it was cut "
            f"from ({src.name} {window}). The inset is only allowed to share the "
            "scene's pinned colour mapping because it holds the SAME CELLS; a crop "
            "that resampled them holds different elevations and would map the same "
            "ground to a different colour in two halves of one picture."
        )

    return {
        "rule": INSET_CROP_RULE,
        "extent": extent.name,
        "source_path": str(src.resolve()),
        "source_sha256": src_sha256 or sha256_of(src),
        "inset_path": str(dst.resolve()),
        "inset_sha256": sha256_of(dst),
        "window": {
            "col_off": int(window.col_off), "row_off": int(window.row_off),
            "width": int(window.width), "height": int(window.height),
        },
        "requested_bounds_wsen": list(extent.bounds()),
        "cut_bounds_wsen": [float(v) for v in cut_bounds],
        "cells": int(window.width) * int(window.height),
        "cells_compared_after_write": int(written.size),
        "cells_differing_after_write": 0,
        "verdict": "CELL_FOR_CELL_IDENTICAL_TO_THE_SOURCE_WINDOW",
    }


# ---------------------------------------------------------------------------
# THE THREE SILLS
# ---------------------------------------------------------------------------

#: The Manych crest, on the MODERN GEBCO 15-arcsec COG, at native resolution, in
#: F0.  Re-derived this session by a connected-path minimax (priority flood) between
#: the Caspian and Pontic seeds of :data:`SEED_BOXES`: **+25.000000 m at
#: 44.102083 E / 45.706250 N**, route 13 375 cells, route maximum equal to the crest
#: to the last digit.  That reproduces ``bake_khvalynian.MANYCH_SILL_PATH_M`` and the
#: crest cell already pinned by ``test_manych_path_constant_matches_a_native_flood``.
MANYCH_CREST_F0_M = 25.00
MANYCH_CREST_LON, MANYCH_CREST_LAT = 44.102083, 45.706250

#: The Bosphorus crest, ESTABLISHED the same way in the same flood rather than
#: quoted: **-20.000000 m at 28.997917 E / 41.010417 N**, route 6 878 cells.
#:
#: Measured on the MODERN COG, never on the ``khv_*.tif`` this sequence then guards
#: -- measuring a crest on the raster you are checking is self-referential and
#: certifies nothing (phase-1 contradiction C7).
BOSPHORUS_CREST_F0_M = -20.00
BOSPHORUS_CREST_LON, BOSPHORUS_CREST_LAT = 28.997917, 41.010417


def manych_expectation(epoch_ka: float, dem_frame: str) -> SillExpectation:
    """The ``caspian<->pontic`` expectation for one epoch, built from the data.

    The quantity is ``MANYCH_SILL_PATH_M`` (+25.00 m, F0, the continuous-path
    controlling col of the modern GEBCO raster) **plus** this epoch's
    ``dz_sill_m``, i.e. the bake's own ``sill_path_f1_m`` -- +27.701 m at 16.5 ka.

    It is deliberately NOT ``MANYCH_SILL_M`` (+23.00 m).  ``bake_khvalynian``
    documents that constant as the max of per-column minima, which is a LOWER BOUND
    on the path crest: armed as the expectation it false-fails by +2.30 m even on
    the native grid, where the terrain and the frame in fact agree to +0.30 m.
    """
    from scripts import bake_khvalynian as bk

    dz_sill_m, block = bk.sill_solid_displacement(float(epoch_ka))
    return SillExpectation(
        value_m=float(MANYCH_CREST_F0_M + dz_sill_m),
        frame=dem_frame,
        source=(
            f"MANYCH_SILL_PATH_M {MANYCH_CREST_F0_M:+.2f} m (F0, connected-path "
            f"minimax on the modern GEBCO 15-arcsec COG at native resolution, "
            f"re-derived 2026-08-03 as {MANYCH_CREST_F0_M:+.6f} m at "
            f"{MANYCH_CREST_LON:.6f} E / {MANYCH_CREST_LAT:.6f} N) + dz_sill_m "
            f"{dz_sill_m:+.6f} m (bake_khvalynian.sill_solid_displacement at "
            f"{float(epoch_ka):g} ka, ICE-6G_C VM5a 10min; 3x3 cell block "
            f"{block[0]:+.6f}..{block[1]:+.6f} m) = "
            f"{MANYCH_CREST_F0_M + dz_sill_m:+.6f} m"
        ),
        measured_by=(
            "connected-path minimax (priority flood) between the registry's Caspian "
            "and Pontic seeds on the modern GEBCO COG, converted to the DEM's frame "
            "at the sill cell -- NOT a single-cell reading of the sill point, which "
            "is a different quantity by up to 8 m on this raster, and NOT measured "
            "on the khv_*.tif being guarded, which would be self-referential"
        ),
        lon=MANYCH_CREST_LON,
        lat=MANYCH_CREST_LAT,
    )


def bosphorus_unmeasured(epoch_ka: float, dz_bosphorus_m: float) -> UnmeasuredSill:
    """``pontic<->mediterranean``: measured, and declared uncheckable on this grid.

    The crest WAS established, with provenance, by the same recipe as the Manych:
    ``-20.000000 m at 28.997917 E / 41.010417 N`` on the modern GEBCO COG at native
    resolution, F0, over a 6 878-cell route, in the same flood that re-derived
    ``MANYCH_SILL_PATH_M`` as ``+25.000000 m`` as its control.  Converting it with
    ``dz`` at that cell gives ``-9.181 m`` at 16.5 ka.

    **It survives an independent check.**  Rendering 16.5 ka with
    ``measure_smoothing_sill_shift=True`` measures the same saddle on the
    UNSMOOTHED native mosaic of ``khv_16.5ka.tif`` -- a different raster, already in
    the DEM's own frame, with no conversion applied by hand -- and gets
    **-9.51 m**.  Two readings, two rasters, two routes to the frame: **0.33 m
    apart**.  That is the same kind of agreement the Manych has (+25.000 F0 + 2.701
    = +27.701 against an independent +27.751, 0.05 m) and it means the crest is a
    measurement, not an estimate.

    It is nevertheless declared UNMEASURED rather than armed as a
    :class:`SillExpectation`, and the reason is the third number: the array the mask
    is actually built from -- ``dem_sm``, the sigma=1.5 smoothing of that same
    mosaic, which is what ``dem_threshold_ocean`` thresholds and therefore what the
    guard is required to compare against -- resolves this saddle at **+13.345 m**.
    The smoothing alone moves it **+22.86 m**, against +0.25 m for the Manych on the
    same grid, because the Bosphorus is about 700 m wide against a 463 x 350 m cell
    and a Gaussian closes what survives the resampling.  So the gap the guard would
    see is +22.53 m, more than eleven times the 2 m tolerance, and every bit of it
    is the smoothing.  Arming the expectation would refuse every frame of the
    sequence for a reason that is about pixel size, while hiding the Manych result
    -- which this same grid DOES resolve to 0.30 m -- behind it.

    HOW MANY TIMES THE NUMBER IS ACTUALLY DISCLOSED -- CORRECTED 2026-08-04
    ----------------------------------------------------------------------
    This docstring used to end "the pair is disclosed twice and checked never", and
    said the number travels into the sidecar and the manifest under
    ``sill_expectation_status``.  Both statements were wrong about the artifacts
    that shipped.

    * ``sill_expectation_status`` is an attribute of the renderer's ``BasinField``
      dataclass.  It is **not** a key in ``<frame>.basins.json`` (which spells it
      ``expectation_status``) and **not** a key in ``sequence_manifest.json`` at all
      -- grepping the 2026-08-03 manifest for either spelling returns zero hits.
    * "Twice" counts a disclosure that only exists when a frame RENDERS.  The
      2026-08-03 deliverable rendered none: all 17 frames refused, no
      ``.basins.json`` was written for any of them, and ``smoothing_shift_m``
      appears 27 times in that manifest and is ``null`` all 27 times.  So in the
      shipped artifacts the +22.86 m was disclosed **once** -- here, in this reason
      string, which reaches ``frames[].sill[].unmeasured_reason``.

    What is true, and is now what the manifest records as a value rather than a
    claim: this reason string is the disclosure that always ships, and a rendered
    frame adds a second one in its own ``.basins.json`` under
    ``sills[].smoothing_shift_m``.  ``_sill_block`` counts which of the two are
    actually present for each frame and writes the list into
    ``frames[].sill[].smoothing_shift_disclosed_in``, so the count comes off the
    artifacts.  (The ``SmoothingShiftWarning`` itself is a runtime warning; it is not
    an artifact and is not counted as a disclosure.)

    Declining to check is allowed; declining silently is not.  The guard's coverage
    is honestly two pairs declared and one checked, and that is now also a value:
    ``manifest.sill_verdicts.by_pair``.

    A second reason not to arm it, recorded so nobody re-litigates from the number
    alone: the Pontic LEVEL at 16.5 and 14.7 ka is itself the modern Bosphorus
    sediment-sill depth used as an overflow proxy (contract sections 3 and 7.1).  A crest
    measured in the same strait is not independent evidence about whether the Pontic
    cleared it; the two differ here by the 10 m between the curve's -30 mbsl
    sediment sill and GEBCO's -20 m path minimax over the whole channel, which is a
    disagreement about which quantity the outlet is, not about the flood.
    """
    crest_f1 = BOSPHORUS_CREST_F0_M + float(dz_bosphorus_m)
    return UnmeasuredSill(
        reason=(
            f"the Bosphorus crest IS measured -- {BOSPHORUS_CREST_F0_M:+.6f} m (F0) "
            f"at {BOSPHORUS_CREST_LON:.6f} E / {BOSPHORUS_CREST_LAT:.6f} N by "
            "connected-path minimax over a 6,878-cell route on the modern GEBCO "
            "15-arcsec COG at native resolution, in the same flood that re-derived "
            "MANYCH_SILL_PATH_M as +25.000000 m as its control; + dz "
            f"{float(dz_bosphorus_m):+.6f} m at that cell = {crest_f1:+.6f} m in the "
            f"DEM's frame at {float(epoch_ka):g} ka. It is independently confirmed AT "
            "16.5 ka: the same saddle measured on the UNSMOOTHED native mosaic of "
            "khv_16.5ka.tif, a different raster already in the DEM's frame, reads "
            "-9.51 m against this recipe's -9.181 m -- 0.33 m apart. It is NOT armed "
            "as an expectation because the array the mask is built from is dem_sm, "
            "the sigma=1.5 smoothing of that mosaic, and at 16.5 ka dem_sm puts this "
            "saddle at +13.345 m: the smoothing alone moves it +22.857 m there "
            "(against +0.250 m for the Manych on the same grid), because the "
            "Bosphorus is ~700 m wide against a 463x350 m cell and a Gaussian closes "
            "what survives the resampling. The guard's gap at that epoch would "
            "therefore be +22.53 m, 11x the 2 m tolerance, and all of it is the "
            "smoothing -- so checking it would "
            "refuse every frame for a reason about pixel size while hiding the "
            "Manych result, which this same grid resolves to 0.30 m. THOSE FOUR "
            "FIGURES ARE 16.5 ka MEASUREMENTS AND NONE OF THEM IS A WINDOW-WIDE "
            "CONSTANT: this epoch's own values are in the frame's sill block as "
            "measured_saddle_m, smoothing_shift_m and gap_m, and the run's range is "
            "summarised in manifest.sill_expectations.by_pair. THE NUMBER IS "
            "CARRIED HERE, and this string is the one disclosure that ships on every "
            "frame, refused or rendered; a frame that RENDERS adds a second one in "
            "its own .basins.json under sills[].smoothing_shift_m, and the manifest "
            "counts which of the two are actually present per frame in "
            "sill[].smoothing_shift_disclosed_in rather than asserting a number of "
            "them here. (The 2026-08-03 deliverable rendered no frame, so it "
            "disclosed this once, not twice, and the sentence that used to stand "
            "here said twice.) "
            "Note also that the Pontic LEVEL at 16.5/14.7 ka is itself the modern "
            "Bosphorus sill depth used as an overflow proxy (contract sections 3 and 7.1), "
            "so a crest from the same strait is not fully independent evidence about "
            "whether the Pontic cleared it"
        )
    )


#: ``caspian<->mediterranean``: there is no such sill, and that is the entry.
CASPIAN_MEDITERRANEAN_UNMEASURED = UnmeasuredSill(
    reason=(
        "these two basins share no direct sill. Any connection between them runs "
        "through the Pontic, so a 'caspian<->mediterranean crest' would be either "
        "the Manych or the Bosphorus wearing the wrong pair's name. Measured on "
        "khv_16.5ka.tif at max_px=7800 with the three registry seeds: their "
        "territories never touch at all, so the priority flood returns no saddle "
        "for this pair and assign_basins records NaN -- which is the correct "
        "answer and must not be read as agreement. assign_basins builds pair_keys "
        "over ALL pairs, so the entry has to exist; this is what it says"
    )
)


def sill_expectations_for(
    epoch_ka: float, dem_frame: str, dz_bosphorus_m: float
) -> dict[tuple[str, str], Any]:
    """All three unordered pairs, one checked and two declared.

    ``assign_basins`` builds ``pair_keys`` over every unordered pair and refuses a
    mapping that leaves any of them out -- a partial mapping leaves those pairs
    exactly as unchecked as ``sill_expectations=None`` did.
    """
    return {
        ("caspian", "pontic"): manych_expectation(epoch_ka, dem_frame),
        ("mediterranean", "pontic"): bosphorus_unmeasured(epoch_ka, dz_bosphorus_m),
        ("caspian", "mediterranean"): CASPIAN_MEDITERRANEAN_UNMEASURED,
    }


def dz_at_bosphorus(epoch_ka: float) -> float:
    """``dz_solid`` at the measured Bosphorus crest cell, this epoch."""
    from scripts import bake_khvalynian as bk

    dz, lats, lons = bk.solid_displacement_field(float(epoch_ka))
    i, j = bk._cell_index(lats, lons, BOSPHORUS_CREST_LON, BOSPHORUS_CREST_LAT)
    value = float(dz[i, j])
    if not np.isfinite(value):
        raise BasinLevelUnavailableError(
            f"dz at the Bosphorus crest cell is not finite at {float(epoch_ka):g} ka"
        )
    return value


# ---------------------------------------------------------------------------
# THE THREE LEVELS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinLevel:
    """One basin's water surface for one frame, and where the number came from."""

    basin: str
    level_m: float
    frame: str
    rule: str
    provenance: str

    def to_dict(self) -> dict:
        return {
            "basin": self.basin,
            "level_m": float(self.level_m),
            "frame": self.frame,
            "rule": self.rule,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CurveFile:
    """One level-curve CSV, fingerprinted, with the basin it drives named.

    WHY THIS EXISTS
    ---------------
    Until 2026-08-04 the manifest fingerprinted every RASTER and identified
    **neither level curve**.  Two of the three basin surfaces in every frame are
    read out of a CSV -- the Caspian from the Khvalynian curve, the Pontic from the
    Black Sea curve -- and those two files were never named, never pathed and never
    hashed anywhere in the deliverables.  A frame's caption could therefore not be
    checked against the numbers it was drawn from: the rasters were auditable and
    the water surfaces were not.

    ``sha256`` is not optional and there is no ``"unknown"`` sentinel.  Both files
    are hand-edited research artifacts under version control in a **dirty** tree
    (phase-1 contradiction C6 measured 9 tracked changes + ~140 untracked in
    ``Paleo-Europe_3D``), so a path alone identifies a file name, not its contents.
    """

    basin: str
    role: str
    constant: str
    path: Optional[str]
    sha256: Optional[str]
    bytes: Optional[int]
    status: str
    read_with: str

    def to_dict(self) -> dict:
        return {
            "basin": self.basin,
            "role": self.role,
            "constant": self.constant,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "status": self.status,
            "read_with": self.read_with,
        }


def _fingerprint_curve(
    basin: str, role: str, constant: str, path: Any, read_with: str
) -> CurveFile:
    """One curve file, hashed, or the reason it could not be."""
    try:
        resolved = Path(path)
    except Exception as exc:  # noqa: BLE001 - a bad constant is data, not a crash
        return CurveFile(basin, role, constant, None, None, None,
                         f"UNRESOLVED: {type(exc).__name__}: {exc}", read_with)
    if not resolved.exists():
        return CurveFile(
            basin, role, constant, str(resolved), None, None,
            f"MISSING: {resolved} does not exist, so the level this basin was drawn "
            "at cannot be checked against the file it came from",
            read_with,
        )
    return CurveFile(
        basin=basin,
        role=role,
        constant=constant,
        path=str(resolved.resolve()),
        sha256=sha256_of(resolved),
        bytes=resolved.stat().st_size,
        status="FINGERPRINTED",
        read_with=read_with,
    )


def level_curve_provenance() -> list[dict]:
    """Path + sha256 for every file a basin surface in this sequence is read from.

    THREE basins, TWO curves, and the third entry says so out loud.  The
    Mediterranean is ``-offset(t)`` computed from the ICE-6G_C distribution
    (vertical-datum contract section 2); there is no third CSV and a reader must not
    go looking for one, so the absence is recorded as an entry rather than as a
    silence.
    """
    entries: list[CurveFile] = []
    try:
        from scripts import bake_khvalynian as bk

        entries.append(
            _fingerprint_curve(
                "caspian",
                "F2 level column, scenario 'defensible'; converted to F1 at the "
                "threshold by bake_khvalynian.caspian_threshold",
                "bake_khvalynian.CASPIAN_CSV",
                bk.CASPIAN_CSV,
                "pandas.read_csv(comment='#') via bake_khvalynian.load_curve, then "
                "validate_basin_curve for every scenario + check_no_lgm_lowstand",
            )
        )
    except Exception as exc:  # noqa: BLE001
        entries.append(
            CurveFile("caspian", "F2 level column", "bake_khvalynian.CASPIAN_CSV",
                      None, None, None,
                      f"UNRESOLVED: {type(exc).__name__}: {exc}", "n/a")
        )
    try:
        from scripts import bake_black_sea as bbs

        entries.append(
            _fingerprint_curve(
                "pontic",
                "F2 level column, scenario 'gradual'; converted to F1 at the "
                "threshold by bake_black_sea.pontic_threshold",
                "bake_black_sea.PONTIC_CSV",
                bbs.PONTIC_CSV,
                "pandas.read_csv(comment='#') -- mandatory: the file leads with a "
                "'#' header block and a bare read_csv raises ParserError on it "
                "(vertical-datum contract section 8)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        entries.append(
            CurveFile("pontic", "F2 level column", "bake_black_sea.PONTIC_CSV",
                      None, None, None,
                      f"UNRESOLVED: {type(exc).__name__}: {exc}", "n/a")
        )
    entries.append(
        CurveFile(
            basin="mediterranean",
            role="F3 = -offset(t); NO level curve exists or is wanted for this basin",
            constant="bake_khvalynian.sea_surface_offset",
            path=None,
            sha256=None,
            bytes=None,
            status=(
                "NOT_A_CURVE: the Mediterranean surface is DERIVED from the ICE-6G_C "
                "distribution, not read from a CSV -- offset(t) is the median dTopo "
                "over far-field ice-free tropical continental land taken from the "
                "GLOBAL field, and the vertical-datum contract section 2 proves that "
                "scalar is the EXACT land/sea companion of H_paleo for a cell "
                "ICE-6G treats as ocean. This entry exists so that 'three basins, "
                "two curve files' is a statement in the data rather than a gap in it"
            ),
            read_with="bake_khvalynian.sea_surface_offset(epoch), cross-checked "
                      "against the five figures the contract prints (see "
                      "manifest.offset_cross_check)",
        )
    )
    return [entry.to_dict() for entry in entries]


def epoch_grid(
    window_ka: tuple[float, float] = FINE_WINDOW_KA,
    step_ka: float = FINE_STEP_KA,
) -> tuple[float, ...]:
    """The animation's epochs, taken from the bake's own grid.

    Not typed out: ``bake_khvalynian.EPOCHS_KA`` is where the 0.25 ka fine window is
    defined, and a second hand-written list is a second definition that can acquire
    a gap without anybody noticing.

    *step_ka* SELECTS FROM that grid, it does not generate one.  Every returned epoch
    is a member of ``EPOCHS_KA``, so a coarser grid can only ever be a subset of the
    baked rasters -- there is no arithmetic here that could produce an epoch nothing
    was baked for.  ``0.5`` gives the nine epochs whose Pontic rows exist; ``0.25``
    gives all seventeen.  The result is checked for uniformity before it is returned:
    a "0.5 ka grid" with a 0.75 ka step in the middle of it would be a caption that
    lies about its own contents, and every number computed over such a set -- a
    duration, a rate of rise, a count of epochs -- would inherit the lie.
    """
    from scripts import bake_khvalynian as bk

    lo, hi = float(window_ka[0]), float(window_ka[1])
    step = float(step_ka)
    if step <= 0:
        raise ValueError(f"step_ka must be positive, got {step_ka!r}")
    inside = [e for e in bk.EPOCHS_KA if lo <= e <= hi]
    grid = tuple(
        e for e in inside if abs(round((e - lo) / step) * step - (e - lo)) < 1e-9
    )
    steps = {round(a - b, 10) for a, b in zip(grid, grid[1:])}
    if len(grid) < 2 or steps not in ({step}, set()):
        raise ValueError(
            f"selecting step {step} ka out of bake_khvalynian.EPOCHS_KA over "
            f"{lo}-{hi} ka gives {len(grid)} epoch(s) with step(s) {sorted(steps)}, "
            f"not a uniform {step} ka grid. The bake's grid over this window is "
            f"{inside}; a step this does not divide would produce a sequence whose "
            "name does not describe its own contents."
        )
    return grid


def grid_description(
    epochs: Sequence[float],
    *,
    step_ka: float,
    window_ka: tuple[float, float] = FINE_WINDOW_KA,
    companion: Optional[str] = None,
    n_rendered: Optional[int] = None,
) -> dict:
    """What grid this sequence is on, and -- by name -- what it therefore leaves out.

    The 0.5 ka deliverable is not "the 0.25 ka animation with the gaps closed"; it is
    a different sequence over a different set of epochs, and the eight it does not
    carry are listed here with the reason, in the sequence's OWN manifest, so a
    reader never has to hold two files side by side to find out which one they have.

    *n_rendered* is how many of these slots produced imagery.  It exists because
    ``do_not_call_this`` used to read "a sequence of 9 rendered frames -- 8 of these
    17 slots are refusals" with all three numbers TYPED IN: on the 2026-08-04 run
    they happened to be right, and on any other run of the same grid they would have
    been a sentence contradicting the ``frames`` array beside them.  Passing it
    ``None`` says so rather than guessing.
    """
    epochs = [float(e) for e in epochs]
    full = list(epoch_grid(window_ka, FINE_STEP_KA))
    omitted = [e for e in full if e not in epochs]
    if step_ka != FINE_STEP_KA:
        do_not_call_this = f"a {FINE_STEP_KA:g} ka animation"
    elif n_rendered is None:
        do_not_call_this = (
            f"a sequence of {len(epochs)} rendered frames -- how many of these "
            f"{len(epochs)} slots carry imagery is counted in 'n_rendered' at the "
            "top of this manifest, and was not available when this block was built"
        )
    else:
        do_not_call_this = (
            f"a sequence of {int(n_rendered)} rendered frames -- "
            f"{len(epochs) - int(n_rendered)} of these {len(epochs)} slots are "
            "refusals"
        )
    return {
        "step_ka": float(step_ka),
        "window_ka": [float(window_ka[0]), float(window_ka[1])],
        "n_slots": len(epochs),
        "epochs": epochs,
        "source_of_the_epochs": (
            "selected out of bake_khvalynian.EPOCHS_KA, never generated -- every "
            "epoch here is one the bake wrote a raster for"
        ),
        "finest_grid_in_this_window_ka": FINE_STEP_KA,
        "n_slots_on_the_finest_grid": len(full),
        "quarter_ka_epochs_not_in_this_sequence": omitted,
        "why_those_are_absent": (
            PONTIC_GAP_RULE if omitted else
            "none: this sequence carries every epoch of the finest grid in the window"
        ),
        "companion_sequence": companion,
        "n_rendered_in_this_sequence": n_rendered,
        "do_not_call_this": do_not_call_this,
    }


def raster_for(epoch_ka: float, topo_dir: Path | str = KHV_TOPO_DIR) -> Path:
    """The baked raster for one epoch, refusing a substitute.

    The name is built the same way ``bake_khvalynian.build_paleo_dem`` builds it
    (``khv_<epoch:g>ka.tif``), so the two cannot disagree about 17 vs 17.0.
    """
    path = Path(topo_dir) / f"khv_{float(epoch_ka):g}ka.tif"
    if not path.exists():
        raise MissingEpochRasterError(
            f"no raster for {float(epoch_ka):g} ka at {path}. A neighbouring "
            "epoch's raster is NOT substituted: it carries the GIA field of its own "
            "epoch, so borrowing it would move every elevation in the frame and "
            "label the result with the wrong age."
        )
    return path


def assert_raster_tree_complete(
    topo_dir: Path | str, epochs: Sequence[float]
) -> dict:
    """Refuse a raster directory that cannot supply every epoch of the sequence.

    Returns the census that goes in the log: how many of the requested epochs the
    directory holds, how many were needed, and both spellings of the directory.

    WHY THIS IS A RAISE AND NOT A PER-EPOCH REFUSAL FRAME
    ----------------------------------------------------
    A refusal frame is the honest answer to "this epoch has no defensible level" --
    the slot is kept, the reason is drawn on it, time is not compressed.  It is the
    WRONG answer to "the directory is not the directory", because then every epoch
    refuses for a reason that is about configuration rather than about the data, and
    a run that produced 2 frames out of 17 would ship a manifest naming the 13-17 ka
    window.  That is not hypothetical: it is what the default did until 2026-08-04
    (see :data:`KHV_TOPO_DIR`), and it is why the two dataset-gated acceptance tests
    of this module failed.

    Raises
    ------
    IncompleteRasterTreeError
        The directory is missing, or holds fewer than the requested epochs.
    """
    directory = Path(topo_dir)
    wanted = [float(e) for e in epochs]
    present = [e for e in wanted if (directory / f"khv_{e:g}ka.tif").exists()]
    missing = [e for e in wanted if e not in present]
    census = {
        "topo_dir": path_record(directory),
        "epochs_needed": len(wanted),
        "epochs_found": len(present),
        "epochs_missing": missing,
        "tree_of_record": path_record(CANONICAL_TOPO_DIR),
        "is_the_tree_of_record": (
            str(directory).lower() == str(CANONICAL_TOPO_DIR).lower()
        ),
    }
    if missing:
        exists = directory.exists()
        n_tifs = len(list(directory.glob("khv_*ka.tif"))) if exists else 0
        raise IncompleteRasterTreeError(
            f"{directory} holds {len(present)} of the {len(wanted)} epoch rasters "
            f"this sequence needs"
            + (
                f" (it contains {n_tifs} file(s) matching khv_*ka.tif in total)"
                if exists
                else " -- the directory does not exist"
            )
            + f". Missing: {missing}. A short sequence is NOT rendered: the run "
            "would produce a manifest naming the whole 13-17 ka window over "
            f"{len(present)} frame(s). The tree of record is {CANONICAL_TOPO_DIR} "
            f"(= {path_record(CANONICAL_TOPO_DIR)['resolved']}); point --topo-dir or "
            "PALEO_OUT_ROOT at a tree that holds every epoch, or ask for only the "
            "epochs that exist with --epochs."
        )
    return census


def load_mirror_records(
    *search_dirs: Path | str, name: str = MIRROR_PROVENANCE_NAME
) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    """Every ``{source, mirror, sha256}`` record found in a mirror sidecar.

    Returns ``(records, files_read)``.  A sidecar that is absent, unreadable or
    shaped differently contributes nothing and is not an error -- a run that reads
    the canonical store directly needs no sidecar at all.
    """
    records: list[dict] = []
    files_read: list[str] = []
    seen: set[str] = set()
    for directory in search_dirs:
        try:
            candidate = Path(directory) / name
        except Exception:  # noqa: BLE001
            continue
        key = str(candidate).lower()
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        payload = _read_json(candidate)
        if not isinstance(payload, dict):
            continue
        rows = payload.get("files")
        if not isinstance(rows, list):
            continue
        kept = [
            row for row in rows
            if isinstance(row, dict) and row.get("sha256") and row.get("source")
        ]
        if kept:
            records.extend(kept)
            files_read.append(str(candidate.resolve()))
    return tuple(records), tuple(files_read)


def resolve_raster_source(
    path_as_read: Path | str,
    sha256: str,
    *,
    canonical_dir: Path | str = CANONICAL_TOPO_DIR,
    mirror_records: Sequence[dict] = (),
    mirror_files: Sequence[str] = (),
    verify_canonical_sha256: bool = False,
) -> dict:
    """Put the CANONICAL path back on a raster fingerprint, or say it could not be.

    THE DEFECT THIS CLOSES
    ----------------------
    Every raster path in the 2026-08-03 manifest pointed into a per-session agent
    scratchpad under ``%TEMP%``.  The bytes were right -- the mirror was sha256-
    verified on both sides -- but the seventeen paths were dead the moment the
    session ended, so the manifest fingerprinted files nobody could ever open.

    THE RULE
    --------
    A source path is recorded only when it is established BY SHA256:

    * the raster was read straight out of *canonical_dir* -- then the path read IS
      the source and nothing needs matching;
    * a mirror sidecar carries a record whose ``sha256`` equals this raster's, and
      that record names the source it was copied from;
    * *verify_canonical_sha256* is on and ``canonical_dir/<name>`` hashes equal.

    A file of the same NAME in the canonical store is **not** accepted as evidence
    on its own.  This project re-bakes rasters in place under unchanged names --
    :func:`sha256_of` says so in as many words -- so a matching name is not a
    matching scene, and the honest output is ``source_path=None`` with a status
    that says what would settle it.
    """
    read_path = Path(path_as_read)
    canonical = Path(canonical_dir)
    try:
        read_resolved = read_path.resolve()
    except Exception:  # noqa: BLE001
        read_resolved = read_path

    def _under(child: Path, parents: Sequence[Path]) -> bool:
        for parent in parents:
            try:
                child.relative_to(parent)
                return True
            except Exception:  # noqa: BLE001
                continue
        return False

    canonical_forms = [canonical]
    try:
        canonical_forms.append(canonical.resolve())
    except Exception:  # noqa: BLE001
        pass

    if _under(read_resolved, canonical_forms) or _under(read_path, canonical_forms):
        return {
            "source_path": str(read_path),
            "status": "CANONICAL",
            "matched_by": "the raster was read directly out of the canonical store",
            "canonical_dir": str(canonical),
            "evidence": (
                f"{read_path} lies under {canonical}; no matching was needed "
                "and none was done"
            ),
        }

    for record in mirror_records:
        if str(record.get("sha256", "")).lower() != str(sha256).lower():
            continue
        mirror_path = record.get("mirror")
        same_file = False
        if mirror_path:
            try:
                same_file = Path(mirror_path).resolve() == read_resolved
            except Exception:  # noqa: BLE001
                same_file = False
        return {
            "source_path": str(record.get("source")),
            "status": "MIRRORED",
            "matched_by": "sha256, against a mirror provenance sidecar",
            "canonical_dir": str(canonical),
            "evidence": (
                f"sha256 {sha256} appears in {list(mirror_files) or ['<sidecar>']} "
                f"as the fingerprint of {record.get('source')!r}, copied to "
                f"{mirror_path!r}"
                + (
                    "; that mirror path is the file this run read"
                    if same_file
                    else "; NOTE the sidecar's mirror path is not the path this run "
                         "read, so the match rests on the hash alone -- which is "
                         "the stronger evidence of the two"
                )
            ),
        }

    twin = canonical / read_path.name
    if verify_canonical_sha256 and twin.exists():
        twin_digest = sha256_of(twin)
        if twin_digest == sha256:
            return {
                "source_path": str(twin.resolve()),
                "status": "CANONICAL_VERIFIED",
                "matched_by": "sha256, recomputed on the canonical store",
                "canonical_dir": str(canonical),
                "evidence": f"{twin} hashes to {twin_digest}, equal to the raster read",
            }
        return {
            "source_path": None,
            "status": "CONFLICT",
            "matched_by": None,
            "canonical_dir": str(canonical),
            "evidence": (
                f"{twin} exists but hashes to {twin_digest}, NOT {sha256}. The "
                "canonical store holds a different scene under this name; the frame "
                "was drawn from the file that was read and this is recorded rather "
                "than reconciled"
            ),
        }

    return {
        "source_path": None,
        "status": "UNRESOLVED",
        "matched_by": None,
        "canonical_dir": str(canonical),
        "evidence": (
            f"{read_resolved} is not under {canonical}, no mirror provenance record "
            f"carries sha256 {sha256} "
            f"(sidecars read: {list(mirror_files) or 'none'}), and the canonical "
            "store was not hashed"
            + (
                f". A file named {read_path.name!r} does exist there; a NAME is not "
                "a fingerprint on this project, which re-bakes rasters in place "
                "under unchanged names, so it is not accepted as evidence. Pass "
                "verify_canonical_sha256=True, or write a "
                f"{MIRROR_PROVENANCE_NAME} beside the rasters"
                if twin.exists()
                else f". No file named {read_path.name!r} exists there either"
            )
        ),
    }


def basin_levels_for(epoch_ka: float) -> dict[str, BasinLevel]:
    """The three surfaces for one frame, each with the rule that produced it.

    The order is Caspian, Mediterranean, Pontic, and it is deliberate: the Pontic is
    the one that refuses -- its curve is on a 0.5 ka grid through this window, so
    eight of these seventeen epochs have no row -- and computing it LAST means its
    refusal carries the other two levels with it, into the placeholder frame, its
    sidecar and the manifest.  With the Pontic computed in the middle the frame
    would have reported one resolved level when two were in hand.

    Raises
    ------
    BasinLevelUnavailableError
        Any one of the three cannot be produced.  The message carries the original
        exception's class and text, because the epoch becomes a refusal frame and
        the frame's whole content is that sentence; ``.resolved`` carries whichever
        levels were already in hand when it failed.
    """
    from scripts import bake_black_sea as bbs
    from scripts import bake_khvalynian as bk

    epoch = float(epoch_ka)
    levels: dict[str, BasinLevel] = {}

    # --- Caspian: the F2 curve level converted at the threshold ---------------
    try:
        caspian = bk.caspian_threshold(bk.load_curve(), epoch, "defensible")
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed refusal
        raise BasinLevelUnavailableError(
            f"caspian level unavailable at {epoch:g} ka: "
            f"{type(exc).__name__}: {exc}",
            resolved=levels,
        ) from exc
    levels["caspian"] = BasinLevel(
        basin="caspian",
        level_m=float(caspian.threshold_m),
        frame="F1 (F2 curve level + dz_solid at the locus, converted at the threshold)",
        rule=f"bake_khvalynian.caspian_threshold(..., 'defensible') rule={caspian.rule}",
        provenance=caspian.provenance,
    )

    # --- Mediterranean: F3, derived rather than chosen -------------------------
    try:
        offset = float(bk.sea_surface_offset(epoch))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed refusal
        raise BasinLevelUnavailableError(
            f"mediterranean level unavailable at {epoch:g} ka: "
            f"{type(exc).__name__}: {exc}",
            resolved=levels,
        ) from exc
    levels["mediterranean"] = BasinLevel(
        basin="mediterranean",
        level_m=-offset,
        frame="F3 (-offset(t), the DEM's own ocean companion)",
        rule="-bake_khvalynian.sea_surface_offset(epoch)",
        provenance=(
            f"T = -offset({epoch:g} ka) = {-offset:+.6f} m. offset(t) is the median "
            "dTopo over far-field ice-free tropical continental land taken from the "
            "GLOBAL ICE-6G_C field, and the vertical-datum contract section 2 proves that "
            "for a cell ICE-6G treats as OCEAN this scalar is the EXACT land/sea "
            "companion of H_paleo: submerged <=> H_paleo < -offset(t), with no "
            "assumption that the sea-surface change is spatially uniform, because "
            "whatever local residual is baked into dz_engine cancels against the "
            "same scalar in the threshold. The Mediterranean is connected to the "
            "world ocean, so this is derived, not adopted. CAVEAT, stated rather "
            "than left to be discovered: ICE-6G has no lake model, so the Marmara "
            "-- which at -77.9 m is cut off from the Aegean by the ~-70 m "
            "Dardanelles and was a lake in reality -- is an ocean cell to the "
            "engine and inherits this surface. That is the same limitation "
            "contract section 7.6 records for the Black Sea, and it is why the Pontic has "
            "a curve of its own and the Marmara does not"
        ),
    )

    # --- Pontic: the F2 curve level converted at the threshold ----------------
    # This is the one that refuses at the eight quarter-ka epochs: the curve is on a
    # 0.5 ka grid through this window and basin_level will not take a neighbour's
    # row, because at 16.75 ka the two candidates are -58.3 m and -30.0 m and
    # "nearest" would be decided by DataFrame row order.
    try:
        import pandas as pd

        pontic_curve = pd.read_csv(bbs.PONTIC_CSV, comment="#")
        pontic = bbs.pontic_threshold(pontic_curve, epoch, "gradual")
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed refusal
        raise BasinLevelUnavailableError(
            f"pontic level unavailable at {epoch:g} ka: {type(exc).__name__}: {exc}",
            resolved=levels,
        ) from exc
    levels["pontic"] = BasinLevel(
        basin="pontic",
        level_m=float(pontic.threshold_m),
        frame="F1 (F2 curve level + dz_solid at the marker/locus, converted at the threshold)",
        rule=f"bake_black_sea.pontic_threshold(..., 'gradual') rule={pontic.rule}",
        provenance=pontic.provenance,
    )

    return levels


# ---------------------------------------------------------------------------
# THE PLAN
# ---------------------------------------------------------------------------


@dataclass
class EpochPlan:
    """Everything one frame needs, or the reason it has none."""

    epoch_ka: float
    raster_path: Optional[str] = None
    raster_sha256: Optional[str] = None
    raster_source: Optional[dict] = None
    dem_frame: Optional[str] = None
    levels: dict[str, BasinLevel] = field(default_factory=dict)
    seeds: tuple[ResolvedSeed, ...] = ()
    dz_bosphorus_m: Optional[float] = None
    refusal: Optional[str] = None

    @property
    def renderable(self) -> bool:
        return self.refusal is None

    def basin_seeds(self) -> list[BasinSeed]:
        return [s.to_basin_seed() for s in self.seeds]

    def to_dict(self) -> dict:
        return {
            "epoch_ka": float(self.epoch_ka),
            "raster_path_as_read": self.raster_path,
            "raster_source_path": (self.raster_source or {}).get("source_path"),
            "raster_sha256": self.raster_sha256,
            "raster_source_resolution": self.raster_source,
            "dem_vertical_frame": self.dem_frame,
            "levels": {k: v.to_dict() for k, v in self.levels.items()},
            "seeds": [s.to_dict() for s in self.seeds],
            "dz_bosphorus_m": self.dz_bosphorus_m,
            "refusal": self.refusal,
        }


def plan_epoch(
    epoch_ka: float,
    *,
    topo_dir: Path | str = KHV_TOPO_DIR,
    mirror_records: Sequence[dict] = (),
    mirror_files: Sequence[str] = (),
    verify_canonical_sha256: bool = False,
) -> EpochPlan:
    """Resolve one epoch: raster, its canonical source, frame tag, levels, seeds.

    Never raises for a *content* problem -- an epoch that cannot be planned comes
    back with ``refusal`` set and keeps its slot in the sequence.

    ``dz_bosphorus_m`` is resolved BEFORE the three basin levels, and the order is
    load-bearing rather than tidy.  It depends only on the epoch and the ICE-6G_C
    field, not on either curve, so computing it first means the eight quarter-ka
    epochs -- which refuse on the Pontic curve and only on the Pontic curve -- still
    carry it, and therefore still carry a full three-pair sill block naming what was
    EXPECTED at that epoch.  With it computed after the levels, as it was until
    2026-08-04, ``_sill_block`` saw ``dz_bosphorus_m is None`` and emitted an EMPTY
    sill list for exactly those eight frames: the manifest recorded no expectation,
    no measurement and no verdict for 8 of 17 slots, and their silence was
    indistinguishable from agreement.
    """
    plan = EpochPlan(epoch_ka=float(epoch_ka))
    try:
        raster = raster_for(epoch_ka, topo_dir)
        plan.raster_path = str(raster)
        plan.raster_sha256 = sha256_of(raster)
        plan.raster_source = resolve_raster_source(
            raster,
            plan.raster_sha256,
            mirror_records=mirror_records,
            mirror_files=mirror_files,
            verify_canonical_sha256=verify_canonical_sha256,
        )
        plan.dem_frame = declared_vertical_frame({"khv": str(raster)})
        if not plan.dem_frame:
            raise BasinLevelUnavailableError(
                f"{raster.name} declares no 'vertical_frame' tag. A raster that "
                "does not say what its elevations are above cannot have a sill "
                "expectation checked against it."
            )
        plan.dz_bosphorus_m = dz_at_bosphorus(epoch_ka)
        plan.levels = basin_levels_for(epoch_ka)
        plan.seeds = resolve_seeds(
            raster,
            {name: lv.level_m for name, lv in plan.levels.items()},
            epoch_ka=epoch_ka,
            raster_sha256=plan.raster_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - the refusal IS the frame's content
        plan.refusal = f"{type(exc).__name__}: {exc}"
        # Whatever DID resolve is kept.  Eight epochs of this window refuse only on
        # the Pontic curve; a frame reporting no levels at all would overstate its
        # own ignorance, and the Caspian and Mediterranean surfaces it already has
        # are the two a reader most wants beside that refusal.  Seeds are NOT kept:
        # a seed is checked against its own basin's level and a frame missing one
        # level was never checked as a whole.
        plan.levels = dict(getattr(exc, "resolved", None) or {})
        plan.seeds = ()
    return plan


def plan_sequence(
    epochs: Optional[Sequence[float]] = None,
    *,
    topo_dir: Path | str = KHV_TOPO_DIR,
    out_root: Optional[Path | str] = None,
    verify_canonical_sha256: bool = False,
) -> list[EpochPlan]:
    """One :class:`EpochPlan` per epoch, in descending age -- the order it plays in.

    The mirror provenance sidecar is read ONCE here and handed to every epoch, so
    seventeen epochs cost one file read rather than seventeen.

    The raster directory is checked for COMPLETENESS first
    (:func:`assert_raster_tree_complete`), before a single epoch is planned.  A
    directory that cannot supply the requested epochs raises
    :class:`IncompleteRasterTreeError`; it does not become seventeen refusal frames
    and it never becomes a short sequence wearing the window's name.
    """
    epochs = tuple(epoch_grid()) if epochs is None else tuple(float(e) for e in epochs)
    assert_raster_tree_complete(topo_dir, epochs)
    topo = Path(topo_dir)
    search = [topo, topo.parent, topo.parent.parent]
    if out_root is not None:
        search.append(Path(out_root))
    records, files = load_mirror_records(*search)
    return [
        plan_epoch(
            e,
            topo_dir=topo_dir,
            mirror_records=records,
            mirror_files=files,
            verify_canonical_sha256=verify_canonical_sha256,
        )
        for e in epochs
    ]


@dataclass
class InsetPlan:
    """One epoch's inset: the cropped raster, its own seeds, or the reason it has none.

    Separate from :class:`EpochPlan` because it is a separate grid with a separate
    registry.  Sharing the dataclass would have let an inset frame inherit the full
    scene's seeds by defaulting, which is the one thing a second extent must not do.
    """

    epoch_ka: float
    extent: InsetExtent
    raster_path: Optional[str] = None
    raster_sha256: Optional[str] = None
    crop: Optional[dict] = None
    dem_frame: Optional[str] = None
    seeds: tuple[ResolvedSeed, ...] = ()
    refusal: Optional[str] = None

    @property
    def renderable(self) -> bool:
        return self.refusal is None and self.raster_path is not None

    def basin_seeds(self) -> list[BasinSeed]:
        return [s.to_basin_seed() for s in self.seeds]

    def to_dict(self) -> dict:
        return {
            "epoch_ka": float(self.epoch_ka),
            "extent": self.extent.to_dict(),
            "raster_path": self.raster_path,
            "raster_sha256": self.raster_sha256,
            "crop": self.crop,
            "dem_vertical_frame": self.dem_frame,
            "seeds": [s.to_dict() for s in self.seeds],
            "refusal": self.refusal,
        }


def inset_sill_expectations(plan: InsetPlan) -> dict:
    """The expectation for every pair the INSET declares, and only those.

    ``assign_basins`` refuses a key naming basins the call does not have, so the
    three-pair mapping :func:`sill_expectations_for` builds for the full scene cannot
    be handed to a two-basin inset.  The VALUE is the same quantity either way --
    ``MANYCH_SILL_PATH_M`` converted at this epoch -- because the expectation is a
    property of the terrain and the vertical frame, not of the grid.  What is
    grid-specific is the MEASUREMENT it is checked against, and that is the whole
    point of arming the guard on the inset separately: a coarser grid inflates this
    saddle by +8.30 m at 2000 px and +5.24 m at 4000 px, which is the difference
    between 'endorheic' and 'spilling'.
    """
    if plan.dem_frame is None:
        raise BasinLevelUnavailableError(
            f"the {plan.extent.name!r} inset raster at {plan.epoch_ka:g} ka declares "
            "no vertical_frame, so no expectation can be frame-checked against it."
        )
    out: dict[tuple[str, str], Any] = {}
    for pair in plan.extent.pairs():
        if set(pair) == {"caspian", "pontic"}:
            out[pair] = manych_expectation(plan.epoch_ka, plan.dem_frame)
        else:  # pragma: no cover - the registry declares exactly one pair today
            raise InsetExtentError(
                f"the inset declares the basin pair {pair!r} and this module has no "
                "measured crest for it. An UnmeasuredSill would let the frame render "
                "with that saddle unchecked; state the crest or drop the basin."
            )
    return out


def plan_inset(
    plan: EpochPlan,
    extent: InsetExtent,
    out_dir: Path | str,
) -> InsetPlan:
    """Crop *plan*'s raster to *extent* and resolve the inset's own seeds in it.

    Never raises for a content problem, exactly like :func:`plan_epoch`: an inset
    that cannot be planned comes back with ``refusal`` set, and the frame is then
    composed without it rather than silently dropped -- a composed canvas that is
    missing its inset without saying so would be indistinguishable from a run that
    was never asked for one.
    """
    inset = InsetPlan(epoch_ka=float(plan.epoch_ka), extent=extent)
    if not plan.renderable:
        inset.refusal = (
            f"the full-extent frame at {plan.epoch_ka:g} ka has no imagery, so there "
            f"is nothing to inset into: {plan.refusal}"
        )
        return inset
    try:
        target = Path(out_dir) / f"khv_{plan.epoch_ka:g}ka__{extent.name}.tif"
        inset.crop = crop_to_extent(
            plan.raster_path, extent, target, src_sha256=plan.raster_sha256
        )
        inset.raster_path = inset.crop["inset_path"]
        inset.raster_sha256 = inset.crop["inset_sha256"]
        inset.dem_frame = declared_vertical_frame({"khv_inset": inset.raster_path})
        if inset.dem_frame != plan.dem_frame:
            raise InsetCropError(
                f"the crop declares vertical_frame {inset.dem_frame!r} and the "
                f"raster it was cut from declares {plan.dem_frame!r}. The crop copies "
                "the source tags; a difference here means it did not, and an "
                "expectation checked against the wrong frame is wrong by dz_sill_m."
            )
        levels = {
            basin: plan.levels[basin].level_m
            for basin in extent.basins()
            if basin in plan.levels
        }
        missing = [b for b in extent.basins() if b not in levels]
        if missing:
            raise BasinLevelUnavailableError(
                f"the inset declares basin(s) {missing!r} and the epoch carries no "
                f"level for them at {plan.epoch_ka:g} ka."
            )
        inset.seeds = resolve_seeds(
            inset.raster_path,
            levels,
            epoch_ka=plan.epoch_ka,
            raster_sha256=inset.raster_sha256,
            boxes=extent.seed_boxes,
        )
    except Exception as exc:  # noqa: BLE001 - the refusal IS the inset's content
        inset.refusal = f"{type(exc).__name__}: {exc}"
        inset.seeds = ()
    return inset


def assert_route_inside_the_inset(
    inset: InsetPlan, basins_sidecar: Optional[dict]
) -> dict:
    """After the flood: is the measured crest, and the route through it, inside?

    The bounds are checked against a MEASUREMENT here, not against a claim.  The
    crest cell is re-measured by every render; the route extent is what the sidecar
    records for the pair.  A crest outside the extent means the inset is drawing a
    different channel from the one the scene is about, and that is a refusal.
    """
    out: dict[str, Any] = {
        "extent": inset.extent.name,
        "bounds_wsen": list(inset.extent.bounds()),
        "checked": False,
        "crest_lonlat": None,
        "crest_inside": None,
        "route_lat_span": None,
        "route_lat_inside": None,
        "route_lon_span": (
            "NOT RECORDED: sill_route_stats carries the route's north_lat and "
            "south_lat and no longitudes at all, so the longitude half of this check "
            "cannot be run from the sidecar. It is left null rather than inferred "
            "from the crest longitude, which is one cell of a route thousands of "
            "cells long"
        ),
        "verdict": "NOT_CHECKED: the inset produced no basins sidecar",
    }
    sills = (basins_sidecar or {}).get("sills") or []
    entry = next(
        (
            s for s in sills
            if sorted(s.get("pair") or []) == ["caspian", "pontic"]
        ),
        None,
    )
    if entry is None or entry.get("lon") is None:
        return out
    lon, lat = float(entry["lon"]), float(entry["lat"])
    out.update(
        checked=True,
        crest_lonlat=[lon, lat],
        crest_inside=inset.extent.contains(lon, lat),
    )
    stats = entry.get("route_stats") or {}
    south, north = stats.get("south_lat"), stats.get("north_lat")
    if south is not None and north is not None:
        out["route_lat_span"] = [float(south), float(north)]
        out["route_lat_inside"] = bool(
            inset.extent.lat_min <= float(south)
            and float(north) <= inset.extent.lat_max
        )
    out["verdict"] = (
        "CREST_INSIDE"
        if out["crest_inside"] and out["route_lat_inside"] is not False
        else "CREST_OUTSIDE_THE_INSET"
        if not out["crest_inside"]
        else "ROUTE_LEAVES_THE_INSET_IN_LATITUDE"
    )
    if not out["crest_inside"]:
        raise InsetExtentError(
            f"the {inset.extent.name!r} inset measured its caspian<->pontic crest at "
            f"{lon:.6f} E / {lat:.6f} N, outside its own bounds "
            f"{inset.extent.bounds()}. rasterio clips a window to the raster, so this "
            "cannot be an out-of-range read: it means the inset's controlling col is "
            "a different col from the scene's, and the picture would be about a "
            "different channel from the caption."
        )
    return out


def sequence_limits_for(
    plans: Sequence[EpochPlan],
    *,
    config: Optional[RenderConfig] = None,
    computed_by: str = "scripts/animate_khvalynian_spill.py",
    inset_plans: Optional[Sequence[InsetPlan]] = None,
) -> Optional[SequenceLimits]:
    """One colour mapping over every RENDERABLE epoch of the sequence.

    The depth ramp is built on the **Pontic** surface and says so.  That is a
    decision, not a default: on the 16.5 ka frame the three levels are +29.944 m,
    -18.020 m and -77.906 m, and building the ramp on the Caspian instead would
    move every Pontic depth by 47.96 m.  The Pontic is the receiving basin -- the
    basin the spill is ABOUT -- and it is the one whose bathymetry the sequence
    exists to show filling.  The other two levels ride along in
    ``other_basin_levels_m`` so the envelope still bounds them.

    Returns ``None`` when no epoch is renderable: a mapping shared by an empty
    sequence is not a mapping.

    THE INSET IS PASSED THROUGH THE SAME PASS, ON PURPOSE
    ----------------------------------------------------
    When *inset_plans* is given, each epoch's ``SequenceEpoch`` names BOTH rasters.
    ``compute_sequence_limits`` opens every raster path separately -- it does not
    merge them -- so the inset raster is genuinely READ, and the fingerprint the
    limits record for it is a fingerprint of a file the pass actually saw.  That
    matters for two independent reasons:

    * ``SequenceLimits.fingerprints_for`` refuses a frame built from a raster the
      limits never read.  Rendering the inset through the scene's limits therefore
      requires the inset raster to be IN them; the alternative -- appending a
      fingerprint the pass never opened -- would put a false statement in the one
      object whose whole purpose is provenance.
    * the aggregation is min/max over rasters, so adding the inset can only WIDEN the
      land ramp, never crop it.  The inset holds a strict subset of the source's
      cells (:func:`crop_to_extent` proves that cell for cell), so the true envelope
      ``dem_data_*`` cannot move at all; only the percentile ENDPOINTS can, and only
      outwards.  Both are measured and reported rather than assumed -- see
      ``inset.colour_mapping`` in the manifest.
    """
    usable = [p for p in plans if p.renderable]
    if not usable:
        return None
    insets = {
        float(i.epoch_ka): i
        for i in (inset_plans or ())
        if i.renderable
    }

    def _tiles(plan: EpochPlan) -> dict:
        tiles = {"khv": plan.raster_path}
        inset = insets.get(float(plan.epoch_ka))
        if inset is not None:
            tiles["khv_inset"] = inset.raster_path
        return tiles

    epochs = [
        SequenceEpoch.from_basin_seeds(
            p.epoch_ka,
            _tiles(p),
            p.basin_seeds(),
            depth_reference="pontic",
            water_level_rule=(
                "the Pontic surface, "
                f"{p.levels['pontic'].level_m:+.4f} m at {p.epoch_ka:g} ka, from "
                f"{p.levels['pontic'].rule}. Nominated over the Caspian "
                f"({p.levels['caspian'].level_m:+.4f} m) and the Mediterranean "
                f"({p.levels['mediterranean'].level_m:+.4f} m) because the Pontic is "
                "the RECEIVING basin: the sequence is about its bathymetry filling, "
                "and AT THIS EPOCH the three surfaces span "
                f"{max(lv.level_m for lv in p.levels.values()) - min(lv.level_m for lv in p.levels.values()):.2f} m, "
                "so the choice moves every depth in the frame"
            ),
        )
        for p in usable
    ]
    return compute_sequence_limits(epochs, config=config, computed_by=computed_by)


#: Which extremum each pinned quantity is: ``"min"`` = the sequence takes the
#: smallest per-epoch value, ``"max"`` = the largest.  Used to attribute each one
#: back to the epoch that produced it.
_EXTREMUM_KIND = {
    "land_min_m": "min",
    "land_max_m": "max",
    "depth_max_m": "max",
    "ice_max_m": "max",
    "dem_data_min_m": "min",
    "dem_data_max_m": "max",
    "depth_data_max_m": "max",
}


def attribute_extrema(
    plans: Sequence[EpochPlan],
    limits: SequenceLimits,
    *,
    config: Optional[RenderConfig] = None,
    inset_plans: Optional[Sequence[InsetPlan]] = None,
) -> dict:
    """Say WHICH epoch produced each pinned extremum, and what every epoch's own was.

    ``SequenceLimits`` records the epochs it was computed over and the raster bytes
    that fed it, but not which of them each number came off -- and a pinned mapping
    whose owner cannot be named is a mapping nobody can argue with.  This re-runs
    the same helper one epoch at a time and matches.

    It costs a second limits pass, about 1.5 % of one native frame for a nine-epoch
    sequence, and it reads the same bytes the same way -- so a value here that does
    NOT match the sequence's is a real disagreement rather than a different method.
    It is reported as ``"UNATTRIBUTED"`` rather than rounded into agreement.

    ``shade_min`` / ``shade_max`` are deliberately absent: they come from
    :func:`~paleoeurope.viz.paleocostline_render.analytic_shade_range`, which is
    geometry, so no epoch produces them and naming one would be a fiction.
    """
    usable = [p for p in plans if p.renderable]
    insets = {
        float(i.epoch_ka): i for i in (inset_plans or ()) if i.renderable
    }
    per_epoch: dict[float, dict] = {}
    for plan in usable:
        tiles = {"khv": plan.raster_path}
        inset = insets.get(float(plan.epoch_ka))
        if inset is not None:
            # The SAME raster set the sequence pass saw for this epoch.  Attributing
            # a widened limit against the full raster alone would report
            # "UNATTRIBUTED" for a number the inset legitimately produced.
            tiles["khv_inset"] = inset.raster_path
        one = compute_sequence_limits(
            [
                SequenceEpoch.from_basin_seeds(
                    plan.epoch_ka,
                    tiles,
                    plan.basin_seeds(),
                    depth_reference="pontic",
                    water_level_rule=(
                        "single-epoch pass, for extremum attribution only"
                    ),
                )
            ],
            config=config,
            depth_percentile=limits.depth_percentile,
            computed_by="attribute_extrema",
        )
        per_epoch[plan.epoch_ka] = {name: getattr(one, name) for name in _EXTREMUM_KIND}

    attribution: dict[str, Any] = {}
    for name, kind in _EXTREMUM_KIND.items():
        target = getattr(limits, name)
        owners = [
            epoch
            for epoch, values in per_epoch.items()
            if values[name] is not None
            and target is not None
            and float(values[name]) == float(target)
        ]
        attribution[name] = {
            "value": None if target is None else float(target),
            "extremum": kind,
            "produced_by_epoch_ka": owners or "UNATTRIBUTED",
            "per_epoch": {
                f"{epoch:g}": (None if values[name] is None else float(values[name]))
                for epoch, values in sorted(per_epoch.items(), reverse=True)
            },
        }
    attribution["method"] = (
        "each quantity recomputed by compute_sequence_limits over ONE epoch at a "
        "time, then matched against the sequence value by exact float equality. "
        "'UNATTRIBUTED' means no single epoch reproduces the sequence number, which "
        "would be a real disagreement and is reported rather than rounded away. "
        "shade_min/shade_max are absent because analytic_shade_range is geometry: "
        "no epoch produces them and none can."
    )
    return attribution


# ---------------------------------------------------------------------------
# FRAMES
# ---------------------------------------------------------------------------


#: The refusal placeholder is laid out on a figure this many inches tall, whatever
#: the canvas is in pixels, and the dpi is derived from the two.  Point sizes are
#: physical, so hard-coding a dpi instead would make the text shrink to illegibility
#: exactly when the canvas grows -- measured: 28 pt on a 21-inch figure (2100 px at
#: dpi=100) reads as a smudge, and the refusal is the entire content of this frame.
#: 14.0 is ``RenderConfig.fig_size_in``, so a refusal frame and a rendered frame put
#: text on the page at the same scale.
REFUSAL_FIGURE_IN = 14.0


def refusal_frame(
    epoch_ka: float,
    refusal: str,
    out_path: Path | str,
    size_px: tuple[int, int],
    *,
    levels: Optional[dict[str, BasinLevel]] = None,
    dpi: Optional[float] = None,
) -> Path:
    """A visibly hatched placeholder that keeps a refused epoch's slot.

    It is deliberately unmistakable -- diagonal hatching, a red border, the word
    REFUSED at the top -- because the alternative failure mode is a viewer taking a
    blank or a repeated frame for data.  It carries the full refusal text, so the
    animation itself says why the epoch is missing without anybody opening the
    manifest, and a sidecar JSON beside it carries
    ``{"threshold": null, "refusal": "<class>: <message>"}`` verbatim.

    *size_px* is not negotiable and not derived from the text: it is the sequence's
    canvas, so this frame is byte-compatible with a rendered one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width_px, height_px = int(size_px[0]), int(size_px[1])
    if dpi is None:
        dpi = max(width_px, height_px) / REFUSAL_FIGURE_IN
    dpi = float(dpi)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            facecolor="#f4eaea", edgecolor="#b00020", linewidth=8,
            hatch="////", zorder=0,
        )
    )
    ax.text(
        0.5, 0.945, "FRAME REFUSED", ha="center", va="center",
        fontsize=34, fontweight="bold", color="#b00020",
    )
    ax.text(
        0.5, 0.895, f"{float(epoch_ka):g} ka BP", ha="center", va="center",
        fontsize=24, color="#222222",
    )
    ax.text(
        0.5, 0.855,
        "slot kept, full duration held -- dropping it would compress time, "
        "interpolating it would invent a level",
        ha="center", va="center", fontsize=12, style="italic", color="#444444",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.94,
              "edgecolor": "none"},
    )
    ax.text(
        0.5, 0.80, _wrap(refusal, 96), ha="center", va="top",
        multialignment="left",
        fontsize=11, color="#111111", family="monospace", linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "alpha": 0.94,
              "edgecolor": "#b00020"},
    )
    if levels:
        known = "\n".join(
            f"{name:>14s}  {lv.level_m:+11.4f} m   {lv.frame}"
            for name, lv in sorted(levels.items())
        )
        ax.text(
            0.5, 0.22,
            "levels that DID resolve at this epoch\n" + known,
            ha="center", va="top", multialignment="left",
            fontsize=12, color="#222222", family="monospace",
            linespacing=1.4,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "alpha": 0.94,
                  "edgecolor": "#999999"},
        )
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    # The rendered frames come back RGB; keep the sequence one mode so the writer
    # is not quietly flattening an alpha channel on some frames and not others.
    from PIL import Image

    with Image.open(out_path) as image:
        image.convert("RGB").save(out_path)

    sidecar = out_path.with_suffix(".refusal.json")
    sidecar.write_text(
        json.dumps(
            {
                "epoch_ka": float(epoch_ka),
                "threshold": None,
                "refusal": refusal,
                "levels_that_resolved": (
                    {k: v.to_dict() for k, v in levels.items()} if levels else {}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path.resolve()


def _wrap(text: str, width: int, max_lines: int = 34) -> str:
    """Wrap a refusal onto the frame, saying so out loud if it did not fit.

    A truncated refusal that does not admit it is a frame claiming to state its own
    reason while withholding part of it; the full text is always in the sidecar and
    the manifest, and the marker is how a reader of the PICTURE knows to go there.
    """
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1]
        lines.append(
            f"[... truncated to fit the frame; the full {len(str(text)):,}-character "
            "refusal is in the .refusal.json sidecar and the manifest]"
        )
    return "\n".join(lines)


def read_png_size(path: Path | str) -> tuple[int, int]:
    """``(width, height)`` in pixels."""
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def canvas_size(
    sizes: Iterable[tuple[int, int]],
    fallback: tuple[int, int] = FALLBACK_CANVAS_PX,
) -> tuple[int, int]:
    """The one canvas every frame is padded to: the element-wise maximum.

    Maximum rather than minimum because cropping would remove picture, and this
    step must not be able to change what a frame shows.  *fallback* is used only
    when the sequence produced no rendered frame at all, i.e. when there is no
    measured size to pad to.
    """
    sizes = list(sizes)
    if not sizes:
        return (int(fallback[0]), int(fallback[1]))
    return (max(w for w, _ in sizes), max(h for _, h in sizes))


def pad_to_canvas(
    src: Path | str, dst: Path | str, size_px: tuple[int, int]
) -> dict:
    """Centre one frame on the sequence canvas; never scale, never crop.

    Returns the padding actually applied, which goes in the manifest: a run where
    every ``pad_*`` is 0 is a run where nothing moved, and that is a measurement
    rather than an assurance.
    """
    from PIL import Image

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        frame = image.convert("RGB")
        width, height = frame.size
        target_w, target_h = int(size_px[0]), int(size_px[1])
        if width > target_w or height > target_h:
            raise FrameSizeError(
                f"{Path(src).name} is {width}x{height} px, larger than the canvas "
                f"{target_w}x{target_h}. Padding never crops, so this cannot be "
                "resolved here: the canvas is the element-wise maximum over the "
                "frames and something produced a frame outside the set it was "
                "computed from."
            )
        left = (target_w - width) // 2
        top = (target_h - height) // 2
        canvas = Image.new("RGB", (target_w, target_h), CANVAS_PAD_RGB)
        canvas.paste(frame, (left, top))
        canvas.save(dst)
    return {
        "source": str(Path(src).resolve()),
        "canvas": str(dst.resolve()),
        "source_size_px": [width, height],
        "canvas_size_px": [target_w, target_h],
        "pad_left_px": int(left),
        "pad_top_px": int(top),
        "pad_right_px": int(target_w - width - left),
        "pad_bottom_px": int(target_h - height - top),
    }


def assert_uniform_size(paths: Sequence[Path | str]) -> tuple[int, int]:
    """Refuse a sequence whose frames are not all one size."""
    sizes = [read_png_size(p) for p in paths]
    unique = sorted(set(sizes))
    if len(unique) != 1:
        detail = ", ".join(
            f"{Path(p).name} {w}x{h}" for p, (w, h) in zip(paths, sizes)
        )
        raise FrameSizeError(
            f"the sequence holds {len(unique)} distinct frame sizes {unique}; an "
            f"animation built from them jitters. Frames: {detail}"
        )
    return unique[0]


# ---------------------------------------------------------------------------
# THE ONE PALETTE  (the colour mapping the ENCODER must not re-derive)
# ---------------------------------------------------------------------------

#: Entries in the colour table.  GIF stores the table size as a 3-bit exponent, so
#: the only legal sizes are the powers of two from 2 to 256, and 256 is the ceiling
#: the format offers.  There is no larger GIF: see
#: :func:`describe_palette_headroom` for what that costs on this content and what
#: the alternatives are.
GIF_PALETTE_SIZE = 256

#: How a frame's colours are mapped onto the master palette.  ``"NONE"`` is the
#: shipped choice and it is a decision with a cost, stated here rather than left in
#: a default: Floyd-Steinberg would lower the per-pixel error (it trades a colour
#: error for a spatial one) but its error-diffusion pattern is a function of the
#: frame's own pixels, so it moves between adjacent epochs even where the terrain
#: has not.  That is the same shimmer ``RenderConfig.sequence_limits`` exists to
#: remove, reintroduced as texture instead of as hue.  :func:`measure_palette_cost`
#: reports BOTH, so the trade is a measurement in the manifest and not an assertion
#: here.
PALETTE_DITHER = "NONE"

#: How the one palette is derived from the sequence.  **The decision is taken; it
#: was open until 2026-08-04 and an open decision in a shipped artifact is a
#: decision made by whoever reads it last.**
#:
#: THE MEASUREMENT THAT SETTLED IT -- ON THE REAL FRAMES, NOT ON A PROXY
#: --------------------------------------------------------------------
#: The predecessor left MEDIANCUT vs MAXCOVERAGE open with numbers taken on 13
#: ``black_sea_corrected`` map frames -- same renderer, same ramp, a different scene
#: -- and said in as many words that it should be re-measured when a khvalynian
#: frame existed.  Nine now do.  Measured on the nine real canvas frames of
#: ``spill_0.25ka_17_slots`` (2964x1250 px, ``dither="NONE"``, 256 entries, all
#: three quantizers filling all 256), error being the per-pixel RGB Euclidean
#: distance to the source PNG on a 0..441.67 scale, reproduced twice:
#:
#:     recipe        mean/px   worst frame   max/px    pixels changed
#:     MEDIANCUT      2.9808      3.0126     212.899      49.3 %
#:     MAXCOVERAGE    4.8884      4.9196      22.561      51.2 %
#:     FASTOCTREE     3.2847      3.3010      60.473      99.0 %
#:
#: **MAXCOVERAGE is taken, and what it costs is stated with what it buys.**  The
#: deciding quantity is the WORST pixel, not the mean, because on a hypsometric map
#: the colour ramp IS the data: a reader takes an elevation off a hue, so one pixel
#: drawn 212.9 RGB units from its true colour -- 48.2 % of the largest error the
#: metric can express -- is a wrong elevation, while 4.9 units everywhere (2.82 per
#: channel, i.e. at the scale of the just-noticeable difference between two flat
#: patches rather than far above it) costs contrast and not correctness.  The mean
#: cost is stated, not dismissed.  MAXCOVERAGE bounds the worst pixel
#: **9.4x tighter** (22.561 against 212.899) for **+1.908 RGB units of mean** -- a
#: 64 % higher mean error that is invisible, against a 90 % smaller worst case that
#: is not.  FASTOCTREE is dominated: worse than MEDIANCUT on the mean AND 2.7x worse
#: than MAXCOVERAGE on the worst pixel, while touching 99 % of pixels.
#:
#: The cost of the switch is that the proxy-set numbers this module used to quote
#: were MEDIANCUT numbers; they have been replaced above by the real-frame
#: measurement, and :func:`compare_quantizers` re-measures all three arms on every
#: run's own frames.  :func:`palette_method_decision` writes the rule, the arms and
#: the verdict into ``master_palette.json``, and it CONTRADICTS ITSELF OUT LOUD if a
#: future run's frames no longer support the choice.
PALETTE_METHOD = "MAXCOVERAGE"


def _pil_quantize_method(name: str):
    from PIL import Image

    try:
        return getattr(Image.Quantize, str(name).upper())
    except AttributeError as exc:  # noqa: PERF203
        raise PaletteError(
            f"unknown quantizer {name!r}; PIL offers "
            f"{[m.name for m in Image.Quantize]!r}"
        ) from exc


def _pil_dither(name: str):
    from PIL import Image

    try:
        return getattr(Image.Dither, str(name).upper())
    except AttributeError as exc:  # noqa: PERF203
        raise PaletteError(
            f"unknown dither {name!r}; PIL offers {[d.name for d in Image.Dither]!r}"
        ) from exc


@dataclass(frozen=True)
class MasterPalette:
    """The ONE colour table every frame of the animation is mapped through.

    WHY IT IS AN OBJECT AND NOT 768 BYTES ON THE SIDE
    ------------------------------------------------
    A palette without the sequence it was derived from is exactly as re-derivable
    as a ``BasinSeed`` without its box -- which is to say not at all.  This carries
    the frames it was quantized over, their sha256s, the quantizer, the number of
    distinct colours the sequence actually holds, and how many table slots the
    quantizer filled, so a reader can rebuild it or contradict it.

    Attributes
    ----------
    entries
        ``size`` RGB triples, **pairwise distinct** -- see :func:`build_master_palette`
        for why a repeated entry is a correctness problem rather than waste.
    n_from_quantizer
        How many of the entries the quantizer produced.  Only these are ever
        referenced by a pixel; the rest are filler.
    n_distinct_in_sequence
        How many distinct RGB triples the source frames hold between them.  The
        ratio of this to ``n_from_quantizer`` is the compression the format forces.
    """

    entries: tuple[tuple[int, int, int], ...]
    size: int
    method: str
    dither: str
    n_from_quantizer: int
    n_distinct_in_sequence: int
    filler_rule: str
    built_over: tuple[str, ...]
    frames_sha256: tuple[str, ...]
    frame_size_px: tuple[int, int]
    #: Pixels the quantizer actually saw: ``n_frames * width * height``.  Recorded
    #: because "quantized once over the sequence" is only true if the whole sequence
    #: went in, and this is the number that says so.
    quantized_over_px: int = 0

    @property
    def n_filler(self) -> int:
        return int(self.size - self.n_from_quantizer)

    def as_bytes(self) -> bytes:
        """Exactly ``size * 3`` bytes, RGBRGB..., ready for PIL's ``palette=``."""
        return bytes(channel for triple in self.entries for channel in triple)

    def palette_image(self, *, entries: Optional[int] = None):
        """A 1x1 ``P`` image carrying this palette -- what ``Image.quantize`` wants.

        *entries* truncates the table.  The mapping is done against the quantizer's
        own entries ONLY (``entries=n_from_quantizer``), so a filler slot can never
        win a nearest-colour contest and repaint a pixel.
        """
        from PIL import Image

        count = self.n_from_quantizer if entries is None else int(entries)
        image = Image.new("P", (1, 1))
        image.putpalette(
            bytes(
                channel for triple in self.entries[:count] for channel in triple
            ),
            "RGB",
        )
        return image

    def to_dict(self) -> dict:
        return {
            "size": int(self.size),
            "method": self.method,
            "dither": self.dither,
            "n_from_quantizer": int(self.n_from_quantizer),
            "n_filler": int(self.n_filler),
            "n_distinct_in_sequence": int(self.n_distinct_in_sequence),
            "filler_rule": self.filler_rule,
            "entries_rgb": [list(triple) for triple in self.entries],
            "sha256_of_the_table": hashlib.sha256(self.as_bytes()).hexdigest(),
            "built_over": list(self.built_over),
            "frames_sha256": list(self.frames_sha256),
            "frame_size_px": list(self.frame_size_px),
            "quantized_over_px": int(self.quantized_over_px),
            "quantized_over": (
                f"{len(self.built_over)} frames x {self.frame_size_px[0]}x"
                f"{self.frame_size_px[1]} px = {self.quantized_over_px:,} pixels, "
                "in ONE pass. Not a sample: a palette derived from an unstated "
                "subset would map every colour the subset missed to a neighbour "
                "with no record of it"
            ),
        }


#: The rule that fills a colour table the quantizer did not fill, recorded on every
#: palette so a reader never has to guess whether a slot carries data.
FILLER_RULE = (
    "the lowest 0xRRGGBB values, in ascending order, that the quantizer did not "
    "already produce. A filler slot is NEVER referenced by a pixel: the frames are "
    "mapped against a palette image truncated to n_from_quantizer entries, so a "
    "filler colour cannot win a nearest-colour contest. They exist only so that the "
    "colour table holds 2^k PAIRWISE DISTINCT entries -- PIL's "
    "GifImagePlugin._normalize_palette builds a colour->index dict from the table, "
    "so a colour appearing at two indices resolves to one of them, the other index "
    "is filled from whatever slot is left over, and remap_palette then moves pixels "
    "onto an index whose colour it did not have. A repeated entry is therefore a "
    "silent repaint, not wasted space"
)


def _distinct_filler(used: set[tuple[int, int, int]], count: int) -> list[tuple[int, int, int]]:
    """``count`` RGB triples not in *used*, by :data:`FILLER_RULE`."""
    out: list[tuple[int, int, int]] = []
    value = 0
    limit = 1 << 24
    while len(out) < count:
        if value >= limit:  # pragma: no cover - needs >16.7M distinct colours
            raise PaletteError(
                "the 24-bit RGB space is exhausted; a colour table cannot be filled "
                "with distinct entries"
            )
        triple = ((value >> 16) & 255, (value >> 8) & 255, value & 255)
        if triple not in used:
            out.append(triple)
            used.add(triple)
        value += 1
    return out


def _frame_arrays(frame_paths: Sequence[Path | str]) -> list[np.ndarray]:
    """Every frame as an ``(h, w, 3)`` uint8 array, refusing a mixed sequence."""
    from PIL import Image

    if not frame_paths:
        raise PaletteError(
            "a palette cannot be quantized over an empty sequence: a mapping shared "
            "by no frames is not a mapping"
        )
    arrays: list[np.ndarray] = []
    for path in frame_paths:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise FrameSizeError(
            f"the frames handed to the quantizer have {len(shapes)} distinct shapes "
            f"{sorted(shapes)}; pad them to one canvas first. One palette over "
            "frames of different sizes would weight the quantizer by frame area."
        )
    return arrays


def _packed_colours(array: np.ndarray) -> np.ndarray:
    flat = array.reshape(-1, 3).astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def build_master_palette(
    frame_paths: Sequence[Path | str],
    *,
    size: int = GIF_PALETTE_SIZE,
    method: str = PALETTE_METHOD,
    dither: str = PALETTE_DITHER,
) -> MasterPalette:
    """Quantize ONCE, over the whole sequence, and refuse a malformed table.

    THE MECHANISM
    -------------
    Every frame is concatenated into one image and quantized once.  PIL's median cut
    works from the image's colour histogram, so concatenation is exactly "quantize
    the sequence": each frame contributes its own pixels, in proportion, and no
    frame gets a table of its own.  The frames are required to be one size first --
    otherwise a larger frame would weight the palette by area rather than by
    content, which is a decision nobody took.

    WHAT ONE SHARED TABLE COSTS ON REAL MAP FRAMES, MEASURED
    --------------------------------------------------------
    The proxy this paragraph used to quote -- 13 ``black_sea_corrected`` frames,
    used "because the 17 frames this sequence ships today are hatched refusal
    placeholders" -- is retired: **nine real map frames exist**, and the numbers
    below are theirs.  Measured on the 17-slot run's own canvas (2964x1250 px,
    9 map frames + 8 hatched placeholders), ``dither="NONE"``, error being the
    per-pixel RGB Euclidean distance to the source PNG:

    ===============================  =========  =============  ===============
    arm                              mean/px    max/px         pixels changed
    ===============================  =========  =============  ===============
    ONE shared table                   3.765      89.04            (per run)
    a table per frame (ADAPTIVE)       1.246     224.72            (per run)
    ===============================  =========  =============  ===============

    So one shared table costs **+2.52 RGB units of mean per-pixel error** on that
    set, and its WORST pixel is **better by 135.7 units** -- a shared table has no
    frame it is allowed to neglect, while a per-frame table can spend all 256
    entries on an easy frame and still miss badly on a hard one.  Those 17 frames
    hold **128 443 distinct colours**, i.e. **501.7 per palette entry**, so 256
    entries are visibly tight on this content and the error above is the price --
    see :func:`describe_palette_headroom`, and note that "use more colours" is not a
    setting GIF has.  (Both figures are MEDIANCUT-era, i.e. the shipped GIF of
    2026-08-04; the quantizer changed the same day and the per-run numbers below are
    the ones to read.)

    :func:`measure_palette_cost` recomputes both columns on the run's own frames and
    puts them in the manifest, so no run relies on the numbers above.

    WHY EVERY ENTRY IS DISTINCT
    ---------------------------
    See :data:`FILLER_RULE`.  A repeated colour-table entry makes PIL's palette
    normalisation permute indices non-identically, which repaints pixels silently.
    The invariant is enforced here and asserted again after the file is written.

    THE WHOLE SEQUENCE, NEVER A SAMPLE
    ----------------------------------
    A 34-frame native sequence is ~150 M pixels once stacked, over PIL's
    ``MAX_IMAGE_PIXELS`` decompression-bomb guard (89 478 485).  That guard is
    lifted for the duration of the quantization, deliberately and with the reason
    recorded here: it exists to stop a small hostile FILE from decoding into a huge
    array, and this array is not decoded from anything -- it is built in memory from
    frames this module produced and has already measured.  The alternative would be
    to quantize a SAMPLE, and a palette derived from an unstated subset is exactly
    the class of defect this module exists to remove: every colour the sample missed
    would be silently mapped to a neighbour and would appear in no record.  The
    stacked pixel count is written into the palette's own record either way.

    Raises
    ------
    PaletteError
        *size* is not a power of two in 2..256, or the sequence is empty.
    FrameSizeError
        The frames are not all one size.
    """
    from PIL import Image

    size = int(size)
    if size < 2 or size > 256 or (size & (size - 1)):
        raise PaletteError(
            f"size={size} is not a power of two in 2..256. GIF stores the colour "
            "table size as a 3-bit exponent, so no other value can be written."
        )
    arrays = _frame_arrays(frame_paths)
    height, width, _ = arrays[0].shape
    total_px = height * len(arrays) * width

    distinct = np.unique(np.concatenate([_packed_colours(a) for a in arrays]))
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        stack = Image.fromarray(np.concatenate(arrays, axis=0), mode="RGB")
        quantized = stack.quantize(colors=size, method=_pil_quantize_method(method))
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    raw = quantized.getpalette() or []

    entries: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index in range(0, min(len(raw), size * 3), 3):
        triple = (int(raw[index]), int(raw[index + 1]), int(raw[index + 2]))
        if triple in seen:
            continue
        seen.add(triple)
        entries.append(triple)
    n_from_quantizer = len(entries)
    entries.extend(_distinct_filler(seen, size - n_from_quantizer))

    if len(entries) != size or len(set(entries)) != size:  # pragma: no cover
        raise PaletteError(
            f"the master palette came out with {len(entries)} entries of which "
            f"{len(set(entries))} are distinct; {size} pairwise-distinct entries "
            "are required. See FILLER_RULE for why."
        )

    return MasterPalette(
        entries=tuple(entries),
        size=size,
        method=str(method).upper(),
        dither=str(dither).upper(),
        n_from_quantizer=n_from_quantizer,
        n_distinct_in_sequence=int(distinct.size),
        filler_rule=FILLER_RULE,
        built_over=tuple(str(Path(p).resolve()) for p in frame_paths),
        frames_sha256=tuple(sha256_of(p) for p in frame_paths),
        frame_size_px=(int(width), int(height)),
        quantized_over_px=int(total_px),
    )


def map_frame_through(
    frame: Any, master: MasterPalette, *, dither: Optional[str] = None
):
    """One frame as a ``P`` image whose palette is byte-identical to *master*.

    *frame* may be a path, a PIL image or an ``(h, w, 3)`` uint8 array.

    TWO STEPS, AND THE SECOND ONE IS THE POINT
    ------------------------------------------
    1. ``quantize`` against the master palette **truncated to the quantizer's own
       entries**, so no pixel can be mapped onto a filler slot.
    2. ``putpalette`` the FULL master table onto the result.  Step 1 leaves the
       frame carrying a short palette that PIL pads with zeros; without step 2 the
       frames would reach the writer with palettes that are equal in their used
       entries and different in their bytes, and PIL's ``_normalize_palette`` would
       have to remap -- which is the machinery this whole section exists to keep
       out of the write path.  After step 2 every frame's palette is the same 768
       bytes and the remap is the identity.
    """
    from PIL import Image

    if isinstance(frame, (str, Path)):
        with Image.open(frame) as opened:
            source = opened.convert("RGB")
    elif isinstance(frame, np.ndarray):
        source = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8), "RGB")
    else:
        source = frame.convert("RGB")

    mapped = source.quantize(
        palette=master.palette_image(),
        dither=_pil_dither(master.dither if dither is None else dither),
    )
    mapped.putpalette(master.as_bytes(), "RGB")
    return mapped


# ---------------------------------------------------------------------------
# THE ANIMATION FILES
# ---------------------------------------------------------------------------


def _check_duration_ms(duration_ms: int) -> int:
    duration_ms = int(duration_ms)
    if duration_ms <= 0:
        raise GifDurationError(
            f"duration_ms={duration_ms!r}. A zero or negative frame delay plays the "
            "sequence as fast as the decoder can go, which is the exact failure a "
            "seconds-shaped duration produces silently -- on imageio 2.37.2, and on "
            "PIL too: measured through this module's own writer, duration=0.25 comes "
            "back [None, None, None]."
        )
    if duration_ms % 10:
        raise GifDurationError(
            f"duration_ms={duration_ms} is not a multiple of 10. GIF stores the "
            "frame delay in centiseconds, so this value cannot survive the round "
            "trip and the read-back assertion would be reporting the file format "
            "rather than this code."
        )
    return duration_ms


def write_gif(
    frame_paths: Sequence[Path | str],
    out_path: Path | str,
    *,
    duration_ms: int = FRAME_DURATION_MS,
    loop: int = 0,
    master: Optional[MasterPalette] = None,
) -> Path:
    """Write the GIF: ONE colour table, and the duration in MILLISECONDS.

    THE UNIT IS ONE BUG
    -------------------
    ``imageio``'s ``duration`` has meant seconds in some versions and milliseconds
    in others.  On the installed **2.37.2** it is milliseconds, and a seconds-shaped
    value is not rejected -- measured here, ``duration=0.25`` over three frames
    produces a valid GIF whose per-frame delays PIL reads back as ``[None, 0, 0]``.
    No exception, no warning, a file that opens.  Four of the five GIF builders in
    the sibling ``Paleo-Europe_3D/scripts`` tree pass a seconds-shaped value today.

    So this function writes the value and the caller is expected to READ IT BACK --
    :func:`assert_frame_durations_ms` does, and :func:`build_animation` calls it.
    Nothing here trusts the write.

    250 ms is also chosen for the format: GIF stores the delay in CENTISECONDS, so
    only multiples of 10 ms survive the round trip.  A value like 205 ms would come
    back 200 or 210 and the assertion would be reporting the format, not the code.

    THE COLOUR TABLE IS THE OTHER BUG, AND IT IS WHY THIS NO LONGER GOES THROUGH
    ``imageio``
    ---------------------------------------------------------------------------
    ``imageio.v3.imwrite(..., extension=".gif")`` hands PIL RGB frames and no
    palette.  PIL then does two things, both in ``PIL.GifImagePlugin``:
    ``_normalize_mode`` runs ``im.convert("P", palette=Image.Palette.ADAPTIVE)`` on
    **each frame separately**, and ``_write_multiple_frames`` sets
    ``encoderinfo["include_color_table"] = True`` for every delta frame *because*
    the ``palette`` argument is ``None``.  The 2026-08-03 deliverable is the proof:
    parsing its image descriptors found **16 of 17 frames carrying a local 256-entry
    colour table**, adjacent tables sharing as few as 103 of 256 entries.  Eight
    per-frame colour limits pinned in the renderer, undone by the encoder.

    This writer instead maps every frame through *master* (:func:`map_frame_through`)
    and passes the same table to PIL as ``palette=``.  That is the branch that keeps
    ``include_color_table`` unset, so the delta frames reference the global table.
    The result is still not trusted: :func:`assert_one_global_palette` parses the
    written blocks back out and :func:`assert_gif_frames_are_the_mapped_frames`
    decodes them.

    Returns the path; the palette used is *master*, or one built here over these
    frames if none was given.
    """
    duration_ms = _check_duration_ms(duration_ms)
    if master is None:
        master = build_master_palette(frame_paths)
    mapped = [map_frame_through(path, master) for path in frame_paths]
    if not mapped:
        raise PaletteError("no frames to write")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped[0].save(
        out_path,
        save_all=True,
        append_images=mapped[1:],
        duration=duration_ms,
        loop=int(loop),
        palette=master.as_bytes(),
        optimize=False,
    )
    # Returned UNRESOLVED, so the caller's path_record() can carry both spellings.
    # ``.resolve()`` here would make ``as_written`` and ``resolved`` the same UNC
    # string and the drive-letter spelling of the deliverable's own animation would
    # appear nowhere in the manifest -- see PATH_SPELLING_DECISION.
    return out_path


def write_webp(
    frame_paths: Sequence[Path | str],
    out_path: Path | str,
    *,
    duration_ms: int = FRAME_DURATION_MS,
    loop: int = 0,
    lossless: bool = True,
) -> Path:
    """The frames as animated WebP: **24-bit, lossless, and NOT palette-mapped.**

    WHAT CHANGED ON 2026-08-04, AND WHY IT HAD TO
    ---------------------------------------------
    Until 2026-08-04 this wrote the frames *after* :func:`map_frame_through`, so
    the WebP carried exactly the same 256 colours as the GIF.  In the same run,
    :func:`describe_palette_headroom` told the reader that when 256 entries are not
    enough the escape is "animated WebP written LOSSLESS (24-bit, one file, which
    this run already writes beside the GIF)".  The two statements sat in the same
    ``master_palette.json``: the shipped 0.5 ka deliverable's ``webp_audit`` read
    ``n_distinct_colours_decoded: 222``, i.e. the named remedy was a file with FEWER
    colours than the palette it was supposed to escape.  A sentence pointing at a
    remedy the artifact does not provide is worse than no sentence.

    So the WebP is now written from the ORIGINAL canvas PNGs.  It is the 24-bit
    escape the headroom note names, and :func:`assert_webp_is_a_lossless_24bit_copy`
    proves it by DECODING the written file and comparing every frame to its source
    bit for bit.

    THE PER-FRAME COLOUR MAPPING IS STILL RULED OUT -- MORE STRONGLY, NOT LESS
    -------------------------------------------------------------------------
    The defect a WebP can have is the one a lossy encoder introduces: it re-derives
    each frame's colours from that frame's own pixels, which is a per-frame colour
    transform by another name (the pre-2026-08-03 writer used ``quality=80``).
    LOSSLESS removes it completely -- the decoded frame equals the source array
    exactly, so no transform of any kind was applied -- and the bit-exact assertion
    is what proves that, where "every colour is in the master table" only ever
    proved the weaker thing.

    The GIF remains the deliverable and the pinned-palette artifact.  Which file is
    smaller depends entirely on the frames, so the sizes go in the manifest rather
    than in a claim here; both are measured per run.
    """
    from PIL import Image

    frames = []
    for path in frame_paths:
        with Image.open(path) as opened:
            frames.append(opened.convert("RGB"))
    if not frames:
        raise PaletteError("no frames to write")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=int(loop),
        lossless=bool(lossless),
        quality=100 if lossless else 80,
    )
    # Unresolved, for the reason in :func:`write_gif`.
    return out_path


def read_frame_durations_ms(path: Path | str) -> list[Optional[int]]:
    """Per-frame delay as PIL reads it back out of the file.

    ``None`` is preserved rather than mapped to 0: PIL reports ``None`` for a frame
    that carries no delay at all, and that is a different defect from a delay of
    zero.  Both are what a seconds-shaped ``duration`` produces.

    ``load()`` IS THE WHOLE FUNCTION -- WITHOUT IT THIS REPORTED A DEFECT IT HAD MADE
    ---------------------------------------------------------------------------------
    Until 2026-08-04 this seeked and read ``image.info["duration"]`` immediately.
    That works for a GIF, because ``GifImagePlugin`` fills ``info`` from the graphic
    control extension during the seek; it does NOT work for a WebP, because
    ``WebPImagePlugin`` only sets ``info["duration"]`` in ``load()``.  So every WebP
    this module wrote came back ``[None, None, ...]``, and the manifest shipped a
    ``webp_duration_note`` explaining that "PIL exposes no per-frame duration for the
    animated WebP this writes -- measured, ``Image.info['duration']`` reads ``None``
    on every frame".

    **That sentence was false, and the measurement behind it was this reader.**
    Measured on the shipped 0.5 ka deliverable's own
    ``khvalynian_spill.webp`` with a ``load()`` added: ``[250] * 9``.  The delays
    were in the file the whole time.  The WebP's delay is therefore asserted now, on
    the same terms as the GIF's, and the note that excused not asserting it is gone.
    """
    from PIL import Image

    durations: list[Optional[int]] = []
    with Image.open(path) as image:
        index = 0
        while True:
            try:
                image.seek(index)
            except EOFError:
                break
            image.load()
            value = image.info.get("duration")
            durations.append(None if value is None else int(value))
            index += 1
    return durations


def check_webp_durations(
    path: Path | str, expected_ms: int, n_frames: int
) -> tuple[list[Optional[int]], bool, str]:
    """Assert the WebP's delays -- except for the ONE case where the format cannot.

    Returns ``(durations, asserted, note)``.

    From 2026-08-04 the WebP's per-frame delay is asserted exactly like the GIF's:
    it always survived the write, and the ``[None, None, ...]`` this module used to
    record was :func:`read_frame_durations_ms` failing to ``load()`` the frame, not
    the file failing to carry a delay.

    There is one measured exception, and it is a property of the container rather
    than of this code.  Written with PIL 12.1.0 on this machine, ``duration=250``
    over N frames reads back:

        N = 1   webp ``[0]``          gif ``[250]``
        N = 2   webp ``[250, 250]``   gif ``[250, 250]``
        N = 3   webp ``[250]*3``      gif ``[250]*3``

    A lone frame has no inter-frame interval to get wrong, and libwebp's animation
    muxer writes zero for it.  So a one-frame sequence records the delay it read and
    says ``asserted=False`` with this reason; it does not quietly pass an assertion
    that did not run, and it does not fail a sequence over a delay that has nothing
    to separate.
    """
    durations = read_frame_durations_ms(path)
    if int(n_frames) < 2:
        return (
            durations,
            False,
            "NOT ASSERTED, and this is the only case in which it is not: a "
            "single-frame animated WebP written by PIL 12.1.0 reads its delay back "
            f"as {durations!r} whatever was written (measured on this machine: "
            "duration=250 over 1 frame reads [0], over 2 frames reads [250, 250]). "
            "A lone frame has no inter-frame interval, so there is nothing for the "
            "delay to separate. The GIF carries 250 even at one frame and IS "
            "asserted",
        )
    assert_frame_durations_ms(path, expected_ms, n_frames)
    return (
        durations,
        True,
        f"ASSERTED: read back with PIL after load() and compared against the "
        f"{int(expected_ms)} ms written, frame by frame. Until 2026-08-04 this was "
        "recorded as [null, ...] and excused with a note saying PIL exposes no WebP "
        "duration -- the reader was not calling load(), and the delays were in the "
        "file the whole time",
    )


def assert_frame_durations_ms(
    path: Path | str, expected_ms: int, n_frames: int
) -> list[Optional[int]]:
    """Read the delays back out of the written file and refuse a mismatch."""
    durations = read_frame_durations_ms(path)
    if len(durations) != int(n_frames):
        raise GifDurationError(
            f"{Path(path).name} holds {len(durations)} frames, not {int(n_frames)}. "
            "A frame that vanished in the write is an epoch missing from the "
            "sequence, which is the compression of time this module exists to "
            "prevent."
        )
    wrong = [(i, d) for i, d in enumerate(durations) if d != int(expected_ms)]
    if wrong:
        raise GifDurationError(
            f"{Path(path).name} was written with duration={int(expected_ms)} ms but "
            f"PIL reads back {durations!r}; frames {[i for i, _ in wrong]} disagree. "
            "On imageio 2.37.2 this is what a seconds-shaped duration does: "
            "duration=0.25 writes delays PIL reads as [None, 0, 0], silently, in a "
            "valid file. The write is never trusted; this is the check."
        )
    return durations


# ---------------------------------------------------------------------------
# READING THE WRITTEN FILE BACK  (the palette half of "never trust the write")
# ---------------------------------------------------------------------------


def parse_gif_structure(path: Path | str) -> dict:
    """Walk a GIF's blocks and report every colour table it actually carries.

    This is a BYTE parse of the file, not a query to PIL.  PIL will happily decode
    an animation whose frames each carry their own colour table and report nothing
    unusual, because that is a legal GIF -- the defect is not a decoding failure, it
    is a provenance failure, and it is only visible in the container.

    The GIF89a grammar walked here is the whole of it: a 13-byte header and logical
    screen descriptor, an optional global colour table of ``2^(N+1)`` RGB triples,
    then a stream of blocks -- ``0x21`` extension (graphic control, comment,
    application, plain text; all sub-block chains), ``0x2C`` image descriptor
    (10 bytes, optional local colour table, LZW minimum code size, sub-block chain),
    ``0x3B`` trailer.

    Returns
    -------
    dict
        ``version``, ``logical_screen_px``, ``global_color_table`` (bytes or
        ``None``), ``global_color_table_entries``, and ``images``: one record per
        image descriptor with ``offset_px``, ``size_px``, ``local_color_table``
        (bytes or ``None``), ``local_color_table_entries`` and
        ``local_equals_global``.
    """
    data = Path(path).read_bytes()
    if data[:3] != b"GIF":
        raise PaletteError(
            f"{Path(path).name} does not start with the GIF signature; its first "
            f"six bytes are {data[:6]!r}"
        )
    version = data[:6].decode("ascii", "replace")
    cursor = 6
    screen_w = int.from_bytes(data[cursor:cursor + 2], "little")
    screen_h = int.from_bytes(data[cursor + 2:cursor + 4], "little")
    packed = data[cursor + 4]
    cursor += 7  # 4 size bytes + packed + background index + aspect ratio
    global_table: Optional[bytes] = None
    global_entries = 0
    if packed & 0x80:
        global_entries = 2 ** ((packed & 0x07) + 1)
        global_table = bytes(data[cursor:cursor + 3 * global_entries])
        cursor += 3 * global_entries

    def _skip_sub_blocks(position: int) -> int:
        while True:
            size = data[position]
            position += 1
            if size == 0:
                return position
            position += size

    images: list[dict] = []
    while cursor < len(data):
        block = data[cursor]
        if block == 0x3B:  # trailer
            break
        if block == 0x21:  # extension introducer
            cursor += 2  # introducer + label
            cursor = _skip_sub_blocks(cursor)
            continue
        if block == 0x2C:  # image descriptor
            descriptor_offset = cursor
            cursor += 1
            left = int.from_bytes(data[cursor:cursor + 2], "little")
            top = int.from_bytes(data[cursor + 2:cursor + 4], "little")
            width = int.from_bytes(data[cursor + 4:cursor + 6], "little")
            height = int.from_bytes(data[cursor + 6:cursor + 8], "little")
            image_packed = data[cursor + 8]
            cursor += 9
            local_table: Optional[bytes] = None
            local_entries = 0
            if image_packed & 0x80:
                local_entries = 2 ** ((image_packed & 0x07) + 1)
                local_table = bytes(data[cursor:cursor + 3 * local_entries])
                cursor += 3 * local_entries
            cursor += 1  # LZW minimum code size
            cursor = _skip_sub_blocks(cursor)
            images.append(
                {
                    "index": len(images),
                    # Byte offsets into the file, so a reader (or a test) can point
                    # at the descriptor that carries a table rather than describing
                    # where it is.
                    "descriptor_byte_offset": descriptor_offset,
                    "packed_byte_offset": descriptor_offset + 9,
                    "offset_px": [left, top],
                    "size_px": [width, height],
                    "local_color_table": local_table,
                    "local_color_table_entries": local_entries or None,
                    "local_equals_global": (
                        None if local_table is None else local_table == global_table
                    ),
                }
            )
            continue
        raise PaletteError(
            f"{Path(path).name}: unknown block 0x{block:02x} at byte {cursor}; the "
            "file is not a GIF this parser understands and its colour tables cannot "
            "be certified"
        )
    return {
        "version": version,
        "logical_screen_px": [screen_w, screen_h],
        "global_color_table": global_table,
        "global_color_table_entries": global_entries or None,
        "images": images,
    }


def assert_one_global_palette(
    path: Path | str, master: Optional[MasterPalette] = None
) -> dict:
    """Refuse a GIF that carries more than one colour mapping.

    Three things are checked, in order, against the file's own bytes:

    1. a global colour table exists at all;
    2. if *master* is given, that table is byte-identical to it -- the animation
       must carry the palette the frames were mapped through, not some other one
       the encoder derived;
    3. no image descriptor carries a local colour table.  A local table that is
       byte-identical to the global one is tolerated and REPORTED, because it is
       redundant rather than wrong; a local table that differs is the defect.

    Returns a report for the manifest.  A caller that wants the audit in the data
    without the exception can call :func:`parse_gif_structure` directly.
    """
    structure = parse_gif_structure(path)
    name = Path(path).name
    global_table = structure["global_color_table"]
    if global_table is None:
        raise PaletteError(
            f"{name} carries NO global colour table, so every frame must be "
            "carrying its own. One shared mapping is the whole point."
        )
    if master is not None and global_table != master.as_bytes():
        raise PaletteError(
            f"{name}'s global colour table is not the palette its frames were "
            f"mapped through: {len(global_table)} bytes with sha256 "
            f"{hashlib.sha256(global_table).hexdigest()} in the file against "
            f"{len(master.as_bytes())} bytes with sha256 "
            f"{hashlib.sha256(master.as_bytes()).hexdigest()} in the master. The "
            "encoder re-derived the table."
        )
    differing = [
        image for image in structure["images"]
        if image["local_color_table"] is not None
        and not image["local_equals_global"]
    ]
    redundant = [
        image for image in structure["images"]
        if image["local_color_table"] is not None and image["local_equals_global"]
    ]
    if differing:
        shared_counts = []
        global_set = {
            global_table[i:i + 3] for i in range(0, len(global_table), 3)
        }
        for image in differing[:5]:
            table = image["local_color_table"]
            local_set = {table[i:i + 3] for i in range(0, len(table), 3)}
            shared_counts.append(
                f"frame {image['index']}: {len(global_set & local_set)} of "
                f"{len(local_set)} entries shared with the global table"
            )
        raise PaletteError(
            f"{name} carries {len(differing)} local colour tables that differ from "
            f"the global one, in {len(structure['images'])} image descriptors "
            f"({[i['index'] for i in differing]}). That is a per-frame colour "
            "mapping laid on top of the pinned one: the same RGB triple reaches a "
            "different index, and the same index a different triple, from one epoch "
            "to the next. " + "; ".join(shared_counts)
        )
    return {
        "file": str(Path(path).resolve()),
        "version": structure["version"],
        "logical_screen_px": structure["logical_screen_px"],
        "n_image_descriptors": len(structure["images"]),
        "global_color_table_entries": structure["global_color_table_entries"],
        "global_color_table_sha256": hashlib.sha256(global_table).hexdigest(),
        "global_color_table_matches_master": (
            None if master is None else global_table == master.as_bytes()
        ),
        "n_frames_with_a_local_color_table": len(differing) + len(redundant),
        "n_frames_with_a_DIFFERING_local_color_table": len(differing),
        "n_frames_with_a_redundant_local_color_table": len(redundant),
        "verdict": "ONE_GLOBAL_PALETTE",
        "checked_by": (
            "parse_gif_structure -- a byte walk of the GIF89a block grammar over the "
            "written file. Not PIL, and not the writer's return value: a per-frame "
            "colour table is a legal GIF, so no decoder reports it as an error and "
            "it is only visible in the container"
        ),
    }


def assert_gif_frames_are_the_mapped_frames(
    path: Path | str,
    master: MasterPalette,
    frame_paths: Sequence[Path | str],
) -> dict:
    """Decode the animation and refuse a frame that is not the frame that went in.

    The palette audit proves ONE table reached the file.  This proves the PIXELS
    reached it too: every frame is decoded, composited (GIF delta frames are
    sub-rectangles over their predecessor) and compared, exactly, against the same
    frame mapped through *master*.  Together they close the loop -- the mapping is
    the pinned one, and it is the mapping the file actually applies.

    Zero tolerance, deliberately.  Both sides are index images through the same
    256-entry table, so "close" has no meaning here: any difference at all is a
    frame the encoder changed.
    """
    from PIL import Image

    mismatches: list[str] = []
    n_decoded = 0
    with Image.open(path) as animation:
        for index, frame_path in enumerate(frame_paths):
            try:
                animation.seek(index)
            except EOFError:
                mismatches.append(
                    f"frame {index} ({Path(frame_path).name}) is not in the file at "
                    "all"
                )
                break
            n_decoded += 1
            decoded = np.asarray(animation.convert("RGB"), dtype=np.uint8)
            expected = np.asarray(
                map_frame_through(frame_path, master).convert("RGB"), dtype=np.uint8
            )
            if decoded.shape != expected.shape:
                mismatches.append(
                    f"frame {index} decodes {decoded.shape} against an expected "
                    f"{expected.shape}"
                )
                continue
            differing = int(np.count_nonzero(np.any(decoded != expected, axis=-1)))
            if differing:
                worst = int(
                    np.abs(decoded.astype(np.int16) - expected.astype(np.int16)).max()
                )
                mismatches.append(
                    f"frame {index} ({Path(frame_path).name}) differs in "
                    f"{differing:,} of {decoded.shape[0] * decoded.shape[1]:,} "
                    f"pixels, worst channel delta {worst}"
                )
    if mismatches:
        raise PaletteError(
            f"{Path(path).name} does not decode to the frames that were mapped into "
            "it, so the encoder changed the picture after the palette was pinned: "
            + "; ".join(mismatches[:5])
        )
    return {
        "n_frames_decoded_and_compared": n_decoded,
        "pixels_differing": 0,
        "tolerance": (
            "exact. Both sides are index images through the same 256-entry table, "
            "so any difference is a frame the encoder rewrote"
        ),
    }


def assert_webp_is_a_lossless_24bit_copy(
    path: Path | str,
    frame_paths: Sequence[Path | str],
    *,
    master: Optional[MasterPalette] = None,
) -> dict:
    """Decode the WebP and refuse it unless every frame equals its source exactly.

    This is the assertion that makes the "24-bit escape from the 256-colour
    ceiling" sentence in :func:`describe_palette_headroom` a fact about the file
    rather than a hope about the format.  Three things come out of the parse and
    all three are reported as values:

    1. **bit-exactness** -- every decoded frame compared, element by element,
       against the source canvas PNG.  Any difference at all refuses: a lossless
       encode that changed a pixel is not lossless, and a lossy one changes many.
       This is strictly stronger than the "no colour outside the master palette"
       check it replaces, which could be passed by a file that had been re-mapped
       as long as it re-mapped into the table.
    2. **how many colours actually reached the file** -- the distinct RGB triples
       over all frames.  When *master* is given, this is compared against the
       palette size, and that comparison is the whole point: on the 2026-08-03/04
       deliverable the number was **222 against a 256-entry table**, i.e. the file
       offered as the escape held fewer colours than the ceiling it escaped.
    3. **whether the escape is real for THIS sequence** -- ``escapes_the_gif_ceiling``
       is ``True`` only when more distinct colours decode than the GIF table can
       hold.  It can legitimately be ``False``: a sequence of flat hatched refusal
       frames has few colours and nothing to escape.  ``False`` is then a fact about
       the content, and it is written down rather than assumed away.
    """
    from PIL import Image

    mismatches: list[str] = []
    n_frames = 0
    distinct: set[int] = set()
    per_frame: list[dict] = []
    with Image.open(path) as animation:
        for index, frame_path in enumerate(frame_paths):
            try:
                animation.seek(index)
            except EOFError:
                mismatches.append(
                    f"frame {index} ({Path(frame_path).name}) is not in the file at all"
                )
                break
            n_frames += 1
            decoded = np.asarray(animation.convert("RGB"), dtype=np.uint8)
            with Image.open(frame_path) as opened:
                source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            packed = np.unique(_packed_colours(decoded))
            distinct |= set(int(v) for v in packed.tolist())
            if decoded.shape != source.shape:
                mismatches.append(
                    f"frame {index} decodes {decoded.shape} against a source "
                    f"{source.shape}"
                )
                continue
            differing = int(np.count_nonzero(np.any(decoded != source, axis=-1)))
            if differing:
                worst = int(
                    np.abs(decoded.astype(np.int16) - source.astype(np.int16)).max()
                )
                mismatches.append(
                    f"frame {index} ({Path(frame_path).name}) differs from its source "
                    f"in {differing:,} of {decoded.shape[0] * decoded.shape[1]:,} "
                    f"pixels, worst channel delta {worst}"
                )
            per_frame.append(
                {
                    "frame": str(Path(frame_path).name),
                    "distinct_colours_decoded": int(packed.size),
                    "pixels_differing_from_the_source": differing,
                }
            )
    if mismatches:
        raise PaletteError(
            f"{Path(path).name} does not decode to the frames that were written into "
            "it, so it is not a lossless 24-bit copy and must not be offered as the "
            "escape from the GIF's 256-colour ceiling: "
            + "; ".join(mismatches[:5])
            + ". A lossy WebP re-derives every frame's colours from that frame's own "
            "pixels, which is a per-frame colour mapping by another name -- write it "
            "lossless, from the unquantized canvas frames."
        )
    ceiling = GIF_PALETTE_SIZE if master is None else int(master.size)
    return {
        "file": path_record(path),
        "n_frames_decoded": n_frames,
        "n_distinct_colours_decoded": len(distinct),
        "gif_colour_table_ceiling": ceiling,
        "escapes_the_gif_ceiling": len(distinct) > ceiling,
        "pixels_differing_from_the_source": 0,
        "per_frame": per_frame,
        "verdict": "BIT_EXACT_24_BIT",
        "checked_by": (
            "every frame decoded out of the written file and compared element by "
            "element against the source canvas PNG. Exact equality, deliberately: a "
            "lossless encode that moved a pixel is not lossless. This replaces the "
            "'no colour outside the master palette' check of 2026-08-03, which the "
            "old palette-mapped WebP passed while carrying 222 distinct colours -- "
            "fewer than the 256-entry table the same file's headroom note said it "
            "escaped"
        ),
    }


# ---------------------------------------------------------------------------
# WHAT ONE SHARED PALETTE COSTS
# ---------------------------------------------------------------------------


def measure_palette_cost(
    frame_paths: Sequence[Path | str],
    master: MasterPalette,
    *,
    compare_with_per_frame_adaptive: bool = True,
    webp_audit: Optional[dict] = None,
) -> dict:
    """What the ONE table costs against a table per frame, in per-pixel error.

    Error is the RGB Euclidean distance between the source PNG and the frame as it
    will be drawn -- ``sqrt(dR^2 + dG^2 + dB^2)``, 0..441.67 -- reported as the mean
    over pixels and the maximum over pixels, per frame and over the sequence.

    The comparison arm is the DEFECT: ``im.convert("P", palette=ADAPTIVE)`` per
    frame, which is literally what ``PIL.GifImagePlugin._normalize_mode`` did to
    every frame of the 2026-08-03 deliverable.  So this is not a hypothetical
    baseline -- it is the file that shipped.

    A lower per-pixel error is NOT the deciding quantity and the report says so:
    the adaptive arm wins on mean error precisely because it is free to give each
    frame a different mapping, which is the property being removed.
    """
    from PIL import Image

    per_frame: list[dict] = []
    for path in frame_paths:
        with Image.open(path) as opened:
            source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        shared = np.asarray(
            map_frame_through(path, master).convert("RGB"), dtype=np.uint8
        )
        record = {
            "frame": str(Path(path).name),
            "distinct_colours_in_the_source": int(
                np.unique(_packed_colours(source)).size
            ),
            "shared_palette": _error_stats(source, shared),
        }
        if compare_with_per_frame_adaptive:
            with Image.open(path) as opened:
                adaptive = np.asarray(
                    opened.convert("RGB")
                    .convert("P", palette=Image.Palette.ADAPTIVE, colors=master.size)
                    .convert("RGB"),
                    dtype=np.uint8,
                )
            record["per_frame_adaptive"] = _error_stats(source, adaptive)
        per_frame.append(record)

    def _roll(arm: str, key: str, how) -> Optional[float]:
        values = [f[arm][key] for f in per_frame if arm in f]
        return None if not values else float(how(values))

    report: dict[str, Any] = {
        "metric": (
            "per-pixel RGB Euclidean distance between the source canvas PNG and the "
            "frame as drawn, sqrt(dR^2+dG^2+dB^2), range 0..441.67"
        ),
        "dither": master.dither,
        "n_frames": len(per_frame),
        "distinct_colours_in_the_sequence": master.n_distinct_in_sequence,
        "palette_entries_used": master.n_from_quantizer,
        "shared_palette": {
            "mean_of_frame_means": _roll("shared_palette", "mean", np.mean),
            "worst_frame_mean": _roll("shared_palette", "mean", np.max),
            "max_per_pixel": _roll("shared_palette", "max", np.max),
            "max_per_channel": _roll("shared_palette", "max_channel", np.max),
            "fraction_of_pixels_changed": _roll(
                "shared_palette", "fraction_changed", np.mean
            ),
        },
        "per_frame": per_frame,
    }
    if compare_with_per_frame_adaptive:
        report["per_frame_adaptive"] = {
            "what_it_is": (
                "im.convert('P', palette=ADAPTIVE) per frame -- exactly what "
                "PIL.GifImagePlugin._normalize_mode did to every frame of the "
                "2026-08-03 deliverable. The comparison arm is the defect, not a "
                "hypothetical"
            ),
            "mean_of_frame_means": _roll("per_frame_adaptive", "mean", np.mean),
            "worst_frame_mean": _roll("per_frame_adaptive", "mean", np.max),
            "max_per_pixel": _roll("per_frame_adaptive", "max", np.max),
            "max_per_channel": _roll("per_frame_adaptive", "max_channel", np.max),
            "fraction_of_pixels_changed": _roll(
                "per_frame_adaptive", "fraction_changed", np.mean
            ),
        }
        shared_mean = report["shared_palette"]["mean_of_frame_means"]
        adaptive_mean = report["per_frame_adaptive"]["mean_of_frame_means"]
        shared_max = report["shared_palette"]["max_per_pixel"]
        adaptive_max = report["per_frame_adaptive"]["max_per_pixel"]
        report["cost_of_sharing"] = {
            "extra_mean_per_pixel_error": (
                None if None in (shared_mean, adaptive_mean)
                else float(shared_mean - adaptive_mean)
            ),
            "extra_max_per_pixel_error": (
                None if None in (shared_max, adaptive_max)
                else float(shared_max - adaptive_max)
            ),
            "how_to_read_it": (
                "a POSITIVE mean is what one shared table costs; a NEGATIVE max is "
                "the shared table being better at its worst pixel, which it can be "
                "because it has no frame it is allowed to neglect. Lower error is "
                "not the deciding quantity: the adaptive arm wins on the mean "
                "precisely by giving each frame its own mapping, which is the "
                "property being removed"
            ),
        }
    report["headroom"] = describe_palette_headroom(master, webp_audit=webp_audit)
    return report


def _error_stats(source: np.ndarray, drawn: np.ndarray) -> dict:
    delta = source.astype(np.int16) - drawn.astype(np.int16)
    euclid = np.sqrt((delta.astype(np.float64) ** 2).sum(axis=-1))
    return {
        "mean": float(euclid.mean()),
        "max": float(euclid.max()),
        "max_channel": int(np.abs(delta).max()),
        "fraction_changed": float((euclid > 0).mean()),
    }


def compare_quantizers(
    frame_paths: Sequence[Path | str],
    *,
    methods: Sequence[str] = ("MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE"),
) -> dict:
    """Re-measure every quantizer arm on the frames of the sequence it reports in.

    :data:`PALETTE_METHOD` records the trade this settles, measured on the nine real
    canvas frames.  This function re-runs the same measurement on whatever frames it
    is given, so the decision recorded beside an animation is a measurement of THAT
    animation's own frames and never a number carried over from another set.

    Each method builds a WHOLE master palette over the WHOLE set by the same path the
    deliverable uses, so nothing here is a different recipe: same
    :func:`build_master_palette`, same size, same ``dither="NONE"``, same
    :func:`measure_palette_cost` metric.  The per-frame-adaptive arm is switched off
    because it does not depend on the method and would triple the cost to produce the
    same number three times.

    It answers ONE question -- which quantizer, at 256 entries, on this content -- and
    it does not answer whether 256 entries are enough.  That is
    :func:`describe_palette_headroom`, and the answer there is a property of the GIF
    format rather than of the choice made here.
    """
    arms: dict[str, Any] = {}
    for method in methods:
        try:
            master = build_master_palette(frame_paths, method=method)
        except Exception as exc:  # noqa: BLE001 - an unavailable method is data
            arms[method] = {"status": f"UNAVAILABLE: {type(exc).__name__}: {exc}"}
            continue
        cost = measure_palette_cost(
            frame_paths, master, compare_with_per_frame_adaptive=False
        )
        arms[method] = {
            "status": "MEASURED",
            "entries_from_the_quantizer": master.n_from_quantizer,
            "filler_entries": master.n_filler,
            "mean_of_frame_means": cost["shared_palette"]["mean_of_frame_means"],
            "worst_frame_mean": cost["shared_palette"]["worst_frame_mean"],
            "max_per_pixel": cost["shared_palette"]["max_per_pixel"],
            "fraction_of_pixels_changed": cost["shared_palette"][
                "fraction_of_pixels_changed"
            ],
        }
    measured = {k: v for k, v in arms.items() if v.get("status") == "MEASURED"}
    best_mean = min(measured, key=lambda k: measured[k]["mean_of_frame_means"], default=None)
    best_max = min(measured, key=lambda k: measured[k]["max_per_pixel"], default=None)
    report = {
        "question": (
            "which PIL quantizer, at 256 entries, on THESE frames -- re-measured on "
            "every run so the decision beside an animation is a measurement of that "
            "animation's own frames"
        ),
        "n_frames": len(list(frame_paths)),
        "size": GIF_PALETTE_SIZE,
        "dither": PALETTE_DITHER,
        "shipped": PALETTE_METHOD,
        "arms": arms,
        "lowest_mean_error": best_mean,
        "lowest_worst_pixel": best_max,
        "how_to_read_it": (
            "mean and worst pixel disagree here, as they do on every set this "
            "project has measured. Which one matters depends on the picture -- on a "
            "hypsometric ramp a viewer reads an elevation off a hue, so a large "
            "excursion on a few pixels is a wrong elevation while a small one "
            "everywhere is invisible. 'decision' below states which quantity was "
            "given the casting vote, and re-derives the choice from the arms above "
            "rather than restating PALETTE_METHOD"
        ),
    }
    report["decision"] = palette_method_decision(report)
    return report


#: The quantity that decides the quantizer, and why it and not the other one.
PALETTE_DECISION_RULE = (
    "the WORST per-pixel error decides, not the mean. On a hypsometric map the "
    "colour ramp IS the data -- a reader takes an elevation off a hue -- so one "
    "pixel drawn far from its true colour is a wrong elevation, while a small error "
    "spread over every pixel costs contrast rather than correctness. The mean is "
    "reported beside it as the price, in what_it_costs, both as the Euclidean figure "
    "and as its per-channel equivalent -- the latter is the one to compare against a "
    "just-noticeable difference, which is of order 2-3 RGB units per channel between "
    "two flat patches. The rule is defensible while the per-channel cost stays at "
    "that scale; a run whose mean climbs well past it has to re-argue the rule "
    "rather than re-apply it"
)


def palette_method_decision(
    quantizers: dict, *, shipped: str = PALETTE_METHOD
) -> dict:
    """Which arm the rule picks on THESE frames, and whether that is what shipped.

    Written into ``master_palette.json`` beside the arms it is derived from, so the
    decision can be audited against the numbers in the same object instead of
    against a sentence in a source file the reader does not have.

    It never silently agrees with :data:`PALETTE_METHOD`: when the measurement on
    this run's own frames selects a different arm, the verdict says
    ``DISAGREES_WITH_THE_SHIPPED_METHOD`` and carries both, because a constant that
    the data no longer supports is exactly the kind of thing that survives by being
    unexamined.
    """
    arms = {
        name: arm
        for name, arm in (quantizers.get("arms") or {}).items()
        if arm.get("status") == "MEASURED"
    }
    if not arms:
        return {
            "rule": PALETTE_DECISION_RULE,
            "shipped": shipped,
            "selected_by_the_rule": None,
            "verdict": "NOT_MEASURED: no quantizer arm completed on these frames",
        }
    selected = min(arms, key=lambda name: arms[name]["max_per_pixel"])
    lowest_mean = min(arms, key=lambda name: arms[name]["mean_of_frame_means"])
    chosen, other = arms[selected], arms[lowest_mean]
    worst_ratio = (
        None
        if not chosen["max_per_pixel"]
        else float(other["max_per_pixel"] / chosen["max_per_pixel"])
    )
    return {
        "rule": PALETTE_DECISION_RULE,
        "metric": (
            "per-pixel RGB Euclidean distance to the source canvas PNG, "
            "sqrt(dR^2+dG^2+dB^2), range 0..441.67"
        ),
        "shipped": shipped,
        "selected_by_the_rule": selected,
        "verdict": (
            "AGREES_WITH_THE_SHIPPED_METHOD"
            if selected == shipped
            else "DISAGREES_WITH_THE_SHIPPED_METHOD"
        ),
        "what_it_buys": {
            "arm": selected,
            "max_per_pixel": chosen["max_per_pixel"],
            "against_the_lowest_mean_arm": lowest_mean,
            "its_max_per_pixel": other["max_per_pixel"],
            "worst_pixel_tighter_by_factor": worst_ratio,
            "worst_pixel_as_fraction_of_the_metric_maximum": {
                selected: float(chosen["max_per_pixel"] / 441.6729559300637),
                lowest_mean: float(other["max_per_pixel"] / 441.6729559300637),
            },
        },
        "what_it_costs": {
            "extra_mean_per_pixel_error": float(
                chosen["mean_of_frame_means"] - other["mean_of_frame_means"]
            ),
            "mean_per_pixel_error": chosen["mean_of_frame_means"],
            "mean_per_channel_equivalent": float(
                chosen["mean_of_frame_means"] / (3 ** 0.5)
            ),
            "note": (
                "the per-channel equivalent divides the Euclidean mean by sqrt(3), "
                "i.e. what it would be if the error were spread evenly over R, G "
                "and B. It is the number to compare against a just-noticeable "
                "difference, which the Euclidean figure is not"
            ),
        },
        "arms_it_was_taken_from": {
            name: {
                "mean_of_frame_means": arm["mean_of_frame_means"],
                "max_per_pixel": arm["max_per_pixel"],
            }
            for name, arm in arms.items()
        },
        "measured_over": {
            "n_frames": quantizers.get("n_frames"),
            "palette_entries": quantizers.get("size"),
            "dither": quantizers.get("dither"),
        },
        "when_this_disagrees": (
            "if 'verdict' reads DISAGREES_WITH_THE_SHIPPED_METHOD then the frames "
            "of THIS run no longer support PALETTE_METHOD, and the animation beside "
            "this file was written with the shipped arm anyway -- the decision is "
            "not switched mid-run, because every other number in this manifest was "
            "measured through the shipped palette. Change the constant and re-run"
        ),
    }


def describe_palette_headroom(
    master: MasterPalette, *, webp_audit: Optional[dict] = None
) -> dict:
    """Is 256 entries enough for this content -- and if not, what instead.

    The ratio of distinct colours in the sequence to entries in the table is the
    compression the GIF format imposes, and it is not a knob: **there is no GIF with
    more than 256 colours per table.**  So when this is tight the answer is a
    different container, and the alternatives are named here rather than left for a
    reader to work out.

    THE ESCAPE IS NAMED ONLY WHEN IT WAS MEASURED
    ---------------------------------------------
    This note used to end "animated WebP written LOSSLESS (24-bit, one file, which
    this run already writes beside the GIF)" **unconditionally**, and the file it
    named was the palette-mapped one: 222 distinct colours against a 256-entry
    table, in the same ``master_palette.json``.  So *webp_audit* is now taken as an
    argument.  With it, the sentence states what parsing that file found -- how many
    colours decoded, and whether that is more than the GIF ceiling.  Without it, the
    alternatives are described as alternatives and this run is not credited with
    writing any of them.
    """
    used = max(int(master.n_from_quantizer), 1)
    distinct = int(master.n_distinct_in_sequence)
    ratio = distinct / used
    if webp_audit is None:
        escape = (
            "the alternatives are containers, not settings: a PNG SEQUENCE "
            "(lossless 24-bit, one file per epoch, no animation), animated WebP "
            "written LOSSLESS from the UNQUANTIZED frames (24-bit, one file), or "
            "MP4/H.264 (24-bit but lossy in a different way -- chroma subsampling, "
            "which smears the sharp colour boundaries of a hypsometric ramp). NO "
            "such file is claimed for this report: it was produced without a WebP "
            "audit, so nothing here says one was written"
        )
    else:
        decoded = int(webp_audit.get("n_distinct_colours_decoded") or 0)
        escapes = bool(webp_audit.get("escapes_the_gif_ceiling"))
        escape = (
            "the alternatives are containers, not settings: a PNG SEQUENCE "
            "(lossless 24-bit, one file per epoch, no animation), animated WebP "
            "written LOSSLESS from the UNQUANTIZED frames (24-bit, one file), or "
            "MP4/H.264 (24-bit but lossy in a different way -- chroma subsampling, "
            "which smears the sharp colour boundaries of a hypsometric ramp). This "
            f"run wrote the second one beside the GIF, and PARSING it back found "
            f"{decoded:,} distinct colours over "
            f"{webp_audit.get('n_frames_decoded')} frame(s) against the format's "
            f"{webp_audit.get('gif_colour_table_ceiling')}-entry GIF ceiling, every "
            "frame bit-identical to its source PNG ("
            f"{webp_audit.get('verdict')}). So the escape "
            + (
                "IS available in this deliverable"
                if escapes
                else "is available in principle but is NOT exercised by THESE frames "
                "-- they hold no more colours than a GIF table can carry, so the "
                "WebP is 24-bit and has nothing to escape"
            )
            + ". The GIF stays the deliverable only while the measured error above "
            "is acceptable"
        )
    return {
        "distinct_colours_in_the_sequence": distinct,
        "palette_entries_available": int(master.size),
        "palette_entries_the_quantizer_filled": int(master.n_from_quantizer),
        "colours_per_entry": float(ratio),
        "gif_ceiling": (
            "256. GIF stores the colour table size as a 3-bit exponent, so 2^8 is "
            "the format's maximum and 'use more colours' is not available here"
        ),
        "webp_escape_measured": webp_audit is not None,
        "if_256_is_not_enough": escape,
    }


# ---------------------------------------------------------------------------
# THE RUN
# ---------------------------------------------------------------------------


def _log(message: str, log_path: Optional[Path] = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing sidecar is data, not a failure
        return None


# ---------------------------------------------------------------------------
# THE ONE PINNED LIMIT THAT IS A THEOREM RATHER THAN A MEASUREMENT
# ---------------------------------------------------------------------------

#: How far above ``SequenceLimits.shade_max`` a cell may sit and still be called
#: arithmetic.  ``shade_max`` is +1.0 from ``analytic_shade_range``: the largest
#: value ``n . d`` can take for a unit surface normal and a unit light direction.
#: Nothing can genuinely exceed it, so an excursion is floating point -- but only up
#: to the precision of the floating point.  The DEM arrives float32 and the
#: normalisation inside ``LightSource.shade_normals`` is where the rounding happens,
#: so the budget is one float32 epsilon, 1.1920929e-07.  Measured on
#: ``khv_16.5ka.tif`` at ``max_px=7800`` the actual excursion is 2.96e-08, i.e. 0.25
#: of this budget; the budget is not fitted to that number, it is the arithmetic's
#: own precision, and :func:`measure_analytic_shade_excursion` re-measures per epoch.
SHADE_EXCURSION_BUDGET = float(np.finfo(np.float32).eps)

#: The policy this module renders under, and it is a decision with a stated cost.
#: See :class:`ClipPolicyBreachError`.  ``"raise"`` cannot be used because it aborts
#: a frame on the analytic ceiling above, against limits computed from that same
#: frame -- a refusal no recomputation can satisfy.  ``"mark"`` plus
#: :func:`assert_only_the_analytic_shade_ceiling_was_marked` is strictly stronger on
#: every quantity that is actually measured.
CLIP_POLICY = "mark"

#: Out-of-range quantities whose limit was MEASURED over the sequence's rasters.  A
#: non-zero count on any of these is real information about the frame and stops the
#: run.  ``shade_below_shade_min`` is in this list although its bound is also
#: analytic: ``-cos(45 deg)`` = -0.707107 is an infimum that needs a vertical face,
#: which ``numpy.gradient`` cannot produce, and the native 16.5 ka frame's smallest
#: raw intensity is -0.577424 -- 0.129683 clear of it, eleven million times the
#: float32 epsilon.  Rounding cannot reach that floor, so a cell below it would be a
#: real finding.
MEASURED_LIMIT_QUANTITIES = (
    "land_elevation_above_dem_data_max_m",
    "land_elevation_below_dem_data_min_m",
    "depth_above_depth_data_max_m",
    "ice_thickness_above_ice_max_m",
    "shade_below_shade_min",
)

#: The one quantity a frame may be marked for, and only inside
#: :data:`SHADE_EXCURSION_BUDGET`.
ARITHMETIC_LIMIT_QUANTITIES = ("shade_above_shade_max",)


def measure_analytic_shade_excursion(
    raster_path: Path | str,
    *,
    config: RenderConfig,
) -> dict:
    """How far this epoch's hillshade actually goes past its analytic ceiling.

    Reproduces the three renderer steps that produce ``raw_shade`` and nothing else:
    ``build_epoch_mosaic`` at the frame's own ``max_px``, the sigma=1.5
    ``gaussian_filter`` that makes ``dem_sm``, and ``_raw_hillshade_intensity`` with
    the frame's own ``vert_exag`` and the module's azimuth/altitude.  Same functions,
    same constants, same raster -- so a number here is comparable with the count the
    frame's own ``.palette.json`` reports, and a disagreement between them would be a
    real disagreement rather than two different recipes.

    THE ONE PLACE THE REPRODUCTION CAN DIVERGE, AND WHAT HAPPENS THEN
    ----------------------------------------------------------------
    ``render_paleocostline_epoch`` replaces ``dem_fs`` with an ice-dome blend
    wherever ``ice > ice_threshold_m``.  This function does NOT model that blend, so
    on a raster carrying ice its answer would be a measurement of a different array.
    It therefore counts the ice cells and, if there are any, returns
    ``status="NOT_COMPARABLE"`` with the count and no excursion -- it does not report
    a number it cannot stand behind.  On the Khvalynian rasters the count is 0 and
    ``SequenceLimits.ice_max_m`` is ``None`` for the same reason.

    WHAT IT COSTS, MEASURED
    -----------------------
    One full-resolution read and one hillshade per epoch.  Called from
    :func:`build_animation` immediately after that epoch's own render, it costs
    **5.37 s at 17.0 ka and 5.48 s at 16.5 ka against renders of 189.04 s and
    210.82 s** -- 2.8 % and 2.6 %.  The read is nearly free at that point because the
    render just read the same 42 MB file and the OS page cache still holds it; called
    cold it would pay the full read.  Both numbers are in ``seconds`` in the result,
    so no run relies on these two.
    """
    from matplotlib.colors import LightSource
    from scipy.ndimage import gaussian_filter

    from paleoeurope.viz.paleocostline_render import (
        HILLSHADE_ALTDEG,
        HILLSHADE_AZDEG,
        _raw_hillshade_intensity,
        analytic_shade_range,
        build_epoch_mosaic,
    )

    started = time.time()
    shade_min, shade_max = analytic_shade_range(HILLSHADE_ALTDEG)
    dem, ice, xform, bounds, _crs, _res = build_epoch_mosaic(
        {"khv": str(raster_path)}, config
    )
    nodata = np.isnan(dem)
    n_ice = int(((~nodata) & (ice > config.ice_threshold_m)).sum())
    common = {
        "raster": str(raster_path),
        "max_px": int(config.max_px),
        "vert_exag": float(config.vert_exag),
        "azdeg": float(HILLSHADE_AZDEG),
        "altdeg": float(HILLSHADE_ALTDEG),
        "shade_min_analytic": float(shade_min),
        "shade_max_analytic": float(shade_max),
        "ice_cells": n_ice,
        "budget": SHADE_EXCURSION_BUDGET,
        "budget_rule": (
            "one float32 epsilon. The DEM is read float32 and the unit-normal "
            "normalisation inside LightSource.shade_normals is where n.d rounds "
            "through a ceiling it can attain exactly; an excursion larger than the "
            "precision of that arithmetic is not arithmetic"
        ),
    }
    if n_ice:
        del dem, ice
        return {
            **common,
            "status": "NOT_COMPARABLE",
            "why": (
                f"{n_ice:,} cells carry ice above ice_threshold_m, and the renderer "
                "substitutes an ice-dome blend into dem_fs there. This function does "
                "not model that blend, so any excursion it measured would be taken "
                "off a different array than the frame was drawn from"
            ),
            "seconds": round(time.time() - started, 2),
        }

    dem_sm = gaussian_filter(np.where(nodata, 0.0, dem), sigma=1.5)
    dem_sm = np.where(nodata, np.nan, dem_sm)
    dem_fs = np.where(nodata, 0.0, dem_sm)
    del dem_sm
    cos_lat = np.cos(np.radians((bounds.bottom + bounds.top) / 2.0))
    light = LightSource(azdeg=HILLSHADE_AZDEG, altdeg=HILLSHADE_ALTDEG)
    raw = _raw_hillshade_intensity(
        dem_fs,
        light,
        config.vert_exag,
        abs(xform.a) * 111_320.0 * cos_lat,
        abs(xform.e) * 111_320.0,
    )
    above = raw > shade_max
    below = raw < shade_min
    raw_max = float(raw.max())
    raw_min = float(raw.min())
    n_above = int(above.sum())
    n_below = int(below.sum())
    n_cells = int(raw.size)
    where: list[dict] = []
    if n_above:
        rows, cols = np.nonzero(above)
        for row, col in list(zip(rows, cols))[:8]:
            lon, lat = xform * (float(col) + 0.5, float(row) + 0.5)
            where.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "raw_intensity": float(raw[row, col]),
                    "dem_fs_m": float(dem_fs[row, col]),
                }
            )
    del dem, ice, dem_fs, raw, above, below
    excursion = max(raw_max - shade_max, 0.0)
    return {
        **common,
        "status": "MEASURED",
        "cells_above_shade_max": n_above,
        "cells_below_shade_min": n_below,
        "mosaic_cells": n_cells,
        "valid_cells_note": (
            "counts are over the whole mosaic; the frame's own .palette.json counts "
            "only cells that are drawn as land or ice, so its number is <= this one"
        ),
        "raw_intensity_max": raw_max,
        "raw_intensity_min": raw_min,
        "excursion_above_ceiling": excursion,
        "excursion_in_float32_eps": excursion / SHADE_EXCURSION_BUDGET,
        "clearance_below_floor": float(raw_min - shade_min),
        "within_budget": excursion <= SHADE_EXCURSION_BUDGET,
        "where": where,
        "seconds": round(time.time() - started, 2),
    }


def assert_only_the_analytic_shade_ceiling_was_marked(
    records: Sequence[dict],
) -> dict:
    """Per quantity, not per frame: what ``clip_policy="raise"`` would have done, better.

    ``clip_policy="raise"`` is all-or-nothing over six quantities, five of which are
    extrema MEASURED over the sequence's rasters and one of which -- ``shade_max`` --
    is a geometric supremum the terrain attains, so the blunt policy is unusable
    here.  This splits them:

    * any of :data:`MEASURED_LIMIT_QUANTITIES` non-zero on any frame ->
      :class:`ClipPolicyBreachError`, naming the frame and the count.  This is what
      ``"raise"`` would have done, and it runs on every frame rather than stopping at
      the first, so the report is the whole sequence.
    * ``shade_above_shade_max`` is allowed, and every epoch it is non-zero on must
      carry a :func:`measure_analytic_shade_excursion` result from the SAME run whose
      excursion is inside :data:`SHADE_EXCURSION_BUDGET`.  A frame marked for the
      ceiling with no measurement beside it is refused too: an unmeasured excursion
      is not a small one.

    Returns the tally that goes in the manifest.
    """
    breaches: list[str] = []
    unmeasured: list[str] = []
    over_budget: list[str] = []
    marked: dict[str, int] = {}
    per_frame: list[dict] = []
    for record in records:
        if not record.get("png"):
            continue
        palette = record.get("palette_sidecar") or {}
        by_quantity = (palette.get("out_of_range") or {}).get("by_quantity") or {}
        epoch = record.get("epoch_ka")
        for name in MEASURED_LIMIT_QUANTITIES:
            count = int(by_quantity.get(name) or 0)
            if count:
                breaches.append(f"{epoch:g} ka: {name}={count:,}")
        ceiling = int(by_quantity.get("shade_above_shade_max") or 0)
        marked[f"{epoch:g}"] = ceiling
        measurement = record.get("shade_excursion") or {}
        if ceiling:
            if measurement.get("status") != "MEASURED":
                unmeasured.append(
                    f"{epoch:g} ka: {ceiling:,} cell(s) marked, measurement status "
                    f"{measurement.get('status', 'ABSENT')!r}"
                )
            elif not measurement.get("within_budget"):
                over_budget.append(
                    f"{epoch:g} ka: excursion "
                    f"{measurement['excursion_above_ceiling']:.6e} = "
                    f"{measurement['excursion_in_float32_eps']:.3f} float32 eps"
                )
        per_frame.append(
            {
                "epoch_ka": epoch,
                "shade_above_shade_max": ceiling,
                "measured_quantities_out_of_range": {
                    name: int(by_quantity.get(name) or 0)
                    for name in MEASURED_LIMIT_QUANTITIES
                },
                "excursion": measurement or None,
            }
        )

    if breaches:
        raise ClipPolicyBreachError(
            "these frames were MARKED out of range for a quantity whose limit was "
            "measured over the sequence's own rasters, which clip_policy='raise' "
            "would have refused and so does this: "
            + "; ".join(breaches)
            + ". The limits were computed over the epochs that rendered, so this is "
            "not a stale-limits problem -- recompute nothing until the frame is "
            "understood."
        )
    if unmeasured:
        raise ShadeExcursionError(
            "these frames were marked at the analytic hillshade ceiling with no "
            "excursion measurement from this run beside them, and an unmeasured "
            "excursion is not a small one: " + "; ".join(unmeasured)
        )
    if over_budget:
        raise ShadeExcursionError(
            "these frames rose above the analytic hillshade ceiling by more than one "
            f"float32 epsilon ({SHADE_EXCURSION_BUDGET:.6e}), which is more than the "
            "arithmetic can explain: " + "; ".join(over_budget)
        )
    return {
        "policy": CLIP_POLICY,
        "rule": (
            "clip_policy='mark' plus a per-QUANTITY check. The five measured "
            "quantities must be zero on every frame -- that is exactly what "
            "clip_policy='raise' enforces, and it is enforced here over the whole "
            "sequence instead of stopping at the first frame. shade_above_shade_max "
            "is allowed because SequenceLimits.shade_max is +1.0 from "
            "analytic_shade_range, a geometric supremum the terrain ATTAINS at "
            "native resolution, so a cell rounds through it in float32 and 'raise' "
            "would abort a frame against limits computed from that same frame. Every "
            "epoch it is non-zero on has its excursion measured on its own raster in "
            "this run and checked against one float32 epsilon"
        ),
        "measured_quantities_that_must_be_zero": list(MEASURED_LIMIT_QUANTITIES),
        "cells_marked_at_the_analytic_ceiling_by_epoch": marked,
        "total_cells_marked": sum(marked.values()),
        "budget_float32_eps": SHADE_EXCURSION_BUDGET,
        "per_frame": per_frame,
    }


def render_one(
    plan: EpochPlan,
    out_dir: Path,
    *,
    config: RenderConfig,
    sill_tolerance_m: float,
    corridor: str = "step",
) -> dict:
    """Render one planned epoch.  Returns a record; never raises for a bad frame.

    ``corridor`` is the caller's, and there is no fallback.  ``"step"`` draws every
    basin flat at its own declared level and MARKS the corridor; ``"ramp"``
    interpolates the surface across the ONE donor->receiver corridor and draws it.
    ``assign_basins`` raises ``CorridorNotRampableError`` when more than one pair is
    donor->receiver at once, and that exception arrives here as a REFUSED FRAME
    naming the class -- it is never caught into a silent step render, because a
    sequence in which some frames interpolate a surface and others do not, without
    saying which, is the substitution this project keeps deleting.  Whether ramp
    could raise on this sequence is measured per epoch by :func:`ramp_legality` from
    this run's own frames.
    """
    from scripts import bake_black_sea as bbs

    record: dict[str, Any] = {"epoch_ka": plan.epoch_ka, "corridor_requested": corridor}
    started = time.time()
    try:
        png = render_paleocostline_epoch(
            plan.epoch_ka,
            {"khv": plan.raster_path},
            bbs.SPRATT_CSV,
            bbs.PALEOCOASTLINES_SHP,
            out_dir,
            config,
            data_source_label="khvalynian",
            basin_seeds=plan.basin_seeds(),
            basin_corridor=corridor,
            basin_sill_expectations=sill_expectations_for(
                plan.epoch_ka, plan.dem_frame, plan.dz_bosphorus_m
            ),
            basin_sill_tolerance_m=sill_tolerance_m,
        )
        record["png"] = str(png)
        record["refusal"] = None
        record["basins_sidecar"] = _read_json(Path(png).with_suffix(".basins.json"))
        record["palette_sidecar"] = _read_json(Path(png).with_suffix(".palette.json"))
    except Exception as exc:  # noqa: BLE001 - a refused frame keeps its slot
        record["png"] = None
        record["refusal"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
    record["wall_s"] = round(time.time() - started, 2)
    return record


#: What the inset panel is titled, in the figure the renderer draws.  A separate
#: constant because it is the only thing that distinguishes the two panels for a
#: reader who has the PNG and not the manifest.
#:
#: IT USED TO READ "same cells", FULL STOP, AND THAT WAS TOO STRONG.  The CROP holds
#: the same cells -- :func:`crop_to_extent` proves it cell for cell -- but the WATER
#: is re-derived on the crop, and the de-speckle inside the renderer treats everything
#: outside the array as not-water.  So a water body the full frame continues past the
#: crop line loses a rim along that line, and cells drawn blue in the panel above are
#: drawn as land here.  How deep the rim is is measured, not asserted, by
#: :func:`measure_despeckle_edge_bite`; what it cost on this run's own panels is in
#: ``manifest.inset.edge_erosion`` and in the composed caption.  A reader who has only
#: the PNG gets the warning in this line, which is why the line carries it.
INSET_FIGURE_TITLE = (
    "INSET — the Manych corridor: the same source cells and the same colour mapping, "
    "but its WATER IS RE-FLOODED ON THIS CROP, so the shoreline along the panel's own "
    "four edges is eroded against the full frame above (see the caption)"
)


def measure_despeckle_edge_bite(size: int = 40) -> dict:
    """How deep the renderer's own de-speckle bites into water at an array edge.

    MEASURED, on a solid-water square, by running the exact operator the renderer
    runs (:func:`~paleoeurope.viz.paleocostline_render.despeckle_ocean_mask`, its
    closing then its opening) and reading how many rows of the result are dry.  It
    costs microseconds and it carries no typed constant: if the iteration counts or
    the structuring element ever change, this number changes with them.

    Why it exists: ``scipy.ndimage``'s binary erosion treats cells outside the array
    as FALSE, so a water body that the full scene continues past the crop line is
    eroded from that line inwards on the inset -- an observation boundary being drawn
    as a shore.  The full frame has the same behaviour at ITS edges, but its edges are
    the scene's own and mostly stand on land; the inset's four edges cut open sea.
    """
    from paleoeurope.viz.paleocostline_render import (
        OCEAN_CLOSING_ITERATIONS,
        OCEAN_OPENING_ITERATIONS,
        despeckle_ocean_mask,
    )

    n = int(size)
    water = np.ones((n, n), dtype=bool)
    kept = despeckle_ocean_mask(water, np.zeros((n, n), dtype=bool))
    middle = kept[n // 2]
    edge_cells = int(np.argmax(middle)) if middle.any() else n // 2
    return {
        "measured_on": (
            f"a solid-water {n}x{n} array with no nodata, through "
            "paleoeurope.viz.paleocostline_render.despeckle_ocean_mask -- the same "
            "call the multi-basin path makes on BasinField.water_mask"
        ),
        "closing_iterations": int(OCEAN_CLOSING_ITERATIONS),
        "opening_iterations": int(OCEAN_OPENING_ITERATIONS),
        "edge_bite_cells": edge_cells,
        "cells_lost_of_the_probe": int(water.sum() - kept.sum()),
        "rule": (
            "edge_bite_cells is how many cells deep the straight side of an "
            "all-water array is turned to land by the de-speckle alone, with no "
            "terrain in it. It is a property of the OPERATOR and of the array "
            "boundary, not of this scene, and it is what an inset panel pays along "
            "every edge of its crop that cuts water. The corners lose more than the "
            "sides, which is why cells_lost_of_the_probe exceeds the perimeter times "
            "the bite"
        ),
    }


def inset_edge_erosion(
    full_sidecar: Optional[dict],
    inset_sidecar: Optional[dict],
    crop: Optional[dict],
) -> dict:
    """What the crop's own four edges cost the inset panel's water, as numbers.

    Everything here is read off the two panels' own ``.basins.json`` morphology
    blocks plus the crop window, and the operator's bite is measured in this call.
    Nothing is inferred from the pictures and no per-cell mask is available to this
    module, so what CANNOT be separated is said rather than estimated: the
    de-speckle's ``cells_removed`` counts the opening over the WHOLE panel, of which
    the edge rim is a part, and this block reports both it and the rim's own upper
    bound instead of a difference it cannot measure.
    """
    bite = measure_despeckle_edge_bite()
    inset_morph = ((inset_sidecar or {}).get("morphology") or {})
    full_morph = ((full_sidecar or {}).get("morphology") or {})
    full_grid = ((full_sidecar or {}).get("grid") or {})
    window = ((crop or {}).get("window") or {})
    width, height = window.get("width"), window.get("height")
    perimeter = (
        None if width is None or height is None
        else 2 * (int(width) + int(height)) - 4
    )
    full_perimeter = (
        None if not full_grid.get("width") or not full_grid.get("height")
        else 2 * (int(full_grid["width"]) + int(full_grid["height"])) - 4
    )
    depth = bite["edge_bite_cells"]
    upper_bound = None if perimeter is None else perimeter * depth
    removed = inset_morph.get("cells_removed")
    full_removed = full_morph.get("cells_removed")
    return {
        "what_this_is": (
            "the inset panel's shoreline is NOT the full panel's along the crop's own "
            "four edges. The DEM cells are identical (crop.verdict is "
            "CELL_FOR_CELL_IDENTICAL_TO_THE_SOURCE_WINDOW) and the colour mapping is "
            "identical (inset.colour_mapping), but the water mask is re-derived on "
            "the crop and the morphological de-speckle inside the renderer treats "
            "everything outside the array as not-water. So water that the full frame "
            "carries across the crop line is eroded back from that line, and cells "
            "painted blue in the panel above are painted as land here"
        ),
        "operator_bite": bite,
        "crop_window_cells": (
            None if width is None else {"width": int(width), "height": int(height)}
        ),
        "crop_perimeter_cells": perimeter,
        "upper_bound_on_the_eroded_rim_cells": upper_bound,
        "how_the_upper_bound_is_built": (
            "crop_perimeter_cells x operator_bite.edge_bite_cells: every edge cell of "
            "the crop, as if every one of them were water. It is an UPPER BOUND and "
            "not a measurement -- the edges that cut land lose nothing -- and it is "
            "reported as one because the drawn masks are not in the sidecar and this "
            "module will not estimate a number it cannot read"
        ),
        "inset_panel_despeckle": {
            "cells_removed": removed,
            "cells_added": inset_morph.get("cells_added"),
            "cells_drawn": inset_morph.get("cells_drawn"),
        },
        "full_panel_despeckle": {
            "cells_removed": full_morph.get("cells_removed"),
            "cells_added": full_morph.get("cells_added"),
            "cells_drawn": full_morph.get("cells_drawn"),
        },
        "why_the_two_removed_counts_are_worth_comparing": (
            "the de-speckle's opening removes speckle AND the edge rim, and it cannot "
            "be split from here. The comparison that makes the rim visible anyway is "
            "the two panels' cells_removed against the area each was run over: the "
            "inset is a small fraction of the scene and its four edges cut open sea, "
            "so a cells_removed of the same order as the full frame's is the rim and "
            "not speckle"
        ),
        "cells_removed_per_perimeter_cell": (
            None if not perimeter or removed is None
            else round(float(removed) / float(perimeter), 4)
        ),
        "full_panel_crop_perimeter_cells": full_perimeter,
        "full_panel_cells_removed_per_perimeter_cell": (
            None if not full_perimeter or full_removed is None
            else round(float(full_removed) / float(full_perimeter), 4)
        ),
        "how_to_read_the_two_ratios": (
            "cells the de-speckle removed, per cell of the panel's own border. The "
            "full frame's border is the scene's own and stands mostly on land, so "
            "its ratio is what speckle alone costs; the inset's four edges cut open "
            "sea on two of them. A ratio near the operator's bite is a rim, not "
            "speckle. Both are null when the panel's grid was not recorded, and "
            "neither is estimated"
        ),
        "what_is_NOT_affected": (
            "the sill verdict, the crest cell, the seeds and the colour mapping. The "
            "saddle is a minimax over a path that runs nowhere near the crop line "
            "(assert_route_inside_the_inset measures its margins), and the crop is "
            "cell-for-cell the source window. This is a statement about the drawn "
            "SHORELINE at the panel's own border, and about nothing else"
        ),
    }


def render_inset(
    inset: InsetPlan,
    out_dir: Path,
    *,
    config: RenderConfig,
    sill_tolerance_m: float,
    corridor: str = "step",
) -> dict:
    """Render one epoch's inset.  Same shape as :func:`render_one`, own registry.

    The *config* handed in must be the SAME object the full frame is rendered with,
    apart from ``extra_figure_title``: the two pictures are composed into one canvas
    and quantized into one colour table, so a second ``SequenceLimits`` would map the
    same elevation to two different colours in two halves of one image.  That the
    limits are in fact the same object AND field-identical is asserted by
    :func:`assert_one_colour_mapping`, not assumed here.
    """
    from scripts import bake_black_sea as bbs

    record: dict[str, Any] = {
        "epoch_ka": inset.epoch_ka,
        "extent": inset.extent.name,
        "corridor_requested": corridor,
    }
    started = time.time()
    if not inset.renderable:
        record.update(png=None, refusal=inset.refusal, wall_s=0.0)
        return record
    try:
        cfg = dataclasses.replace(config, extra_figure_title=INSET_FIGURE_TITLE)
        # Asserted on the OBJECT that is about to be handed to the renderer, not on
        # one built beside it for the check.
        record["colour_mapping"] = assert_one_colour_mapping(config, cfg)
        png = render_paleocostline_epoch(
            inset.epoch_ka,
            {"khv_inset": inset.raster_path},
            bbs.SPRATT_CSV,
            bbs.PALEOCOASTLINES_SHP,
            out_dir,
            cfg,
            data_source_label="khvalynian-manych-inset",
            basin_seeds=inset.basin_seeds(),
            basin_corridor=corridor,
            basin_sill_expectations=inset_sill_expectations(inset),
            basin_sill_tolerance_m=sill_tolerance_m,
        )
        record["png"] = str(png)
        record["refusal"] = None
        record["basins_sidecar"] = _read_json(Path(png).with_suffix(".basins.json"))
        record["palette_sidecar"] = _read_json(Path(png).with_suffix(".palette.json"))
        record["extent_check"] = assert_route_inside_the_inset(
            inset, record["basins_sidecar"]
        )
    except Exception as exc:  # noqa: BLE001 - a refused inset keeps its slot
        record["png"] = None
        record["refusal"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
    record["wall_s"] = round(time.time() - started, 2)
    return record


class PaletteNotSharedError(AssertionError):
    """The two panels of one canvas were not drawn through one colour mapping."""


#: The colour-bearing fields of :class:`SequenceLimits`.  Everything a pixel's colour
#: depends on; ``epochs``, ``rasters`` and ``computed_from`` are provenance and are
#: expected to be equal here too, but they are checked separately so a failure says
#: which kind it is.
_COLOUR_FIELDS = (
    "depth_min_m", "depth_max_m", "land_min_m", "land_max_m", "ice_max_m",
    "shade_min", "shade_max", "dem_data_min_m", "dem_data_max_m",
    "depth_data_max_m", "depth_percentile", "land_percentile_lo",
    "land_percentile_hi", "vert_exag", "azdeg", "altdeg",
)


def assert_one_colour_mapping(
    full_config: RenderConfig, inset_config: RenderConfig
) -> dict:
    """Refuse to compose two panels that were not drawn through one mapping.

    Checked three ways, because two of them can pass while the picture is still
    wrong: object IDENTITY (the strongest -- there is one mapping, not two that
    agree), field-by-field EQUALITY of every colour-bearing number, and equality of
    the clip policy, which decides what happens to a cell the mapping cannot
    represent.  A frame drawn with ``clip_policy='mark'`` beside one drawn with
    ``'clip'`` shares a ramp and not a picture.
    """
    a, b = full_config.sequence_limits, inset_config.sequence_limits
    if a is None or b is None:
        raise PaletteNotSharedError(
            "one of the two panels has no pinned SequenceLimits "
            f"(full={a is not None}, inset={b is not None}). Without it the "
            "compositor normalises that panel by ITS OWN extremes, so the same "
            "elevation gets two colours in one image -- which is the entire reason "
            "the inset is not allowed its own limits."
        )
    differing = [
        {"field": name, "full": getattr(a, name), "inset": getattr(b, name)}
        for name in _COLOUR_FIELDS
        if getattr(a, name) != getattr(b, name)
    ]
    if differing:
        raise PaletteNotSharedError(
            "the two panels' colour mappings differ in "
            f"{len(differing)} field(s): {differing}. Composing them would recolour "
            "the same elevation differently in two halves of one picture, and the "
            "GIF quantizer would then spend its 256 entries reconciling a "
            "disagreement nobody intended."
        )
    if full_config.clip_policy != inset_config.clip_policy:
        raise PaletteNotSharedError(
            f"clip policies differ: full={full_config.clip_policy!r}, "
            f"inset={inset_config.clip_policy!r}."
        )
    return {
        "same_object": a is b,
        "colour_fields_compared": list(_COLOUR_FIELDS),
        "colour_fields_differing": 0,
        "clip_policy": full_config.clip_policy,
        "epochs_equal": tuple(a.epochs) == tuple(b.epochs),
        "raster_sets_equal": (
            sorted(fp.path for fp in a.rasters) == sorted(fp.path for fp in b.rasters)
        ),
        "verdict": (
            "ONE_MAPPING_SAME_OBJECT" if a is b else "ONE_MAPPING_BY_VALUE"
        ),
        "why_identity_is_checked_as_well": (
            "field equality can hold between two objects that were computed over "
            "different raster sets and happened to agree; identity cannot. Both are "
            "reported so a run where they diverge is visible rather than argued "
            "about"
        ),
    }


#: How :func:`measure_legend_occlusion` decides what it is looking at.  Every step is
#: a property of the SAVED PICTURE, because that is the only thing this module may
#: read: ``render_paleocostline_epoch`` cannot be imported for its axes geometry and
#: cannot be edited to record it.
LEGEND_OCCLUSION_RULE = (
    "measured on the panel's own PNG, never on the figure object. THE MAP RECTANGLE "
    "is the axes' spine box, found by the LONGEST CONTIGUOUS RUN of dark pixels in "
    "each row: the map's top and bottom spines are the longest unbroken dark lines "
    "in the picture, and where each run starts and ends is where the left and right "
    "spines are. It is NOT found by counting dark pixels per row and column, which "
    "is what this function did first and which was wrong: the colour bar is a second "
    "framed box, taller than the map, so its two vertical edges hold MORE dark pixels "
    "than the map's own spines do and the rectangle grew rightwards to swallow it -- "
    "measured on the 16.5 ka corridor panel, a map reported 3078 px wide that is "
    "2940, a colour bar counted as water, and every figure downstream wrong in the "
    "safe-looking direction. A run length is immune to that because the colour bar's "
    "horizontal edges are only as long as the colour bar is wide. THE LEGEND "
    "RECTANGLE is the "
    "bounding box of the largest 4-connected dark component inside that map "
    "rectangle, MINUS A TWO-PIXEL RING along the spines. The ring is not cosmetic: "
    "the spines are themselves a dark connected component, and a legend pushed into "
    "a corner touches them, at which point the largest dark component is the spine "
    "box and its bounding box is the whole map -- the detector would report a legend "
    "covering everything. Its fill ratio (component cells over bounding-box cells) is "
    "reported beside it, because a legend is a filled rounded box and a ratio far "
    "below one means the detector found something else and the numbers should not be "
    "believed. "
    "WATER is counted by hue rather than by brightness -- blue at least as strong as "
    "green and at least four levels above red -- because the legend patch is "
    "TRANSLUCENT: it darkens the terrain under it without replacing its colour, so a "
    "brightness test would report the water it hides as absent and understate itself. "
    "What is reported is therefore what the legend COVERS, not what it erases; the "
    "cells beneath it are still on the page, they are merely unreadable behind a "
    "swatch table. Nothing here is compared against the renderer's own land/sea "
    "percentages, which count RASTER CELLS of the whole crop and not pixels of a "
    "drawn axes."
)

#: How far in from the spines :func:`measure_legend_occlusion` starts looking for the
#: legend.  Two pixels, because a spine is one pixel wide and a legend pushed against
#: it would otherwise 4-connect to the spine box -- whose bounding box is the whole
#: map.  That is not a hypothetical: it is what this function did on its first
#: synthetic frame, and it reported a legend covering 100 % of the picture.
LEGEND_SPINE_MARGIN_PX = 2

#: Digit-free.  Every value is interpolated from the frame the caption belongs to.
LEGEND_OCCLUSION_TEMPLATE = (
    "THE LEGEND SITS INSIDE THE MAP AND THIS IS WHAT IT COVERS, measured on this "
    "panel: its box is {legend_w} x {legend_h} px of a {axes_w} x {axes_h} px map, "
    "{legend_pct} % of it, and {water_pct} % of the water pixels this panel draws "
    "are under it ({water_under} of {water_total}). It is translucent, so those "
    "cells are darkened rather than deleted. It cannot be moved from here: the "
    "renderer that places it is not writable by this module, so the only lever this "
    "build has is the figure it is drawn into -- the legend is fixed in points and "
    "the map is fixed in inches, so a larger figure shrinks the legend against the "
    "map and nothing else about the picture changes."
)


def _longest_dark_run_per_row(dark) -> dict:
    """For every row of a boolean array: the longest unbroken True run and where it starts.

    Vectorised over rows because a rendered frame is a few thousand of them and this
    runs once per panel per epoch.  Returns ``{"length": (h,), "start": (h,)}``.
    """
    import numpy as _np

    h, w = dark.shape
    padded = _np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = dark
    diff = _np.diff(padded.astype(_np.int8), axis=1)
    length = _np.zeros(h, dtype=_np.int64)
    start = _np.zeros(h, dtype=_np.int64)
    for row in range(h):
        starts = _np.flatnonzero(diff[row] == 1)
        if starts.size == 0:
            continue
        ends = _np.flatnonzero(diff[row] == -1)
        runs = ends - starts
        best = int(_np.argmax(runs))
        length[row] = int(runs[best])
        start[row] = int(starts[best])
    return {"length": length, "start": start}


def measure_legend_occlusion(
    png: Path | str,
    extent: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """How much of the drawn map the legend box covers.  See :data:`LEGEND_OCCLUSION_RULE`.

    *extent* is ``(west, south, east, north)`` and is optional: without it the pixel
    numbers are still measured, and only the geographic window the legend covers is
    reported as ``None``.  It is never guessed from the picture.
    """
    import numpy as _np
    from PIL import Image as _Image
    from scipy import ndimage as _ndimage

    rgb = _np.asarray(_Image.open(png).convert("RGB")).astype(_np.int16)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = luma < 128
    if not dark.any():
        return {
            "rule": LEGEND_OCCLUSION_RULE,
            "status": "NO_MAP_RECTANGLE_FOUND",
            "why": "the picture holds no dark pixel at all, so it has no spines",
        }
    runs = _longest_dark_run_per_row(dark)
    best = int(_np.argmax(runs["length"]))
    span = int(runs["length"][best])
    if span < 2:
        return {
            "rule": LEGEND_OCCLUSION_RULE,
            "status": "NO_MAP_RECTANGLE_FOUND",
            "why": "no row holds an unbroken dark line, so there is no spine to find",
        }
    ax_x0, ax_x1 = int(runs["start"][best]), int(runs["start"][best]) + span - 1
    # Every row whose longest run is that long AND starts and ends where this one
    # does is a horizontal spine of the SAME box; the topmost and bottommost of them
    # are the map's two.  The tolerance is in pixels because a spine is antialiased
    # at its ends, not because a different box would be accepted: a colour bar's
    # horizontal edges are hundreds of pixels short of this length.
    same = (
        (runs["length"] >= span - 2)
        & (_np.abs(runs["start"] - ax_x0) <= 2)
    )
    spine_rows = _np.flatnonzero(same)
    ax_y0, ax_y1 = int(spine_rows[0]), int(spine_rows[-1])
    axes_w, axes_h = ax_x1 - ax_x0 + 1, ax_y1 - ax_y0 + 1
    sub = rgb[ax_y0 : ax_y1 + 1, ax_x0 : ax_x1 + 1]
    water = (sub[..., 2] >= sub[..., 1]) & (sub[..., 2] - sub[..., 0] >= 4)
    water_total = int(water.sum())
    out: dict[str, Any] = {
        "rule": LEGEND_OCCLUSION_RULE,
        "status": "MEASURED",
        "png": str(Path(png).resolve()),
        "image_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "map_rect_px": [ax_x0, ax_y0, ax_x1, ax_y1],
        "map_px": [axes_w, axes_h],
        "map_area_px": axes_w * axes_h,
        "spine_run_px": span,
        "spine_rows_found": int(spine_rows.size),
        "water_px_in_map": water_total,
    }
    if extent is not None:
        west, south, east, north = (float(v) for v in extent)
        out["extent_deg"] = [west, south, east, north]
        out["px_per_deg_lon"] = round(axes_w / (east - west), 4)
        out["px_per_deg_lat"] = round(axes_h / (north - south), 4)
    ring = LEGEND_SPINE_MARGIN_PX
    out["spine_margin_px"] = ring
    if axes_w <= 2 * ring or axes_h <= 2 * ring:
        out["legend"] = {
            "found": False,
            "why": "the map is not wider than the spine ring the search excludes",
        }
        return out
    inner_luma = 0.299 * sub[..., 0] + 0.587 * sub[..., 1] + 0.114 * sub[..., 2]
    inner = (inner_luma < 110)[ring : axes_h - ring, ring : axes_w - ring]
    labels, n = _ndimage.label(inner)
    if n == 0:
        out["legend"] = {"found": False, "why": "no dark component inside the map"}
        return out
    sizes = _ndimage.sum(inner, labels, range(1, n + 1))
    which = int(_np.argmax(sizes)) + 1
    ys, xs = _np.nonzero(labels == which)
    lx0, lx1 = int(xs.min()) + ring, int(xs.max()) + ring
    ly0, ly1 = int(ys.min()) + ring, int(ys.max()) + ring
    lw, lh = lx1 - lx0 + 1, ly1 - ly0 + 1
    under = _np.zeros(water.shape, dtype=bool)
    under[ly0 : ly1 + 1, lx0 : lx1 + 1] = True
    water_under = int((water & under).sum())
    legend: dict[str, Any] = {
        "found": True,
        "rect_in_map_px": [lx0, ly0, lx1, ly1],
        "rect_in_image_px": [lx0 + ax_x0, ly0 + ax_y0, lx1 + ax_x0, ly1 + ax_y0],
        "size_px": [lw, lh],
        "area_px": lw * lh,
        "component_px": int(sizes[which - 1]),
        "fill_ratio": round(float(sizes[which - 1]) / (lw * lh), 4),
        "pct_of_map": round(100.0 * lw * lh / (axes_w * axes_h), 3),
        "water_px_under": water_under,
        "pct_of_water_under": (
            round(100.0 * water_under / water_total, 3) if water_total else None
        ),
    }
    if extent is not None:
        west, south, east, north = (float(v) for v in extent)
        legend["window_deg"] = [
            round(west + (east - west) * lx0 / axes_w, 4),
            round(north - (north - south) * (ly1 + 1) / axes_h, 4),
            round(west + (east - west) * (lx1 + 1) / axes_w, 4),
            round(north - (north - south) * ly0 / axes_h, 4),
        ]
    out["legend"] = legend
    return out


#: How the two panels are put on one page.  No scaling anywhere: a resample would
#: invent colours neither render produced, and the whole claim of this composition is
#: that every pixel in it came out of a render that consumed the pinned mapping.
COMPOSE_RULE = (
    "the full-extent frame above, the inset below, both at their OWN pixel size, "
    "centred on a canvas as wide as the wider of the two and as tall as their sum "
    "plus the caption band. NEVER scaled and NEVER cropped: any resampling would "
    "blend palette-mapped pixels into colours no render produced, and the composed "
    "canvas is then quantized as one image. NO LOCATOR RECTANGLE is drawn on the "
    "full frame: render_paleocostline_epoch saves with bbox_inches='tight', so the "
    "axes' pixel box is not recoverable from the PNG, and a rectangle placed by "
    "eye would be a typed coordinate on a figure whose whole subject is a measured "
    "one. The inset bounds are stated in the caption instead."
)

#: The close-up build's own composition rule.  It differs from :data:`COMPOSE_RULE`
#: in exactly one thing -- WHICH panels reach the page -- and it is a separate
#: constant rather than a branch inside that sentence because both spellings ship,
#: each in its own manifest, and a reader of either must be told what THAT canvas is.
CLOSEUP_COMPOSE_RULE = (
    "the corridor frame ALONE, at its own pixel size, on a canvas as wide as it is "
    "plus the caption band. The whole-scene panel is still RENDERED, hashed and kept "
    "beside it under frames/full, and every per-epoch verdict, sill, seed and clip "
    "audit in the manifest is that panel's; it is simply not put on this page, "
    "because the subject of this build is the corridor and a whole-scene panel above "
    "it is what made the corridor small. IT IS NOT SHRUNK INTO A LOCATOR. A locator "
    "would have to be a resampled copy, and resampling blends palette-mapped pixels "
    "into colours no render produced -- the composed canvas is quantized as one "
    "image through ONE master palette, so invented colours would cost palette "
    "entries the corridor's own water and terrain need. The reader places the "
    "corridor from the caption's extent, from the axes' own degree labels, and from "
    "the whole-scene panel on disk."
)


def _caption_font(size_px: int):
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size_px)
    except Exception:  # pragma: no cover - font discovery is environment-specific
        return ImageFont.load_default()


def _wrap_to_width(text: str, font, max_px: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words, current = paragraph.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.getlength(candidate) <= max_px or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def compose_full_and_inset(
    full_png: Path | str,
    inset_png: Optional[Path | str],
    dst: Path | str,
    *,
    caption: str,
    font_px: int = 22,
    margin_px: int = 24,
    drop_full: bool = False,
) -> dict:
    """One canvas holding both panels and the caption.  Returns what it did, measured.

    *inset_png* may be ``None``: an epoch whose inset refused is composed as the full
    frame plus a caption that SAYS the inset is missing and why.  The alternative --
    quietly emitting the full frame alone -- would make a run with a broken inset
    look like a run that was never asked for one.

    ``drop_full=True`` is the CLOSE-UP build: the inset panel goes on the page alone
    and the whole-scene frame is recorded in ``full_panel_omitted`` -- path, sha256
    and the reason -- so the composed record still names the panel that carries the
    verdict.  It is REFUSED when there is no inset to put on the page: a canvas with
    neither panel would be a caption band claiming to be a frame.
    """
    from PIL import Image, ImageDraw

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if drop_full and inset_png is None:
        raise ValueError(
            "compose_full_and_inset(drop_full=True) with inset_png=None would put "
            "NOTHING on the page. The close-up build drops the whole-scene panel "
            "because the corridor panel replaces it; with no corridor panel there is "
            "nothing to compose, and an epoch whose inset refused must reach the "
            "sequence as a refusal frame carrying that refusal, not as a caption "
            "band. Compose it with drop_full=False, or let the epoch refuse."
        )
    panels: list[tuple[str, Path]] = [] if drop_full else [("full", Path(full_png))]
    if inset_png is not None:
        panels.append(("inset", Path(inset_png)))
    images = [(role, Image.open(p).convert("RGB"), p) for role, p in panels]
    try:
        width = max(im.width for _, im, _ in images) + 2 * margin_px
        font = _caption_font(font_px)
        lines = _wrap_to_width(caption, font, width - 2 * margin_px)
        line_h = int(round(font_px * 1.35))
        caption_h = len(lines) * line_h + 2 * margin_px
        height = sum(im.height for _, im, _ in images) + caption_h + margin_px
        canvas = Image.new("RGB", (width, height), CANVAS_PAD_RGB)
        placed: list[dict] = []
        y = 0
        for role, im, path in images:
            x = (width - im.width) // 2
            canvas.paste(im, (x, y))
            placed.append(
                {
                    "role": role,
                    "source": str(path.resolve()),
                    "source_sha256": sha256_of(path),
                    "size_px": [im.width, im.height],
                    "at_px": [int(x), int(y)],
                    "scaled": False,
                    "cropped": False,
                }
            )
            y += im.height
        draw = ImageDraw.Draw(canvas)
        text_y = y + margin_px
        for line in lines:
            draw.text((margin_px, text_y), line, fill=(20, 20, 20), font=font)
            text_y += line_h
        canvas.save(dst)
        size = (canvas.width, canvas.height)
    finally:
        for _, im, _ in images:
            im.close()
    omitted = None
    if drop_full:
        omitted = {
            "role": "full",
            "source": str(Path(full_png).resolve()),
            "source_sha256": sha256_of(full_png),
            "why": CLOSEUP_COMPOSE_RULE,
        }
    return {
        "rule": CLOSEUP_COMPOSE_RULE if drop_full else COMPOSE_RULE,
        "composed": str(dst.resolve()),
        "composed_sha256": sha256_of(dst),
        "size_px": [int(size[0]), int(size[1])],
        "panels": placed,
        "full_panel_omitted": omitted,
        "inset_present": inset_png is not None,
        "caption": caption,
        "caption_lines": len(lines),
        "font_px": int(font_px),
    }


#: What the two panels do NOT share, said on the canvas a PNG-only reader has.
#: Digit-free; every value is this frame's own.  See :func:`inset_edge_erosion`.
INSET_EROSION_TEMPLATE = (
    "THE TWO PANELS DO NOT DRAW THE SAME SHORELINE AT THE INSET'S OWN EDGES. The "
    "cells are the same and the colour mapping is the same, but the inset's water is "
    "re-flooded on the crop and the renderer's de-speckle treats everything outside "
    "an array as not-water, so water the full frame carries across the crop line is "
    "eroded back from it. Measured on the operator itself, a straight edge of solid "
    "water loses {bite} cell(s); this crop's perimeter is {perimeter} cell(s), so at "
    "most {bound} cell(s) of this panel's shoreline are the crop's edge rather than "
    "the terrain, against {removed} cell(s) its de-speckle removed in total. Nothing "
    "else moves with it: the sill verdict, the crest cell, the seeds and the colour "
    "mapping are the full frame's own."
)


def compose_caption(
    epoch_ka: float,
    addendum: Optional[dict],
    inset_record: Optional[dict],
    inset_plan: Optional[InsetPlan],
    *,
    edge_erosion: Optional[dict] = None,
    closeup: bool = False,
    legend_occlusion: Optional[dict] = None,
) -> str:
    """The caption band's text, built from measurements of THIS frame.

    Every numeral in it is interpolated from a record; the three sentences that carry
    none are the ones that say what the panels ARE.

    *edge_erosion* is what :func:`inset_edge_erosion` measured for this frame.  It is
    a keyword with a default because the caption must still be buildable for a frame
    whose inset refused -- and when it is absent the caption says nothing about the
    erosion rather than implying there was none.

    *closeup* says the whole-scene panel is not on this page (see
    :data:`CLOSEUP_COMPOSE_RULE`), and *legend_occlusion* is what
    :func:`measure_legend_occlusion` measured on the panel that IS.  Both default to
    the two-panel spelling, so nothing that already ships changes wording.
    """
    parts: list[str] = [
        (
            f"{epoch_ka:g} ka — the Manych corridor, and nothing else. "
            f"{CLOSEUP_COMPOSE_RULE}"
        )
        if closeup
        else f"{epoch_ka:g} ka — full scene above, Manych inset below. {COMPOSE_RULE}"
    ]
    if legend_occlusion and (legend_occlusion.get("legend") or {}).get("found"):
        legend = legend_occlusion["legend"]
        parts.append(
            LEGEND_OCCLUSION_TEMPLATE.format(
                legend_w=_fmt(legend["size_px"][0], ","),
                legend_h=_fmt(legend["size_px"][1], ","),
                axes_w=_fmt(legend_occlusion["map_px"][0], ","),
                axes_h=_fmt(legend_occlusion["map_px"][1], ","),
                legend_pct=_fmt(legend.get("pct_of_map"), ".2f"),
                water_pct=_fmt(legend.get("pct_of_water_under"), ".2f"),
                water_under=_fmt(legend.get("water_px_under"), ","),
                water_total=_fmt(legend_occlusion.get("water_px_in_map"), ","),
            )
        )
    if inset_plan is not None:
        west, south, east, north = inset_plan.extent.bounds()
        parts.append(
            f"INSET EXTENT '{inset_plan.extent.name}': {west:g}–{east:g} E / "
            f"{south:g}–{north:g} N."
        )
    if inset_record is not None and inset_record.get("refusal"):
        parts.append(
            "THE INSET REFUSED AND IS NOT DRAWN: " + str(inset_record["refusal"])
        )
    elif inset_record is not None:
        sidecar = inset_record.get("basins_sidecar") or {}
        entry = next(
            (
                s for s in (sidecar.get("sills") or [])
                if sorted(s.get("pair") or []) == ["caspian", "pontic"]
            ),
            {},
        )
        measured = entry.get("elevation_m")
        parts.append(
            "THE INSET IS GUARDED SEPARATELY, because it is a different grid: it "
            "measures its OWN caspian<->pontic saddle at "
            f"{_fmt(measured, '+.4f')} m and is refused if that leaves the same "
            "tolerance the full frame is held to. "
            + (entry.get("expectation_status") or "no expectation status recorded")
        )
        if edge_erosion:
            parts.append(
                INSET_EROSION_TEMPLATE.format(
                    bite=_fmt(
                        (edge_erosion.get("operator_bite") or {}).get(
                            "edge_bite_cells"
                        ),
                        ",",
                    ),
                    perimeter=_fmt(edge_erosion.get("crop_perimeter_cells"), ","),
                    bound=_fmt(
                        edge_erosion.get("upper_bound_on_the_eroded_rim_cells"), ","
                    ),
                    removed=_fmt(
                        (edge_erosion.get("inset_panel_despeckle") or {}).get(
                            "cells_removed"
                        ),
                        ",",
                    ),
                )
            )
    if addendum is not None:
        parts.append(addendum["text"])
    return "\n".join(parts)



class PlanOnlyWouldOverwriteARenderedManifest(RuntimeError):
    """``--plan-only`` was pointed at a build that already has rendered frames."""


def _refuse_to_overwrite_a_rendered_manifest(manifest_path):
    """Guard the manifest against a ``--plan-only`` run aimed at a real build.

    ``--plan-only`` renders nothing and then writes the manifest anyway, so
    aiming it at an existing build replaces the record of N rendered frames with
    one whose every frame carries no picture. The PNGs survive; the file that
    says what they are does not, and a sequence whose manifest says
    ``png=False`` everywhere is indistinguishable from one that was never
    rendered.

    Refuses rather than warns: by the time a warning is read the file is gone.
    Silent when there is no manifest, when it cannot be parsed, or when it
    describes no rendered frame -- those are all cases where nothing is lost.
    """
    if not manifest_path.exists():
        return
    try:
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return          # nothing legible to protect
    rendered = [
        f.get("epoch_ka")
        for f in (prev.get("frames") or ())
        if f.get("png") or (f.get("inset") or {}).get("png") or f.get("composed_png")
    ]
    if not rendered:
        return
    raise PlanOnlyWouldOverwriteARenderedManifest(
        "REFUSING to write a plan-only manifest over %s, which describes %d "
        "RENDERED frames (%s). --plan-only renders nothing, but it writes the "
        "manifest, so this would replace the record of those frames with one "
        "that says every frame has no picture -- and the pictures themselves "
        "would still be sitting beside it. Point --out-root at a scratch "
        "directory instead; this repo's own convention is a sibling named "
        "_planonly_scratch_<tag> (see renders/_ABORTED_2026-08-04_*/"
        "_planonly_scratch_rfinal). If you really mean to discard the record, "
        "move or delete the manifest yourself first, deliberately."
        % (manifest_path, len(rendered),
           " ".join("%g" % float(e) for e in rendered[:12])
           + (" ..." if len(rendered) > 12 else ""))
    )


def build_animation(
    *,
    out_root: Path | str = ANIM_ROOT,
    topo_dir: Path | str = KHV_TOPO_DIR,
    max_px: int = NATIVE_MAX_PX,
    duration_ms: int = FRAME_DURATION_MS,
    epochs: Optional[Sequence[float]] = None,
    grid_step_ka: float = FINE_STEP_KA,
    companion: Optional[str] = None,
    plan_only: bool = False,
    sill_tolerance_m: Optional[float] = None,
    fallback_canvas_px: Optional[tuple[int, int]] = None,
    verify_canonical_sha256: bool = False,
    corridor: str = "step",
    inset: Optional[InsetExtent] = None,
    fig_size_in: Optional[float] = None,
    dpi: Optional[float] = None,
    panels: str = "both",
) -> dict:
    """Plan, render, normalise, animate, and write the manifest.

    Returns the manifest dict.  The only exceptions that escape are the THREE the
    sequence must not survive: a GIF whose read-back delay is wrong
    (:class:`GifDurationError`), frames that are not one size
    (:class:`FrameSizeError`), and an animation that does not carry the one palette
    its frames were mapped through (:class:`PaletteError`).  Everything else becomes
    a refusal frame.

    ``verify_canonical_sha256`` re-hashes the canonical store to put a source path
    back on a raster read from a mirror that left no provenance sidecar.  It is off
    by default because it costs a full read of every raster -- 17 x 42 MB, which the
    SMB share measured at 0.4-3.4 MB/s -- and because the sidecar route needs no
    such read.  With it off and no sidecar, the manifest says ``UNRESOLVED`` and
    what would settle it; it never guesses from a file name.

    ``fig_size_in`` / ``dpi`` are the figure the renderer draws into.  ``None`` (the
    default) leaves ``RenderConfig``'s own values, which is what every sequence
    before 2026-08-05 was built with.  Raising ``fig_size_in`` is the ONLY lever this
    module has on the legend that sits inside the map -- see
    :data:`CLOSEUP_FIGURE_SIZE_IN` for the measurement and
    :func:`measure_legend_occlusion` for the guard that reports what it bought.

    ``panels`` is which panels reach the composed page: ``"both"`` (default, the
    whole scene above the inset) or ``"inset"`` (the CLOSE-UP build -- the corridor
    alone, with the whole-scene frame still rendered and hashed beside it; see
    :data:`CLOSEUP_COMPOSE_RULE`).  ``"inset"`` without an ``inset`` extent is
    refused rather than silently composing the full frame it was told not to show.
    """
    from paleoeurope.viz.paleocostline_render import SILL_TOLERANCE_M

    if corridor not in ("step", "ramp"):
        raise ValueError(
            f"corridor={corridor!r}; assign_basins implements 'step' and 'ramp' and "
            "this module passes the string through unchanged rather than mapping an "
            "unknown one onto a default."
        )
    if panels not in ("both", "inset"):
        raise ValueError(
            f"panels={panels!r}; this module composes 'both' (the whole scene above "
            "the inset) or 'inset' (the corridor alone). It does not map an unknown "
            "value onto a default, because the value decides what a reader SEES."
        )
    if panels == "inset" and inset is None:
        raise ValueError(
            "panels='inset' with inset=None: there is no inset to put on the page, "
            "and composing the whole-scene frame instead would hand back exactly the "
            "picture the caller asked not to be shown. Name a registered extent with "
            f"--inset ({[e.name for e in INSET_EXTENTS]!r}) or compose 'both'."
        )
    fig_size_in = (
        RenderConfig().fig_size_in if fig_size_in is None else float(fig_size_in)
    )
    # RenderConfig.dpi is declared `int`; a float here would put 150.0 in the
    # manifest's render_config beside a renderer that typed it 150, and the two
    # builds would then differ in a field neither run changed.
    dpi = RenderConfig().dpi if dpi is None else int(dpi)
    if fallback_canvas_px is None:
        fallback_canvas_px = fallback_canvas_for(fig_size_in, dpi)
    out_root = Path(out_root)
    frames_dir = out_root / "frames"
    full_dir = frames_dir if inset is None else frames_dir / "full"
    inset_dir = frames_dir / "inset"
    inset_raster_dir = out_root / "inset_rasters"
    composed_dir = out_root / "composed"
    canvas_dir = out_root / "canvas"
    log_path = out_root / "run.log"
    out_root.mkdir(parents=True, exist_ok=True)
    tolerance = SILL_TOLERANCE_M if sill_tolerance_m is None else float(sill_tolerance_m)

    _log("=" * 72, log_path)
    _log(f"khvalynian spill animation: out_root={out_root}", log_path)
    _log(f"max_px={max_px}  duration_ms={duration_ms}  topo_dir={topo_dir}", log_path)
    _log(
        f"corridor={corridor!r}  inset="
        + (f"{inset.name!r} {inset.bounds()}" if inset is not None else "None"),
        log_path,
    )
    _log(
        f"figure {fig_size_in:g} in x {dpi:g} dpi "
        f"(untrimmed {fallback_canvas_px[0]}x{fallback_canvas_px[1]} px)  "
        f"panels={panels!r}",
        log_path,
    )

    plans = plan_sequence(
        epochs,
        topo_dir=topo_dir,
        out_root=out_root,
        verify_canonical_sha256=verify_canonical_sha256,
    )
    unresolved = [
        p.epoch_ka for p in plans
        if (p.raster_source or {}).get("source_path") is None and p.raster_path
    ]
    if unresolved:
        _log(
            f"WARNING: {len(unresolved)} raster(s) have no canonical source path: "
            f"{unresolved}. The manifest records the path READ and says UNRESOLVED; "
            "it does not invent one.",
            log_path,
        )
    _log(
        f"planned {len(plans)} epochs: "
        f"{sum(1 for p in plans if p.renderable)} renderable, "
        f"{sum(1 for p in plans if not p.renderable)} refused at planning",
        log_path,
    )
    for plan in plans:
        if plan.renderable:
            spelled = "  ".join(
                f"{k}={v.level_m:+.4f}" for k, v in sorted(plan.levels.items())
            )
            _log(f"  {plan.epoch_ka:6.2f} ka  {spelled}", log_path)
        else:
            _log(f"  {plan.epoch_ka:6.2f} ka  REFUSED  {plan.refusal[:160]}", log_path)

    # ── the inset's own rasters, cut before the limits pass reads them ────────
    # Order is load-bearing: compute_sequence_limits opens every raster of every
    # SequenceEpoch, so the crops must exist first for the limits to be computed
    # OVER them.  A fingerprint appended to a limits object afterwards would name a
    # file that pass never read.
    inset_plans: list[InsetPlan] = []
    if inset is not None and not plan_only:
        for plan in plans:
            cut = plan_inset(plan, inset, inset_raster_dir)
            inset_plans.append(cut)
            _log(
                f"  inset {plan.epoch_ka:6.2f} ka  "
                + (
                    f"{Path(cut.raster_path).name} "
                    f"{cut.crop['window']['width']}x{cut.crop['window']['height']} "
                    "cells, "
                    + " ".join(
                        f"{s.basin}={s.elevation_m:+.2f}m@{s.lon:.4f}/{s.lat:.4f}"
                        for s in cut.seeds
                    )
                    if cut.renderable
                    else f"REFUSED  {cut.refusal[:180]}"
                ),
                log_path,
            )

    config = RenderConfig(
        max_px=int(max_px),
        ice_model_label="ICE-6G_C",
        measure_smoothing_sill_shift=True,
        fig_size_in=float(fig_size_in),
        dpi=dpi,
    )
    #: The same config WITHOUT the pinned limits.  ``measure_analytic_shade_excursion``
    #: must not consume them: it re-derives the raw intensity the limits are compared
    #: against, and handing it the limits would make the measurement depend on the
    #: thing it exists to check.
    base_config = config
    limits: Optional[SequenceLimits] = None
    extrema: Optional[dict] = None
    limits_full_only: Optional[SequenceLimits] = None
    if not plan_only:
        started = time.time()
        limits = sequence_limits_for(plans, config=config, inset_plans=inset_plans)
        if inset_plans and limits is not None:
            # What the inset raster COST the mapping, measured rather than argued.
            # It can only widen the percentile ramp (the aggregation is min/max over
            # rasters) and it cannot move the true envelope at all (the crop holds a
            # strict subset of the source's cells), but both are claims and both are
            # checked here on this run's own numbers.
            limits_full_only = sequence_limits_for(plans, config=config)
        if limits is not None:
            _log(
                f"sequence limits over {len(limits.epochs)} epochs in "
                f"{time.time() - started:.1f} s: land "
                f"{limits.land_min_m:+.3f}..{limits.land_max_m:+.3f} m, depth "
                f"0..{limits.depth_max_m:.3f} m, shade {limits.shade_min:.6f}.."
                f"{limits.shade_max:.6f}, envelope "
                f"{limits.dem_data_min_m:+.3f}..{limits.dem_data_max_m:+.3f} m",
                log_path,
            )
            config = dataclasses.replace(
                config, sequence_limits=limits, clip_policy=CLIP_POLICY
            )
            started = time.time()
            extrema = attribute_extrema(
                plans, limits, config=base_config, inset_plans=inset_plans
            )
            _log(
                f"extrema attributed in {time.time() - started:.1f} s: "
                + ", ".join(
                    f"{name}={extrema[name]['produced_by_epoch_ka']}"
                    for name in ("land_min_m", "land_max_m", "depth_max_m",
                                 "dem_data_min_m", "dem_data_max_m")
                ),
                log_path,
            )
        else:
            _log("no renderable epoch: no sequence limits computed", log_path)

    records: list[dict] = []
    if not plan_only:
        for index, plan in enumerate(plans, start=1):
            if not plan.renderable:
                _log(
                    f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka refused at "
                    f"planning; no render attempted",
                    log_path,
                )
                records.append(
                    {"epoch_ka": plan.epoch_ka, "png": None, "refusal": plan.refusal}
                )
                continue
            _log(f"[{index}/{len(plans)}] rendering {plan.epoch_ka:g} ka ...", log_path)
            record = render_one(
                plan, full_dir, config=config, sill_tolerance_m=tolerance,
                corridor=corridor,
            )
            records.append(record)
            _log(
                f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka "
                + (
                    f"OK {record['png']} ({record['wall_s']} s)"
                    if record["refusal"] is None
                    else f"REFUSED ({record['wall_s']} s) {record['refusal'][:200]}"
                ),
                log_path,
            )
            # The clip policy is a decision (see CLIP_POLICY), so the number it rests
            # on is measured HERE, on this epoch's own raster, in this run -- not
            # carried over from another epoch and not asserted in a docstring.
            if record.get("png"):
                record["shade_excursion"] = measure_analytic_shade_excursion(
                    plan.raster_path, config=base_config
                )
                excursion = record["shade_excursion"]
                _log(
                    f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka analytic shade "
                    f"ceiling: status={excursion['status']}"
                    + (
                        f", max raw {excursion['raw_intensity_max']!r}, "
                        f"{excursion['cells_above_shade_max']:,} of "
                        f"{excursion['mosaic_cells']:,} cells above +1.0, excursion "
                        f"{excursion['excursion_above_ceiling']:.6e} = "
                        f"{excursion['excursion_in_float32_eps']:.3f} float32 eps, "
                        f"floor clearance {excursion['clearance_below_floor']:.6f} "
                        f"({excursion['seconds']} s)"
                        if excursion["status"] == "MEASURED"
                        else f" -- {excursion.get('why', '')}"
                    ),
                    log_path,
                )

            # ── the inset, then the one canvas both panels go on ──────────────
            cut = next(
                (i for i in inset_plans if float(i.epoch_ka) == float(plan.epoch_ka)),
                None,
            )
            if cut is not None and record.get("png"):
                inset_record = render_inset(
                    cut, inset_dir, config=config, sill_tolerance_m=tolerance,
                    corridor=corridor,
                )
                record["inset"] = inset_record
                if inset_record.get("png"):
                    # The inset is a frame drawn through the pinned mapping, so it
                    # gets the same clip audit as the full frame -- including the
                    # excursion measurement, without which a marked ceiling cell
                    # would be refused as unmeasured rather than accepted as
                    # arithmetic.  It costs one mosaic over the crop.
                    inset_record["shade_excursion"] = (
                        measure_analytic_shade_excursion(
                            cut.raster_path, config=base_config
                        )
                    )
                _log(
                    f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka inset "
                    + (
                        f"OK {inset_record['png']} ({inset_record['wall_s']} s), "
                        f"colour mapping {inset_record['colour_mapping']['verdict']}, "
                        f"extent {inset_record['extent_check']['verdict']}"
                        if inset_record.get("png")
                        else f"REFUSED {str(inset_record.get('refusal'))[:220]}"
                    ),
                    log_path,
                )
            if record.get("png"):
                # The addendum is derived ONCE, here, and then used twice -- drawn on
                # the composed canvas and carried into the manifest -- so the picture
                # and the record cannot come to say different things.
                basins = record.get("basins_sidecar") or {}
                frame_like = {
                    "epoch_ka": plan.epoch_ka,
                    "basin_levels": {
                        k: v.to_dict() for k, v in plan.levels.items()
                    },
                    "sill": _sill_block(
                        plan, basins, tolerance_m=tolerance,
                        frame_refusal=None, frame_rendered=True,
                    ),
                    # The corridor as DRAWN, so the added sentence quotes the same
                    # fields the renderer's own disclosure quotes rather than a
                    # second reading of the same frame.  ``corridor`` and
                    # ``morphology`` are here because ramp_new_wet_ground needs
                    # BOTH -- whether the corridor was painted at all, and the
                    # de-speckle's closing, which bounds what a flat frame could
                    # have drawn of the same ground.
                    "spill": {
                        "corridor": basins.get("corridor"),
                        "connectivity": basins.get("connectivity"),
                        "painted_water": basins.get("painted_water"),
                        "morphology": basins.get("morphology"),
                    },
                }
                record["addendum"] = corridor_addendum(
                    frame_like,
                    corridor_mode=(basins.get("corridor") or {}).get("mode"),
                )
                if cut is None:
                    record["composed"] = record["png"]
                else:
                    inset_record = record.get("inset") or {}
                    # What the crop's own edges cost the lower panel's water,
                    # measured here so the caption a PNG-only reader sees carries it
                    # and the manifest carries the same object.
                    erosion = (
                        inset_edge_erosion(
                            basins, inset_record.get("basins_sidecar"), cut.crop
                        )
                        if inset_record.get("png")
                        else None
                    )
                    inset_record["edge_erosion"] = erosion
                    # What the legend costs the panel that is actually on the page.
                    # Measured on the saved PNG of THAT panel, so the caption and the
                    # manifest quote one object -- and measured whether it improved
                    # or not, because the figure was raised to change this number and
                    # a knob that reports only its successes is not a measurement.
                    occlusion = (
                        measure_legend_occlusion(
                            inset_record["png"], cut.extent.bounds()
                        )
                        if inset_record.get("png")
                        else None
                    )
                    inset_record["legend_occlusion"] = occlusion
                    if panels == "inset" and not inset_record.get("png"):
                        record["refusal"] = (
                            "CloseupPanelRefused: this build composes the corridor "
                            "panel ALONE (panels='inset'), and this epoch's corridor "
                            "panel refused, so the slot refuses with it rather than "
                            "falling back to the whole-scene frame the build exists "
                            "not to show. The whole-scene frame DID render and is on "
                            f"disk at {record.get('png')}. The corridor panel's own "
                            f"refusal: {inset_record.get('refusal')}"
                        )
                        _log(
                            f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka NOT "
                            "composed: the corridor panel refused and this build "
                            "shows only the corridor",
                            log_path,
                        )
                    else:
                        record["compose"] = compose_full_and_inset(
                            record["png"],
                            inset_record.get("png"),
                            composed_dir / f"composed_{plan.epoch_ka:g}ka.png",
                            caption=compose_caption(
                                plan.epoch_ka, record.get("addendum"), inset_record,
                                cut,
                                edge_erosion=erosion,
                                closeup=(panels == "inset"),
                                legend_occlusion=occlusion,
                            ),
                            drop_full=(panels == "inset"),
                        )
                        record["composed"] = record["compose"]["composed"]
                        _log(
                            f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka composed "
                            f"{record['compose']['size_px']} px from "
                            f"{len(record['compose']['panels'])} panel(s), "
                            f"inset_present={record['compose']['inset_present']}, "
                            "full_panel_omitted="
                            f"{record['compose']['full_panel_omitted'] is not None}",
                            log_path,
                        )
                        if occlusion and (occlusion.get("legend") or {}).get("found"):
                            legend = occlusion["legend"]
                            _log(
                                f"[{index}/{len(plans)}] {plan.epoch_ka:g} ka legend "
                                f"{legend['size_px'][0]}x{legend['size_px'][1]} px "
                                f"covers {legend['pct_of_map']:.2f} % of the "
                                f"{occlusion['map_px'][0]}x{occlusion['map_px'][1]} "
                                "px map and "
                                f"{legend['pct_of_water_under']:.2f} % of its water; "
                                f"{occlusion.get('px_per_deg_lon')} px/deg lon "
                                f"(fill ratio {legend['fill_ratio']})",
                                log_path,
                            )

    # ── what the marks mean, before anything is encoded ───────────────────────
    # Runs BEFORE the canvas and the palette, so a sequence with a real limits
    # breach in it costs a refusal rather than a GIF nobody should trust.
    clipping: dict[str, Any] = {}
    clipping_inset: dict[str, Any] = {}
    if not plan_only and any((r.get("inset") or {}).get("png") for r in records):
        # The SAME audit on the inset panels.  They are drawn through the same pinned
        # mapping onto the same page, so "the limits could not represent a cell" is
        # exactly as disqualifying there -- and an inset panel that was never audited
        # would be a half-checked picture wearing a fully-checked caption.
        clipping_inset = assert_only_the_analytic_shade_ceiling_was_marked(
            [r["inset"] for r in records if (r.get("inset") or {}).get("png")]
        )
        _log(
            f"inset clip policy {clipping_inset['policy']!r}: "
            f"{clipping_inset['total_cells_marked']} cell(s) marked at the analytic "
            "hillshade ceiling across the inset panels "
            f"({clipping_inset['cells_marked_at_the_analytic_ceiling_by_epoch']})",
            log_path,
        )
    if not plan_only and any(r.get("png") for r in records):
        clipping = assert_only_the_analytic_shade_ceiling_was_marked(records)
        _log(
            f"clip policy {clipping['policy']!r}: "
            f"{clipping['total_cells_marked']} cell(s) marked at the analytic "
            "hillshade ceiling across the sequence "
            f"({clipping['cells_marked_at_the_analytic_ceiling_by_epoch']}); every "
            "measured limit is zero on every frame",
            log_path,
        )

    # ── one canvas ───────────────────────────────────────────────────────────
    # The canvas is measured over the COMPOSED frames, which are the files that are
    # actually encoded.  With no inset the composed frame IS the render's own PNG,
    # so this is the behaviour that shipped; with one, padding the un-composed frame
    # would size the sequence to a picture nobody sees.
    rendered = [Path(r["composed"]) for r in records if r.get("composed")]
    rendered_epochs = [r["epoch_ka"] for r in records if r.get("composed")]
    sizes = [read_png_size(p) for p in rendered]
    canvas = canvas_size(sizes, fallback=fallback_canvas_px)
    canvas_from = (
        "the element-wise maximum over the "
        f"{len(rendered)} composed frame(s) this run actually produced, epochs "
        f"{[float(e) for e in rendered_epochs]}, whose own sizes were "
        f"{sorted({tuple(s) for s in sizes})} px"
        if rendered
        else (
            "declared fallback: NO frame rendered, so there is no measured size. "
            f"{fallback_canvas_px[0]}x{fallback_canvas_px[1]} is fig_size_in x dpi, "
            "the figure before bbox_inches='tight' trims it, i.e. an upper bound on "
            "every frame this config can produce -- it is NOT a measurement of this "
            "sequence"
        )
    )
    _log(f"canvas {canvas[0]}x{canvas[1]} px from {canvas_from}", log_path)

    frame_files: list[Path] = []
    padding: list[dict] = []
    if not plan_only:
        for index, (plan, record) in enumerate(zip(plans, records)):
            stem = f"frame_{index:02d}_{plan.epoch_ka:g}ka.png"
            target = canvas_dir / stem
            if record.get("composed"):
                padding.append(pad_to_canvas(record["composed"], target, canvas))
            else:
                refusal = record.get("refusal") or plan.refusal or "unknown refusal"
                refusal_frame(
                    plan.epoch_ka, refusal, target, canvas, levels=plan.levels or None
                )
                padding.append(
                    {
                        "source": None,
                        "canvas": str(target.resolve()),
                        "source_size_px": None,
                        "canvas_size_px": [canvas[0], canvas[1]],
                        "pad_left_px": 0, "pad_top_px": 0,
                        "pad_right_px": 0, "pad_bottom_px": 0,
                        "note": "hatched refusal placeholder drawn AT the canvas size",
                    }
                )
            # The fingerprint of the file that is actually encoded.  It is what lets
            # write_subsequence re-encode these frames instead of re-rendering them
            # and still PROVE, rather than assume, that the two animations show the
            # same pictures.
            padding[-1]["canvas_sha256"] = sha256_of(target)
            frame_files.append(target)

    animation: dict[str, Any] = {}
    master: Optional[MasterPalette] = None
    palette_report: dict[str, Any] = {}
    if frame_files:
        uniform = assert_uniform_size(frame_files)
        _log(f"all {len(frame_files)} frames are {uniform[0]}x{uniform[1]} px", log_path)

        started = time.time()
        master = build_master_palette(frame_files)
        _log(
            f"master palette in {time.time() - started:.1f} s: "
            f"{master.n_from_quantizer} of {master.size} entries from "
            f"{master.method}, {master.n_filler} filler, over "
            f"{master.n_distinct_in_sequence:,} distinct colours in the sequence "
            f"({master.n_distinct_in_sequence / max(master.n_from_quantizer, 1):.1f} "
            "colours per entry)",
            log_path,
        )
        gif = write_gif(frame_files, out_root / "khvalynian_spill.gif",
                        duration_ms=duration_ms, master=master)
        durations = assert_frame_durations_ms(gif, duration_ms, len(frame_files))
        gif_palette_audit = assert_one_global_palette(gif, master)
        gif_pixel_audit = assert_gif_frames_are_the_mapped_frames(
            gif, master, frame_files
        )
        _log(
            f"gif palette audit: {gif_palette_audit['n_image_descriptors']} image "
            f"descriptors, "
            f"{gif_palette_audit['n_frames_with_a_local_color_table']} local colour "
            f"tables, global table sha256 "
            f"{gif_palette_audit['global_color_table_sha256'][:16]}...; "
            f"{gif_pixel_audit['n_frames_decoded_and_compared']} frames decoded and "
            "identical to the mapped frames",
            log_path,
        )

        # The WebP is written from the UNQUANTIZED canvas frames -- it is the 24-bit
        # escape the headroom note names, and the note is now handed the audit of
        # this very file instead of asserting one exists.  Both files' delays are
        # asserted; until 2026-08-04 the WebP's were not, because the reader never
        # called load() and so measured None where the file said 250.
        webp = write_webp(frame_files, out_root / "khvalynian_spill.webp",
                          duration_ms=duration_ms)
        webp_durations, webp_delay_asserted, webp_delay_note = check_webp_durations(
            webp, duration_ms, len(frame_files)
        )
        webp_audit = assert_webp_is_a_lossless_24bit_copy(
            webp, frame_files, master=master
        )

        started = time.time()
        cost = measure_palette_cost(frame_files, master, webp_audit=webp_audit)
        _log(
            f"palette cost in {time.time() - started:.1f} s: shared mean "
            f"{cost['shared_palette']['mean_of_frame_means']:.4f} / max "
            f"{cost['shared_palette']['max_per_pixel']:.2f} against per-frame "
            f"adaptive mean "
            f"{cost['per_frame_adaptive']['mean_of_frame_means']:.4f} / max "
            f"{cost['per_frame_adaptive']['max_per_pixel']:.2f} "
            "(RGB Euclidean per pixel)",
            log_path,
        )
        # The quantizer decision is re-measured on THIS run's own frames, in every
        # run, so master_palette.json carries the arms it was taken from rather than
        # a constant to be trusted.  Cost is logged beside it.
        started = time.time()
        quantizers = compare_quantizers(frame_files)
        _log(
            f"quantizer comparison in {time.time() - started:.1f} s over "
            f"{len(frame_files)} frame(s) ("
            f"{len(rendered)} map frame(s) + "
            f"{len(frame_files) - len(rendered)} refusal placeholder(s)): "
            + "; ".join(
                f"{name} mean {arm['mean_of_frame_means']:.4f} / max "
                f"{arm['max_per_pixel']:.3f}"
                for name, arm in quantizers["arms"].items()
                if arm.get("status") == "MEASURED"
            )
            + f" -- rule selects {quantizers['decision']['selected_by_the_rule']}, "
            f"shipped {quantizers['shipped']}, "
            f"{quantizers['decision']['verdict']}",
            log_path,
        )

        palette_report = {
            "master": master.to_dict(),
            "cost": cost,
            "quantizer_comparison": quantizers,
            "headroom": cost["headroom"],
            "gif_audit": gif_palette_audit,
            "gif_pixel_audit": gif_pixel_audit,
            "webp_audit": webp_audit,
        }
        animation = {
            "gif": path_record(gif),
            "gif_bytes": gif.stat().st_size,
            "gif_frame_durations_ms_read_back_with_pil": durations,
            "gif_palette_verdict": gif_palette_audit["verdict"],
            "webp": path_record(webp),
            "webp_bytes": webp.stat().st_size,
            "webp_frame_durations_ms_read_back_with_pil": webp_durations,
            "webp_delay_asserted": webp_delay_asserted,
            "webp_delay_note": webp_delay_note,
            "webp_verdict": webp_audit["verdict"],
            "webp_is_24_bit_note": (
                "the WebP is written from the UNQUANTIZED canvas frames, LOSSLESS, "
                "and every frame of it was decoded and compared to its source PNG "
                "bit for bit -- see palette.webp_audit for the distinct-colour count "
                "and whether it exceeds the GIF's 256-entry ceiling. It is NOT the "
                "palette-mapped copy it was until 2026-08-04, which carried 222 "
                "distinct colours while the same file's headroom note offered it as "
                "the escape FROM 256"
            ),
            "frame_duration_ms": int(duration_ms),
            "unit_note": (
                "MILLISECONDS, in BOTH files, read back and asserted in both. On "
                "imageio 2.37.2 a seconds-shaped duration such as 0.25 writes delays "
                "PIL reads back as [None, 0, 0] -- silently, in a valid file. The "
                "WebP delays are asserted from 2026-08-04: they always survived the "
                "write, but the reader did not call load() and so measured None"
            ),
            "frame_size_px": list(uniform),
        }
        _log(
            f"gif {animation['gif_bytes']:,} B, webp {animation['webp_bytes']:,} B "
            f"({webp_audit['n_distinct_colours_decoded']:,} distinct colours "
            f"decoded, escapes_the_gif_ceiling="
            f"{webp_audit['escapes_the_gif_ceiling']}), delays gif "
            f"{durations[:3]}... webp {webp_durations[:3]}...",
            log_path,
        )

    manifest = build_manifest(
        plans, records, padding, limits, animation,
        config=config, canvas=canvas, canvas_from=canvas_from,
        duration_ms=duration_ms, sill_tolerance_m=tolerance, extrema=extrema,
        palette=palette_report, grid_step_ka=grid_step_ka, companion=companion,
        clipping=clipping, corridor=corridor, inset=inset, inset_plans=inset_plans,
        limits_full_only=limits_full_only, clipping_inset=clipping_inset,
        panels=panels,
    )
    manifest_path = out_root / "sequence_manifest.json"
    if plan_only:
        _refuse_to_overwrite_a_rendered_manifest(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"manifest {manifest_path}", log_path)
    if corridor == "ramp":
        # AFTER the manifest is on disk, so the refusal leaves its evidence behind.
        # This is the second of two independent guards and neither replaces the
        # other: assign_basins raises CorridorNotRampableError at render time and
        # that arrives as a refused frame, while this one re-derives the same
        # question from the saddles and the levels and refuses the SEQUENCE.
        assert_ramp_is_legal(manifest["ramp_legality"])

    # The pinned mapping also stands on its own, next to the animation, because the
    # phase acceptance asks for a file that names WHICH epoch produced each
    # extremum and a reader chasing a colour should not have to walk a 17-frame
    # manifest to find it.  Same objects, no second derivation.
    if limits is not None:
        (out_root / "sequence_limits.json").write_text(
            json.dumps(
                {
                    "limits": limits.to_dict(),
                    "covers": manifest["sequence_limits_cover"],
                    "extrema_by_epoch": extrema,
                    "consumed_with": {
                        "clip_policy": config.clip_policy,
                        "verify_sequence_raster_hashes": (
                            config.verify_sequence_raster_hashes
                        ),
                        "max_px": config.max_px,
                    },
                    "clipping": clipping or None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _log(
            f"sequence limits {out_root / 'sequence_limits.json'} -- computed over "
            f"{len(limits.epochs)} epoch(s) of a {len(plans)}-slot sequence; "
            f"{manifest['sequence_limits_cover']['verdict']}",
            log_path,
        )

    # The palette stands on its own beside the animation for the same reason the
    # limits do: a reader chasing a colour should be able to open ONE file and see
    # the 256 entries, what they cost, and the audit of the written bytes -- without
    # walking a 17-frame manifest.  Same objects, no second derivation.
    if palette_report:
        (out_root / "master_palette.json").write_text(
            json.dumps(palette_report, indent=2), encoding="utf-8"
        )
        _log(f"master palette {out_root / 'master_palette.json'}", log_path)
    return manifest


#: What the deliverable directory does NOT carry after :func:`_copy_frames_into`,
#: named so a reader who copies it knows exactly what stayed behind and where.
EXTERNAL_DEPENDENCIES_NOTE = (
    "this directory is self-contained FOR THE ANIMATION: canvas/ holds the exact "
    "PNGs the GIF and the WebP were encoded from, each re-hashed against the sha256 "
    "the source run recorded, so copying this directory alone loses no source image. "
    "What it deliberately does NOT copy is the SOURCE RUN's frames/ tree -- the "
    "renderer's own paleo_render_<epoch>ka.png plus its .basins.json and "
    ".palette.json sidecars, which are the pre-padding output and the per-frame "
    "audit trail. Those live at 'derived_from.source_run/frames', they are ~2.7 MB "
    "per frame against the 2.5 MB canvas PNG already copied here, and every FIELD "
    "this manifest needs out of them is carried inline in frames[]: the sill block, "
    "the spill and connectivity report, the clipping counts and the "
    "analytic-shade-ceiling measurement. The rasters are not copied either and are "
    "not meant to be -- they are 42 MB each and are identified by sha256 plus a "
    "canonical path in frames[].raster. THE SAME APPLIES TWICE OVER TO AN INSET "
    "RUN, and is named here rather than discovered: composed/ (the two panels on "
    "one page, which IS what canvas/ was padded from and is therefore already "
    "represented byte for byte), frames/inset/ (the inset render and its own two "
    "sidecars) and inset_rasters/ (the crops) all stay in the source run. Every "
    "field the manifest needs out of them is inline in frames[].inset -- the crop "
    "window and both sha256s, the resolved seeds, the inset's own measured saddle "
    "and its verdict -- and the crop is reproducible from the source raster and "
    "the named extent by crop_to_extent alone"
)


def _copy_frames_into(
    frame_files: Sequence[Path],
    canvas_dir: Path,
    frames: Sequence[dict],
    *,
    source_root: Path,
) -> tuple[list[Path], dict]:
    """Copy the verified canvas PNGs into the deliverable and re-verify each copy.

    THE DEFECT THIS CLOSES
    ----------------------
    ``spill_0.5ka_9_epochs`` shipped a GIF, a WebP, a manifest and a palette, and
    **no frames of its own**: every ``canvas_frame`` in it pointed into
    ``spill_0.25ka_17_slots/canvas``.  Copying the deliverable directory alone lost
    the source images, and the manifest said nothing about the dependency.

    COST, MEASURED BEFORE IT WAS SPENT
    ----------------------------------
    Nine canvas PNGs at 2964x1250 px are **22.2 MB** together (2.45-2.49 MB each);
    the volume they are written to had **4.8 TB free**.  Self-contained is therefore
    the answer rather than a README pointing elsewhere, and the actual byte count is
    recorded per run in the block this returns.

    The copy is re-hashed and compared against the sha256 the source run recorded
    for the original -- not against the copy's own freshly-computed digest, which
    would only prove the file was read back.  A mismatch refuses: an animation
    encoded from bytes nobody can attribute is the artifact this project keeps
    having to retract.
    """
    import shutil

    canvas_dir.mkdir(parents=True, exist_ok=True)
    local: list[Path] = []
    copied: list[dict] = []
    total = 0
    for source_file, entry in zip(frame_files, frames):
        destination = canvas_dir / source_file.name
        shutil.copy2(source_file, destination)
        digest = sha256_of(destination)
        recorded = entry.get("canvas_frame_sha256")
        if digest != recorded:
            raise SubsequenceSourceError(
                f"the copy of {source_file} at {destination} hashes to {digest}, "
                f"but the source run recorded {recorded} for it. The copy is not the "
                "frame that was measured and it is NOT encoded."
            )
        size = destination.stat().st_size
        total += size
        entry["canvas_frame"] = str(destination)
        entry["canvas_frame_spellings"] = path_record(destination)
        entry["canvas_frame_sha256_reverified"] = digest
        copied.append(
            {
                "epoch_ka": entry.get("epoch_ka"),
                "from": entry.get("canvas_frame_in_the_source_run"),
                "to": path_record(destination),
                "bytes": size,
                "sha256": digest,
            }
        )
        local.append(destination)
    return local, {
        "verdict": "SELF_CONTAINED_FOR_THE_ANIMATION",
        "canvas_dir": path_record(canvas_dir),
        "files_copied": len(copied),
        "bytes_copied": total,
        "mib_copied": round(total / (1024 * 1024), 2),
        "copied": copied,
        "rule": (
            "each copy is re-hashed and compared against the sha256 the SOURCE RUN "
            "recorded for the original -- not against a digest computed from the "
            "copy itself, which would only prove the file could be read back"
        ),
        "what_is_still_external": EXTERNAL_DEPENDENCIES_NOTE,
        "source_run": path_record(source_root),
    }


def write_subsequence(
    source_root: Path | str,
    out_root: Path | str,
    *,
    grid_step_ka: float = HALF_KA_STEP,
    window_ka: tuple[float, float] = FINE_WINDOW_KA,
    duration_ms: int = FRAME_DURATION_MS,
    keep: Optional[Sequence[float]] = None,
) -> dict:
    """The second grid, re-encoded from the first run's frames -- never re-rendered.

    WHY NOT JUST RUN IT AGAIN
    -------------------------
    The two deliverables differ in exactly one thing: whether the eight epochs with
    no Pontic curve row occupy a slot.  They do NOT differ in imagery, because
    ``sequence_limits_for`` measures over the *renderable* plans and those are the
    same nine epochs either way, and ``render_one`` is independent of the other
    slots.  Rendering nine native frames a second time would therefore cost about
    forty minutes to produce bytes that must be identical -- and would leave "must
    be identical" as an argument.  Re-encoding the first run's own canvas frames
    makes it a construction, and the sha256 check below makes it a measurement.

    WHAT IS RE-DERIVED AND WHAT IS CARRIED
    --------------------------------------
    * **Re-derived:** the master palette, and every number that hangs off it.  A
      palette quantized over nine map frames is not a palette quantized over nine map
      frames and eight hatched red-and-white placeholders -- the placeholders spend
      colour-table entries on hatching, so the deliverable gets its own table over
      its own frames, and its own written-bytes audit.
    * **Carried, unchanged, from the source manifest:** every per-frame block --
      levels, seeds, raster fingerprints and source paths, sill expectation /
      measured saddle / verdict, connectivity, clipping.  Those describe the frame,
      and the frame is the same file.  Re-deriving them would be a second answer to a
      question already answered, and two answers can disagree.

    REFUSALS
    --------
    Nothing is re-encoded that cannot be proved to be the source run's own frame:
    :class:`SubsequenceSourceError` for a missing manifest, an epoch the source did
    not carry, a slot that never rendered, a canvas file that is gone, or a canvas
    file whose sha256 is not the one the source recorded.
    """
    source_root = Path(source_root)
    out_root = Path(out_root)
    manifest_path = source_root / "sequence_manifest.json"
    log_path = out_root / "run.log"
    out_root.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise SubsequenceSourceError(
            f"{manifest_path} does not exist. A sub-sequence is re-encoded from a "
            "finished run's own frames and its own record of them; without the "
            "manifest there is nothing to verify the frames against, and encoding "
            "them unverified is the defect this function exists to avoid."
        )
    try:
        same_root = source_root.resolve() == out_root.resolve()
    except Exception:  # noqa: BLE001
        same_root = str(source_root) == str(out_root)
    if same_root:
        raise SubsequenceSourceError(
            f"--subsequence-of and --out-root are the same directory ({source_root}). "
            "The sub-sequence copies the source run's canvas frames into its own "
            "canvas/ so the deliverable can be moved on its own; pointed at itself it "
            "would copy each frame over itself and then overwrite the manifest it is "
            "reading from. Give the deliverable its own directory."
        )
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha = sha256_of(manifest_path)

    _log("=" * 72, log_path)
    _log(f"sub-sequence at {grid_step_ka:g} ka from {source_root}", log_path)

    wanted = (
        [float(e) for e in keep]
        if keep is not None
        else list(epoch_grid(window_ka, grid_step_ka))
    )
    by_epoch = {float(f["epoch_ka"]): f for f in source.get("frames", [])}
    missing = [e for e in wanted if e not in by_epoch]
    if missing:
        raise SubsequenceSourceError(
            f"the source run carries no slot for {missing} (it has "
            f"{sorted(by_epoch)}). A sub-sequence selects, it does not render."
        )

    frames: list[dict] = []
    frame_files: list[Path] = []
    for slot, epoch in enumerate(wanted):
        entry = dict(by_epoch[epoch])
        canvas_path = entry.get("canvas_frame")
        recorded = entry.get("canvas_frame_sha256")
        if entry.get("refusal") or not entry.get("png"):
            raise SubsequenceSourceError(
                f"{epoch:g} ka produced no frame in the source run "
                f"({entry.get('refusal')!r}). A sub-sequence whose whole point is "
                "that every frame is imagery cannot carry a placeholder; if this "
                "epoch belongs in the sequence, the sequence is the other grid."
            )
        if not canvas_path or not Path(canvas_path).exists():
            raise SubsequenceSourceError(
                f"{epoch:g} ka names canvas frame {canvas_path!r}, which does not "
                "exist. The bytes this would re-encode are gone."
            )
        actual = sha256_of(canvas_path)
        if not recorded:
            raise SubsequenceSourceError(
                f"{epoch:g} ka carries no canvas_frame_sha256 in the source "
                "manifest, so there is nothing to check the file against. Re-run the "
                "source; a frame that cannot be verified is not re-encoded."
            )
        if actual != recorded:
            raise SubsequenceSourceError(
                f"{epoch:g} ka: {canvas_path} hashes to {actual}, but the source run "
                f"recorded {recorded}. The file has changed since it was measured "
                "and it is NOT re-encoded -- an animation assembled from frames "
                "nobody can attribute is exactly the artifact this project keeps "
                "having to retract."
            )
        entry["slot"] = slot
        entry["canvas_frame_sha256_reverified"] = actual
        entry["canvas_frame_in_the_source_run"] = path_record(canvas_path)
        frames.append(entry)
        frame_files.append(Path(canvas_path))

    uniform = assert_uniform_size(frame_files)
    _log(
        f"{len(frame_files)} frames re-verified by sha256 and all "
        f"{uniform[0]}x{uniform[1]} px",
        log_path,
    )

    # ── the deliverable keeps its own copy of the images it is made of ────────
    frame_files, self_contained = _copy_frames_into(
        frame_files, out_root / "canvas", frames, source_root=source_root
    )
    _log(
        f"self-contained: {self_contained['files_copied']} canvas PNG(s), "
        f"{self_contained['bytes_copied']:,} B copied into {out_root / 'canvas'}; "
        f"every copy re-hashed and equal to the source run's recorded sha256",
        log_path,
    )

    master = build_master_palette(frame_files)
    gif = write_gif(
        frame_files, out_root / "khvalynian_spill.gif",
        duration_ms=duration_ms, master=master,
    )
    durations = assert_frame_durations_ms(gif, duration_ms, len(frame_files))
    gif_palette_audit = assert_one_global_palette(gif, master)
    gif_pixel_audit = assert_gif_frames_are_the_mapped_frames(gif, master, frame_files)
    webp = write_webp(
        frame_files, out_root / "khvalynian_spill.webp", duration_ms=duration_ms,
    )
    webp_durations, webp_delay_asserted, webp_delay_note = check_webp_durations(
        webp, duration_ms, len(frame_files)
    )
    webp_audit = assert_webp_is_a_lossless_24bit_copy(
        webp, frame_files, master=master
    )
    cost = measure_palette_cost(frame_files, master, webp_audit=webp_audit)
    # The MEDIANCUT/MAXCOVERAGE trade is re-measured on the frames of THIS sequence
    # -- all of them real map frames here -- and the rule that decides it is applied
    # to those arms in palette_method_decision rather than restated from the
    # constant.
    quantizers = compare_quantizers(frame_files)
    _log(
        "quantizer comparison on the real frames: "
        + "; ".join(
            f"{name} mean {arm['mean_of_frame_means']:.4f} / max "
            f"{arm['max_per_pixel']:.3f}"
            for name, arm in quantizers["arms"].items()
            if arm.get("status") == "MEASURED"
        )
        + f" -- rule selects {quantizers['decision']['selected_by_the_rule']}, "
        f"shipped {quantizers['shipped']}, {quantizers['decision']['verdict']}",
        log_path,
    )
    _log(
        f"gif {gif.stat().st_size:,} B ({gif_palette_audit['verdict']}, "
        f"{gif_pixel_audit['n_frames_decoded_and_compared']} frames decoded and "
        f"identical), webp {webp.stat().st_size:,} B ({webp_audit['verdict']}, "
        f"{webp_audit['n_distinct_colours_decoded']:,} distinct colours, "
        f"escapes_the_gif_ceiling={webp_audit['escapes_the_gif_ceiling']}), delays "
        f"gif {durations[:3]}... webp {webp_durations[:3]}...",
        log_path,
    )

    palette_report = {
        "master": master.to_dict(),
        "cost": cost,
        "quantizer_comparison": quantizers,
        "headroom": cost["headroom"],
        "gif_audit": gif_palette_audit,
        "gif_pixel_audit": gif_pixel_audit,
        "webp_audit": webp_audit,
    }
    manifest = dict(source)
    manifest["grid"] = grid_description(
        wanted, step_ka=grid_step_ka, window_ka=window_ka,
        companion=str((source_root / "sequence_manifest.json").resolve()),
        n_rendered=len(wanted),
    )
    manifest["what"] = (
        "The Khvalynian (Caspian) transgression spilling west through the Manych "
        f"into the Pontic, {max(wanted):.2f}-{min(wanted):.2f} ka, on the "
        f"{grid_step_ka:g} ka grid: {len(wanted)} epochs, EVERY ONE OF THEM REAL "
        "IMAGERY. See 'grid' for the eight quarter-ka epochs this sequence does not "
        "carry and 'derived_from' for the run whose frames these are."
    )
    manifest["generated_by"] = (
        "scripts/animate_khvalynian_spill.py::write_subsequence"
    )
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["epochs"] = wanted
    manifest["n_frames"] = len(wanted)
    manifest["n_rendered"] = len(wanted)
    manifest["n_refused"] = 0
    manifest["frames"] = frames
    manifest["palette"] = palette_report
    manifest["sill_verdicts"] = _sill_verdict_summary(frames)
    # Two blocks of the source manifest are SUMMARIES OVER ITS SLOTS, so carrying
    # them verbatim would put "17" inside a 9-frame file and invite exactly the
    # misreading this whole pair of deliverables exists to prevent.  They are
    # recounted over the frames THIS sequence carries, from those frames' own
    # records -- no raster is re-hashed and no limit is recomputed.
    statuses: dict[str, int] = {}
    no_source: list[float] = []
    for frame in frames:
        raster = frame.get("raster") or {}
        status = (raster.get("source_resolution") or {}).get("status", "UNRESOLVED")
        statuses[status] = statuses.get(status, 0) + 1
        if not raster.get("source_path"):
            no_source.append(frame["epoch_ka"])
    manifest["raster_source_resolution"] = {
        **(source.get("raster_source_resolution") or {}),
        "by_status": statuses,
        "epochs_with_no_source_path": no_source,
        "counted_over": (
            f"the {len(frames)} epochs of THIS sequence, recounted from their own "
            "per-frame records. The source run's tally covered "
            f"{(source.get('grid') or {}).get('n_slots')} slots and is not repeated "
            "here, because a count of 17 inside a 9-frame manifest is a number that "
            "describes a different file"
        ),
    }
    manifest["sequence_limits_cover"] = {
        **(source.get("sequence_limits_cover") or {}),
        "n_slots_in_the_sequence": len(frames),
        "slots": [f["epoch_ka"] for f in frames],
        "slots_carrying_no_frame": [],
        "note_for_this_subsequence": (
            f"every one of these {len(frames)} slots carries a frame. The limits "
            "object above is the SOURCE run's, unchanged and un-recomputed, because "
            "it was computed over exactly these epochs -- the source's other slots "
            "refused at planning and contributed nothing to it. "
            "'epochs_the_limits_were_computed_over' and "
            "'epochs_that_produced_a_frame' are the source run's and are the same "
            "list; 'n_slots_in_the_sequence' and 'slots' have been recounted for "
            "this file"
        ),
    }
    manifest["animation"] = {
        "gif": path_record(gif),
        "gif_bytes": gif.stat().st_size,
        "gif_frame_durations_ms_read_back_with_pil": durations,
        "gif_palette_verdict": gif_palette_audit["verdict"],
        "webp": path_record(webp),
        "webp_bytes": webp.stat().st_size,
        "webp_frame_durations_ms_read_back_with_pil": webp_durations,
        "webp_delay_asserted": webp_delay_asserted,
        "webp_delay_note": webp_delay_note,
        "webp_verdict": webp_audit["verdict"],
        "webp_is_24_bit_note": (
            "the WebP is written from the UNQUANTIZED canvas frames, LOSSLESS, and "
            "every frame of it was decoded and compared to its source PNG bit for "
            "bit -- see palette.webp_audit for the distinct-colour count and whether "
            "it exceeds the GIF's 256-entry ceiling"
        ),
        "frame_duration_ms": int(duration_ms),
        "unit_note": (
            "MILLISECONDS, in BOTH files, read back with PIL and asserted in both"
        ),
        "frame_size_px": list(uniform),
    }
    manifest["self_contained"] = self_contained
    manifest["derived_from"] = {
        "source_run": path_record(source_root),
        "source_manifest": path_record(manifest_path),
        "source_manifest_sha256": source_sha,
        "source_grid_step_ka": (source.get("grid") or {}).get("step_ka"),
        "source_slots": (source.get("grid") or {}).get("n_slots"),
        "what_was_re_encoded": [
            {
                "epoch_ka": f["epoch_ka"],
                "canvas_frame_in_the_source_run": f["canvas_frame_in_the_source_run"],
                "canvas_frame_copied_here": f["canvas_frame_spellings"],
                "sha256_recorded_by_the_source_run": f["canvas_frame_sha256"],
                "sha256_reverified_here": f["canvas_frame_sha256_reverified"],
            }
            for f in frames
        ],
        "rule": (
            "no frame was rendered for this sequence. Each canvas PNG named above "
            "was re-hashed and matched the sha256 the source run recorded for it, "
            "then COPIED into this directory's own canvas/ and re-hashed again "
            "against that same recorded value before it was encoded -- so the two "
            "animations show the same pictures as a matter of the same bytes rather "
            "than of a claim, and this directory carries those bytes itself. What IS "
            "re-derived here is the master palette -- a table quantized over nine map "
            "frames is not a table quantized over nine map frames plus eight hatched "
            "placeholders -- together with its cost, its quantizer decision and its "
            "written-bytes audits"
        ),
        "what_is_carried_unchanged": (
            "every per-frame block: basin levels and their rules, seeds, raster "
            "sha256 and canonical source path, the sill expectation / measured "
            "saddle / verdict per pair, the spill and connectivity report, the "
            "clipping counts and the analytic-shade-ceiling measurement. They "
            "describe the frame, and the frame is the same file; re-deriving them "
            "would be a second answer to an answered question. level_curves is "
            "carried for a stronger reason: the frames were drawn from the CSVs as "
            "the source run read them, and re-hashing those files now could report a "
            "different fingerprint than the pictures were made with"
        ),
    }
    _apply_computed_blocks(manifest, frames)
    (out_root / "sequence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_root / "master_palette.json").write_text(
        json.dumps(palette_report, indent=2), encoding="utf-8"
    )
    if manifest.get("sequence_limits") is not None:
        (out_root / "sequence_limits.json").write_text(
            json.dumps(
                {
                    "limits": manifest["sequence_limits"],
                    "covers": manifest["sequence_limits_cover"],
                    "extrema_by_epoch": manifest.get(
                        "sequence_limits_extrema_by_epoch"
                    ),
                    "clipping": manifest.get("analytic_shade_ceiling"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _log(f"manifest {out_root / 'sequence_manifest.json'}", log_path)
    return manifest


#: What the ``corridor`` field is, and what it is NOT an answer to.  Every NUMBER
#: that used to live in this string is now computed per run by :func:`corridor_note`,
#: which is why this constant carries none: the shipped sentence quoted a single
#: epoch's saddle (+13.345 m, 16.5 ka) as if it held across the window.
#:
#: The nine rendered frames of the 0.5 ka deliverable in fact measured, IN EPOCH
#: ORDER from 17.0 to 13.0 ka, +13.476 / +13.345 / +13.325 / +13.528 / +14.099 /
#: +14.092 / +16.064 / +15.363 / +14.919 m -- a series that rises, falls, rises and
#: falls again.  Its SMALLEST value is at 16.0 ka and its LARGEST at 14.0 ka, so
#: "13.325 to 16.064" is the ASCENDING SORT's endpoints and not a run through time;
#: it was written here as though it were, which is why ``corridor_note`` now ships
#: the series by epoch beside the sorted pair and names each for what it is.
CORRIDOR_RULE = (
    "The brief asked whether TWO pairs are donor->receiver at once -- the case "
    "assign_basins refuses in 'ramp' mode with CorridorNotRampableError. The answer "
    "is measured per epoch below, on two independent readings: the "
    "pontic<->mediterranean saddle this grid actually measured on dem_sm, and the "
    "independently established Bosphorus F0 path crest converted at that cell. The "
    "corridor mode this run REQUESTED is the 'corridor' field; it is the caller's "
    "and this module never falls back from one to the other. 'step' marks the "
    "corridor and draws every basin flat at its own level; 'ramp' interpolates a "
    "surface across the one corridor and draws it, and what that interpolation is "
    "and is not is stated per frame in ramp_addendum"
)


def _is_monotone(values: Sequence[float]) -> bool:
    """Whether a series never changes direction -- so a sorted pair may stand for it.

    The one question that decides whether a ``[min, max]`` pair can be read as "it
    ran from here to there".  On the nine shipped 0.5 ka frames the answer is False
    for the ``mediterranean<->pontic`` saddle, which is how a sorted pair came to be
    written as a run through time.
    """
    series = [float(v) for v in values]
    return series == sorted(series) or series == sorted(series, reverse=True)


def corridor_note(frames: Sequence[dict], corridor: str = "step") -> dict:
    """Whether two pairs are ever donor->receiver at once -- computed, per epoch.

    Reads only this run's own frame records: the Pontic surface, ``dz_bosphorus_m``,
    and the ``mediterranean|pontic`` saddle the render measured.  Frames that did not
    render contribute their level and their crest and are marked as carrying no
    measured saddle, rather than being dropped -- a summary taken over the frames
    that happened to succeed is a summary of a different sequence.
    """
    rows: list[dict] = []
    for frame in frames:
        levels = frame.get("basin_levels") or {}
        pontic = (levels.get("pontic") or {}).get("level_m")
        dz = frame.get("dz_bosphorus_m")
        entry = next(
            (
                s for s in (frame.get("sill") or [])
                if s.get("pair") == ["mediterranean", "pontic"]
            ),
            {},
        )
        saddle = entry.get("measured_saddle_m")
        crest = None if dz is None else BOSPHORUS_CREST_F0_M + float(dz)
        rows.append(
            {
                "epoch_ka": frame.get("epoch_ka"),
                "pontic_level_m": pontic,
                "measured_saddle_m": saddle,
                "pontic_below_the_measured_saddle": (
                    None if pontic is None or saddle is None else pontic < saddle
                ),
                "bosphorus_crest_f1_m": crest,
                "pontic_below_the_independent_crest": (
                    None if pontic is None or crest is None else pontic < crest
                ),
                "headroom_to_the_independent_crest_m": (
                    None if pontic is None or crest is None else crest - pontic
                ),
            }
        )
    measured = [r for r in rows if r["pontic_below_the_measured_saddle"] is not None]
    independent = [
        r for r in rows if r["pontic_below_the_independent_crest"] is not None
    ]
    headrooms = [
        r for r in independent if r["headroom_to_the_independent_crest_m"] is not None
    ]
    narrowest = (
        min(headrooms, key=lambda r: r["headroom_to_the_independent_crest_m"])
        if headrooms
        else None
    )
    widest = (
        max(headrooms, key=lambda r: r["headroom_to_the_independent_crest_m"])
        if headrooms
        else None
    )
    return {
        "rule": CORRIDOR_RULE,
        "corridor": str(corridor),
        "per_epoch": rows,
        "n_frames": len(rows),
        "on_the_render_grid": {
            "epochs_with_a_measured_saddle": len(measured),
            "pontic_below_it_at_every_one": all(
                r["pontic_below_the_measured_saddle"] for r in measured
            ) if measured else None,
            # SORTED, and named so. The pair below is min..max over the frames that
            # carry a value; it is NOT the first and last epoch's readings and it
            # must not be read as a run through time -- on the nine shipped 0.5 ka
            # frames the series rises, falls, rises and falls again, and its
            # smallest and largest values sit at the THIRD and SEVENTH epochs.
            "measured_saddle_min_max_m_sorted_not_chronological": (
                [
                    min(r["measured_saddle_m"] for r in measured),
                    max(r["measured_saddle_m"] for r in measured),
                ]
                if measured
                else None
            ),
            # The same numbers IN EPOCH ORDER, each beside the epoch it was measured
            # at, so a reader who wants the series through time has it and never has
            # to reconstruct it from a sorted pair.
            "measured_saddle_by_epoch": [
                [r["epoch_ka"], r["measured_saddle_m"]] for r in rows
            ],
            "measured_saddle_lowest_at_epoch_ka": (
                min(measured, key=lambda r: r["measured_saddle_m"])["epoch_ka"]
                if measured else None
            ),
            "measured_saddle_highest_at_epoch_ka": (
                max(measured, key=lambda r: r["measured_saddle_m"])["epoch_ka"]
                if measured else None
            ),
            "measured_saddle_is_monotone_in_time": (
                None if len(measured) < 2
                else _is_monotone([r["measured_saddle_m"] for r in measured])
            ),
            "note": (
                "the saddle is NOT a constant across the window and the sentence "
                "this block replaces quoted one epoch's value as if it were. Two "
                "fields are given instead of one because a min..max pair had been "
                "read as a run through time: "
                "measured_saddle_min_max_m_sorted_not_chronological is the ASCENDING "
                "SORT's endpoints, measured_saddle_by_epoch is the series in epoch "
                "order, and measured_saddle_is_monotone_in_time says whether the two "
                "can be read for each other on THIS run"
            ),
        },
        "on_the_independent_crest": {
            "epochs_checked": len(independent),
            "pontic_below_it_at_every_one": all(
                r["pontic_below_the_independent_crest"] for r in independent
            ) if independent else None,
            "crest_f0_m": BOSPHORUS_CREST_F0_M,
            "crest_f1_range_m": (
                [
                    min(r["bosphorus_crest_f1_m"] for r in independent),
                    max(r["bosphorus_crest_f1_m"] for r in independent),
                ]
                if independent
                else None
            ),
            "narrowest_headroom": narrowest,
            "widest_headroom": widest,
        },
        "verdict": (
            "AT_MOST_ONE_DONOR_RECEIVER_PAIR"
            if (measured or independent)
            and all(r["pontic_below_the_measured_saddle"] for r in measured)
            and all(r["pontic_below_the_independent_crest"] for r in independent)
            else "NOT_ESTABLISHED_ON_THESE_FRAMES"
        ),
        "what_the_verdict_means": (
            "AT_MOST_ONE_DONOR_RECEIVER_PAIR: the Pontic stands below both readings "
            "of its own western outlet at every epoch checked, so it never spills "
            "west and caspian->pontic is the only donor->receiver pair -- which "
            "means BOTH modes are available and the one in the 'corridor' field "
            f"above ({str(corridor)!r}, what this run asked for) is kept by choice "
            "and not by refusal. The sentence that shipped here until 2026-08-04 "
            "said 'step is kept by choice' whatever the run had asked for, three "
            "fields under a 'corridor' that read 'ramp'. "
            "NOT_ESTABLISHED_ON_THESE_FRAMES: at least one epoch has no reading, or "
            "has one that puts the Pontic above its outlet; read per_epoch"
        ),
    }


class RampNotLegalError(ValueError):
    """``corridor="ramp"`` was asked for at an epoch where it cannot be defined."""


#: WHY COUNTING PAIRS ABOVE THEIR OWN SADDLE IS A PROOF, NOT A HEURISTIC.
#:
#: No digit here: every number this block ships is a field of the row beside the
#: sentence, so a reader checks the claim against the measurement and not against a
#: remembered constant.  The predecessor's version of this claim was a single saddle
#: value measured at ONE epoch and generalised over nine.
RAMP_LEGALITY_RULE = (
    "assign_basins raises CorridorNotRampableError when more than ONE unordered "
    "basin pair is donor->receiver at one epoch. Its own spill test is: basin A "
    "spills into B iff a path exists from A's seed to B's seed on which every cell "
    "lies below A's OWN declared surface. The lowest such path's maximum IS the "
    "connected-path minimax saddle between those two seeds -- the quantity every "
    "frame already measures and records as sills[].elevation_m. So a pair can be "
    "donor->receiver ONLY IF the higher of its two declared surfaces stands above "
    "its own measured saddle, and counting the pairs for which that holds is an "
    "UPPER BOUND on the number of donor->receiver pairs at that epoch. An upper "
    "bound of one is a proof that ramp cannot raise; it is not evidence about it. A "
    "pair whose territories never met under the flood ceiling carries no saddle and "
    "cannot merge. A pair with no measurement at all is UNDECIDED and is never "
    "counted as safe -- substituting 'no reading' for 'does not merge' is the "
    "substitution this project has already paid for once."
)


def ramp_pair_candidates(frame: dict) -> dict:
    """Per pair on ONE frame: could it be donor->receiver?  Measured, or UNDECIDED."""
    levels = {
        name: (entry or {}).get("level_m")
        for name, entry in (frame.get("basin_levels") or {}).items()
    }
    rows: list[dict] = []
    for entry in frame.get("sill") or []:
        pair = entry.get("pair") or []
        if len(pair) != 2:
            continue
        a, b = pair
        la, lb = levels.get(a), levels.get(b)
        saddle = entry.get("measured_saddle_m")
        higher = None if la is None or lb is None else max(float(la), float(lb))
        if higher is None:
            state, why = "UNDECIDED", "one of the two basins carries no level"
        elif saddle is None:
            state, why = (
                "CANNOT_MERGE",
                "the two territories never met under this frame's flood ceiling, so "
                "no path between the seeds exists at any level this frame draws",
            )
        elif higher > float(saddle):
            state, why = (
                "COULD_BE_DONOR_RECEIVER",
                "the higher declared surface stands above the measured saddle",
            )
        else:
            state, why = (
                "CANNOT_MERGE",
                "both declared surfaces stand below the measured saddle",
            )
        rows.append(
            {
                "pair": list(pair),
                "higher_level_m": higher,
                "measured_saddle_m": saddle,
                "headroom_m": (
                    None if higher is None or saddle is None
                    else higher - float(saddle)
                ),
                "state": state,
                "why": why,
            }
        )
    could = [r for r in rows if r["state"] == "COULD_BE_DONOR_RECEIVER"]
    undecided = [r for r in rows if r["state"] == "UNDECIDED"]
    return {
        "epoch_ka": frame.get("epoch_ka"),
        "pairs": rows,
        "n_pairs_measured": len(rows),
        "n_could_be_donor_receiver": len(could),
        "n_undecided": len(undecided),
        "pairs_that_could_merge": [r["pair"] for r in could],
        "verdict": (
            "UNDECIDED" if undecided
            else "RAMP_IS_DEFINED" if len(could) <= 1
            else "RAMP_WOULD_RAISE"
        ),
    }


def ramp_legality(frames: Sequence[dict], corridor: str = "step") -> dict:
    """Whether ``corridor="ramp"`` is defined at EVERY epoch of THIS run.

    Computed from this run's own frames, epoch by epoch.  It is deliberately
    computed even when the run is in step mode -- that is the measurement that tells
    a later run whether it may switch -- and it is reported beside what actually
    happened, so "ramp was legal" and "ramp was used" can never be read off each
    other.
    """
    per_epoch = [ramp_pair_candidates(f) for f in frames]
    raised = sorted(
        float(f["epoch_ka"])
        for f in frames
        if "CorridorNotRampableError" in str(f.get("refusal") or "")
    )
    undecided = [e for e in per_epoch if e["verdict"] == "UNDECIDED"]
    would_raise = [e for e in per_epoch if e["verdict"] == "RAMP_WOULD_RAISE"]
    modes = sorted(
        {
            (f.get("spill") or {}).get("corridor", {}).get("mode")
            for f in frames
            if (f.get("spill") or {}).get("corridor")
        }
    )
    if not per_epoch:
        verdict = "NOT_ESTABLISHED: this run has no frames"
    elif would_raise:
        verdict = "RAMP_WOULD_RAISE_AT: " + ", ".join(
            f"{e['epoch_ka']:g} ka" for e in would_raise
        )
    elif undecided:
        verdict = "UNDECIDED_AT: " + ", ".join(
            f"{e['epoch_ka']:g} ka" for e in undecided
        )
    else:
        verdict = "RAMP_IS_DEFINED_AT_EVERY_EPOCH_OF_THIS_RUN"
    return {
        "rule": RAMP_LEGALITY_RULE,
        "corridor_requested": str(corridor),
        "corridor_modes_the_frames_actually_report": modes,
        "n_epochs_examined": len(per_epoch),
        "per_epoch": per_epoch,
        "epochs_that_raised_CorridorNotRampableError": raised,
        "verdict": verdict,
        "what_the_verdict_means": (
            "RAMP_IS_DEFINED_AT_EVERY_EPOCH_OF_THIS_RUN: at every epoch at most one "
            "basin pair has its higher declared surface above its own measured "
            "saddle, so assign_basins cannot find two donor->receiver pairs and "
            "corridor='ramp' is defined. RAMP_WOULD_RAISE_AT / UNDECIDED_AT name the "
            "epochs and nothing is inferred about the rest. "
            "epochs_that_raised_CorridorNotRampableError is what ACTUALLY happened "
            "in this run, kept separate from what was predicted: if the two ever "
            "disagree, the prediction is wrong and the exception is right"
        ),
    }


def assert_ramp_is_legal(legality: dict) -> None:
    """Refuse a ramp run whose own frames say ramp is not defined everywhere."""
    if not legality["verdict"].startswith("RAMP_IS_DEFINED_AT_EVERY_EPOCH"):
        raise RampNotLegalError(
            "corridor='ramp' was requested and this run's own measurements say it is "
            f"not defined throughout: {legality['verdict']}. Nothing here falls back "
            "to 'step': half a sequence with an interpolated corridor surface and "
            "half without, unlabelled, is worse than either. Re-run in step mode, or "
            "build only the epochs the measurement clears."
        )


#: MY sentence, added beside the renderer's own.  It contains NO DIGIT: every value
#: is interpolated from the frame it is written for.
#:
#: The renderer's ramp-mode disclosure already says the surface is "interpolated
#: linearly in along-channel graph distance ... (a first-order water-surface profile,
#: not hydraulics)", and its common preamble already says the verdict is measured on
#: the terrain.  What it does NOT say -- checked verbatim on a real frame -- is that
#: the interpolation is not a MEASURED surface either, and it never puts the crest's
#: own number beside the two declared levels the ramp runs between.  With a connected
#: blue ribbon in the picture that distinction becomes load-bearing, so it is added
#: HERE, in this module's manifest and on this module's composed canvas, rather than
#: by editing a renderer that has no test file this phase.  It is
#: 6,405 lines long, counted 2026-08-04; ``tests/test_viz.py``'s own incident stub
#: says 6 369, which is wrong and is outside this module's write scope to correct.
RAMP_ADDENDUM_TEMPLATE = (
    "WHAT IS MEASURED HERE, AND WHAT IS NOT. The measured quantity is the CREST: "
    "{measured} m at {crest_lon} E / {crest_lat} N, the connected-path minimax "
    "between this frame's own two basin seeds on dem_sm, the array the water mask is "
    "built from — checked against an independently established {expected} m "
    "({tolerance} m tolerance, gap {gap} m; {source}) and NOT measured on the raster "
    "it is guarding. The donor {donor} stands at {donor_level} m and the receiver "
    "{receiver} at {receiver_level} m, so the crest is passable and the verdict "
    "'{verdict}' rests on those numbers and on nothing in this picture. The SLOPE "
    "between them is NEITHER measured NOR modelled: inside the corridor the surface "
    "is interpolated LINEARLY IN ALONG-CHANNEL GRAPH DISTANCE from the donor's "
    "declared level to the receiver's. No gauge on it was read, no discharge, "
    "roughness or backwater enters it, and no hydraulic model produced it — it is a "
    "first-order profile drawn between two numbers that came from level curves. "
    "{drawn} {claim} Read whatever water you can see along the corridor as 'a "
    "continuous channel exists whose highest point the donor clears', never as 'the "
    "water stood at this height here', and never as 'the two seas met' -- whether "
    "they did in the picture is the basins_joined field in the sentence above."
)

#: The same distinction for a frame drawn in step mode, where there is no ribbon and
#: therefore nothing interpolated.  Also digit-free.
STEP_ADDENDUM_TEMPLATE = (
    "WHAT IS MEASURED HERE. The CREST: {measured} m at {crest_lon} E / {crest_lat} N, "
    "the connected-path minimax between this frame's own two basin seeds on dem_sm, "
    "checked against an independently established {expected} m ({tolerance} m "
    "tolerance, gap {gap} m; {source}). The donor {donor} stands at {donor_level} m "
    "and the receiver {receiver} at {receiver_level} m. Nothing in this frame is "
    "interpolated: corridor='step' draws every cell at its own basin's flat declared "
    "level and only MARKS the corridor, so the water you can see is the terrain "
    "compared against those two numbers, cell by cell. {drawn} {claim}"
)

#: What the corridor ACTUALLY came out as on the frame being written.  Digit-free,
#: like the two above, and filled from ``connectivity.corridor_if_painted`` and
#: ``painted_water.basins_joined`` -- the same fields the renderer's own disclosure
#: quotes, so the two sentences cannot come to disagree.
#:
#: THIS CLAUSE EXISTS BECAUSE THE PROBE FALSIFIED THE SENTENCE THAT WAS HERE FIRST.
#: The 2026-08-04 16.5 ka ramp probe was written with a tail reading "read the
#: CONNECTED RIBBON as ...".  Measured on that very frame the ramp drew 15,381
#: corridor cells in 112 DISJOINT pieces with ``basins_joined=false`` -- so the
#: phrase asserted a connectedness the picture does not have, which is the fifth
#: instance of the defect class this project has already retracted four sentences
#: for.  The clause now states what was drawn instead of what was expected.
CORRIDOR_DRAWN_TEMPLATE = (
    "WHAT IT ACTUALLY DREW HERE: {corridor_cells} corridor cell(s), "
    "{corridor_area} sq km, in {corridor_components} disjoint piece(s), and the "
    "basins {joined} in this picture (painted_water.basins_joined)."
)

#: THE SIZE OF THE CLAIM THE RAMP MAKES, on the frame being written.  Digit-free.
#:
#: This is the quantity the milestone was asked for and that nothing stated: how much
#: ground the interpolated surface draws as WATER that a flat surface leaves DRY.  It
#: is not a detail of the corridor -- it IS the corridor's whole content as a claim
#: about the world, and until it is a number a reader cannot tell whether the ramp
#: moved a coastline or a rounding error.
#:
#: It must never be read off ``painted_water.corridor_cells_restored_by_the_or``,
#: which is the only field that looks like it.  That field counts the cells the
#: morphological OPENING had already removed and the corridor OR then put back --
#: 243 on the 2026-08-04 full-frame probe and 230 on its inset -- i.e. a repair to the
#: de-speckle, roughly two orders of magnitude smaller than the claim.  The two are
#: reported side by side, with the ratio computed, so the small one can never be
#: quoted for the large one.
RAMP_CLAIM_TEMPLATE = (
    "HOW BIG THE NEW CLAIM IS: the interpolated surface draws {claim_cells} cell(s), "
    "{claim_area} sq km, in {claim_pieces} disjoint piece(s), as WATER that a flat "
    "surface leaves DRY -- {claim_share} of the water painted in this frame. Those "
    "cells are the corridor itself: assign_basins carves it out of the donor-level "
    "flood MINUS the donor's own water and MINUS the receiver's own connected water, "
    "so no cell of it belongs to either basin at its own flat level, and this frame "
    "checks that against its own terrain rather than asserting it -- {bed_check}. "
    "DO NOT read this off painted_water.corridor_cells_restored_by_the_or, which is "
    "{or_cells} cell(s) here, {or_ratio}: that field counts only what the de-speckle "
    "had opened away and the corridor OR put back. What a flat frame could still "
    "paint of this ground through the de-speckle's closing is bounded above by "
    "morphology.cells_added, {closing_cells} cell(s) over the whole frame."
)

#: Why the inset carries a guard of its own, said with THIS inset's own numbers.
#:
#: WHAT IT REPLACED.  Until 2026-08-04 this sentence read "a coarser grid inflates
#: this saddle -- +8.30 m at 2000 px and +5.24 m at 4000 px against the native
#: +0.30 m on the full scene".  Two of those three figures are readings at grids the
#: object shipping the sentence never used and carries no field for, so a reader
#: could not check them against anything beside them -- which is exactly the break
#: the full frame's own templates were rewritten to close.  The point survives; the
#: numbers are now this object's own, and the grid each was measured on is the
#: ``measured_on`` string the renderer wrote, quoted rather than summarised.
INSET_GUARD_TEMPLATE = (
    "the EXPECTATION is a property of the terrain and the vertical frame, so it is "
    "the same number the full frame is held to: {expected} m. The MEASUREMENT is a "
    "property of the GRID and is not, so this panel measures its own and is refused "
    "if it leaves the same tolerance: {measured} m on this inset's array, gap {gap} m "
    "against a tolerance of {tolerance} m. The grid it came off is stated in "
    "measured_on beside this sentence -- '{grid}' -- because a saddle without the "
    "grid it was measured on can only be trusted, never re-derived: on this scene a "
    "coarser grid fills the channel in and turns 'spilling' into 'endorheic', which "
    "is why --max-px below the native grid is a refusal rather than a knob (see the "
    "CLI's own message, which carries the sub-native readings and the fields they "
    "are compared against)."
)

#: The same quantity for a frame that does NOT paint its corridor.  Digit-free.
NO_RAMP_CLAIM_TEMPLATE = (
    "HOW BIG THE NEW CLAIM IS: none. This frame's corridor is {corridor_state}, so "
    "no cell is drawn as water on the strength of an interpolated surface and every "
    "blue cell stands below its own basin's flat declared level. What painting it "
    "WOULD have claimed is measured anyway and is {would_be_cells} cell(s), "
    "{would_be_area} sq km."
)


def _fmt(value, spec: str) -> str:
    """A measured value for a sentence, or the word for its absence."""
    return "not measured" if value is None else format(value, spec)


def shipped_templates() -> dict[str, str]:
    """Every SHIPPED TEMPLATE this module carries, found by inspection.

    A template here is a module-level ``str`` constant that either

    * is named ``*_TEMPLATE``, or
    * carries at least one replacement field whose name is a Python identifier --
      which is what distinguishes ``"{measured} m at {crest_lon} E"`` from prose that
      merely contains braces, such as ``CONNECTIVITY_STATES_RULE``'s
      ``"isfinite(dem) & (dem < ...)"``.

    NOTHING IS ENUMERATED.  A template added tomorrow is in this dict tomorrow, with
    no list to remember to extend -- which is the point: five sentences carrying a
    figure that the object beside them did not have have now shipped on this project,
    each one caught by a reader rather than by a guard, and each fix so far has been
    to delete the offending digits from ONE named string.  The test that consumes
    this refuses ANY digit in ANY of them.

    What it deliberately does NOT cover, stated so the coverage is not overread:
    format strings built inside a function body, and f-strings, whose literal halves
    legitimately carry digits (``f"{v:.6f} m"`` and ``"3x3 cell block"`` alike).  The
    rule this module works to is that a sentence which SHIPS and carries a number
    must be a module-level template filled from the artifact being written -- and
    that is exactly the set below.
    """
    import string as _string

    found: dict[str, str] = {}
    for name, value in sorted(globals().items()):
        if name.startswith("_") or not isinstance(value, str):
            continue
        if name.endswith("_TEMPLATE"):
            found[name] = value
            continue
        try:
            fields = [
                field
                for _, field, _, _ in _string.Formatter().parse(value)
                if field is not None
            ]
        except ValueError:          # not parseable as a format string at all
            continue
        if any(field.isidentifier() for field in fields):
            found[name] = value
    return found


def ramp_new_wet_ground(
    frame: dict, *, corridor_mode: Optional[str] = None
) -> dict:
    """How much ground THIS frame draws as water that a flat surface leaves dry.

    Computed at write time from the frame's own ``.basins.json`` blocks -- nothing
    here is carried between frames and nothing is recomputed from a raster.

    WHY THE CORRIDOR IS THE WHOLE OF IT
    -----------------------------------
    ``assign_basins`` builds the corridor as
    ``candidates = extent & ~donor_water & ~receiver_water`` (minus any third basin's
    territory), where ``extent`` is the flood component at the DONOR's level holding
    the donor seed, ``donor_water`` is the donor's own water inside it and
    ``receiver_water`` is the connected component at the RECEIVER's level holding the
    receiver seed.  A cell in that set is therefore in neither basin's own connected
    flat water, by construction: a flat frame does not paint it.  In ramp mode the
    drawn corridor is that set intersected with ``dem < the ramped surface`` and then
    with the connected mask, so it can only be smaller.

    WHAT IS CHECKED RATHER THAN ARGUED
    ----------------------------------
    ``bed_min_m > receiver_level_m`` says every corridor cell's bed stands above the
    receiver's flat surface, so a flat frame leaves it dry on ELEVATION alone,
    without appealing to connectivity at all.  Both figures are in the same
    ``corridor_if_painted`` object, so the check is one subtraction a reader can
    redo.  When it does NOT hold the block says so and the count is reported as an
    upper bound -- some of those cells would then have been dry only for want of a
    route, which is a weaker claim and is labelled as one.

    The residue a flat frame could still paint here comes from the de-speckle's
    CLOSING, and is bounded above by ``morphology.cells_added`` on this frame.
    """
    spill = frame.get("spill") or {}
    corridor = spill.get("corridor") or {}
    connectivity = spill.get("connectivity") or {}
    cost = connectivity.get("corridor_if_painted") or {}
    painted_water = spill.get("painted_water") or {}
    morphology = spill.get("morphology") or (frame.get("morphology") or {})

    # Three sources, in order, and the one that answered is RECORDED.  A frame whose
    # sidecar says nothing about its corridor is not silently assumed to have none.
    mode = corridor.get("mode") or corridor_mode
    painted = corridor.get("painted")
    painted_from = "spill.corridor.painted, written by the renderer"
    if painted is None:
        painted = mode == "ramp"
        painted_from = (
            "derived from the corridor MODE, because spill.corridor.painted is absent "
            f"on this frame (mode={mode!r}, taken from "
            + (
                "spill.corridor.mode"
                if corridor.get("mode")
                else "the mode this call was given"
            )
            + "); assign_basins paints the corridor in ramp mode and only marks it "
            "otherwise"
        )

    bed_min = cost.get("bed_min_m")
    receiver_level = cost.get("receiver_level_m")
    above_receiver = (
        None if bed_min is None or receiver_level is None
        else bool(float(bed_min) > float(receiver_level))
    )
    or_cells = painted_water.get("corridor_cells_restored_by_the_or")
    cells = cost.get("cells") if painted else 0
    block = {
        "what_it_counts": (
            "cells this frame draws as WATER on the strength of the interpolated "
            "corridor surface, which a flat surface at each basin's own declared "
            "level leaves DRY. It is the size of the claim the ramp makes about the "
            "world, in cells and in square kilometres"
        ),
        "corridor_is_painted": bool(painted),
        "corridor_is_painted_read_from": painted_from,
        "corridor_mode": mode,
        "cells_the_ramp_wets_that_a_flat_surface_leaves_dry": cells,
        "km2_the_ramp_wets_that_a_flat_surface_leaves_dry": (
            cost.get("km2") if painted else 0.0
        ),
        "pieces": cost.get("components") if painted else 0,
        "share_of_this_frames_painted_water": (
            None
            if not painted or not cells or not painted_water.get("cells")
            else round(100.0 * float(cells) / float(painted_water["cells"]), 4)
        ),
        "every_cell_stands_above_the_receivers_flat_surface": above_receiver,
        "bed_min_m": bed_min,
        "receiver_level_m": receiver_level,
        "bed_min_above_receiver_level_by_m": (
            None if bed_min is None or receiver_level is None
            else float(bed_min) - float(receiver_level)
        ),
        "donor": cost.get("donor"),
        "donor_level_m": cost.get("donor_level_m"),
        "receiver": cost.get("receiver"),
        "is_an_upper_bound": (
            None if above_receiver is None else (not above_receiver)
        ),
        "the_margin_is_thin_by_construction": (
            "bed_min_above_receiver_level_by_m is the deepest corridor cell against "
            "the receiver's surface, and the corridor ENDS at that surface: the "
            "interpolated profile tends to the receiver's level at the receiving "
            "end, so the deepest corridor cell is the one immediately landward of "
            "the receiver's own shoreline and the margin is a fraction of a metre "
            "whatever the frame. Its SIGN is the measurement; its size is not"
        ),
        "why_it_would_be_an_upper_bound": (
            "every corridor cell's bed stands above the receiver's own flat surface "
            "on this frame, so a flat frame leaves all of them dry on elevation "
            "alone and the count is exact"
            if above_receiver
            else "at least one corridor cell's bed lies BELOW the receiver's flat "
            "surface, so that cell is dry in a flat frame only because no route "
            "reaches it. The count then mixes 'above every surface' with 'below a "
            "surface but unreachable', which are different claims, and the sidecar "
            "carries no per-cell mask to split them here"
            if above_receiver is False
            else "not decided: this frame carries no corridor bed measurement"
        ),
        "not_to_be_confused_with": {
            "field": "painted_water.corridor_cells_restored_by_the_or",
            "value": or_cells,
            "what_that_field_counts": (
                "cells the morphological OPENING removed from the water mask and the "
                "ramp-mode OR then restored -- a repair to the de-speckle, not a "
                "claim about the terrain. It is a small fraction of this quantity "
                "and must never be quoted for it"
            ),
            "this_quantity_is_larger_by": (
                None
                if not or_cells or not cells
                else round(float(cells) / float(or_cells), 2)
            ),
        },
        "flat_frame_residue_upper_bound": {
            "field": "morphology.cells_added",
            "value": morphology.get("cells_added"),
            "why": (
                "the de-speckle's CLOSING fills small holes, so a flat frame can "
                "paint a corridor cell it never owned. Every such cell on the whole "
                "frame is inside cells_added, which therefore bounds the part of "
                "this count a flat frame would have drawn anyway"
            ),
        },
        "if_it_were_painted": (
            None
            if painted
            else {
                "cells": cost.get("cells"),
                "km2": cost.get("km2"),
                "components": cost.get("components"),
                "note": (
                    "corridor='step' marks the corridor and does not paint it, so the "
                    "claim above is zero and this is what the same set would cost"
                ),
            }
        ),
    }
    return block


def ramp_claim_clause(claim: Optional[dict]) -> str:
    """One measured sentence naming the size of the claim.  No digit typed."""
    claim = claim or {}
    if not claim.get("corridor_is_painted"):
        mode = claim.get("corridor_mode")
        return NO_RAMP_CLAIM_TEMPLATE.format(
            corridor_state=(
                "MARKED and not painted"
                if mode in ("step", "ramp")
                else "reported as absent -- the basins do not connect on this frame"
                if mode == "none"
                else "not recorded on this frame"
                if mode is None
                else f"reported as {mode!r}"
            ),
            would_be_cells=_fmt(
                (claim.get("if_it_were_painted") or {}).get("cells"), ","
            ),
            would_be_area=_fmt(
                (claim.get("if_it_were_painted") or {}).get("km2"), ",.0f"
            ),
        )
    cells = claim.get("cells_the_ramp_wets_that_a_flat_surface_leaves_dry")
    share = claim.get("share_of_this_frames_painted_water")
    ratio = (claim.get("not_to_be_confused_with") or {}).get(
        "this_quantity_is_larger_by"
    )
    margin = claim.get("bed_min_above_receiver_level_by_m")
    return RAMP_CLAIM_TEMPLATE.format(
        claim_cells=_fmt(cells, ","),
        claim_area=_fmt(
            claim.get("km2_the_ramp_wets_that_a_flat_surface_leaves_dry"), ",.0f"
        ),
        claim_pieces=_fmt(claim.get("pieces"), ","),
        claim_share=(
            "a share this frame does not carry"
            if share is None
            else f"{share:.4f} %"
        ),
        bed_check=(
            "its lowest corridor bed stands "
            f"{_fmt(margin, '+.6f')} m ABOVE the receiver's flat surface, so the flat "
            "frame leaves every one of them dry on elevation alone. That margin is "
            "THIN BY CONSTRUCTION and its size means nothing: the corridor ends at "
            "the receiver's own shoreline, where the interpolated surface tends to "
            "the receiver's level, so the deepest corridor cell is always the one "
            "just landward of it. What the check asks is only whether the margin is "
            "positive"
            if claim.get("every_cell_stands_above_the_receivers_flat_surface")
            else "its lowest corridor bed lies at or below the receiver's flat "
            "surface, so part of this count is dry in a flat frame for want of a "
            "route rather than for want of depth, and the figure is an UPPER BOUND"
            if claim.get("every_cell_stands_above_the_receivers_flat_surface") is False
            else "this frame carries no corridor bed measurement, so the check could "
            "not be run and the figure is reported unchecked"
        ),
        or_cells=_fmt(
            (claim.get("not_to_be_confused_with") or {}).get("value"), ","
        ),
        or_ratio=(
            "a ratio this frame does not carry"
            if ratio is None
            else f"smaller by a factor of {ratio:,.2f}"
        ),
        closing_cells=_fmt(
            (claim.get("flat_frame_residue_upper_bound") or {}).get("value"), ","
        ),
    )


def ramp_claim_summary(frames: Sequence[dict]) -> dict:
    """The size of the claim over the whole run, per epoch and never summed.

    The counts are NOT added across epochs, for the same reason
    :data:`CONNECTIVITY_NO_SUM_RULE` gives: it is the same ground in every frame, and
    a sum of it describes no surface.  The extremes name the epoch that produced
    them, and every row keeps its epoch's slot even when the corridor was not
    painted -- an epoch that claims nothing is a measurement, not an absence.
    """
    rows = []
    for frame in frames:
        claim = frame.get("ramp_new_wet_ground")
        if claim is None:
            if frame.get("png"):
                rows.append(
                    {
                        "epoch_ka": frame.get("epoch_ka"),
                        "cells": None,
                        "km2": None,
                        "note": "this frame carries no ramp_new_wet_ground block",
                    }
                )
            continue
        rows.append(
            {
                "epoch_ka": frame.get("epoch_ka"),
                "corridor_mode": claim.get("corridor_mode"),
                "corridor_is_painted": claim.get("corridor_is_painted"),
                "cells": claim.get(
                    "cells_the_ramp_wets_that_a_flat_surface_leaves_dry"
                ),
                "km2": claim.get("km2_the_ramp_wets_that_a_flat_surface_leaves_dry"),
                "pieces": claim.get("pieces"),
                "share_of_this_frames_painted_water": claim.get(
                    "share_of_this_frames_painted_water"
                ),
                "every_cell_stands_above_the_receivers_flat_surface": claim.get(
                    "every_cell_stands_above_the_receivers_flat_surface"
                ),
                "corridor_cells_restored_by_the_or": (
                    (claim.get("not_to_be_confused_with") or {}).get("value")
                ),
                "this_quantity_is_larger_by": (
                    (claim.get("not_to_be_confused_with") or {}).get(
                        "this_quantity_is_larger_by"
                    )
                ),
            }
        )
    painted = [r for r in rows if r.get("cells")]
    return {
        "what_it_counts": (
            "per frame, the ground the interpolated corridor surface draws as WATER "
            "that a flat surface at each basin's own declared level leaves DRY -- "
            "the size of the claim this sequence's corridor makes. Computed at write "
            "time by ramp_new_wet_ground from each frame's own sidecar"
        ),
        "per_epoch": rows,
        "n_frames_with_a_painted_corridor": len(painted),
        "epochs_that_claim_nothing": [
            r["epoch_ka"] for r in rows if not r.get("cells")
        ],
        "most_cells": (
            max(painted, key=lambda r: r["cells"]) if painted else None
        ),
        "fewest_cells": (
            min(painted, key=lambda r: r["cells"]) if painted else None
        ),
        "why_the_counts_are_not_summed": (
            "the same ground appears in every frame, so a total over epochs measures "
            "no surface. Read the per-epoch rows, or most_cells / fewest_cells, each "
            "of which names the epoch it belongs to"
        ),
        "do_not_confuse_it_with": (
            "painted_water.corridor_cells_restored_by_the_or, carried in each row "
            "beside this quantity together with the ratio between them. That field "
            "counts only the cells the de-speckle's OPENING had removed and the "
            "ramp-mode OR restored -- a repair to the morphology, two orders of "
            "magnitude smaller, and the only field in the sidecar that looks like "
            "this one"
        ),
    }


def corridor_drawn_clause(spill: Optional[dict]) -> str:
    """One measured sentence about what the corridor came out as.  No digit typed."""
    spill = spill or {}
    cost = ((spill.get("connectivity") or {}).get("corridor_if_painted")) or {}
    joined = (spill.get("painted_water") or {}).get("basins_joined")
    return CORRIDOR_DRAWN_TEMPLATE.format(
        corridor_cells=_fmt(cost.get("cells"), ","),
        corridor_area=_fmt(cost.get("km2"), ",.0f"),
        corridor_components=_fmt(cost.get("components"), ","),
        joined=(
            "did NOT visibly merge"
            if joined is False
            else "DID visibly merge into one water body"
            if joined is True
            else "were not counted for merging"
        ),
    )


def corridor_addendum(frame: dict, *, corridor_mode: Optional[str]) -> Optional[dict]:
    """The added disclosure for ONE frame, or ``None`` when it has no crest to state.

    Raises :class:`KeyError` on a corridor mode this module has no sentence for,
    exactly as :func:`spill_not_drawn_disclosure` does: a disclosure that is not
    about the algorithm that ran is not a disclosure.
    """
    entry = next(
        (
            s for s in (frame.get("sill") or [])
            if sorted(s.get("pair") or []) == ["caspian", "pontic"]
        ),
        None,
    )
    if entry is None or entry.get("measured_saddle_m") is None:
        return None
    levels = {
        name: (value or {}).get("level_m")
        for name, value in (frame.get("basin_levels") or {}).items()
    }
    caspian, pontic = levels.get("caspian"), levels.get("pontic")
    if caspian is None or pontic is None:
        return None
    donor, receiver = (
        ("caspian", "pontic") if caspian >= pontic else ("pontic", "caspian")
    )
    template = {
        "ramp": RAMP_ADDENDUM_TEMPLATE,
        "step": STEP_ADDENDUM_TEMPLATE,
        "none": STEP_ADDENDUM_TEMPLATE,
        None: STEP_ADDENDUM_TEMPLATE,
    }[corridor_mode]
    # The size of the claim, derived from THIS frame before the sentence is built, so
    # the manifest field and the drawn caption are the same object rather than two
    # readings of it.
    claim = ramp_new_wet_ground(frame, corridor_mode=corridor_mode)
    lonlat = entry.get("measured_lonlat") or [None, None]
    text = template.format(
        measured=_fmt(entry.get("measured_saddle_m"), "+.3f"),
        crest_lon=_fmt(lonlat[0], ".6f"),
        crest_lat=_fmt(lonlat[1], ".6f"),
        expected=_fmt(entry.get("expected_m"), "+.3f"),
        tolerance=_fmt(entry.get("tolerance_m"), ".2f"),
        gap=_fmt(entry.get("gap_m"), "+.3f"),
        source=entry.get("expectation_source") or "no source recorded",
        donor=donor,
        receiver=receiver,
        donor_level=_fmt(levels[donor], "+.3f"),
        receiver_level=_fmt(levels[receiver], "+.3f"),
        verdict=entry.get("verdict") or "no verdict recorded",
        drawn=corridor_drawn_clause(frame.get("spill")),
        claim=ramp_claim_clause(claim),
    )
    return {
        "corridor_mode": corridor_mode,
        "text": text,
        "template_carries_no_digit": not any(c.isdigit() for c in template),
        "ramp_new_wet_ground": claim,
        "why_it_is_here_and_not_in_the_renderer": (
            "paleoeurope/viz/paleocostline_render.py is off limits this phase: its "
            "test file was destroyed on 2026-08-04 by a full-disk truncation and the "
            "replacement is a stub that fails on purpose, so the module has no "
            "coverage to change it under. Its own ramp sentence does say the surface "
            "is interpolated and is not hydraulics; what it does not say is that the "
            "interpolation is not a MEASURED surface either, it never puts the "
            "crest's own value beside the two levels the ramp runs between, and it "
            "never states HOW MUCH GROUND the interpolation newly wets. That is "
            "what this adds, in the manifest and on the composed canvas"
        ),
    }


def sill_expectation_summary(frames: Sequence[dict]) -> dict:
    """Per pair, what was expected and what was measured -- summarised from frames.

    This replaces three hand-written paragraphs that carried unlabelled constants
    (``+13.345 m``, ``+22.86 m``, ``+22.53 m``) measured at 16.5 ka and shipped in
    manifests spanning 17-13 ka, where every one of them varies epoch by epoch.  The
    REASONS still ship -- they are per-epoch strings built by
    :func:`sill_expectations_for` and carried in ``frames[].sill[]`` -- and what is
    summarised here is only what can be counted over this run's own frames.
    """
    pairs: dict[str, dict] = {}
    for frame in frames:
        for entry in frame.get("sill") or []:
            key = "|".join(entry.get("pair") or [])
            block = pairs.setdefault(
                key,
                {
                    "pair": entry.get("pair"),
                    "expectation_kind": entry.get("expectation_kind"),
                    "n_frames": 0,
                    "verdicts": {},
                    "expected_m_range": None,
                    "measured_saddle_m_range": None,
                    "gap_m_range": None,
                    "smoothing_shift_m_range": None,
                    "tolerance_m": entry.get("tolerance_m"),
                    "reason_is_per_epoch": bool(entry.get("unmeasured_reason")),
                    "where_the_reason_is": (
                        "frames[].sill[].unmeasured_reason -- built for each epoch "
                        "from that epoch's own dz, so it is not repeated here"
                        if entry.get("unmeasured_reason")
                        else None
                    ),
                    "where_the_source_is": (
                        "frames[].sill[].expectation_source -- built for each epoch "
                        "from that epoch's own dz_sill_m"
                        if entry.get("expectation_source")
                        else None
                    ),
                },
            )
            block["n_frames"] += 1
            verdict = str(entry.get("verdict"))
            block["verdicts"][verdict] = block["verdicts"].get(verdict, 0) + 1
            for name, source_key in (
                ("expected_m_range", "expected_m"),
                ("measured_saddle_m_range", "measured_saddle_m"),
                ("gap_m_range", "gap_m"),
                ("smoothing_shift_m_range", "smoothing_shift_m"),
            ):
                value = entry.get(source_key)
                if value is None:
                    continue
                current = block[name]
                block[name] = (
                    [float(value), float(value)]
                    if current is None
                    else [min(current[0], float(value)), max(current[1], float(value))]
                )
    return {
        "rule": (
            "counted over the frames of THIS sequence. Ranges are min..max over the "
            "frames that carry a value; a pair with no measurement on any frame has "
            "None, which is 'not reached' and never 'agrees'. The per-epoch reason "
            "and source strings are NOT copied up here: they are built per epoch and "
            "a single copy of one of them at the top of a 9- or 17-frame manifest "
            "would be one epoch's number wearing the sequence's name"
        ),
        "by_pair": pairs,
    }


#: The reference the offset cross-check is run against, WITH the document that
#: prints each figure and at what precision.
#:
#: WHAT THIS REPLACED, AND WHY IT HAD TO
#: -------------------------------------
#: The block shipped in every manifest until 2026-08-04 said it compared against
#: "the five figures the contract prints" and listed 21.0 ka 87.332, 18.0 81.959,
#: 16.5 77.906, 14.5 66.373, 12.0 30.005.  Checked against the document it names:
#:
#: * the vertical-datum contract carries **no 21.0 ka row at all**, so 87.332 had no
#:   source anywhere in either repository -- it appeared only in this module;
#: * its offset column prints **two** decimals, not three, so 81.959 / 66.373 /
#:   30.005 are a precision the cited document does not have;
#: * that column also carries its own 2026-07-31 amendment saying **"do not quote a
#:   number out of this table"**, which is exactly what the block was doing;
#: * and the third decimal was wrong twice where a source does exist: the engine
#:   reads 81.958069 and 66.374115, i.e. 81.958 and 66.374 to three places.
#:
#: The reference below is a table that IS printed at three decimals, in a document
#: that states it was measured with the same estimator the bake uses:
#: ``2026-07-31-manych-sill-history.md``.  Five of its six epochs are epochs this
#: animation renders, so the comparison lands on the sequence's own frames -- the
#: Mediterranean surface of each is ``-offset(t)`` and can be read straight out of
#: ``frames[].basin_levels.mediterranean.level_m``.
OFFSET_REFERENCE = (
    (18.0, 81.958),
    (17.0, 79.467),
    (16.5, 77.906),
    (15.0, 69.792),
    (14.5, 66.374),
    (14.0, 45.997),
)

#: Half of the last printed digit of :data:`OFFSET_REFERENCE`: three decimals, so a
#: difference at or under 0.0005 m is rounding in the printed figure and not a
#: disagreement.  A difference LARGER than this is reported as one.
OFFSET_REFERENCE_PRECISION_M = 0.0005

OFFSET_REFERENCE_SOURCE = (
    "Paleo-Europe_3D/docs/superpowers/research/2026-07-31-manych-sill-history.md "
    "-- the offset(t) column of its sill-displacement table, printed to three "
    "decimals, measured there from data/raw/ICE6G/I6_C.VM5a_10min.*.nc with the "
    "far-field-debiased estimator the bake uses. Chosen over the vertical-datum "
    "contract's own offset column because that column prints two decimals, carries "
    "no 21 ka row, and its 2026-07-31 amendment says in as many words 'do not quote "
    "a number out of this table'"
)


def offset_cross_check() -> dict:
    """``sea_surface_offset`` against a table that actually prints those digits.

    Recomputed at manifest write time -- it used to be five hard-coded rows carrying
    a measured column, a "contract" column and a difference, true when they were
    written and unfalsifiable afterwards because nothing in the run touched them.
    See :data:`OFFSET_REFERENCE` for what was wrong with the reference itself.
    """
    rows: list[dict] = []
    status = "MEASURED"
    try:
        from scripts import bake_khvalynian as bk
    except Exception as exc:  # noqa: BLE001 - an unimportable bake is data
        return {
            "status": f"UNAVAILABLE: {type(exc).__name__}: {exc}",
            "note": (
                "bake_khvalynian.sea_surface_offset could not be imported, so the "
                "cross-check was NOT run. No previous run's numbers are reproduced "
                "here in its place"
            ),
        }
    for epoch, printed in OFFSET_REFERENCE:
        try:
            measured = float(bk.sea_surface_offset(epoch))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "epoch_ka": epoch,
                    "measured_m": None,
                    "reference_m": printed,
                    "difference_m": None,
                    "within_the_references_printing_precision": None,
                    "status": f"UNAVAILABLE: {type(exc).__name__}: {exc}",
                }
            )
            status = "PARTIAL"
            continue
        difference = measured - printed
        rows.append(
            {
                "epoch_ka": epoch,
                "measured_m": measured,
                "reference_m": printed,
                "difference_m": difference,
                "within_the_references_printing_precision": (
                    abs(difference) <= OFFSET_REFERENCE_PRECISION_M
                ),
            }
        )
    differences = [
        abs(r["difference_m"]) for r in rows if r.get("difference_m") is not None
    ]
    return {
        "status": status,
        "note": (
            "bake_khvalynian.sea_surface_offset recomputed at manifest write time "
            "against a table printed to THREE decimals in the document named below. "
            f"A difference at or under {OFFSET_REFERENCE_PRECISION_M} m is rounding "
            "in that printed figure; anything larger is a disagreement and is "
            "flagged per row rather than absorbed into a tolerance"
        ),
        "reference_source": OFFSET_REFERENCE_SOURCE,
        "reference_printing_precision_m": OFFSET_REFERENCE_PRECISION_M,
        "rows": rows,
        "largest_absolute_difference_m": max(differences) if differences else None,
        "all_within_the_references_printing_precision": (
            all(
                r["within_the_references_printing_precision"]
                for r in rows
                if r.get("within_the_references_printing_precision") is not None
            )
            if rows
            else None
        ),
        "where_to_check_it_against_this_sequence": (
            "five of these six epochs are epochs this animation renders, and the "
            "Mediterranean surface of each is -offset(t) by construction -- so "
            "frames[].basin_levels.mediterranean.level_m should be the negative of "
            "measured_m at the same epoch, to the bit"
        ),
    }


def paths_block(manifest: dict) -> dict:
    """The two spellings, the drive map, and which keys carry which -- measured.

    See :data:`PATH_SPELLING_DECISION`.  The drive map is built from the paths this
    run actually wrote, and the per-key inventory is built by looking at the values
    in the manifest rather than by describing them, so a key that changes spelling
    shows up here instead of silently contradicting a sentence.
    """
    frames = manifest.get("frames") or []
    seen: list[str] = []
    inventory: dict[str, dict[str, int]] = {}

    def _note(key: str, value: Any) -> None:
        if isinstance(value, dict):
            for name in ("as_written", "resolved"):
                if value.get(name):
                    _note(f"{key}.{name}", value[name])
            return
        if not isinstance(value, str) or not value:
            return
        seen.append(value)
        tally = inventory.setdefault(key, {"unc": 0, "drive_letter": 0, "other": 0})
        if value.startswith("\\\\"):
            tally["unc"] += 1
        elif len(value) > 1 and value[1] == ":":
            tally["drive_letter"] += 1
        else:
            tally["other"] += 1

    for frame in frames:
        raster = frame.get("raster") or {}
        _note("frames[].raster.path_as_read", raster.get("path_as_read"))
        _note("frames[].raster.source_path", raster.get("source_path"))
        _note("frames[].png", frame.get("png"))
        _note("frames[].canvas_frame", frame.get("canvas_frame"))
        for seed in frame.get("seeds") or []:
            _note("frames[].seeds[].raster_path", seed.get("raster_path"))
    for curve in manifest.get("level_curves") or []:
        _note("level_curves[].path", curve.get("path"))
    animation = manifest.get("animation") or {}
    _note("animation.gif", animation.get("gif"))
    _note("animation.webp", animation.get("webp"))
    for entry in ((manifest.get("palette") or {}).get("master") or {}).get(
        "built_over"
    ) or []:
        _note("palette.master.built_over[]", entry)
    return {
        "decision": PATH_SPELLING_DECISION,
        "drive_map": drive_map(
            seen + [str(CANONICAL_TOPO_DIR), str(ANIM_ROOT), str(KHV_TOPO_DIR)]
        ),
        "spelling_by_key": inventory,
        "how_to_read_spelling_by_key": (
            "counts of the values actually written into this file, per key: 'unc' "
            "begins \\\\, 'drive_letter' begins X:, 'other' is neither. A key with "
            "both counts non-zero is not an inconsistency to fix -- it means some "
            "of those paths went through Path.resolve() and some did not, and "
            "drive_map converts either way"
        ),
        "topo_dir": path_record(KHV_TOPO_DIR),
        "tree_of_record": path_record(CANONICAL_TOPO_DIR),
        "anim_root_default": path_record(ANIM_ROOT),
    }


#: What the three connectivity states are, and which of them is a measurement.
#:
#: No digit appears here on purpose: every number this block ships is a FIELD of the
#: object that carries it, so a reader can check the sentence against the array beside
#: it.  The one thing a sentence may say is what the states MEAN.
CONNECTIVITY_STATES_RULE = (
    "assign_basins sorts every component of the flat mask "
    "{isfinite(dem) & (dem < that cell's own declared surface)} into three states, "
    "and only CONNECTED is painted as water. CLOSED and UNOBSERVABLE are NOT the "
    "same thing and this block exists so the difference survives into the manifest. "
    "CLOSED is a MEASUREMENT: the body holds no basin seed and its rim lies wholly "
    "inside the extent, above the surface the body carries, so this raster does "
    "observe a barrier and drawing the body as land states what the terrain says. "
    "UNOBSERVABLE is the ABSENCE of one: the body holds no seed and reaches the edge "
    "of what this raster can observe -- an array side, or a nodata cell -- so the "
    "route that would decide it runs outside the extent, and it is drawn as neither "
    "water nor land, in a reserved hatched colour named in the figure's own legend "
    "and caption. Until the revision that added this block the two were one field "
    "spelled 'dropped' and "
    "both were painted as dry land, so every shipped frame asserted a barrier nobody "
    "had observed -- the same defect as painting them would have been, which asserts "
    "a connection nobody observed."
)

#: Why the per-epoch cell counts are NOT added up.
CONNECTIVITY_NO_SUM_RULE = (
    "cells are counted per epoch on that epoch's own raster, and the epochs of this "
    "sequence overlap almost entirely in space. Adding them would count the same "
    "ground once per frame and produce a number describing no surface that ever "
    "existed, so this block reports the range and names the epoch at each end."
)


def unobservable_summary(frames: Sequence[dict]) -> dict:
    """The third state, counted over the frames of THIS sequence.

    Every number is read out of ``frames[].spill.connectivity`` and
    ``frames[].spill.unobservable_drawn`` -- the blocks the renderer wrote into each
    frame's own ``.basins.json``.  Nothing here is recomputed from a raster and
    nothing is carried in from another run.

    THE CHECK THAT MAKES THIS AUDITABLE
    -----------------------------------
    ``partition_is_exact_on_every_frame`` is the statement that
    ``connected + unobservable + closed == flat_mask_cells`` on every frame that
    carries a measurement, and the same for the component counts.  It is what turns
    "there are three states" from a claim into an arithmetic identity a reader can
    verify against the four fields beside it.  Any frame where it does not hold is
    NAMED in ``frames_where_the_partition_does_not_close``, never averaged away.

    A frame that rendered but carries no connectivity block is listed in
    ``epochs_with_no_measurement`` rather than skipped: a summary taken over the
    frames that happened to report is a summary of a different sequence.
    """
    rows: list[dict] = []
    no_measurement: list[float] = []
    broken: list[dict] = []
    for frame in frames:
        spill = frame.get("spill") or {}
        conn = spill.get("connectivity") or {}
        drawn = spill.get("unobservable_drawn") or {}
        if not conn:
            if frame.get("png"):
                no_measurement.append(frame.get("epoch_ka"))
            continue
        connected = conn.get("connected") or {}
        unobs = conn.get("unobservable") or {}
        closed = conn.get("closed") or {}
        flat_cells = conn.get("flat_mask_cells")
        flat_components = conn.get("flat_components")
        cells_sum = sum(
            int(block.get("cells") or 0) for block in (connected, unobs, closed)
        )
        comp_sum = sum(
            int(block.get("components") or 0) for block in (connected, unobs, closed)
        )
        cells_close = flat_cells is not None and int(flat_cells) == cells_sum
        comps_close = flat_components is not None and int(flat_components) == comp_sum
        row = {
            "epoch_ka": frame.get("epoch_ka"),
            "flat_mask_cells": flat_cells,
            "flat_components": flat_components,
            "connected_cells": connected.get("cells"),
            "connected_components": connected.get("components"),
            "unobservable_cells": unobs.get("cells"),
            "unobservable_km2": unobs.get("km2"),
            "unobservable_components": unobs.get("components"),
            "unobservable_policy": unobs.get("policy"),
            "unobservable_by_edge_cells": {
                edge: block.get("cells")
                for edge, block in sorted((unobs.get("by_edge") or {}).items())
            },
            "unobservable_bodies": [
                {
                    "cells": body.get("cells"),
                    "km2": body.get("km2"),
                    "touched_edges": body.get("touched_edges"),
                    "surface_min_m": body.get("surface_min_m"),
                    "surface_max_m": body.get("surface_max_m"),
                    "painted_at_basin": body.get("painted_at_basin"),
                    "lon": body.get("lon"),
                    "lat": body.get("lat"),
                }
                for body in (unobs.get("bodies") or [])
            ],
            "closed_cells": closed.get("cells"),
            "closed_km2": closed.get("km2"),
            "closed_components": closed.get("components"),
            "cells_drawn_as_unobservable": drawn.get("cells"),
            "cells_classified_as_unobservable": drawn.get("cells_classified"),
            "cells_the_morphology_took_from_the_third_state": (
                None
                if drawn.get("cells") is None or drawn.get("cells_classified") is None
                else int(drawn["cells_classified"]) - int(drawn["cells"])
            ),
            "stripe_period_px": drawn.get("stripe_period_px"),
            "partition_closes_on_cells": cells_close,
            "partition_closes_on_components": comps_close,
        }
        rows.append(row)
        if not (cells_close and comps_close):
            broken.append(
                {
                    "epoch_ka": row["epoch_ka"],
                    "flat_mask_cells": flat_cells,
                    "connected_plus_unobservable_plus_closed_cells": cells_sum,
                    "flat_components": flat_components,
                    "connected_plus_unobservable_plus_closed_components": comp_sum,
                }
            )

    def _extreme(key: str, how) -> Optional[dict]:
        have = [r for r in rows if r.get(key) is not None]
        if not have:
            return None
        pick = how(have, key=lambda r: r[key])
        return {"epoch_ka": pick["epoch_ka"], key: pick[key]}

    return {
        "rule": CONNECTIVITY_STATES_RULE,
        "why_the_counts_are_not_summed": CONNECTIVITY_NO_SUM_RULE,
        "n_frames_with_a_measurement": len(rows),
        "epochs_with_no_measurement": no_measurement,
        "per_epoch": rows,
        "unobservable_cells_fewest": _extreme("unobservable_cells", min),
        "unobservable_cells_most": _extreme("unobservable_cells", max),
        "unobservable_km2_most": _extreme("unobservable_km2", max),
        "closed_cells_most": _extreme("closed_cells", max),
        "edges_reached_on_any_frame": sorted(
            {
                edge
                for row in rows
                for edge in (row["unobservable_by_edge_cells"] or {})
            }
        ),
        "policy_on_every_frame": sorted(
            {row["unobservable_policy"] for row in rows if row["unobservable_policy"]}
        ),
        "partition_is_exact_on_every_frame": bool(rows) and not broken,
        "frames_where_the_partition_does_not_close": broken,
        "where_the_full_record_is": (
            "frames[].spill.connectivity carries every body of both states with its "
            "cells, area, the edges it touches and the surface it sits under; "
            "frames[].spill.unobservable_drawn carries what the figure actually "
            "painted in the reserved colour and the stripe period it used"
        ),
    }


def _apply_computed_blocks(manifest: dict, frames: Sequence[dict]) -> dict:
    """The blocks whose every number must come off THIS run, applied to a manifest.

    Called by :func:`build_manifest` and again by :func:`write_subsequence`, which
    starts from a copy of the source run's manifest -- so a sub-sequence never
    inherits a note whose numbers were summarised over a different set of frames.
    """
    corridor = str(manifest.get("corridor_requested") or "step")
    manifest["corridor_note"] = corridor_note(frames, corridor)
    manifest["ramp_legality"] = ramp_legality(frames, corridor)
    manifest["ramp_claim"] = ramp_claim_summary(frames)
    manifest["sill_expectations"] = sill_expectation_summary(frames)
    manifest["connectivity_states"] = unobservable_summary(frames)
    manifest["offset_cross_check"] = offset_cross_check()
    manifest["paths"] = paths_block(manifest)
    return manifest


def build_manifest(
    plans: Sequence[EpochPlan],
    records: Sequence[dict],
    padding: Sequence[dict],
    limits: Optional[SequenceLimits],
    animation: dict,
    *,
    config: RenderConfig,
    canvas: tuple[int, int],
    canvas_from: str,
    duration_ms: int,
    sill_tolerance_m: float,
    extrema: Optional[dict] = None,
    palette: Optional[dict] = None,
    grid_step_ka: float = FINE_STEP_KA,
    companion: Optional[str] = None,
    clipping: Optional[dict] = None,
    corridor: str = "step",
    inset: Optional[InsetExtent] = None,
    inset_plans: Sequence[InsetPlan] = (),
    limits_full_only: Optional[SequenceLimits] = None,
    clipping_inset: Optional[dict] = None,
    panels: str = "both",
) -> dict:
    """The file any single frame can be audited from, without re-running anything.

    Per frame: the epoch, the three basin levels and where each came from, the sill
    expectation / measured saddle / VERDICT per pair, the clipped-cell counts, the
    raster sha256 AND the canonical path that sha256 belongs to.  Plus, once for the
    sequence: the two level-curve CSVs the water surfaces are read from, path and
    sha256, and the one colour table every frame is drawn through.
    """
    frames: list[dict] = []
    for index, plan in enumerate(plans):
        record = records[index] if index < len(records) else {}
        pad = padding[index] if index < len(padding) else None
        basins = record.get("basins_sidecar") or {}
        frame_palette = record.get("palette_sidecar") or {}
        frames.append(
            {
                "slot": index,
                "epoch_ka": plan.epoch_ka,
                "played_for_ms": int(duration_ms),
                "threshold": (
                    None
                    if record.get("refusal") or plan.refusal
                    else {k: v.level_m for k, v in sorted(plan.levels.items())}
                ),
                "refusal": record.get("refusal") or plan.refusal,
                "raster": {
                    # ``path_as_read`` is where THIS run opened the file, which may
                    # be a mirror; ``source_path`` is where the file LIVES.  The two
                    # are separate keys because the 2026-08-03 manifest recorded only
                    # the first, under %TEMP%, and every fingerprint in it therefore
                    # pointed at a directory designed to be discarded.
                    "path_as_read": plan.raster_path,
                    "source_path": (plan.raster_source or {}).get("source_path"),
                    "source_resolution": plan.raster_source,
                    "sha256": plan.raster_sha256,
                    "vertical_frame": plan.dem_frame,
                },
                "basin_levels": {k: v.to_dict() for k, v in plan.levels.items()},
                "seeds": [s.to_dict() for s in plan.seeds],
                "dz_bosphorus_m": plan.dz_bosphorus_m,
                "sill": _sill_block(
                    plan, basins, tolerance_m=sill_tolerance_m,
                    frame_refusal=record.get("refusal") or plan.refusal,
                    frame_rendered=bool(record.get("png")),
                ),
                "spill": {
                    "spilling": basins.get("spilling"),
                    "connected_groups": basins.get("connected_groups"),
                    "corridor": basins.get("corridor"),
                    "flood_ceiling_m": basins.get("flood_ceiling_m"),
                    "basins": basins.get("basins"),
                    "painted_water": basins.get("painted_water"),
                    "spill_not_drawn": basins.get("spill_not_drawn"),
                    # THE THIRD STATE, carried up out of the sidecar.  Until
                    # 2026-08-04 this block held ``painted_water`` and nothing at all
                    # about what was NOT painted, so a reader auditing a frame from
                    # the manifest -- which is the one thing this file exists for --
                    # could not learn that some of its cells are drawn as neither
                    # water nor land, nor separate the cells this raster measures to
                    # be CLOSED from the cells whose connectivity it cannot observe.
                    # The renderer measured both and the sidecar recorded both; the
                    # manifest simply did not pick them up, and a state that is absent
                    # from the audit surface is indistinguishable from a state that
                    # does not exist.
                    "connectivity": basins.get("connectivity"),
                    "unobservable_drawn": basins.get("unobservable_drawn"),
                    # The de-speckle's own counts, carried up for two readers:
                    # ramp_new_wet_ground bounds the flat-frame residue with
                    # cells_added, and inset_edge_erosion compares cells_removed
                    # between the two panels.  Both were reading a field the
                    # manifest did not carry until 2026-08-04.
                    "morphology": basins.get("morphology"),
                    # ... and the grid those counts were measured on, without which
                    # cells_removed cannot be turned into a per-border-cell rate and
                    # the two panels' de-speckles cannot be compared at all.
                    "grid": basins.get("grid"),
                },
                # HOW BIG THE CLAIM IS, per frame, computed at write time from the
                # blocks above.  See ramp_new_wet_ground: it is NOT
                # painted_water.corridor_cells_restored_by_the_or, and the block
                # says so beside the ratio between them.
                "ramp_new_wet_ground": ramp_new_wet_ground(
                    {
                        "spill": {
                            "corridor": basins.get("corridor"),
                            "connectivity": basins.get("connectivity"),
                            "painted_water": basins.get("painted_water"),
                            "morphology": basins.get("morphology"),
                        }
                    }
                ),
                "clipping": {
                    "out_of_range": frame_palette.get("out_of_range"),
                    "saturated_at_ramp_end": frame_palette.get(
                        "saturated_at_ramp_end"
                    ),
                    "clip_policy": frame_palette.get("clip_policy"),
                },
                "png": record.get("png"),
                "composed_png": record.get("composed"),
                "compose": record.get("compose"),
                # The added disclosure, derived once at render time from this frame's
                # own crest, levels and verdict, and DRAWN on the composed canvas
                # from the same string.  Its template carries no digit; the flag
                # beside it is checked rather than claimed.
                "disclosure_addendum": record.get("addendum"),
                "inset": _inset_frame_block(
                    next(
                        (
                            i for i in inset_plans
                            if float(i.epoch_ka) == float(plan.epoch_ka)
                        ),
                        None,
                    ),
                    record.get("inset"),
                    tolerance_m=sill_tolerance_m,
                ),
                "canvas_frame": (pad or {}).get("canvas"),
                "canvas_frame_sha256": (pad or {}).get("canvas_sha256"),
                "canvas_padding": pad,
                "analytic_shade_ceiling": record.get("shade_excursion"),
                "render_wall_s": record.get("wall_s"),
            }
        )

    epochs = [p.epoch_ka for p in plans]
    rendered_epochs = [
        r["epoch_ka"] for r in records if r.get("png")
    ]
    grid = grid_description(
        epochs, step_ka=grid_step_ka, companion=companion,
        n_rendered=len(rendered_epochs),
    )
    manifest = {
        "what": (
            "The Khvalynian (Caspian) transgression spilling west through the Manych "
            f"into the Pontic, 17.00-13.00 ka, on the {grid_step_ka:g} ka grid: "
            f"{grid['n_slots']} slot(s) selected out of bake_khvalynian.EPOCHS_KA, "
            f"{len(rendered_epochs)} of them rendered, three declared basins, one "
            "pinned colour mapping. See 'grid' for what this sequence does not carry "
            "and why."
        ),
        "grid": grid,
        "generated_by": "scripts/animate_khvalynian_spill.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epochs": [p.epoch_ka for p in plans],
        "n_frames": len(plans),
        "n_rendered": sum(1 for r in records if r.get("png")),
        "n_refused": sum(1 for f in frames if f["refusal"]),
        "scenario": {
            "caspian": "defensible (the reliably dated central line, peaking at +35 m)",
            "pontic": "gradual",
            "mediterranean": "F3 = -offset(t); no scenario, it is the engine's own datum",
        },
        "level_curves": level_curve_provenance(),
        "raster_source_resolution": _raster_source_summary(plans),
        "render_config": {
            "max_px": config.max_px,
            "dpi": config.dpi,
            "fig_size_in": config.fig_size_in,
            "vert_exag": config.vert_exag,
            "land_elev_percentile_lo": config.land_elev_percentile_lo,
            "land_elev_percentile_hi": config.land_elev_percentile_hi,
            "clip_policy": config.clip_policy,
            "ice_model_label": config.ice_model_label,
            "sill_tolerance_m": sill_tolerance_m,
            "figure_is_the_only_lever_on_the_legend": FIGURE_LEVER_RULE,
            "untrimmed_figure_px": list(fallback_canvas_for(config.fig_size_in,
                                                            config.dpi)),
        },
        "panels_composed": {
            "panels": str(panels),
            "rule": CLOSEUP_COMPOSE_RULE if panels == "inset" else COMPOSE_RULE,
        },
        "seed_registry": {
            "rule": SEED_RULE,
            "boxes": [b.to_dict() for b in SEED_BOXES],
            "why_it_is_load_bearing": (
                "B1 measured the caspian<->pontic saddle to be seed-INDEPENDENT to "
                "the last float64 bit over 1 953 seed pairs "
                "(+28.000944137573242 m at 44.110417 E / 45.735417 N, spread 0.00), "
                "so the VERDICT is safe against a bad seed. The PAINTED CORRIDOR is "
                "not: moving the Pontic seed inside the same connected component "
                "changed it by 2 383 005 cells, a factor of 4.68. The registry is "
                "what makes the drawn water reproducible."
            ),
        },
        "sequence_limits": limits.to_dict() if limits is not None else None,
        "sequence_limits_cover": _limits_cover(limits, plans, records),
        "sequence_limits_extrema_by_epoch": extrema,
        "corridor_requested": str(corridor),
        "inset": _inset_block(
            inset, inset_plans, frames, limits, limits_full_only, clipping_inset
        ),
        "analytic_shade_ceiling": clipping or None,
        "canvas": {
            "size_px": list(canvas),
            "derived_from": canvas_from,
            "why": (
                "render_paleocostline_epoch saves with bbox_inches='tight', so its "
                "output size follows the widest line of text in the figure. That "
                "file is not writable from here, so the frames are padded to one "
                "canvas instead -- centred, never scaled, never cropped, on the "
                "figure's own white background. Per-frame padding is recorded so "
                "'nothing moved' is a number."
            ),
        },
        "palette": palette or None,
        "sill_verdicts": _sill_verdict_summary(frames),
        "animation": animation,
        "frames": frames,
    }
    # corridor_note / sill_expectations / offset_cross_check / paths: every number
    # in them is summarised over THIS run's own frames or recomputed here, which is
    # why they are applied rather than written above.
    return _apply_computed_blocks(manifest, frames)


def _inset_frame_block(
    plan: Optional[InsetPlan],
    record: Optional[dict],
    *,
    tolerance_m: float,
) -> Optional[dict]:
    """One frame's inset, as values: the crop, the seeds, the saddle, the verdict.

    ``None`` means this run asked for no inset at all -- which is a different claim
    from an inset that was asked for and refused, and that one carries its refusal.
    """
    if plan is None:
        return None
    record = record or {}
    sidecar = record.get("basins_sidecar") or {}
    entry = next(
        (
            s for s in (sidecar.get("sills") or [])
            if sorted(s.get("pair") or []) == ["caspian", "pontic"]
        ),
        {},
    )
    measured = entry.get("elevation_m")
    expected = None
    if plan.dem_frame is not None:
        try:
            expected = float(
                manych_expectation(plan.epoch_ka, plan.dem_frame).value_m
            )
        except Exception:  # noqa: BLE001 - an unresolvable expectation is a null here
            expected = None
    gap = None if measured is None or expected is None else float(measured) - expected
    return {
        "extent": plan.extent.to_dict(),
        "crop": plan.crop,
        "raster_path": plan.raster_path,
        "raster_sha256": plan.raster_sha256,
        "dem_vertical_frame": plan.dem_frame,
        "seeds": [s.to_dict() for s in plan.seeds],
        "refusal": plan.refusal or record.get("refusal"),
        "png": record.get("png"),
        "render_wall_s": record.get("wall_s"),
        "colour_mapping": record.get("colour_mapping"),
        "extent_check": record.get("extent_check"),
        # The object the composed caption's erosion sentence was drawn from, so the
        # picture and the record cannot come to disagree about it.
        "edge_erosion": record.get("edge_erosion"),
        # Likewise for the legend sentence.  This key was MISSING from the first
        # close-up build's manifest while render_config.figure_is_the_only_lever_on
        # _the_legend told a reader to look for it here -- a pointer to nothing, in
        # a file whose whole purpose is that a frame can be audited from it.  The
        # numbers were in frames[].compose.caption the whole time; that is where a
        # reader had to go, without being told.
        "legend_occlusion": record.get("legend_occlusion"),
        "sill": {
            "pair": ["caspian", "pontic"],
            "expected_m": expected,
            "measured_saddle_m": measured,
            "measured_lonlat": (
                None if entry.get("lon") is None
                else [entry.get("lon"), entry.get("lat")]
            ),
            "tolerance_m": float(tolerance_m),
            "gap_m": gap,
            "verdict": (
                "NOT_MEASURED_ON_THIS_GRID" if measured is None
                else "WITHIN_TOLERANCE" if abs(gap) <= float(tolerance_m)
                else "OUTSIDE_TOLERANCE"
            ),
            "measured_on": entry.get("measured_on"),
            "expectation_status": entry.get("expectation_status"),
            "smoothing_shift_m": entry.get("smoothing_shift_m"),
            "route_cells": entry.get("route_cells"),
            "why_the_inset_is_guarded_separately": INSET_GUARD_TEMPLATE.format(
                expected=_fmt(expected, "+.4f"),
                measured=_fmt(measured, "+.4f"),
                gap=_fmt(gap, "+.4f"),
                tolerance=_fmt(float(tolerance_m), ".2f"),
                grid=(
                    entry.get("measured_on")
                    or "no grid recorded for this measurement"
                ),
            ),
        },
        "spill": {
            "spilling": sidecar.get("spilling"),
            "connected_groups": sidecar.get("connected_groups"),
            "corridor": sidecar.get("corridor"),
            "painted_water": sidecar.get("painted_water"),
            "spill_not_drawn": sidecar.get("spill_not_drawn"),
            "connectivity": sidecar.get("connectivity"),
            "morphology": sidecar.get("morphology"),
        },
        # The same quantity the full frame reports, on the inset's own grid.
        "ramp_new_wet_ground": ramp_new_wet_ground(
            {
                "spill": {
                    "corridor": sidecar.get("corridor"),
                    "connectivity": sidecar.get("connectivity"),
                    "painted_water": sidecar.get("painted_water"),
                    "morphology": sidecar.get("morphology"),
                }
            }
        ),
        "clipping": {
            "out_of_range": (record.get("palette_sidecar") or {}).get("out_of_range"),
            "saturated_at_ramp_end": (record.get("palette_sidecar") or {}).get(
                "saturated_at_ramp_end"
            ),
            "clip_policy": (record.get("palette_sidecar") or {}).get("clip_policy"),
        },
    }


def _limits_delta(
    limits: Optional[SequenceLimits], full_only: Optional[SequenceLimits]
) -> dict:
    """What including the inset raster in the limits pass moved, per field."""
    if limits is None or full_only is None:
        return {
            "measured": False,
            "why": (
                "no second pass was run: either the sequence produced no limits, or "
                "this run has no inset and there is nothing to compare"
            ),
        }
    rows = []
    for name in _COLOUR_FIELDS:
        with_inset, without = getattr(limits, name), getattr(full_only, name)
        if with_inset != without:
            rows.append(
                {
                    "field": name,
                    "full_extent_rasters_only": (
                        None if without is None else float(without)
                    ),
                    "with_the_inset_raster": (
                        None if with_inset is None else float(with_inset)
                    ),
                }
            )
    envelope_moved = [
        r["field"] for r in rows
        if r["field"] in ("dem_data_min_m", "dem_data_max_m", "depth_data_max_m")
    ]
    return {
        "measured": True,
        "fields_that_moved": rows,
        "n_fields_that_moved": len(rows),
        "true_envelope_moved": envelope_moved,
        "what_this_should_look_like": (
            "the crop holds a strict SUBSET of the source raster's cells, proved "
            "cell for cell by crop_to_extent, so dem_data_min_m / dem_data_max_m / "
            "depth_data_max_m -- the TRUE extrema -- cannot move, and "
            "true_envelope_moved must be empty. The percentile ramp endpoints CAN "
            "move, and only outwards, because the aggregation is a min/max over "
            "rasters: a sub-extent whose land is lower or higher than the whole "
            "scene's 2nd/98th percentile widens the ramp. A widened ramp only ever "
            "gives a frame MORE headroom, never less"
        ),
    }


def _inset_block(
    inset: Optional[InsetExtent],
    inset_plans: Sequence[InsetPlan],
    frames: Sequence[dict],
    limits: Optional[SequenceLimits],
    limits_full_only: Optional[SequenceLimits],
    clipping_inset: Optional[dict] = None,
) -> Optional[dict]:
    """The sequence-level inset block: the extent, and what it measured everywhere."""
    if inset is None:
        return None
    per_epoch = []
    erosion_rows: list[dict] = []
    for frame in frames:
        block = frame.get("inset")
        if block is None:
            continue
        if block.get("png"):
            # The object the caption was drawn from, when the run measured one;
            # otherwise re-measured here from the same fields.  Never two readings.
            measured = block.get("edge_erosion") or inset_edge_erosion(
                {
                    "morphology": (frame.get("spill") or {}).get("morphology"),
                    "grid": (frame.get("spill") or {}).get("grid"),
                },
                {"morphology": ((block.get("spill") or {}).get("morphology"))},
                block.get("crop"),
            )
            erosion_rows.append({"epoch_ka": frame.get("epoch_ka"), **measured})
        per_epoch.append(
            {
                "epoch_ka": frame.get("epoch_ka"),
                "rendered": bool(block.get("png")),
                "refusal": block.get("refusal"),
                "ramp_new_wet_ground_cells": (
                    (block.get("ramp_new_wet_ground") or {}).get(
                        "cells_the_ramp_wets_that_a_flat_surface_leaves_dry"
                    )
                ),
                "measured_saddle_m": (block.get("sill") or {}).get(
                    "measured_saddle_m"
                ),
                "expected_m": (block.get("sill") or {}).get("expected_m"),
                "gap_m": (block.get("sill") or {}).get("gap_m"),
                "verdict": (block.get("sill") or {}).get("verdict"),
                "crest_lonlat": (block.get("sill") or {}).get("measured_lonlat"),
                "extent_verdict": (block.get("extent_check") or {}).get("verdict"),
                "colour_mapping_verdict": (block.get("colour_mapping") or {}).get(
                    "verdict"
                ),
                "spilling": (block.get("spill") or {}).get("spilling"),
                "basins_joined": (
                    (block.get("spill") or {}).get("painted_water") or {}
                ).get("basins_joined"),
            }
        )
    verdicts = {row["verdict"] for row in per_epoch if row["verdict"]}
    return {
        "extent": inset.to_dict(),
        "n_epochs_planned": len(inset_plans),
        "n_epochs_rendered": sum(1 for r in per_epoch if r["rendered"]),
        "per_epoch": per_epoch,
        "sill_verdicts_seen": sorted(verdicts),
        "verdict": (
            "EVERY_INSET_FRAME_RESOLVES_ITS_OWN_SILL"
            if per_epoch and verdicts == {"WITHIN_TOLERANCE"}
            else "NOT_EVERY_INSET_FRAME_RESOLVES_ITS_OWN_SILL"
        ),
        "colour_mapping": {
            "rule": (
                "the inset consumes the SEQUENCE's SequenceLimits object, not one of "
                "its own. The two panels are composed into one canvas and quantized "
                "into one 256-entry table, so two mappings would recolour the same "
                "elevation differently in two halves of one picture. The inset "
                "raster is passed through the limits PASS as well, in the same "
                "SequenceEpoch as the full raster, because SequenceLimits refuses to "
                "be consumed by a frame built from a raster it never read -- and "
                "appending a fingerprint the pass never opened would be a false "
                "statement in the one object whose purpose is provenance"
            ),
            "per_frame_check": (
                "frames[].inset.colour_mapping, asserted on the RenderConfig object "
                "handed to the renderer by assert_one_colour_mapping"
            ),
            "cost_of_including_the_inset_in_the_limits_pass": _limits_delta(
                limits, limits_full_only
            ),
        },
        # WHAT THE PANEL'S OWN EDGES COST ITS WATER.  The figure title says the two
        # panels share their cells and their colour mapping; it cannot also be read
        # as saying they draw the same shoreline, and this block is the measurement
        # that keeps a reader from doing so.
        "edge_erosion": {
            "per_epoch": erosion_rows,
            "n_epochs_measured": len(erosion_rows),
            "note": (
                "one row per rendered inset panel, each measured on that panel's own "
                "de-speckle counts and its own crop window. The counts are NEVER "
                "summed across epochs: the same ground appears in every frame"
            ),
        },
        "clipping": clipping_inset or None,
        "clipping_note": (
            "the SAME per-quantity audit the full frames get "
            "(assert_only_the_analytic_shade_ceiling_was_marked), run over the inset "
            "panels alone. The five MEASURED limit quantities must be zero on every "
            "inset panel too: the crop holds a subset of the source's cells, so a "
            "cell outside the sequence envelope would mean the mapping and the crop "
            "disagree about the same terrain. null = this run drew no inset panel"
        ),
    }


def _limits_cover(
    limits: Optional[SequenceLimits],
    plans: Sequence[EpochPlan],
    records: Sequence[dict],
) -> dict:
    """Which epochs the pinned mapping was measured over, against which it is USED on.

    THE DEFECT THIS CLOSES
    ----------------------
    The 2026-08-03 ``sequence_limits.json`` recorded ``epochs: [17.0, 16.5, 16.0,
    15.5, 15.0, 14.5, 14.0, 13.5, 13.0]`` -- nine -- while the sequence it was
    shipped beside had SEVENTEEN slots, and the file said nothing about the
    difference.  A reader comparing the two counts had no way to tell whether eight
    frames had been dropped, whether the limits were stale, or whether the sequence
    held placeholders.

    Three sets, never conflated:

    * the slots of the sequence -- what plays;
    * the epochs the limits were computed over -- the *renderable* plans, taken
      before any render;
    * the epochs that actually produced a frame.

    The second and third can differ: a plan can resolve every level and seed and then
    refuse inside the renderer, which is exactly what all nine did on 2026-08-03.
    When they do, the limits describe an epoch that is not drawn, which can only
    WIDEN the envelope the drawn frames are mapped through -- never clip a drawn
    cell -- and the direction is stated rather than left to be worked out.
    """
    slots = [float(p.epoch_ka) for p in plans]
    rendered = [float(r["epoch_ka"]) for r in records if r.get("png")]
    if limits is None:
        return {
            "n_slots_in_the_sequence": len(slots),
            "epochs_the_limits_were_computed_over": None,
            "epochs_that_produced_a_frame": rendered,
            "verdict": "NO_LIMITS: no epoch of this sequence was renderable",
        }
    measured = [float(e) for e in limits.epochs]
    in_limits_not_drawn = [e for e in measured if e not in rendered]
    drawn_not_in_limits = [e for e in rendered if e not in measured]
    slots_not_in_limits = [e for e in slots if e not in measured]
    if drawn_not_in_limits:
        verdict = (
            "DISAGREEMENT: a frame was drawn for an epoch the limits never measured, "
            "which the renderer's own assert_consumable_by should have refused"
        )
    elif in_limits_not_drawn:
        verdict = (
            "WIDER THAN DRAWN: the limits were measured over epochs that then "
            "produced no frame, so the envelope is an envelope of more rasters than "
            "the animation shows"
        )
    else:
        verdict = (
            "EXACT: the limits were computed over precisely the epochs that produced "
            "a frame"
        )
    return {
        "n_slots_in_the_sequence": len(slots),
        "slots": slots,
        "epochs_the_limits_were_computed_over": measured,
        "epochs_that_produced_a_frame": rendered,
        "slots_carrying_no_frame": slots_not_in_limits,
        "epochs_in_the_limits_that_produced_no_frame": in_limits_not_drawn,
        "epochs_drawn_that_the_limits_never_measured": drawn_not_in_limits,
        "verdict": verdict,
        "rule": (
            "the limits are computed over the RENDERABLE plans, before any render, "
            "because RenderConfig needs them to render. A slot that refuses at "
            "planning contributes nothing to them and holds a hatched placeholder "
            "instead of a frame, so a sequence with more slots than limit-epochs is "
            "correct and is not a dropped frame -- 'slots_carrying_no_frame' names "
            "every one of them. What would NOT be correct is silence about it, which "
            "is what the 2026-08-03 file shipped"
        ),
        "direction_of_the_error_when_it_is_not_EXACT": (
            "pinning over a superset can only widen a ramp, never clip a drawn cell, "
            "so a WIDER THAN DRAWN sequence loses contrast and never misreports an "
            "elevation. The reverse -- drawing an epoch the limits never saw -- would "
            "be unsound and is refused by SequenceLimits.assert_consumable_by before "
            "the frame is computed"
        ),
    }


def _raster_source_summary(plans: Sequence[EpochPlan]) -> dict:
    """How many rasters got a canonical source path back, and by what evidence."""
    statuses: dict[str, int] = {}
    unresolved: list[float] = []
    for plan in plans:
        if not plan.raster_path:
            continue
        status = (plan.raster_source or {}).get("status", "UNRESOLVED")
        statuses[status] = statuses.get(status, 0) + 1
        if (plan.raster_source or {}).get("source_path") is None:
            unresolved.append(plan.epoch_ka)
    return {
        "canonical_dir": str(CANONICAL_TOPO_DIR),
        "by_status": statuses,
        "epochs_with_no_source_path": unresolved,
        "rule": (
            "a source path is recorded only when it is established BY SHA256 -- the "
            "raster was read out of the canonical store, or a mirror provenance "
            "sidecar carries a record with this fingerprint, or the canonical store "
            "was re-hashed and agreed. A file of the same NAME is not accepted as "
            "evidence: this project re-bakes rasters in place under unchanged names, "
            "so a matching name is not a matching scene"
        ),
        "why_it_exists": (
            "the 2026-08-03 manifest recorded seventeen raster paths inside a "
            "per-session agent scratchpad under %TEMP% -- a directory designed to be "
            "discarded -- so every fingerprint in it was attached to a path that was "
            "dead on arrival"
        ),
    }


#: What a sill verdict can say, as VALUES rather than as an inference from which
#: exception a run did or did not raise.
SILL_VERDICT_RULE = (
    "WITHIN_TOLERANCE / OUTSIDE_TOLERANCE compare |measured_saddle_m - expected_m| "
    "against tolerance_m, both read out of this frame's own artifacts. "
    "NOT_MEASURED means the frame produced no saddle for this pair -- either the "
    "frame did not render, or the saddle stands above the flood ceiling and the "
    "sidecar records null, which is 'not reached' and never 'agrees'. "
    "NOT_CHECKED means the pair is a declared UnmeasuredSill: no expectation was "
    "armed, on purpose, with the reason carried beside it. "
    "There is no verdict that is inferred from an exception type: until 2026-08-04 "
    "'the Manych guard passes' was read off the ABSENCE of a SillExpectationError, "
    "which is compatible with the guard never having run"
)


def _sill_verdict_summary(frames: Sequence[dict]) -> dict:
    """Per pair, the verdict tally over the sequence -- the aggregate as values."""
    tally: dict[str, dict[str, int]] = {}
    for frame in frames:
        for entry in frame.get("sill") or []:
            key = "|".join(entry.get("pair") or [])
            verdict = str(entry.get("verdict"))
            tally.setdefault(key, {})
            tally[key][verdict] = tally[key].get(verdict, 0) + 1
    return {
        "rule": SILL_VERDICT_RULE,
        "by_pair": tally,
        "n_frames": len(frames),
    }


def _sill_verdict(
    expectation: Any,
    expected_m: Optional[float],
    measured_m: Optional[float],
    tolerance_m: float,
    frame_refusal: Optional[str],
    frame_rendered: bool,
) -> tuple[str, Optional[float], str]:
    """``(verdict, gap_m, why)`` -- the three values a reader needs, computed here.

    The gap is ``measured - expected`` and is ``None`` when either side is missing.
    It is never filled in from a neighbouring epoch, a nominal constant, or the
    absence of an exception.

    ``frame_rendered`` is separate from ``frame_refusal`` on purpose: a frame can
    have no saddle for three DIFFERENT reasons and they must not be collapsed --
    the frame refused, no render was attempted at all (``--plan-only``), or the
    frame rendered and the saddle stands above the flood ceiling.  Only the third
    is a statement about the terrain.
    """
    if not isinstance(expectation, SillExpectation):
        return (
            "NOT_CHECKED",
            None,
            "the pair is a declared UnmeasuredSill; no expectation was armed and the "
            "reason is carried in unmeasured_reason",
        )
    if expected_m is None:
        return (
            "NOT_MEASURED",
            None,
            "the expectation carries no value_m, so nothing could be compared",
        )
    if measured_m is None:
        if frame_refusal:
            return (
                "NOT_MEASURED",
                None,
                "this frame did not render, so no saddle was measured on it. The "
                f"expectation stands at {float(expected_m):+.3f} m and is recorded "
                "beside this verdict; the refusal is in the frame's 'refusal' field. "
                "An unrun guard is NOT a passed guard",
            )
        if not frame_rendered:
            return (
                "NOT_MEASURED",
                None,
                "no render was attempted for this frame -- this is a plan-only "
                "manifest -- so no saddle exists to compare. The expectation stands "
                f"at {float(expected_m):+.3f} m. An unrun guard is NOT a passed guard",
            )
        return (
            "NOT_MEASURED",
            None,
            "the frame rendered but records no saddle for this pair -- the sidecar's "
            "elevation_m is null, which means the saddle stands above the flood "
            "ceiling and was not reached. 'Not reached' is not 'agrees'",
        )
    gap = float(measured_m) - float(expected_m)
    within = abs(gap) <= float(tolerance_m)
    return (
        "WITHIN_TOLERANCE" if within else "OUTSIDE_TOLERANCE",
        gap,
        f"|{float(measured_m):+.3f} - {float(expected_m):+.3f}| = {abs(gap):.3f} m "
        f"{'<=' if within else '>'} {float(tolerance_m):.2f} m",
    )


def _sill_block(
    plan: EpochPlan,
    basins: dict,
    *,
    tolerance_m: float,
    frame_refusal: Optional[str] = None,
    frame_rendered: bool = False,
) -> list[dict]:
    """Per pair: what was expected, what this grid measured, and the VERDICT.

    Every one of the three is a value in the record.  Until 2026-08-04 no sill
    verdict was written anywhere in the deliverables at all: the expectation was
    recorded, the measured saddle was recorded when a frame rendered, and whether
    the two agreed had to be inferred -- in practice from the fact that no
    ``SillExpectationError`` had been raised, which is equally consistent with the
    guard never having run.  Eight of the seventeen frames did not even carry the
    expectation, because ``dz_bosphorus_m`` was resolved after the levels and so was
    ``None`` on every epoch that refused; see :func:`plan_epoch`.
    """
    out: list[dict] = []
    measured: dict[str, dict] = {}
    for entry in basins.get("sills", []) or []:
        pair = entry.get("pair")
        key = "|".join(sorted(pair)) if isinstance(pair, (list, tuple)) else str(pair)
        measured[key] = entry
    if plan.dem_frame is None or plan.dz_bosphorus_m is None:
        expectations: dict[tuple[str, str], Any] = {}
    else:
        expectations = sill_expectations_for(
            plan.epoch_ka, plan.dem_frame, plan.dz_bosphorus_m
        )
    for pair, expectation in sorted(expectations.items()):
        key = "|".join(sorted(pair))
        entry = measured.get(key, {})
        expected_m = getattr(expectation, "value_m", None)
        measured_m = entry.get("elevation_m")
        verdict, gap_m, why = _sill_verdict(
            expectation, expected_m, measured_m, tolerance_m, frame_refusal,
            frame_rendered,
        )
        out.append(
            {
                "pair": sorted(pair),
                "expectation_kind": type(expectation).__name__,
                "expected_m": expected_m,
                "tolerance_m": float(tolerance_m),
                "gap_m": gap_m,
                "verdict": verdict,
                "verdict_why": why,
                "expectation_source": getattr(expectation, "source", None),
                "expectation_measured_by": getattr(expectation, "measured_by", None),
                "expectation_lonlat": (
                    [expectation.lon, expectation.lat]
                    if isinstance(expectation, SillExpectation)
                    else None
                ),
                "unmeasured_reason": getattr(expectation, "reason", None),
                # Where the smoothing shift ACTUALLY is for THIS frame, counted
                # rather than claimed.  The Bosphorus reason used to assert the
                # number was "disclosed twice"; on a refused frame there is no
                # sidecar, so it is disclosed once, and the count now comes from the
                # artifacts instead of from a sentence.
                "smoothing_shift_disclosed_in": (
                    ["unmeasured_reason", "basins_sidecar.sills[].smoothing_shift_m"]
                    if entry.get("smoothing_shift_m") is not None
                    and getattr(expectation, "reason", None)
                    else ["unmeasured_reason"]
                    if getattr(expectation, "reason", None)
                    else (
                        ["basins_sidecar.sills[].smoothing_shift_m"]
                        if entry.get("smoothing_shift_m") is not None
                        else []
                    )
                ),
                # ``elevation_m`` is null in the sidecar when the saddle stands above
                # the flood ceiling; that is "not reached", never "agrees".
                "measured_saddle_m": measured_m,
                "measured_lonlat": (
                    None
                    if entry.get("lon") is None
                    else [entry.get("lon"), entry.get("lat")]
                ),
                "measured_on": entry.get("measured_on"),
                # The renderer's own sentence, copied verbatim from
                # <frame>.basins.json's sills[].expectation_status. Named for where
                # it comes from: 'status' alone read like a verdict this module had
                # computed, and it is null on every frame that did not render.
                "sidecar_expectation_status": entry.get("expectation_status"),
                "smoothing_shift_m": entry.get("smoothing_shift_m"),
                "route_cells": entry.get("route_cells"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the 17-frame Khvalynian spill animation.",
    )
    parser.add_argument("--out-root", default=str(ANIM_ROOT))
    parser.add_argument("--topo-dir", default=str(KHV_TOPO_DIR))
    parser.add_argument("--max-px", type=int, default=NATIVE_MAX_PX)
    parser.add_argument("--duration-ms", type=int, default=FRAME_DURATION_MS)
    parser.add_argument(
        "--epochs",
        default="",
        help=(
            "comma-separated epochs to build instead of the whole 13-17 ka window. "
            "For probes only: a subset is not the sequence, and its pinned colour "
            "limits are computed over the subset, so its frames are NOT comparable "
            "with the full run's."
        ),
    )
    parser.add_argument(
        "--grid-step-ka",
        type=float,
        default=FINE_STEP_KA,
        help=(
            f"which grid to build, SELECTED out of bake_khvalynian.EPOCHS_KA. "
            f"{FINE_STEP_KA} (default) is the owner's request, 17 slots, 8 of which "
            f"can carry no Pontic level and become refusal frames. {HALF_KA_STEP} is "
            "the 9 epochs that all resolve. Both are shipped and each manifest names "
            "the other."
        ),
    )
    parser.add_argument(
        "--subsequence-of",
        default=None,
        help=(
            "build this grid by RE-ENCODING a finished run's own canvas frames "
            "instead of rendering them again. Each frame is re-hashed against the "
            "sha256 that run recorded before it is used; a mismatch refuses. Use "
            "with --grid-step-ka."
        ),
    )
    parser.add_argument(
        "--companion",
        default=None,
        help="path of the other grid's manifest, recorded in this one's 'grid' block",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="resolve levels, seeds and refusals; render nothing",
    )
    parser.add_argument(
        "--verify-canonical-sha256",
        action="store_true",
        help=(
            "re-hash the canonical raster store to put a source path back on a "
            "raster read from a mirror that left no provenance sidecar. Costs a full "
            "read of every raster (17 x 42 MB, measured at 0.4-3.4 MB/s over SMB). "
            "Without it and without a sidecar the manifest records UNRESOLVED and "
            "what would settle it; it never guesses from a file name."
        ),
    )
    parser.add_argument(
        "--corridor",
        choices=("step", "ramp"),
        default="step",
        help=(
            "how the cells BETWEEN two connected basins are drawn. 'step' (default, "
            "and what the shipped sequence uses) draws every basin flat at its own "
            "declared level and only MARKS the corridor. 'ramp' interpolates the "
            "surface linearly in along-channel graph distance between the two "
            "declared levels and DRAWS it -- a first-order water-surface profile, "
            "not hydraulics. There is no fallback: if any epoch has two "
            "donor->receiver pairs, ramp refuses and names the epoch."
        ),
    )
    parser.add_argument(
        "--inset",
        choices=("none", *(e.name for e in INSET_EXTENTS)),
        default="none",
        help=(
            "compose a second, tighter render into the same canvas as each frame. "
            "Every choice is a REGISTERED extent -- its own seed boxes, its own "
            "per-bound reason and its own sill guard on its own grid -- and both "
            "panels share ONE SequenceLimits. "
            + " ".join(
                f"{e.name!r}: {e.lon_min:g}-{e.lon_max:g} E / "
                f"{e.lat_min:g}-{e.lat_max:g} N "
                f"({e.lon_max - e.lon_min:g} x {e.lat_max - e.lat_min:g} deg)."
                for e in INSET_EXTENTS
            )
        ),
    )
    parser.add_argument(
        "--panels",
        choices=("both", "inset"),
        default="both",
        help=(
            "which panels reach the composed page. 'both' (default, and what the "
            "shipped sequences use) puts the whole scene above the inset. 'inset' is "
            "the CLOSE-UP build: the corridor panel alone, with the whole-scene frame "
            "still rendered, hashed and kept under frames/full so every verdict, "
            "sill, seed and clip audit in the manifest is still that panel's. The "
            "omitted panel is NOT shrunk into a locator -- resampling would blend "
            "palette-mapped pixels into colours no render produced, and the canvas is "
            "quantized as one image through one master palette. Requires --inset."
        ),
    )
    parser.add_argument(
        "--fig-size-in",
        type=float,
        default=None,
        help=(
            "the square figure side, inches. Default: the renderer's own 14.0. This "
            "is the ONLY lever this module has on the legend that sits inside the "
            "map, which is sized in points and so covers a fraction of the map that "
            "goes as 1/fig_size_in; it raises the map's pixel count by the same "
            "factor. Measured on the 16.5 ka corridor panel at 14.0 in the legend "
            "box is 21.67 per cent of the map and covers 27.68 per cent of the water "
            f"the panel draws; at {CLOSEUP_FIGURE_SIZE_IN:g} in it is 8.11 and 17.86 "
            "per cent of the same two things, and the map goes from 186.88 to 306.25 "
            "pixels per degree of longitude."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=(
            "output DPI. Default: the renderer's own 150. It buys pixels and NO "
            "relief on the legend -- text and map scale with dpi alike, so the ratio "
            "between them does not move. Use --fig-size-in for that."
        ),
    )
    parser.add_argument(
        "--i-know-this-grid-cannot-resolve-the-sill",
        action="store_true",
        help=(
            "required with --max-px below 7800. At 2000 px the caspian<->pontic "
            "saddle reads +36.01 m against the measured +27.70 m."
        ),
    )
    args = parser.parse_args(argv)

    if args.max_px < NATIVE_MAX_PX and not getattr(
        args, "i_know_this_grid_cannot_resolve_the_sill"
    ):
        parser.error(
            f"--max-px {args.max_px} is below the native {NATIVE_MAX_PX}. The "
            "two-basin frame is renderable only on the native grid: the "
            "caspian<->pontic saddle on dem_sm reads +36.005 m at 2000 px and "
            "+32.942 m at 4000 px against the independently measured +27.701 m, "
            "which is the difference between 'endorheic' and 'spilling'. This is a "
            "refusal, not a knob. Pass "
            "--i-know-this-grid-cannot-resolve-the-sill to build a PROBE (not a "
            "deliverable) anyway."
        )

    epochs = (
        [float(token) for token in args.epochs.split(",") if token.strip()]
        or None
    )

    if args.subsequence_of:
        manifest = write_subsequence(
            args.subsequence_of,
            args.out_root,
            grid_step_ka=args.grid_step_ka,
            duration_ms=args.duration_ms,
            keep=epochs,
        )
        print(
            f"\nsub-sequence at {args.grid_step_ka:g} ka: {manifest['n_frames']} "
            f"frames, all imagery, re-encoded from {args.subsequence_of} with every "
            "frame re-verified by sha256."
        )
        return 0

    if epochs is None:
        epochs = list(epoch_grid(FINE_WINDOW_KA, args.grid_step_ka))
    manifest = build_animation(
        out_root=args.out_root,
        topo_dir=args.topo_dir,
        max_px=args.max_px,
        duration_ms=args.duration_ms,
        epochs=epochs,
        grid_step_ka=args.grid_step_ka,
        companion=args.companion,
        plan_only=args.plan_only,
        verify_canonical_sha256=args.verify_canonical_sha256,
        corridor=args.corridor,
        inset=None if args.inset == "none" else inset_extent_by_name(args.inset),
        fig_size_in=args.fig_size_in,
        dpi=args.dpi,
        panels=args.panels,
    )
    if args.plan_only:
        planned = manifest["n_frames"] - manifest["n_refused"]
        print(
            f"\nplan only: {planned} of {manifest['n_frames']} epochs planned, "
            f"{manifest['n_refused']} refused before any render was attempted."
        )
    else:
        print(
            f"\n{manifest['n_rendered']} of {manifest['n_frames']} frames rendered; "
            f"{manifest['n_refused']} refused."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
