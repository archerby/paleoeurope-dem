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
from pathlib import Path

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
# Simplified curve: 0 ka -> 0 m (the full CSV has a +8.49 m present-datum offset
# and must NOT be used for the global-eustatic reference frames).
SPRATT_CSV = (
    _RAW_DIR / "sea_level_curves" / "spratt_lisiecki_2016_simplified.csv"
)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_filename(epoch: float) -> str:
    """ICE-6G filename token for an epoch.

    Integers use the bare int (``16`` -> ``...16.nc``); half-ka use one decimal
    (``0.5`` -> ``...0.5.nc``).  ``0`` -> ``...0.nc``.
    """
    if float(epoch).is_integer():
        return f"I6_C.VM5a_10min.{int(epoch)}.nc"
    return f"I6_C.VM5a_10min.{epoch}.nc"


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


def basin_level(
    df: pd.DataFrame,
    epoch: float,
    scenario: str = "gradual",
    basin: str = "pontic",
) -> float:
    """Nearest-age closed-basin level (m vs present global SL) for *epoch*.

    *scenario* selects the reconnection column (``gradual`` or ``catastrophic``).
    *basin* selects the datum-sanity window applied to the returned scalar.  That
    window is a blow-up guard only — the sign of the curve is validated once, at
    load time, by :func:`validate_basin_curve`.
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
    idx = (df["age_ka"] - float(epoch)).abs().idxmin()
    level = float(df.loc[idx, col])
    return _check_datum_sanity(level, basin, f"{float(epoch):g} ka")


def pontic_level(df: pd.DataFrame, epoch: float, scenario: str = "gradual") -> float:
    """Nearest-age **Pontic** basin level (m vs present global SL) for *epoch*.

    Thin wrapper over :func:`basin_level` pinned to the ``pontic`` range, kept for
    the Black-Sea callers.  Same values and same nearest-age selection as before.
    """
    return basin_level(df, epoch, scenario, basin="pontic")


def _icemask_subset(ds_sub) -> np.ndarray | None:
    """``sftgif`` (percent) from an already-subset ICE-6G dataset, or ``None``.

    ``None`` means the file carries no icemask at all, which is *no ice
    information* — not *no ice*.  Callers must not conflate the two; see
    :func:`gia_delta_subset`.
    """
    if ICE6G_ICEMASK_VAR not in ds_sub.variables:
        return None
    return ds_sub[ICE6G_ICEMASK_VAR].values.astype(np.float64)


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
        topo_t = sub_t["Topo"].values.astype(np.float64)
        topo_0 = sub_0["Topo"].values.astype(np.float64)
        model_lats = sub_t["lat"].values.astype(np.float64)
        model_lons = sub_t["lon"].values.astype(np.float64)
        sftgif_t = _icemask_subset(sub_t)
        sftgif_0 = _icemask_subset(sub_0)

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
    epoch: float, *, gia_nan_policy: str = BAKE_GIA_NAN_POLICY
) -> Path:
    """Bake the GIA-corrected paleo-DEM GeoTIFF for *epoch*; return its path.

    Parameters
    ----------
    gia_nan_policy : {'raise', 'nodata'}
        What to do when the GIA correction is unknown over part of the window
        because ICE-6G ``Topo`` is the ice surface there.  See
        :data:`BAKE_GIA_NAN_POLICIES`.  There is no ``'zero'``.

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

    return out_tif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pontic_df = pd.read_csv(PONTIC_CSV, comment="#")
    # Validate the curve ONCE, before any epoch lookup: sign flips are only
    # detectable on the curve as a whole, never on an individual scalar.
    for scenario in SCENARIOS:
        validate_basin_curve(pontic_df, "pontic", scenario)

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
    header = f"{'epoch':>6} | {'global_sl':>10} | {'grad':>7} | {'catas':>7}"
    print(header)
    print("-" * len(header))

    for epoch in EPOCHS_KA:
        paleo_tif = build_paleo_dem(epoch)
        global_sl = get_sea_level_for_epoch(epoch, SPRATT_CSV)

        # CORRECT — one frame per scenario, DEM-driven at the basin level.
        levels = {}
        for scenario in SCENARIOS:
            bs_sl = pontic_level(pontic_df, epoch, scenario)
            levels[scenario] = bs_sl
            render_paleocostline_epoch(
                epoch,
                {"bs": str(paleo_tif)},
                sea_level_csv=SPRATT_CSV,
                paleocoastlines_shp=PALEOCOASTLINES_SHP,
                output_dir=out_correct[scenario],
                config=cfg,
                data_source_label=f"Pontic-{scenario}",
                sea_level_override_m=bs_sl,
                force_dem_mask=True,
                preloaded_coast_gdf=coast_gdf,
            )

        # WRONG — global eustatic, DEM-driven at the global level (comparison ref).
        render_paleocostline_epoch(
            epoch,
            {"bs": str(paleo_tif)},
            sea_level_csv=SPRATT_CSV,
            paleocoastlines_shp=PALEOCOASTLINES_SHP,
            output_dir=out_wrong,
            config=cfg,
            data_source_label="global-eustatic",
            sea_level_override_m=None,
            force_dem_mask=True,
            preloaded_coast_gdf=coast_gdf,
        )

        print(f"{epoch:>6} | {global_sl:>+10.2f} | {levels['gradual']:>+7.1f} | "
              f"{levels['catastrophic']:>+7.1f}")

    print()
    print(f"Correct (gradual):      {out_correct['gradual']}")
    print(f"Correct (catastrophic): {out_correct['catastrophic']}")
    print(f"Wrong (global):         {out_wrong}")


if __name__ == "__main__":
    main()
