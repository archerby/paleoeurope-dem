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

OUTPUT
------
  $PALEO_OUT_ROOT/khvalynian_caspian/topo/khv_<epoch>ka.tif   (paleo-DEM, tagged)
  $PALEO_OUT_ROOT/khvalynian_caspian/khvalynian_sill_test.csv (THE sill report)

ROOTS
-----
``PALEO_DATA_ROOT`` / ``PALEO_OUT_ROOT``, resolved by ``bake_black_sea`` and reused
verbatim so the two bakes cannot end up pointing at different trees.  The working
raster is located through ``KHVALYNIAN_DEM_PATH`` (default: the verified COG on Y:).

Run (repo NOT pip-installed):
    PYTHONPATH=$(pwd) python scripts/bake_khvalynian.py --sill-report-only
    PYTHONPATH=$(pwd) python scripts/bake_khvalynian.py --epochs 16.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
from scripts.bake_black_sea import (  # noqa: E402
    _MARKER_COLUMNS,
    BAKE_GIA_NAN_POLICIES,
    BS_NODATA_M,
    GIA_SIGMA_BLUR,
    ICE6G_DIR,
    ICE6G_ICEMASK_VAR,
    MANYCH_SILL_M,
    PALEO_DATA_ROOT,
    PALEO_OUT_ROOT,
    BasinThreshold,
    UnconvertibleThresholdError,
    UnknownGiaInBakeError,
    _check_threshold_sanity,
    _epoch_filename,
    _parse_marker_position,
    _require_model_grid_brackets_tile,
    basin_level,
    validate_basin_curve,
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

#: Scene bbox = the working raster's own extent.  Stated as constants rather than
#: read from the file so that _require_model_grid_brackets_tile has something to
#: check the raster AGAINST, and so a swapped raster is caught instead of silently
#: redefining the scene.
LAT_MIN, LAT_MAX = 36.0, 54.0
LON_MIN, LON_MAX = 25.5, 58.0

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


@lru_cache(maxsize=1)
def ice6g_ages_ka() -> tuple[float, ...]:
    """Ages (ka) for which an ICE-6G file exists, ascending.

    Read from the directory, never assumed: the distribution steps 0.5 ka to 21 ka
    and 1 ka beyond, and hard-coding that would silently mis-bracket if a file were
    added or missing.
    """
    ages: list[float] = []
    for path in ICE6G_DIR.glob("I6_C.VM5a_10min.*.nc"):
        token = path.name[len("I6_C.VM5a_10min.") : -len(".nc")]
        try:
            ages.append(float(token))
        except ValueError:  # pragma: no cover - a stray file, not an age
            continue
    if not ages:
        raise FileNotFoundError(f"no ICE-6G files under {ICE6G_DIR}")
    return tuple(sorted(ages))


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


def _time_interpolation_note(epoch: float) -> str:
    """Human/tag-readable statement of how *epoch* got its GIA field."""
    lo, hi, w = _ice6g_bracket(epoch)
    if lo == hi:
        return f"exact ICE-6G epoch {lo:g} ka (no time interpolation)"
    return (
        f"linear in time between ICE-6G {lo:g} and {hi:g} ka, w={w:.4f} "
        "(each endpoint computed exactly; the blend is the only approximation)"
    )


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

    Raises
    ------
    ValueError
        Unknown scenario, or a converted threshold outside the sanity window.
    UnconvertibleThresholdError
        The locus is unusable at this epoch (too few finite cells).
    """
    col = f"level_m_global_{scenario}"
    if col not in df.columns:
        raise ValueError(
            f"unknown scenario {scenario!r}; available columns={list(df.columns)}"
        )
    level = basin_level(df, epoch, scenario, basin="caspian")
    where = f"{float(epoch):g} ka / {scenario}"

    # --- rule 1: a located marker, if the curve ever grows one ----------------
    if all(c in df.columns for c in _MARKER_COLUMNS):
        idx = (df["age_ka"] - float(epoch)).abs().idxmin()
        position = _parse_marker_position(df.loc[idx], scenario)
        if position is not None:
            lon, lat = position
            dz, lats, lons = solid_displacement_field(epoch)
            i, j = _cell_index(lats, lons, lon, lat)
            conversion = float(dz[i, j])
            block = dz[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
            finite = block[np.isfinite(block)]
            threshold = _check_threshold_sanity(level + conversion, where)
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
                    f"{threshold:+.3f} m"
                ),
            )

    # --- rule 2: the documented basin band + its p5-p95 bar -------------------
    median, (p5, p95), n_cells = locus_solid_displacement(epoch)
    threshold = _check_threshold_sanity(level + median, where)
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
            f"{CASPIAN_SHORELINE_BAND.why}"
        ),
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
    gia_time_interpolation: str
    provenance: str


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

    lo_b, hi_b = threshold.bracket_m if threshold.bracket_m else (surface, surface)
    robust = _verdict(lo_b - sill_f1) == _verdict(hi_b - sill_f1) == _verdict(margin)

    margin_modern = threshold.level_m - MANYCH_SILL_M
    margin_mixed = threshold.level_m - sill_f1

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


def spill_window_ka(report: pd.DataFrame, scenario: str) -> tuple[float, float] | None:
    """``(opens_ka, closes_ka)`` — oldest and youngest ``spilling_west`` epoch.

    ``None`` when the scenario never spills.  Ages, so *opens* is the LARGER number.
    Reported off :attr:`SillTest.verdict`, i.e. the F1-vs-F1 test.
    """
    hit = report[(report["scenario"] == scenario) & (report["verdict"] == "spilling_west")]
    if hit.empty:
        return None
    return float(hit["epoch_ka"].max()), float(hit["epoch_ka"].min())


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


def build_paleo_dem(
    epoch: float,
    *,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
    provenance: Mapping[str, str] | None = None,
    dem_path: Path | None = None,
    out_dir: Path | None = None,
    strip_rows: int = STRIP_ROWS,
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
                dst.write(paleo.astype(np.float32), 1, window=window)

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
    header = (
        f"{'epoch':>6} | {'scenario':>11} | {'level_F2':>9} {'surface_F1':>10} | "
        f"{'crest_F1':>8} {'margin':>8} | {'verdict':>13} {'robust':>6} | "
        f"{'+25m path':>13}"
    )
    print()
    print(header)
    print("-" * len(header))
    for _i, r in report.iterrows():
        print(
            f"{r.epoch_ka:>6g} | {r.scenario:>11} | {r.level_f2_m:>+9.2f} "
            f"{r.surface_f1_m:>+10.2f} | {r.sill_f1_m:>+8.2f} {r.margin_m:>+8.2f} | "
            f"{r.verdict:>13} {str(bool(r.verdict_is_robust)):>6} | "
            f"{r.verdict_path:>13}"
        )


def main(
    epochs: Sequence[float] | None = None,
    scenarios: Sequence[str] = SCENARIOS,
    *,
    gia_nan_policy: str = KHV_GIA_NAN_POLICY,
    write_rasters: bool = True,
    csv_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Bake the Khvalynian side and write the sill report.  Returns the report.

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
    out_base.mkdir(parents=True, exist_ok=True)
    report_path = out_base / SILL_REPORT_CSV.name
    report.to_csv(report_path, index=False)

    if write_rasters:
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
        if window is None:
            print(f"{scenario:>11}: never clears the sill on this grid")
        else:
            print(
                f"{scenario:>11}: spilling_west {window[0]:g} -> {window[1]:g} ka "
                "(F1 vs F1, +23.00 m point sill)"
            )
    print()
    print(f"Sill report: {report_path}")
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
        help="compute and write the sill report, write no rasters (seconds, not hours)",
    )
    parser.add_argument(
        "--nan-policy",
        choices=BAKE_GIA_NAN_POLICIES,
        default=KHV_GIA_NAN_POLICY,
        help="what to do where the GIA correction is unknown (no 'zero' exists)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(
        epochs=args.epochs,
        gia_nan_policy=args.nan_policy,
        write_rasters=not args.sill_report_only,
    )


if __name__ == "__main__":
    _cli()
