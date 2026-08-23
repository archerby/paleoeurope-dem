r"""``drainage_implied`` as a PIPELINE STEP, not a probe.

WHAT THIS PRODUCES, AND WHAT IT IS NOT
======================================
For each epoch, the drainage that the epoch's own **land surface** implies under
D8 routing once the epoch's own water has been removed. Every raster it writes
carries that sentence in full, in its own tags:

    a palaeo-river network. It is the drainage the epoch's own land surface
    implies under D8 routing after the epoch's own water was removed. No channel
    here was observed; none is dated; width, discharge and permanence are not
    claimed.

The 2026-08-13 scoping stands and is not weakened here: on modern land this is
*modern incision at a palaeo base level*; in the Caspian lowland the channels are
the ICE-6G interpolation lattice; on the exposed shelf there is no morphology to
detect. Those verdicts are about INTERPRETATION. This step is about producing the
raster reproducibly, with its provenance and its refusals in the file.

WHY IT IS A NEW FILE RATHER THAN AN EDIT OF THE PROBE
====================================================
``scripts/rivers_probe/`` is committed AS FOUND, deliberately, so that this can be
read as a diff against something. Nothing in it is modified. The pure helpers it
already got right -- ``grid.row_area_km2`` (latitude taken from the transform at
the global row, which is the trap this project has a row for), ``grid.d8_step_len_km``
and the three guards -- are IMPORTED, not re-derived.

WHAT CHANGED FROM THE PROBE, AND WHY
====================================
* **The frame is not a literal.** The probe read
  ``khvalynian_basin_fields`` -- the 51 N tree -- as a hardcoded path. Two trees
  exist and they differ in 645 802 cells. The fields root now comes from
  ``PALEO_BASIN_FIELDS_ROOT`` and REFUSES when unset, and the window comes from
  the field raster's own ``crop_window`` tag, so the frame follows the tree
  instead of being asserted beside it.
* **The predictor-free rewrite is a STEP, not a workaround.** WhiteboxTools
  2.4.0 panics on ``PREDICTOR=3`` and its Python wrapper returns 0 -- an
  artifact-less success. The rewrite is done here, explicitly, and the WBT call
  is judged by its ARTIFACT existing, never by the return code.
* **The threshold is swept per epoch, and the cross-epoch question is MEASURED
  rather than decided.** The probe chose one epoch's threshold from its own
  sweep. Doing that per epoch makes each epoch self-consistent and does NOT make
  the series a time series -- a drainage density computed at a different
  threshold each epoch cannot be differenced across epochs. So the manifest also
  reports whether ONE common threshold lies on EVERY epoch's plateau, and names
  it if so. That is a measurement. Choosing which to publish is not made here.

THE THRESHOLD RULE, VERBATIM FROM THE PROBE
===========================================
The largest threshold whose local log-log slope of drainage density against
threshold still lies within ``PLATEAU_TOL`` of the plateau exponent the sweep
itself exhibits at its finest end. The sweep is the justification; no closed form
is used. The hard upper bound is the first swept threshold at which the Manych
corridor window carries zero channel length -- any threshold at or above it
erases the feature the scene exists to show -- and the step REFUSES if the chosen
threshold reaches it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as nd
from rasterio.windows import Window

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "rivers_probe"))

from grid import d8_step_len_km, row_area_km2                       # noqa: E402
from guards import (                                                # noqa: E402
    GuardFailure, guard_mask_partition, guard_no_channel_on_water_or_unobservable,
    guard_third_state_not_collapsed,
)
from paleoprobe import sha256_arr, sha256_file, utcnow, write_tif    # noqa: E402

S8 = np.ones((3, 3), bool)
NODATA = -9999.0
WATER_BIT, UNOBS_BIT = 1, 2

#: Thresholds swept, in accumulated cells: QUARTER-DECADE steps from 100 cells,
#: eighteen of them, which is the ladder the rivers probe used and therefore the
#: ladder its shipped product was chosen on. Derived rather than typed out, so it
#: cannot drift from what this comment says it is.
#:
#: The ladder is not cosmetic. Sweeping powers of two instead put the chosen
#: threshold at 3200 cells against the probe's 3162.2777 (10**3.5), and the two
#: products then differed in 1437 cells -- all of them probe-only, the signature
#: of a threshold higher by 1.2 %. A sweep's points are where you looked, and
#: the answer is one of them.
SWEEP_DECADE_STEPS = 4
SWEEP_START_CELLS = 100.0
SWEEP_N = 18
SWEEP_CELLS = tuple(SWEEP_START_CELLS * 10.0 ** (k / float(SWEEP_DECADE_STEPS))
                    for k in range(SWEEP_N))

#: How close a local log-log slope has to sit to the plateau exponent to count as
#: still on the plateau. The probe's value, kept so 16.5 ka reproduces.
PLATEAU_TOL = 0.01

#: How much the five finest slopes may spread before there is no plateau to speak
#: of. MEASURED, not chosen: across the 51 N reproduction and all five 53 N spill
#: epochs the spread runs 0.0070-0.0089, so this sits at better than twice the
#: worst observed. It exists because the refusal it replaces could not fire --
#: with five values the median is one of them, so "some slope is within
#: PLATEAU_TOL of the median" is true of every input there has ever been.
PLATEAU_SPREAD_MAX = 0.02

#: The Manych corridor window the upper bound is measured in (w, s, e, n).
#: The rivers probe's own diagnostic window, kept identical because the upper
#: bound is READ OUT of it -- a different rectangle makes the bound a statement
#: about different ground. (Not the same box as khv_story_ramp's, which is
#: 40.5-46.9 E / 45.0-47.2 N and serves a different question.)
MANYCH_BOX = (42.00, 45.00, 48.00, 47.50)


class DrainageRefusal(RuntimeError):
    """A refusal that must stop the epoch rather than be worked around."""


# ---------------------------------------------------------------------------
# locations -- every one refuses rather than defaulting
# ---------------------------------------------------------------------------

def fields_root(explicit=None):
    raw = explicit or os.environ.get("PALEO_BASIN_FIELDS_ROOT", "").strip()
    if not raw:
        raise DrainageRefusal(
            "PALEO_BASIN_FIELDS_ROOT is not set and there is no default. Two "
            "basin-field trees exist -- khvalynian_basin_fields (51 N) and "
            "khvalynian_basin_fields_n53 (53 N) -- and they differ in 645 802 "
            "cells. A default here would pick a FRAME on the reader's behalf "
            "and nothing downstream could tell which one a product was built "
            "on. The rivers probe read the 51 N tree as a literal; that is the "
            "defect this replaces.")
    p = Path(raw)
    if not p.is_dir():
        raise DrainageRefusal("PALEO_BASIN_FIELDS_ROOT is %r, which is not an "
                              "existing directory." % raw)
    return p


def topo_dir(explicit=None):
    raw = explicit or os.environ.get("PALEO_OUT_ROOT", "").strip()
    if not raw:
        raise DrainageRefusal(
            "PALEO_OUT_ROOT is not set, so the paleo-DEM cannot be located. "
            "Set it (Linux: export PALEO_OUT_ROOT=/mnt/tank/paleo/renders) or "
            "pass --topo-dir.")
    p = Path(raw)
    p = p if p.name == "topo" else p / "khvalynian_caspian" / "topo"
    if not p.is_dir():
        raise DrainageRefusal("no topo directory at %s" % p)
    return p


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def _wbt(work_dir):
    import whitebox
    w = whitebox.WhiteboxTools()
    w.work_dir = str(work_dir)
    w.set_verbose_mode(False)
    return w


def _run_wbt(label, fn, *args, artifact=None, **kw):
    """Run a WhiteboxTools call and judge it BY ITS ARTIFACT.

    WBT panics inside Rust and its Python wrapper still returns 0 -- a success
    with no output file. This project's rule is to judge by artifacts; here it
    has to be applied to a third-party tool.
    """
    t0 = time.time()
    lines = []
    fn(*args, callback=lambda s: lines.append(s), **kw)
    bad = [l for l in lines if "panic" in l.lower() or "error" in l.lower()]
    dt = time.time() - t0
    if artifact is not None and not Path(artifact).exists():
        raise DrainageRefusal(
            "%s produced no artifact at %s after %.1f s. WhiteboxTools returns 0 "
            "from a Rust panic, so the return code is not evidence; the file is. "
            "Tool output: %s" % (label, artifact, dt, bad or lines[-3:]))
    if bad:
        raise DrainageRefusal("%s reported: %s" % (label, bad[:3]))
    return dt


def _crop_window(field_tags):
    kv = dict(p.split("=", 1) for p in field_tags["crop_window"].split(";"))
    return Window(int(kv["col_off"]), int(kv["row_off"]),
                  int(kv["width"]), int(kv["height"]))


def _box_mask(transform, shape, box):
    w, s, e, n = box
    rows = np.arange(shape[0])
    cols = np.arange(shape[1])
    lat = transform.f + (rows + 0.5) * transform.e
    lon = transform.c + (cols + 0.5) * transform.a
    return ((lat >= s) & (lat <= n))[:, None] & ((lon >= w) & (lon <= e))[None, :]


def sweep(acc, land, steplen, area_row, manych, thresholds=SWEEP_CELLS):
    """Drainage density and Manych channel length at each threshold."""
    land_area = float((land.sum(axis=1) * area_row).sum())
    rows = []
    for t in thresholds:
        sm = land & (acc >= t)
        length = float(steplen[sm].sum())
        dens = length / land_area if land_area else 0.0
        man = float(steplen[sm & manych].sum())
        rows.append((float(t), int(sm.sum()), length, dens, man))
    return rows, land_area


def choose_threshold(rows, requested=None):
    """The probe's rule, verbatim, plus the refusals it implies.

    *requested*, when given, overrides the choice -- but NOT the Manych bound:
    a threshold asked for is still refused if it erases the corridor at this
    epoch, because a common threshold that quietly kills the feature at one
    epoch is worse than no common threshold.
    """
    ts = np.array([r[0] for r in rows], float)
    dens = np.array([r[3] for r in rows], float)
    man = np.array([r[4] for r in rows], float)
    lt = np.log10(ts)
    ld = np.log10(np.maximum(dens, 1e-12))
    slope = np.gradient(ld, lt)
    plateau = float(np.median(slope[:5]))
    spread = float(np.max(slope[:5]) - np.min(slope[:5]))
    if spread > PLATEAU_SPREAD_MAX:
        raise DrainageRefusal(
            "the five finest sweep points spread %.4f in log-log slope, over the "
            "%g this calls flat, so there is no plateau at the fine end and the "
            "exponent %+.4f is a median of a slope that never settled. A "
            "threshold picked from it would be an arithmetic guess."
            % (spread, PLATEAU_SPREAD_MAX, plateau))
    # Implied by the line above, not a second guard: the median of five values IS
    # one of them, so this is always true once the spread test has passed.
    on = np.abs(slope - plateau) <= PLATEAU_TOL
    dead = man == 0
    upper = float(ts[dead].min()) if dead.any() else float("inf")
    chosen = float(ts[on][-1]) if requested is None else float(requested)
    if chosen >= upper:
        raise DrainageRefusal(
            "the %s threshold %.0f cells is at or above %.0f, the first "
            "threshold at which the Manych corridor carries no channel at all. "
            "Any threshold there erases the feature this scene exists to show."
            % ("requested" if requested is not None else "chosen", chosen, upper))
    return chosen, plateau, upper, slope.tolist()


def build_epoch(epoch, fields, topo, out_dir, work_dir, keep_intermediate=False,
                requested_threshold=None):
    ep = "%g" % float(epoch)
    ws_p = fields / "water_state" / ("water_state_%ska.tif" % ep)
    topo_p = topo / ("khv_%ska.tif" % ep)
    for p in (ws_p, topo_p):
        if not p.exists():
            raise DrainageRefusal("epoch %s: %s is not there" % (ep, p))

    with rasterio.open(ws_p) as d:
        ws = d.read(1)
        wtags = d.tags()
        wtr = d.transform
    win = _crop_window(wtags)
    with rasterio.open(topo_p) as d:
        dem = d.read(1, window=win)
        prof = d.profile.copy()
        ttags = d.tags()
        tr = d.window_transform(win)
        prof.update(width=win.width, height=win.height, transform=tr)
    if tuple(tr)[:6] != tuple(wtr)[:6]:
        raise DrainageRefusal(
            "epoch %s: the topo window's affine %r is not the field's %r -- the "
            "two are not on one grid and every count below would be about "
            "different cells." % (ep, tuple(tr)[:6], tuple(wtr)[:6]))

    water = (ws & WATER_BIT) > 0
    unobs = (ws & UNOBS_BIT) > 0
    masked = dem.astype(np.float32).copy()
    masked[water | unobs] = NODATA

    g_part = guard_mask_partition(masked, NODATA, ws, "mask_partition@%s" % ep)
    g_third = guard_third_state_not_collapsed(masked, NODATA, ws, "third_state@%s" % ep)

    work_dir.mkdir(parents=True, exist_ok=True)
    # The predictor-free rewrite: a STEP, because WBT 2.4.0 panics on PREDICTOR=3
    # and its wrapper returns 0 -- a success with no file.
    land_dem = work_dir / ("dem_land_only_%ska.tif" % ep)
    write_tif(str(land_dem), masked, prof, NODATA, {
        "product": "dem_land_only",
        "method": ("the epoch's own water_state bits water|unobservable set to "
                   "nodata; land elevations untouched; no resampling"),
        "rule": ("WhiteboxTools treats nodata as the edge of the DEM, so the "
                 "epoch's shoreline becomes a flow outlet without inventing one"),
        "third_state_policy": ("UNOBSERVABLE is masked out, NOT drained and NOT "
                               "called land; refusal over substitution of state"),
        "predictor_policy": ("written WITHOUT a compression predictor on purpose: "
                             "WhiteboxTools 2.4.0 panics on PREDICTOR=3 and its "
                             "Python wrapper still returns 0, i.e. a success with "
                             "no output file"),
        "input_dem": str(topo_p), "input_dem_sha256": sha256_file(str(topo_p)),
        "input_water_state": str(ws_p),
        "input_water_state_sha256": sha256_file(str(ws_p)),
        "crop_window_as_read": wtags["crop_window"],
        "crop_north_lat_deg_as_read": wtags.get("crop_north_lat_deg"),
        "epoch_ka": ttags.get("epoch_ka") or ep,
        "cells_water": int(water.sum()),
        "cells_unobservable": int(unobs.sum()),
        "cells_masked_nodata": int((masked == NODATA).sum()),
        "cells_land": int((masked != NODATA).sum()),
    }, predictor=None)

    wbt = _wbt(work_dir)
    fill = work_dir / ("land_fill_%s.tif" % ep)
    pntr = work_dir / ("land_pntr_%s.tif" % ep)
    acc_p = work_dir / ("land_acc_%s.tif" % ep)
    t_fill = _run_wbt("FillDepressions", wbt.fill_depressions,
                      str(land_dem), str(fill), artifact=fill)
    t_pntr = _run_wbt("D8Pointer", wbt.d8_pointer, str(fill), str(pntr),
                      artifact=pntr)
    t_acc = _run_wbt("D8FlowAccumulation", wbt.d8_flow_accumulation, str(fill),
                     str(acc_p), artifact=acc_p, out_type="cells")

    with rasterio.open(acc_p) as d:
        acc = d.read(1)
    with rasterio.open(pntr) as d:
        pn = d.read(1)

    land = masked != NODATA
    steplen = d8_step_len_km(pn, tr, masked.shape[0])
    area_row = row_area_km2(tr, masked.shape[0])
    manych = _box_mask(tr, masked.shape, MANYCH_BOX)

    rows, land_area = sweep(acc, land, steplen, area_row, manych)
    chosen, plateau, upper, slopes = choose_threshold(
        rows, requested=requested_threshold)

    sm = land & (acc >= chosen)
    bad_w, bad_u = guard_no_channel_on_water_or_unobservable(
        sm, ws, "drainage_implied@%s" % ep)
    length = float(steplen[sm].sum())
    _lab, frag = nd.label(sm, structure=S8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("drainage_implied_%ska.tif" % ep)
    write_tif(str(out), sm.astype(np.uint8), prof, 0, {
        "product": "drainage_implied",
        "product_is_not": (
            "a palaeo-river network. It is the drainage the epoch's own land "
            "surface implies under D8 routing after the epoch's own water was "
            "removed. No channel here was observed; none is dated; width, "
            "discharge and permanence are not claimed"),
        "method": ("water_state bits water|unobservable -> nodata; WBT "
                   "FillDepressions on the land only; WBT D8Pointer; WBT "
                   "D8FlowAccumulation in cells; channel = accumulation >= "
                   "threshold"),
        "threshold_cells": chosen,
        "threshold_rule": (
            "largest threshold whose local log-log slope of drainage density vs "
            "threshold still lies within %g of the plateau exponent the sweep "
            "itself exhibits at its finest end; the sweep is the justification, "
            "no closed form was used" % PLATEAU_TOL),
        "threshold_plateau_exponent": plateau,
        "threshold_upper_bound_cells": upper,
        "threshold_upper_bound_rule": (
            "the first swept threshold at which the Manych corridor window "
            "carries zero channel length; any threshold at or above it erases "
            "the feature this scene exists to show"),
        "threshold_is_per_epoch": (
            "TRUE. This threshold was chosen from THIS epoch's own sweep. A "
            "drainage density computed at a different threshold at each epoch "
            "is NOT a time series and must not be differenced across epochs; "
            "see the manifest for whether one common threshold lies on every "
            "epoch's plateau"
            if requested_threshold is None else
            "FALSE. This threshold was REQUESTED and is the same at every epoch "
            "of this run, so densities here CAN be differenced across epochs. It "
            "was still checked against this epoch's own Manych bound and would "
            "have been refused if it erased the corridor here"),
        "epoch_ka": ep,
        "fields_root": str(fields),
        "crop_window_as_read": wtags["crop_window"],
        "crop_north_lat_deg_as_read": wtags.get("crop_north_lat_deg"),
        "input_dem": str(topo_p), "input_dem_sha256": sha256_file(str(topo_p)),
        "input_water_state_sha256": sha256_file(str(ws_p)),
        "wbt_version": "WhiteboxTools v2.4.0",
        "channel_cells": int(sm.sum()),
        "channel_length_km": length,
        "network_fragments_8connected": int(frag),
        "land_area_km2": land_area,
        "drainage_density_km_per_km2": (length / land_area) if land_area else 0.0,
        "guard_channel_cells_on_water": int(bad_w),
        "guard_channel_cells_on_unobservable": int(bad_u),
        "guard_mask_partition_nodata_cells": int(g_part),
        "guard_third_state_leaked_cells": int(g_third),
        "value_encoding": ("1 = channel cell, 0 = not a channel cell; 0 is also "
                           "the declared nodata, and every masked cell is 0 "
                           "because no channel can be asserted there"),
    }, predictor=None)

    if not keep_intermediate:
        for p in (fill, pntr, acc_p, land_dem):
            try:
                p.unlink()
            except OSError:
                pass

    return {
        "epoch_ka": ep,
        "product": str(out),
        "product_sha256": sha256_file(str(out)),
        "pixel_sha256": sha256_arr(sm.astype(np.uint8)),
        "threshold_cells": chosen,
        "threshold_plateau_exponent": plateau,
        "threshold_upper_bound_cells": upper,
        "sweep": [{"threshold_cells": r[0], "channel_cells": r[1],
                   "length_km": r[2], "density_km_per_km2": r[3],
                   "manych_length_km": r[4]} for r in rows],
        "sweep_log_log_slopes": slopes,
        "channel_cells": int(sm.sum()),
        "channel_length_km": length,
        "network_fragments_8connected": int(frag),
        "land_area_km2": land_area,
        "drainage_density_km_per_km2": (length / land_area) if land_area else 0.0,
        "crop_window": wtags["crop_window"],
        "crop_north_lat_deg": wtags.get("crop_north_lat_deg"),
        "seconds": {"fill": t_fill, "pointer": t_pntr, "accumulation": t_acc},
    }


def common_threshold(records):
    """Is there ONE threshold on EVERY epoch's plateau? Measured, not decided."""
    if not records:
        return {"exists": False, "why": "no epochs"}
    sets = []
    for r in records:
        ts = np.array([s["threshold_cells"] for s in r["sweep"]], float)
        sl = np.array(r["sweep_log_log_slopes"], float)
        plateau = r["threshold_plateau_exponent"]
        ok = set(ts[np.abs(sl - plateau) <= PLATEAU_TOL].tolist())
        ok = {t for t in ok if t < r["threshold_upper_bound_cells"]}
        sets.append(ok)
    shared = set.intersection(*sets) if sets else set()
    return {
        "exists": bool(shared),
        "candidates_cells": sorted(shared),
        "largest_cells": max(shared) if shared else None,
        "per_epoch_chosen_cells": {r["epoch_ka"]: r["threshold_cells"]
                                   for r in records},
        "meaning": (
            "a threshold on every epoch's plateau AND below every epoch's "
            "Manych upper bound. If it exists, a series built at it CAN be "
            "differenced across epochs; the per-epoch thresholds shipped in the "
            "rasters cannot. Which to publish is not decided here."),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields-root", default=None)
    ap.add_argument("--topo-dir", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--threshold-cells", type=float, default=None,
                    help="use THIS threshold at every epoch instead of choosing "
                         "one per epoch from its own sweep. The comparable "
                         "series; still refused at any epoch where it would "
                         "erase the Manych corridor.")
    a = ap.parse_args(argv)

    fields = fields_root(a.fields_root)
    topo = topo_dir(a.topo_dir)
    out_dir = Path(a.out)
    work = Path(a.work_dir) if a.work_dir else out_dir / "_work"

    print("fields  %s" % fields, flush=True)
    print("topo    %s" % topo, flush=True)
    print("out     %s" % out_dir, flush=True)

    records, refused = [], {}
    for ep in a.epochs:
        try:
            rec = build_epoch(ep, fields, topo, out_dir, work,
                              keep_intermediate=a.keep_intermediate,
                              requested_threshold=a.threshold_cells)
        except (DrainageRefusal, GuardFailure) as exc:
            refused["%g" % float(ep)] = str(exc)
            print("[D] %s ka REFUSED: %s" % (ep, exc), flush=True)
            continue
        records.append(rec)
        print("[D] %-6s ka  T=%-8.0f cells  %d channel cells  %.1f km  "
              "density %.6f km/km2  %d fragments"
              % (rec["epoch_ka"], rec["threshold_cells"], rec["channel_cells"],
                 rec["channel_length_km"], rec["drainage_density_km_per_km2"],
                 rec["network_fragments_8connected"]), flush=True)

    manifest = {
        "product": "drainage_implied",
        "generated_utc": utcnow(),
        "generated_by": "scripts/drainage_implied.py",
        "fields_root": str(fields),
        "topo_dir": str(topo),
        "sweep_thresholds_cells": list(SWEEP_CELLS),
        "plateau_tolerance": PLATEAU_TOL,
        "plateau_spread_max": PLATEAU_SPREAD_MAX,
        "requested_threshold_cells": None,
        "manych_box_wsen": list(MANYCH_BOX),
        "epochs_built": [r["epoch_ka"] for r in records],
        "epochs_refused": refused,
        "common_threshold": common_threshold(records),
        "epochs": records,
    }
    manifest["requested_threshold_cells"] = a.threshold_cells
    manifest["threshold_mode"] = (
        "per-epoch: each raster carries a threshold chosen from its OWN sweep, "
        "so the densities are NOT differenceable across epochs"
        if a.threshold_cells is None else
        "common: every raster carries the same requested threshold, so the "
        "densities ARE differenceable across epochs")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest %s  built=%d refused=%d" % (mp, len(records), len(refused)),
          flush=True)
    ct = manifest["common_threshold"]
    print("common threshold across the built epochs: %s"
          % ("largest %.0f cells (candidates %s)"
             % (ct["largest_cells"], ct["candidates_cells"])
             if ct["exists"] else "NONE -- the series is not differenceable"),
          flush=True)
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
