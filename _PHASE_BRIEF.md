# DESIGN BRIEF — Khvalynian→Pontic spill, implementation phase

> ### AMENDED 2026-08-03 — the implementation falsified three things this brief instructs. Read this before following any row below.
>
> Nothing is deleted; the original instruction and its correction stand side by side, as the
> vertical-datum contract does.
>
> 1. **§5 item 13 mandates the literal `"UNKNOWN:not_recorded_in_the_curve"` for `band_provenance`.
>    That literal is FALSE and has been retired.** The Caspian curve's `note` column DOES record the
>    envelope edges' sources on the rows the headline number rests on — the 15.0 ka note names
>    HESS2024's "+22 to +35 m a.s.l. reliably dated to 18(17)-13 kyr BP" and attributes the +22 m
>    bottom edge to Makshaev & Tkach. What is true is only that there is no machine-readable
>    `unc_source` / `unc_interp_flag` COLUMN. The shipped string now says that instead.
> 2. **The brief calls `unc_low`/`unc_high` a per-scenario uncertainty band throughout. It is not.**
>    It is an inter-school scenario ENVELOPE carried once per curve ROW and shared by all six
>    Caspian level columns: at 16.0 ka it is [22.0, 50.0] and brackets both `defensible` (+35.0) and
>    `maximal` (+48.0). Verified: no level column falls outside it at any of the 71 rows.
>    Consequently the triple is scenario-INVARIANT, and any field or caption naming a scenario
>    beside an envelope edge asserts something the data does not carry.
> 3. **§4's A-RENDER acceptance row expects
>    `count(ocean_mask ∧ corridor_mask ∧ dem ≥ water_level) == 0`. That quantity is not zero and
>    should not be.** It is 1 054 on the native frame, and every one of those cells comes from the
>    morphological CLOSING inside `dem_threshold_ocean`, which fills 1-px holes and always has. The
>    quantity that isolates the step-corridor overpaint fix, and that must be exactly zero, is
>    `corridor_overpaint_from_the_or`.
>
> Corrected in the code rather than here: the figure "over bed as high as +203.71 m", which shipped
> in the step-mode disclosure, is impossible on the cell set that sentence names — `candidates` is
> carved out of `components(levels[donor])`, i.e. `dem < donor level`, so the set is bounded above
> by the donor's own +29.94 m surface.

Contract of record: `Paleo-Europe_3D/docs/superpowers/research/2026-07-30-vertical-datum-contract.md`. Its header requires a re-read before every bake; every implementer re-reads §6, §7.1, §8 before first edit. Cite by section/symbol only.

---

## 1. Contradictions resolved

### C1 — "the guard's expectation": `MANYCH_SILL_M` (+23.00) vs `MANYCH_SILL_PATH_M` (+25.00)
**Resolved: `MANYCH_SILL_PATH_M`, converted to F1.** Not a disagreement — a frame error in one reader.

| Reader | Raster | Frame | Native path saddle |
|---|---|---|---|
| A1 | modern GEBCO COG | F0 | **+25.000 m** @ 44.1021E/45.7062N |
| A6 | `khv_16.5ka.tif` | F1 (GIA-corrected) | **+27.751 m** |
| A4 | `khv_16.5ka.tif`, `dem_sm` | F1 | +28.001 m |

+25.000 (F0) + `dz_sill_m` 2.701 = **+27.701** = the bake's own `sill_path_f1_m`. A6's independent flood reads +27.751 — agreement 0.05 m. Two readers, two rasters, two frames, one constant. `MANYCH_SILL_M` = +23.00 is documented in `bake_khvalynian` as the max of per-column minima, i.e. a **lower bound** on the path crest; armed as the expectation it gives a +2.30 m gap on the native grid and false-fails (A6 verified). A5's claim that "the guard cannot be satisfied at any resolution" is **refuted**: A5 compared the F1 saddle against the bare F0 constant without `dz_sill_m` — exactly the §6 unconverted-comparison error.

### C2 — "the frame is correct at ≥4000 px"
**Refuted. Only the native grid (max_px ≥ 7800) is correct.** A1's "+25.00 at 4000" was on the raw mosaic of the modern COG; `render_paleocostline_epoch` thresholds `dem_sm` (σ=1.5), not the mosaic. Triple-confirmed on `khv_16.5ka.tif` by A4, A5 (sidecars) and A6, to the centimetre:

| max_px | raw mosaic | `dem_sm` | gap vs +27.701 (`dem_sm`) | connects at level +29.944? |
|---|---|---|---|---|
| 2000 (shipped bake) | +32.709 | **+36.005** | +8.30 | no — contradicts `khvalynian_sill_test.csv` |
| 4000 | +27.899 | **+32.942** | +5.24 | no — contradicts the CSV |
| 7800 native | +27.751 | **+28.001** | **+0.30** | yes |

### C3 — "~5 m of resampling inflation at 2000" vs the `SillResolutionError` docstring's "+1.00 m"
**Both correct; different quantities on the same grid.** Docstring = bilinear value of the single cell at 44.119E/45.773N. Guard = connected-path minimax. Point-cell residual at 2000 px +1.19 m; path residual +4.96 m raw / +8.25 m on `dem_sm` (A6). A1 measured the same magnitude on the COG (+5.00 m at 2000). The docstring's framing of +1.00/+2.00 m as "the honest cost of downsampling ... inside the 2 m tolerance" is **wrong for the guard** and must be corrected in place. Point-cell reading is also non-monotone (1000→+27, 2000→+24, 3000→+25, 4000→+25, 6000→+26, native→+23) — bilinear phase, not a resolution cost.

### C4 — Pontic re-bake: 32.44 % vs 21.69 % of pixels changed
**Believe A2 on the fact, treat the percentage as undefined.** A2 has the unambiguous result: rendering 5 frames at HEAD with `legacy_degree_grid_blend=True` reproduces the on-disk PNGs with **identical sha256**. That simultaneously (a) proves the re-bake is not a no-op and (b) proves the seam blend is the *only* behavioural delta in 2a38b33 for these frames — which **resolves A6's unknown #1** ("part of the difference may come from other changes in the commit"). The two percentages differ by the diff threshold (A6 counted `>1/255`, A2 counted any nonzero). Non-load-bearing; the implementer pins one definition.

### C5 — the band's lower bound
**Both readers agree to the centimetre.** A3: `unc_low` endorheic at all 17 epochs 17.00→13.00 (16.5 ka margin −5.657 m). A6: same, 11 epochs of the window, 16.5 ka −5.657. Both find `unc_low + p95(locus)` reaches `at_sill` at 16.00/15.75/15.50/15.25/15.00. Residual: A3 quotes the corner range as −0.13…−0.07, A6 lists −0.130/−0.169/−0.176/−0.125/−0.071. **Implementer pins A6's five values and reconciles the 15.5 ka entry** (same CSV, same code path — one of the two rounded).

### C6 — "both repos CLEAN"
**Refuted by A2, A5, A6 independently.** `Paleo-Europe_3D` @ 8164d9a has 9 tracked changes + ~140 untracked. `paleoeurope-dem` @ 2a38b33 is clean. Consequence: **no implementer writes to `Paleo-Europe_3D`.** Curve-builder changes (`unc_source`, `unc_interp_flag`) are out of scope this phase.

### C7 — in-guard native re-measure (A1) vs expectation from the sill test (A6)
**A6's mechanism, A1's type.** A1's `native_source=` re-measure is **degenerate here**: the "native source" is the same `khv_*.tif` the mosaic is built from, so at max_px ≥ 7800 the guard would compare its own measurement against itself and certify nothing. It also carries A1's own unresolved windowing question (33.7 M cells / ~2 GB for a full-grid pure-Python flood). The F0 derivation A1 wants has **already been performed once** (A1's own +25.000 native flood) and belongs in a pinned test, not in every frame. Keep A1's typed `SillExpectation(value_m, frame, source, measured_by, lon, lat)` — a bare float is refused.

### BLOCKED (cannot be resolved from what was returned)

| ID | Question | Single measurement that settles it | Blocks |
|---|---|---|---|
| B1 | Is the saddle seed-independent? All three readers invented `BasinSeed` coords; none swept. | Sweep seeds over all cells below −200 m in each basin on `khv_16.5ka.tif` at max_px=7800; assert `sill_m[('caspian','pontic')]` invariant to ±0.05 m. | A-RENDER acceptance |
| B2 | Does the guard hold at the other 16 epochs of 17–13 ka? Only 2 of 43 rasters exist. | Bake the fine window (15 rasters, ~714 MB) and re-run the native guard per epoch. | E-ANIM entirely |
| B3 | Is a full native render viable at 34 frames? 139 s/frame measured, ~1.5–2 GB/worker. | Time one full `render_paleocostline_epoch` at max_px=7800 with peak-RSS instrumentation. | E-ANIM scheduling |
| B4 | Is `khvalynian_sill_test.csv` (mtime 2026-08-03 10:16, newer than HEAD) reproducible at 2a38b33? | Regenerate with `--sill-report-only` at a clean 2a38b33 checkout and diff. | A-RENDER's expectation source |

---

## 2. Task order by dependency

| Wave | Task | Why here | Status |
|---|---|---|---|
| **0** | **T2 — archive + re-bake the 39 Pontic frames** | Zero source edits; establishes the byte-baseline every later wave must not break. Must precede A because `bake_black_sea.py` imports `paleocostline_render.py`. | GO |
| **1** | **A — T1 + T4 (renderer)** | Decides the expectation, the guarded array, the refusal. Everything visual depends on it. | GO, B1 pending |
| **1** | **C — T3 (band) + `basin_level` guard + ICE-6G bracket** | Disjoint files from A. Its ICE-6G bracket is a hard precondition for E. | GO |
| **2** | **E — T5 (animation)** | Serialized after A (same two files) and after C (quarter-ka Pontic thresholds). Additionally blocked on B2/B3. | **DEFER** |

### Dropped / refused (established by measurement, not preference)

| Item | Verdict | Evidence |
|---|---|---|
| Bump `max_px` to 4000 in `bake_black_sea.main` | **DROP** | ≥3415 px is a no-op (`step` clamps to 1.0); it is a hardcoded literal, not a flag; no measurement says it matters for Pontic (A2). |
| Re-run `scripts/render_black_sea_corrected.py` | **DO NOT RUN** | Still reads the retired `spratt_lisiecki_2016_simplified.csv` and calls `get_sea_level_for_epoch` raw — the contract's own KNOWN GAP. |
| Minimum-channel-width in the morphology | **DROP** | Connecting body's median cross-section is 36–134 px; only 1 column ≤2 px at 2000 px. The break is a 261-cell stretch where the ramp is 1.46–9.26 m *below* the bed — dilation cannot reach it (A4). |
| Paint the corridor cells | **REFUSE** | 69,523 cells / 158,680 km² over bed up to +36 m, in 44 disjoint pieces — 4× the 38,040 km² of spurious drowning contract §5a exists to remove. |
| Minimum-`max_px` precondition as the gate | **DROP as a gate** | Non-monotone: 2048 resolves 1.00 m worse than 2000 (A1). Legitimate only as text inside the `SillResolutionError` message. |
| In-guard `native_source=` re-measure | **DROP** | Self-referential (C7). |
| Synthesized "low" scenario column in the curve | **FORBIDDEN** | `validate_basin_curve` check A requires `max(level) ≥ MANYCH_SILL_M` = +23.0; `max(unc_low)` = +22.0 → raises. Contract §8: convert at the threshold, not in the curve. |
| Remove the σ=1.5 `dem_sm` from the mask path | **NOT THIS PHASE** | It feeds `dem_threshold_ocean` on *all* paths — removing it changes all 39 Pontic frames and destroys the Wave-0 baseline. Measure and report it instead. |

---

## 3. File partition (no file appears twice)

| Implementer | Exclusive write scope | Tasks |
|---|---|---|
| **W0-REBAKE** | *(no source files)* — writes only `D:\pasha\RESEARCH\renders\black_sea_corrected*`, `paleoeurope-dem\outputs\_bs_paleo\` | T2 |
| **A-RENDER** | `paleoeurope-dem\paleoeurope\viz\paleocostline_render.py`, `paleoeurope-dem\tests\test_viz.py` | T1 + T4 (**merged** — same two files, and T4's step-overpaint bug is in the line T1 must gate) |
| **C-BAND** | `paleoeurope-dem\scripts\bake_khvalynian.py`, `paleoeurope-dem\scripts\bake_black_sea.py`, `paleoeurope-dem\tests\test_bake_khvalynian.py`, `paleoeurope-dem\tests\test_bake_black_sea.py` | T3 + `basin_level` age-gap guard + `_ice6g_bracket` shared into `pontic_threshold` + gate `main()`'s unconditional CSV write |
| **E-ANIM** (wave 2, serialized) | `paleoeurope-dem\scripts\animate_khvalynian_spill.py` (new) — **inherits A-RENDER's two files after A merges**, never concurrently | T5 |

Serialization rules:
- W0 runs to completion before A or C starts (the bake imports the renderer).
- A and C run in parallel; **C is additive only** — it may not change `MANYCH_SILL_M`, `MANYCH_SILL_PATH_M`, `AT_SILL_TOL_M`, or the existing `SillTest` column set, which A's tests import read-only.
- E starts only after A merges and B2/B3 are answered.
- `D:\pasha\RESEARCH\Paleo-Europe_3D` is **read-only for all four** (dirty tree, known-red loader).

---

## 4. Per-task specification and acceptance tests

### W0-REBAKE (T2)

Steps: rename `renders\black_sea_corrected` → `renders\black_sea_corrected_PRE_SEAM_FIX_2026-08-01` (rename, not copy — D: has 9.4 GB free); write `provenance.json` **into the archive**; run `python scripts\bake_black_sea.py` from the repo root (no env vars, no PYTHONPATH, no `skip_unconvertible` — preflight resolves 26/26); then `python scripts\make_bs_qc_gallery.py`. Budget ~5 min cold.

Acceptance (all four):
1. `renders\black_sea_corrected_PRE_SEAM_FIX_2026-08-01\provenance.json` exists and contains the recipe string `2a38b33 + RenderConfig(max_px=2000, ice_model_label="ICE-6G_C", legacy_degree_grid_blend=True)` plus **sha256 for all 39 archived PNGs**.
2. Re-rendering `16.5 ka gradual` at 2a38b33 with `legacy_degree_grid_blend=True` reproduces the **archived** file with identical sha256 (A2 verified this for 5 frames; the archive sidecar must let anyone repeat it for all 39).
3. The new `black_sea_corrected\topo` tree contains exactly 39 PNGs across `correct/gradual`, `correct/catastrophic`, `wrong`, and the new `16.5 ka gradual` differs from the archived one in ≥ 700,000 of 2,376,900 pixels under an *explicitly documented* diff threshold.
4. `black_sea_corrected\topo\bs_qc_gallery.html` exists (the archive has one; the current tree does not).

No commit, no source edit. `renders\` is not a git repo — the sidecar *is* the provenance.

### A-RENDER (T1 + T4)

Design decisions, each with its evidence:

| Decision | Value | Evidence |
|---|---|---|
| Expectation quantity | `MANYCH_SILL_PATH_M + dz_sill_m` = the bake's `sill_path_f1_m` (+27.701 at 16.5 ka) | C1 |
| Expectation type | frozen `SillExpectation(value_m, frame, source, measured_by, lon, lat)`; bare float refused; `frame` must match the raster's `vertical_frame` tag | §6, §7.1 |
| Guarded array | `dem_sm` — the array that produces the mask. Guarding `dem` while painting `dem_sm` certifies a quantity nobody draws. | C2 |
| Tolerance | keep 2.0 m, **re-derive in writing** from the spill window (2 m ≈ 0.3 ka against a 0.25-ka grid) and delete the circular citation between `sill_tolerance_m` and `seam_blend_max_shift_m` | A1 |
| Consequence | the two-basin frame is renderable **only at max_px ≥ 7800**. This is a refusal, not a knob. | C2 |
| Smoothing | keep σ=1.5; **measure** `dem` vs `dem_sm` sill and emit `SmoothingShiftWarning` + sidecar field. Never subtract a modelled σ-correction. | A4, A6 |

Acceptance (named assertions in `tests/test_viz.py`):

| Test | Assertion |
|---|---|
| `test_two_basin_frame_refuses_below_native_resolution` | `render_paleocostline_epoch(khv_16.5ka.tif, basin_seeds=…, max_px=2000)` raises `SillResolutionError` whose message contains measured `+36.01`, expected `+27.70`, gap `+8.30`; same at 4000 (`+32.94`, gap `+5.24`); at 7800 it does **not** raise (gap `+0.30`). |
| `test_multi_seed_without_expectation_raises` | `assign_basins(..., seeds=2, sill_expectations=None)` raises. Today it is a silent no-op. |
| `test_bare_float_expectation_is_refused` | passing `{('caspian','pontic'): 27.701}` raises `TypeError`. |
| `test_expectation_frame_must_match_raster` | an F0-framed `SillExpectation` against an F1 raster raises (this is the exact error A5 made). |
| `test_manych_path_constant_matches_a_native_flood` | on the modern COG at native, path saddle == `25.00 ± 0.05` (pins A1's derivation). Marked `@pytest.mark.dataset`; **required in the phase acceptance run**, not in the 397. |
| `test_step_corridor_is_never_painted` | with `corridor="step"`, `count(final ocean_mask ∧ corridor_mask ∧ dem ≥ water_level) == 0`. Today: 69,453 cells / 158,516 km². |
| `test_unseeded_water_body_inside_extent_raises` | two seeds at max_px=7800 raises, listing the 545,978-cell Aegean and 83,001-cell Levantine components, instead of painting them at an invented −17.6…−3.3 m surface. |
| `test_smoothing_shift_is_reported` | sidecar carries `sill[].smoothing_shift_m`; `SmoothingShiftWarning` raised when `|shift| > seam_blend_max_shift_m`. |
| `test_sidecar_records_what_it_was_measured_on` | `basins.json` has `grid.{max_px,height,width,pixel_deg}`, `sill[].{expected_m,tolerance_m,measured_on,smoothing_shift_m}`, `painted_water.{cells,components,basins_joined,gap_cells}`, `morphology.{closing_iterations,opening_iterations,cells_added,cells_removed}`, `spill_not_drawn` (fixed disclosure string). |
| `test_pontic_single_basin_path_is_byte_stable` | re-rendering `16.5 ka gradual` after A sha256-matches the **W0 output**. |
| `test_ocean_lake_label_keys_on_basin_field` | two endorheic lakes are not labelled "Ocean" (the ternary currently tests only `sea_level_override_m`). |

Artifact acceptance: one `.basins.json` at max_px=7800 for 16.5 ka defensible showing `spilling=true`, `painted_water.basins_joined=false`, `gap_cells=1096`, and the `spill_not_drawn` disclosure — i.e. the disagreement is **stated in the data**, not painted away. Drawing the minimax route (keep the predecessor array `_priority_flood_ownership` already computes, discard nothing) as an annotated polyline is in scope; painting water is not.

### C-BAND (T3 + curve-reading guards)

Acceptance (`tests/test_bake_khvalynian.py`, `tests/test_bake_black_sea.py`):

| Test | Assertion |
|---|---|
| `test_unc_band_is_absolute_not_a_width` | `unc_low ≤ level ≤ unc_high` on all 71 Caspian rows for all four distinct level columns. |
| `test_band_lower_bound_never_spills` | `spill_window_ka(report, "defensible", verdict_column="verdict_low") is None`; same for `"maximal"`. |
| `test_band_upper_bound_spills_17_to_13_75` | `verdict_high == "spilling_west"` at 17.00…13.75, `"endorheic"` at 13.50/13.25/13.00. |
| `test_joint_corner_is_not_robust_on_the_mwp1a_ramp` | `verdict_is_jointly_robust` False at exactly {16.00, 15.75, 15.50, 15.25, 15.00}, margins pinned to the measured −0.130/−0.169/−0.176/−0.125/−0.071 (reconcile C5 first). |
| `test_verdict_is_robust_is_unchanged` | `verdict_is_robust` computation byte-identical; `test_the_only_non_robust_verdict_is_the_mwp1a_epoch` still green. |
| `test_band_that_does_not_contain_its_central_is_refused` | Pontic catastrophic at 9.0/8.5/8.0/7.5 ka yields `band_status="REFUSED: band does not contain its central (level −100.0, band [−27.0, −17.0])"`, all four band fields `None`. |
| `test_pontic_quarter_ka_epoch_resolves` | `pontic_threshold(df, 16.75, "gradual")` returns a threshold via `_ice6g_bracket`, tagged `ice6g_interpolated=True` — not a raw `FileNotFoundError`. |
| `test_basin_level_refuses_a_far_row` | `basin_level(df, 16.75, "gradual")` no longer silently returns the 17.0 ka row's −58.3 m over the 16.5 ka row's −30.0 m (28.3 m decided by DataFrame row order). |
| `test_nothing_is_unconvertible_any_more_so_the_plain_bake_does_not_abort` | still green — `preflight_thresholds` still 26/26, 0 unconvertible. |
| `test_sill_report_csv_write_is_gated` | `--sill-report-only` no longer unconditionally rewrites `$PALEO_OUT_ROOT/khvalynian_caspian/khvalynian_sill_test.csv`; requires an explicit `--write`. |

Artifact acceptance: the report CSV gains `verdict_low/verdict_high/band_verdict/verdict_is_band_robust/verdict_is_jointly_robust`, and the caption string it produces reads verbatim: *"at the central estimate the Caspian spilled west 16.5→14.25 ka (defensible; 16.75 on maximal); at the published band's lower bound it never cleared the Manych at any epoch of 17–13 ka — but by as little as 0.73 m at 16.0 ka, inside the sill's own ±2 m GIA bar."*

One deliberate catch, at exactly one site: `basin_level_band()` **raises** `MissingBandError`; `pontic_threshold` catches it and records `band_status` + `None` fields, because the band is not load-bearing for the Pontic threshold and the 39 correct frames plus the 13 deliberately-wrong frames must keep being produced. `caspian_threshold` does **not** catch — there the band drives a verdict.

### E-ANIM (T5) — DEFER

Preconditions before it may start: A merged; C's `_ice6g_bracket` merged; B2 (15 rasters baked); B3 (native render cost/memory). Acceptance when it runs: `sequence_limits.json` naming which epoch produced each extremum; all 34 frames identical in pixel size (`tight_bbox=False`); GIF `duration` read back with PIL == 250 ms per frame (`duration=0.2` writes 0 ms silently on imageio 2.37.2 — four of the five existing builders are wrong today); a refused epoch produces a hatched frame carrying `{"threshold": null, "refusal": "<class>: <message>"}`, never a gap and never an interpolated level; a single master palette (per-frame `ADAPTIVE` differs in 211/256 entries between adjacent frames).

---

## 5. RAISE-not-compute inventory

Every site where the honest implementation refuses, and the number that would otherwise be substituted.

| # | Site (symbol) | Substitution risk today | Required behaviour | Owner |
|---|---|---|---|---|
| 1 | `assign_basins`, `sill_expectations=None` with >1 seed | loop body never runs — a frame is shipped unchecked | RAISE | A |
| 2 | `sill_expectations` bare float | a guessed number is indistinguishable from a measured one (§7.1) | RAISE; require `SillExpectation` | A |
| 3 | expectation frame vs raster frame | F0 constant silently compared to an F1 saddle (§6) — the error A5 made | RAISE on mismatch | A |
| 4 | two-basin frame below native resolution | picture contradicts the raster tag `sill_verdict_defensible=spilling_west` | RAISE with measured/expected/gap; **never auto-bump `max_px`** (non-monotone) | A |
| 5 | unseeded water body inside `extent` | 545,978 Aegean + 83,001 Levantine cells painted at an invented −17.6…−3.3 m surface | RAISE listing components and sizes | A |
| 6 | `BasinSeed` lon/lat | invented by all three readers; the repo registers none | RAISE unless from a provenanced registry; derive by deterministic argmin over a named box and record box + result + raster hash in the data | A (B1) |
| 7 | `dem_sm` sill shift | +0.25…+5.04 m, silent; the seam audit covers a step that is a no-op for single-file input | measure + `SmoothingShiftWarning` + sidecar field; never subtract a modelled correction | A |
| 8 | `basin_level` nearest-age, no tolerance | 16.75 ka silently takes −58.3 m instead of −30.0 m, tie broken by row order | RAISE on age gap > tol | C |
| 9 | `pontic_threshold` → `_open_bs_window` missing `.nc` | raw `FileNotFoundError` at 8 of 17 quarter-ka epochs | typed refusal, or `_ice6g_bracket` **tagged as interpolated** | C |
| 10 | missing / NaN / inverted band edge | would collapse the triple to a boolean | `MissingBandError`; never substitute the central, never borrow the neighbouring row (band width varies 28 m → 2 m) | C |
| 11 | band not containing its central | Pontic catastrophic 9.0/8.5/8.0/7.5 ka; 9.0 and 7.5 are baked epochs | refuse, record `band_status` | C |
| 12 | F4 (`ocean_offset`) row band | converting a band on a row whose level is not the threshold | all four fields `None`, `band_rule="not_applicable_F4"` | C |
| 13 | `band_provenance` | curve records no `unc_source`/`unc_interp_flag`; 12 of 17 engine epochs read an interpolated edge | literal `"UNKNOWN:not_recorded_in_the_curve"`; caption says "band as carried in the curve", not "as published" | C |
| 14 | synthesized low-scenario curve column | trips `validate_basin_curve` check A (+22.0 < +23.0) and violates §8 | FORBIDDEN | C |
| 15 | colour-limit clipping in a sequence | silent clip | reserved out-of-range colour + count in sidecar; `clip_policy="raise"` for production | E |
| 16 | refused epoch in the animation | dropping compresses time, interpolating invents a level | hatched refusal frame keeping its slot and duration | E |

---

## 6. Must not change

| Invariant | Guard |
|---|---|
| **397 green tests in `paleoeurope-dem`** | All new `RenderConfig` fields default to `None` = today's behaviour. All new `BasinThreshold`/`SillTest` fields are additive. Full suite must be 397 green (+ new tests) before any merge. |
| **`verdict_is_robust`** | Computation stays byte-identical (locus/GIA spread only, `bracket_m = (level+p5, level+p95)` with the level held fixed). Only its docstring changes. `test_the_only_non_robust_verdict_is_the_mwp1a_epoch` must not be edited. |
| **Known-red `src/data/fabdem_loader.py` (Paleo-Europe_3D)** | That repo is read-only for this phase and is additionally dirty (9 tracked + ~140 untracked). Do not touch, do not fix, do not run its test suite as a gate. |
| **The 13 deliberately-wrong eustatic frames in `bake_black_sea.main`** | `eustatic_wrong_frame_m`, the `wrong/` tree and the F4 comparison stay wrong on purpose. C must not "correct" them; the re-bake must still emit all 13. |
| **The single-basin (`sea_level_override_m`) render path** | Byte-stable across A: `16.5 ka gradual` sha256 after A == W0 output. This is why `dem_sm` stays on the mask path. |
| **`legacy_degree_grid_blend`** | Must remain and must keep reproducing the archived frames byte-for-byte — it is the only provenance the pre-seam-fix tree has. |
| **`MANYCH_SILL_M`, `MANYCH_SILL_PATH_M`, `AT_SILL_TOL_M`, existing `SillTest` columns** | C may not change them; A's tests import them read-only. |
| **`preflight_thresholds` = 26/26, 0 unconvertible** | The plain `python scripts/bake_black_sea.py` must keep running without `skip_unconvertible`. |

### Contract §8 exposure

| Row | Exposure | How the plan avoids it |
|---|---|---|
| "convert at the threshold, not in the curve" (the one §8 rule quoted back by a reader) | (a) a synthesized `unc_low` scenario column; (b) baking `dz` into a curve cell; (c) comparing a raw F2 band edge to an F1 crest | (a) forbidden — see #14; (b) band conversion lives in `BasinThreshold` only; (c) both band edges converted with the **same** `conversion_m` and `rule` as the central, or refused |
| Remaining §8 rows | **UNKNOWN — not quoted by any reader.** | Each implementer re-reads §8 in full before first edit (the contract header requires it) and reports any additional row this plan trips **before** writing code. |

Ancillary risk, recorded not fixed: `dz` for the band's low edge strictly belongs to a *different* site (Makshaev Lower Volga vs Semikolennykh Manych spillway). §6's 2026-07-31 update measures `dz(sill) − dz(marker)` at +0.03…+0.18 m on the band median — below the 0.73 m headroom at 16.0 ka but not negligible against it. Record `band_provenance`; do not model a per-edge `dz`.