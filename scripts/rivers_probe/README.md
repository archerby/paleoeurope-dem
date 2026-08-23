# Rivers probe — RESCUED AS FOUND, 2026-08-23

These 23 Python files produced `drainage_implied_16.5ka.tif` and the measurements
the 2026-08-13 rivers scoping stands on. Until this commit they existed in
**exactly one copy**, under `/mnt/tank/paleo/renders/_rivers_probe_2026-08-18/`,
with no remote and no history — the same exposure the Khvalynian render scripts
were in until 2026-08-18, on a project that has already lost one 5 310-line
source file to a full disk.

**They are committed exactly as they were found.** Nothing has been ported,
tidied, renamed or fixed. That order is deliberate and it is the order this repo
already used once: the only copy gets a history *before* anyone edits it, so a
port can be reviewed as a diff against something rather than being the first and
last version.

## What is here

| file(s) | what it does |
|---|---|
| `paleoprobe.py` | the prologue every job assumes: `PROBE`, `write_tif` (with an explicit `predictor` argument), `sha256_*`, `toolchain()` |
| `guards.py` | three guards, all raising: no channel on water or UNOBSERVABLE; the third state not collapsed into land; the nodata mask is exactly water∪unobservable |
| `job1_*` | the predictor-free rewrite, and the proof WhiteboxTools can then read it |
| `job2a…2g` | fill, outlets, breach, minimax, channel, geometry, windows |
| `job3a…3d` | build, the threshold sweep, **the product**, and the guard-proof that plants each guard's own defect |
| `job4*` | the final checks and the "this is not a product" accounting |
| `minimax.py`, `grid.py`, `probe_*.py` | connectivity structures, grid helpers and the small probes the above import |
| `final_check.py` | the last read-back over the finished product |

`job3c_product.py` is the one to read first. It picks its channel threshold **from
the sweep** — the largest threshold whose local log-log slope of drainage density
still sits within 0.01 of the plateau exponent the sweep itself exhibits — and it
stamps the rule, the plateau exponent and the hard upper bound (the first
threshold at which the Manych corridor carries zero channel) into the raster.
It also stamps `probe_product_is_not`, in full, in the file.

## What is wrong with them, stated rather than left to be discovered

* **They are a PROBE, not a pipeline step.** One epoch (16.5 ka), one window,
  driven by a prologue that has to be in scope; there is no CLI, no epoch loop
  and no manifest. Making `drainage_implied` a pipeline step is the open item,
  and it is open.
* **`job3c_product.py` reads `khvalynian_basin_fields` as a LITERAL** — the
  51 N tree — while everything else in this repo now goes through
  `PALEO_BASIN_FIELDS_ROOT`. Two basin-field trees exist and they differ in
  645 802 cells, so this must be wired to the variable before the product is
  rebuilt, not after.
* **`PROBE` is an absolute path inside `paleoprobe.py`** and several jobs
  redefine it themselves, so the location is written down more than once — the
  defect class this repo spent 2026-08-18 removing from the mesh half.
* The scripts were run from inside their own output directory, so relative paths
  in them mean "beside the artifacts", not "beside the repo".

## The one thing not to do

Do not "fix" these in place and call it a pipeline step. Port them the way the
mesh campaign and the render scripts were ported: replace the location literals
with the refusing helper, keep every measured number, correct any prose the
change falsifies in the same edit, and prove the result by artifact — a rebuild
that reproduces `drainage_implied_16.5ka.tif` where it should and differs only
where the fix intends.
