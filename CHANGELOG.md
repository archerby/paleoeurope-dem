# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work towards v1.1 (not yet released). Corrections to the method described in the
v1.0 preprint (https://doi.org/10.31223/X59T9W):

### Fixed

- **GIA delta.** v1.0 used `Orog(t) − Orog(0)`. ICE-6G_C `Orog` is clamped to 0 over
  water and referred to the contemporaneous sea surface, so the correction vanished
  under the palaeo-shoreline and carried the eustatic term over land. The engine now
  uses the present-datum displacement `Δz_solid = [Topo(t) − Topo(0)] − offset(t)`
  (`paleoeurope/gia/ice6g_topo.py`), with `offset(t)` referenced to far-field,
  never-glaciated tropical land; the land/sea threshold is `−offset(t)`. The loader
  refuses the `Orog` path.
- **Sea-level curve.** The file used as "Spratt & Lisiecki 2016" was a hand-drawn
  sketch. The engine now reads the published shortPC1 series (pinned by tests) and
  interpolates half-step epochs linearly instead of snapping to a neighbour.
- Closed basins (Black Sea, Caspian) get their own levels, converted between frames
  (`pontic_threshold`), instead of the global eustatic value.

### Known limitations

- Cells covered by ice at either endpoint have no bedrock estimate (NaN): the
  10-arcmin ICE-6G_C files carry no ice thickness.
- `tests/test_viz.py` is a stub that fails on purpose (see its docstring).
- Two legacy figure scripts, `scripts/render_black_sea_corrected.py` and
  `scripts/render_black_sea_hero.py`, still read the sketch curve
  (`spratt_lisiecki_2016_simplified.csv`). Do not use them for science.

## [0.1.0-alpha] — 2026-02-21

### Added

- Initial repository structure following C-1 specification from PAPER1_SUBMISSION_PLAN.md
- `paleoeurope.fusion` package: FABDEM loader, GEBCO loader, EGM2008 datum corrector, alpha blender, fusion pipeline
- `paleoeurope.gia` package: ICE-6G_C loader, ICE-7G_NA loader, delta-method deformation, ice envelope
- `paleoeurope.utils` package: raster and grid utilities
- `scripts/generate_synthetic_data.py` — deterministic synthetic test data (seed=42)
- `tests/` — unit tests for fusion and GIA with synthetic data
- `configs/default.yml` and `configs/europe_full.yml`
- GitHub Actions CI: test, lint, notebook execution
- MIT LICENSE, CITATION.cff

### Not yet included (Paper 2 scope)

- Topological analysis (persistent homology)
- WhiteboxTools hydrology pipeline (HAND, terraces)
- Blender/mesh generation
