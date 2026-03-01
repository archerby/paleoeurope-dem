# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
