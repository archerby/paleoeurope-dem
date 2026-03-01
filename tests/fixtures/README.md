# Test Fixtures — Real Data Subsets

These files are **committed to git** and provide a reproducible, real-data
baseline for the pipeline test suite.  No download is needed to run `pytest`.

## Provenance

| File | Source | Licence | Generated from |
|------|--------|---------|---------------|
| `fabdem/N62E006_FABDEM_V1-2_fixture.tif` | Hawker et al. (2022) doi:10.5194/essd-14-4677-2022 | CC BY 4.0 | N62E006_FABDEM_V1-2.tif — full 1°×1° tile, 3600×3600 px |
| `gebco/GEBCO_2024_fixture.tif` | GEBCO Compilation Group (2024) doi:10.5285/1c44ce99-… | CC0 | Regional GeoTIFF, clipped to full 1° tile |
| `egm2008/egm2008_fixture.tif` | Pavlis et al. (2012) doi:10.1029/2011JB008916 | Public Domain | us_nga_egm08_25.tif, clipped |
| `ice6g/ICE6G_fixture.nc` | Peltier et al. (2015) doi:10.1002/2014JB011176 | Cite | I6_C.VM5a_10min.{0..21}.nc, 43 epochs 0–21 ka |

## Coverage

Tile: **N62E006** (62-63°N, 6-7°E — Sunnmøre / Ålesund, Norway)  
Bbox: `6.0°E  62.0°N → 7.0°E  63.0°N`

Full 1°×1° square covering Norwegian fjords (~49 % land, ~51 % ocean).
Exercises coastal blending (`RasterBlender`), datum correction
(`DatumCorrector`) and GIA deformation (`Ice6gLoader`) with real topography.

## Regeneration

If you need to regenerate these fixtures (e.g. after updating the source
data version), run from the repo root:

```bash
python scripts/prepare_test_fixtures.py --force
```

Source data must be available at the local paths listed at the top of that
script.  The fixtures directory is **tracked by git** (`tests/fixtures/`
is NOT in `.gitignore`).
