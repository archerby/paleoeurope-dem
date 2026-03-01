# Data Catalogue — paleoeurope-dem pipeline

All datasets used by the pipeline.  Column meanings:
- **Size** — approximate uncompressed size for the Europe domain
- **Bundled** — included directly in this repository
- **Download script** — key in `scripts/download_sample_data.py`

---

## 1. FABDEM v1.2
| Field | Value |
|-------|-------|
| **Full name** | Forest And Buildings removed Copernicus DEM |
| **Authors** | Hawker, L. et al. |
| **Year** | 2022 |
| **License** | CC BY 4.0 |
| **DOI** | [10.5194/essd-14-4677-2022](https://doi.org/10.5194/essd-14-4677-2022) |
| **Download** | https://doi.org/10.5523/bris.25wfy0f9ukibjp75gb8laq1ghq |
| **Format** | GeoTIFF, 1°×1° tiles, Float32, ~30 m resolution |
| **Coverage** | Global, 80°S–80°N |
| **Size** | ~8 000 tiles × ~5 MB = ~40 GB (Europe subset ≈ 2 GB) |
| **Bundled** | ❌ — download via script |
| **Download script** | `fabdem` |
| **Pipeline use** | Land terrain base for coastal fusion |

---

## 2. GEBCO 2024
| Field | Value |
|-------|-------|
| **Full name** | General Bathymetric Chart of the Oceans, 2024 edition |
| **Authors** | GEBCO Compilation Group |
| **Year** | 2024 |
| **License** | CC-0 Public Domain |
| **DOI** | [10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f](https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f) |
| **Download** | https://download.gebco.net/ |
| **Format** | NetCDF / GeoTIFF, 15 arc-second grid, Int16 |
| **Coverage** | Global |
| **Size** | ~7 GB (global); clipped Europe bbox ≈ 500 MB |
| **Bundled** | ❌ — download via script |
| **Download script** | `gebco` |
| **Pipeline use** | Bathymetry fill for coastal fusion |

---

## 3. EGM2008 Geoid
| Field | Value |
|-------|-------|
| **Full name** | Earth Gravitational Model 2008 |
| **Authors** | Pavlis, N. K. et al. |
| **Year** | 2012 |
| **License** | Public Domain (US Government work) |
| **DOI** | [10.1029/2011JB008916](https://doi.org/10.1029/2011JB008916) |
| **Download** | https://earth-info.nga.mil/index.php?dir=wgs84 |
| **Format** | GeoTIFF / ASCII, 2.5 arc-minute grid, Float32 |
| **Coverage** | Global |
| **Size** | ~200 MB (global) |
| **Bundled** | ❌ — download via script |
| **Download script** | `egm2008` |
| **Pipeline use** | Geoid undulation correction (FABDEM ↔ GEBCO datum match) |

---

## 4. ICE-6G_C (VM5a)
| Field | Value |
|-------|-------|
| **Full name** | ICE-6G_C VM5a glacial isostatic adjustment model |
| **Authors** | Peltier, W. R. et al. |
| **Year** | 2015 |
| **License** | Academic use — cite Peltier et al. (2015) |
| **DOI** | [10.1002/2014JB011176](https://doi.org/10.1002/2014JB011176) |
| **Download** | https://www.physics.utoronto.ca/~peltier/data.html |
| **Format** | NetCDF, 0.5° global grid, Float32, 0–26 ka |
| **Coverage** | Global |
| **Size** | ~50 MB (full); Europe subset 3 epochs ≈ 3 MB |
| **Bundled** | ❌ — download via script |
| **Download script** | `ice6g` |
| **Pipeline use** | GIA deformation correction (Stages 0, and GIA module) |

---

## 5. ICE-7G_NA (VM7)
| Field | Value |
|-------|-------|
| **Full name** | ICE-7G_NA VM7 North American/global ice model |
| **Authors** | Roy, K. & Peltier, W. R. |
| **Year** | 2018 |
| **License** | Academic use — cite Roy & Peltier (2018) |
| **DOI** | [10.1016/j.quascirev.2017.11.016](https://doi.org/10.1016/j.quascirev.2017.11.016) |
| **Download** | https://www.physics.utoronto.ca/~peltier/data.html |
| **Format** | NetCDF, 1° global grid, 45 epochs (0–26 ka at 0.5 ka steps) |
| **Coverage** | Global |
| **Size** | ~45 × 1° tiles per epoch |
| **Bundled** | ❌ — download separately |
| **Download script** | *(not yet — add `ice7g` entry when URL confirmed)* |
| **Pipeline use** | Ice-sheet thickness overlay in `viz` renders (white mask layer) |

---

## 6. Paleocoastlines GIS Dataset  ⬅ **required for Paleocostline Overlay Render**
| Field | Value |
|-------|-------|
| **Full name** | Paleocoastlines GIS dataset |
| **Authors** | Zickel, M., Becker, D., Verheul, J., Yener, Y. & Willmes, C. |
| **Year** | 2016 |
| **License** | CC BY 4.0 |
| **DOI** | [10.5880/SFB806.19](https://doi.org/10.5880/SFB806.19) |
| **Download** | http://crc806db.uni-koeln.de/layer/show/327/ |
| **Format** | FlatGeobuf (`.fgb`) — clipped derivative; full Shapefile available via DOI |
| **Coverage (bundled)** | −5°→15°E, 50°→63°N  (North Sea + Danish/Norwegian shelf) |
| **Coverage (full)** | Europe + Near East + N. Africa (−21°W–62°E, 1°N–62°N) |
| **Size (bundled)** | ~12 MB — **committed to repo** at `data/paleocostlines/paleocoastlines_north_sea.fgb` |
| **Size (full)** | ~46 MB (zip); ~94 MB extracted |
| **Bundled** | ✅ `data/paleocostlines/paleocoastlines_north_sea.fgb` |
| **Download script** | `paleocoastlines` (downloads full global dataset) |
| **SHA256 (full zip)** | `c6cba2a7dda0af3cfb62980b6914842cbe3e25358c5a77528aa7c8cd541a0735` |
| **Pipeline use** | Land/sea binary mask per epoch in `paleoeurope.viz.render_paleocostline_epoch` |

> **Derivation note**: the bundled FlatGeobuf was produced by
> `scripts/prepare_paleocostlines_clip.py` — see
> `data/paleocostlines/README.md` for full provenance.  Geometries simplified
> at 0.001° ≈ 100 m; all 24 sea-level steps retained.

### Sea-level steps in the dataset
The dataset contains **24 sea-level steps** (metres relative to present):

```
−130, −120, −115, −81, −80, −77, −74, −65, −50, −45, −40,
−27, −25, −24, −21, −20, −19, −18, −16, −14, −5, −3, 0, +5
```

The `filter_paleocostlines_for_epoch` function selects the closest step to the
Spratt & Lisiecki (2016) sea level for the requested epoch.

---

## 7. Spratt & Lisiecki (2016) Sea-Level Curve  ⬅ **bundled in repo**
| Field | Value |
|-------|-------|
| **Full name** | A Late Pleistocene sea-level stack |
| **Authors** | Spratt, R. M. & Lisiecki, L. E. |
| **Year** | 2016 |
| **License** | CC BY 3.0 |
| **DOI** | [10.5194/cp-12-1079-2016](https://doi.org/10.5194/cp-12-1079-2016) |
| **Full data DOI** | [10.1594/PANGAEA.856145](https://doi.org/10.1594/PANGAEA.856145) |
| **Format** | CSV (3 columns: age_bp, sea_level_m, uncertainty_m) |
| **Coverage** | 0–800 ka (full); bundled extract covers **0–26 ka at 0.5 ka steps** |
| **Size** | 4 KB (bundled extract) |
| **Bundled** | ✅ `data/sea_level_curves/spratt_lisiecki_2016_simplified.csv` |
| **Download script** | `spratt_csv` (downloads full PANGAEA tabular data) |
| **Pipeline use** | Maps epoch → sea level → selects Paleocoastlines step |

> **Note**: The bundled CSV covers 0–26 ka.  For older epochs, download the  
> full PANGAEA dataset:  
> `python scripts/download_sample_data.py --only spratt_csv`

---

## Reproducibility notes

All datasets are publicly available under open licenses or academic-use terms.
To reproduce the paper results:

```bash
# 1. Download all sample datasets (~60 MB total)
python scripts/download_sample_data.py

# 2. Download Paleocoastlines specifically
python scripts/download_sample_data.py --only paleocoastlines

# 3. Verify checksums
python scripts/download_sample_data.py --only paleocoastlines --skip-verify=false
```

Datasets are never stored in the git repository (except the bundled Spratt CSV).
SHA256 checksums are recorded in `tests/data/real_sample/checksums.sha256` after
first download.
