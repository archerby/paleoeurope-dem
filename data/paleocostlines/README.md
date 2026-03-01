# Paleocoastlines — North Sea / European Shelf Clip

## File

| File | Format | Size | Area |
|------|--------|------|------|
| `paleocoastlines_north_sea.fgb` | FlatGeobuf | ~12 MB | −5°→15°E, 50°→63°N |

## Contents

24 sea-level steps (land polygons) derived from the global Paleocoastlines GIS
dataset (Zickel et al. 2016). Each polygon represents the land extent at one of
the following sea-level configurations:

```
−130, −120, −115, −81, −80, −77, −74, −65, −50, −45, −40,
−27, −25, −24, −21, −20, −19, −18, −16, −14, −5, −3, 0, +5 m
```

Column: **`Sea level`** (integer, metres relative to present).

## Source & Licence

> Zickel, M., Knitter, D., Jung, R., Alt, K.W., Kjeld, A., Kehl, M., Weninger,
> B., Edinborough, K., Zimmermann, A. (2016). **Paleocoastlines GIS dataset**.
> *CRC806-Database*.
> DOI: [10.5880/SFB806.19](https://doi.org/10.5880/SFB806.19)
>
> Licence: **CC BY 4.0**
> Full download: <http://crc806db.uni-koeln.de/layer/show/327/>

SHA-256 of the full source zip (`Paleocoastlines.zip`):
`c6cba2a7dda0af3cfb62980b6914842cbe3e25358c5a77528aa7c8cd541a0735`

## Derivation

This file was produced by `scripts/prepare_paleocostlines_clip.py`:
1. Load global SHP with `pyogrio` (`on_invalid='ignore'`)
2. Repair invalid geometries with `shapely.validation.make_valid`
3. Clip to bbox `(−5, 50, 15, 63)` — covers North Sea + Norwegian margin
4. Simplify at 0.001° tolerance (~100 m) — preserves all coastline features
   relevant to 1 : 250 000-scale paleo-DEM visualisations
5. Export as FlatGeobuf (compact, streamable, single-file)

To regenerate from the full global dataset:
```
python scripts/prepare_paleocostlines_clip.py \
  --src /path/to/Paleocoastlines.shp \
  --out data/paleocostlines/paleocoastlines_north_sea.fgb
```
