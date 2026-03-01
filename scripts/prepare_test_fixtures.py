#!/usr/bin/env python3
"""
scripts/prepare_test_fixtures.py

Cut a full 1°×1° fixture set from LOCAL real data and write it to
tests/fixtures/ so it can be committed to git alongside the code.

Source data (local, not redistributed via git)
----------------------------------------------
  FABDEM   : /home/pp/paleo_project/data/raw/FABDEM/         (CC BY 4.0)
  GEBCO    : /home/pp/paleo_project/data/raw/GEBCO/…/*.tif   (CC0)
  EGM2008  : /home/pp/paleo_project/data/raw/GSHHG/egm2008/  (PD)
  ICE-6G   : /home/pp/paleo_project/data/raw/ICE6G/          (cite only)
    ICE-7G   : /home/pp/paleo_project/data/raw/ICE7G/          (cite only)

Output (committed to git)
-----------------------------------------
tests/fixtures/
├── README.md                  ← provenance + attributions
├── fabdem/
│   └── N62E006_FABDEM_V1-2_fixture.tif   3600×3600 px  1 arc-sec (full tile)
├── gebco/
│   └── GEBCO_2024_fixture.tif            ~240×240 px  15 arc-sec (full 1° tile)
├── egm2008/
│   └── egm2008_fixture.tif               ~5×5 px      2.5 arc-min
└── ice6g/
    └── ICE6G_fixture.nc                  full 1° bbox, 43 epochs: 0–21 ka (0.5 ka steps)

Optionally (if local ICE-7G sources are present), this script can also write:

└── ice7g/
    └── ICE7G_fixture.nc                  clipped bbox, selected epochs (for open-demo DMK)

Tile: N62E006 (62-63°N, 6-7°E) — Sunnmøre / Ålesund, Norway.
      Full 1°×1° square so that fusion pipeline and GIA correction operate
      on complete, non-cropped input data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
FIXTURES  = REPO_ROOT / "tests" / "fixtures"

# ── local source paths ───────────────────────────────────────────────────────
FAB_SRC  = Path("/home/pp/paleo_project/data/raw/FABDEM/N62E006_FABDEM_V1-2.tif")
GEBCO_SRC = next(
    Path("/home/pp/paleo_project/data/raw/GEBCO").rglob("*.tif"), None
)
EGM_SRC  = Path("/home/pp/paleo_project/data/raw/GSHHG/egm2008/us_nga_egm08_25.tif")
ICE6G_DIR = Path("/home/pp/paleo_project/data/raw/ICE6G")
ICE7G_DIR = Path("/home/pp/paleo_project/data/raw/ICE7G")

TILE_ID   = "N62E006"   # Sunnmøre / Ålesund, Norway — 49 % land, 51 % ocean
# Full 1°×1° tile bounds (SW corner derived from TILE_ID)
TILE_BOUNDS = (6.0, 62.0, 7.0, 63.0)   # lon_min, lat_min, lon_max, lat_max
# ICE6G epochs to include: 0 ka to 21 ka in 0.5 ka steps
ICE6G_EPOCHS = [round(k * 0.5, 1) for k in range(43)]   # 0.0 … 21.0 ka

# ICE-7G epochs needed by the publication/demo notebooks.
# These correspond to per-epoch files named like: I7G_NA.VM7_1deg.{epoch}.nc
ICE7G_EPOCHS = [0, 8, 12, 21]


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "fabdem").mkdir(exist_ok=True)
    (FIXTURES / "gebco").mkdir(exist_ok=True)
    (FIXTURES / "egm2008").mkdir(exist_ok=True)
    (FIXTURES / "ice6g").mkdir(exist_ok=True)
    (FIXTURES / "ice7g").mkdir(exist_ok=True)

    _cut_fabdem()
    _cut_gebco()
    _cut_egm2008()
    _cut_ice6g()
    _cut_ice7g()
    _write_readme()

    print(f"\n✓  Fixtures written to {FIXTURES}")
    _print_sizes()
    return 0


# ── FABDEM ────────────────────────────────────────────────────────────────────

def _cut_fabdem() -> None:
    """Copy the full 1°×1° FABDEM tile, remapping water (0.0) → nodata (-9999)."""
    import rasterio

    fname = FAB_SRC.stem + "_fixture.tif"   # N62E006_FABDEM_V1-2_fixture.tif
    dest = FIXTURES / "fabdem" / fname
    if dest.exists():
        print(f"  [fabdem]  {dest.name} already present")
        return

    print(f"  [fabdem]  Reading full tile {FAB_SRC.name} …")
    with rasterio.open(FAB_SRC) as src:
        data = src.read(1).astype(np.float32)
        t = src.transform
        crs = src.crs
        h, w = src.height, src.width

    # FABDEM always stores water bodies as 0.0 (sea level) regardless of what
    # the nodata header tag says (which is -9999 for "outside coverage").
    # Remap 0.0 → -9999 so the blender can distinguish land from ocean.
    out_nodata = -9999.0
    data[data == 0.0] = out_nodata

    _write_tif(dest, data, t, nodata=out_nodata, crs=crs)
    print(f"  [fabdem]  ✓ {dest.name}  {w}×{h} px  ({_sz(dest)})")


# ── GEBCO ─────────────────────────────────────────────────────────────────────

def _cut_gebco() -> None:
    """Clip GEBCO to the exact 1°×1° tile bounds (TILE_BOUNDS)."""
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds

    dest = FIXTURES / "gebco" / "GEBCO_2024_fixture.tif"
    if dest.exists():
        print(f"  [gebco]   {dest.name} already present")
        return

    if GEBCO_SRC is None:
        print("  [gebco]   GEBCO source not found — skipping", file=sys.stderr)
        return

    lon_min, lat_min, lon_max, lat_max = TILE_BOUNDS
    buf = 0.1  # small buffer to avoid edge artefacts after resampling
    print(f"  [gebco]   Clipping {GEBCO_SRC.name} to {TILE_BOUNDS} …")

    with rasterio.open(GEBCO_SRC) as src:
        win = win_from_bounds(
            lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf,
            src.transform,
        )
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)
        t_win = src.window_transform(win)
        _write_tif(dest, data.astype(np.float32), t_win,
                   nodata=float(src.nodata or -9999), crs=src.crs)

    print(f"  [gebco]   ✓ {dest.name}  {data.shape[1]}×{data.shape[0]} px  ({_sz(dest)})")


def _cut_ice7g() -> None:
    """Build a small ICE-7G thickness fixture (stgit) for the demo epochs.

    Expects local ICE-7G per-epoch NetCDF files, e.g.:
      ICE7G_DIR/I7G_NA.VM7_1deg.21.nc
    """
    import xarray as xr

    dest = FIXTURES / "ice7g" / "ICE7G_fixture.nc"
    if dest.exists():
        print(f"  [ice7g]   {dest.name} already present")
        return

    if not ICE7G_DIR.exists():
        print("  [ice7g]   ICE-7G source dir not found — skipping", file=sys.stderr)
        return

    lon_min, lat_min, lon_max, lat_max = TILE_BOUNDS
    buf = 1.0
    ds_list = []

    for ep in ICE7G_EPOCHS:
        src = ICE7G_DIR / f"I7G_NA.VM7_1deg.{int(ep)}.nc"
        if not src.exists():
            print(f"  [ice7g]   missing {src.name} — skipping ICE-7G fixture", file=sys.stderr)
            return

        with xr.open_dataset(src) as ds:
            if "stgit" not in ds.data_vars:
                raise KeyError(f"ICE-7G file lacks 'stgit': {src}")

            lat_dim = next((d for d in ("lat", "latitude", "Lat") if d in ds.dims or d in ds.coords), None)
            lon_dim = next((d for d in ("lon", "longitude", "Lon") if d in ds.dims or d in ds.coords), None)
            if lat_dim is None or lon_dim is None:
                raise KeyError(f"Cannot detect lat/lon dims in {src}; dims={list(ds.dims)}")

            lat = ds[lat_dim].values
            lon = ds[lon_dim].values
            lat_sel = (lat >= lat_min - buf) & (lat <= lat_max + buf)
            lon_sel = (lon >= lon_min - buf) & (lon <= lon_max + buf)

            stgit = ds["stgit"].isel({lat_dim: lat_sel, lon_dim: lon_sel}).astype(np.float32)
            stgit = stgit.rename({lat_dim: "lat", lon_dim: "lon"})
            ds_list.append(stgit.expand_dims({"time_ka": [float(ep)]}))

    stgit_all = xr.concat(ds_list, dim="time_ka")
    out = xr.Dataset({"stgit": stgit_all})

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(dest)
    print(f"  [ice7g]   ✓ {dest.name}  ({_sz(dest)})")


# ── EGM2008 ───────────────────────────────────────────────────────────────────

def _cut_egm2008() -> None:
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds

    dest = FIXTURES / "egm2008" / "egm2008_fixture.tif"
    if dest.exists():
        print(f"  [egm2008] {dest.name} already present")
        return

    lon_min, lat_min, lon_max, lat_max = TILE_BOUNDS
    buf = 1.0  # EGM2008 is coarse (2.5 arcmin), wide buffer needed for interp
    print(f"  [egm2008] Clipping {EGM_SRC.name} …")

    with rasterio.open(EGM_SRC) as src:
        win = win_from_bounds(
            lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf,
            src.transform,
        )
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)
        t_win = src.window_transform(win)
        _write_tif(dest, data.astype(np.float32), t_win,
                   nodata=float(src.nodata or -9999), crs=src.crs)

    print(f"  [egm2008] ✓ {dest.name}  {data.shape[1]}×{data.shape[0]} px  ({_sz(dest)})")


# ── ICE-6G ────────────────────────────────────────────────────────────────────

def _cut_ice6g() -> None:
    """Merge all ICE6G_EPOCHS (0–21 ka, 0.5 ka steps) into one NetCDF for TILE_BOUNDS."""
    import netCDF4 as nc

    dest = FIXTURES / "ice6g" / "ICE6G_fixture.nc"
    if dest.exists():
        print(f"  [ice6g]   {dest.name} already present")
        return

    lon_min, lat_min, lon_max, lat_max = TILE_BOUNDS
    buf = 5.0  # wide: ICE-6G is 10-arcmin, needs generous margin

    # Build epoch → filename-suffix map (0 → "0", 0.5 → "0.5", 1 → "1", …)
    def _suffix(ka: float) -> str:
        return str(int(ka)) if ka == int(ka) else str(ka)

    epochs = {ka: _suffix(ka) for ka in ICE6G_EPOCHS}
    data_by_epoch: dict[float, dict] = {}

    for epoch_ka, suffix in epochs.items():
        nc_path = ICE6G_DIR / f"I6_C.VM5a_10min.{suffix}.nc"
        if not nc_path.exists():
            print(f"  [ice6g]   WARNING: {nc_path.name} not found", file=sys.stderr)
            continue

        with nc.Dataset(str(nc_path)) as ds:
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            lat_indexes = np.where((lats >= lat_min - buf) & (lats <= lat_max + buf))[0]
            lon_indexes = np.where((lons >= lon_min - buf) & (lons <= lon_max + buf))[0]
            lat_slice = slice(int(lat_indexes[0]), int(lat_indexes[-1]) + 1)
            lon_slice = slice(int(lon_indexes[0]), int(lon_indexes[-1]) + 1)

            data_by_epoch[epoch_ka] = {
                "lats": lats[lat_slice],
                "lons": lons[lon_slice],
            }
            for vname in ("Topo", "Orog", "sftlf", "sftgif"):
                if vname in ds.variables:
                    v = ds.variables[vname]
                    arr = v[lat_slice, lon_slice]
                    data_by_epoch[epoch_ka][vname] = dict(
                        data=np.array(arr, dtype=np.float32),
                        units=getattr(v, "units", ""),
                        long_name=getattr(v, "long_name", vname),
                    )

    if not data_by_epoch:
        print("  [ice6g]   ERROR: no epoch files found", file=sys.stderr)
        return

    found = sorted(data_by_epoch.keys())
    missing = [e for e in ICE6G_EPOCHS if e not in data_by_epoch]
    if missing:
        print(f"  [ice6g]   WARNING: {len(missing)} epochs not found: {missing[:5]}…")
    print(f"  [ice6g]   Found {len(found)} epochs: {found[0]}–{found[-1]} ka")
    epochs_list = found
    ref = data_by_epoch[epochs_list[0]]
    nlat, nlon = len(ref["lats"]), len(ref["lons"])

    with nc.Dataset(str(dest), "w", format="NETCDF4") as out:
        out.title = (
            "ICE-6G_C VM5a fixture for paleoeurope tests. "
            f"Real data subset: tile {TILE_ID} bbox {TILE_BOUNDS}. "
            "Cite: Peltier et al. 2015 doi:10.1002/2014JB011176"
        )
        out.Conventions = "CF-1.8"
        out.source = (
            f"I6_C.VM5a_10min.{{0..21}}.nc — real ICE-6G_C VM5a data, "
            f"{len(epochs_list)} epochs 0–21 ka at 0.5 ka steps"
        )

        out.createDimension("time", len(epochs_list))
        out.createDimension("lat", nlat)
        out.createDimension("lon", nlon)

        t_v = out.createVariable("time", "f4", ("time",))
        t_v[:] = np.array(epochs_list, dtype=np.float32) * -1  # 0 → 0, 21 → -21
        t_v.units = "ka BP"
        t_v.long_name = "model time (0=modern, negative=past)"

        lat_v = out.createVariable("lat", "f4", ("lat",))
        lat_v[:] = ref["lats"]
        lat_v.units = "degrees_north"

        lon_v = out.createVariable("lon", "f4", ("lon",))
        lon_v[:] = ref["lons"]
        lon_v.units = "degrees_east"

        # Write variables present in the first epoch
        for vname in ("Topo", "Orog", "sftlf", "sftgif"):
            if vname not in ref:
                continue
            v_out = out.createVariable(vname, "f4", ("time", "lat", "lon"),
                                       fill_value=-9999.0)
            stack = np.stack(
                [data_by_epoch[e].get(vname, {}).get("data",
                   np.full((nlat, nlon), -9999.0, dtype=np.float32))
                 for e in epochs_list],
                axis=0,
            )
            v_out[:] = stack
            v_out.units = ref[vname]["units"]
            v_out.long_name = ref[vname]["long_name"]

        # Also expose Topo as "stgr" alias (what our pipeline expects for GIA delta)
        # GIA delta = Topo[t] - Topo[0]
        if "Topo" in ref:
            topo_stack = np.stack(
                [data_by_epoch[e].get("Topo", {}).get("data",
                   np.full((nlat, nlon), 0.0, dtype=np.float32))
                 for e in epochs_list], axis=0,
            )
            topo_modern = topo_stack[0:1]          # epoch 0 ka
            gia_delta = topo_stack - topo_modern   # delta from modern
            stgr = out.createVariable("stgr", "f4", ("time", "lat", "lon"),
                                      fill_value=-9999.0)
            stgr[:] = gia_delta.astype(np.float32)
            stgr.units = "m"
            stgr.long_name = "Bedrock deformation delta from modern (GIA)"

        # Ice thickness alias  (sftgif: ice fraction 0–1 → proxy mask)
        sice = out.createVariable("sice", "f4", ("time", "lat", "lon"),
                                  fill_value=-9999.0)
        if "sftgif" in ref:
            ice_stack = np.stack(
                [data_by_epoch[e].get("sftgif", {}).get("data",
                   np.zeros((nlat, nlon), dtype=np.float32))
                 for e in epochs_list], axis=0,
            )
            # sftgif is 0–1 fraction; scale to 0–3000 m range for thickness proxy
            sice[:] = (ice_stack * 3000.0).astype(np.float32)
        else:
            sice[:] = np.zeros((len(epochs_list), nlat, nlon), dtype=np.float32)
        sice.units = "m"
        sice.long_name = "Ice thickness (proxy from sftgif * 3000 m)"

        # RSL placeholder — sealevel is NOT used for GIA delta in the pipeline;
        # the delta is pre-computed in 'stgr' (= Topo[t] − Topo[0]) above.
        # Ice6gLoader detects 'stgr' and uses it directly, bypassing sealevel.
        # We keep this variable for schema compatibility with real ICE-6G files.
        rsl = out.createVariable("sealevel", "f4", ("time", "lat", "lon"),
                                 fill_value=-9999.0)
        rsl[:] = np.zeros((len(epochs_list), nlat, nlon), dtype=np.float32)
        rsl.units = "m"
        rsl.long_name = "Relative sea level change (placeholder — GIA delta is in stgr)"

    print(f"  [ice6g]   ✓ {dest.name}  {nlat}×{nlon} pts  ({_sz(dest)})")


# ── README ────────────────────────────────────────────────────────────────────

def _write_readme() -> None:
    dest = FIXTURES / "README.md"
    lon_min, lat_min, lon_max, lat_max = TILE_BOUNDS
    n_epochs = len(ICE6G_EPOCHS)
    content = f"""# Test Fixtures — Real Data Subsets

These files are **committed to git** and provide a reproducible, real-data
baseline for the pipeline test suite.  No download is needed to run `pytest`.

## Provenance

| File | Source | Licence | Generated from |
|------|--------|---------|---------------|
| `fabdem/{FAB_SRC.stem}_fixture.tif` | Hawker et al. (2022) doi:10.5194/essd-14-4677-2022 | CC BY 4.0 | {FAB_SRC.name} — full 1°×1° tile, 3600×3600 px |
| `gebco/GEBCO_2024_fixture.tif` | GEBCO Compilation Group (2024) doi:10.5285/1c44ce99-… | CC0 | Regional GeoTIFF, clipped to full 1° tile |
| `egm2008/egm2008_fixture.tif` | Pavlis et al. (2012) doi:10.1029/2011JB008916 | Public Domain | us_nga_egm08_25.tif, clipped |
| `ice6g/ICE6G_fixture.nc` | Peltier et al. (2015) doi:10.1002/2014JB011176 | Cite | I6_C.VM5a_10min.{{0..21}}.nc, {n_epochs} epochs 0–21 ka |

## Coverage

Tile: **{TILE_ID}** (62-63°N, 6-7°E — Sunnmøre / Ålesund, Norway)
Bbox: `{lon_min:.1f}°E  {lat_min:.1f}°N → {lon_max:.1f}°E  {lat_max:.1f}°N`

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
"""
    dest.write_text(content)
    print(f"  [readme]  {dest.name} written")


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_tif(path: Path, data: np.ndarray, transform, nodata: float,
               crs=None) -> None:
    import rasterio
    from rasterio.crs import CRS

    arr = data.astype(np.float32)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = nodata

    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32",
        crs=crs or CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
        compress="LZW",
        tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr, 1)


def _sz(p: Path) -> str:
    n = p.stat().st_size
    for u in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n //= 1024
    return f"{n:.1f} GB"


def _print_sizes() -> None:
    total = 0
    for f in sorted(FIXTURES.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            print(f"    {f.relative_to(FIXTURES)}  {sz//1024} KB")
    print(f"  Total: {total//1024} KB  ({total/1024/1024:.2f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing fixtures")
    args = parser.parse_args()
    if args.force:
        import shutil
        shutil.rmtree(FIXTURES, ignore_errors=True)
        print("  [force] Removed existing fixtures")
    sys.exit(main())
