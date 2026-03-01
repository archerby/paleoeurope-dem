#!/usr/bin/env python3
"""
scripts/generate_synthetic_data.py   — fallback data generator (CI / offline)

Primary test data (committed to git, always preferred):
  tests/fixtures/  — real data subsets cut by prepare_test_fixtures.py

This script is a FALLBACK for environments where real fixtures are absent
(e.g. a fresh clone without git-lfs, or CI with network access to download
FABDEM/GEBCO/EGM2008).  The conftest.py session fixture calls this script
only when tests/fixtures/ is incomplete.

Downloads real 1°×1° sample data for tile N51E000 (English Channel coast)
---  FABDEM, GEBCO, EGM2008 are downloaded from their official sources.
---  ICE-6G remains synthetic (see NOTE below) and is saved to tests/data/.

NOTE on ICE-6G:
  Full ICE-6G_C VM5a dataset (~50 MB) requires registration with the Peltier
  group (University of Toronto).  Committed real fixtures (tests/fixtures/)
  contain a genuine ICE-6G subset produced by prepare_test_fixtures.py from
  locally obtained files.  The synthetic fallback produced here (ICE6G_sample.nc)
  uses calibrated-but-random values and is labelled "SYNTHETIC" in its metadata.
  NEVER use tests/data/ice6g/ICE6G_sample.nc as a figure or result source.

Downloaded datasets (when run online)
-------------------
  FABDEM v1.2 — tile N51E000_V1-2_FABDEM.tif
    Source  : University of Bristol data store (CC BY 4.0)
    doi     : 10.5194/essd-14-4677-2022  (Hawker et al. 2022)
    Size    : ~10 MB  (3600×3600 px, 1 arc-sec, float32)

  GEBCO 2024 — GEBCO_2024_N51E000.tif
    Source  : GEBCO WCS  (CC0 Public Domain)
    doi     : 10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f
    Size    : ~0.5 MB  (240×240 px, 15 arc-sec, int16)

  EGM2008 — egm2008_N51E000.tif
    Source  : OpenTopography / NGA  (Public Domain)
    doi     : 10.1029/2011JB008916  (Pavlis et al. 2012)
    Size    : ~250 KB  (clipped from global 2.5 arc-min grid)

  ICE-6G_C VM5a — ICE6G_sample.nc   ← SYNTHETIC FALLBACK
    Reason  : Full dataset requires registration; see NOTE above.
    Cite    : 10.1002/2014JB011176  (Peltier et al. 2015)

Output structure (tests/data/, gitignored — not the committed fixtures)
--------
tests/data/
├── .generated                        ← sentinel
├── tile_list.txt                     ← single entry: N51E000
├── fabdem/
│   └── N51E000_V1-2_FABDEM.tif      ← REAL if download succeeds; SYNTHETIC fallback
├── gebco/
│   └── GEBCO_2024_N51E000.tif       ← REAL if download succeeds; SYNTHETIC fallback
├── egm2008/
│   └── egm2008_N51E000.tif          ← REAL if download succeeds; SYNTHETIC fallback
└── ice6g/
    └── ICE6G_sample.nc              ← ALWAYS SYNTHETIC (see NOTE above)

Fallback behaviour
------------------
If any network download fails (CI without internet, rate-limit, etc.) the
script falls back to generating realistic synthetic data for FABDEM/GEBCO/EGM2008
only.  A WARNING is printed for each fallback.  ICE-6G is always synthetic.
When tests/fixtures/ is present (normal case), this script is not called at all.

Usage
-----
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --output-dir tests/data
    python scripts/generate_synthetic_data.py --force       # re-download
    python scripts/generate_synthetic_data.py --offline     # synthetic only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── tile constants ──────────────────────────────────────────────────────────
TILE_ID = "N51E000"
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 0.0, 51.0, 1.0, 52.0

# buffer around the tile used for EGM2008 / ICE-6G to avoid edge interpolation
BUFFER = 1.0

# ── upstream URLs ───────────────────────────────────────────────────────────
FABDEM_ZIP_URL = (
    "https://data.bris.ac.uk/datasets/d5hqmjcdj8yo2ibzi9b4ew3sn/"
    "N51E000_V1-2_FABDEM.tif.zip"
)

# GEBCO WCS 2.0 — returns a clipped GeoTIFF, no auth required
GEBCO_WCS_URL = (
    "https://www.gebco.net/data_and_products/gebco_web_services/web_map_services/"
    "mapserv?service=WCS&version=2.0.1&request=GetCoverage"
    "&coverageid=gebco_latest&format=image/tiff"
    "&subset=Lat(51,52)&subset=Long(0,1)"
)

# OpenTopography global EGM2008 2.5 arc-min GeoTIFF (Public Domain, ~5 MB)
EGM2008_GLOBAL_URL = (
    "https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/"
    "OTDS.012013.4326.1/raster/egm2008-25.tif"
)

# ── synthetic fallback resolution ───────────────────────────────────────────
# 360 px/° ≈ 10 arc-sec — realistic size, fast to generate
SYNTH_PX_DEG = 360
SEED = 42


# ===========================================================================
# Main
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tests" / "data"),
        help="Destination directory (default: tests/data)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all downloads; generate realistic synthetic data only",
    )
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    sentinel = out / ".generated"

    if sentinel.exists() and not args.force:
        print(f"[data] Already present at {out}  (use --force to re-download)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    _get_fabdem(out, offline=args.offline)
    _get_gebco(out, offline=args.offline)
    _get_egm2008(out, offline=args.offline)
    _generate_ice6g(out, rng)          # always synthetic — see module docstring
    _write_tile_list(out)

    sentinel.touch()
    print(f"[data] Done.  Output: {out}")
    return 0


# ===========================================================================
# FABDEM
# ===========================================================================


def _get_fabdem(base: Path, offline: bool) -> None:
    fab_dir = base / "fabdem"
    fab_dir.mkdir(parents=True, exist_ok=True)
    dest = fab_dir / f"{TILE_ID}_V1-2_FABDEM.tif"

    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  [fabdem] {dest.name} already present")
        return

    if not offline:
        try:
            _download_fabdem_real(dest)
            return
        except Exception as exc:
            _warn(f"FABDEM download failed ({exc}) — using synthetic fallback")

    _generate_fabdem_synthetic(dest, np.random.default_rng(SEED))


def _download_fabdem_real(dest: Path) -> None:
    """Download real FABDEM tile N51E000 from Bristol data store."""
    print(f"  [fabdem] Downloading {TILE_ID} from Bristol data store …")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{TILE_ID}_V1-2_FABDEM.tif.zip"
        _http_get(FABDEM_ZIP_URL, zip_path)
        print("  [fabdem] Extracting …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(Path(tmp))
        tifs = list(Path(tmp).rglob("*.tif"))
        if not tifs:
            raise RuntimeError("No .tif found in downloaded zip")
        shutil.copy2(tifs[0], dest)

    print(f"  [fabdem] ✓ {dest.name}  ({_sz(dest)} — REAL 1 arc-sec, CC BY 4.0)")


def _generate_fabdem_synthetic(dest: Path, rng: np.random.Generator) -> None:
    """Fallback: realistic synthetic FABDEM (360 px = 10 arc-sec resolution)."""
    from rasterio.transform import from_bounds
    from scipy.ndimage import gaussian_filter

    h = w = SYNTH_PX_DEG
    terrain = rng.uniform(0, 350, (h, w)).astype(np.float32)
    terrain = gaussian_filter(terrain, sigma=8).astype(np.float32)

    lats = np.linspace(LAT_MAX, LAT_MIN, h)
    ocean_rows = lats < 51.5
    terrain[ocean_rows, :] = np.nan

    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, w, h)
    _write_tif(dest, terrain, transform, nodata=-9999.0)
    print(f"  [fabdem] ✓ {dest.name}  ({_sz(dest)} — SYNTHETIC fallback, {SYNTH_PX_DEG} px/°)")


# ===========================================================================
# GEBCO
# ===========================================================================


def _get_gebco(base: Path, offline: bool) -> None:
    geb_dir = base / "gebco"
    geb_dir.mkdir(parents=True, exist_ok=True)
    dest = geb_dir / "GEBCO_2024_N51E000.tif"

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [gebco]  {dest.name} already present")
        return

    if not offline:
        try:
            _download_gebco_wcs(dest)
            return
        except Exception as exc:
            _warn(f"GEBCO WCS failed ({exc}) — trying gdal_translate …")

    if not offline and shutil.which("gdal_translate"):
        try:
            _download_gebco_gdal(dest)
            return
        except Exception as exc:
            _warn(f"gdal_translate GEBCO failed ({exc}) — using synthetic fallback")

    _generate_gebco_synthetic(dest, np.random.default_rng(SEED))


def _download_gebco_wcs(dest: Path) -> None:
    """Fetch 1° GEBCO tile via WCS 2.0."""
    print("  [gebco]  Requesting 1° tile via GEBCO WCS …")
    _http_get(GEBCO_WCS_URL, dest, timeout=60)
    import rasterio
    with rasterio.open(dest) as ds:
        _ = ds.meta
    print(f"  [gebco]  ✓ {dest.name}  ({_sz(dest)} — REAL GEBCO 2024, CC0)")


def _download_gebco_gdal(dest: Path) -> None:
    """Fallback: gdal_translate clip from the global GEBCO via /vsicurl/."""
    print("  [gebco]  Clipping via gdal_translate /vsicurl/ …")
    src = (
        "/vsicurl/https://www.gebco.net/data_and_products/gridded_bathymetry_data/"
        "gebco_2024/GEBCO_2024_sub_ice_topo_geotiff.zip/GEBCO_2024.tif"
    )
    cmd = [
        "gdal_translate", "-of", "GTiff",
        "-projwin", "0", "52", "1", "51",      # ulx, uly, lrx, lry
        "-co", "COMPRESS=LZW", "-co", "TILED=YES",
        src, str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    print(f"  [gebco]  ✓ {dest.name}  ({_sz(dest)} — REAL GEBCO 2024, CC0)")


def _generate_gebco_synthetic(dest: Path, rng: np.random.Generator) -> None:
    """Fallback: realistic synthetic GEBCO matching the tile's coast at 51.5°N."""
    from rasterio.transform import from_bounds
    from scipy.ndimage import gaussian_filter

    h = w = 240   # 15 arc-sec = 240 px/° (matches real GEBCO resolution)
    lats = np.linspace(LAT_MAX, LAT_MIN, h)
    data = np.where(
        lats[:, None] < 51.5,
        rng.uniform(-120, -5, (h, w)).astype(np.float32),
        rng.uniform(0, 250, (h, w)).astype(np.float32),
    ).astype(np.float32)
    data = gaussian_filter(data, sigma=2).astype(np.float32)

    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, w, h)
    _write_tif(dest, data, transform, nodata=-32767.0)
    print(f"  [gebco]  ✓ {dest.name}  ({_sz(dest)} — SYNTHETIC fallback, 240 px/°)")


# ===========================================================================
# EGM2008
# ===========================================================================


def _get_egm2008(base: Path, offline: bool) -> None:
    egm_dir = base / "egm2008"
    egm_dir.mkdir(parents=True, exist_ok=True)
    dest = egm_dir / "egm2008_N51E000.tif"

    if dest.exists() and dest.stat().st_size > 5_000:
        print(f"  [egm2008] {dest.name} already present")
        return

    if not offline:
        try:
            _download_egm2008(dest)
            return
        except Exception as exc:
            _warn(f"EGM2008 download failed ({exc}) — using synthetic fallback")

    _generate_egm2008_synthetic(dest, np.random.default_rng(SEED))


def _download_egm2008(dest: Path) -> None:
    """Download global EGM2008 from OpenTopography then clip to tile+buffer."""
    print("  [egm2008] Downloading global egm2008-25.tif from OpenTopography (~5 MB) …")
    with tempfile.TemporaryDirectory() as tmp:
        global_tif = Path(tmp) / "egm2008_global.tif"
        _http_get(EGM2008_GLOBAL_URL, global_tif, timeout=120)

        bb = [LON_MIN - BUFFER, LAT_MIN - BUFFER, LON_MAX + BUFFER, LAT_MAX + BUFFER]

        if shutil.which("gdal_translate"):
            cmd = [
                "gdal_translate", "-of", "GTiff",
                "-projwin", str(bb[0]), str(bb[3]), str(bb[2]), str(bb[1]),
                "-co", "COMPRESS=LZW",
                str(global_tif), str(dest),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:300])
        else:
            import rasterio
            from rasterio.windows import from_bounds as win_from_bounds

            with rasterio.open(global_tif) as src:
                win = win_from_bounds(bb[0], bb[1], bb[2], bb[3], src.transform)
                data = src.read(1, window=win)
                t = src.window_transform(win)
            _write_tif(dest, data.astype(np.float32), t, nodata=-9999.0)

    print(f"  [egm2008] ✓ {dest.name}  ({_sz(dest)} — REAL EGM2008, Public Domain)")


def _generate_egm2008_synthetic(dest: Path, rng: np.random.Generator) -> None:
    """Fallback: realistic synthetic EGM2008 (NW Europe: ~47–52 m undulation)."""
    from rasterio.transform import from_bounds

    h, w = 6, 6
    transform = from_bounds(
        LON_MIN - BUFFER, LAT_MIN - BUFFER,
        LON_MAX + BUFFER, LAT_MAX + BUFFER,
        w, h,
    )
    undulation = rng.uniform(47.0, 52.0, (h, w)).astype(np.float32)
    _write_tif(dest, undulation, transform, nodata=-9999.0)
    print(f"  [egm2008] ✓ {dest.name}  ({_sz(dest)} — SYNTHETIC fallback, 47–52 m)")


# ===========================================================================
# ICE-6G  (always synthetic)
# ===========================================================================


def _generate_ice6g(base: Path, rng: np.random.Generator) -> None:
    """
    Realistic synthetic ICE-6G–like NetCDF calibrated to NW Europe.

    Epochs  : 0 ka (modern), 12 ka (early Holocene), 21 ka (LGM)
    stgr    : bedrock GIA uplift — 0 m at 51°N, ~300 m at 60°N+
    sice    : Fennoscandian ice sheet north of 56°N at 21 ka
    sealevel: RSL −120 m at 21 ka, −60 m at 12 ka

    Cite the real model:
      Peltier et al. (2015) doi:10.1002/2014JB011176
    """
    import netCDF4 as nc

    ice_dir = base / "ice6g"
    ice_dir.mkdir(parents=True, exist_ok=True)
    dest = ice_dir / "ICE6G_sample.nc"

    if dest.exists() and dest.stat().st_size > 2_000:
        print(f"  [ice6g]  {dest.name} already present")
        return

    lat_vals = np.arange(LAT_MIN - BUFFER - 1, LAT_MAX + BUFFER + 11, 0.5, dtype=np.float32)
    lon_vals = np.arange(LON_MIN - BUFFER - 1, LON_MAX + BUFFER + 2, 0.5, dtype=np.float32)
    time_vals = np.array([0.0, -12.0, -21.0], dtype=np.float32)

    nlat, nlon, ntime = len(lat_vals), len(lon_vals), len(time_vals)
    lat_grid = lat_vals[:, None] * np.ones((1, nlon))

    # bedrock deformation: scales with latitude (proxy for distance to Fennoscandia)
    uplift_factor = np.clip((lat_grid - 51.0) / 10.0, 0.0, 1.0)
    stgr = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    stgr[1] = (rng.uniform(100, 150, (nlat, nlon)) * uplift_factor).astype(np.float32)
    stgr[2] = (rng.uniform(250, 320, (nlat, nlon)) * uplift_factor).astype(np.float32)

    # ice sheet north of 56°N
    sice = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    ice_mask = lat_grid > 56.0
    sice[1][ice_mask] = rng.uniform(500, 2000, ice_mask.sum()).astype(np.float32)
    sice[2][ice_mask] = rng.uniform(1500, 3000, ice_mask.sum()).astype(np.float32)

    # relative sea level
    rsl = np.zeros((ntime, nlat, nlon), dtype=np.float32)
    rsl[1, :, :] = rng.uniform(-65, -55, (nlat, nlon)).astype(np.float32)
    rsl[2, :, :] = rng.uniform(-125, -115, (nlat, nlon)).astype(np.float32)

    with nc.Dataset(str(dest), "w", format="NETCDF4") as ds:
        ds.title = (
            "Synthetic ICE-6G_C VM5a-like data for paleoeurope tests. "
            "NW Europe: 51-52N, 0-1E. Epochs: 0/12/21 ka. "
            "Cite real model: Peltier et al. 2015 doi:10.1002/2014JB011176"
        )
        ds.Conventions = "CF-1.8"
        ds.note = "SYNTHETIC — calibrated but not the real ICE-6G_C dataset"

        ds.createDimension("time", ntime)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)

        t_v = ds.createVariable("time", "f4", ("time",))
        t_v[:] = time_vals
        t_v.units = "ka BP"
        t_v.long_name = "model time (0=modern, negative=past)"

        lat_v = ds.createVariable("lat", "f4", ("lat",))
        lat_v[:] = lat_vals
        lat_v.units = "degrees_north"

        lon_v = ds.createVariable("lon", "f4", ("lon",))
        lon_v[:] = lon_vals
        lon_v.units = "degrees_east"

        for vname, data, units, lname in [
            ("stgr",      stgr, "m", "Bedrock deformation (GIA delta)"),
            ("sice",      sice, "m", "Ice sheet thickness"),
            ("sealevel",  rsl,  "m", "Relative sea level change"),
        ]:
            v = ds.createVariable(vname, "f4", ("time", "lat", "lon"), fill_value=-9999.0)
            v[:] = data
            v.units = units
            v.long_name = lname

    print(
        f"  [ice6g]  ✓ {dest.name}  ({_sz(dest)} — SYNTHETIC, "
        "epochs: 0/12/21 ka, NW-Europe calibrated)"
    )


# ===========================================================================
# Tile list
# ===========================================================================


def _write_tile_list(base: Path) -> None:
    tile_list = base / "tile_list.txt"
    tile_list.write_text(f"{TILE_ID}\n")
    print(f"  [tiles]  tile_list.txt written  (1 real tile: {TILE_ID})")


# ===========================================================================
# Utilities
# ===========================================================================


def _http_get(url: str, dest: Path, timeout: int = 90) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "paleoeurope/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(dest, "wb") as f:
                while True:
                    buf = resp.read(65536)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total:
                        print(
                            f"\r    {100*done//total:3d}%  {_human(done)} / {_human(total)}   ",
                            end="", flush=True,
                        )
            print()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc


def _write_tif(path: Path, data: np.ndarray, transform: object, nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.crs import CRS

    arr = np.where(np.isnan(data), nodata, data).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32",
        crs=CRS.from_epsg(4326), transform=transform,
        nodata=nodata, compress="LZW",
    ) as dst:
        dst.write(arr, 1)


def _sz(p: Path) -> str:
    return _human(p.stat().st_size)


def _human(n: int) -> str:
    for u in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n //= 1024
    return f"{n:.1f} GB"


def _warn(msg: str) -> None:
    print(f"  WARNING: {msg}", file=sys.stderr)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
