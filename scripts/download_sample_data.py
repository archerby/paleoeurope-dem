#!/usr/bin/env python3
"""
scripts/download_sample_data.py

Download a minimal real-data sample for integration tests and demo notebooks.

What is downloaded
------------------
1. FABDEM v1.2 — tile N51E000 (England/Channel coast, 51-52°N 0-1°E)
   Source : CEDA Archive via Bristol University
   License: CC BY 4.0  (Hawker et al. 2022, doi:10.5194/essd-14-4677-2022)
   Size   : ~5 MB (zipped), ~10 MB (GeoTIFF)

2. GEBCO 2024 — bbox 0°W–3°E, 51°N–54°N  (clipped with GDAL)
   Source : https://download.gebco.net/  (tile service)
   License: CC0 Public Domain
   Size   : ~2 MB (Int16 GeoTIFF at 15 arc-sec)

3. EGM2008 undulation — bbox -1°–4°E, 50°–55°N (from NGA WCS or static file)
   Source : NGA / OpenTopography mirror
   License: Public Domain (US Government work)
   Size   : ~500 KB (Float32, 2.5 arc-min grid)

4. ICE-6G_C VM5a — Europe subset, epochs 0 ka + 12 ka + 21 ka
   Source : U Toronto Peltier group (direct URL from their data page)
   License: Academic use — cite Peltier et al. 2015. No redistribution clause,
            but extract/subset for testing is standard academic practice.
   Size   : ~3 MB (NetCDF, 0.5° global subset bbox 45-60°N -5-20°E, 3 epochs)

5. Paleocoastlines GIS dataset — land masks for sea levels -130 m to +5 m
   Source : CRC806-Database  https://doi.org/10.5880/SFB806.19
            Direct download: http://crc806db.uni-koeln.de/layer/show/327/
   License: CC BY 4.0 — Zickel et al. (2016)
   Size   : ~46 MB (zip archive containing Shapefile)
   Note   : Contains 23 sea-level steps (130 m below to 5 m above MSL).
            If the DOI resolves to a landing page requiring manual download,
            the script will print instructions.

6. Spratt & Lisiecki (2016) sea-level curve — simplified 0–26 ka CSV
   Source : Bundled in data/sea_level_curves/ (4 KB — included in repo)
            Full dataset (0–800 ka): https://doi.org/10.1594/PANGAEA.856145
   License: CC BY 3.0  — Spratt & Lisiecki (2016)
   Note   : The bundled CSV is a simplified extract; download the full
            PANGAEA dataset for production accuracy beyond 26 ka.

Output layout
-------------
tests/data/real_sample/
├── .downloaded                              ← sentinel
├── fabdem/
│   └── N51E000_V1-2_FABDEM.tif
├── gebco/
│   └── GEBCO_2024_sample.tif               ← 0-3°E 51-54°N
├── egm2008/
│   └── egm2008_sample.tif                  ← -1-4°E 50-55°N
├── ice6g/
│   └── ICE6G_C_VM5a_sample.nc              ← subset 3 epochs
├── paleocostlines/
│   └── Paleocoastlines/                    ← extracted zip
│       └── Paleocoastlines/
│           ├── Paleocoastlines.shp
│           ├── Paleocoastlines.dbf
│           └── …
└── checksums.sha256

Usage
-----
    python scripts/download_sample_data.py
    python scripts/download_sample_data.py --dest tests/data/real_sample
    python scripts/download_sample_data.py --force        # re-download even if present
    python scripts/download_sample_data.py --skip-verify  # skip SHA256 check
    python scripts/download_sample_data.py --only gebco fabdem
   python scripts/download_sample_data.py --only paleocoastlines spratt_csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# sha256 is filled after first download (see _record_checksums).
# Leave as None to skip verification for datasets fetched dynamically.

DATASETS: dict[str, dict] = {
    "fabdem": {
        "desc": "FABDEM v1.2 tile N51E000  (CC BY 4.0)",
        "cite": "Hawker et al. (2022) doi:10.5194/essd-14-4677-2022",
        # Direct tile download from the CEDA/Bristol data store
        # The zip contains N51E000_V1-2_FABDEM.tif
        "url": (
            "https://data.bris.ac.uk/datasets/25wfy0f9ukibjp75gb8laq1ghq/"
            "N51E000_V1-2_FABDEM.tif.zip"
        ),
        "dest_subdir": "fabdem",
        "filename": "N51E000_V1-2_FABDEM.tif.zip",
        "unzip": True,
        "final_file": "N51E000_V1-2_FABDEM.tif",
        "sha256": None,  # filled after first download
    },
    "gebco": {
        "desc": "GEBCO 2024 clipped 0-3°E 51-54°N  (CC0)",
        "cite": "GEBCO Compilation Group (2024) doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f",
        # GEBCO tile-service WCS URL (bbox, resolution, format)
        # This fetches a pre-clipped GeoTIFF directly — no login required.
        "url": (
            "https://www.gebco.net/data_and_products/gridded_bathymetry_data/"
            "gebco_2024/gebco_2024_tid_raster.zip"
        ),
        # Alternative: GEBCO tile service (subset). We use gdal_translate for clipping.
        "use_gdal_clip": True,
        "gdal_source": (
            "https://opendap.earthdata.nasa.gov/providers/ASTER/collections/"
            "GEBCO_LATEST_TID/granules/GEBCO_LATEST_TID.nc"
        ),
        "gdal_clip_bounds": [0.0, 51.0, 3.0, 54.0],  # xmin ymin xmax ymax
        "dest_subdir": "gebco",
        "final_file": "GEBCO_2024_sample.tif",
        "sha256": None,
    },
    "egm2008": {
        "desc": "EGM2008 undulation -1-4°E 50-55°N  (Public Domain)",
        "cite": "Pavlis et al. (2012) doi:10.1029/2011JB008916",
        # OpenTopography hosts a GeoTIFF version of the EGM2008 25-arcmin grid
        "url": (
            "https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/"
            "OTDS.012013.4326.1/raster/egm2008-25.tif"
        ),
        "use_gdal_clip": True,
        "gdal_clip_bounds": [-1.0, 50.0, 4.0, 55.0],
        "dest_subdir": "egm2008",
        "final_file": "egm2008_sample.tif",
        "sha256": None,
    },
    "ice6g": {
        "desc": "ICE-6G_C VM5a global NetCDF — Europe subset 3 epochs  (cite Peltier et al. 2015)",
        "cite": "Peltier et al. (2015) doi:10.1002/2014JB011176",
        # The Peltier group provides downloads at:
        # https://www.physics.utoronto.ca/~peltier/data.html
        # Direct URL for ICE-6G_C VM5a O512 file:
        "url": (
            "https://www.physics.utoronto.ca/~peltier/data_ICE6GC/"
            "ICE-6G_C_VM5a_O512.nc"
        ),
        "use_nc_subset": True,
        "nc_bbox": [-5.0, 45.0, 20.0, 60.0],       # lon_min lat_min lon_max lat_max
        "nc_epochs_ka": [0.0, 12.0, 21.0],           # keep only these epochs
        "dest_subdir": "ice6g",
        "final_file": "ICE6G_C_VM5a_sample.nc",
        "sha256": None,
    },
    "paleocoastlines": {
        "desc": "Paleocoastlines GIS dataset — 23 sea-level land masks (CC BY 4.0)",
        "cite": (
            "Zickel, M., Becker, D., Verheul, J., Yener, Y. & Willmes, C. (2016): "
            "Paleocoastlines GIS dataset. CRC806-Database, doi:10.5880/SFB806.19"
        ),
        # Primary download from CRC806 database:
        #   http://crc806db.uni-koeln.de/layer/show/327/
        # The download button on that page provides the zip ("Download as zip").
        # If the URL below fails (DOI landing page redirect), download manually
        # and place the zip at <dest>/paleocostlines/Paleocoastlines.zip
        "url": "http://crc806db.uni-koeln.de/layer/show/327/download/?format=shp",
        "dest_subdir": "paleocostlines",
        "filename": "Paleocoastlines.zip",
        "unzip": True,
        "final_file": "Paleocoastlines/Paleocoastlines/Paleocoastlines.shp",
        # SHA256 of the Paleocoastlines.zip sourced from production data:
        "sha256": "c6cba2a7dda0af3cfb62980b6914842cbe3e25358c5a77528aa7c8cd541a0735",
        "manual_fallback": (
            "Could not download Paleocoastlines automatically.\n"
            "  1. Visit https://doi.org/10.5880/SFB806.19\n"
            "  2. Download the shapefile archive\n"
            "  3. Place the zip as:  <dest>/paleocostlines/Paleocoastlines.zip"
        ),
    },
    "spratt_csv": {
        "desc": "Spratt & Lisiecki (2016) sea-level curve 0–800 ka  (CC BY 3.0)",
        "cite": (
            "Spratt, R. M. & Lisiecki, L. E. (2016). A Late Pleistocene sea-level "
            "stack. Climate of the Past, 12, 1079–1092. "
            "doi:10.5194/cp-12-1079-2016"
        ),
        # Full tabular data on PANGAEA:
        "url": "https://doi.pangaea.de/10.1594/PANGAEA.856145?format=textfile",
        "dest_subdir": "sea_level_curves",
        "final_file": "spratt_lisiecki_2016_full.tab",
        "sha256": None,
        "note": (
            "The repo bundles a simplified 0-26 ka extract at "
            "data/sea_level_curves/spratt_lisiecki_2016_simplified.csv. "
            "Download the full PANGAEA file for epochs beyond 26 ka."
        ),
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest",
        default=str(REPO_ROOT / "tests" / "data" / "real_sample"),
        help="Output directory (default: tests/data/real_sample)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--skip-verify", action="store_true", help="Skip SHA256 verification")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DATASETS),
        metavar="DATASET",
        help="Download only these datasets (e.g. --only fabdem gebco)",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    sentinel = dest / ".downloaded"

    if sentinel.exists() and not args.force:
        logger.info("Sample data already present at %s  (use --force to re-download)", dest)
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    targets = args.only or list(DATASETS)
    failed: list[str] = []

    for name in targets:
        spec = DATASETS[name]
        logger.info("─── %s ───", spec["desc"])
        try:
            _fetch(name, spec, dest, skip_verify=args.skip_verify)
        except Exception as exc:
            logger.error("FAILED %s: %s", name, exc)
            failed.append(name)

    # Write checksums file
    _record_checksums(dest, targets)

    if failed:
        logger.error("Failed datasets: %s", ", ".join(failed))
        logger.error("Re-run with --only %s to retry.", " ".join(failed))
        return 1

    sentinel.touch()
    logger.info("✓ Download complete → %s", dest)
    return 0


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _fetch(name: str, spec: dict, dest: Path, skip_verify: bool) -> None:
    """Dispatch to the right fetch strategy for this dataset."""
    out_dir = dest / spec["dest_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / spec["final_file"]

    if final.exists() and not _is_empty(final):
        logger.info("  Already present: %s", final.name)
        if not skip_verify and spec["sha256"]:
            # For shapefiles, verify the zip, not the individual .shp
            zip_path = out_dir / spec.get("filename", final.name)
            if zip_path.exists() and spec.get("sha256"):
                _verify_sha256(zip_path, spec["sha256"])
        return

    try:
        if spec.get("use_gdal_clip"):
            _fetch_gdal_clip(spec, final)
        elif spec.get("use_nc_subset"):
            _fetch_nc_subset(spec, final)
        else:
            _fetch_url(spec, out_dir, final)
    except Exception as exc:
        fallback = spec.get("manual_fallback")
        if fallback:
            logger.error(
                "  Download failed: %s\n  Manual download instructions:\n  %s",
                exc, fallback
            )
            return
        raise

    if not final.exists():
        logger.warning("  Expected output not found: %s", final)
        return

    if not skip_verify and spec["sha256"]:
        # For zipped shapefiles, verify the zip archive
        zip_path = out_dir / spec.get("filename", final.name)
        verify_target = zip_path if zip_path.exists() else final
        _verify_sha256(verify_target, spec["sha256"])

    size_str = _human_size(final) if final.is_file() else f"dir {final.parent.name}/"
    logger.info("  ✓ %s  (%s)", final.name, size_str)


def _fetch_url(spec: dict, out_dir: Path, final: Path) -> None:
    """Download a URL, optionally unzip, and move to final path."""
    url = spec["url"]
    filename = spec.get("filename", final.name)
    raw_path = out_dir / filename

    logger.info("  Downloading %s", url)
    _download(url, raw_path)

    if spec.get("unzip"):
        import zipfile

        logger.info("  Extracting %s", raw_path.name)
        with zipfile.ZipFile(raw_path) as zf:
            zf.extractall(out_dir)
        # Keep the zip (it may be needed for SHA256 verification)

        if not final.exists():
            # For flat zip → single raster files, find by extension
            extracted = [
                p for p in out_dir.rglob("*")
                if p.suffix in (".tif", ".nc", ".shp")
                and not p.name.startswith(".")
            ]
            if not extracted:
                raise RuntimeError(
                    f"Expected {final.name!r} not found after unzip in {out_dir}"
                )
            # If there's exactly one match and it's not already at final, rename
            if len(extracted) == 1 and extracted[0].resolve() != final.resolve():
                extracted[0].rename(final)


def _fetch_gdal_clip(spec: dict, final: Path) -> None:
    """Use gdal_translate or gdalwarp to clip a remote or local source."""
    _require_gdal()
    source = spec.get("gdal_source") or spec["url"]
    bb = spec["gdal_clip_bounds"]  # [xmin, ymin, xmax, ymax]

    logger.info("  GDAL-clipping  %s  →  bbox %s", source, bb)

    # Try gdal_translate (works for NetCDF/TIF with subdataset)
    cmd = [
        "gdal_translate",
        "-projwin", str(bb[0]), str(bb[3]), str(bb[2]), str(bb[1]),  # ulx uly lrx lry
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        source,
        str(final),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("  gdal_translate failed, trying gdalwarp:\n%s", result.stderr)
        # Fallback to gdalwarp
        cmd2 = [
            "gdalwarp",
            "-te", str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3]),
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            source,
            str(final),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(f"gdalwarp failed:\n{result2.stderr}")


def _fetch_nc_subset(spec: dict, final: Path) -> None:
    """Download a NetCDF, then subset by bbox and selected epochs."""
    url = spec["url"]
    bb = spec["nc_bbox"]
    epochs_ka = spec["nc_epochs_ka"]

    logger.info("  Downloading full NetCDF from %s", url)
    logger.info("  (this may take a few minutes — ~50 MB file)")

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _download(url, tmp_path)
        logger.info("  Subsetting bbox=%s  epochs=%s ka", bb, epochs_ka)
        _nc_subset(tmp_path, final, bbox=bb, epochs_ka=epochs_ka)
    finally:
        tmp_path.unlink(missing_ok=True)


def _nc_subset(
    src: Path,
    dst: Path,
    bbox: list[float],
    epochs_ka: list[float],
) -> None:
    """Subset a NetCDF by bounding box and epoch list."""
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(src, engine="netcdf4")

    # Detect dimension names
    lat_dim = _dim(ds, ("lat", "latitude", "Lat"))
    lon_dim = _dim(ds, ("lon", "longitude", "Lon"))
    time_dim = _dim(ds, ("time", "Time", "t"))

    lat = ds[lat_dim].values
    lon = ds[lon_dim].values
    time_vals = ds[time_dim].values

    lon_min, lat_min, lon_max, lat_max = bbox
    buf = 1.0
    lat_sel = (lat >= lat_min - buf) & (lat <= lat_max + buf)
    lon_sel = (lon >= lon_min - buf) & (lon <= lon_max + buf)

    # Match epochs (time axis may be negative or positive ka)
    time_indices = []
    for eka in epochs_ka:
        target = -float(eka) if time_vals.min() < 0 else float(eka)
        time_indices.append(int(np.argmin(np.abs(time_vals - target))))

    ds_sub = ds.isel({lat_dim: lat_sel, lon_dim: lon_sel, time_dim: time_indices})
    ds_sub.to_netcdf(dst)
    ds.close()
    logger.info("  Subset written: %s  (%s)", dst.name, _human_size(dst))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress indicator."""
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536
            with open(dest, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = 100 * downloaded // total
                        print(f"\r  {pct:3d}%  {_human_bytes(downloaded)} / {_human_bytes(total)}   ", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} downloading {url}\n"
            f"Check the URL manually: {url}\n"
            f"If the file requires registration, download manually and place at: {dest}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error downloading {url}: {exc.reason}") from exc


def _verify_sha256(path: Path, expected: str) -> None:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    actual = sha.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path.name}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )
    logger.info("  ✓ SHA256 verified: %s", path.name)


def _record_checksums(dest: Path, names: list[str]) -> None:
    """Write checksums.sha256 for all downloaded final files."""
    lines: list[str] = []
    for name in names:
        spec = DATASETS[name]
        final = dest / spec["dest_subdir"] / spec["final_file"]
        if not final.exists():
            continue
        sha = hashlib.sha256()
        with open(final, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        digest = sha.hexdigest()
        lines.append(f"{digest}  {spec['dest_subdir']}/{spec['final_file']}")
        # Also update the in-memory spec so re-runs verify
        DATASETS[name]["sha256"] = digest

    checksum_file = dest / "checksums.sha256"
    checksum_file.write_text("\n".join(lines) + "\n")
    logger.info("Checksums written to %s", checksum_file)


def _require_gdal() -> None:
    if not shutil.which("gdal_translate"):
        raise EnvironmentError(
            "gdal_translate not found.  Install GDAL:\n"
            "  conda: conda install -c conda-forge gdal\n"
            "  apt:   sudo apt-get install gdal-bin"
        )


def _dim(ds: object, candidates: tuple[str, ...]) -> str:
    import xarray as xr

    assert isinstance(ds, xr.Dataset)
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError(f"Dimension not found; tried: {candidates}. Available: {list(ds.dims)}")


def _is_empty(path: Path) -> bool:
    return path.stat().st_size < 1024


def _human_size(path: Path) -> str:
    return _human_bytes(path.stat().st_size)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
