"""Shared helpers for the rivers probe. No numeric literal is ever written into
a tag or caption: every value in a tag is computed from the artifact at write time."""
import hashlib, datetime, os, subprocess, sys
import numpy as np, rasterio

PROBE = "/mnt/tank/paleo/renders/_rivers_probe_2026-08-18"

def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(buf), b""):
            h.update(c)
    return h.hexdigest()

def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def toolchain():
    import rasterio as r
    return "python=%s;numpy=%s;rasterio=%s;gdal=%s" % (
        sys.version.split()[0], np.__version__, r.__version__, r.__gdal_version__)

def write_tif(path, arr, profile_src, nodata, tags, dtype=None, predictor=None):
    """Write a GeoTIFF. predictor=None => predictor-free (WBT-readable)."""
    prof = dict(profile_src)
    dt = dtype or arr.dtype.name
    prof.update(driver="GTiff", dtype=dt, count=1, nodata=nodata,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    prof.pop("predictor", None)
    if predictor is not None:
        prof["predictor"] = predictor
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr.astype(dt), 1)
        base = dict(tags)
        base["probe_written_utc"] = utcnow()
        base["probe_toolchain"] = toolchain()
        base["probe_pixel_sha256"] = sha256_arr(arr.astype(dt))
        base["probe_pixel_sha256_of"] = ("band 1 as %s, C order, row-major, %d rows x %d cols, "
            "exactly the bytes rasterio read(1) returns" % (dt, arr.shape[0], arr.shape[1]))
        ds.update_tags(**{k: str(v) for k, v in base.items()})
    return path

def read_tags(p):
    with rasterio.open(p) as d:
        return d.tags()
