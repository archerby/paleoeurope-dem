import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
import rasterio, numpy as np, whitebox, os

SRC="/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_16.5ka.tif"
DST=PROBE+"/in/khv_16.5ka_nopred.tif"
with rasterio.open(SRC) as d:
    a=d.read(1); prof=d.profile; st=d.tags()
    src_pred=d.tags(ns='IMAGE_STRUCTURE').get('PREDICTOR','<absent>')
print("source PREDICTOR tag:", src_pred, "shape", a.shape, "dtype", a.dtype)
src_sha = sha256_file(SRC)
write_tif(DST, a, prof, d_nodata if False else -9999.0, {
  "probe_product":"predictor_free_rewrite_of_source_dem",
  "probe_method":"rasterio copy of band 1, no resampling, no reprojection, PREDICTOR omitted",
  "probe_why":"WhiteboxTools 2.4.0 GeoTIFF reader panics on floating-point predictors; this rewrite is a pipeline step, not a fix to the source",
  "probe_input_path":SRC,
  "probe_input_sha256":src_sha,
  "probe_input_PREDICTOR":src_pred,
  "probe_input_pixel_sha256_as_read":st.get("pixel_sha256","<absent>"),
  "probe_epoch_ka":st.get("epoch_ka","<absent>"),
  "probe_rows":a.shape[0], "probe_cols":a.shape[1],
  "probe_nodata_cells_measured": int((a==-9999.0).sum()),
  "probe_min_m": float(np.nanmin(a)), "probe_max_m": float(np.nanmax(a)),
})
# prove the rewrite is bit-identical in pixel content
with rasterio.open(DST) as d2:
    b=d2.read(1)
    print("PREDICTOR after rewrite:", d2.tags(ns='IMAGE_STRUCTURE').get('PREDICTOR','<absent>'))
    print("pixel identical to source:", bool(np.array_equal(a,b)))
    print("source pixel_sha256 tag  :", st.get("pixel_sha256"))
    print("rewrite pixel sha256 recomputed:", sha256_arr(b))
print("size src %.1f MB  dst %.1f MB" % (os.path.getsize(SRC)/1e6, os.path.getsize(DST)/1e6))

wbt=whitebox.WhiteboxTools(); wbt.set_verbose_mode(True); wbt.work_dir=PROBE+"/tmp"
L=[]; wbt.raster_summary_stats(DST, callback=lambda s:L.append(s))
print("\n---- WBT reading the rewritten file ----")
for l in L:
    if l.startswith("Progress"): continue
    print(l)
