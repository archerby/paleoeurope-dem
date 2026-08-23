import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
import rasterio, numpy as np, whitebox, os, time
T=PROBE+"/tmp"
wbt=whitebox.WhiteboxTools(); wbt.work_dir=T; wbt.set_verbose_mode(False)

def rewrite(ep):
    src="/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_%ska.tif"%ep
    dst=PROBE+"/in/khv_%ska_nopred.tif"%ep
    if os.path.exists(dst): return src,dst
    with rasterio.open(src) as d:
        a=d.read(1); prof=d.profile; st=d.tags()
    write_tif(dst,a,prof,-9999.0,{
      "probe_product":"predictor_free_rewrite_of_source_dem",
      "probe_method":"rasterio copy of band 1, no resampling, PREDICTOR omitted",
      "probe_input_path":src,"probe_input_sha256":sha256_file(src),
      "probe_input_pixel_sha256_as_read":st.get("pixel_sha256"),
      "probe_epoch_ka":st.get("epoch_ka"),
      "probe_rows":a.shape[0],"probe_cols":a.shape[1]})
    return src,dst

def rowcol(tr, lon, lat):
    inv=~tr; c,r = inv*(lon,lat); return int(r), int(c)

for ep, slon, slat in [("16.5",44.118750,45.710417), ("13",44.118750,45.743750)]:
    print("="*78); print("EPOCH %s ka  sill lon %.6f lat %.6f"%(ep,slon,slat))
    src,dem_p = rewrite(ep)
    with rasterio.open(dem_p) as d: dem=d.read(1); tr=d.transform; prof=d.profile
    r,c = rowcol(tr, slon, slat)
    clon,clat = tr*(c+0.5,r+0.5)
    print(" sill cell row %d col %d  cell centre lon %.6f lat %.6f"%(r,c,clon,clat))
    print(" NATIVE dem at sill cell : %.6f m"%dem[r,c])
    with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%ep) as w:
        wt=w.tags()
    m=dict(kv.split('=',1) for kv in wt['sill_caspian__pontic'].split(';') if '=' in kv)
    print(" shipped measured_saddle_m: %s   spilling=%s"%(m['measured_saddle_m'], wt['spilling']))
    out=T+"/breach_%s.tif"%ep
    t0=time.time(); L=[]
    wbt.breach_depressions(dem_p, out, callback=lambda s:L.append(s))
    print(" BreachDepressions(defaults) %.1fs  cmdline: %s"%(time.time()-t0, L[0].strip() if L else "?"))
    bad=[l for l in L if 'panic' in l.lower() or 'error' in l.lower()]
    print(" err lines:", bad or "none", " artifact:", os.path.exists(out), os.path.getsize(out) if os.path.exists(out) else 0)
    with rasterio.open(out) as d: br=d.read(1)
    print(" BREACHED dem at sill cell: %.6f m   (drop %.6f m)"%(br[r,c], br[r,c]-dem[r,c]))
    dz = br.astype(np.float64)-dem.astype(np.float64)
    print(" cells lowered (dz<0): %d (%.4f%%)   cells raised (dz>0): %d (%.4f%%)"%(
      int((dz<0).sum()),100.0*(dz<0).sum()/dz.size,int((dz>0).sum()),100.0*(dz>0).sum()/dz.size))
    print(" min dz %.6f  max dz %.6f  sum|dz| %.3f"%(dz.min(),dz.max(),np.abs(dz).sum()))
    np.save(T+"/dz_breach_%s.npy"%ep, dz.astype(np.float32))
    del br, dz, dem
