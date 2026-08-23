import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
import rasterio, whitebox, numpy as np, os
PROBE="/mnt/tank/paleo/renders/_rivers_probe_2026-08-18"
print("--- does the TOPO raster carry a sill_caspian__pontic tag, as the brief states? ---")
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_16.5ka.tif") as d: tt=d.tags()
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_16.5ka.tif") as d: wt=d.tags()
print("  on topo        :", 'sill_caspian__pontic' in tt, "| sill-ish topo tags:", sorted(k for k in tt if 'sill' in k))
print("  on water_state :", 'sill_caspian__pontic' in wt)
print("\n--- shipped product reads back, tags intact, WBT-readable ---")
out=PROBE+"/out/drainage_implied_16.5ka.tif"
with rasterio.open(out) as d:
    a=d.read(1); t=d.tags()
    print("  ",out, d.dtypes, d.shape, "PREDICTOR:", d.tags(ns='IMAGE_STRUCTURE').get('PREDICTOR','<absent>'))
    for k in sorted(t):
        if k.startswith('probe_'): print("    %-42s = %s"%(k,str(t[k])[:150]))
    print("   channel cells re-counted from the file:", int((a==1).sum()))
wbt=whitebox.WhiteboxTools(); wbt.work_dir=PROBE+"/tmp"; wbt.set_verbose_mode(False)
L=[]; wbt.raster_summary_stats(out, callback=lambda s:L.append(s))
print("  WBT on the product:", [l.strip() for l in L if 'cells' in l or 'panic' in l.lower()][:3])
print("\n--- inventory ---")
for root,dirs,files in os.walk(PROBE):
    for f in sorted(files):
        p=os.path.join(root,f)
        print("   %8.1f MB  %s"%(os.path.getsize(p)/1e6, p.replace(PROBE+"/","")))
