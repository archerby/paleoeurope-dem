import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from guards import *
import rasterio, numpy as np, whitebox, os, time
from rasterio.windows import Window
T=PROBE+"/tmp"; EP="16.5"
TOPO="/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_%ska.tif"%EP
WS="/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP
BI="/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_%ska.tif"%EP

with rasterio.open(WS) as w:
    ws=w.read(1); wprof=w.profile; wtr=w.transform; wtags=w.tags()
cw = dict(kv.split('=',1) for kv in wtags['crop_window'].split(';'))
r_off=int(cw['row_off']); c_off=int(cw['col_off']); W=int(cw['width']); H=int(cw['height'])
print("crop_window from the shipped field's own tag:", cw)
with rasterio.open(TOPO) as d:
    dem = d.read(1, window=Window(c_off, r_off, W, H))
    prof = d.profile.copy(); ttags=d.tags()
    prof.update(width=W, height=H, transform=d.window_transform(Window(c_off,r_off,W,H)))
print("dem window", dem.shape, "transform matches field:", tuple(prof['transform'])[:6]==tuple(wtr)[:6])

water = (ws & 1) > 0; unobs = (ws & 2) > 0; corr = (ws & 4) > 0
print("water %d  unobservable %d  corridor-mark %d  overlap(water&unobs) %d  overlap(water&corr) %d"%(
    water.sum(), unobs.sum(), corr.sum(), (water&unobs).sum(), (water&corr).sum()))
ND = -9999.0
masked = dem.astype(np.float32).copy()
masked[water|unobs] = ND
print("masked-to-nodata cells: %d (%.4f%% of window)"%((masked==ND).sum(), 100.0*(masked==ND).sum()/masked.size))
print("land cells remaining  : %d"%int((masked!=ND).sum()))

# guards, on the real artifact
g1 = guard_mask_partition(masked, ND, ws, "mask_partition")
g2 = guard_third_state_not_collapsed(masked, ND, ws, "third_state")
print("GUARD mask_partition PASS (nodata=%d)   GUARD third_state PASS (leaked=%d)"%(g1,g2))

MDEM=PROBE+"/in/dem_land_only_%ska.tif"%EP
write_tif(MDEM, masked, prof, ND, {
 "probe_product":"dem_land_only",
 "probe_method":"epoch's own water_state bits water|unobservable set to nodata; land elevations untouched; no resampling",
 "probe_rule":"WhiteboxTools treats nodata as the edge of the DEM, so the epoch's shoreline becomes a flow outlet without inventing one",
 "probe_third_state_policy":"UNOBSERVABLE is masked out, NOT drained and NOT called land; refusal over substitution of state",
 "probe_input_dem":TOPO, "probe_input_dem_sha256":sha256_file(TOPO),
 "probe_input_water_state":WS, "probe_input_water_state_sha256":sha256_file(WS),
 "probe_water_state_bits_as_read":wtags['water_state_bits'],
 "probe_crop_window_as_read":wtags['crop_window'],
 "probe_epoch_ka":ttags.get('epoch_ka'),
 "probe_cells_water":int(water.sum()), "probe_cells_unobservable":int(unobs.sum()),
 "probe_cells_corridor_mark":int(corr.sum()),
 "probe_cells_masked_nodata":int((masked==ND).sum()),
 "probe_cells_land":int((masked!=ND).sum()),
 "probe_land_min_m":float(masked[masked!=ND].min()), "probe_land_max_m":float(masked[masked!=ND].max()),
 "probe_nodata_value":ND,
})
print("wrote", MDEM, os.path.getsize(MDEM))

wbt=whitebox.WhiteboxTools(); wbt.work_dir=T; wbt.set_verbose_mode(False)
def run(n,f,*a,**k):
    t0=time.time(); L=[]; f(*a,callback=lambda s:L.append(s),**k)
    bad=[l for l in L if 'panic' in l.lower() or 'error' in l.lower()]
    print("[%s] %.1fs err:%s"%(n,time.time()-t0,bad or "none")); return L
FILL=T+"/land_fill_%s.tif"%EP; PNTR=T+"/land_pntr_%s.tif"%EP; ACC=T+"/land_acc_%s.tif"%EP
run("FillDepressions(land only)", wbt.fill_depressions, MDEM, FILL)
run("D8Pointer", wbt.d8_pointer, FILL, PNTR)
run("D8FlowAccumulation", wbt.d8_flow_accumulation, FILL, ACC, out_type="cells")
for f in (FILL,PNTR,ACC): print(" artifact", os.path.basename(f), os.path.exists(f), os.path.getsize(f))
with rasterio.open(FILL) as d: fl=d.read(1)
land = masked!=ND
dz = np.where(land, fl.astype(np.float64)-masked.astype(np.float64), 0.0)
print("=== land-only fill delta (over LAND cells only) ===")
print(" land cells %d ; moved %d (%.4f%% of land, %.4f%% of window)"%(
  land.sum(), (dz>0).sum(), 100.0*(dz>0).sum()/land.sum(), 100.0*(dz>0).sum()/land.size))
print(" max dz %.6f  sum dz %.3f  mean over moved %.6f"%(dz.max(), dz.sum(), dz[dz>0].mean()))
