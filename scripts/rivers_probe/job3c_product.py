import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from guards import *
from grid import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd, json, os
T=PROBE+"/tmp"; EP="16.5"; ND=-9999.0

# --- threshold chosen FROM THE SWEEP, not from arithmetic ---
rows=json.load(open(T+"/sweep_%s.json"%EP))
Ts=np.array([r[0] for r in rows]); dens=np.array([r[3] for r in rows]); man=np.array([r[5] for r in rows])
lt=np.log10(Ts); ld=np.log10(np.maximum(dens,1e-12)); sl=np.gradient(ld,lt)
PLATEAU=np.median(sl[:5])                      # the exponent the sweep itself exhibits at its fine end
ok=np.abs(sl-PLATEAU)<=0.01
T_CHOSEN=float(Ts[ok][-1])                     # largest T still on the plateau the data shows
T_MANYCH_DEAD=float(Ts[man==0].min())          # first T at which the Manych corridor carries no channel
print("sweep plateau exponent (median of the 5 finest T): %+.4f"%PLATEAU)
print("largest T with |slope-plateau|<=0.01  -> T_CHOSEN = %.0f cells"%T_CHOSEN)
print("first T at which manych_km == 0        -> %.0f cells (hard upper bound)"%T_MANYCH_DEAD)
print("chosen T is below that bound by a factor of %.1f"%(T_MANYCH_DEAD/T_CHOSEN))

with rasterio.open(PROBE+"/in/dem_land_only_%ska.tif"%EP) as d: dem=d.read(1); tr=d.transform; prof=d.profile; shp=d.shape
with rasterio.open(T+"/land_acc_%s.tif"%EP) as d: acc=d.read(1)
with rasterio.open(T+"/land_pntr_%s.tif"%EP) as d: pntr=d.read(1)
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP) as d: ws=d.read(1); wstags=d.tags()
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_%ska.tif"%EP) as d: bid=d.read(1)
land=dem!=ND
steplen=d8_step_len_km(pntr,tr,shp[0]); ra=row_area_km2(tr,shp[0])
sm = land & (acc>=T_CHOSEN)
L=float(steplen[sm].sum()); lab,frag=nd.label(sm,structure=S8)

# GUARDS on the real product
bw,bu = guard_no_channel_on_water_or_unobservable(sm, ws, "drainage_implied@T_CHOSEN")
print("GUARD no_channel_on_water_or_unobservable PASS (water=%d unobservable=%d)"%(bw,bu))

out=PROBE+"/out/drainage_implied_%ska.tif"%EP
write_tif(out, sm.astype(np.uint8), prof, 0, {
 "probe_product":"drainage_implied",
 "probe_product_is_not":"a palaeo-river network. It is the drainage the epoch's own land surface implies under D8 routing after the epoch's own water was removed. No channel here was observed; none is dated; width, discharge and permanence are not claimed",
 "probe_method":"water_state bits water|unobservable -> nodata; WBT FillDepressions on the land only; WBT D8Pointer; WBT D8FlowAccumulation in cells; channel = accumulation >= threshold",
 "probe_threshold_cells":T_CHOSEN,
 "probe_threshold_rule":"largest threshold whose local log-log slope of drainage density vs threshold still lies within one hundredth of the plateau exponent the sweep itself exhibits at its finest end; the sweep is the justification, no closed form was used",
 "probe_threshold_plateau_exponent":PLATEAU,
 "probe_threshold_upper_bound_cells":T_MANYCH_DEAD,
 "probe_threshold_upper_bound_rule":"the first swept threshold at which the Manych corridor window carries zero channel length; any threshold at or above it erases the feature this scene exists to show",
 "probe_epoch_ka":EP,
 "probe_input_dem":PROBE+"/in/dem_land_only_%ska.tif"%EP,
 "probe_input_dem_sha256":sha256_file(PROBE+"/in/dem_land_only_%ska.tif"%EP),
 "probe_input_water_state_sha256":sha256_file("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP),
 "probe_wbt_version":"WhiteboxTools v2.4.0",
 "probe_channel_cells":int(sm.sum()),
 "probe_channel_length_km":L,
 "probe_network_fragments_8connected":int(frag),
 "probe_land_area_km2":float((land.sum(axis=1)*ra).sum()),
 "probe_drainage_density_km_per_km2":L/float((land.sum(axis=1)*ra).sum()),
 "probe_guard_channel_cells_on_water":int(bw),
 "probe_guard_channel_cells_on_unobservable":int(bu),
 "probe_value_encoding":"1 = channel cell, 0 = not a channel cell; 0 is also the declared nodata, and every masked cell is 0 because no channel can be asserted there",
})
print("wrote", out, os.path.getsize(out))
print("channel cells %d  length %.1f km  fragments %d"%(sm.sum(), L, frag))
np.save(T+"/streams_honest_%s.npy"%EP, sm)
np.save(T+"/steplen_%s.npy"%EP, steplen.astype(np.float32))
open(T+"/T_CHOSEN.txt","w").write(repr(T_CHOSEN))
