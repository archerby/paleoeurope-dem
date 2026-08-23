import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from guards import *
import numpy as np, rasterio
PROBE="/mnt/tank/paleo/renders/_rivers_probe_2026-08-18"; T=PROBE+"/tmp"; EP="16.5"; ND=-9999.0
sm=np.load(T+"/streams_honest_%s.npy"%EP)
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP) as d: ws=d.read(1)
with rasterio.open(PROBE+"/in/dem_land_only_%ska.tif"%EP) as d: dem=d.read(1)
print("=== PLANTED-FAILURE PROOFS: each guard is shown to fire on the exact defect it exists to catch ===")
def prove(name, fn):
    try:
        fn(); print("  %-40s DID NOT FIRE  <<< THE GUARD IS USELESS"%name)
    except GuardFailure as e:
        print("  %-40s FIRED: %s"%(name, e))

# 1) plant a channel cell on a WATER cell
wr,wc = np.argwhere((ws&1)>0)[0]
bad=sm.copy(); bad[wr,wc]=True
prove("no_channel_on_water (planted water)", lambda: guard_no_channel_on_water_or_unobservable(bad, ws, "PLANTED"))
# 2) plant a channel cell on an UNOBSERVABLE cell
ur,uc = np.argwhere((ws&2)>0)[0]
bad2=sm.copy(); bad2[ur,uc]=True
prove("no_channel_on_unobservable (planted)", lambda: guard_no_channel_on_water_or_unobservable(bad2, ws, "PLANTED"))
# 3) collapse ONE unobservable cell into land in the DEM
bad3=dem.copy(); bad3[ur,uc]=0.0
prove("third_state_not_collapsed (planted)", lambda: guard_third_state_not_collapsed(bad3, ND, ws, "PLANTED"))
# 4) break the partition: un-mask one water cell
bad4=dem.copy(); bad4[wr,wc]=1.0
prove("mask_partition (planted)", lambda: guard_mask_partition(bad4, ND, ws, "PLANTED"))
print("=== and the same four guards on the UNMODIFIED artifact ===")
print("  no_channel_on_water_or_unobservable :", guard_no_channel_on_water_or_unobservable(sm, ws, "real"))
print("  third_state_not_collapsed           :", guard_third_state_not_collapsed(dem, ND, ws, "real"))
print("  mask_partition                      :", guard_mask_partition(dem, ND, ws, "real"))
