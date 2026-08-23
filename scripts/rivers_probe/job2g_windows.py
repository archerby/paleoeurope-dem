import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
import rasterio, numpy as np
T=PROBE+"/tmp"
def rowcol(tr,lon,lat):
    inv=~tr; c,r=inv*(lon,lat); return int(r),int(c)
for ep in ("13","16.5"):
    with rasterio.open(PROBE+"/in/khv_%ska_nopred.tif"%ep) as d: dem=d.read(1); tr=d.transform
    with rasterio.open(T+"/breach_%s.tif"%ep) as d: br=d.read(1).astype(np.float32)
    print("EPOCH %s ka - min breached elevation inside nested windows centred on the Manych sill"%ep)
    for half_lon, half_lat in ((0.25,0.25),(0.5,0.5),(1.0,0.5),(2.0,1.0),(3.0,1.25),(4.0,1.5)):
        r0,c0=rowcol(tr,44.11875-half_lon,45.72+half_lat); r1,c1=rowcol(tr,44.11875+half_lon,45.72-half_lat)
        w=br[r0:r1,c0:c1]; wn=dem[r0:r1,c0:c1]
        print("   +/-%.2f lon +/-%.2f lat : native min %10.3f  breached min %11.3f"%(half_lon,half_lat,wn.min(),w.min()))
    del dem,br
