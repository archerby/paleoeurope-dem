import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
import rasterio, numpy as np, whitebox, os, time
T=PROBE+"/tmp"
wbt=whitebox.WhiteboxTools(); wbt.work_dir=T; wbt.set_verbose_mode(False)
FILL=T+"/fill_naive.tif"; PNTR=T+"/fill_pntr.tif"; ACC=T+"/fill_acc.tif"; BAS=T+"/fill_basins.tif"
def run(n,f,*a,**k):
    t0=time.time(); L=[]; f(*a,callback=lambda s:L.append(s),**k)
    bad=[l for l in L if 'panic' in l.lower() or 'error' in l.lower()]
    print("[%s] %.1fs err:%s"%(n,time.time()-t0,bad or "none"))
run("D8Pointer",wbt.d8_pointer,FILL,PNTR)
run("D8FlowAccumulation",wbt.d8_flow_accumulation,FILL,ACC,out_type="cells")
run("Basins",wbt.basins,PNTR,BAS)
for f in (PNTR,ACC,BAS): print(" artifact",os.path.basename(f),os.path.exists(f),os.path.getsize(f))

with rasterio.open(FILL) as d: tr=d.transform; shp=d.shape
with rasterio.open(BAS) as d: bas=d.read(1)
with rasterio.open(ACC) as d: acc=d.read(1)
with rasterio.open(PNTR) as d: pnt=d.read(1)
n=bas.size
vals,cnts=np.unique(bas,return_counts=True)
order=np.argsort(-cnts)
print("=== BASINS draining to the DEM edge ===")
print("distinct basin ids:",len(vals))
print("top 10 by cell count:")
for i in order[:10]:
    v=int(vals[i]); c=int(cnts[i])
    m = bas==v
    # outlet = max-accumulation cell in the basin
    idx=int(np.argmax(np.where(m,acc,-1)))
    r,cc=np.unravel_index(idx,shp); lon,lat=tr*(cc+0.5,r+0.5)
    print("  id %-6d cells %-10d (%6.3f%%)  outlet lon %.5f lat %.5f  acc %.0f  pntr %d"%(
        v,c,100.0*c/n,lon,lat,acc[r,cc],pnt[r,cc]))
gmax=int(np.argmax(acc)); r,cc=np.unravel_index(gmax,shp); lon,lat=tr*(cc+0.5,r+0.5)
print("global max accumulation: %.0f cells (%.4f%% of grid) at lon %.5f lat %.5f  basin id %d"%(
    acc.flat[gmax],100.0*acc.flat[gmax]/n,lon,lat,bas[r,cc]))
print("share of grid in the single largest basin: %.4f%%"%(100.0*cnts[order[0]]/n))
