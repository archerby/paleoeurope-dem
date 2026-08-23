import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
import rasterio, numpy as np, whitebox, os, time

DEM=PROBE+"/in/khv_16.5ka_nopred.tif"
T=PROBE+"/tmp"
wbt=whitebox.WhiteboxTools(); wbt.work_dir=T; wbt.set_verbose_mode(False)

def run(name, fn, *a, **k):
    t0=time.time(); L=[]
    fn(*a, callback=lambda s: L.append(s), **k)
    bad=[l for l in L if 'panic' in l or 'Error' in l or 'error' in l]
    print("[%s] %.1fs  panic/err lines: %s" % (name, time.time()-t0, bad if bad else "none"))
    return L

FILL=T+"/fill_naive.tif"
run("FillDepressions(default)", wbt.fill_depressions, DEM, FILL)
print("artifact exists:", os.path.exists(FILL), os.path.getsize(FILL) if os.path.exists(FILL) else 0)

with rasterio.open(DEM) as d: dem=d.read(1); prof=d.profile; tr=d.transform
with rasterio.open(FILL) as d: fl=d.read(1)
print("fill dtype", fl.dtype, "shape", fl.shape)
dz = fl.astype(np.float64) - dem.astype(np.float64)
n = dem.size
moved = dz > 0
nmoved = int(moved.sum())
print("=== FILL DELTA ===")
print("total cells                 :", n)
print("cells moved (dz>0)          : %d  (%.4f%%)" % (nmoved, 100.0*nmoved/n))
for eps in (0.001,0.01,0.1,1.0,10.0,100.0):
    print("  cells moved > %8.3f m : %d (%.4f%%)" % (eps,int((dz>eps).sum()),100.0*(dz>eps).sum()/n))
print("max dz m                    : %.6f" % dz.max())
print("sum dz m                    : %.6f" % dz.sum())
print("mean dz over moved cells m  : %.6f" % (dz[moved].mean() if nmoved else float('nan')))
print("cells lowered (dz<0)        :", int((dz<0).sum()), " min dz:", float(dz.min()))
# where is the max lift
r,c = np.unravel_index(int(np.argmax(dz)), dz.shape)
lon,lat = tr*(c+0.5, r+0.5)
print("max-lift cell               : row %d col %d  lon %.6f lat %.6f  dem %.3f -> fill %.3f" % (r,c,lon,lat,dem[r,c],fl[r,c]))
# area of moved cells in km2 (cells are lat-dependent)
import math
lats = np.array([ (tr*(0.5, i+0.5))[1] for i in range(dem.shape[0]) ])
dxdeg = tr.a; dydeg = -tr.e
R=6371.0088
row_area = (R*math.pi/180*dydeg) * (R*math.pi/180*dxdeg*np.cos(np.radians(lats)))  # km2 per cell in that row
moved_per_row = moved.sum(axis=1)
print("moved area km2              : %.1f" % float((moved_per_row*row_area).sum()))
print("total area km2              : %.1f" % float((dem.shape[1]*row_area).sum()))
np.save(PROBE+"/tmp/row_area_km2_4320.npy", row_area)
