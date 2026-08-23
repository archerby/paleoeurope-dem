import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from grid import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd, json
from rasterio.windows import Window
T=PROBE+"/tmp"; EP="16.5"; ND=-9999.0
TCH=float(open(T+"/T_CHOSEN.txt").read())
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP) as d: ws=d.read(1); tr=d.transform
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_%ska.tif"%EP) as d: bid=d.read(1)
with rasterio.open(PROBE+"/in/dem_land_only_%ska.tif"%EP) as d: dem=d.read(1); shp=d.shape
honest=np.load(T+"/streams_honest_%s.npy"%EP); naive=np.load(T+"/streams_naive.npy")
land = dem!=ND
print("=== 1) is the briefed Jaccard 0.985 a LAND-RESTRICTED Jaccard? ===")
def jac(a,b,restrict=None):
    if restrict is not None: a=a&restrict; b=b&restrict
    u=(a|b).sum(); return (a&b).sum()/u if u else 0.0, int((a&b).sum()), int(u)
print("  whole window        : %.6f  (inter %d / union %d)"%jac(naive,honest))
print("  restricted to LAND  : %.6f  (inter %d / union %d)"%jac(naive,honest,land))
print("  naive cells on land %d, on water %d, on unobservable %d"%(
    int((naive&land).sum()), int((naive&((ws&1)>0)).sum()), int((naive&((ws&2)>0)).sum())))
print("  fraction of the HONEST network that the naive network also draws: %.6f"%((naive&honest).sum()/honest.sum()))

print("\n=== 2) the Caspian-lake-bed length is threshold-dependent: same measurement across the sweep ===")
with rasterio.open(T+"/x_naive_nomask_acc.tif") as d: acc_n=d.read(1)
with rasterio.open(T+"/x_naive_nomask_pntr.tif") as d: pn_n=d.read(1)
sl_n=d8_step_len_km(pn_n,tr,shp[0])
casp_water=((ws&1)>0)&(bid==1)
print("  %-12s %-14s %-16s %-12s"%("T_cells","total_km","on_caspian_km","pct"))
for Tv in (1000,1778,3162,5623,10000,17783,31623,56234,100000):
    sm=acc_n>=Tv; L=float(sl_n[sm].sum()); Lc=float(sl_n[sm&casp_water].sum())
    print("  %-12d %-14.1f %-16.1f %-12.2f"%(Tv,L,Lc,100.0*Lc/L if L else 0))

print("\n=== 3) outlets: naive vs honest ===")
for nm, pth, valid in (("naive(unmasked)",T+"/x_naive_nomask_pntr.tif",np.ones(shp,bool)),
                       ("honest(masked)", T+"/land_pntr_%s.tif"%EP, land)):
    with rasterio.open(pth) as d: pn=d.read(1)
    term = valid & (pn==0)
    lab,nb = nd.label(valid, structure=S8)
    print("  %-16s terminal cells (pointer==0) inside the valid domain: %d"%(nm,int(term.sum())))
    with rasterio.open(T+("/x_naive_nomask_acc.tif" if "naive" in nm else "/land_acc_%s.tif"%EP)) as d: ac=d.read(1)
    tv=ac[term]
    if tv.size:
        o=np.argsort(-tv)[:5]
        rr,cc=np.nonzero(term)
        print("    top-5 outlets by contributing cells:")
        for i in o:
            lon,lat=tr*(cc[i]+0.5,rr[i]+0.5)
            print("      lon %9.5f lat %8.5f  acc %12.0f  (%.3f%% of the valid domain)"%(lon,lat,tv[i],100.0*tv[i]/valid.sum()))
        print("    share of the valid domain draining to the single largest outlet: %.4f%%"%(100.0*tv.max()/valid.sum()))

print("\n=== 4) what CANNOT be measured with the shipped fields ===")
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_%ska.tif"%EP) as d:
    ftr=d.transform; fshp=d.shape
ra_full=row_area_km2(ftr,fshp[0])
north = float((fshp[1]*ra_full[:720]).sum())
print("  the topo raster spans lat %.4f..%.4f (%d rows); the water_state/basin_id fields span lat %.4f..%.4f (%d rows)"%(
   (ftr*(0,0))[1], (ftr*(0,fshp[0]))[1], fshp[0], (tr*(0,0))[1], (tr*(0,shp[0]))[1], shp[0]))
print("  rows of topo with NO water_state at all: %d  = %.1f km2 (%.2f%% of the topo extent)"%(
   720, north, 100.0*north/float((fshp[1]*ra_full).sum())))
print("  -> for those cells the epoch's water state is not shipped, so their drainage state is UNOBSERVABLE.")
print("     They are excluded, not called land. The contributing area they would send south across 51 N is")
print("     NOT MEASURABLE from the shipped fields: computing it requires a water mask north of 51 N that does not exist.")
ra=row_area_km2(tr,shp[0])
print("  cells the field calls CLOSED (below their basin level, rim inside the extent, not water):")
print("     not directly encoded in water_state; derivable only with water_level, and this probe did not separate them.")
print("  corridor-marked cells carrying channel: %d cells, %.1f km"%(
   int((honest&((ws&4)>0)).sum()), float(np.load(T+"/steplen_%s.npy"%EP).astype(np.float64)[honest&((ws&4)>0)].sum())))
