import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from grid import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd, math, json
T=PROBE+"/tmp"; EP="16.5"; ND=-9999.0
with rasterio.open(PROBE+"/in/dem_land_only_%ska.tif"%EP) as d:
    dem=d.read(1); tr=d.transform; shp=d.shape
with rasterio.open(T+"/land_acc_%s.tif"%EP) as d: acc=d.read(1)
with rasterio.open(T+"/land_pntr_%s.tif"%EP) as d: pntr=d.read(1)
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP) as d: ws=d.read(1)
land = dem!=ND
steplen = d8_step_len_km(pntr, tr, shp[0])
ra = row_area_km2(tr, shp[0])
land_area = float((land.sum(axis=1)*ra).sum())
print("land area km2: %.1f"%land_area)
# where is the max fill lift on land?
with rasterio.open(T+"/land_fill_%s.tif"%EP) as d: fl=d.read(1)
dzl = np.where(land, fl.astype(np.float64)-dem.astype(np.float64), 0.0)
r,c=np.unravel_index(int(np.argmax(dzl)),shp); lon,lat=tr*(c+0.5,r+0.5)
print("max land-fill lift %.3f m at lon %.4f lat %.4f (dem %.2f -> %.2f); basin_id/water_state there = %d"%(
  dzl.max(),lon,lat,dem[r,c],fl[r,c],ws[r,c]))
del fl, dzl

# Manych diagnostic window
inv=~tr
def rc(lon,lat):
    cc,rr=inv*(lon,lat); return int(rr),int(cc)
mr0,mc0=rc(42.0,47.5); mr1,mc1=rc(48.0,45.0)
manych=np.zeros(shp,bool); manych[mr0:mr1,mc0:mc1]=True
corr=(ws&4)>0

print("\nthreshold sweep: streams = {D8 accumulation >= T}, T in CELLS of contributing area")
print("%-10s %-12s %-13s %-12s %-11s %-9s %-12s %-10s"%(
  "T_cells","km2_equiv","stream_cells","length_km","dens_km/km2","frags","manych_km","corr_km"))
rows=[]
Ts=[]
v=100.0
while v<=3.0e6:
    Ts.append(v); v*= 10**0.25
for Tv in Ts:
    sm = land & (acc>=Tv)
    n=int(sm.sum())
    if n==0:
        rows.append((Tv,0,0.0,0.0,0,0.0,0.0)); continue
    L=float(steplen[sm].sum())
    dens=L/land_area
    lab,frag=nd.label(sm,structure=S8)
    Lm=float(steplen[sm&manych].sum()); Lc=float(steplen[sm&corr].sum())
    # cell area equivalent of T at the median stream latitude
    km2=float(Tv*ra.mean())
    rows.append((Tv,n,L,dens,frag,Lm,Lc))
    print("%-10.0f %-12.1f %-13d %-12.1f %-11.6f %-9d %-12.1f %-10.1f"%(Tv,km2,n,L,dens,frag,Lm,Lc))
json.dump([[float(x) for x in r] for r in rows], open(T+"/sweep_%s.json"%EP,"w"))
# monotonicity check
print("\nmonotonicity of each column as T increases:")
import itertools
names=["stream_cells","length_km","density","fragments","manych_km","corridor_km"]
arr=np.array([[r[i+1] for r in rows] for i in range(6)],float)
for i,nm in enumerate(names):
    d=np.diff(arr[i]); ups=int((d>0).sum()); downs=int((d<0).sum())
    print("  %-14s strictly-decreasing:%s  increases at %d of %d steps"%(nm,(ups==0),ups,len(d)))
    if ups:
        bad=[(rows[j][0],arr[i][j],arr[i][j+1]) for j in np.nonzero(d>0)[0]]
        for b in bad[:6]: print("      NON-MONOTONE at T=%.0f : %.4f -> %.4f"%b)
# log-log local slope of density vs T
lt=np.log10(np.array([r[0] for r in rows])); ld=np.log10(np.maximum(arr[2],1e-12))
sl=np.gradient(ld,lt)
print("\nlocal log-log slope d(log density)/d(log T):")
for j in range(len(rows)):
    print("   T=%-10.0f dens=%.6f  slope=%+.4f"%(rows[j][0],arr[2][j],sl[j]))
