import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from grid import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd, whitebox, os, time, json
from rasterio.windows import Window
T=PROBE+"/tmp"; EP="16.5"; ND=-9999.0
TCH=float(open(T+"/T_CHOSEN.txt").read())
print("threshold in use: %.0f cells"%TCH)
wbt=whitebox.WhiteboxTools(); wbt.work_dir=T; wbt.set_verbose_mode(False)
def run(n,f,*a,**k):
    t0=time.time(); L=[]; f(*a,callback=lambda s:L.append(s),**k)
    bad=[l for l in L if 'panic' in l.lower() or 'error' in l.lower()]
    if bad: print("  [%s] ERR %s"%(n,bad))
    return L
def rc_from(tr): 
    inv=~tr
    return lambda lon,lat: (int((inv*(lon,lat))[1]), int((inv*(lon,lat))[0]))

TOPO="/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_%ska.tif"%EP
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%EP) as d:
    ws=d.read(1); tr=d.transform; wprof=d.profile
with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_%ska.tif"%EP) as d: bid=d.read(1)
with rasterio.open(TOPO) as d:
    demraw=d.read(1,window=Window(0,720,7800,3600)); prof=d.profile.copy()
    prof.update(width=7800,height=3600,transform=d.window_transform(Window(0,720,7800,3600)))
shp=demraw.shape; ra=row_area_km2(tr,shp[0])
honest=np.load(T+"/streams_honest_%s.npy"%EP)
steplen_h=np.load(T+"/steplen_%s.npy"%EP).astype(np.float64)

def pipeline(tag, dem_in, nodata):
    p=T+"/x_%s.tif"%tag
    write_tif(p, dem_in, prof, nodata, {"probe_product":"probe_intermediate_dem","probe_variant":tag})
    f=T+"/x_%s_fill.tif"%tag; pt=T+"/x_%s_pntr.tif"%tag; ac=T+"/x_%s_acc.tif"%tag
    run("fill",wbt.fill_depressions,p,f); run("pntr",wbt.d8_pointer,f,pt); run("acc",wbt.d8_flow_accumulation,f,ac,out_type="cells")
    with rasterio.open(ac) as d: acc=d.read(1)
    with rasterio.open(pt) as d: pn=d.read(1)
    valid = dem_in!=nodata
    sm = valid & (acc>=TCH)
    sl = d8_step_len_km(pn,tr,shp[0])
    return sm, sl

def report(name, sm, sl):
    L=float(sl[sm].sum()); lab,frag=nd.label(sm,structure=S8)
    onw=float(sl[sm&((ws&1)>0)].sum()); onu=float(sl[sm&((ws&2)>0)].sum())
    casp=float(sl[sm&((ws&1)>0)&(bid==1)].sum()); pont=float(sl[sm&((ws&1)>0)&(bid==2)].sum()); med=float(sl[sm&((ws&1)>0)&(bid==3)].sum())
    inter=int((sm&honest).sum()); union=int((sm|honest).sum())
    print("  %-26s cells %-9d len %9.1f km  frags %-5d | on 16.5ka WATER %9.1f km (%5.2f%%)  on UNOBSERVABLE %7.1f km (%.3f%%)"%(
        name, sm.sum(), L, frag, onw, 100.0*onw/L if L else 0, onu, 100.0*onu/L if L else 0))
    print("  %-26s   of that water length: caspian %.1f km, pontic %.1f km, mediterranean %.1f km"%("",casp,pont,med))
    print("  %-26s   Jaccard vs honest-16.5 network = %.6f  (inter %d / union %d)"%("",inter/union if union else 0,inter,union))
    return dict(cells=int(sm.sum()),length_km=L,frags=int(frag),on_water_km=onw,on_unobs_km=onu,
                casp_km=casp,pont_km=pont,med_km=med,jaccard=inter/union if union else 0)

print("\n=== A) NAIVE FILL, no water mask at all, same window, same threshold ===")
sm_n, sl_n = pipeline("naive_nomask", demraw.astype(np.float32), -9999.0)
res_n=report("naive (unmasked)", sm_n, sl_n)
print("  honest network for reference:")
res_h=report("honest (16.5ka mask)", honest, steplen_h)

print("\n=== B) same DEM, NEIGHBOUR EPOCH's water mask ===")
out={}
for nep in ("16","17"):
    wp="/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%nep
    with rasterio.open(wp) as d:
        wsn=d.read(1); assert tuple(d.transform)[:6]==tuple(tr)[:6], "grid mismatch"
        tg=d.tags()
    m=(wsn&1)>0; u=(wsn&2)>0
    dm=demraw.astype(np.float32).copy(); dm[m|u]=ND
    print("  mask from %s ka: water %d unobservable %d (16.5ka has water %d unobs %d); masked %d vs %d"%(
        nep, m.sum(), u.sum(), ((ws&1)>0).sum(), ((ws&2)>0).sum(), (m|u).sum(), (((ws&1)>0)|((ws&2)>0)).sum()))
    print("    that epoch: spilling=%s connected=%s caspian_level=%s"%(tg['spilling'],tg['connected_groups'],tg['basin_level_m_caspian']))
    sm_x, sl_x = pipeline("mask%s"%nep, dm, ND)
    out[nep]=report("mask=%s ka"%nep, sm_x, sl_x)
    # how much of THAT network lands on 16.5ka's own water/unobservable -> the error of using the wrong epoch
    np.save(T+"/streams_mask%s.npy"%nep, sm_x)
json.dump({"naive":res_n,"honest":res_h,"neighbour":out,"threshold_cells":TCH}, open(T+"/job4.json","w"), indent=1)
np.save(T+"/streams_naive.npy", sm_n)
