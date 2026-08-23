import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from minimax import minimax, connected_at, S4, S8
import rasterio, numpy as np, time, scipy.ndimage as nd

T=PROBE+"/tmp"
def rowcol(tr,lon,lat):
    inv=~tr; c,r=inv*(lon,lat); return int(r),int(c)

for ep in ("13","16.5"):
    print("="*78); print("EPOCH",ep,"ka")
    with rasterio.open(PROBE+"/in/khv_%ska_nopred.tif"%ep) as d:
        dem=d.read(1); tr=d.transform
    with rasterio.open(T+"/breach_%s.tif"%ep) as d:
        br=d.read(1).astype(np.float32)
    with rasterio.open("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_%ska.tif"%ep) as w:
        wt=w.tags()
    sc = dict(kv.split('=',1) for kv in wt['basin_seed_caspian'].split(';') if '=' in kv)
    sp = dict(kv.split('=',1) for kv in wt['basin_seed_pontic'].split(';') if '=' in kv)
    A = rowcol(tr, float(sc['lon']), float(sc['lat']))
    B = rowcol(tr, float(sp['lon']), float(sp['lat']))
    print(" caspian seed lon %s lat %s -> row,col %s dem %.3f (tag elev %s)"%(sc['lon'],sc['lat'],A,dem[A],sc['elevation_m']))
    print(" pontic  seed lon %s lat %s -> row,col %s dem %.3f (tag elev %s)"%(sp['lon'],sp['lat'],B,dem[B],sp['elevation_m']))
    print(" shipped sill (4-conn, project routine): %s"%dict(kv.split('=',1) for kv in wt['sill_caspian__pontic'].split(';') if '=' in kv)['measured_saddle_m'])
    print(" caspian level %s  pontic level %s  spilling=%s"%(wt['basin_level_m_caspian'],wt['basin_level_m_pontic'],wt['spilling']))
    for name, arr in (("NATIVE",dem),("BREACHED",br)):
        for cn, st in (("4-conn",S4),("8-conn",S8)):
            t0=time.time(); v,it = minimax(arr, A, B, st)
            print("  %-8s %-7s minimax = %s   (%d bisections, %.0fs)"%(
                name, cn, ("%.6f m"%v) if v is not None else "NOT CONNECTED AT ANY LEVEL", it, time.time()-t0))
    # the guard question, verbatim: does the caspian seed's 4-connected flood at its OWN level reach the pontic seed?
    lvl=float(wt['basin_level_m_caspian'])
    for name,arr in (("NATIVE",dem),("BREACHED",br)):
        for cn,st in (("4-conn",S4),("8-conn",S8)):
            print("  GUARD %-8s %-7s at caspian level %.6f m: caspian seed reaches pontic seed = %s"%(
                name,cn,lvl,connected_at(arr,lvl,A,B,st)))
    del dem,br
