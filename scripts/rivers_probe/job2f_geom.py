import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd
T=PROBE+"/tmp"
def rowcol(tr,lon,lat):
    inv=~tr; c,r=inv*(lon,lat); return int(r),int(c)

for ep, slon, slat, casp_min in (("13",44.118750,45.743750,-1047.642578),("16.5",44.118750,45.710417,-1048.861206)):
    print("="*78); print("EPOCH",ep,"ka")
    with rasterio.open(PROBE+"/in/khv_%ska_nopred.tif"%ep) as d: dem=d.read(1); tr=d.transform
    with rasterio.open(T+"/breach_%s.tif"%ep) as d: br=d.read(1).astype(np.float32)
    print(" global min: native %.6f  breached %.6f  (breach descends %.6f m below native min)"%(
        dem.min(), br.min(), br.min()-dem.min()))
    # Manych window
    r0,c0 = rowcol(tr,42.0,47.5); r1,c1 = rowcol(tr,48.0,45.0)
    w_n = dem[r0:r1, c0:c1]; w_b = br[r0:r1, c0:c1]
    print(" Manych window rows %d..%d cols %d..%d (lon 42-48E lat 45-47.5N), %d cells"%(r0,r1,c0,c1,w_n.size))
    print("  native   min %.6f max %.6f"%(w_n.min(), w_n.max()))
    print("  breached min %.6f max %.6f"%(w_b.min(), w_b.max()))
    idx=np.unravel_index(int(np.argmin(w_b)), w_b.shape)
    lon,lat = tr*(c0+idx[1]+0.5, r0+idx[0]+0.5)
    print("  deepest breached cell in window: lon %.6f lat %.6f  native %.6f -> breached %.6f"%(
        lon,lat,w_n[idx],w_b[idx]))
    # THE CARVED CORRIDOR: cells of the breached DEM at or below the caspian floor
    thr = casp_min + 1e-3
    M = br <= thr
    lab8,n8 = nd.label(M, structure=S8); lab4,n4 = nd.label(M, structure=S4)
    A = rowcol(tr,50.743750,38.952083)  # caspian seed
    comp8 = lab8[A]; comp4 = lab4[A]
    m8 = lab8==comp8; m4 = lab4==comp4
    print("  corridor {breached <= %.6f}: %d cells; 8-conn components %d, 4-conn components %d"%(
        thr, int(M.sum()), n8, n4))
    print("  component holding the caspian seed: 8-conn %d cells, 4-conn %d cells  -> ratio %.1fx"%(
        int(m8.sum()), int(m4.sum()), m8.sum()/max(m4.sum(),1)))
    # restrict to the Manych window and count the cross-divide chain
    sub8 = m8[r0:r1, c0:c1]
    print("  in Manych window: 8-conn caspian corridor holds %d cells"%int(sub8.sum()))
    if sub8.any():
        l4s,n4s = nd.label(sub8, structure=S4); l8s,n8s = nd.label(sub8, structure=S8)
        sz4 = np.bincount(l4s.ravel())[1:]; 
        print("   that same cell set, relabelled: 8-conn comps %d, 4-conn comps %d"%(n8s,n4s))
        print("   4-conn component sizes: min %d median %d max %d ; %d of %d comps are single cells (%.1f%%)"%(
            sz4.min(), int(np.median(sz4)), sz4.max(), int((sz4==1).sum()), len(sz4), 100.0*(sz4==1).sum()/len(sz4)))
        rr,cc = np.nonzero(sub8)
        print("   window bbox of corridor rows %d..%d cols %d..%d ; cells per row: max %d, median %d"%(
            rr.min(),rr.max(),cc.min(),cc.max(), int(np.bincount(rr).max()), int(np.median(np.bincount(rr)[np.bincount(rr)>0]))))
    del dem,br,M,lab8,lab4,m8,m4
