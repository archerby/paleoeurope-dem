import rasterio, numpy as np, glob, re
ws="/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_16.5ka.tif"
bi="/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_16.5ka.tif"
wl="/mnt/tank/paleo/renders/khvalynian_basin_fields/water_level/water_level_16.5ka.tif"
with rasterio.open(ws) as d:
    t=d.tags()
    for k in t:
        if 'bit' in k.lower() or 'state' in k.lower() or 'unobserv' in k.lower():
            print("WS TAG %-34s = %s"%(k,t[k][:900]))
    a=d.read(1); u,c=np.unique(a,return_counts=True)
    print("WS unique:", list(zip(u.tolist(),c.tolist())))
print("-"*70)
with rasterio.open(bi) as d:
    t=d.tags()
    for k in ('basin_code_map','nodata_declaration','product'):
        if k in t: print("BI TAG %-24s = %s"%(k,t[k][:400]))
    a=d.read(1); u,c=np.unique(a,return_counts=True); print("BI unique:", list(zip(u.tolist(),c.tolist())))
print("-"*70)
with rasterio.open(wl) as d:
    t=d.tags(); print("WL dtype",d.dtypes,"nodata",d.nodatavals)
    for k in ('product','product_note','nodata_declaration','nodata_value'):
        if k in t: print("WL TAG %-24s = %s"%(k,t[k][:400]))
    a=d.read(1); u,c=np.unique(a,return_counts=True)
    print("WL n_unique",len(u), "sample", list(zip(u.tolist()[:12],c.tolist()[:12])))
