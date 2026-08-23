import rasterio, glob, re, os
rows=[]
for f in sorted(glob.glob("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_*ka.tif")):
    ep=float(re.search(r'_([0-9.]+)ka',f).group(1))
    with rasterio.open(f) as d: t=d.tags()
    sill=t.get('sill_caspian__pontic','')
    m=dict(kv.split('=',1) for kv in sill.split(';') if '=' in kv) if sill else {}
    rows.append((ep, t.get('spilling','?'), t.get('connected_groups','?'),
                 m.get('measured_saddle_m','?'), m.get('expected_m','?'),
                 m.get('lon','?'), m.get('lat','?'),
                 t.get('basin_level_m_caspian','?'), t.get('basin_level_m_pontic','?'),
                 t.get('connectivity_unobservable_cells','?')))
rows.sort()
print("%-7s %-8s %-24s %-14s %-12s %-11s %-11s %-12s %-12s %s"%(
  "ka","spilling","connected_groups","saddle_m","expected_m","lon","lat","casp_lvl","pont_lvl","unobs"))
for r in rows: print("%-7s %-8s %-24s %-14s %-12s %-11s %-11s %-12s %-12s %s"%r)
# is 'endorheic' a word anywhere?
print("\n--- grep endorheic across water_state tags ---")
for f in sorted(glob.glob("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_*ka.tif")):
    with rasterio.open(f) as d: t=d.tags()
    hits=[k for k,v in t.items() if 'endorheic' in str(v).lower() or 'endorheic' in k.lower()]
    if hits: print(os.path.basename(f), hits[:6])
print("\n--- grep endorheic across topo tags (16.5 & neighbours) ---")
for f in ["/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_%ska.tif"%e for e in ("16.5","17","16","20","21","22","23","24","25","26")]:
    with rasterio.open(f) as d: t=d.tags()
    hits=[(k,str(v)[:120]) for k,v in t.items() if 'endorheic' in str(v).lower()]
    print(os.path.basename(f), "verdict_def=",t.get('sill_verdict_defensible'), "verdict_max=",t.get('sill_verdict_maximal'), "hits:",hits[:3])
