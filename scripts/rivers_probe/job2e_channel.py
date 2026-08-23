import sys; sys.path.insert(0,"/mnt/tank/paleo/renders/_rivers_probe_2026-08-18")
from paleoprobe import *
from minimax import S4,S8
import rasterio, numpy as np, scipy.ndimage as nd, glob, re
T=PROBE+"/tmp"
print("--- caspian seed elevation across epochs (looking for -1080.13) ---")
for f in sorted(glob.glob("/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_*ka.tif"),
                key=lambda x: float(re.search(r'_([0-9.]+)ka',x).group(1))):
    ep=re.search(r'_([0-9.]+)ka',f).group(1)
    with rasterio.open(f) as d: t=d.tags()
    sc=dict(kv.split('=',1) for kv in t['basin_seed_caspian'].split(';') if '=' in kv)
    print("  %-6s caspian seed elev %s"%(ep, sc['elevation_m']))
