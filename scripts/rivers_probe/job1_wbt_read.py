import whitebox, os, sys, io, contextlib
wbt = whitebox.WhiteboxTools()
wbt.set_verbose_mode(True)
wbt.work_dir = "/mnt/tank/paleo/renders/_rivers_probe_2026-08-18/tmp"
print("WBT:", wbt.version().strip().splitlines()[0])

SRC = "/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_16.5ka.tif"
print("\n### A) WBT FillDepressions directly on PREDICTOR=3 source")
lines=[]
rc = wbt.fill_depressions(SRC, "/mnt/tank/paleo/renders/_rivers_probe_2026-08-18/tmp/a_fill.tif",
                          callback=lambda s: lines.append(s))
print("return code:", rc)
print("---- WBT stdout verbatim ----")
for l in lines: print(l)
print("---- end ----")
print("output exists:", os.path.exists("/mnt/tank/paleo/renders/_rivers_probe_2026-08-18/tmp/a_fill.tif"))

print("\n### B) WBT on a basin field (no PREDICTOR tag) - control")
WS="/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_16.5ka.tif"
lines2=[]
rc2 = wbt.raster_summary_stats(WS, callback=lambda s: lines2.append(s))
print("return code:", rc2)
for l in lines2: print(l)
