import rasterio
for path in [
 "/mnt/tank/paleo/renders/khvalynian_caspian/topo/khv_16.5ka.tif",
 "/mnt/tank/paleo/renders/khvalynian_basin_fields/water_state/water_state_16.5ka.tif",
 "/mnt/tank/paleo/renders/khvalynian_basin_fields/basin_id/basin_id_16.5ka.tif",
 "/mnt/tank/paleo/renders/khvalynian_basin_fields/water_level/water_level_16.5ka.tif",
]:
    print("="*80); print(path)
    with rasterio.open(path) as ds:
        print("  driver",ds.driver,"dtype",ds.dtypes,"shape",ds.shape,"crs",ds.crs)
        print("  nodata",ds.nodatavals)
        print("  transform",tuple(ds.transform)[:6])
        print("  profile:", {k:v for k,v in ds.profile.items() if k!='transform'})
        print("  IMAGE_STRUCTURE:", ds.tags(ns='IMAGE_STRUCTURE'))
        t = ds.tags()
        for k in sorted(t): print("   tag %-42s = %s" % (k, t[k]))
        print("  band descriptions:", ds.descriptions)
