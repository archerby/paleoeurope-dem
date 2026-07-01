"""Tests for the ICE-6G Topo → present-datum solid displacement helpers."""
import numpy as np
import pytest

from paleoeurope.gia.ice6g_topo import implied_sea_surface_offset, solid_displacement


def test_offset_is_deep_ocean_median_of_delta():
    # Two deep-ocean cells (topo0 < -1000) both rose by exactly the sea-surface
    # amount (+112); one shallow/land cell (topo0 = 500) must NOT enter the median.
    topo_0 = np.array([[-2000.0, -3000.0, 500.0]])
    topo_t = np.array([[-2000.0 + 112.0, -3000.0 + 112.0, 500.0 + 999.0]])
    assert implied_sea_surface_offset(topo_t, topo_0) == pytest.approx(112.0)


def test_solid_displacement_zero_in_far_field():
    # Pure sea-surface change over deep ocean → solid displacement ~ 0.
    topo_0 = np.array([[-4000.0, -3500.0]])
    topo_t = topo_0 + 112.0
    offset = implied_sea_surface_offset(topo_t, topo_0)
    dz = solid_displacement(topo_t, topo_0, offset)
    assert np.allclose(dz, 0.0, atol=1e-6)


def test_solid_displacement_recovers_uplift_over_land():
    # Deep ocean sets the offset (+112); a land cell that additionally rose 50 m
    # must report Δz_solid = +50, NOT +162 (the bare-ΔTopo double-count bug).
    topo_0 = np.array([[-4000.0, -3500.0, 300.0]])
    topo_t = np.array([[-4000.0 + 112.0, -3500.0 + 112.0, 300.0 + 112.0 + 50.0]])
    offset = implied_sea_surface_offset(topo_0=topo_0, topo_t=topo_t)
    dz = solid_displacement(topo_t, topo_0, offset)
    assert dz[0, 2] == pytest.approx(50.0)
    assert np.allclose(dz[0, :2], 0.0, atol=1e-6)


def test_offset_raises_when_no_deep_ocean_cells():
    # All cells are shallow/land (topo_0 >= -1000) → no deep-ocean subset exists,
    # so there is no far-field basis for the sea-surface offset.
    topo_0 = np.array([[10.0, 500.0]])
    topo_t = topo_0 + 5.0
    with pytest.raises(ValueError):
        implied_sea_surface_offset(topo_t, topo_0)


def test_offset_ignores_nan_deep_cell():
    # Three deep-ocean cells: two rose by +112, one is NaN (an ICE-6G _FillValue).
    # nanmedian must ignore the NaN and still recover +112 (plain median would NaN).
    topo_0 = np.array([[-2000.0, -3000.0, -4000.0]])
    topo_t = np.array([[-2000.0 + 112.0, -3000.0 + 112.0, np.nan]])
    assert implied_sea_surface_offset(topo_t, topo_0) == pytest.approx(112.0)


# append to tests/test_gia_topo.py
import os
from pathlib import Path
import xarray as xr

ICE6G_DIR = Path(
    r"D:/pasha/RESEARCH/Paleo-Europe_3D/data/raw/ICE6G"
)

pytestmark_data = pytest.mark.skipif(
    not (ICE6G_DIR / "I6_C.VM5a_10min.18.nc").exists(),
    reason="ICE-6G data not present",
)


@pytestmark_data
def test_far_field_solid_displacement_near_zero_real_data():
    with xr.open_dataset(ICE6G_DIR / "I6_C.VM5a_10min.18.nc") as ds_t, \
         xr.open_dataset(ICE6G_DIR / "I6_C.VM5a_10min.0.nc") as ds_0:
        topo_t = ds_t["Topo"].values.astype(np.float64)
        topo_0 = ds_0["Topo"].values.astype(np.float64)
        offset = implied_sea_surface_offset(topo_t, topo_0)
        dz = solid_displacement(topo_t, topo_0, offset)
        # Three far-field deep-ocean points (far from ice): mid-Pacific, Indian, S-Atlantic.
        for lat, lon in [(-20.0, 200.0), (-30.0, 80.0), (-30.0, 340.0)]:
            v = float(
                xr.DataArray(dz, coords={"lat": ds_t["lat"], "lon": ds_t["lon"]},
                             dims=("lat", "lon")).sel(lat=lat, lon=lon, method="nearest")
            )
            assert abs(v) < 8.0, f"far-field Δz_solid={v:.1f} m at ({lat},{lon}) should be ~0"
        # And the sea-surface offset at 18 ka must be a large positive lowstand value.
        assert 90.0 < offset < 130.0, f"18 ka sea-surface offset {offset:.1f} m out of range"
