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
