"""
tests/test_correction_nan_policy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NaN policy for the correction-matrix path.

:func:`paleoeurope.gia.ice6g_topo.solid_displacement` returns **NaN** over cells
where ICE-6G ``Topo`` is the ice surface rather than the solid surface — these
files ship no bedrock-under-ice field at all, so there is no number to give.  The
whole value of that NaN is that it must never be turned into a number: a 0.0 in a
GIA correction reads as *"no isostatic correction here"*, a confident and wrong
statement, where the truth is *"unknown"*.

These tests pin the policy across the two production hops that used to destroy it:

1. ``read_correction_matrix()`` ended with ``np.nan_to_num(result, nan=0.0)``.
2. ``correction_pipeline`` built ``dz_tile`` with exactly that call and added it
   straight onto the modern DEM.

Kept in its own module (rather than in ``test_gia.py`` / ``test_gia_topo.py``)
because it is a cross-module policy test, not a unit test of either loader.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import rasterio.coords
import rasterio.crs
import rasterio.enums
import rasterio.transform

from paleoeurope.gia import correction_pipeline as cp
from paleoeurope.gia.correction_matrix import (
    build_correction_matrix,
    read_correction_matrix,
)
from paleoeurope.gia.ice6g_topo import solid_displacement

# ── Tile geometry: N62E006, the same tile the committed fixtures were cut for ──
TILE_WEST, TILE_SOUTH, TILE_EAST, TILE_NORTH = 6.0, 62.0, 7.0, 63.0
TILE_PX = 60
MODEL_STEP_DEG = 0.25
MODERN_ELEV_M = 200.0
NODATA = -9999.0

#: Model node that is glaciated at the target epoch.  Chosen inside the tile so
#: the resulting NaN patch lands in the tile interior, but small enough that
#: finite cells remain around it — a test that saw *only* NaN could not tell a
#: preserved NaN from a broken read.
ICE_NODE_LAT, ICE_NODE_LON = 62.5, 6.5


def _model_grid() -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(60.5, 64.5 + 1e-9, MODEL_STEP_DEG)
    lons = np.arange(4.5, 8.5 + 1e-9, MODEL_STEP_DEG)
    return lats, lons


def _dz_with_ice_surface_nan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Δz_solid field whose NaN was produced by the real ice-surface guard.

    Returns ``(lats, lons, dz)``.  ``dz`` is a uniform −10 m of subsidence with a
    single-node NaN hole where the icemask says ICE-6G ``Topo`` is the ice
    surface.  −10 m (not 0 m) so that a zero-fill is distinguishable from the
    honest answer.
    """
    lats, lons = _model_grid()
    shape = (lats.size, lons.size)

    topo_0 = np.full(shape, 100.0)
    topo_t = np.full(shape, 90.0)          # 10 m lower at the epoch

    i = int(np.argmin(np.abs(lats - ICE_NODE_LAT)))
    j = int(np.argmin(np.abs(lons - ICE_NODE_LON)))
    ice_t = np.zeros(shape, dtype=bool)
    ice_t[i, j] = True
    ice_0 = np.zeros(shape, dtype=bool)

    dz = solid_displacement(topo_t, topo_0, 0.0, ice_mask_t=ice_t, ice_mask_0=ice_0)

    # Guard the guard: the fixture is only meaningful if the NaN is really there.
    assert not np.isfinite(dz[i, j])
    assert np.isfinite(dz).sum() == dz.size - 1
    return lats, lons, dz


def _tile_bounds() -> rasterio.coords.BoundingBox:
    return rasterio.coords.BoundingBox(TILE_WEST, TILE_SOUTH, TILE_EAST, TILE_NORTH)


def _build_matrix(field: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    return build_correction_matrix(
        field, lats, lons,
        TILE_WEST - 0.5, TILE_SOUTH - 0.5, TILE_EAST + 0.5, TILE_NORTH + 0.5,
        res_deg=1.0 / 60.0,
    )


# ==========================================================================
# 1. read_correction_matrix must not invent a number
# ==========================================================================


class TestReadCorrectionMatrixNaNPolicy:
    def test_ice_surface_nan_reaches_the_tile(self):
        """THE REPRODUCTION: NaN from solid_displacement must survive to dz_tile.

        Before the fix ``read_correction_matrix`` ended with
        ``np.nan_to_num(result, nan=0.0)``, so the ice-surface cells arrived as
        exactly 0.0 — indistinguishable from "GIA is zero here".
        """
        lats, lons, dz = _dz_with_ice_surface_nan()
        matrix, tf = _build_matrix(dz, lats, lons)
        assert not np.isfinite(matrix).all(), "matrix should carry the NaN patch"

        dz_tile = read_correction_matrix(
            matrix, tf, _tile_bounds(), TILE_PX, TILE_PX,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        unknown = ~np.isfinite(dz_tile)
        assert unknown.any(), (
            "ice-surface NaN was destroyed on the way to the tile: "
            f"{int(np.count_nonzero(dz_tile == 0.0))} cells arrived as exactly 0.0"
        )
        # and it must not have been laundered into a plausible-looking zero
        assert not np.any(dz_tile[np.isfinite(dz_tile)] == 0.0)
        # the finite part is still the physical answer
        assert np.allclose(dz_tile[~unknown], -10.0, atol=1e-3)

    def test_preserve_nan_false_refuses_to_zero_fill(self):
        """Opting out of NaN must be a refusal, never a silent zero-fill."""
        lats, lons, dz = _dz_with_ice_surface_nan()
        matrix, tf = _build_matrix(dz, lats, lons)

        with pytest.raises(ValueError, match="(?i)nan"):
            read_correction_matrix(
                matrix, tf, _tile_bounds(), TILE_PX, TILE_PX,
                preserve_nan=False,
            )

    def test_nan_free_read_is_unchanged(self):
        """Regression: on a NaN-free matrix the read is bit-for-bit the old one."""
        lats, lons = _model_grid()
        field = np.linspace(-30.0, 30.0, lats.size * lons.size).reshape(
            lats.size, lons.size
        )
        matrix, tf = _build_matrix(field, lats, lons)

        honest = read_correction_matrix(matrix, tf, _tile_bounds(), TILE_PX, TILE_PX)
        strict = read_correction_matrix(
            matrix, tf, _tile_bounds(), TILE_PX, TILE_PX, preserve_nan=False
        )
        assert np.isfinite(honest).all()
        assert np.array_equal(honest, strict)
        assert honest.dtype == np.float32


# ==========================================================================
# 2. correction_pipeline must not present "unknown" as "no correction"
# ==========================================================================


def _write_fusion_tile(path) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": TILE_PX,
        "height": TILE_PX,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": rasterio.transform.from_bounds(
            TILE_WEST, TILE_SOUTH, TILE_EAST, TILE_NORTH, TILE_PX, TILE_PX
        ),
        "nodata": NODATA,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((TILE_PX, TILE_PX), MODERN_ELEV_M, dtype=np.float32), 1)


class _FakeGiaLoader:
    """Stands in for Ice6gLoader, returning the ice-surface-NaN Δz field.

    Monkeypatched rather than driven off ``ICE6G_fixture.nc`` on purpose: the
    fixture is a 62x61 *regional* cut, and the far-field sea-surface datum is a
    global quantity, so the real loader legitimately refuses it.  This test is
    about the pipeline's NaN policy, not about the datum.
    """

    def __init__(self, _path) -> None:
        self.lats, self.lons, self.dz = _dz_with_ice_surface_nan()

    def get_fields(self, epoch_ka, bounds):
        ice_h = np.zeros_like(self.dz)
        return self.lats, self.lons, self.dz, ice_h

    def close(self) -> None:
        pass


class _FakeIceLoader:
    def __init__(self, _path) -> None:
        self.lats, self.lons = _model_grid()

    def get_thickness(self, *, epoch_ka, bounds, gaussian_sigma=None):
        return self.lats, self.lons, np.zeros((self.lats.size, self.lons.size))

    def close(self) -> None:
        pass


@pytest.fixture
def piped(tmp_path, monkeypatch):
    fusion = tmp_path / "N62E006_fusion.tif"
    _write_fusion_tile(fusion)
    monkeypatch.setattr(cp, "Ice6gLoader", _FakeGiaLoader)
    monkeypatch.setattr(cp, "Ice7gLoader", _FakeIceLoader)

    def _run(**kwargs):
        return cp.run_single_tile_epochs(
            fusion_path=fusion,
            ice6g_path=tmp_path / "ice6g.nc",
            ice7g_path=tmp_path / "ice7g.nc",
            output_dir=tmp_path / "out",
            tile_id="N62E006",
            epochs_ka=[16.5],
            cm_res_deg=1.0 / 60.0,
            verbose=False,
            **kwargs,
        )

    return _run


class TestPipelineUnknownGiaPolicy:
    def test_default_policy_fails_loudly(self, piped):
        """An unknown Δz must stop the tile, not become 0 m of GIA."""
        with pytest.raises(cp.UnknownGiaCorrectionError) as exc:
            piped()
        assert "ice surface" in str(exc.value).lower()

    def test_nodata_policy_marks_unknown_as_nodata_not_zero(self, piped):
        """The documented escape hatch records unknown as nodata, never as 0 m."""
        results = piped(gia_nan_policy="nodata")
        res = results[16.5]

        with rasterio.open(res.paleo_path) as src:
            surface = src.read(1)
            assert src.nodata == NODATA

        unknown = surface == np.float32(NODATA)
        assert unknown.any(), "the ice-surface patch must be marked unknown"
        assert not unknown.all(), "the ice-free part of the tile must survive"

        # The cells that ARE known carry modern − 10 m; none of them is the
        # uncorrected modern elevation, which is what a zero-filled Δz gives.
        known = surface[~unknown]
        assert np.allclose(known, MODERN_ELEV_M - 10.0, atol=1e-2)
        assert not np.any(known == np.float32(MODERN_ELEV_M))

        # And the unknown mask is reported, not just written to disk.
        assert res.gia_unknown_mask.shape == surface.shape
        assert np.array_equal(res.gia_unknown_mask, unknown)

    def test_reported_dz_range_excludes_unknown(self, piped):
        """dz_min/dz_max must stay usable numbers, not NaN."""
        res = piped(gia_nan_policy="nodata")[16.5]
        assert np.isfinite(res.dz_min) and np.isfinite(res.dz_max)
        assert res.dz_min == pytest.approx(-10.0, abs=1e-3)

    def test_unknown_policy_string_is_rejected(self, piped):
        with pytest.raises(ValueError, match="(?i)gia_nan_policy"):
            piped(gia_nan_policy="zero")
