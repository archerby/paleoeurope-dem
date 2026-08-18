"""Guards for the Khvalynian scene table — which working raster IS which scene.

Two working rasters now exist and the second is not a replacement:

* ``…n54.0_s36.0_w25.5_e58.0_cog.tif`` — the frame the shipped 43-epoch set was
  baked on;
* ``…n62.0_s36.0_w25.5_e60.0_cog.tif`` — the drainage_implied frame, extended to
  62 N and to 60 E, which is the eastern edge of the GEBCO tile this project
  holds and therefore not a design choice.

``bake_khvalynian`` keeps the scene bbox in CODE rather than reading it off
whichever file is configured, so that a swapped raster is caught instead of
silently redefining the scene.  These tests pin that, and — the part that
matters — plant the failures the two guards claim to catch and prove each
fires.  Two of them are failures the module could NOT catch before this table
existed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.bake_khvalynian as bk  # noqa: E402


NARROW = "gebco_2024_khvalynian_pontic_n54.0_s36.0_w25.5_e58.0_cog.tif"
WIDE = "gebco_2024_khvalynian_pontic_n62.0_s36.0_w25.5_e60.0_cog.tif"


class TestTheSceneTable:
    def test_both_working_rasters_are_named_and_neither_was_dropped(self):
        assert set(bk.KHV_SCENES) == {NARROW, WIDE}, (
            "a raster was added or removed without this test being updated; the "
            "table is the only record of which frame a shipped product belongs to")

    def test_each_entry_agrees_with_the_bbox_spelled_in_its_own_filename(self):
        """The filename carries the bbox and so does the table.  Two spellings of
        one fact drift; this is what stops them."""
        import re
        for name, (lat_min, lat_max, lon_min, lon_max) in bk.KHV_SCENES.items():
            m = re.search(r"n([\d.]+)_s([\d.]+)_w([\d.]+)_e([\d.]+)", name)
            assert m, "filename %r does not spell its bbox" % name
            n, s, w, e = (float(g) for g in m.groups())
            assert (s, n, w, e) == (lat_min, lat_max, lon_min, lon_max), (
                "table says %r for %s but the filename says N=%g S=%g W=%g E=%g"
                % ((lat_min, lat_max, lon_min, lon_max), name, n, s, w, e))

    def test_the_narrow_scene_is_the_historical_box(self):
        assert bk.KHV_SCENES[NARROW] == (36.0, 54.0, 25.5, 58.0)

    def test_the_wide_scene_stops_at_the_gebco_tiles_own_eastern_edge(self):
        """60 E is not a preference. Widening it needs a different source, and a
        test that let it drift would hide that."""
        assert bk.KHV_SCENES[WIDE][3] == 60.0

    def test_an_unknown_raster_is_refused_by_name(self):
        """PLANTED FAILURE. Before the table existed, pointing
        KHVALYNIAN_DEM_PATH at any file simply used the historical bbox — a wider
        tile's margin would then be baked with 0 m of GIA, which
        apply_gia_delta's fill_value=0.0 produces silently and which is
        indistinguishable from a measured zero over stable far-field ground."""
        with pytest.raises(SystemExit) as exc:
            bk._scene_for(Path("/some/where/a_raster_nobody_declared.tif"))
        msg = str(exc.value)
        assert "REFUSING an unknown working raster" in msg
        assert NARROW in msg and WIDE in msg, "the refusal must list what IS known"

    def test_the_refusal_looks_at_the_name_not_the_directory(self):
        for parent in ("/mnt/tank/paleo/data", "Y:/paleo/data", "."):
            assert bk._scene_for(Path(parent) / NARROW) == (36.0, 54.0, 25.5, 58.0)


def _raster(path: Path, w, s, e, n, res=1.0 / 240.0):
    width = int(round((e - w) / res))
    height = int(round((n - s) / res))
    profile = dict(driver="GTiff", dtype="int16", nodata=-32767.0, count=1,
                   width=width, height=height, crs="EPSG:4326",
                   transform=from_bounds(w, s, e, n, width, height))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.zeros((height, width), dtype="int16"), 1)
    return path


class TestTheDeclaredSceneGuard:
    """``_require_tile_is_the_declared_scene`` is the half the comment above
    KHV_SCENES always promised and the code did not have.

    ``_require_model_grid_brackets_tile`` catches a tile reaching OUTSIDE the GIA
    window.  A tile SMALLER than the declared box sails through it — a smaller
    tile is bracketed more comfortably — and every downstream consumer computes
    its rows from LAT_MIN..LAT_MAX, so a short raster yields a frame that is
    internally consistent, correctly tagged, and about the wrong ground.
    """

    def test_the_declared_scene_passes(self, tmp_path):
        p = _raster(tmp_path / NARROW, 25.5, 36.0, 58.0, 54.0)
        with rasterio.open(p) as src:
            bk._require_tile_is_the_declared_scene(src)  # must not raise

    def test_the_other_declared_scene_passes_against_ITS_OWN_box(self, tmp_path):
        """The bbox comes from the table entry, not from the module-level
        LAT_MIN/LON_MIN. Baking the wide scene must not be checked against the
        narrow one's box just because that is what happens to be imported."""
        p = _raster(tmp_path / WIDE, 25.5, 36.0, 60.0, 62.0)
        with rasterio.open(p) as src:
            bk._require_tile_is_the_declared_scene(src)

    @pytest.mark.parametrize(
        "bounds, edges",
        [((25.5, 36.0, 58.0, 51.0), "N"),      # SHORT in the north — the silent one
         ((25.5, 36.0, 60.0, 54.0), "E"),      # wider east than declared
         ((25.5, 30.0, 58.0, 54.0), "S"),      # extends further south
         ((20.0, 36.0, 58.0, 54.0), "W")],     # extends further west
    )
    def test_a_raster_swapped_under_a_declared_name_is_refused(
            self, tmp_path, bounds, edges):
        """PLANTED FAILURES, one per edge.

        The first case is the one that motivated the guard: a raster three
        degrees SHORT in the north raises nothing from the brackets check, and
        would have produced a complete, correctly-tagged frame of the wrong
        ground."""
        p = _raster(tmp_path / NARROW, *bounds)
        with rasterio.open(p) as src:
            with pytest.raises(ValueError) as exc:
                bk._require_tile_is_the_declared_scene(src)
        msg = str(exc.value)
        assert "not the scene KHV_SCENES declares" in msg
        assert edges in msg, "the refusal must name which edge is off: %s" % msg

    def test_an_UNDECLARED_raster_is_passed_through_and_that_is_the_scope(
            self, tmp_path):
        """The other half of declaring a scope.

        ``build_paleo_dem`` is legitimately called with small synthetic tiles by
        the unit tests. Demanding that a 43-45 E / 45-47 N fixture be the whole
        Caspian scene is a category error, not a guard — the first version of
        this check did exactly that and turned 8 passing tests into errors. The
        claim being checked is "a file was swapped under a DECLARED name", and a
        fixture makes no such claim.
        """
        p = _raster(tmp_path / "a_synthetic_fixture.tif", 43.0, 45.0, 45.0, 47.0)
        with rasterio.open(p) as src:
            bk._require_tile_is_the_declared_scene(src)  # must not raise

    def test_the_brackets_guard_alone_would_NOT_have_caught_the_short_raster(
            self, tmp_path):
        """The reason this guard had to be added rather than relied upon.

        Hands the SAME short raster to ``_require_model_grid_brackets_tile``
        against a model window covering the full declared scene, and shows it
        passes — so before this change a three-degree-short tile was baked in
        silence."""
        p = _raster(tmp_path / "short.tif", 25.5, 36.0, 58.0, 51.0)
        model_lats = np.arange(35.0, 55.01, 1.0 / 6.0)
        model_lons = np.arange(24.5, 59.01, 1.0 / 6.0)
        with rasterio.open(p) as src:
            tile_lats, tile_lons = bk._tile_coordinates(src)
            bk._require_model_grid_brackets_tile(
                model_lats, model_lons, tile_lats, tile_lons)  # passes: no raise

    def test_half_a_pixel_of_slop_is_tolerated_and_two_pixels_is_not(self, tmp_path):
        """The tolerance is a stated quantity, not a feeling: the bbox is whole
        degrees on a 15-arcsec grid, so anything the same scene can legitimately
        differ by is far below half a pixel."""
        res = 1.0 / 240.0
        near = tmp_path / "near"
        near.mkdir()
        _raster(near / NARROW, 25.5, 36.0, 58.0, 54.0 + res * 0.4)
        with rasterio.open(near / NARROW) as src:
            bk._require_tile_is_the_declared_scene(src)
        far = tmp_path / "far"
        far.mkdir()
        _raster(far / NARROW, 25.5, 36.0, 58.0, 54.0 + res * 2.0)
        with rasterio.open(far / NARROW) as src:
            with pytest.raises(ValueError):
                bk._require_tile_is_the_declared_scene(src)
