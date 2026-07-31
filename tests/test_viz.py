"""
tests/test_viz.py

Unit and integration tests for paleoeurope.viz.paleocostline_render.

All tests are self-contained: synthetic data is generated in-memory or via
tmp_path — no external downloads required.

Coverage targets (PAPER1_SUBMISSION_PLAN.md C-2: ≥80 %):
  get_sea_level_for_epoch      — 5 tests
  load_paleocostlines          — 3 tests
  filter_paleocostlines_for_epoch — 3 tests
  rasterize_land_mask          — 3 tests
  build_epoch_mosaic           — 4 tests
  render_paleocostline_epoch   — 4 tests
  RenderConfig                 — 2 tests
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box as shp_box

from paleoeurope.viz.paleocostline_render import (
    RenderConfig,
    SeaLevelCurveResolutionError,
    build_epoch_mosaic,
    filter_paleocostlines_for_epoch,
    get_sea_level_for_epoch,
    load_paleocostlines,
    rasterize_land_mask,
    render_paleocostline_epoch,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _write_spratt_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    """Write a minimal Spratt-format CSV (age_bp, sea_level_m, uncertainty_m)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding=utf-8 is explicit so the fixture matches the production CSV's
    # encoding on Windows (where open() would otherwise default to cp1252 and
    # write a 0x97 byte for the em-dash that pandas then fails to decode).
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Spratt & Lisiecki (2016) — test fixture\n")
        for age_bp, sl, unc in rows:
            fh.write(f"{age_bp},{sl},{unc}\n")


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    bounds: tuple[float, float, float, float] = (6.0, 62.0, 7.0, 63.0),
    crs_epsg: int = 4326,
    nodata: float = -9999.0,
) -> None:
    """Write a single-band float32 GeoTIFF to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = data.shape
    transform = from_bounds(*bounds, width=W, height=H)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _make_coast_gdf(
    sea_levels: list[int] | None = None,
    crs_epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """Build a minimal Paleocoastlines-style GeoDataFrame with one land polygon
    per sea-level step covering roughly half the fixture tile (6–7°E, 62–63°N).
    """
    if sea_levels is None:
        sea_levels = [-120, -80, -40, 0]
    geoms, levels = [], []
    # Land polygon: left half of tile
    land_box = shp_box(6.0, 62.0, 6.5, 63.0)
    for sl in sea_levels:
        geoms.append(land_box)
        levels.append(sl)
    return gpd.GeoDataFrame(
        {"geometry": geoms, "Sea level": levels},
        crs=CRS.from_epsg(crs_epsg),
    )


# ---------------------------------------------------------------------------
# RenderConfig
# ---------------------------------------------------------------------------


class TestRenderConfig:
    def test_defaults(self):
        cfg = RenderConfig()
        assert cfg.max_px == 2048
        assert cfg.dpi == 150
        assert cfg.vert_exag == pytest.approx(8.0)
        assert cfg.ice_threshold_m == pytest.approx(10.0)
        assert cfg.seam_sigma_px == pytest.approx(5.0)

    def test_custom_values_accepted(self):
        cfg = RenderConfig(max_px=512, dpi=72, fig_size_in=7.0, vert_exag=4.0)
        assert cfg.max_px == 512
        assert cfg.dpi == 72
        assert cfg.fig_size_in == pytest.approx(7.0)
        assert cfg.vert_exag == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# get_sea_level_for_epoch
# ---------------------------------------------------------------------------


class TestGetSeaLevelForEpoch:
    @pytest.fixture(autouse=True)
    def csv_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "spratt.csv"
        rows = [
            (0, 0.0, 2.0),
            (500, -5.0, 2.0),
            (1000, -10.0, 2.0),
            (5000, -40.0, 3.0),
            (10000, -80.0, 4.0),
            (15000, -110.0, 5.0),
            (21000, -125.0, 5.0),
        ]
        _write_spratt_csv(p, rows)
        self._path = p
        return p

    def test_exact_match(self, csv_path: Path):
        sl = get_sea_level_for_epoch(21.0, csv_path)
        assert sl == pytest.approx(-125.0)

    def test_gap_wider_than_the_published_step_refuses_to_answer(
        self, csv_path: Path
    ):
        """10.2 ka is bracketed by nodes 5 ka apart — that is not a lookup.

        This test used to assert ``-80.0``: the value of the 10 000 BP node,
        returned by nearest-neighbour across a 5000 yr hole.  A curve that
        coarse does not resolve 10.2 ka, and handing back a neighbour's number
        is exactly the silent substitution this module now refuses.
        """
        with pytest.raises(SeaLevelCurveResolutionError, match="10200"):
            get_sea_level_for_epoch(10.2, csv_path)

    def test_present_day(self, csv_path: Path):
        sl = get_sea_level_for_epoch(0.0, csv_path)
        assert sl == pytest.approx(0.0)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Sea-level CSV not found"):
            get_sea_level_for_epoch(10.0, tmp_path / "nonexistent.csv")

    def test_clamping_warning(self, tmp_path: Path):
        """CSV with only one row → any epoch triggers clamping warning."""
        p = tmp_path / "tiny.csv"
        _write_spratt_csv(p, [(0, 0.0, 1.0)])
        with pytest.warns(UserWarning, match="Spratt CSV"):
            sl = get_sea_level_for_epoch(21.0, p)
        assert sl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_sea_level_for_epoch — the half-step tie
#
# The published Spratt & Lisiecki (2016) stack is on a UNIFORM 1 ka grid.  Four
# of the thirteen epochs the Black-Sea bake ships (16.5, 14.5, 9.5, 7.5 ka) land
# exactly halfway between two nodes.  Under the retired nearest-neighbour lookup
# ``(df.age_bp - want).abs().idxmin()`` returned the FIRST row attaining the
# minimum, and the CSV is ascending in age, so every tie broke toward the
# YOUNGER — i.e. the SHALLOWER — node:
#
#     ka    younger node      older node     returned      linear      error
#   16.5    -98.06 @16 ka  -107.30 @17 ka     -98.06    -102.680    +4.620 m
#   14.5    -76.64 @14 ka   -86.57 @15 ka     -76.64     -81.605    +4.965 m
#    9.5    -15.83 @ 9 ka   -24.59 @10 ka     -15.83     -20.210    +4.380 m
#    7.5     -6.11 @ 7 ka    -9.09 @ 8 ka      -6.11      -7.600    +1.490 m
#
# and it did so silently: the miss is exactly 500 yr, which is exactly half a
# step, which is precisely the boundary every guard in the project tests with a
# STRICT inequality.  16.5 and 14.5 ka are the hero frames.
# ---------------------------------------------------------------------------


_PUBLISHED_SPRATT = (
    Path(os.environ.get("PALEO_DATA_ROOT") or _REPO_ROOT.parents[1] / "Paleo-Europe_3D")
    / "data"
    / "raw"
    / "sea_level_curves"
    / "spratt_lisiecki_2016_full.csv"
)

# Published SeaLev_shortPC1 nodes bracketing each shipped half-ka epoch, and the
# value linear interpolation must produce at the midpoint.  Transcribed from
# NOAA/WDS-Paleo study 19982 (Spratt & Lisiecki 2016, Clim. Past 12, 1079-1092);
# ``test_half_ka_expectations_match_the_shipped_published_curve`` proves this
# table against the shipped file rather than trusting the transcription.
_HALF_KA_TRUTH = {
    # epoch_ka: (younger node m, older node m, expected interpolated m)
    16.5: (-98.06, -107.30, -102.680),
    14.5: (-76.64, -86.57, -81.605),
    9.5: (-15.83, -24.59, -20.210),
    7.5: (-6.11, -9.09, -7.600),
}


class TestHalfStepTie:
    @pytest.fixture()
    def uniform_1ka_csv(self, tmp_path: Path) -> Path:
        """A 1-ka grid whose values are strictly decreasing with age."""
        p = tmp_path / "uniform_1ka.csv"
        _write_spratt_csv(p, [(ka * 1000, -10.0 * ka, 1.0) for ka in range(0, 21)])
        return p

    def test_a_half_step_tie_returns_the_midpoint_not_a_neighbour(
        self, uniform_1ka_csv: Path
    ):
        """16.5 ka must not silently become 16 ka.

        Nodes: -160 m at 16 ka, -170 m at 17 ka.  Nearest-neighbour returned
        -160 (the younger, shallower node).  The honest answer is -165.
        """
        sl = get_sea_level_for_epoch(16.5, uniform_1ka_csv)
        assert sl == pytest.approx(-165.0)
        assert sl != pytest.approx(-160.0)  # the younger node it used to pick
        assert sl != pytest.approx(-170.0)  # nor the older one

    def test_the_tie_is_not_merely_broken_the_other_way(
        self, uniform_1ka_csv: Path
    ):
        """A quarter-step must land a quarter of the way, not on a node."""
        assert get_sea_level_for_epoch(16.25, uniform_1ka_csv) == pytest.approx(-162.5)
        assert get_sea_level_for_epoch(16.75, uniform_1ka_csv) == pytest.approx(-167.5)

    def test_every_node_is_returned_bit_exact(self, uniform_1ka_csv: Path):
        """Interpolation must not perturb the published values themselves."""
        for ka in range(0, 21):
            assert get_sea_level_for_epoch(
                float(ka), uniform_1ka_csv
            ) == pytest.approx(-10.0 * ka, abs=0.0, rel=0.0)

    def test_lookup_is_monotone_across_the_tie(self, uniform_1ka_csv: Path):
        """No step discontinuity at the half-step boundary any more."""
        ages = [16.0, 16.1, 16.4, 16.5, 16.6, 16.9, 17.0]
        vals = [get_sea_level_for_epoch(a, uniform_1ka_csv) for a in ages]
        assert all(b < a for a, b in zip(vals, vals[1:])), vals

    @pytest.mark.parametrize("epoch_ka", sorted(_HALF_KA_TRUTH))
    def test_half_ka_expectations_match_the_shipped_published_curve(self, epoch_ka):
        """Prove the transcription in _HALF_KA_TRUTH against the shipped file."""
        if not _PUBLISHED_SPRATT.exists():
            pytest.skip(f"published curve not present: {_PUBLISHED_SPRATT}")
        younger, older, _ = _HALF_KA_TRUTH[epoch_ka]
        lo_ka, hi_ka = int(epoch_ka - 0.5), int(epoch_ka + 0.5)
        assert get_sea_level_for_epoch(
            float(lo_ka), _PUBLISHED_SPRATT
        ) == pytest.approx(younger, abs=1e-9)
        assert get_sea_level_for_epoch(
            float(hi_ka), _PUBLISHED_SPRATT
        ) == pytest.approx(older, abs=1e-9)

    @pytest.mark.parametrize("epoch_ka", sorted(_HALF_KA_TRUTH))
    def test_shipped_half_ka_epochs_no_longer_take_the_shallower_node(self, epoch_ka):
        """The four ties on the real curve, pinned to the interpolated value."""
        if not _PUBLISHED_SPRATT.exists():
            pytest.skip(f"published curve not present: {_PUBLISHED_SPRATT}")
        younger, _older, expected = _HALF_KA_TRUTH[epoch_ka]
        sl = get_sea_level_for_epoch(epoch_ka, _PUBLISHED_SPRATT)
        assert sl == pytest.approx(expected, abs=1e-9)
        assert sl < younger, "the retired lookup returned the shallower node"

    def test_the_thirteen_shipped_epochs_only_move_where_a_tie_existed(self):
        """Regression fence: integer-ka epochs are nodes and must not budge."""
        if not _PUBLISHED_SPRATT.exists():
            pytest.skip(f"published curve not present: {_PUBLISHED_SPRATT}")
        unchanged = {
            22: -125.82, 18: -113.01, 12: -51.05, 11: -35.85, 10: -24.59,
            9: -15.83, 5: 0.00, 2: 4.01, 0: 8.49,
        }
        for ka, want in unchanged.items():
            assert get_sea_level_for_epoch(
                float(ka), _PUBLISHED_SPRATT
            ) == pytest.approx(want, abs=1e-9), ka


# ---------------------------------------------------------------------------
# get_sea_level_for_epoch — refusals (no silent clamp, no silent hole)
# ---------------------------------------------------------------------------


class TestSeaLevelRefusals:
    def test_off_the_end_by_exactly_one_step_still_warns(self, tmp_path: Path):
        """The old blind spot: the warning tested ``> 1000`` yr.

        A 0-20 ka curve asked for 21 ka misses by EXACTLY 1000 yr, so the clamp
        to the 20 ka row was silent.  Any extrapolation off the end warns now.
        """
        p = tmp_path / "trunc.csv"
        _write_spratt_csv(p, [(ka * 1000, -10.0 * ka, 1.0) for ka in range(0, 21)])
        with pytest.warns(UserWarning, match="Spratt CSV"):
            sl = get_sea_level_for_epoch(21.0, p)
        assert sl == pytest.approx(-200.0)  # clamped to the 20 ka endpoint

    def test_inside_the_range_never_warns(self, tmp_path: Path):
        """A genuine bracketed lookup is not noisy."""
        p = tmp_path / "ok.csv"
        _write_spratt_csv(p, [(ka * 1000, -10.0 * ka, 1.0) for ka in range(0, 21)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_sea_level_for_epoch(16.5, p)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]

    def test_a_hole_in_the_curve_raises_rather_than_bridging_it(self, tmp_path: Path):
        """Interpolating across 5 ka of missing curve is an invention."""
        p = tmp_path / "coarse.csv"
        _write_spratt_csv(p, [(0, 0.0, 1.0), (5000, -10.0, 1.0), (25000, -120.0, 1.0)])
        with pytest.raises(SeaLevelCurveResolutionError) as exc:
            get_sea_level_for_epoch(16.5, p)
        assert "5000" in str(exc.value) and "25000" in str(exc.value)

    def test_a_node_inside_a_hole_is_still_answerable(self, tmp_path: Path):
        """The refusal is about interpolating, not about coarse curves per se."""
        p = tmp_path / "coarse.csv"
        _write_spratt_csv(p, [(0, 0.0, 1.0), (5000, -10.0, 1.0), (25000, -120.0, 1.0)])
        assert get_sea_level_for_epoch(5.0, p) == pytest.approx(-10.0)
        assert get_sea_level_for_epoch(25.0, p) == pytest.approx(-120.0)

    def test_the_gap_budget_is_caller_controllable(self, tmp_path: Path):
        """A caller who owns a 5-ka curve can say so explicitly, in writing."""
        p = tmp_path / "coarse.csv"
        _write_spratt_csv(p, [(0, 0.0, 1.0), (5000, -10.0, 1.0)])
        assert get_sea_level_for_epoch(
            2.5, p, max_node_gap_yr=5_000.0
        ) == pytest.approx(-5.0)

    def test_unsorted_and_duplicated_rows_do_not_change_the_answer(self, tmp_path: Path):
        """np.interp is undefined on unsorted x; the loader must not rely on order."""
        p = tmp_path / "shuffled.csv"
        _write_spratt_csv(
            p,
            [(17000, -170.0, 1.0), (15000, -150.0, 1.0),
             (16000, -160.0, 1.0), (16000, -160.0, 1.0)],
        )
        assert get_sea_level_for_epoch(16.5, p) == pytest.approx(-165.0)


# ---------------------------------------------------------------------------
# The OTHER nearest-neighbour lookup in this module, which must stay nearest
# ---------------------------------------------------------------------------


def test_coastline_step_selection_stays_nearest_and_never_interpolates():
    """``filter_paleocostlines_for_epoch`` keys a CATEGORICAL set of polygons.

    Its "Sea level" values are labels of discrete published coastline layers,
    not samples of a continuous function: there is no polygon halfway between
    the -120 m and the -80 m step, and inventing a level of -100 would select
    nothing at all.  It must keep snapping to an EXISTING step.
    """
    gdf = _make_coast_gdf([-120, -80, -40, 0])
    _, step = filter_paleocostlines_for_epoch(gdf, -100.0)  # exact half-step tie
    assert step in (-120, -80), "must return a real published step, not -100"
    assert isinstance(step, int)


# ---------------------------------------------------------------------------
# load_paleocostlines
# ---------------------------------------------------------------------------


class TestLoadPaleocostlines:
    def test_missing_file_returns_none_with_warning(self, tmp_path: Path):
        with pytest.warns(UserWarning, match="not found"):
            result = load_paleocostlines(tmp_path / "ghost.shp")
        assert result is None

    def test_loads_valid_shapefile(self, tmp_path: Path):
        shp = tmp_path / "coast" / "Paleocoastlines.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        gdf = _make_coast_gdf()
        gdf.to_file(shp)
        result = load_paleocostlines(shp)
        assert result is not None
        assert "Sea level" in result.columns
        assert len(result) == 4

    def test_sea_level_column_is_numeric(self, tmp_path: Path):
        shp = tmp_path / "coast2" / "Paleocoastlines.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        gdf = _make_coast_gdf(sea_levels=[-120, 0])
        gdf.to_file(shp)
        result = load_paleocostlines(shp)
        assert pd.api.types.is_numeric_dtype(result["Sea level"])


# ---------------------------------------------------------------------------
# filter_paleocostlines_for_epoch
# ---------------------------------------------------------------------------


class TestFilterPaleocostlinesForEpoch:
    @pytest.fixture(autouse=True)
    def gdf(self) -> gpd.GeoDataFrame:
        self._gdf = _make_coast_gdf(sea_levels=[-120, -80, -40, 0])
        return self._gdf

    def test_exact_level_selected(self):
        land, matched = filter_paleocostlines_for_epoch(self._gdf, -120.0)
        assert matched == -120
        assert len(land) == 1

    def test_nearest_level_selected(self):
        # -95 is equidistant between -80 and -120; should pick whichever min gives
        _, matched = filter_paleocostlines_for_epoch(self._gdf, -95.0)
        assert matched in (-80, -120)

    def test_none_input_returns_none(self):
        land, matched = filter_paleocostlines_for_epoch(None, -120.0)
        assert land is None
        assert matched is None


# ---------------------------------------------------------------------------
# rasterize_land_mask
# ---------------------------------------------------------------------------


class TestRasterizeLandMask:
    """Tests for rasterize_land_mask using in-memory data only."""

    # Fixture tile: 6–7°E, 62–63°N, 64×64 px
    _bounds = rasterio.coords.BoundingBox(left=6.0, bottom=62.0, right=7.0, top=63.0)
    _crs = CRS.from_epsg(4326)
    _H, _W = 64, 64
    _transform = from_bounds(6.0, 62.0, 7.0, 63.0, width=_W, height=_H)

    def test_half_tile_land_mask(self):
        """Land polygon covering left half → roughly 50 % True pixels."""
        gdf = _make_coast_gdf(sea_levels=[-120])
        land = gdf[gdf["Sea level"] == -120].copy()
        mask = rasterize_land_mask(
            land, self._transform, self._H, self._W, self._crs, self._bounds
        )
        assert mask is not None
        assert mask.dtype == bool
        land_frac = mask.sum() / mask.size
        assert 0.35 < land_frac < 0.65, f"Expected ~50 % land, got {land_frac:.2%}"

    def test_none_gdf_returns_none(self):
        mask = rasterize_land_mask(
            None, self._transform, self._H, self._W, self._crs, self._bounds
        )
        assert mask is None

    def test_nonintersecting_polygon_returns_none(self):
        """Polygon entirely outside the scene extent → None."""
        far = gpd.GeoDataFrame(
            {"geometry": [shp_box(50.0, 0.0, 51.0, 1.0)], "Sea level": [-120]},
            crs=self._crs,
        )
        mask = rasterize_land_mask(
            far, self._transform, self._H, self._W, self._crs, self._bounds
        )
        assert mask is None


# ---------------------------------------------------------------------------
# build_epoch_mosaic
# ---------------------------------------------------------------------------


class TestBuildEpochMosaic:
    """Tests using synthetic single-tile mosaics (no real data)."""

    @pytest.fixture()
    def single_tile(self, tmp_path: Path) -> Path:
        """One 64×64 tile at 62–63°N, 6–7°E with mixed land/ocean."""
        rng = np.random.default_rng(0)
        data = rng.uniform(-50.0, 500.0, (64, 64)).astype("float32")
        p = tmp_path / "N62E006_fusion.tif"
        _write_geotiff(p, data)
        return p

    @pytest.fixture()
    def two_tiles(self, tmp_path: Path) -> list[Path]:
        """Two adjacent 64×64 tiles."""
        rng = np.random.default_rng(1)
        paths = []
        for col, (lon0, lon1) in enumerate([(6.0, 7.0), (7.0, 8.0)]):
            data = rng.uniform(-50.0, 200.0, (64, 64)).astype("float32")
            p = tmp_path / f"tile{col}.tif"
            _write_geotiff(p, data, bounds=(lon0, 62.0, lon1, 63.0))
            paths.append(p)
        return paths

    def test_single_tile_returns_valid_mosaic(self, single_tile: Path):
        cfg = RenderConfig(max_px=64)
        dem, ice, xform, bounds, crs, res = build_epoch_mosaic(
            {str(single_tile): str(single_tile)}, cfg
        )
        assert dem.ndim == 2
        assert dem.shape[0] <= 64
        assert np.isfinite(dem).any()

    def test_ice_array_same_shape_as_dem(self, single_tile: Path):
        cfg = RenderConfig(max_px=64)
        dem, ice, *_ = build_epoch_mosaic({str(single_tile): str(single_tile)}, cfg)
        # ice may be None (no ice file) or an ndarray of matching shape
        if ice is not None:
            assert ice.shape == dem.shape

    def test_two_tile_mosaic_wider(self, two_tiles: list[Path], single_tile: Path):
        cfg = RenderConfig(max_px=256)
        dem_single, _, _, bounds_1, _, _ = build_epoch_mosaic(
            {str(single_tile): str(single_tile)}, cfg
        )
        dem_two, _, _, bounds_2, _, _ = build_epoch_mosaic(
            {str(p): str(p) for p in two_tiles}, cfg
        )
        # Two-tile mosaic covers ~twice the longitudinal extent
        width_1 = bounds_1.right - bounds_1.left
        width_2 = bounds_2.right - bounds_2.left
        assert width_2 > width_1 * 1.5

    def test_max_px_limits_output(self, single_tile: Path):
        cfg = RenderConfig(max_px=32)
        dem, *_ = build_epoch_mosaic({str(single_tile): str(single_tile)}, cfg)
        assert max(dem.shape) <= 32


# ---------------------------------------------------------------------------
# render_paleocostline_epoch  (end-to-end, synthetic only)
# ---------------------------------------------------------------------------


class TestRenderPaleocostlineEpoch:
    """Full render pipeline using synthetic tiles + synthetic coastlines."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.tmp = tmp_path

        # Sea-level CSV
        self.csv = tmp_path / "spratt.csv"
        rows = [(ka * 1000, float(-ka * 5), 2.0) for ka in range(0, 27)]
        _write_spratt_csv(self.csv, rows)

        # One 32×32 synthetic tile
        rng = np.random.default_rng(42)
        data = rng.uniform(-30.0, 300.0, (32, 32)).astype("float32")
        self.tile = tmp_path / "N62E006_fusion.tif"
        _write_geotiff(self.tile, data)

        # Paleocoastlines shapefile
        shp = tmp_path / "coast" / "Paleocoastlines.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        gdf = _make_coast_gdf([-120, -80, -40, 0])
        gdf.to_file(shp)
        self.shp = shp

        self.outdir = tmp_path / "out"
        self.cfg = RenderConfig(max_px=32, dpi=30, fig_size_in=3.0)

    def test_output_file_created(self):
        png = render_paleocostline_epoch(
            epoch_ka=21.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir,
            config=self.cfg,
        )
        assert png.exists()
        assert png.suffix == ".png"

    def test_output_filename_contains_epoch(self):
        png = render_paleocostline_epoch(
            epoch_ka=12.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir,
            config=self.cfg,
        )
        assert "12" in png.name

    def test_missing_shapefile_still_renders(self, tmp_path: Path):
        """Render falls back to DEM zero-contour when shapefile is absent."""
        png = render_paleocostline_epoch(
            epoch_ka=0.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=tmp_path / "nonexistent.shp",
            output_dir=self.outdir,
            config=self.cfg,
        )
        assert png.exists()

    def test_preloaded_coast_gdf_used(self):
        """Caller may pass a pre-loaded GDF to skip shapefile I/O."""
        preloaded = _make_coast_gdf([-120, 0])
        png = render_paleocostline_epoch(
            epoch_ka=21.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir,
            config=self.cfg,
            preloaded_coast_gdf=preloaded,
        )
        assert png.exists()


# ---------------------------------------------------------------------------
# Per-basin sea-level override  (sea_level_override_m)
# ---------------------------------------------------------------------------


class TestSeaLevelOverride:
    """The keyword-only ``sea_level_override_m`` must replace the global Spratt
    value for the coastline-step selection while leaving the default path
    (override=None) byte-for-byte identical to today's behaviour.

    Scientific motivation: inside a sill-dammed basin (the Black Sea was an
    isolated lake until ~9 ka) the global eustatic curve is the WRONG sea level
    — the basin must be driven by its own cited level curve.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.tmp = tmp_path

        # Spratt CSV: 21 ka → -125 m.  We then override with -40 m so the
        # selected coastline step is unambiguously different.
        self.csv = tmp_path / "spratt.csv"
        rows = [(ka * 1000, float(-ka * 6), 2.0) for ka in range(0, 27)]
        _write_spratt_csv(self.csv, rows)  # 21 ka → -126 m

        rng = np.random.default_rng(7)
        data = rng.uniform(-30.0, 300.0, (32, 32)).astype("float32")
        self.tile = tmp_path / "N62E006_fusion.tif"
        _write_geotiff(self.tile, data)

        # Coastline steps far apart so nearest-step picks are unambiguous.
        shp = tmp_path / "coast" / "Paleocoastlines.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        _make_coast_gdf([-120, -80, -40, 0]).to_file(shp)
        self.shp = shp

        self.outdir = tmp_path / "out"
        self.cfg = RenderConfig(max_px=32, dpi=30, fig_size_in=3.0)

    def test_override_selects_step_nearest_override_not_spratt(self):
        """Unit-level: the engine must feed the OVERRIDE (not the Spratt value)
        into filter_paleocostlines_for_epoch.

        Spratt(21 ka) = -126 m  → nearest step -120.
        override      = -40 m   → nearest step  -40.
        These differ, proving the override is what drives selection.
        """
        gdf = _make_coast_gdf([-120, -80, -40, 0])

        spratt_sl = get_sea_level_for_epoch(21.0, self.csv)
        _, step_default = filter_paleocostlines_for_epoch(gdf, spratt_sl)
        _, step_override = filter_paleocostlines_for_epoch(gdf, -40.0)

        assert step_default == -120
        assert step_override == -40
        assert step_default != step_override

    def test_render_with_override_differs_from_default(self):
        """End-to-end: a render WITH override=-40 must encode a different
        coastline step than the default (Spratt) render — visible in the PNG
        bytes — while both still produce a valid file.
        """
        png_default = render_paleocostline_epoch(
            epoch_ka=21.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir / "default",
            config=self.cfg,
        )
        png_override = render_paleocostline_epoch(
            epoch_ka=21.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir / "override",
            config=self.cfg,
            sea_level_override_m=-40.0,
        )
        assert png_default.exists()
        assert png_override.exists()
        # Different coastline step → different rasterized land mask → different
        # rendered pixels.
        assert png_default.read_bytes() != png_override.read_bytes()

    def test_override_none_is_backward_compatible(self):
        """override=None (default) must select the SAME step as a render that
        never passed the kwarg — i.e. the global Spratt path is unchanged.
        """
        spratt_sl = get_sea_level_for_epoch(21.0, self.csv)
        gdf = _make_coast_gdf([-120, -80, -40, 0])
        _, step_spratt = filter_paleocostlines_for_epoch(gdf, spratt_sl)

        png = render_paleocostline_epoch(
            epoch_ka=21.0,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.shp,
            output_dir=self.outdir,
            config=self.cfg,
            sea_level_override_m=None,
        )
        assert png.exists()
        assert step_spratt == -120  # unchanged default selection


from paleoeurope.viz.paleocostline_render import ocean_depth_from_surface


def test_ocean_depth_is_measured_below_the_epoch_surface():
    # Sea floor at -2000 m; lake surface (override) at -80 m.
    dem = np.array([[-2000.0, -50.0, 10.0]])
    ocean = np.array([[True, True, False]])
    depth = ocean_depth_from_surface(dem, ocean, water_level=-80.0)
    # Deep pixel: below the -80 surface: -80 - (-2000) = 1920.
    assert depth[0, 0] == pytest.approx(1920.0)
    # Pixel at -50 m is ABOVE the -80 m surface -> not really submerged -> depth 0.
    assert depth[0, 1] == pytest.approx(0.0)
    # Non-ocean pixel -> 0.
    assert depth[0, 2] == pytest.approx(0.0)


from paleoeurope.viz.paleocostline_render import dem_threshold_ocean


def test_dem_threshold_ocean_uses_the_water_level_not_zero():
    dem = np.full((20, 20), -40.0)   # everywhere at -40 m
    nodata = np.zeros((20, 20), dtype=bool)
    # At level 0 m: -40 < 0 -> ocean. Check the INTERIOR (binary_closing/opening
    # with iterations=2 erodes a ~3-px border on an all-True field via the default
    # border_value=0 — that de-speckle border behavior is inherited from the code
    # this helper extracts and is harmless for an interior basin on a large frame).
    assert dem_threshold_ocean(dem, nodata, 0.0)[4:-4, 4:-4].all()
    # At level -80 m (basin lowstand): -40 is NOT below -80 -> no ocean anywhere.
    assert not dem_threshold_ocean(dem, nodata, -80.0).any()
