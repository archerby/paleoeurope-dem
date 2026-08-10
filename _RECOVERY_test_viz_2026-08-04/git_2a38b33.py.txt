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
from scipy.ndimage import gaussian_filter
from shapely.geometry import box as shp_box

from paleoeurope.viz.paleocostline_render import (
    RenderConfig,
    SeaLevelCurveResolutionError,
    SeamBlendShiftWarning,
    build_epoch_mosaic,
    filter_paleocostlines_for_epoch,
    get_sea_level_for_epoch,
    interior_tile_seams,
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


# ---------------------------------------------------------------------------
# Two basins at two levels in ONE frame  (basin_seeds / assign_basins)
# ---------------------------------------------------------------------------
#
# Scientific motivation: the hero frame of the Khvalynian project is the Caspian
# at ~+35 m and the Pontic at ~-30 m rendered SIMULTANEOUSLY, with water crossing
# the Manych spillway between them.  A single scalar ``sea_level_override_m``
# cannot express that.  The tests below pin four properties:
#
#   1. basin membership is HYDROLOGICAL (a cell belongs to the basin it drains
#      to), not a bounding box;
#   2. the spill is OBSERVED FROM THE TERRAIN (the two seeded fills merge), not
#      asserted from a curve;
#   3. the corridor between two basins at different levels is a RAMP, not a
#      vertical wall of blue, and it does not fill the receiving basin to the
#      donor's level;
#   4. the single-scalar path is untouched.

from paleoeurope.viz.paleocostline_render import (  # noqa: E402
    BasinSeed,
    CorridorNotRampableError,
    assign_basins,
)

# Synthetic two-bowl world -------------------------------------------------
# Two circular cones rising from -100 m at their apices, capped at +300 m:
#
#     dem = min(-100 + 4*dist_to_A, -100 + 4*dist_to_B, 300)
#
# so the saddle between them sits at the midpoint, where both distances are 35
# cells: -100 + 4*35 = +40.00 m.  That number is ANALYTIC, not measured off a
# picture, which is what makes it usable as an expected value for the sill.
_BASIN_H, _BASIN_W = 60, 160
_BASIN_BOUNDS = (0.0, 0.0, 16.0, 6.0)          # west, south, east, north
_BASIN_CELL_DEG = 0.1
_A_RC, _B_RC = (30, 40), (30, 110)             # bowl apices (row, col)
_SADDLE_M = 40.0                               # -100 + 4 * 35


def _basin_dem() -> np.ndarray:
    r, c = np.indices((_BASIN_H, _BASIN_W))
    d_a = np.hypot(r - _A_RC[0], c - _A_RC[1])
    d_b = np.hypot(r - _B_RC[0], c - _B_RC[1])
    dem = np.minimum(-100.0 + 4.0 * d_a, -100.0 + 4.0 * d_b)
    return np.minimum(dem, 300.0)


def _basin_transform():
    return from_bounds(*_BASIN_BOUNDS, width=_BASIN_W, height=_BASIN_H)


def _rc_to_lonlat(rc: tuple[int, int]) -> tuple[float, float]:
    row, col = rc
    west, south, east, north = _BASIN_BOUNDS
    return (
        west + (col + 0.5) * _BASIN_CELL_DEG,
        north - (row + 0.5) * _BASIN_CELL_DEG,
    )


def _seeds(level_a: float, level_b: float) -> list[BasinSeed]:
    lon_a, lat_a = _rc_to_lonlat(_A_RC)
    lon_b, lat_b = _rc_to_lonlat(_B_RC)
    return [
        BasinSeed(name="west", lon=lon_a, lat=lat_a, level_m=level_a),
        BasinSeed(name="east", lon=lon_b, lat=lat_b, level_m=level_b),
    ]


class TestBasinAssignment:
    """``assign_basins`` -- hydrologically honest membership + observed spill."""

    def test_membership_follows_the_drainage_divide_not_a_bounding_box(self):
        """Every cell must be owned by the bowl it drains into.  In this world
        that divide is the column-75 midline, and it is derivable analytically:
        a cell is closer to A iff it is west of the midline.
        """
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(-40.0, -60.0))

        west_id = field.names.index("west") + 1
        east_id = field.names.index("east") + 1
        assert field.basin_id[_A_RC] == west_id
        assert field.basin_id[_B_RC] == east_id

        # Interior of each bowl, well clear of the divide, is owned outright.
        assert field.basin_id[30, 45] == west_id
        assert field.basin_id[30, 105] == east_id
        # A cell EAST of the divide but at the SAME longitude band a bounding
        # box would hand to the west basin still drains east.
        assert field.basin_id[30, 80] == east_id
        assert field.basin_id[30, 70] == west_id

    def test_sill_elevation_is_measured_from_the_terrain(self):
        """The saddle between the two bowls is analytically +40.00 m; the
        assignment must recover it from the DEM without being told."""
        dem = _basin_dem()
        # Ceiling must clear the saddle, else the flood cannot see it.
        field = assign_basins(dem, _basin_transform(), _seeds(-40.0, -60.0))
        assert field.sill_m[("east", "west")] == pytest.approx(_SADDLE_M, abs=0.6)
        lon, lat = field.sill_lonlat[("east", "west")]
        assert lon == pytest.approx(7.55, abs=0.25)   # column 75
        assert lat == pytest.approx(2.95, abs=0.25)   # row 30

    def test_sill_above_the_flood_ceiling_is_reported_as_unknown_not_guessed(self):
        """If both levels sit so far below the saddle that the flood never
        reaches it, the sill is NaN -- never a plausible-looking number."""
        dem = _basin_dem()
        field = assign_basins(
            dem, _basin_transform(), _seeds(-40.0, -60.0), ceiling_margin_m=10.0
        )
        assert np.isnan(field.sill_m[("east", "west")])
        assert field.spilling is False

    def test_not_spilling_when_both_levels_are_below_the_sill(self):
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(20.0, -20.0))
        assert field.spilling is False
        assert field.connected_groups == (("east",), ("west",))
        assert not field.corridor_mask.any()
        assert field.corridor_mode == "none"

    def test_spilling_is_observed_when_the_donor_level_clears_the_sill(self):
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        assert field.spilling is True
        assert field.connected_groups == (("east", "west"),)

    def test_each_basin_is_thresholded_against_its_own_level(self):
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(20.0, -20.0))
        west_id = field.names.index("west") + 1
        east_id = field.names.index("east") + 1
        assert field.water_level_m[30, 45] == pytest.approx(20.0)
        assert field.water_level_m[30, 105] == pytest.approx(-20.0)
        assert field.water_mask[_A_RC] and field.water_mask[_B_RC]
        # dem == 0 m happens at distance 25 cells from each apex.
        assert field.basin_id[30, 65] == west_id and dem[30, 65] == pytest.approx(0.0)
        assert field.water_mask[30, 65]           # 0 < +20 -> west water
        assert field.basin_id[30, 85] == east_id and dem[30, 85] == pytest.approx(0.0)
        assert not field.water_mask[30, 85]       # 0 > -20 -> east dry

    def test_determinism(self):
        dem = _basin_dem()
        a = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        b = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        np.testing.assert_array_equal(a.basin_id, b.basin_id)
        np.testing.assert_allclose(a.water_level_m, b.water_level_m, equal_nan=True)

    def test_seed_off_the_raster_raises(self):
        dem = _basin_dem()
        with pytest.raises(ValueError, match="outside"):
            assign_basins(
                dem,
                _basin_transform(),
                [BasinSeed("nowhere", lon=99.0, lat=0.0, level_m=0.0)],
            )

    def test_seed_that_is_not_submerged_at_its_own_level_raises(self):
        """A basin seed must be under its own water.  A seed on dry land silently
        produces an empty basin, which is exactly the class of quiet wrongness
        this project refuses."""
        dem = _basin_dem()
        with pytest.raises(ValueError, match="above its own level"):
            assign_basins(dem, _basin_transform(), _seeds(-40.0, -200.0))

    def test_duplicate_basin_names_raise(self):
        dem = _basin_dem()
        lon_a, lat_a = _rc_to_lonlat(_A_RC)
        lon_b, lat_b = _rc_to_lonlat(_B_RC)
        with pytest.raises(ValueError, match="unique"):
            assign_basins(
                dem,
                _basin_transform(),
                [
                    BasinSeed("same", lon_a, lat_a, -40.0),
                    BasinSeed("same", lon_b, lat_b, -60.0),
                ],
            )


class TestSpillCorridor:
    """The water surface between two basins at different levels is a GRADIENT."""

    def test_corridor_is_a_ramp_not_a_step(self):
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        assert field.corridor_mode == "ramp"
        assert field.corridor_mask.any()

        ramp = field.water_level_m[field.corridor_mask]
        # Strictly between the two basin levels, and a genuine spread -- a step
        # would put every corridor cell at one of the two endpoint values.
        assert ramp.min() > -20.0
        assert ramp.max() <= 60.0 + 1e-9
        assert ramp.max() - ramp.min() > 5.0
        # Endpoint-exact at the donor end: the first corridor cells past the
        # sill stand essentially at the donor level.
        assert ramp.max() > 55.0

    def test_the_ramp_does_not_fill_the_receiving_basin_to_the_donor_level(self):
        """The failure mode this exists to prevent: flooding the whole receiving
        basin to the donor's surface because ``dem < donor_level`` is true there.
        """
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        far = (30, 135)                      # 25 cells east of the east apex
        assert dem[far] == pytest.approx(0.0)
        assert dem[far] < 60.0                # a naive donor-level fill WOULD flood it
        assert not field.water_mask[far]      # the ramp does not
        assert field.water_level_m[far] < 0.0

    def test_donor_basin_keeps_its_own_flat_surface(self):
        dem = _basin_dem()
        field = assign_basins(dem, _basin_transform(), _seeds(60.0, -20.0))
        west_id = field.names.index("west") + 1
        west_water = (
            field.water_mask & (field.basin_id == west_id) & (~field.corridor_mask)
        )
        assert west_water.any()
        assert np.allclose(field.water_level_m[west_water], 60.0)

    def test_step_mode_marks_the_corridor_instead_of_pretending(self):
        dem = _basin_dem()
        field = assign_basins(
            dem, _basin_transform(), _seeds(60.0, -20.0), corridor="step"
        )
        assert field.corridor_mode == "step"
        # The corridor is MARKED (so a renderer can hatch it) but no cell was
        # given an invented intermediate surface.
        assert field.corridor_mask.any()
        levels = set(np.unique(field.water_level_m[np.isfinite(field.water_level_m)]))
        assert levels <= {60.0, -20.0}

    def test_more_than_one_spilling_pair_raises_rather_than_guessing(self):
        """Three basins all merged is a corridor topology this ramp does not
        model; it must say so instead of drawing something plausible."""
        dem = _basin_dem()
        lon_a, lat_a = _rc_to_lonlat(_A_RC)
        lon_b, lat_b = _rc_to_lonlat(_B_RC)
        lon_c, lat_c = _rc_to_lonlat((10, 40))
        seeds = [
            BasinSeed("west", lon_a, lat_a, 60.0),
            BasinSeed("east", lon_b, lat_b, -20.0),
            BasinSeed("north", lon_c, lat_c, 55.0),
        ]
        with pytest.raises(CorridorNotRampableError):
            assign_basins(dem, _basin_transform(), seeds)


class TestDemThresholdOceanAcceptsAField:
    """``dem_threshold_ocean`` must take a per-cell water surface, and must be
    bit-identical to the scalar call when that surface is constant."""

    def test_constant_field_equals_the_scalar_call(self):
        rng = np.random.default_rng(3)
        dem = rng.uniform(-120.0, 40.0, (40, 40))
        nodata = np.zeros((40, 40), dtype=bool)
        scalar = dem_threshold_ocean(dem, nodata, -30.0)
        field = dem_threshold_ocean(dem, nodata, np.full((40, 40), -30.0))
        np.testing.assert_array_equal(scalar, field)

    def test_a_varying_field_thresholds_per_cell(self):
        dem = np.full((40, 40), -40.0)
        nodata = np.zeros((40, 40), dtype=bool)
        level = np.full((40, 40), -80.0)
        level[:, 20:] = 0.0
        out = dem_threshold_ocean(dem, nodata, level)
        assert not out[4:-4, :16].any()      # -40 is above the -80 surface
        assert out[4:-4, 24:-4].all()        # -40 is below the 0 m surface


class TestRenderTwoBasinsInOneFrame:
    """End-to-end: one PNG, two basins, two levels, and a machine-readable
    sidecar recording the spill state observed from the terrain."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.csv = tmp_path / "spratt.csv"
        _write_spratt_csv(
            self.csv, [(ka * 1000, float(-ka * 5), 2.0) for ka in range(0, 27)]
        )
        self.tile = tmp_path / "twobowl_fusion.tif"
        _write_geotiff(self.tile, _basin_dem().astype("float32"), bounds=_BASIN_BOUNDS)
        self.outdir = tmp_path / "out"
        self.cfg = RenderConfig(
            max_px=_BASIN_W, dpi=40, fig_size_in=4.0, seam_sigma_px=0.1
        )

    def test_render_writes_png_and_basin_sidecar(self):
        png = render_paleocostline_epoch(
            epoch_ka=16.5,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.tile.parent / "nonexistent.shp",
            output_dir=self.outdir,
            config=self.cfg,
            basin_seeds=_seeds(60.0, -20.0),
        )
        assert png.exists()
        sidecar = png.parent / (png.stem + ".basins.json")
        assert sidecar.exists()
        import json

        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["spilling"] is True
        assert meta["corridor"]["mode"] == "ramp"
        assert {b["name"] for b in meta["basins"]} == {"west", "east"}
        # The renderer assigns basins on its SMOOTHED DEM (gaussian sigma=1.5 px),
        # not on the raw tile, and the saddle here is a tent crest of slope
        # 4 m/cell.  Convolving |x| with a Gaussian lowers the apex by
        # sigma*sqrt(2/pi) per unit of slope: 4 * 1.5 * 0.7979 = 4.79 m.  So the
        # measured sill is PREDICTED near +35.2 m, not the raw +40.0 m.  That is a
        # property of the surface the renderer thresholds on, not an error in the
        # sill measurement, and it is asserted here as a prediction rather than
        # back-fitted to the observed value.
        assert meta["sills"][0]["elevation_m"] == pytest.approx(
            _SADDLE_M - 4.79, abs=1.0
        )

    def test_two_levels_render_differently_from_one(self):
        two = render_paleocostline_epoch(
            epoch_ka=16.5,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.tile.parent / "nonexistent.shp",
            output_dir=self.outdir / "two",
            config=self.cfg,
            basin_seeds=_seeds(20.0, -20.0),
        )
        one = render_paleocostline_epoch(
            epoch_ka=16.5,
            tile_paths={str(self.tile): str(self.tile)},
            sea_level_csv=self.csv,
            paleocoastlines_shp=self.tile.parent / "nonexistent.shp",
            output_dir=self.outdir / "one",
            config=self.cfg,
            sea_level_override_m=20.0,
            force_dem_mask=True,
        )
        assert two.read_bytes() != one.read_bytes()

    def test_basin_seeds_with_a_scalar_override_raises(self):
        with pytest.raises(ValueError, match="sea_level_override_m"):
            render_paleocostline_epoch(
                epoch_ka=16.5,
                tile_paths={str(self.tile): str(self.tile)},
                sea_level_csv=self.csv,
                paleocoastlines_shp=self.tile.parent / "nonexistent.shp",
                output_dir=self.outdir,
                config=self.cfg,
                basin_seeds=_seeds(20.0, -20.0),
                sea_level_override_m=-30.0,
            )


# ---------------------------------------------------------------------------
# The sill the FRAME can resolve vs the sill the TERRAIN has
# ---------------------------------------------------------------------------
#
# Measured on the real Khvalynian raster, Manych point sill 44.119E / 45.773N,
# through build_epoch_mosaic itself:
#
#                                             max_px=2000   max_px=4000
#   native 15 arcsec, untouched ............. +23.00 m       +23.00 m  <- the terrain
#   + bilinear downsample to the mosaic ..... +24.00 m       +25.00 m
#   + the seam blend as it stood 2026-08-01 . +40.76 m       +30.57 m  <- the frame
#
# The last row is gone: that blend ran along every whole degree line whether or
# not a tile boundary was there, and the single-file inputs both bakes pass have
# no seams at all (see TestSeamBlendDoesNotMoveTheManychCrest).  What is left is
# the honest cost of downsampling a spillway narrower than the working pixel,
# +1.00 m and +2.00 m, and the renderer's own dem_sm gaussian on top of it.
#
# The 16.76 m that used to ride along was the dangerous part: at a +35 m
# Khvalynian highstand it flips the frame's spill verdict from SPILLING to
# endorheic -- silently, and in the direction that deletes the subject of the
# animation.  `spilling` is only as honest as the DEM it is observed on, so a
# caller who has measured the sill at native resolution can state it and have the
# frame refuse to disagree -- which is the guard that stays put whether or not any
# one smoothing step is currently behaving.

from paleoeurope.viz.paleocostline_render import SillResolutionError  # noqa: E402


class TestSillResolutionGuard:
    def test_a_sill_the_grid_cannot_resolve_raises(self):
        """Smoothing the tent crest lowers it ~4.8 m (see the sidecar test); an
        expectation pinned to the unsmoothed terrain must then fail loudly."""
        dem = gaussian_filter(_basin_dem(), sigma=1.5)
        with pytest.raises(SillResolutionError, match="resolve"):
            assign_basins(
                dem,
                _basin_transform(),
                _seeds(-40.0, -60.0),
                sill_expectations={("west", "east"): _SADDLE_M},
                sill_tolerance_m=2.0,
            )

    def test_an_expectation_the_frame_agrees_with_is_accepted(self):
        dem = gaussian_filter(_basin_dem(), sigma=1.5)
        field = assign_basins(
            dem,
            _basin_transform(),
            _seeds(-40.0, -60.0),
            sill_expectations={("east", "west"): _SADDLE_M - 4.79},
            sill_tolerance_m=2.0,
        )
        assert np.isfinite(field.sill_m[("east", "west")])

    def test_an_expectation_for_a_sill_above_the_ceiling_raises(self):
        """"I could not reach that saddle" must not read as "it agrees"."""
        dem = _basin_dem()
        with pytest.raises(SillResolutionError, match="ceiling"):
            assign_basins(
                dem,
                _basin_transform(),
                _seeds(-40.0, -60.0),
                ceiling_margin_m=10.0,
                sill_expectations={("west", "east"): _SADDLE_M},
            )

    def test_an_expectation_naming_an_unknown_basin_raises(self):
        dem = _basin_dem()
        with pytest.raises(ValueError, match="not a basin pair"):
            assign_basins(
                dem,
                _basin_transform(),
                _seeds(-40.0, -60.0),
                sill_expectations={("west", "danube"): 10.0},
            )

    def test_the_renderer_passes_the_guard_through(self, tmp_path: Path):
        csv = tmp_path / "spratt.csv"
        _write_spratt_csv(csv, [(ka * 1000, float(-ka * 5), 2.0) for ka in range(0, 27)])
        tile = tmp_path / "twobowl_fusion.tif"
        _write_geotiff(tile, _basin_dem().astype("float32"), bounds=_BASIN_BOUNDS)
        with pytest.raises(SillResolutionError):
            render_paleocostline_epoch(
                epoch_ka=16.5,
                tile_paths={str(tile): str(tile)},
                sea_level_csv=csv,
                paleocoastlines_shp=tmp_path / "nonexistent.shp",
                output_dir=tmp_path / "out",
                config=RenderConfig(
                    max_px=_BASIN_W, dpi=40, fig_size_in=4.0, seam_sigma_px=0.1
                ),
                basin_seeds=_seeds(20.0, -20.0),
                basin_sill_expectations={("west", "east"): _SADDLE_M},
                basin_sill_tolerance_m=1.0,
            )


# ---------------------------------------------------------------------------
# Three basins in a line: connectivity must be a DONOR-level flood
# ---------------------------------------------------------------------------
#
# The hero frame is a THREE-basin configuration -- Caspian, Pontic, Mediterranean
# -- and the pairwise test ``dem < max(level_i, level_j)`` manufactures a
# connection across it.  Flooding to the pairwise max floods every basin in
# BETWEEN to the donor's level as well, so a Caspian at +35 m "reaches" the
# Mediterranean straight through a Pontic basin standing 65 m lower, whatever the
# Bosphorus sill does.  The frame then reports two spilling pairs and
# ``corridor="ramp"`` refuses to draw it at all.
#
# The world below is that configuration reduced to its analytic bones: a channel
# with three bowls at -100 m separated by two crests, +20 m between caspian and
# pontic and +5 m between pontic and mediterranean.  Every number in it is a
# control point of the profile, not something read off a picture.
#
#   caspian  +35 m  clears the +20 m crest  -> spills into pontic
#   pontic   -30 m  is BELOW the +5 m crest -> does not spill into mediterranean
#   caspian  +35 m  is above BOTH crests    -> the old pairwise test connected it
#                                              to the mediterranean as well

import itertools  # noqa: E402

_CHAIN_H, _CHAIN_W = 40, 180
_CHAIN_BOUNDS = (0.0, 0.0, 18.0, 4.0)          # 0.1 deg cells, both axes
_CHAIN_CELL_DEG = 0.1
#                     col    elevation
_CHAIN_PROFILE = ((0, 300.0), (30, -100.0), (60, 20.0), (90, -100.0),
                  (120, 5.0), (150, -100.0), (179, 300.0))
_CHAIN_SEED_COL = {"caspian": 30, "pontic": 90, "mediterranean": 150}
_CHAIN_SEED_ROW = 20
_CHAIN_CREST_CASPIAN_PONTIC_M = 20.0
_CHAIN_CREST_PONTIC_MED_M = 5.0
_CHAIN_LEVELS = {"caspian": 35.0, "pontic": -30.0, "mediterranean": -45.0}


def _chain_dem() -> np.ndarray:
    cols = [c for c, _ in _CHAIN_PROFILE]
    elev = [e for _, e in _CHAIN_PROFILE]
    profile = np.interp(np.arange(_CHAIN_W, dtype=float), cols, elev)
    return np.tile(profile, (_CHAIN_H, 1))


def _chain_transform():
    return from_bounds(*_CHAIN_BOUNDS, width=_CHAIN_W, height=_CHAIN_H)


def _chain_seeds() -> list[BasinSeed]:
    west, _south, _east, north = _CHAIN_BOUNDS
    return [
        BasinSeed(
            name=name,
            lon=west + (_CHAIN_SEED_COL[name] + 0.5) * _CHAIN_CELL_DEG,
            lat=north - (_CHAIN_SEED_ROW + 0.5) * _CHAIN_CELL_DEG,
            level_m=_CHAIN_LEVELS[name],
        )
        for name in ("caspian", "pontic", "mediterranean")
    ]


class TestSpillIsADonorLevelFloodNotAPairwiseMax:
    """Connectivity must never flood an intermediate basin to the donor's level."""

    def test_a_high_donor_does_not_connect_through_an_intermediate_basin(self):
        """The whole three-basin hero frame in one assertion.

        caspian spills into pontic across the +20 m crest, and STOPS there: the
        water it delivers joins the pontic water body at the pontic's own -30 m
        surface, which is 35 m below the +5 m crest to the mediterranean.  Under
        the pairwise-max rule this frame reported TWO spilling pairs and
        ``corridor='ramp'`` raised rather than render.
        """
        field = assign_basins(_chain_dem(), _chain_transform(), _chain_seeds())
        assert field.spilling is True
        assert field.connected_groups == (("caspian", "pontic"), ("mediterranean",))
        assert field.corridor_mode == "ramp"

    def test_the_verdict_does_not_depend_on_the_order_the_basins_are_listed(self):
        seeds = _chain_seeds()
        dem, xform = _chain_dem(), _chain_transform()
        reference = assign_basins(dem, xform, seeds)
        for order in itertools.permutations(range(3)):
            field = assign_basins(dem, xform, [seeds[i] for i in order])
            assert field.spilling == reference.spilling
            assert field.connected_groups == reference.connected_groups
            assert field.sill_m.keys() == reference.sill_m.keys()
            for pair, expected in reference.sill_m.items():
                # NaN == NaN is False, and "this saddle was never reached" is a
                # verdict that has to survive a re-ordering just like a number.
                assert field.sill_m[pair] == pytest.approx(expected, nan_ok=True)

    def test_the_uninvolved_basin_keeps_its_own_water_surface(self):
        """A basin that is neither donor nor receiver must not be repainted by the
        corridor ramp.  The mediterranean stands at -45 m; the ramp interpolates
        between +35 m and -30 m and would put it 15 m too high.
        """
        field = assign_basins(_chain_dem(), _chain_transform(), _chain_seeds())
        med = field.basin_id == (field.names.index("mediterranean") + 1)
        assert med.any()
        assert np.allclose(field.water_level_m[med], _CHAIN_LEVELS["mediterranean"])
        assert not field.corridor_mask[med].any()

    def test_the_two_crests_are_measured_and_the_far_pair_never_touches(self):
        field = assign_basins(_chain_dem(), _chain_transform(), _chain_seeds())
        assert field.sill_m[("caspian", "pontic")] == pytest.approx(
            _CHAIN_CREST_CASPIAN_PONTIC_M, abs=0.5
        )
        assert field.sill_m[("mediterranean", "pontic")] == pytest.approx(
            _CHAIN_CREST_PONTIC_MED_M, abs=0.5
        )
        # The pontic basin lies bodily between them, so these two territories have
        # no common border and there is no saddle to report.
        assert np.isnan(field.sill_m[("caspian", "mediterranean")])

    def test_a_pontic_high_enough_to_clear_its_own_crest_does_connect_onward(self):
        """The rule must not be "never transitive" -- it must be "each hop on its
        own level".  Raise the pontic above the +5 m crest and the three become one
        connected group through TWO genuine corridors (which ramp mode then refuses,
        by design, rather than guessing how the discharge divides).
        """
        seeds = [
            BasinSeed(s.name, s.lon, s.lat, 10.0 if s.name == "pontic" else s.level_m)
            for s in _chain_seeds()
        ]
        field = assign_basins(_chain_dem(), _chain_transform(), seeds, corridor="step")
        assert field.connected_groups == (("caspian", "mediterranean", "pontic"),)


# ---------------------------------------------------------------------------
# The seam blend must not move a sill
# ---------------------------------------------------------------------------
#
# ``build_epoch_mosaic`` used to Gaussian-blur a strip of +/-3 sigma around EVERY
# integer degree line inside the mosaic, whether or not a tile boundary was there.
# On the single-file inputs both bakes actually pass, that is pure interior
# smoothing, and measured on the real working raster
# (``gebco_2024_khvalynian_pontic_...cog.tif``, 2026-08-01) it moved the Manych
# point sill at 44.119 E / 45.773 N by
#
#     max_px=2000 (the Pontic bake's own setting) : +24.00 m -> +40.76 m
#     max_px=4000                                 : +25.00 m -> +30.57 m
#
# and changed 41-75 % of all mosaic cells by more than a centimetre.  A +16.76 m
# rise turns a Caspian spilling over a +23.00 m sill at a +35 m highstand into an
# endorheic lake -- silently, and in the direction that deletes the subject of the
# animation.

_MANYCH_LON, _MANYCH_LAT = 44.119, 45.773
_MANYCH_SILL_M = 23.00
_MANYCH_BOUNDS = (43.0, 45.0, 45.5, 46.5)      # west, south, east, north
_MANYCH_W, _MANYCH_H = 160, 96                 # -> 0.015625 deg/px, both axes
_MANYCH_ROW, _MANYCH_COL = 46, 71              # the cell holding the sill point
_MANYCH_SHOULDER_M = 80.0


def _manych_dem() -> np.ndarray:
    """A narrow E-W channel with an analytic crest at the real Manych point.

    The channel is three cells tall -- the Manych spillway really is narrower than
    a 0.015625 deg working pixel is comfortable with -- and its floor is a tent of
    slope 6 m/px peaking at exactly ``_MANYCH_SILL_M`` in cell
    ``(_MANYCH_ROW, _MANYCH_COL)``, which is the cell containing 44.119 E /
    45.773 N.  Nothing here is measured off a picture.
    """
    dem = np.full((_MANYCH_H, _MANYCH_W), _MANYCH_SHOULDER_M, dtype="float32")
    cols = np.arange(_MANYCH_W)
    floor = np.maximum(_MANYCH_SILL_M - 6.0 * np.abs(cols - _MANYCH_COL), -60.0)
    dem[_MANYCH_ROW - 1 : _MANYCH_ROW + 2, :] = floor.astype("float32")
    return dem


class TestSeamBlendDoesNotMoveTheManychCrest:
    @pytest.fixture()
    def manych_tile(self, tmp_path: Path) -> Path:
        p = tmp_path / "manych_fusion.tif"
        _write_geotiff(p, _manych_dem(), bounds=_MANYCH_BOUNDS)
        return p

    def test_the_sill_point_really_is_the_crest_of_the_raw_raster(self):
        """Guard the fixture itself: the number the mosaic is pinned against has
        to be a property of the raster, not of the test's arithmetic."""
        dem = _manych_dem()
        assert dem[_MANYCH_ROW, _MANYCH_COL] == pytest.approx(_MANYCH_SILL_M)
        channel = dem[_MANYCH_ROW - 1 : _MANYCH_ROW + 2, :]
        assert channel.max() == pytest.approx(_MANYCH_SILL_M)
        # The sill point really is where the fixture says it is.
        xform = from_bounds(*_MANYCH_BOUNDS, width=_MANYCH_W, height=_MANYCH_H)
        col_f, row_f = ~xform * (_MANYCH_LON, _MANYCH_LAT)
        assert (int(row_f), int(col_f)) == (_MANYCH_ROW, _MANYCH_COL)

    def test_a_single_file_mosaic_reproduces_the_manych_crest_exactly(
        self, manych_tile: Path
    ):
        """A single file has no seams, so the mosaic must BE the raster."""
        cfg = RenderConfig(max_px=_MANYCH_W)          # native resolution, no resample
        dem, _ice, _xf, _bnds, _crs, _res = build_epoch_mosaic(
            {"m": str(manych_tile)}, cfg
        )
        assert dem.shape == (_MANYCH_H, _MANYCH_W)
        assert dem[_MANYCH_ROW, _MANYCH_COL] == pytest.approx(_MANYCH_SILL_M, abs=1e-4)
        np.testing.assert_allclose(dem, _manych_dem(), rtol=0, atol=1e-4)

    def test_the_legacy_degree_grid_blend_is_what_moved_the_crest(
        self, manych_tile: Path
    ):
        """The defect, pinned as a measurement rather than described in prose.

        43-45 E and 45-46 N all cross this window, the sill point sits 7.6 px from
        the 44 E line and 14.5 px from the 46 N line, and sigma=5 px reaches 15 px
        -- so the old rule blurred the shoulders straight into the spillway.
        """
        cfg = RenderConfig(max_px=_MANYCH_W, legacy_degree_grid_blend=True)
        with pytest.warns(SeamBlendShiftWarning):
            dem, *_ = build_epoch_mosaic({"m": str(manych_tile)}, cfg)
        risen = float(dem[_MANYCH_ROW, _MANYCH_COL])
        # Far more than the 2 m the project's own SillResolutionError tolerates.
        assert risen > _MANYCH_SILL_M + 5.0

    def test_a_zero_seam_sigma_no_longer_crashes(self, manych_tile: Path):
        """The mitigation a previous step wrote down -- "pass a small
        seam_sigma_px" -- dies with ZeroDivisionError deep inside scipy at the one
        value that actually means "do not blend"."""
        cfg = RenderConfig(max_px=_MANYCH_W, seam_sigma_px=0.0)
        dem, *_ = build_epoch_mosaic({"m": str(manych_tile)}, cfg)
        np.testing.assert_allclose(dem, _manych_dem(), rtol=0, atol=1e-4)

    def test_a_negative_seam_sigma_is_refused_in_words(self, manych_tile: Path):
        cfg = RenderConfig(max_px=_MANYCH_W, seam_sigma_px=-1.0)
        with pytest.raises(ValueError, match="seam_sigma_px"):
            build_epoch_mosaic({"m": str(manych_tile)}, cfg)


_SEAM_COL = 110                                # -> 44.71875 E, 39 px east of the sill
_SEAM_LON = _MANYCH_BOUNDS[0] + _SEAM_COL * 0.015625
_SEAM_STEP_M = 12.0                            # the tile disagreement a blend hides


def _manych_two_tile_raw() -> np.ndarray:
    """The Manych window as two tiles that disagree by ``_SEAM_STEP_M`` at the seam.

    A seam blend exists to hide a step like this.  It is put 39 px east of the
    sill point so that "blend the seam" and "leave the sill alone" are separable
    claims rather than the same claim.
    """
    raw = _manych_dem().copy()
    raw[:, _SEAM_COL:] += np.float32(_SEAM_STEP_M)
    return raw


class TestSeamBlendIsSeamLocalAndAudited:
    @pytest.fixture()
    def abutting_tiles(self, tmp_path: Path) -> list[Path]:
        """Two tiles sharing the 44.71875 E edge; the Manych channel crosses both.

        The shared edge is deliberately NOT a whole degree: a seam is where the
        tiles are.  The old rule could not see one that failed to land on a whole
        degree any more than it could avoid one that was not there at all.
        """
        raw = _manych_two_tile_raw()
        a = tmp_path / "left.tif"
        b = tmp_path / "right.tif"
        _write_geotiff(a, raw[:, :_SEAM_COL], bounds=(43.0, 45.0, _SEAM_LON, 46.5))
        _write_geotiff(b, raw[:, _SEAM_COL:], bounds=(_SEAM_LON, 45.0, 45.5, 46.5))
        return [a, b]

    def test_the_seam_is_found_from_the_tiles_not_from_the_degree_grid(
        self, abutting_tiles: list[Path]
    ):
        from rasterio.coords import BoundingBox

        tiles = [BoundingBox(43.0, 45.0, _SEAM_LON, 46.5),
                 BoundingBox(_SEAM_LON, 45.0, 45.5, 46.5)]
        lons, lats = interior_tile_seams(
            tiles, BoundingBox(43.0, 45.0, 45.5, 46.5), 0.015625
        )
        assert lons == [pytest.approx(_SEAM_LON)]
        assert lats == []          # both tiles span the full latitude range
        # A single tile has no interior edge at all, whatever its extent.
        assert interior_tile_seams(
            [BoundingBox(43.0, 45.0, 45.5, 46.5)],
            BoundingBox(43.0, 45.0, 45.5, 46.5),
            0.015625,
        ) == ([], [])

    def test_a_genuine_seam_is_still_blended_but_only_near_the_seam(
        self, abutting_tiles: list[Path]
    ):
        cfg = RenderConfig(max_px=_MANYCH_W)
        with pytest.warns(SeamBlendShiftWarning):
            dem, *_ = build_epoch_mosaic({str(p): str(p) for p in abutting_tiles}, cfg)
        raw = _manych_two_tile_raw()
        hw = int(np.ceil(cfg.seam_sigma_px * 3))
        # Blended where the tiles meet ...
        assert not np.allclose(
            dem[:, _SEAM_COL - 2 : _SEAM_COL + 2],
            raw[:, _SEAM_COL - 2 : _SEAM_COL + 2],
        )
        # ... and untouched everywhere else, the Manych crest included.
        np.testing.assert_allclose(
            dem[:, : _SEAM_COL - hw], raw[:, : _SEAM_COL - hw], rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(
            dem[:, _SEAM_COL + hw + 1 :],
            raw[:, _SEAM_COL + hw + 1 :],
            rtol=0,
            atol=1e-4,
        )
        assert dem[_MANYCH_ROW, _MANYCH_COL] == pytest.approx(_MANYCH_SILL_M, abs=1e-4)

    def test_the_shift_a_blend_causes_is_reported_with_a_number_and_a_place(
        self, abutting_tiles: list[Path]
    ):
        cfg = RenderConfig(max_px=_MANYCH_W)
        with pytest.warns(SeamBlendShiftWarning) as record:
            build_epoch_mosaic({str(p): str(p) for p in abutting_tiles}, cfg)
        message = str(record[0].message)
        assert " m at " in message and " E / " in message and " N" in message
        assert "seam_blend_max_shift_m" in message

    def test_a_blend_that_stays_inside_its_bound_says_nothing(self, tmp_path: Path):
        flat = np.full((_MANYCH_H, _MANYCH_W), 12.0, dtype="float32")
        a = tmp_path / "flat_left.tif"
        b = tmp_path / "flat_right.tif"
        _write_geotiff(a, flat[:, :_SEAM_COL], bounds=(43.0, 45.0, _SEAM_LON, 46.5))
        _write_geotiff(b, flat[:, _SEAM_COL:], bounds=(_SEAM_LON, 45.0, 45.5, 46.5))
        with warnings.catch_warnings():
            warnings.simplefilter("error", SeamBlendShiftWarning)
            build_epoch_mosaic(
                {str(a): str(a), str(b): str(b)}, RenderConfig(max_px=_MANYCH_W)
            )
