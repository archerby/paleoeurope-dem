"""Tests for ``scripts/export_basin_fields.py``.

WHAT IS CHECKED HERE AND WHAT IS NOT
------------------------------------
The partition itself is ``assign_basins``' and is not re-tested here — this
module's job is the four things wrapped around it:

* the CROP, decision D-A: that it takes whole source rows, refuses a bound it
  would have to round, and reports how much more latitude the chosen bound
  discards than the measured minimum required;
* the GUARDS: an exact partition, a nodata sentinel that marks exactly the
  unowned cells, an inherited sentinel that is refused when it could be mistaken
  for a surface, and a seed registry that must lie inside the window;
* the TAGS: that every raster written carries the provenance set, that the code
  map describes the array beside it, and that **no tag template in the module
  carries a typed measurement** — the defect that has already forced six
  retractions on this project;
* the REFUSALS: that a content problem produces a named error and no raster,
  never a substituted value.

The two dataset-backed acceptance tests are marked ``integration`` and read the
artifacts of a real run; they skip, with the reason, when that run has not
happened on this machine.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paleoeurope.viz.paleocostline_render import (  # noqa: E402
    BasinSeed,
    UnmeasuredSill,
    assign_basins,
)
from scripts import export_basin_fields as ebf  # noqa: E402
from scripts.animate_khvalynian_spill import (  # noqa: E402
    SEED_BOXES,
    SEED_RULE,
    BasinLevel,
    ResolvedSeed,
    SeedBox,
)

# ---------------------------------------------------------------------------
# Fixtures: a tiny two-basin world we can flood in milliseconds
# ---------------------------------------------------------------------------

PIXEL_DEG = 0.5


def _write_dem(path: Path, array: np.ndarray, nodata: float = -9999.0) -> Path:
    """A GeoTIFF with the project's own tag vocabulary, small enough to be fast."""
    height, width = array.shape
    transform = from_origin(0.0, float(height) * PIXEL_DEG, PIXEL_DEG, PIXEL_DEG)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as out:
        out.write(array.astype("float32"), 1)
        out.update_tags(
            vertical_frame="F0 test frame",
            epoch_ka="1",
            vertical_datum_contract="test",
            gia_unknown_ice6g_cells="7",
            dem_source="synthetic",
        )
    return path


def _two_basin_dem() -> np.ndarray:
    """Two bowls separated by a ridge, plus an isolated pit and an edge body.

    The pit is below both declared levels but connected to neither seed: it is
    the CLOSED state.  The strip on the array edge is the UNOBSERVABLE state.
    Together they make the three-state distinction visible in a 24x24 array.
    """
    dem = np.full((24, 24), 50.0)   # dry ground, below the flood ceiling: OWNED
    dem[0:2, :] = 5000.0            # above the ceiling: UNOWNED, so nodata is exercised
    dem[4:10, 2:10] = -100.0        # west bowl
    dem[4:10, 14:22] = -100.0       # east bowl
    dem[14:17, 4:8] = -50.0         # a closed pit, below both levels, reaching no seed
    dem[20:24, 0:3] = -80.0         # touches the array edge -> unobservable
    return dem


def _seed(basin: str, box_name: str, lon: float, lat: float, level: float,
          row: int, col: int, elevation: float) -> ResolvedSeed:
    box = SeedBox(
        basin=basin,
        name=box_name,
        lon_min=lon - PIXEL_DEG,
        lat_min=lat - PIXEL_DEG,
        lon_max=lon + PIXEL_DEG,
        lat_max=lat + PIXEL_DEG,
        why="a synthetic box for the write path, not a geographic claim",
    )
    return ResolvedSeed(
        basin=basin, box=box, epoch_ka=1.0, raster_path="synthetic",
        raster_sha256="0" * 64, row=row, col=col, lon=lon, lat=lat,
        elevation_m=elevation, level_m=level, rule=SEED_RULE,
    )


@pytest.fixture
def flooded(tmp_path: Path):
    """A real ``BasinField`` over the synthetic world, plus everything the tags need."""
    dem = _two_basin_dem()
    height, width = dem.shape
    transform = from_origin(0.0, float(height) * PIXEL_DEG, PIXEL_DEG, PIXEL_DEG)

    def lonlat(row: int, col: int) -> tuple[float, float]:
        lon, lat = transform * (col + 0.5, row + 0.5)
        return float(lon), float(lat)

    west_lonlat = lonlat(6, 5)
    east_lonlat = lonlat(6, 18)
    seeds = [
        BasinSeed(name="west", lon=west_lonlat[0], lat=west_lonlat[1], level_m=-10.0),
        BasinSeed(name="east", lon=east_lonlat[0], lat=east_lonlat[1], level_m=-20.0),
    ]
    expectations = {
        ("west", "east"): UnmeasuredSill(
            reason="a synthetic pair; no crest was measured for this fixture"
        )
    }
    field = assign_basins(
        dem, transform, seeds,
        corridor=ebf.CORRIDOR_MODE,
        sill_expectations=expectations,
        dem_frame="F0 test frame",
        unobservable_policy="mark",
    )
    levels = {
        "west": BasinLevel("west", -10.0, "F0 test frame", "fixture", "fixture level"),
        "east": BasinLevel("east", -20.0, "F0 test frame", "fixture", "fixture level"),
    }
    resolved = [
        _seed("west", "west_box", *west_lonlat, -10.0, 6, 5, -100.0),
        _seed("east", "east_box", *east_lonlat, -20.0, 6, 18, -100.0),
    ]
    return {
        "dem": dem, "transform": transform, "field": field, "levels": levels,
        "seeds": resolved, "expectations": expectations, "tmp_path": tmp_path,
    }


def _survey_stub(height: int = 4320, width: int = 7800) -> dict:
    pixel = 18.0 / height
    return {
        "grid": {
            "height": height, "width": width,
            "top_lat_deg": 54.0, "bottom_lat_deg": 36.0,
            "west_lon_deg": 25.5, "east_lon_deg": 58.0,
            "pixel_deg": pixel, "pixel_arcsec": pixel * 3600.0,
            "crs": "EPSG:4326",
        },
        "minimum_clean_north_lat_deg": 54.0 - 220 * pixel,
        "minimum_clean_north_rule": "stub",
        "per_epoch": {"1": {"nodata_cells_raster": 0}},
    }


def _tags_for(flooded, product: str) -> dict:
    field = flooded["field"]
    partition = ebf._partition_counts(field, int(flooded["dem"].size))
    water = ebf._water_counts(field)
    survey = _survey_stub()
    crop = ebf.crop_record(survey)
    bounds = (0.0, 0.0, 12.0, 12.0)
    from rasterio.windows import Window

    return ebf.build_tags(
        product=product,
        epoch_ka=1.0,
        source_path=Path("synthetic.tif"),
        source_sha256="0" * 64,
        source_tags={"gia_unknown_ice6g_cells": "7", "vertical_datum_contract": "t"},
        dem_frame="F0 test frame",
        window=Window(0, 0, 24, 24),
        window_bounds=bounds,
        pixel_deg=PIXEL_DEG,
        survey=survey,
        crop=crop,
        seed_check={"rule": "stub rule"},
        field=field,
        levels=flooded["levels"],
        seeds=flooded["seeds"],
        expectations=flooded["expectations"],
        partition=partition,
        water=water,
        nodata_in_crop=0,
        field_nodata=-9999.0,
        tools=ebf.toolchain(),
        code=ebf.code_provenance(),
        seconds=1.0,
    )


# ---------------------------------------------------------------------------
# D-A: the crop
# ---------------------------------------------------------------------------


def test_crop_window_takes_whole_source_rows(tmp_path: Path):
    """The window starts on a cell boundary and keeps every column."""
    path = _write_dem(tmp_path / "grid.tif", np.zeros((24, 24)))
    with rasterio.open(path) as dataset:
        window = ebf.crop_window(dataset, north_lat=9.0)
    # Top is 12.0 deg, cells are 0.5 deg, so 9.0 is exactly six rows down.
    assert (window.col_off, window.row_off) == (0, 6)
    assert (window.width, window.height) == (24, 18)


def test_crop_window_refuses_a_bound_it_would_have_to_round(tmp_path: Path):
    """A moved bound is a different scene, so it is refused rather than snapped."""
    path = _write_dem(tmp_path / "grid.tif", np.zeros((24, 24)))
    with rasterio.open(path) as dataset:
        with pytest.raises(ebf.CropBoundError) as excinfo:
            ebf.crop_window(dataset, north_lat=9.1)
    message = str(excinfo.value)
    assert "off this raster's cell grid" in message
    assert "NOT rounded" in message
    assert "9.000000" in message  # it names the nearest boundary it would have moved to


def test_crop_window_refuses_a_bound_outside_the_raster(tmp_path: Path):
    """A bound above the top edge crops nothing while naming a crop."""
    path = _write_dem(tmp_path / "grid.tif", np.zeros((24, 24)))
    with rasterio.open(path) as dataset:
        for bound in (12.0, 20.0, 0.0, -5.0):
            with pytest.raises(ebf.CropBoundError):
                ebf.crop_window(dataset, north_lat=bound)


def test_the_production_bound_is_on_the_production_cell_grid():
    """51 N is exactly 720 rows below the scene's top edge, so no rounding happens.

    Checked arithmetically rather than on the raster so it runs without the
    dataset, and again on the raster in the integration test below.
    """
    pixel = 18.0 / 4320
    rows = (54.0 - ebf.CROP_NORTH_LAT_DEG) / pixel
    assert abs(rows - round(rows)) * pixel < ebf.CROP_SNAP_TOL_DEG


def test_crop_record_states_how_much_more_the_choice_discards():
    """The decision and the measurement stay apart, and the gap is a number."""
    survey = _survey_stub()
    record = ebf.crop_record(survey, north_lat=ebf.CROP_NORTH_LAT_DEG)

    assert record["chosen_north_lat_deg"] == ebf.CROP_NORTH_LAT_DEG
    assert record["measured_minimum_clean_north_lat_deg"] == pytest.approx(
        survey["minimum_clean_north_lat_deg"]
    )
    # The gap is DERIVED from the stub's own measurement and the chosen bound, not
    # pinned: it was 500 rows at the 51.0 bound of 2026-08-05 and is 20 at the 53.0
    # bound of 2026-08-18, and a test that spells the number out asserts the crop
    # rather than the behaviour. What must hold at any bound is that the record
    # states the gap, that it is consistent with the two latitudes it sits between,
    # and that the kept and dropped rows add up to the source.
    pixel = 18.0 / 4320
    expected_extra = round((survey["minimum_clean_north_lat_deg"]
                            - ebf.CROP_NORTH_LAT_DEG) / pixel)
    assert record["extra_rows_discarded"] == expected_extra
    assert record["extra_latitude_discarded_deg"] == pytest.approx(
        expected_extra * pixel)
    assert record["extra_area_discarded_km2"] > 0
    assert record["rows_dropped_of_source"] + record["rows_kept_of_source"] == 4320
    assert record["rows_dropped_of_source"] == round(
        (54.0 - ebf.CROP_NORTH_LAT_DEG) / pixel)


def test_crop_record_reports_zero_extra_when_the_choice_is_the_measurement():
    survey = _survey_stub()
    measured = survey["minimum_clean_north_lat_deg"]
    record = ebf.crop_record(survey, north_lat=measured)
    assert record["extra_rows_discarded"] == 0
    assert record["extra_latitude_discarded_deg"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The survey the decision rests on
# ---------------------------------------------------------------------------


def test_survey_finds_the_southernmost_nodata_row_across_epochs(tmp_path: Path):
    """The minimum clean bound is set by the WORST epoch, not the last one read."""
    topo = tmp_path / "topo"
    topo.mkdir()
    shallow = np.zeros((24, 24))
    shallow[1, 3] = -9999.0                 # nodata in row 1
    deep = np.zeros((24, 24))
    deep[5, 7] = -9999.0                    # nodata in row 5 -> this one decides
    _write_dem(topo / "khv_1ka.tif", shallow)
    _write_dem(topo / "khv_2ka.tif", deep)

    survey = ebf.survey_nodata([1.0, 2.0], topo, strip_rows=8)

    assert survey["southernmost_nodata_row"] == 5
    assert survey["southernmost_nodata_row_epoch_ka"] == 2.0
    assert survey["total_nodata_cells_raster"] == 2
    assert survey["epochs_with_nodata"] == [1.0, 2.0]
    # Top 12.0, cell 0.5: the south edge of row 5 is 12.0 - 6*0.5 = 9.0
    assert survey["minimum_clean_north_lat_deg"] == pytest.approx(9.0)
    assert survey["per_epoch"]["2"]["last_nodata_row"] == 5
    assert survey["per_epoch"]["1"]["nodata_cells_raster"] == 1


def test_survey_copies_the_source_unit_tag_through_without_reconciling_it(
    tmp_path: Path,
):
    """Two units, both named, neither converted into the other."""
    topo = tmp_path / "topo"
    topo.mkdir()
    _write_dem(topo / "khv_1ka.tif", np.zeros((24, 24)))
    survey = ebf.survey_nodata([1.0], topo, strip_rows=8)

    entry = survey["per_epoch"]["1"]
    assert entry["source_tag_gia_unknown_ice6g_cells_as_read"] == "7"
    assert entry["nodata_cells_raster"] == 0
    assert "RASTER CELLS" in survey["unit_note"]
    assert "does not reconcile" in survey["unit_note"]


def test_survey_with_no_nodata_anywhere_asks_for_no_crop_at_all(tmp_path: Path):
    """"Minimum northern bound" means the LEAST that must be cropped.

    With no nodata anywhere that is nothing, so the bound is the raster's own
    northern edge.  Returning the southern edge here would report the maximum
    possible crop as the minimum required one, and every raster written would
    then carry a crop_extra figure with the wrong sign.
    """
    topo = tmp_path / "topo"
    topo.mkdir()
    _write_dem(topo / "khv_1ka.tif", np.zeros((24, 24)))
    survey = ebf.survey_nodata([1.0], topo, strip_rows=8)
    assert survey["southernmost_nodata_row"] == -1
    assert survey["minimum_clean_north_lat_deg"] == pytest.approx(12.0)
    assert "NORTHERN edge" in survey["minimum_clean_north_rule"]

    record = ebf.crop_record(survey, north_lat=9.0)
    assert record["extra_rows_discarded"] == 6
    assert record["chosen_is_north_of_the_clean_bound"] is False


def test_a_bound_north_of_the_clean_one_is_flagged_not_moved(tmp_path: Path):
    """The crop is never widened to fit; the contradiction is stated instead."""
    survey = _survey_stub()
    record = ebf.crop_record(survey, north_lat=54.0)
    assert record["chosen_is_north_of_the_clean_bound"] is True
    assert record["extra_latitude_discarded_deg"] < 0
    assert record["extra_area_discarded_km2"] >= 0
    assert "every epoch that holds one will refuse" in record["extra_note"]
    assert record["chosen_north_lat_deg"] == 54.0


def test_survey_refuses_a_tree_whose_rasters_are_not_one_grid(tmp_path: Path):
    topo = tmp_path / "topo"
    topo.mkdir()
    _write_dem(topo / "khv_1ka.tif", np.zeros((24, 24)))
    _write_dem(topo / "khv_2ka.tif", np.zeros((20, 24)))
    with pytest.raises(ebf.CropBoundError, match="does not describe another"):
        ebf.survey_nodata([1.0, 2.0], topo, strip_rows=8)


# ---------------------------------------------------------------------------
# The seed registry against the window
# ---------------------------------------------------------------------------


def test_every_registered_seed_box_lies_inside_the_chosen_crop():
    """The three production boxes survive D-A; if one did not, the crop would move it."""
    check = ebf.assert_seed_boxes_inside(
        (25.5, 36.0, 58.0, ebf.CROP_NORTH_LAT_DEG), SEED_BOXES
    )
    assert set(check["boxes_checked"]) == {b.name for b in SEED_BOXES}
    assert max(b.lat_max for b in SEED_BOXES) < ebf.CROP_NORTH_LAT_DEG


def test_a_seed_box_outside_the_window_is_refused():
    with pytest.raises(ebf.SeedBoxOutsideCropError) as excinfo:
        ebf.assert_seed_boxes_inside((25.5, 36.0, 58.0, 40.0), SEED_BOXES)
    message = str(excinfo.value)
    assert "pontic_abyssal_plain" in message
    assert "ground the cropped raster does not carry" in message


# ---------------------------------------------------------------------------
# The nodata sentinel
# ---------------------------------------------------------------------------


def test_field_nodata_is_inherited_from_the_source():
    assert ebf.field_nodata_for(-9999.0, [-110.0, -18.0, 29.9]) == -9999.0


def test_field_nodata_refuses_a_source_that_declares_none():
    with pytest.raises(ebf.CropBoundError, match="no finite nodata value"):
        ebf.field_nodata_for(None, [-110.0, 0.0])


def test_field_nodata_refuses_a_sentinel_that_could_pass_for_a_surface():
    """A sentinel inside the data's own range makes absence look like a measurement."""
    with pytest.raises(ebf.CropBoundError, match="cannot be told from a real"):
        ebf.field_nodata_for(-100.0, [-110.0, -95.0, -18.0])


# ---------------------------------------------------------------------------
# The partition guard
# ---------------------------------------------------------------------------


def test_the_partition_guard_accepts_a_real_field(flooded):
    partition = ebf._partition_counts(flooded["field"], int(flooded["dem"].size))
    assert partition["verdict"] == "EXACT_PARTITION"
    assert sum(partition["per_basin"].values()) + partition["unowned"] == 24 * 24
    assert set(partition["per_basin"]) == {"west", "east"}


def test_the_partition_guard_refuses_a_hole(flooded):
    """A cell carrying a label no basin claims is caught, not counted away."""
    field = flooded["field"]
    tampered = field.basin_id.copy()
    tampered[0, 0] = 9
    broken = type(field)(**{**field.__dict__, "basin_id": tampered})
    with pytest.raises(ebf.PartitionError, match="outside the declared code range"):
        ebf._partition_counts(broken, int(flooded["dem"].size))


def test_the_partition_guard_refuses_a_window_size_that_is_not_the_array(flooded):
    with pytest.raises(ebf.PartitionError, match="do not partition the window"):
        ebf._partition_counts(flooded["field"], int(flooded["dem"].size) + 1)


# ---------------------------------------------------------------------------
# The three states
# ---------------------------------------------------------------------------


def test_state_bits_are_distinct_powers_of_two():
    values = [value for _, value, _ in ebf.STATE_BITS]
    assert values == sorted(values)
    assert all(value and not (value & (value - 1)) for value in values)
    assert len(set(values)) == len(values)
    assert all(meaning.strip() for _, _, meaning in ebf.STATE_BITS)


def test_the_flat_test_and_the_connected_mask_disagree_on_this_fixture(flooded):
    """The fixture really does exercise the state the third raster exists for.

    If the flat test and the connected mask agreed here, the round-trip below
    would prove nothing about why ``water_state`` is written at all.
    """
    field = flooded["field"]
    flat = flooded["dem"] < field.water_level_m
    assert int(flat.sum()) > int(field.water_mask.sum())
    assert field.connectivity["flat_mask_cells"] > field.connectivity[
        "connected_mask_cells"
    ]


# ---------------------------------------------------------------------------
# The tags
# ---------------------------------------------------------------------------

#: A digit run counts as a TYPED NUMBER only when it stands alone -- not when it
#: is part of a name.  ``sha256``, ``ice6g``, ``km2``, ``ICE-6G_C`` are
#: identifiers and mean the same thing forever; ``+203.71 m`` is a measurement
#: and can be falsified by the next run.  This is the distinction the rule is
#: about, so it is the distinction the pattern draws.
STANDALONE_NUMBER = re.compile(r"(?<![A-Za-z0-9_\-])\d+(?![A-Za-z0-9_])")

#: The only standalone digit runs a tag TEMPLATE in this module may contain: the
#: unowned code, and the year the three decisions were taken.  Every other figure
#: must be interpolated from a quantity computed for the artifact being written.
ALLOWED_TEMPLATE_NUMBERS = {"0", "2026"}

#: The functions whose string literals become GeoTIFF tags.
TAG_BUILDING_FUNCTIONS = ("build_tags", "_basin_tag_block", "_sill_tag_block")


def test_no_tag_template_in_this_module_carries_a_typed_measurement():
    """The rule that six retractions were paid for, enforced on the source.

    A number typed into a sentence survives the change that falsifies it.  So the
    tag builders may contain no standalone digit run except the unowned code and
    the decision year; every other figure must arrive by interpolation.
    """
    tree = ast.parse(Path(ebf.__file__).read_text(encoding="utf-8"))
    offenders: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in TAG_BUILDING_FUNCTIONS:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Constant) or not isinstance(inner.value, str):
                continue
            for run in STANDALONE_NUMBER.findall(inner.value):
                if run not in ALLOWED_TEMPLATE_NUMBERS:
                    offenders.append((node.name, run, inner.value[:120]))
    assert not offenders, (
        "these tag templates carry a typed number instead of an interpolated "
        f"one: {offenders}"
    )


def test_the_typed_measurement_scanner_would_catch_one():
    """The guard above is only worth having if it fires; here is the proof."""
    assert STANDALONE_NUMBER.findall("over bed as high as +203.71 m") == ["203", "71"]
    assert STANDALONE_NUMBER.findall("gia_unknown_ice6g_cells sha256 km2") == []
    assert STANDALONE_NUMBER.findall("ICE-6G_C VM5a") == []


def test_the_tag_set_names_the_code_map_and_it_describes_the_array(flooded):
    tags = _tags_for(flooded, "basin_id")
    field = flooded["field"]

    mapping = dict(item.split("=", 1) for item in tags["basin_code_map"].split(";"))
    assert mapping["0"] == "unowned"
    for index, name in enumerate(field.names, start=1):
        assert mapping[str(index)] == name
        assert int((field.basin_id == index).sum()) > 0
    assert set(mapping) == {"0"} | {str(i + 1) for i in range(len(field.names))}


def test_every_basin_carries_its_level_its_rule_its_frame_and_its_seed(flooded):
    tags = _tags_for(flooded, "basin_id")
    for name in flooded["field"].names:
        assert tags[f"basin_level_m_{name}"]
        assert tags[f"basin_level_rule_{name}"]
        assert tags[f"basin_level_frame_{name}"]
        assert tags[f"basin_level_provenance_{name}"]
        assert tags[f"basin_seed_{name}"]
        assert tags[f"basin_seed_box_why_{name}"]
    levels = dict(item.split("=", 1) for item in tags["basin_levels_m"].split(";"))
    assert set(levels) == set(flooded["field"].names)


def test_an_unmeasured_sill_says_so_and_carries_no_number(flooded):
    tags = _tags_for(flooded, "basin_id")
    assert tags["sill_status_east__west"].startswith("UNMEASURED: ")
    pair = dict(item.split("=", 1) for item in tags["sill_east__west"].split(";"))
    assert pair["expected_m"] == "none"
    assert pair["gap_m"] == "none"


def test_a_sill_shift_that_was_not_measured_is_not_written_as_zero(flooded):
    """Zero is a measurement.  'Not measured' is a different statement."""
    tags = _tags_for(flooded, "basin_id")
    value = tags["sill_smoothing_shift_m_east__west"]
    assert value.startswith("NOT MEASURED:")
    assert "Zero is not used as a stand-in" in value


def test_the_crop_gap_is_on_every_raster(flooded):
    for product in ("basin_id", "water_level_m", "water_state_bits"):
        tags = _tags_for(flooded, product)
        assert float(tags["crop_extra_latitude_discarded_deg"]) > 0
        # Derived, not pinned -- see the note in
        # test_crop_record_states_how_much_more_the_choice_discards.
        assert int(tags["crop_extra_rows_discarded"]) == round(
            (float(tags["crop_measured_minimum_clean_north_lat_deg"])
             - float(tags["crop_north_lat_deg"])) / (18.0 / 4320))
        assert tags["crop_north_decision"].startswith("OWNER DECISION D-A")
        assert tags["crop_measured_minimum_rule"]


def test_the_nodata_declaration_differs_per_product_and_names_its_unit(flooded):
    ids = _tags_for(flooded, "basin_id")
    levels = _tags_for(flooded, "water_level_m")
    states = _tags_for(flooded, "water_state_bits")

    assert ids["nodata_value"] == "none"
    assert states["nodata_value"] == "none"
    assert float(levels["nodata_value"]) == -9999.0
    assert "declares NO nodata value" in ids["nodata_declaration"]
    assert "RASTER CELLS" in ids["nodata_unit_note"]
    # The two counts that meant different things in two places now say so.
    note = ids["source_tag_gia_unknown_ice6g_cells_note"]
    assert "ICE-6G model grid" in note
    assert "RASTER CELLS" in note
    assert "Two different units" in note
    assert "neither converted into the other" in note
    assert ids["source_tag_gia_unknown_ice6g_cells_as_read"] == "7"


def test_both_the_requested_and_the_as_run_corridor_mode_are_recorded(flooded):
    """The fixture does not spill, so the two differ -- which is the point.

    Recording only the request would claim a corridor was stepped over where the
    terrain has none; recording only the result would lose the fact that the ramp
    was never on the table.
    """
    tags = _tags_for(flooded, "water_level_m")
    assert tags["corridor_mode_requested"] == "step"
    assert tags["corridor_mode_as_run"] == "none"
    assert not flooded["field"].spilling
    assert "D-C" in tags["corridor_mode_decision"]
    assert "D-F" in tags["corridor_mode_decision"]
    assert "TWO VALUES" in tags["corridor_mode_note"]


def test_the_tags_name_the_code_that_produced_them(flooded):
    tags = _tags_for(flooded, "basin_id")
    for key in (
        "code_export_script_sha256",
        "code_renderer_sha256",
        "code_registry_sha256",
    ):
        assert len(tags[key]) == 64
    assert "assign_basins" in tags["code_path"]
    assert "python=" in tags["toolchain"]


def test_the_dem_array_tag_says_it_is_the_raw_source_cells(flooded):
    """The one place this export deviates from the renderer, stated on every file."""
    tags = _tags_for(flooded, "basin_id")
    note = tags["dem_array_assigned"]
    assert "RAW cell values" in note
    assert "no smoothing" in note
    assert "NOT the renderer's smoothed mask array" in note


# ---------------------------------------------------------------------------
# The write path, end to end on the fixture
# ---------------------------------------------------------------------------


def test_the_three_rasters_round_trip(flooded, tmp_path: Path):
    """Write all three products, read them back, and check they describe each other."""
    field = flooded["field"]
    transform = flooded["transform"]
    partition = ebf._partition_counts(field, int(flooded["dem"].size))

    basin_path = tmp_path / "basin_id_1ka.tif"
    ebf._write_raster(
        basin_path, field.basin_id.astype("uint8"), dtype="uint8", nodata=None,
        transform=transform, crs="EPSG:4326", tags=_tags_for(flooded, "basin_id"),
    )
    surface = np.where(
        np.isfinite(field.water_level_m), field.water_level_m, -9999.0
    ).astype("float32")
    level_path = tmp_path / "water_level_1ka.tif"
    ebf._write_raster(
        level_path, surface, dtype="float32", nodata=-9999.0, transform=transform,
        crs="EPSG:4326", tags=_tags_for(flooded, "water_level_m"), predictor=3,
    )
    state = np.zeros(field.basin_id.shape, dtype="uint8")
    masks = {
        "water": field.water_mask,
        "unobservable": field.unobservable_mask,
        "corridor": field.corridor_mask,
    }
    for name, value, _ in ebf.STATE_BITS:
        state[masks[name]] |= np.uint8(value)
    state_path = tmp_path / "water_state_1ka.tif"
    ebf._write_raster(
        state_path, state, dtype="uint8", nodata=None, transform=transform,
        crs="EPSG:4326", tags=_tags_for(flooded, "water_state_bits"),
    )

    with rasterio.open(basin_path) as ds:
        read_ids = ds.read(1)
        assert ds.nodata is None
        assert ds.profile["compress"] == "deflate"
        assert ds.profile["tiled"] is True
        assert len(ds.tags()) > 40
    with rasterio.open(level_path) as ds:
        read_levels = ds.read(1)
        assert ds.nodata == -9999.0
    with rasterio.open(state_path) as ds:
        read_state = ds.read(1)

    assert np.array_equal(read_ids, field.basin_id.astype("uint8"))
    # The nodata cells of the surface are EXACTLY the unowned cells.
    assert int((read_levels == -9999.0).sum()) == partition["unowned"]
    assert np.array_equal(read_levels == -9999.0, read_ids == 0)
    # Every owned cell carries its own basin's level.
    for index, name in enumerate(field.names, start=1):
        owned = read_ids == index
        assert np.allclose(read_levels[owned], field.levels_m[index - 1])
    # The three states are recoverable and the water bit is not the flat test.
    water_bit = (read_state & 1).astype(bool)
    assert np.array_equal(water_bit, field.water_mask)
    assert np.array_equal((read_state & 2).astype(bool), field.unobservable_mask)
    assert np.array_equal((read_state & 4).astype(bool), field.corridor_mask)
    assert int(water_bit.sum()) < int((flooded["dem"] < read_levels).sum())


def test_a_written_raster_carries_the_source_fingerprint_and_the_window(flooded,
                                                                       tmp_path: Path):
    path = tmp_path / "basin_id_1ka.tif"
    ebf._write_raster(
        path, flooded["field"].basin_id.astype("uint8"), dtype="uint8", nodata=None,
        transform=flooded["transform"], crs="EPSG:4326",
        tags=_tags_for(flooded, "basin_id"),
    )
    with rasterio.open(path) as ds:
        tags = ds.tags()
    assert len(tags["source_raster_sha256"]) == 64
    assert "col_off=" in tags["crop_window"] and "row_off=" in tags["crop_window"]
    assert tags["crop_rule"] == ebf.CROP_RULE
    assert tags["partition_total_cells"] == str(24 * 24)
    assert tags["spilling"] in ("true", "false")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_missing_raster_refuses_the_epoch_and_writes_nothing(tmp_path: Path):
    """A content problem comes back as a named refusal, never as a substitute."""
    out = tmp_path / "out"
    result = ebf.export_epoch(
        99.0,
        topo_dir=tmp_path / "empty",
        out_root=out,
        survey=_survey_stub(),
        crop=ebf.crop_record(_survey_stub()),
    )
    assert not result.written
    assert "MissingEpochRasterError" in result.refusal
    assert result.record["epoch_ka"] == 99.0
    assert not out.exists()


def test_a_raster_with_no_vertical_frame_tag_refuses_the_epoch(tmp_path: Path):
    topo = tmp_path / "topo"
    topo.mkdir()
    path = topo / "khv_1ka.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=24, width=24, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 12.0, PIXEL_DEG, PIXEL_DEG), nodata=-9999.0,
    ) as out:
        out.write(np.zeros((24, 24), dtype="float32"), 1)

    result = ebf.export_epoch(
        1.0, topo_dir=topo, out_root=tmp_path / "out",
        survey=_survey_stub(), crop=ebf.crop_record(_survey_stub()),
    )
    assert not result.written
    assert "vertical_frame" in result.refusal


def test_the_manifest_keeps_a_refused_epoch_in_its_slot(tmp_path: Path):
    """A refusal is content, not a gap: the epoch stays in the sequence."""
    manifest = ebf.export_all(
        [99.0], topo_dir=tmp_path / "empty", out_root=tmp_path / "out",
        survey=_survey_stub(),
    )
    assert manifest["summary"]["epochs_written"] == 0
    assert manifest["summary"]["epochs_refused"] == 1
    assert manifest["epochs"][0]["epoch_ka"] == 99.0
    assert manifest["summary"]["refusals"][0]["epoch_ka"] == 99.0
    assert (tmp_path / "out" / "export_manifest.json").exists()


def test_a_survey_from_another_tree_is_not_reused(tmp_path: Path):
    """Reuse saves many minutes; reusing the wrong one puts a stale bound on every file."""
    survey = _survey_stub()
    survey["topo_dir"] = str(tmp_path / "other")
    (tmp_path / "other").mkdir()
    path = tmp_path / "nodata_survey.json"
    path.write_text(json.dumps(survey), encoding="utf-8")

    mine = tmp_path / "mine"
    mine.mkdir()
    with pytest.raises(ebf.StaleSurveyError, match="two scenes"):
        ebf.load_survey(path, [1.0], mine)


def test_a_survey_missing_an_epoch_is_not_reused(tmp_path: Path):
    survey = _survey_stub()
    survey["topo_dir"] = str(tmp_path)
    path = tmp_path / "nodata_survey.json"
    path.write_text(json.dumps(survey), encoding="utf-8")
    with pytest.raises(ebf.StaleSurveyError, match="does not include"):
        ebf.load_survey(path, [1.0, 2.0], tmp_path)


def test_a_survey_that_is_not_there_is_not_invented(tmp_path: Path):
    with pytest.raises(ebf.StaleSurveyError, match="nothing is assumed"):
        ebf.load_survey(tmp_path / "absent.json", [1.0], tmp_path)


def test_free_space_is_checked_before_a_run_starts(tmp_path: Path):
    with pytest.raises(OSError, match="under the"):
        ebf.assert_free_space(tmp_path, needed=1 << 62)


def test_json_writer_refuses_to_stringify_an_unknown_object(tmp_path: Path):
    with pytest.raises(TypeError, match="not silently"):
        ebf._write_json(tmp_path / "x.json", {"bad": object()})
    assert ebf._json_default(np.int64(5)) == 5
    assert ebf._json_default(np.float64(np.nan)) is None


# ---------------------------------------------------------------------------
# Acceptance against the real run
# ---------------------------------------------------------------------------


def _run_records():
    """Every epoch record the production campaign left, across all its manifests.

    The campaign is allowed to have run in parts — it did: a long-lived process
    died mid-grid on a memory-starved machine and the remainder was re-run one
    process per epoch, each leaving its own manifest.  These tests therefore read
    the SET of manifests rather than assuming a single file, because a test that
    only passes when the run happened in one go would silently stop checking
    anything the first time it did not.
    """
    manifests = sorted(ebf.OUT_ROOT.glob("export_manifest*.json"))
    if not manifests:
        pytest.skip(
            f"no export_manifest*.json under {ebf.OUT_ROOT}; the production "
            "export has not run on this machine"
        )
    records: dict[float, dict] = {}
    crop = None
    survey = None
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        crop = payload.get("crop", crop)
        survey = payload.get("nodata_survey", survey)
        for record in payload.get("epochs", []):
            records[float(record["epoch_ka"])] = record
    return records, crop, survey


@pytest.mark.integration
def test_the_measured_minimum_and_the_chosen_bound_are_both_recorded():
    _records, crop, _survey = _run_records()
    assert crop["chosen_north_lat_deg"] == ebf.CROP_NORTH_LAT_DEG
    assert crop["measured_minimum_clean_north_lat_deg"] > crop["chosen_north_lat_deg"]
    assert crop["extra_rows_discarded"] > 0
    assert crop["chosen_is_north_of_the_clean_bound"] is False


@pytest.mark.integration
def test_the_cropped_window_holds_no_nodata_at_any_epoch():
    """The assertion decision D-A rests on, read off the run's own artifacts."""
    records, crop, survey = _run_records()
    assert survey["minimum_clean_north_lat_deg"] >= crop["chosen_north_lat_deg"]
    for record in records.values():
        if record.get("refusal"):
            continue
        assert record["source_nodata_cells_in_crop_window"] == 0


@pytest.mark.integration
def test_every_epoch_of_the_bake_grid_has_a_record():
    """A refusal keeps its slot; an epoch with no record at all would be a gap."""
    records, _crop, _survey = _run_records()
    from scripts import bake_khvalynian as bk

    missing = [e for e in bk.EPOCHS_KA if float(e) not in records]
    assert not missing, f"no record for {missing}"


@pytest.mark.integration
def test_every_written_epoch_partitions_its_window_exactly():
    records, _crop, _survey = _run_records()
    written = [r for r in records.values() if not r.get("refusal")]
    assert written, "the production run wrote no epoch"
    for record in written:
        partition = record["partition"]
        assert partition["verdict"] == "EXACT_PARTITION"
        assert (
            sum(partition["per_basin"].values()) + partition["unowned"]
            == partition["total"]
        )
        assert record["crop_window"]["row_off"] > 0
        assert record["crop_window"]["height"] + record["crop_window"]["row_off"] == 4320


@pytest.mark.integration
def test_every_refusal_is_a_named_error_and_wrote_no_raster():
    """The refusals are the product's honest gaps, not silent omissions."""
    records, _crop, _survey = _run_records()
    refused = {e: r for e, r in records.items() if r.get("refusal")}
    assert refused, "no epoch refused; this test is only meaningful when some did"
    for epoch, record in refused.items():
        assert ":" in record["refusal"], record["refusal"]
        assert "traceback" in record
        assert not (
            ebf.OUT_ROOT / "basin_id" / f"basin_id_{ebf.epoch_tag(epoch)}ka.tif"
        ).exists()


@pytest.mark.integration
def test_the_three_products_cover_the_same_epochs():
    """A missing member of a triple would break every downstream water sheet."""
    records, _crop, _survey = _run_records()
    written = {ebf.epoch_tag(e) for e, r in records.items() if not r.get("refusal")}
    for product, stem in (
        ("basin_id", "basin_id"),
        ("water_level", "water_level"),
        ("water_state", "water_state"),
    ):
        on_disk = {
            p.name[len(stem) + 1: -len("ka.tif")]
            for p in (ebf.OUT_ROOT / product).glob("*.tif")
        }
        assert on_disk == written, f"{product}: {on_disk ^ written}"
