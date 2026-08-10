"""Tests for ``scripts/animate_khvalynian_spill.py``.

Things this sequence must never do, one test class each, plus the seed registry
and the manifest:

* derive a colour limit, a level, a crest or a seed per frame without saying so;
* let the ENCODER re-derive the colour mapping the renderer just pinned -- the
  2026-08-03 deliverable shipped 16 local colour tables in 17 image descriptors,
  and the checks here PARSE THE WRITTEN BYTES rather than trust the writer;
* write a frame delay it did not verify (``imageio`` 2.37.2 takes MILLISECONDS and
  turns a seconds-shaped value into a silent zero);
* let two frames come out different pixel sizes;
* drop or interpolate a refused epoch;
* fingerprint a raster and then point at a path that cannot be opened, or draw a
  water surface out of a CSV it never names, or leave a sill verdict to be inferred
  from which exception a run did not raise.

Everything below the ``@_dataset`` marker runs without any of the project's real
rasters.  The dataset-gated tests need ``Y:\\paleo`` and are turned on with
``PALEO_DATASET_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `scripts`

rasterio = pytest.importorskip("rasterio")
Image = pytest.importorskip("PIL.Image")

import scripts.animate_khvalynian_spill as anim  # noqa: E402
from paleoeurope.viz.paleocostline_render import (  # noqa: E402
    BasinSeed,
    RenderConfig,
    SillExpectation,
    UnmeasuredSill,
)

_dataset = pytest.mark.skipif(
    os.environ.get("PALEO_DATASET_TESTS", "").strip() not in ("1", "true", "TRUE"),
    reason=(
        "real-dataset test: needs the 17 khv_*.tif under "
        "$PALEO_OUT_ROOT/khvalynian_caspian/topo. Set PALEO_DATASET_TESTS=1."
    ),
)

_needs_curves = pytest.mark.skipif(
    not (
        Path(
            os.environ.get("PALEO_DATA_ROOT")
            or r"D:\pasha\RESEARCH\Paleo-Europe_3D"
        ).exists()
    ),
    reason="needs the Paleo-Europe_3D curve CSVs",
)


# ---------------------------------------------------------------------------
# A synthetic raster that the REAL seed boxes resolve on
# ---------------------------------------------------------------------------

#: The scene extent every raster of this project carries, so the registry's real
#: boxes can be resolved against a 0.25-degree toy instead of a 42 MB production
#: file.  Same bounds, same CRS, 130 x 72 cells.
_WEST, _SOUTH, _EAST, _NORTH = 25.5, 36.0, 58.0, 54.0
_PIXEL_DEG = 0.25


def _write_scene_raster(
    path: Path,
    *,
    deeps: dict[str, tuple[float, float, float]] | None = None,
    background: float = 100.0,
    nodata: float = -9999.0,
    hole: tuple[float, float, float, float] | None = None,
    tags: dict | None = None,
) -> Path:
    """A toy raster on the scene's own extent, with a chosen minimum per basin."""
    width = int(round((_EAST - _WEST) / _PIXEL_DEG))
    height = int(round((_NORTH - _SOUTH) / _PIXEL_DEG))
    transform = rasterio.transform.from_origin(_WEST, _NORTH, _PIXEL_DEG, _PIXEL_DEG)
    data = np.full((height, width), background, dtype="float32")
    for lon, lat, value in (deeps or {}).values():
        col, row = ~transform * (lon, lat)
        data[int(row), int(col)] = value
    if hole is not None:
        w, s, e, n = hole
        c0, r1 = ~transform * (w, s)
        c1, r0 = ~transform * (e, n)
        data[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = nodata
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)
        dataset.update_tags(**(tags or {}))
    return path


_TOY_DEEPS = {
    "caspian": (50.75, 38.95, -1000.0),
    "pontic": (35.25, 42.25, -2000.0),
    "mediterranean": (28.75, 36.75, -3000.0),
}
_TOY_LEVELS = {"caspian": 29.9444, "pontic": -18.0204, "mediterranean": -77.9058}


@pytest.fixture()
def toy_raster(tmp_path):
    return _write_scene_raster(tmp_path / "khv_16.5ka.tif", deeps=_TOY_DEEPS)


@pytest.fixture(scope="session")
def untagged_topo_dir(tmp_path_factory):
    """A COMPLETE topo tree whose rasters can none of them be planned.

    Every test below that used to pass ``topo_dir=tmp_path / "nope"`` uses this
    instead, because from 2026-08-04 an INCOMPLETE raster directory RAISES
    (:class:`anim.IncompleteRasterTreeError`) rather than becoming a sequence of
    refusal frames.  The old default pointed at a tree holding 2 of the 17 rasters,
    and a run there produced a two-frame animation wearing the 13-17 ka window's
    name.

    The refusal path is still exercised, one step further in: these rasters carry no
    ``vertical_frame`` tag, so every epoch refuses inside ``plan_epoch`` with
    ``BasinLevelUnavailableError``.  That is the honest shape of the two situations
    -- "the directory is wrong" raises, "this epoch cannot be planned" keeps its
    slot -- and the tests now distinguish them.
    """
    directory = tmp_path_factory.mktemp("khv_topo_untagged")
    for epoch in anim.epoch_grid():
        _write_scene_raster(directory / f"khv_{epoch:g}ka.tif", deeps=_TOY_DEEPS)
    return directory


def _flat_png(path: Path, size_px, value=(200, 30, 30)) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (int(size_px[0]), int(size_px[1])), tuple(value)).save(path)
    return Path(path)


# ---------------------------------------------------------------------------
class TestTheEpochGridIsTheBakesOwn:
    """The 17 epochs are read out of ``bake_khvalynian``, not typed a second time."""

    def test_the_window_is_exactly_seventeen_quarter_ka_epochs(self):
        grid = anim.epoch_grid()
        assert len(grid) == 17
        assert grid[0] == 17.0 and grid[-1] == 13.0
        steps = {round(a - b, 10) for a, b in zip(grid, grid[1:])}
        assert steps == {0.25}, f"the window is not a uniform 0.25 ka grid: {steps}"

    def test_the_grid_is_read_from_the_bake_rather_than_written_out_again(self):
        """A second hand-written list is a second definition that can grow a gap."""
        from scripts import bake_khvalynian as bk

        assert set(anim.epoch_grid()) <= set(bk.EPOCHS_KA)
        assert anim.epoch_grid() == tuple(
            e for e in bk.EPOCHS_KA if 13.0 <= e <= 17.0
        )

    def test_the_raster_name_is_built_the_way_the_bake_builds_it(self, tmp_path):
        """``17`` not ``17.0``: the bake writes ``khv_{epoch:g}ka.tif``."""
        (tmp_path / "khv_17ka.tif").write_bytes(b"")
        (tmp_path / "khv_16.5ka.tif").write_bytes(b"")
        assert anim.raster_for(17.0, tmp_path).name == "khv_17ka.tif"
        assert anim.raster_for(16.5, tmp_path).name == "khv_16.5ka.tif"

    def test_a_missing_epoch_raster_is_refused_and_not_substituted(self, tmp_path):
        (tmp_path / "khv_16.5ka.tif").write_bytes(b"")
        with pytest.raises(anim.MissingEpochRasterError) as excinfo:
            anim.raster_for(16.25, tmp_path)
        assert "NOT substituted" in str(excinfo.value)
        assert "16.25" in str(excinfo.value)


# ---------------------------------------------------------------------------
class TestTheSeedRegistryIsARuleNotACoordinate:
    """RAISE-not-compute item 6: no hand-written lon/lat, and the box is recorded."""

    def test_every_box_is_named_bounded_and_explained(self):
        assert {b.basin for b in anim.SEED_BOXES} == {
            "caspian", "pontic", "mediterranean",
        }
        assert len({b.name for b in anim.SEED_BOXES}) == len(anim.SEED_BOXES)
        for box in anim.SEED_BOXES:
            assert box.lon_min < box.lon_max and box.lat_min < box.lat_max
            assert len(box.why) > 60, f"{box.name} has no written justification"
            assert _WEST <= box.lon_min and box.lon_max <= _EAST
            assert _SOUTH <= box.lat_min and box.lat_max <= _NORTH

    def test_a_box_with_inverted_bounds_is_refused(self):
        with pytest.raises(ValueError, match="not ascending"):
            anim.SeedBox("x", "x", 10.0, 1.0, 5.0, 2.0, why="a" * 80)

    def test_a_box_with_no_written_reason_is_refused(self):
        with pytest.raises(ValueError, match="no default"):
            anim.SeedBox("x", "x", 1.0, 1.0, 2.0, 2.0, why="   ")

    def test_a_seed_is_the_argmin_of_its_box_and_records_where_it_came_from(
        self, toy_raster
    ):
        box = next(b for b in anim.SEED_BOXES if b.basin == "pontic")
        seed = anim.resolve_seed(
            box, toy_raster, level_m=_TOY_LEVELS["pontic"], epoch_ka=16.5
        )
        assert seed.elevation_m == pytest.approx(-2000.0)
        # the CELL CENTRE of the cell holding the planted minimum, not the point
        # itself: a seed is a cell, and the registry reports the cell it resolved
        assert seed.lon == pytest.approx(35.375, abs=1e-9)
        assert seed.lat == pytest.approx(42.125, abs=1e-9)
        # everything needed to re-derive it, in the DATA rather than a heuristic
        payload = seed.to_dict()
        assert payload["box"]["bounds_wsen"] == list(box.bounds())
        assert payload["epoch_ka"] == 16.5
        assert payload["raster_sha256"] == anim.sha256_of(toy_raster)
        assert payload["row"] >= 0 and payload["col"] >= 0
        assert "argmin" in payload["rule"] and "FULL resolution" in payload["rule"]

    def test_the_same_box_on_the_same_raster_resolves_to_the_same_cell(
        self, toy_raster
    ):
        box = next(b for b in anim.SEED_BOXES if b.basin == "caspian")
        first = anim.resolve_seed(
            box, toy_raster, level_m=_TOY_LEVELS["caspian"], epoch_ka=16.5
        )
        second = anim.resolve_seed(
            box, toy_raster, level_m=_TOY_LEVELS["caspian"], epoch_ka=16.5
        )
        assert (first.row, first.col) == (second.row, second.col)
        assert first.lon == second.lon and first.lat == second.lat

    def test_a_seed_that_is_not_under_its_own_water_is_refused(self, toy_raster):
        """Four of twelve plausible hand-written seeds on this project failed here."""
        box = next(b for b in anim.SEED_BOXES if b.basin == "caspian")
        with pytest.raises(anim.SeedAboveItsLevelError) as excinfo:
            anim.resolve_seed(box, toy_raster, level_m=-5000.0, epoch_ka=16.5)
        message = str(excinfo.value)
        assert "-1000" in message and "-5000" in message
        assert "empty basin" in message

    def test_a_box_holding_no_data_is_refused_rather_than_moved(self, tmp_path):
        raster = _write_scene_raster(
            tmp_path / "khv_16.5ka.tif",
            deeps=_TOY_DEEPS,
            hole=(30.0, 42.0, 38.0, 44.5),          # exactly the pontic box
        )
        box = next(b for b in anim.SEED_BOXES if b.basin == "pontic")
        with pytest.raises(anim.SeedBoxEmptyError, match="no finite cell"):
            anim.resolve_seed(
                box, raster, level_m=_TOY_LEVELS["pontic"], epoch_ka=16.5
            )

    def test_resolving_all_three_hashes_the_raster_once_and_records_it(
        self, toy_raster
    ):
        seeds = anim.resolve_seeds(toy_raster, _TOY_LEVELS, epoch_ka=16.5)
        assert [s.basin for s in seeds] == [b.basin for b in anim.SEED_BOXES]
        assert len({s.raster_sha256 for s in seeds}) == 1
        assert all(s.raster_sha256 == anim.sha256_of(toy_raster) for s in seeds)

    def test_a_basin_with_no_level_cannot_be_seeded(self, toy_raster):
        with pytest.raises(anim.BasinLevelUnavailableError, match="mediterranean"):
            anim.resolve_seeds(
                toy_raster,
                {"caspian": 29.9, "pontic": -18.0},
                epoch_ka=16.5,
            )

    def test_a_resolved_seed_becomes_a_plain_basin_seed_unchanged(self, toy_raster):
        seed = anim.resolve_seeds(toy_raster, _TOY_LEVELS, epoch_ka=16.5)[0]
        basin_seed = seed.to_basin_seed()
        assert isinstance(basin_seed, BasinSeed)
        assert basin_seed.name == seed.basin
        assert (basin_seed.lon, basin_seed.lat) == (seed.lon, seed.lat)
        assert basin_seed.level_m == seed.level_m

    def test_the_registry_says_out_loud_that_the_picture_depends_on_it(self):
        """B1: the VERDICT is seed-independent, the painted corridor is not."""
        doc = anim.SeedBox.__doc__ or ""
        assert "28.000944137573242" in doc
        assert "2 383 005" in doc or "2,383,005" in doc

    def test_the_registry_does_not_claim_to_reproduce_both_old_seeds(self):
        """It reproduces one and corrects the other; the comment must say which."""
        import inspect

        source = inspect.getsource(anim)
        block = source[source.index("#: THE REGISTRY."):source.index("SEED_BOXES:")]
        assert "the same cell" in block          # the Caspian
        assert "different\n#:   and deeper cell" in block or "deeper cell" in block
        assert "-2645.555" in block and "-2578.521" in block
        assert "67.03 m deeper" in block


# ---------------------------------------------------------------------------
class TestEveryBasinPairIsAccountedFor:
    """Three basins are three pairs, and a partial mapping is as unchecked as None."""

    _FRAME = "F0 present-day datum (H_modern + dz_solid)"

    def _expectations(self, monkeypatch):
        monkeypatch.setattr(
            anim, "manych_expectation",
            lambda epoch, frame: SillExpectation(
                value_m=27.701019287109375, frame=frame,
                source="test", measured_by="test", lon=44.102083, lat=45.706250,
            ),
        )
        return anim.sill_expectations_for(16.5, self._FRAME, 10.819)

    def test_all_three_unordered_pairs_are_present(self, monkeypatch):
        keys = {tuple(sorted(k)) for k in self._expectations(monkeypatch)}
        assert keys == {
            ("caspian", "pontic"),
            ("mediterranean", "pontic"),
            ("caspian", "mediterranean"),
        }

    def test_exactly_one_pair_is_checked_and_two_are_declared(self, monkeypatch):
        values = self._expectations(monkeypatch)
        checked = [v for v in values.values() if isinstance(v, SillExpectation)]
        declared = [v for v in values.values() if isinstance(v, UnmeasuredSill)]
        assert len(checked) == 1 and len(declared) == 2
        for entry in declared:
            assert len(entry.reason) > 200, "a refusal with no numbers in it"

    def test_the_caspian_mediterranean_entry_says_there_is_no_such_sill(self):
        reason = anim.CASPIAN_MEDITERRANEAN_UNMEASURED.reason
        assert "no direct sill" in reason
        assert "through the Pontic" in reason
        assert "never touch" in reason

    def test_the_bosphorus_entry_carries_the_number_it_declines_to_check(self):
        entry = anim.bosphorus_unmeasured(16.5, 10.818970)
        assert isinstance(entry, UnmeasuredSill)
        # the crest, its position, the conversion, and the gap that stops it being armed
        assert "-20.000000" in entry.reason
        assert "28.997917" in entry.reason and "41.010417" in entry.reason
        assert "-9.181" in entry.reason
        assert "+13.345" in entry.reason
        assert "+22.857" in entry.reason
        assert "-9.51" in entry.reason      # the independent unsmoothed reading
        # ... and every one of those is labelled as a 16.5 ka reading, because the
        # string ships in manifests spanning 17-13 ka where all four move
        assert "16.5 ka MEASUREMENTS" in entry.reason
        assert "NONE OF THEM IS A WINDOW-WIDE CONSTANT" in entry.reason
        assert "measured_saddle_m" in entry.reason

    def test_the_manych_constant_is_the_path_crest_not_the_lower_bound(self):
        from scripts import bake_khvalynian as bk

        assert anim.MANYCH_CREST_F0_M == bk.MANYCH_SILL_PATH_M == 25.00
        assert anim.MANYCH_CREST_F0_M != bk.MANYCH_SILL_M
        doc = anim.manych_expectation.__doc__ or ""
        assert "LOWER BOUND" in doc and "+2.30 m" in doc

    def test_a_bare_float_could_never_be_produced_here(self, monkeypatch):
        for value in self._expectations(monkeypatch).values():
            assert isinstance(value, (SillExpectation, UnmeasuredSill))
            assert not isinstance(value, (int, float))


# ---------------------------------------------------------------------------
class TestADurationIsMillisecondsAndIsReadBack:
    """``imageio`` 2.37.2 turns a seconds-shaped duration into a silent zero."""

    def _frames(self, tmp_path, n=3, size=(24, 24)):
        return [
            _flat_png(tmp_path / f"f{i}.png", size, (10 + 40 * i, 20, 30))
            for i in range(n)
        ]

    def test_the_delay_is_read_back_out_of_the_written_file(self, tmp_path):
        frames = self._frames(tmp_path)
        gif = anim.write_gif(frames, tmp_path / "a.gif", duration_ms=250)
        assert anim.read_frame_durations_ms(gif) == [250, 250, 250]
        assert anim.assert_frame_durations_ms(gif, 250, 3) == [250, 250, 250]

    def test_a_seconds_shaped_duration_is_exactly_what_this_guard_catches(
        self, tmp_path
    ):
        """Written straight through imageio, bypassing :func:`write_gif`."""
        import imageio.v3 as iio

        frames = [np.asarray(Image.open(p).convert("RGB"))
                  for p in self._frames(tmp_path)]
        bad = tmp_path / "seconds.gif"
        iio.imwrite(bad, np.stack(frames), extension=".gif", duration=0.25, loop=0)
        read_back = anim.read_frame_durations_ms(bad)
        assert read_back != [250, 250, 250]
        # the measured failure mode: no exception, a valid file, zero-length frames
        assert set(read_back) <= {None, 0}
        with pytest.raises(anim.GifDurationError) as excinfo:
            anim.assert_frame_durations_ms(bad, 250, 3)
        assert "imageio 2.37.2" in str(excinfo.value)

    def test_write_gif_refuses_a_seconds_shaped_value_before_writing(self, tmp_path):
        frames = self._frames(tmp_path)
        with pytest.raises(anim.GifDurationError, match="zero or negative"):
            anim.write_gif(frames, tmp_path / "b.gif", duration_ms=0)

    def test_a_delay_that_cannot_survive_centiseconds_is_refused(self, tmp_path):
        frames = self._frames(tmp_path)
        with pytest.raises(anim.GifDurationError, match="centiseconds"):
            anim.write_gif(frames, tmp_path / "c.gif", duration_ms=205)

    def test_a_frame_that_vanished_in_the_write_is_refused(self, tmp_path):
        frames = self._frames(tmp_path, n=3)
        gif = anim.write_gif(frames, tmp_path / "d.gif", duration_ms=250)
        with pytest.raises(anim.GifDurationError, match="not 4"):
            anim.assert_frame_durations_ms(gif, 250, 4)

    def test_the_default_duration_is_a_whole_number_of_centiseconds(self):
        assert anim.FRAME_DURATION_MS % 10 == 0
        assert anim.FRAME_DURATION_MS >= 10

    def test_the_new_writers_library_has_the_same_failure_the_guard_was_built_for(
        self, tmp_path
    ):
        """``write_gif`` left imageio for PIL on 2026-08-04; the hazard came along.

        Written straight through PIL with the same palette ``write_gif`` uses, so the
        only difference from the shipped path is the duration handed in.
        """
        frames = _noisy_sequence(tmp_path, n=3, size_px=(24, 24))
        master = anim.build_master_palette(frames)
        mapped = [anim.map_frame_through(p, master) for p in frames]

        def _write(name, duration):
            path = tmp_path / name
            mapped[0].save(path, save_all=True, append_images=mapped[1:],
                           duration=duration, loop=0, palette=master.as_bytes())
            return anim.read_frame_durations_ms(path)

        assert _write("seconds.gif", 0.25) == [None, None, None]
        assert _write("good.gif", 250) == [250, 250, 250]
        assert _write("odd.gif", 205) == [200, 200, 200]   # centiseconds, truncated


# ---------------------------------------------------------------------------
class TestEveryFrameIsTheSameSize:
    """``bbox_inches='tight'`` makes the renderer's output size follow its text."""

    def test_the_canvas_is_the_elementwise_maximum(self):
        assert anim.canvas_size([(10, 20), (30, 5)]) == (30, 20)

    def test_with_no_rendered_frame_the_canvas_is_the_declared_fallback(self):
        assert anim.canvas_size([], fallback=(7, 9)) == (7, 9)
        assert anim.canvas_size([]) == anim.FALLBACK_CANVAS_PX

    def test_padding_centres_and_never_scales_or_crops(self, tmp_path):
        src = _flat_png(tmp_path / "s.png", (20, 10), (7, 8, 9))
        report = anim.pad_to_canvas(src, tmp_path / "d.png", (24, 16))
        assert report["source_size_px"] == [20, 10]
        assert report["canvas_size_px"] == [24, 16]
        assert report["pad_left_px"] == 2 and report["pad_right_px"] == 2
        assert report["pad_top_px"] == 3 and report["pad_bottom_px"] == 3
        with Image.open(tmp_path / "d.png") as out:
            assert out.size == (24, 16)
            assert out.getpixel((0, 0)) == anim.CANVAS_PAD_RGB   # padding, invisible
            assert out.getpixel((12, 8)) == (7, 8, 9)            # picture, unscaled

    def test_a_frame_larger_than_the_canvas_is_refused_not_cropped(self, tmp_path):
        src = _flat_png(tmp_path / "s.png", (40, 40))
        with pytest.raises(anim.FrameSizeError, match="never crops"):
            anim.pad_to_canvas(src, tmp_path / "d.png", (20, 20))

    def test_a_sequence_of_mixed_sizes_is_refused(self, tmp_path):
        a = _flat_png(tmp_path / "a.png", (10, 10))
        b = _flat_png(tmp_path / "b.png", (11, 10))
        assert anim.assert_uniform_size([a, a]) == (10, 10)
        with pytest.raises(anim.FrameSizeError, match="jitters"):
            anim.assert_uniform_size([a, b])

    def test_the_writer_refuses_mixed_shapes_even_if_the_caller_forgot(self, tmp_path):
        a = _flat_png(tmp_path / "a.png", (10, 10))
        b = _flat_png(tmp_path / "b.png", (11, 10))
        with pytest.raises(anim.FrameSizeError, match="distinct shapes"):
            anim.write_gif([a, b], tmp_path / "x.gif", duration_ms=250)


# ---------------------------------------------------------------------------
class TestARefusedEpochKeepsItsSlot:
    """Dropping compresses time; interpolating invents a level."""

    def test_the_placeholder_is_exactly_the_canvas_size(self, tmp_path):
        out = anim.refusal_frame(
            16.75, "SomeError: because", tmp_path / "r.png", (240, 180)
        )
        assert anim.read_png_size(out) == (240, 180)
        with Image.open(out) as image:
            assert image.mode == "RGB"

    def test_the_placeholder_carries_threshold_null_and_the_refusal(self, tmp_path):
        message = "CurveAgeGapError: the curve has no row at 16.75 ka"
        out = anim.refusal_frame(
            16.75, message, tmp_path / "r.png", (240, 180),
            levels={
                "caspian": anim.BasinLevel(
                    "caspian", 24.3507, "F1", "rule", "provenance"
                )
            },
        )
        sidecar = json.loads(Path(out).with_suffix(".refusal.json").read_text())
        assert sidecar["threshold"] is None
        assert sidecar["refusal"] == message
        assert sidecar["epoch_ka"] == 16.75
        assert sidecar["levels_that_resolved"]["caspian"]["level_m"] == 24.3507

    def test_a_truncated_refusal_says_it_was_truncated(self):
        wrapped = anim._wrap("word " * 4000, 90, max_lines=6)
        assert wrapped.count("\n") == 5
        assert "truncated to fit the frame" in wrapped
        assert ".refusal.json" in wrapped

    def test_a_refusal_that_fits_is_not_marked_truncated(self):
        assert "truncated" not in anim._wrap("short refusal", 90)

    def test_every_epoch_keeps_a_slot_when_none_can_be_planned(
        self, tmp_path, untagged_topo_dir
    ):
        """The whole pipeline, over a COMPLETE tree nothing can be planned from."""
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            duration_ms=250,
            fallback_canvas_px=(200, 150),
        )
        assert manifest["n_frames"] == 17
        assert manifest["n_rendered"] == 0
        assert manifest["n_refused"] == 17
        assert [f["epoch_ka"] for f in manifest["frames"]] == list(anim.epoch_grid())
        for frame in manifest["frames"]:
            assert frame["threshold"] is None
            assert frame["refusal"].startswith("BasinLevelUnavailableError")
            assert "vertical_frame" in frame["refusal"]
            assert frame["played_for_ms"] == 250
            assert Path(frame["canvas_frame"]).exists()
        # ... and the animation still exists, with 17 verified delays in BOTH files
        assert manifest["animation"]["gif_frame_durations_ms_read_back_with_pil"] == (
            [250] * 17
        )
        assert manifest["animation"]["webp_frame_durations_ms_read_back_with_pil"] == (
            [250] * 17
        )
        assert manifest["animation"]["webp_delay_asserted"] is True
        assert manifest["animation"]["frame_size_px"] == [200, 150]
        assert anim.assert_uniform_size(
            [f["canvas_frame"] for f in manifest["frames"]]
        ) == (200, 150)

    def test_a_partial_raster_directory_raises_with_the_two_counts(self, tmp_path):
        """The defect finding 2 names: 2 of 17 rasters, and a two-frame 'window'."""
        partial = tmp_path / "partial"
        partial.mkdir()
        for epoch in (16.5, 16.0):
            _write_scene_raster(partial / f"khv_{epoch:g}ka.tif", deeps=_TOY_DEEPS)
        with pytest.raises(anim.IncompleteRasterTreeError) as excinfo:
            anim.build_animation(
                out_root=tmp_path / "out",
                topo_dir=partial,
                fallback_canvas_px=(120, 90),
            )
        message = str(excinfo.value)
        assert "holds 2 of the 17 epoch rasters" in message
        assert "16.75" in message and "13.25" in message, "the missing epochs by name"
        assert str(anim.CANONICAL_TOPO_DIR) in message, "and where the tree of record is"
        assert "A short sequence is NOT rendered" in message
        assert not (tmp_path / "out" / "khvalynian_spill.gif").exists()

    def test_a_missing_raster_directory_raises_rather_than_refusing_seventeen_times(
        self, tmp_path
    ):
        with pytest.raises(anim.IncompleteRasterTreeError) as excinfo:
            anim.plan_sequence(topo_dir=tmp_path / "no-such-topo-dir")
        assert "holds 0 of the 17 epoch rasters" in str(excinfo.value)
        assert "the directory does not exist" in str(excinfo.value)

    def test_the_census_names_both_spellings_of_the_directory(self):
        census = anim.assert_raster_tree_complete(
            anim.CANONICAL_TOPO_DIR, [16.5]
        ) if (anim.CANONICAL_TOPO_DIR / "khv_16.5ka.tif").exists() else None
        if census is None:
            pytest.skip("the tree of record is not mounted on this machine")
        assert census["epochs_found"] == census["epochs_needed"] == 1
        assert census["is_the_tree_of_record"] is True
        assert set(census["topo_dir"]) == {"as_written", "resolved", "differ"}

    def test_a_refused_epoch_never_inherits_a_neighbours_level(
        self, tmp_path, untagged_topo_dir
    ):
        """No frame's threshold may repeat a neighbour's, because none has one."""
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        assert all(f["threshold"] is None for f in manifest["frames"])
        # the refusal lands before any level is read, so there is nothing to carry
        assert all(f["basin_levels"] == {} for f in manifest["frames"])

    def test_a_partial_refusal_keeps_the_levels_that_did_resolve(self):
        """A refusal that reports no levels when it has two overstates itself."""
        error = anim.BasinLevelUnavailableError(
            "pontic level unavailable at 16.75 ka",
            resolved={
                "caspian": anim.BasinLevel("caspian", 24.3507, "F1", "r", "p"),
                "mediterranean": anim.BasinLevel(
                    "mediterranean", -78.6863, "F3", "r", "p"
                ),
            },
        )
        assert set(error.resolved) == {"caspian", "mediterranean"}
        assert "pontic" not in error.resolved
        plain = anim.BasinLevelUnavailableError("nothing resolved")
        assert plain.resolved == {}


# ---------------------------------------------------------------------------
class TestTheManifestCanAuditOneFrameAlone:

    def test_the_manifest_carries_every_field_a_frame_audit_needs(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        for key in (
            "what", "epochs", "n_frames", "scenario", "render_config",
            "corridor_note", "seed_registry", "sill_expectations",
            "offset_cross_check", "sequence_limits",
            "sequence_limits_extrema_by_epoch", "canvas", "animation", "frames",
            "level_curves", "raster_source_resolution", "sill_verdicts", "palette",
            "paths",
        ):
            assert key in manifest, f"manifest lost {key}"
        frame = manifest["frames"][0]
        for key in (
            "slot", "epoch_ka", "played_for_ms", "threshold", "refusal", "raster",
            "basin_levels", "seeds", "sill", "spill", "clipping", "png",
            "canvas_frame", "canvas_padding",
        ):
            assert key in frame, f"frame record lost {key}"
        for key in ("path_as_read", "source_path", "source_resolution", "sha256",
                    "vertical_frame"):
            assert key in frame["raster"], f"raster record lost {key}"
        assert json.loads(
            (Path(manifest["frames"][0]["canvas_frame"]).parent.parent
             / "sequence_manifest.json").read_text()
        )["n_frames"] == 17

    def test_the_seed_registry_travels_into_the_manifest(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        registry = manifest["seed_registry"]
        assert len(registry["boxes"]) == 3
        assert "argmin" in registry["rule"]
        assert "28.000944137573242" in registry["why_it_is_load_bearing"]

    @_needs_curves
    def test_the_offset_cross_check_is_recomputed_at_write_time(
        self, tmp_path, untagged_topo_dir
    ):
        """It used to be five hard-coded rows that no run could falsify."""
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        block = manifest["offset_cross_check"]
        assert block["status"] == "MEASURED"
        rows = block["rows"]
        assert {r["epoch_ka"] for r in rows} == {
            e for e, _ in anim.OFFSET_REFERENCE
        }
        # the reference is a document that prints these digits, named in the block
        assert "manych-sill-history" in block["reference_source"]
        assert "do not quote a number out of this table" in block["reference_source"]
        assert block["reference_printing_precision_m"] == 0.0005
        for row in rows:
            assert abs(row["difference_m"]) <= 0.0005
            assert row["within_the_references_printing_precision"] is True
            assert row["measured_m"] - row["reference_m"] == pytest.approx(
                row["difference_m"], abs=1e-9
            )
        assert block["all_within_the_references_printing_precision"] is True
        # the numbers are the engine's, taken here and not copied from a constant
        from scripts import bake_khvalynian as bk

        assert rows[0]["measured_m"] == pytest.approx(
            float(bk.sea_surface_offset(rows[0]["epoch_ka"])), abs=0.0
        )

    def test_the_retired_offset_reference_is_not_reintroduced(self):
        """21 ka 87.332 had no source in either repo; the contract has no 21 ka row."""
        epochs = {e for e, _ in anim.OFFSET_REFERENCE}
        assert 21.0 not in epochs
        values = {v for _, v in anim.OFFSET_REFERENCE}
        assert 87.332 not in values and 81.959 not in values and 30.005 not in values
        # ... and every reference figure is printed to exactly three decimals
        for _, value in anim.OFFSET_REFERENCE:
            assert round(value, 3) == value

    def test_the_corridor_note_is_computed_per_epoch_from_this_runs_frames(self):
        """It shipped '+13.345 m' as the window's saddle; nine frames read 13.3-16.1."""
        note = anim.corridor_note(
            [
                {
                    "epoch_ka": 16.5,
                    "basin_levels": {"pontic": {"level_m": -18.0204}},
                    "dz_bosphorus_m": 10.81897,
                    "sill": [
                        {
                            "pair": ["mediterranean", "pontic"],
                            "measured_saddle_m": 13.3446865,
                        }
                    ],
                },
                {
                    "epoch_ka": 14.0,
                    "basin_levels": {"pontic": {"level_m": -22.3971}},
                    "dz_bosphorus_m": 13.5514,
                    "sill": [
                        {
                            "pair": ["mediterranean", "pontic"],
                            "measured_saddle_m": 16.0640049,
                        }
                    ],
                },
            ]
        )
        assert "CorridorNotRampableError" in note["rule"]
        assert note["corridor"] == "step"
        assert note["verdict"] == "AT_MOST_ONE_DONOR_RECEIVER_PAIR"
        # the saddle is a RANGE over the sequence, not one epoch's number -- and the
        # field says out loud that the pair is the SORT's endpoints
        grid = note["on_the_render_grid"]
        low, high = grid["measured_saddle_min_max_m_sorted_not_chronological"]
        assert low == pytest.approx(13.3446865) and high == pytest.approx(16.0640049)
        assert "measured_saddle_range_m" not in grid, (
            "the old name said 'range' and was read as a run through time"
        )
        # ... and the series itself is carried IN EPOCH ORDER beside it
        assert grid["measured_saddle_by_epoch"] == [
            [16.5, pytest.approx(13.3446865)],
            [14.0, pytest.approx(16.0640049)],
        ]
        assert grid["measured_saddle_lowest_at_epoch_ka"] == 16.5
        assert grid["measured_saddle_highest_at_epoch_ka"] == 14.0
        assert grid["pontic_below_it_at_every_one"] is True
        # the independent crest is BOSPHORUS_CREST_F0_M + dz, per epoch
        assert note["on_the_independent_crest"]["crest_f1_range_m"] == [
            pytest.approx(anim.BOSPHORUS_CREST_F0_M + 10.81897),
            pytest.approx(anim.BOSPHORUS_CREST_F0_M + 13.5514),
        ]
        assert note["on_the_independent_crest"]["narrowest_headroom"][
            "epoch_ka"
        ] == 16.5
        assert note["on_the_independent_crest"]["narrowest_headroom"][
            "headroom_to_the_independent_crest_m"
        ] == pytest.approx(8.8394, abs=1e-3)

    def test_a_frame_with_no_measured_saddle_is_not_quietly_dropped(self):
        note = anim.corridor_note(
            [{"epoch_ka": 16.75, "basin_levels": {}, "dz_bosphorus_m": None,
              "sill": []}]
        )
        assert note["n_frames"] == 1
        assert note["per_epoch"][0]["measured_saddle_m"] is None
        assert note["per_epoch"][0]["pontic_below_the_measured_saddle"] is None
        assert note["on_the_render_grid"]["epochs_with_a_measured_saddle"] == 0
        assert note["verdict"] == "NOT_ESTABLISHED_ON_THESE_FRAMES"

    def test_the_paths_block_records_both_spellings_and_the_drive_map(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        block = manifest["paths"]
        assert "SAME file" in block["decision"]
        assert set(block["topo_dir"]) == {"as_written", "resolved", "differ"}
        for entry in block["drive_map"]:
            assert entry["drive"].endswith(":")
            assert set(entry["measured_on"]) == {"as_written", "resolved"}
            if entry["expands_to_a_unc_share"]:
                assert entry["unc_target"].startswith("\\\\")
        # the inventory counts the values actually written, per key
        for key, tally in block["spelling_by_key"].items():
            assert set(tally) == {"unc", "drive_letter", "other"}
            assert sum(tally.values()) > 0, key

    def test_path_record_gives_both_spellings_of_a_mapped_drive(self):
        record = anim.path_record(anim.CANONICAL_TOPO_DIR)
        assert record["as_written"] == str(anim.CANONICAL_TOPO_DIR)
        assert record["resolved"] == str(Path(anim.CANONICAL_TOPO_DIR).resolve())
        assert record["differ"] == (record["as_written"] != record["resolved"])
        assert anim.path_record(None) is None


# ---------------------------------------------------------------------------
def _noisy_png(path: Path, size_px=(64, 48), seed=0) -> Path:
    """A frame with thousands of distinct colours, so 256 entries actually bind."""
    rng = np.random.default_rng(seed)
    height, width = int(size_px[1]), int(size_px[0])
    array = np.empty((height, width, 3), dtype=np.uint8)
    array[..., 0] = rng.integers(0, 256, (height, width))
    array[..., 1] = (np.arange(width)[None, :] * 3 + seed * 17) % 256
    array[..., 2] = (np.arange(height)[:, None] * 5 + seed * 29) % 256
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path)
    return Path(path)


def _noisy_sequence(tmp_path: Path, n=4, size_px=(64, 48), first=0) -> list[Path]:
    return [
        _noisy_png(tmp_path / f"n{i}.png", size_px, seed=first + i) for i in range(n)
    ]


class TestOneGlobalPaletteForTheWholeGif:
    """The blocker: the encoder re-quantized per frame, one layer under the renderer.

    Every assertion below reads the WRITTEN FILE -- its bytes, or its decoded
    pixels.  None of them takes the writer's word for anything, for the same reason
    the frame delay is read back: a per-frame colour table is a perfectly legal
    GIF, so nothing in the encode path reports it and no decoder complains.
    """

    def test_the_palette_is_quantized_once_over_the_whole_sequence(self, tmp_path):
        frames = _noisy_sequence(tmp_path)
        master = anim.build_master_palette(frames)
        assert master.size == 256
        assert list(master.built_over) == [str(p.resolve()) for p in frames]
        assert list(master.frames_sha256) == [anim.sha256_of(p) for p in frames]
        assert master.frame_size_px == (64, 48)
        # "over the whole sequence" is a number, not a claim: every pixel of every
        # frame went into the one pass
        assert master.quantized_over_px == 4 * 64 * 48
        assert master.to_dict()["quantized_over_px"] == 4 * 64 * 48
        assert "Not a sample" in master.to_dict()["quantized_over"]
        # the number of colours the sequence actually holds is recorded, so the
        # compression the format forces is a measurement and not an impression
        assert master.n_distinct_in_sequence > master.n_from_quantizer

    def test_a_sequence_over_pils_bomb_guard_is_still_quantized_whole(
        self, tmp_path, monkeypatch
    ):
        """The guard is about hostile FILES; this array is built here, from frames
        this module produced. Sampling instead would be the defect, not the fix.

        The limit is set between ONE frame (3 072 px, which must still open under
        the guard, because a PNG on disk is exactly what the guard is for) and the
        STACK of three (9 216 px, which must not be refused).
        """
        frames = _noisy_sequence(tmp_path, n=3, size_px=(64, 48))
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)
        master = anim.build_master_palette(frames)
        assert master.quantized_over_px == 3 * 64 * 48 == 9216
        assert Image.MAX_IMAGE_PIXELS == 5000, "the guard must be put back"

    def test_the_bomb_guard_still_protects_the_frames_being_opened(
        self, tmp_path, monkeypatch
    ):
        """Lifting it for the in-memory stack must not lift it for files on disk."""
        frames = _noisy_sequence(tmp_path, n=2, size_px=(64, 48))
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        with pytest.raises(Image.DecompressionBombError):
            anim.build_master_palette(frames)

    def test_every_colour_table_entry_is_distinct(self, tmp_path):
        """A repeated entry makes PIL's palette normalisation repaint pixels."""
        frames = _noisy_sequence(tmp_path)
        master = anim.build_master_palette(frames)
        assert len(master.entries) == 256
        assert len(set(master.entries)) == 256
        assert len(master.as_bytes()) == 768

    def test_a_short_sequence_is_padded_with_distinct_filler_that_no_pixel_uses(
        self, tmp_path
    ):
        paths = []
        for index, colour in enumerate([(10, 20, 30), (200, 30, 30), (0, 0, 0)]):
            paths.append(_flat_png(tmp_path / f"f{index}.png", (8, 8), colour))
        master = anim.build_master_palette(paths)
        assert master.n_from_quantizer == 3
        assert master.n_filler == 253
        assert len(set(master.entries)) == 256
        assert "NEVER referenced by a pixel" in master.filler_rule
        # and the filler cannot win a nearest-colour contest: mapping is exact
        for path in paths:
            mapped = np.asarray(anim.map_frame_through(path, master))
            assert int(mapped.max()) < master.n_from_quantizer

    def test_a_table_size_gif_cannot_write_is_refused(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=2)
        with pytest.raises(anim.PaletteError, match="power of two"):
            anim.build_master_palette(frames, size=100)

    def test_an_empty_sequence_has_no_palette(self):
        with pytest.raises(anim.PaletteError, match="empty sequence"):
            anim.build_master_palette([])

    def test_frames_of_different_sizes_are_refused_before_quantizing(self, tmp_path):
        a = _flat_png(tmp_path / "a.png", (10, 10))
        b = _flat_png(tmp_path / "b.png", (11, 10))
        with pytest.raises(anim.FrameSizeError, match="weight the quantizer"):
            anim.build_master_palette([a, b])

    def test_the_written_gif_carries_not_one_local_colour_table(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=5)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        structure = anim.parse_gif_structure(gif)
        assert len(structure["images"]) == 5
        assert structure["global_color_table_entries"] == 256
        assert [i["local_color_table"] for i in structure["images"]] == [None] * 5

    def test_the_global_colour_table_is_the_master_palette_byte_for_byte(
        self, tmp_path
    ):
        frames = _noisy_sequence(tmp_path, n=3)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        structure = anim.parse_gif_structure(gif)
        assert structure["global_color_table"] == master.as_bytes()
        audit = anim.assert_one_global_palette(gif, master)
        assert audit["verdict"] == "ONE_GLOBAL_PALETTE"
        assert audit["n_frames_with_a_local_color_table"] == 0
        assert audit["global_color_table_matches_master"] is True

    def test_the_gif_decodes_to_the_frames_that_were_mapped_into_it(self, tmp_path):
        """The palette audit proves ONE table; this proves it is the one applied."""
        frames = _noisy_sequence(tmp_path, n=4)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        report = anim.assert_gif_frames_are_the_mapped_frames(gif, master, frames)
        assert report["n_frames_decoded_and_compared"] == 4
        assert report["pixels_differing"] == 0

    def test_the_defect_is_reproducible_and_the_checker_catches_it(self, tmp_path):
        """The regression pin: write the OLD way and the audit must refuse.

        ``imageio.v3.imwrite(..., extension='.gif')`` passes no palette, so
        ``PIL.GifImagePlugin`` runs ``convert('P', palette=ADAPTIVE)`` per frame and
        sets ``include_color_table`` on every delta frame.  That is exactly how the
        2026-08-03 deliverable got 16 local tables in 17 descriptors.
        """
        import imageio.v3 as iio

        frames = _noisy_sequence(tmp_path, n=4)
        stack = np.stack(
            [np.asarray(Image.open(p).convert("RGB")) for p in frames]
        )
        old = tmp_path / "old.gif"
        iio.imwrite(old, stack, extension=".gif", duration=250, loop=0)

        structure = anim.parse_gif_structure(old)
        locals_ = [i for i in structure["images"] if i["local_color_table"]]
        assert len(locals_) == len(structure["images"]) - 1, (
            "the defect no longer reproduces on this imageio/Pillow -- re-derive "
            "the claim before relaxing the guard"
        )
        assert all(i["local_equals_global"] is False for i in locals_)

        with pytest.raises(anim.PaletteError) as excinfo:
            anim.assert_one_global_palette(old)
        message = str(excinfo.value)
        assert "local colour tables that differ" in message
        assert "entries shared with the global table" in message

    def test_a_file_with_no_global_colour_table_is_refused(self, tmp_path):
        """Hand-built 14-byte GIF: header, screen descriptor with no table, trailer."""
        bare = tmp_path / "bare.gif"
        bare.write_bytes(
            b"GIF89a"
            + (4).to_bytes(2, "little")
            + (4).to_bytes(2, "little")
            + bytes([0x00, 0x00, 0x00])   # packed: no global colour table
            + b";"
        )
        structure = anim.parse_gif_structure(bare)
        assert structure["global_color_table"] is None
        with pytest.raises(anim.PaletteError, match="NO global colour table"):
            anim.assert_one_global_palette(bare)

    def test_a_local_table_identical_to_the_global_one_is_tolerated_and_counted(
        self, tmp_path
    ):
        """Redundant is not wrong; DIFFERENT is wrong. The two are distinguished."""
        frames = _noisy_sequence(tmp_path, n=3)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        raw = bytearray(gif.read_bytes())
        target = anim.parse_gif_structure(gif)["images"][1]
        packed_at = target["packed_byte_offset"]
        raw[packed_at] |= 0x87          # local colour table flag + 256 entries
        raw[packed_at + 1:packed_at + 1] = master.as_bytes()
        spliced = tmp_path / "redundant.gif"
        spliced.write_bytes(bytes(raw))

        structure = anim.parse_gif_structure(spliced)
        assert structure["images"][1]["local_color_table"] == master.as_bytes()
        assert structure["images"][1]["local_equals_global"] is True
        audit = anim.assert_one_global_palette(spliced, master)
        assert audit["n_frames_with_a_redundant_local_color_table"] == 1
        assert audit["n_frames_with_a_DIFFERING_local_color_table"] == 0
        assert audit["verdict"] == "ONE_GLOBAL_PALETTE"

    def test_a_gif_whose_table_is_not_the_master_is_refused(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=3)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        other = anim.build_master_palette(
            _noisy_sequence(tmp_path / "b", n=3, first=100)
        )
        assert other.as_bytes() != master.as_bytes()
        with pytest.raises(anim.PaletteError, match="not the palette its frames"):
            anim.assert_one_global_palette(gif, other)

    def test_the_delay_and_the_palette_are_both_read_back_from_the_same_file(
        self, tmp_path
    ):
        frames = _noisy_sequence(tmp_path, n=6)
        master = anim.build_master_palette(frames)
        gif = anim.write_gif(frames, tmp_path / "one.gif", duration_ms=250,
                             master=master)
        assert anim.assert_frame_durations_ms(gif, 250, 6) == [250] * 6
        assert anim.assert_one_global_palette(gif, master)["n_image_descriptors"] == 6

    def test_write_gif_builds_a_palette_when_none_is_handed_to_it(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=3)
        gif = anim.write_gif(frames, tmp_path / "auto.gif", duration_ms=250)
        structure = anim.parse_gif_structure(gif)
        assert all(i["local_color_table"] is None for i in structure["images"])


# ---------------------------------------------------------------------------
class TestTheWebpIsTheTwentyFourBitEscapeItIsAdvertisedAs:
    """``master_palette.json`` named the WebP as the escape from 256 colours.

    Until 2026-08-04 the file it named was written from the PALETTE-MAPPED frames,
    so it carried exactly the same 256 colours -- the shipped 0.5 ka deliverable's
    own ``webp_audit`` read ``n_distinct_colours_decoded: 222`` in the same JSON
    object as the sentence offering it as the way out of 256.  The remedy is now
    real, and every assertion below PARSES THE WRITTEN FILE.
    """

    def test_the_written_webp_decodes_bit_identical_to_its_source_frames(
        self, tmp_path
    ):
        frames = _noisy_sequence(tmp_path, n=4)
        webp = anim.write_webp(frames, tmp_path / "a.webp", duration_ms=250)
        report = anim.assert_webp_is_a_lossless_24bit_copy(webp, frames)
        assert report["n_frames_decoded"] == 4
        assert report["pixels_differing_from_the_source"] == 0
        assert report["verdict"] == "BIT_EXACT_24_BIT"

    def test_it_carries_more_colours_than_a_gif_table_could_hold(self, tmp_path):
        """The escape is only an escape if it escapes; this is the measurement."""
        frames = _noisy_sequence(tmp_path, n=4)
        master = anim.build_master_palette(frames)
        webp = anim.write_webp(frames, tmp_path / "a.webp", duration_ms=250)
        report = anim.assert_webp_is_a_lossless_24bit_copy(
            webp, frames, master=master
        )
        assert report["gif_colour_table_ceiling"] == 256
        assert report["n_distinct_colours_decoded"] > 256
        assert report["escapes_the_gif_ceiling"] is True
        # and the GIF of the same frames does NOT: that is the whole point
        gif = anim.write_gif(frames, tmp_path / "a.gif", duration_ms=250,
                             master=master)
        anim.assert_one_global_palette(gif, master)

    def test_a_sequence_with_few_colours_says_so_instead_of_claiming_an_escape(
        self, tmp_path
    ):
        """A flat sequence has nothing to escape, and that is recorded, not assumed."""
        frames = [
            _flat_png(tmp_path / f"f{i}.png", (32, 24), value=(10 * i, 20, 30))
            for i in range(3)
        ]
        webp = anim.write_webp(frames, tmp_path / "flat.webp", duration_ms=250)
        report = anim.assert_webp_is_a_lossless_24bit_copy(webp, frames)
        assert report["n_distinct_colours_decoded"] == 3
        assert report["escapes_the_gif_ceiling"] is False
        assert report["verdict"] == "BIT_EXACT_24_BIT"

    def test_a_lossy_webp_is_refused_because_it_changed_the_picture(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=3, size_px=(96, 96))
        lossy = anim.write_webp(frames, tmp_path / "lossy.webp", duration_ms=250,
                                lossless=False)
        with pytest.raises(anim.PaletteError) as excinfo:
            anim.assert_webp_is_a_lossless_24bit_copy(lossy, frames)
        assert "does not decode to the frames" in str(excinfo.value)
        assert "lossless" in str(excinfo.value)

    def test_the_headroom_note_claims_no_escape_it_was_not_handed(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=3)
        master = anim.build_master_palette(frames)
        without = anim.describe_palette_headroom(master)
        assert without["webp_escape_measured"] is False
        assert "NO such file is claimed" in without["if_256_is_not_enough"]

        webp = anim.write_webp(frames, tmp_path / "a.webp", duration_ms=250)
        audit = anim.assert_webp_is_a_lossless_24bit_copy(webp, frames, master=master)
        with_audit = anim.describe_palette_headroom(master, webp_audit=audit)
        assert with_audit["webp_escape_measured"] is True
        sentence = with_audit["if_256_is_not_enough"]
        assert f"{audit['n_distinct_colours_decoded']:,} distinct colours" in sentence
        assert "BIT_EXACT_24_BIT" in sentence
        assert "IS available in this deliverable" in sentence

    def test_the_webp_delay_is_asserted_from_two_frames_up(self, tmp_path):
        """One frame reads back 0 whatever is written; that is stated, not skipped."""
        frames = _noisy_sequence(tmp_path, n=3)
        two = anim.write_webp(frames[:2], tmp_path / "two.webp", duration_ms=250)
        durations, asserted, note = anim.check_webp_durations(two, 250, 2)
        assert durations == [250, 250] and asserted is True
        assert "ASSERTED" in note

        one = anim.write_webp(frames[:1], tmp_path / "one.webp", duration_ms=250)
        durations, asserted, note = anim.check_webp_durations(one, 250, 1)
        assert asserted is False
        assert "NOT ASSERTED" in note
        assert "no inter-frame interval" in note


# ---------------------------------------------------------------------------
class TestWhatOneSharedPaletteCosts:
    """Requirement 3: quantify it, and name the alternative if 256 is not enough."""

    def test_both_arms_are_measured_and_the_comparison_arm_is_the_defect(
        self, tmp_path
    ):
        frames = _noisy_sequence(tmp_path, n=4)
        master = anim.build_master_palette(frames)
        cost = anim.measure_palette_cost(frames, master)
        assert cost["n_frames"] == 4
        assert cost["shared_palette"]["mean_of_frame_means"] >= 0
        assert cost["shared_palette"]["max_per_pixel"] >= 0
        assert "ADAPTIVE" in cost["per_frame_adaptive"]["what_it_is"]
        assert "2026-08-03 deliverable" in cost["per_frame_adaptive"]["what_it_is"]
        # the cost is a SUBTRACTION of the two measured arms, not an estimate
        assert cost["cost_of_sharing"]["extra_mean_per_pixel_error"] == pytest.approx(
            cost["shared_palette"]["mean_of_frame_means"]
            - cost["per_frame_adaptive"]["mean_of_frame_means"],
            abs=1e-12,
        )

    def test_a_sequence_the_palette_covers_exactly_costs_nothing(self, tmp_path):
        paths = [
            _flat_png(tmp_path / f"f{i}.png", (8, 8), c)
            for i, c in enumerate([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        ]
        master = anim.build_master_palette(paths)
        cost = anim.measure_palette_cost(paths, master)
        assert cost["shared_palette"]["mean_of_frame_means"] == 0.0
        assert cost["shared_palette"]["max_per_pixel"] == 0.0
        assert cost["shared_palette"]["fraction_of_pixels_changed"] == 0.0

    def test_the_headroom_says_gif_has_no_more_colours_and_names_the_alternatives(
        self, tmp_path
    ):
        frames = _noisy_sequence(tmp_path, n=3)
        master = anim.build_master_palette(frames)
        headroom = anim.describe_palette_headroom(master)
        assert headroom["palette_entries_available"] == 256
        assert headroom["colours_per_entry"] == pytest.approx(
            master.n_distinct_in_sequence / master.n_from_quantizer
        )
        alternatives = headroom["if_256_is_not_enough"]
        assert "PNG SEQUENCE" in alternatives
        assert "LOSSLESS" in alternatives and "WebP" in alternatives
        assert "MP4" in alternatives
        assert "not available" in headroom["gif_ceiling"]

    def test_the_metric_is_named_so_the_number_can_be_reproduced(self, tmp_path):
        frames = _noisy_sequence(tmp_path, n=2)
        master = anim.build_master_palette(frames)
        cost = anim.measure_palette_cost(frames, master)
        assert "Euclidean" in cost["metric"] and "441.67" in cost["metric"]
        assert cost["dither"] == "NONE"


# ---------------------------------------------------------------------------
class TestTheManifestIdentifiesTheLevelCurves:
    """Two of the three basin surfaces come from CSVs that were never named."""

    def test_three_basins_are_accounted_for_and_only_two_have_a_curve(self):
        entries = anim.level_curve_provenance()
        assert {e["basin"] for e in entries} == {
            "caspian", "pontic", "mediterranean",
        }
        with_files = [e for e in entries if e["path"]]
        assert {e["basin"] for e in with_files} == {"caspian", "pontic"}
        third = next(e for e in entries if e["basin"] == "mediterranean")
        assert third["status"].startswith("NOT_A_CURVE")
        assert "section 2" in third["status"]

    def test_the_two_curves_name_the_constant_that_points_at_them(self):
        entries = {e["basin"]: e for e in anim.level_curve_provenance()}
        assert entries["caspian"]["constant"] == "bake_khvalynian.CASPIAN_CSV"
        assert entries["pontic"]["constant"] == "bake_black_sea.PONTIC_CSV"
        assert "comment='#'" in entries["pontic"]["read_with"]

    @_needs_curves
    def test_both_curves_are_pathed_and_hashed(self):
        import hashlib

        for entry in anim.level_curve_provenance():
            if not entry["path"]:
                continue
            assert entry["status"] == "FINGERPRINTED"
            assert len(entry["sha256"]) == 64
            data = Path(entry["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            assert entry["bytes"] == len(data)

    def test_a_curve_that_is_not_there_is_reported_rather_than_skipped(self):
        missing = anim._fingerprint_curve(
            "caspian", "role", "const", Path("no-such-curve.csv"), "read_csv"
        )
        assert missing.sha256 is None
        assert missing.status.startswith("MISSING")
        assert "cannot be checked" in missing.status

    def test_the_curves_travel_into_the_manifest(self, tmp_path, untagged_topo_dir):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        basins = {e["basin"] for e in manifest["level_curves"]}
        assert basins == {"caspian", "pontic", "mediterranean"}


# ---------------------------------------------------------------------------
class TestARasterPathPointsAtWhereTheRasterLives:
    """Every path in the 2026-08-03 manifest was inside a discarded scratchpad."""

    def test_a_raster_read_from_the_canonical_store_needs_no_matching(self):
        resolution = anim.resolve_raster_source(
            anim.CANONICAL_TOPO_DIR / "khv_16.5ka.tif", "0" * 64
        )
        assert resolution["status"] == "CANONICAL"
        assert resolution["source_path"].endswith("khv_16.5ka.tif")

    def test_a_mirror_path_is_resolved_to_its_source_by_sha256(self, tmp_path):
        records = [
            {
                "name": "khv_16.5ka.tif",
                "sha256": "ab" * 32,
                "source": r"Y:\paleo\renders\khvalynian_caspian\topo\khv_16.5ka.tif",
                "mirror": str(tmp_path / "mirror" / "khv_16.5ka.tif"),
            }
        ]
        resolution = anim.resolve_raster_source(
            tmp_path / "mirror" / "khv_16.5ka.tif",
            "ab" * 32,
            mirror_records=records,
            mirror_files=["sidecar.json"],
        )
        assert resolution["status"] == "MIRRORED"
        assert resolution["source_path"] == records[0]["source"]
        assert "sha256" in resolution["matched_by"]
        assert "%TEMP%" not in resolution["source_path"]

    def test_a_matching_name_is_not_accepted_as_evidence(self, tmp_path):
        canonical = tmp_path / "store"
        canonical.mkdir()
        (canonical / "khv_16.5ka.tif").write_bytes(b"the real one")
        elsewhere = tmp_path / "scratch" / "khv_16.5ka.tif"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"a different scene under the same name")
        resolution = anim.resolve_raster_source(
            elsewhere, anim.sha256_of(elsewhere), canonical_dir=canonical
        )
        assert resolution["status"] == "UNRESOLVED"
        assert resolution["source_path"] is None
        assert "a NAME is not a fingerprint" in resolution["evidence"]

    def test_re_hashing_the_store_resolves_it_and_a_conflict_is_reported(
        self, tmp_path
    ):
        canonical = tmp_path / "store"
        canonical.mkdir()
        (canonical / "khv_16.5ka.tif").write_bytes(b"identical bytes")
        twin = tmp_path / "mirror" / "khv_16.5ka.tif"
        twin.parent.mkdir()
        twin.write_bytes(b"identical bytes")
        good = anim.resolve_raster_source(
            twin, anim.sha256_of(twin), canonical_dir=canonical,
            verify_canonical_sha256=True,
        )
        assert good["status"] == "CANONICAL_VERIFIED"
        assert good["source_path"] == str((canonical / "khv_16.5ka.tif").resolve())

        twin.write_bytes(b"a re-bake, in place, under the same name")
        conflict = anim.resolve_raster_source(
            twin, anim.sha256_of(twin), canonical_dir=canonical,
            verify_canonical_sha256=True,
        )
        assert conflict["status"] == "CONFLICT"
        assert conflict["source_path"] is None
        assert "recorded rather than reconciled" in conflict["evidence"]

    def test_a_mirror_sidecar_is_read_from_disk(self, tmp_path):
        (tmp_path / "raster_mirror_provenance.json").write_text(
            json.dumps({"files": [
                {"sha256": "cd" * 32, "source": r"Y:\paleo\x.tif",
                 "mirror": r"C:\tmp\x.tif"},
                {"no_sha": True},
            ]}),
            encoding="utf-8",
        )
        records, files = anim.load_mirror_records(tmp_path)
        assert len(records) == 1
        assert len(files) == 1 and files[0].endswith("raster_mirror_provenance.json")

    def test_a_missing_sidecar_is_not_an_error(self, tmp_path):
        records, files = anim.load_mirror_records(tmp_path / "nowhere")
        assert records == () and files == ()

    def test_the_manifest_records_both_the_path_read_and_the_source_path(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        summary = manifest["raster_source_resolution"]
        assert "canonical_dir" in summary
        assert "A file of the same NAME is not accepted as evidence" in summary["rule"]
        assert "%TEMP%" in summary["why_it_exists"]
        frame = manifest["frames"][0]
        assert set(frame["raster"]) >= {"path_as_read", "source_path", "sha256"}


# ---------------------------------------------------------------------------
class TestEveryFrameCarriesASillVerdictAsValues:
    """No sill verdict was recorded anywhere; it was inferred from an exception."""

    _FRAME = "F0 present-day datum (H_modern + dz_solid)"

    def _plan(self):
        return anim.EpochPlan(
            epoch_ka=16.5,
            raster_path="khv_16.5ka.tif",
            raster_sha256="00" * 32,
            dem_frame=self._FRAME,
            dz_bosphorus_m=10.818970,
        )

    def _patch(self, monkeypatch, value_m=27.701019287109375):
        monkeypatch.setattr(
            anim, "manych_expectation",
            lambda epoch, frame: SillExpectation(
                value_m=value_m, frame=frame, source="test", measured_by="test",
                lon=44.102083, lat=45.706250,
            ),
        )

    def test_a_saddle_inside_the_tolerance_reads_within(self, monkeypatch):
        self._patch(monkeypatch)
        basins = {"sills": [{"pair": ["caspian", "pontic"],
                             "elevation_m": 28.000944137573242,
                             "expectation_status": "CHECKED: ...",
                             "smoothing_shift_m": 0.24988174438476562}]}
        block = anim._sill_block(self._plan(), basins, tolerance_m=2.0,
                                 frame_rendered=True)
        entry = next(e for e in block if e["pair"] == ["caspian", "pontic"])
        assert entry["verdict"] == "WITHIN_TOLERANCE"
        assert entry["expected_m"] == pytest.approx(27.701019, abs=1e-6)
        assert entry["measured_saddle_m"] == pytest.approx(28.000944, abs=1e-6)
        assert entry["gap_m"] == pytest.approx(0.299925, abs=1e-6)
        assert entry["tolerance_m"] == 2.0
        assert "0.300 m <= 2.00 m" in entry["verdict_why"]

    def test_a_saddle_outside_the_tolerance_reads_outside(self, monkeypatch):
        self._patch(monkeypatch)
        basins = {"sills": [{"pair": ["caspian", "pontic"], "elevation_m": 36.005}]}
        block = anim._sill_block(self._plan(), basins, tolerance_m=2.0,
                                 frame_rendered=True)
        entry = next(e for e in block if e["pair"] == ["caspian", "pontic"])
        assert entry["verdict"] == "OUTSIDE_TOLERANCE"
        assert entry["gap_m"] == pytest.approx(8.303981, abs=1e-5)

    def test_a_refused_frame_records_the_expectation_and_says_nothing_was_measured(
        self, monkeypatch
    ):
        """An unrun guard is not a passed guard, and now says so in the data."""
        self._patch(monkeypatch)
        block = anim._sill_block(
            self._plan(), {}, tolerance_m=2.0,
            frame_refusal="UnseededWaterBodyError: 181 water bodies ...",
        )
        entry = next(e for e in block if e["pair"] == ["caspian", "pontic"])
        assert entry["verdict"] == "NOT_MEASURED"
        assert entry["measured_saddle_m"] is None
        assert entry["gap_m"] is None
        assert entry["expected_m"] == pytest.approx(27.701019, abs=1e-6)
        assert "did not render" in entry["verdict_why"]
        assert "An unrun guard is NOT a passed guard" in entry["verdict_why"]

    def test_a_saddle_above_the_flood_ceiling_is_not_reached_never_agrees(
        self, monkeypatch
    ):
        self._patch(monkeypatch)
        basins = {"sills": [{"pair": ["caspian", "pontic"], "elevation_m": None}]}
        block = anim._sill_block(self._plan(), basins, tolerance_m=2.0,
                                 frame_rendered=True)
        entry = next(e for e in block if e["pair"] == ["caspian", "pontic"])
        assert entry["verdict"] == "NOT_MEASURED"
        assert "'Not reached' is not 'agrees'" in entry["verdict_why"]

    def test_a_plan_only_manifest_does_not_claim_the_frame_rendered(self, monkeypatch):
        """Three different reasons for 'no saddle', and they must not be collapsed."""
        self._patch(monkeypatch)
        block = anim._sill_block(self._plan(), {}, tolerance_m=2.0)
        entry = next(e for e in block if e["pair"] == ["caspian", "pontic"])
        assert entry["verdict"] == "NOT_MEASURED"
        assert "no render was attempted" in entry["verdict_why"]
        assert "the frame rendered" not in entry["verdict_why"]

    def test_a_declared_pair_reads_not_checked_and_carries_its_reason(
        self, monkeypatch
    ):
        self._patch(monkeypatch)
        block = anim._sill_block(self._plan(), {}, tolerance_m=2.0)
        for entry in block:
            if entry["expectation_kind"] == "UnmeasuredSill":
                assert entry["verdict"] == "NOT_CHECKED"
                assert entry["expected_m"] is None
                assert len(entry["unmeasured_reason"]) > 200

    def test_all_three_pairs_are_present_even_on_a_frame_that_refused(
        self, monkeypatch
    ):
        """Eight of seventeen frames used to carry an EMPTY sill list."""
        self._patch(monkeypatch)
        block = anim._sill_block(
            self._plan(), {}, tolerance_m=2.0, frame_refusal="CurveAgeGapError: ...",
        )
        assert [e["pair"] for e in block] == [
            ["caspian", "mediterranean"], ["caspian", "pontic"],
            ["mediterranean", "pontic"],
        ]
        assert all("verdict" in e for e in block)

    def test_the_rule_says_a_verdict_is_never_inferred_from_an_exception(self):
        assert "inferred from an exception type" in anim.SILL_VERDICT_RULE
        assert "compatible with the guard never having run" in anim.SILL_VERDICT_RULE

    def test_the_summary_tallies_the_verdicts_by_pair(self):
        frames = [
            {"sill": [{"pair": ["caspian", "pontic"], "verdict": "NOT_MEASURED"}]},
            {"sill": [{"pair": ["caspian", "pontic"], "verdict": "WITHIN_TOLERANCE"}]},
            {"sill": [{"pair": ["caspian", "pontic"], "verdict": "NOT_MEASURED"}]},
        ]
        summary = anim._sill_verdict_summary(frames)
        assert summary["by_pair"]["caspian|pontic"] == {
            "NOT_MEASURED": 2, "WITHIN_TOLERANCE": 1,
        }
        assert summary["n_frames"] == 3


# ---------------------------------------------------------------------------
class TestTheBosphorusDisclosureCountIsCounted:
    """It said "disclosed twice" and named a field that exists in no artifact."""

    def test_the_reason_no_longer_claims_a_number_of_disclosures_it_cannot_know(self):
        reason = anim.bosphorus_unmeasured(16.5, 10.818970).reason
        assert "disclosed twice" not in reason
        assert "THE NUMBER IS CARRIED HERE" in reason
        assert "smoothing_shift_disclosed_in" in reason

    def test_the_reason_does_not_name_a_field_no_artifact_has(self):
        """``sill_expectation_status`` is a BasinField attribute, not a JSON key."""
        reason = anim.bosphorus_unmeasured(16.5, 10.818970).reason
        assert "sill_expectation_status" not in reason

    def test_a_refused_frame_discloses_the_shift_once(self, monkeypatch):
        monkeypatch.setattr(
            anim, "manych_expectation",
            lambda epoch, frame: SillExpectation(
                value_m=27.7, frame=frame, source="t", measured_by="t",
                lon=44.1, lat=45.7,
            ),
        )
        plan = anim.EpochPlan(
            epoch_ka=16.5, dem_frame="F0 present-day datum (H_modern + dz_solid)",
            dz_bosphorus_m=10.818970,
        )
        block = anim._sill_block(plan, {}, tolerance_m=2.0, frame_refusal="X: y")
        entry = next(e for e in block if e["pair"] == ["mediterranean", "pontic"])
        assert entry["smoothing_shift_disclosed_in"] == ["unmeasured_reason"]

    def test_a_rendered_frame_discloses_it_twice(self, monkeypatch):
        monkeypatch.setattr(
            anim, "manych_expectation",
            lambda epoch, frame: SillExpectation(
                value_m=27.7, frame=frame, source="t", measured_by="t",
                lon=44.1, lat=45.7,
            ),
        )
        plan = anim.EpochPlan(
            epoch_ka=16.5, dem_frame="F0 present-day datum (H_modern + dz_solid)",
            dz_bosphorus_m=10.818970,
        )
        basins = {"sills": [{"pair": ["mediterranean", "pontic"],
                             "elevation_m": None, "smoothing_shift_m": 22.86}]}
        block = anim._sill_block(plan, basins, tolerance_m=2.0, frame_rendered=True)
        entry = next(e for e in block if e["pair"] == ["mediterranean", "pontic"])
        assert entry["smoothing_shift_disclosed_in"] == [
            "unmeasured_reason", "basins_sidecar.sills[].smoothing_shift_m",
        ]


# ---------------------------------------------------------------------------
class TestTheCliRefusesAGridThatCannotResolveTheSill:

    def test_a_sub_native_max_px_is_refused_without_the_explicit_flag(self):
        with pytest.raises(SystemExit):
            anim._cli(["--max-px", "2000", "--plan-only"])

    def test_the_native_grid_is_the_default(self):
        assert anim.NATIVE_MAX_PX == 7800


# ---------------------------------------------------------------------------
class TestTwoGridsNeitherPresentedAsTheOther:
    """The 0.5 ka deliverable must not be able to call itself a 0.25 ka animation."""

    def test_the_half_ka_grid_is_nine_epochs_and_a_subset_of_the_bakes_own(self):
        from scripts import bake_khvalynian as bk

        nine = anim.epoch_grid(step_ka=anim.HALF_KA_STEP)
        assert nine == (17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0)
        assert set(nine) <= set(bk.EPOCHS_KA), (
            "a coarser grid must SELECT from the baked epochs, never generate one"
        )
        assert set(nine) <= set(anim.epoch_grid())
        assert {round(a - b, 10) for a, b in zip(nine, nine[1:])} == {0.5}

    def test_a_step_the_baked_grid_does_not_divide_is_refused(self):
        """A '0.3 ka grid' would be a caption that lies about its own contents."""
        with pytest.raises(ValueError) as excinfo:
            anim.epoch_grid(step_ka=0.3)
        assert "not a uniform" in str(excinfo.value)
        with pytest.raises(ValueError):
            anim.epoch_grid(step_ka=-0.5)

    def test_the_grid_block_names_the_eight_it_does_not_carry_and_why(self):
        grid = anim.grid_description(
            anim.epoch_grid(step_ka=0.5), step_ka=0.5, companion="other.json"
        )
        assert grid["step_ka"] == 0.5
        assert grid["n_slots"] == 9
        assert grid["quarter_ka_epochs_not_in_this_sequence"] == [
            16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25
        ]
        assert "CurveAgeGapError" in grid["why_those_are_absent"]
        assert "-58.3" in grid["why_those_are_absent"]
        assert "-30.0" in grid["why_those_are_absent"]
        assert grid["companion_sequence"] == "other.json"
        assert grid["do_not_call_this"] == "a 0.25 ka animation"

    def test_the_seventeen_slot_grid_omits_nothing_and_says_so(self):
        """'8 of these 17 slots are refusals' must be COUNTED, not typed."""
        grid = anim.grid_description(anim.epoch_grid(), step_ka=0.25, n_rendered=9)
        assert grid["n_slots"] == 17
        assert grid["quarter_ka_epochs_not_in_this_sequence"] == []
        assert "carries every epoch" in grid["why_those_are_absent"]
        assert grid["n_rendered_in_this_sequence"] == 9
        assert "8 of these 17 slots are refusals" in grid["do_not_call_this"]

        # the same grid with a different outcome must not repeat 9 and 8
        other = anim.grid_description(anim.epoch_grid(), step_ka=0.25, n_rendered=17)
        assert "0 of these 17 slots are refusals" in other["do_not_call_this"]
        unknown = anim.grid_description(anim.epoch_grid(), step_ka=0.25)
        assert unknown["n_rendered_in_this_sequence"] is None
        assert "was not available" in unknown["do_not_call_this"]

    def test_the_manifest_headline_names_the_step_it_was_built_on(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            epochs=anim.epoch_grid(step_ka=0.5),
            grid_step_ka=0.5,
            fallback_canvas_px=(120, 90),
        )
        assert manifest["grid"]["step_ka"] == 0.5
        assert manifest["grid"]["n_slots"] == 9
        assert "0.5 ka grid" in manifest["what"]
        assert manifest["n_frames"] == 9


# ---------------------------------------------------------------------------
class TestTheLimitsSayWhichEpochsTheyCover:
    """The 2026-08-03 limits described 9 epochs beside a 17-slot sequence, silently."""

    class _Limits:
        def __init__(self, epochs):
            self.epochs = tuple(epochs)

    def _plans(self, epochs):
        return [anim.EpochPlan(epoch_ka=e) for e in epochs]

    def test_exact_when_the_limits_epochs_are_the_drawn_epochs(self):
        cover = anim._limits_cover(
            self._Limits([17.0, 16.5]),
            self._plans([17.0, 16.75, 16.5]),
            [
                {"epoch_ka": 17.0, "png": "a.png"},
                {"epoch_ka": 16.75, "png": None},
                {"epoch_ka": 16.5, "png": "b.png"},
            ],
        )
        assert cover["verdict"].startswith("EXACT")
        assert cover["n_slots_in_the_sequence"] == 3
        assert cover["slots_carrying_no_frame"] == [16.75]
        assert cover["epochs_in_the_limits_that_produced_no_frame"] == []

    def test_wider_than_drawn_when_a_planned_epoch_refused_in_the_renderer(self):
        cover = anim._limits_cover(
            self._Limits([17.0, 16.5]),
            self._plans([17.0, 16.5]),
            [
                {"epoch_ka": 17.0, "png": "a.png"},
                {"epoch_ka": 16.5, "png": None, "refusal": "boom"},
            ],
        )
        assert cover["verdict"].startswith("WIDER THAN DRAWN")
        assert cover["epochs_in_the_limits_that_produced_no_frame"] == [16.5]
        assert "can only widen a ramp" in (
            cover["direction_of_the_error_when_it_is_not_EXACT"]
        )

    def test_a_frame_drawn_for_an_epoch_the_limits_never_saw_is_a_disagreement(self):
        cover = anim._limits_cover(
            self._Limits([17.0]),
            self._plans([17.0, 16.5]),
            [
                {"epoch_ka": 17.0, "png": "a.png"},
                {"epoch_ka": 16.5, "png": "b.png"},
            ],
        )
        assert cover["verdict"].startswith("DISAGREEMENT")
        assert cover["epochs_drawn_that_the_limits_never_measured"] == [16.5]

    def test_it_reaches_the_manifest_and_the_sidecar_file(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        cover = manifest["sequence_limits_cover"]
        assert cover["verdict"].startswith("NO_LIMITS")
        assert cover["n_slots_in_the_sequence"] == 17


# ---------------------------------------------------------------------------
class TestTheAnalyticShadeCeilingIsArithmeticNotData:
    """``shade_max`` is +1.0 by geometry, and native terrain attains it."""

    def _record(self, epoch, **by_quantity):
        return {
            "epoch_ka": epoch,
            "png": f"{epoch}.png",
            "palette_sidecar": {"out_of_range": {"by_quantity": dict(by_quantity)}},
        }

    def test_the_budget_is_the_arithmetics_own_precision(self):
        assert anim.SHADE_EXCURSION_BUDGET == float(np.finfo(np.float32).eps)
        assert anim.CLIP_POLICY == "mark"
        assert "shade_below_shade_min" in anim.MEASURED_LIMIT_QUANTITIES
        assert anim.ARITHMETIC_LIMIT_QUANTITIES == ("shade_above_shade_max",)

    def test_a_measured_limit_breach_stops_the_run_exactly_as_raise_would(self):
        with pytest.raises(anim.ClipPolicyBreachError) as excinfo:
            anim.assert_only_the_analytic_shade_ceiling_was_marked(
                [
                    self._record(16.5, shade_above_shade_max=1),
                    self._record(16.0, depth_above_depth_data_max_m=7),
                ]
            )
        assert "16 ka: depth_above_depth_data_max_m=7" in str(excinfo.value)

    def test_a_cell_below_the_analytic_floor_is_treated_as_data_not_rounding(self):
        """-cos(45 deg) is an infimum nothing attains; rounding cannot reach it."""
        with pytest.raises(anim.ClipPolicyBreachError) as excinfo:
            anim.assert_only_the_analytic_shade_ceiling_was_marked(
                [self._record(15.0, shade_below_shade_min=1)]
            )
        assert "shade_below_shade_min=1" in str(excinfo.value)

    def test_a_marked_ceiling_with_no_measurement_beside_it_is_refused(self):
        with pytest.raises(anim.ShadeExcursionError) as excinfo:
            anim.assert_only_the_analytic_shade_ceiling_was_marked(
                [self._record(16.5, shade_above_shade_max=1)]
            )
        assert "unmeasured excursion is not a small one" in str(excinfo.value)

    def test_an_excursion_larger_than_the_arithmetic_can_explain_is_refused(self):
        record = self._record(16.5, shade_above_shade_max=4)
        record["shade_excursion"] = {
            "status": "MEASURED",
            "within_budget": False,
            "excursion_above_ceiling": 1e-3,
            "excursion_in_float32_eps": 1e-3 / anim.SHADE_EXCURSION_BUDGET,
        }
        with pytest.raises(anim.ShadeExcursionError) as excinfo:
            anim.assert_only_the_analytic_shade_ceiling_was_marked([record])
        assert "float32 epsilon" in str(excinfo.value)

    def test_a_measured_in_budget_excursion_is_allowed_and_counted(self):
        record = self._record(16.5, shade_above_shade_max=1)
        record["shade_excursion"] = {
            "status": "MEASURED",
            "within_budget": True,
            "excursion_above_ceiling": 2.9629851017176634e-08,
            "excursion_in_float32_eps": 0.2486,
        }
        clean = self._record(16.0)
        clean["shade_excursion"] = {"status": "MEASURED", "within_budget": True}
        tally = anim.assert_only_the_analytic_shade_ceiling_was_marked([record, clean])
        assert tally["total_cells_marked"] == 1
        assert tally["cells_marked_at_the_analytic_ceiling_by_epoch"] == {
            "16.5": 1, "16": 0
        }
        assert tally["policy"] == "mark"
        assert tally["budget_float32_eps"] == anim.SHADE_EXCURSION_BUDGET

    def test_a_frame_that_did_not_render_contributes_nothing(self):
        tally = anim.assert_only_the_analytic_shade_ceiling_was_marked(
            [{"epoch_ka": 16.75, "png": None, "refusal": "CurveAgeGapError: ..."}]
        )
        assert tally["total_cells_marked"] == 0
        assert tally["per_frame"] == []

    def test_the_excursion_is_measured_on_a_raster_and_reports_both_bounds(
        self, toy_raster
    ):
        got = anim.measure_analytic_shade_excursion(
            toy_raster, config=RenderConfig(max_px=200)
        )
        assert got["status"] == "MEASURED"
        assert got["shade_max_analytic"] == 1.0
        assert got["shade_min_analytic"] == pytest.approx(-(0.5 ** 0.5), abs=1e-12)
        assert got["ice_cells"] == 0
        assert got["excursion_above_ceiling"] >= 0.0
        assert got["clearance_below_floor"] >= 0.0
        assert got["budget"] == anim.SHADE_EXCURSION_BUDGET
        assert got["raw_intensity_max"] <= 1.0 + anim.SHADE_EXCURSION_BUDGET

    def test_it_refuses_to_report_a_number_it_cannot_stand_behind(self, tmp_path):
        """With ice the renderer draws a dome blend this function does not model."""
        raster = _write_scene_raster(tmp_path / "khv_16.5ka.tif", deeps=_TOY_DEEPS)
        _write_scene_raster(tmp_path / "khv_16.5ka_ice.tif", background=900.0)
        got = anim.measure_analytic_shade_excursion(
            raster, config=RenderConfig(max_px=200)
        )
        assert got["status"] == "NOT_COMPARABLE"
        assert got["ice_cells"] > 0
        assert "ice-dome blend" in got["why"]
        assert "excursion_above_ceiling" not in got


# ---------------------------------------------------------------------------
class TestTheSubsequenceIsProvedRatherThanAssumed:
    """The deliverable re-encodes the other run's frames; it never re-renders them."""

    def _source_run(self, tmp_path, epochs=(17.0, 16.5, 16.0), refuse=()):
        root = tmp_path / "source"
        (root / "canvas").mkdir(parents=True, exist_ok=True)
        frames = []
        for slot, epoch in enumerate(epochs):
            if epoch in refuse:
                frames.append(
                    {
                        "slot": slot, "epoch_ka": epoch, "png": None,
                        "refusal": "CurveAgeGapError: no row",
                        "canvas_frame": None, "canvas_frame_sha256": None,
                        "sill": [],
                    }
                )
                continue
            png = _noisy_png(
                root / "canvas" / f"frame_{slot:02d}_{epoch:g}ka.png",
                size_px=(48, 36), seed=slot,
            )
            frames.append(
                {
                    "slot": slot, "epoch_ka": epoch, "png": str(png),
                    "refusal": None,
                    "canvas_frame": str(png),
                    "canvas_frame_sha256": anim.sha256_of(png),
                    "basin_levels": {"pontic": {"level_m": -18.0204}},
                    "sill": [{"pair": ["caspian", "pontic"],
                              "verdict": "WITHIN_TOLERANCE"}],
                }
            )
        manifest = {
            "what": "source", "frames": frames, "epochs": list(epochs),
            "n_frames": len(epochs), "grid": {"step_ka": 0.25, "n_slots": len(epochs)},
            "sequence_limits": {"epochs": [e for e in epochs if e not in refuse]},
            "sequence_limits_cover": {"verdict": "EXACT"},
            "level_curves": [{"basin": "pontic"}],
        }
        (root / "sequence_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return root

    def test_it_re_encodes_the_same_bytes_and_records_both_hashes(self, tmp_path):
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.0], grid_step_ka=0.5
        )
        assert manifest["n_frames"] == 2 and manifest["n_refused"] == 0
        assert [f["epoch_ka"] for f in manifest["frames"]] == [17.0, 16.0]
        assert [f["slot"] for f in manifest["frames"]] == [0, 1]
        for row in manifest["derived_from"]["what_was_re_encoded"]:
            assert (
                row["sha256_recorded_by_the_source_run"]
                == row["sha256_reverified_here"]
            )
        assert manifest["derived_from"]["source_manifest_sha256"] == anim.sha256_of(
            source / "sequence_manifest.json"
        )
        assert Path(manifest["animation"]["gif"]["as_written"]).exists()
        assert manifest["animation"][
            "gif_frame_durations_ms_read_back_with_pil"
        ] == [250, 250]
        assert manifest["animation"]["gif_palette_verdict"] == "ONE_GLOBAL_PALETTE"
        assert manifest["animation"]["webp_verdict"] == "BIT_EXACT_24_BIT"
        assert manifest["animation"]["webp_delay_asserted"] is True
        assert manifest["animation"][
            "webp_frame_durations_ms_read_back_with_pil"
        ] == [250, 250]

    def test_the_deliverable_carries_the_frames_it_is_made_of(self, tmp_path):
        """Copying the directory alone used to lose every source image."""
        source = self._source_run(tmp_path)
        out = tmp_path / "out"
        manifest = anim.write_subsequence(
            source, out, keep=[17.0, 16.0], grid_step_ka=0.5
        )
        block = manifest["self_contained"]
        assert block["verdict"] == "SELF_CONTAINED_FOR_THE_ANIMATION"
        assert block["files_copied"] == 2
        assert block["bytes_copied"] > 0
        for frame in manifest["frames"]:
            local = Path(frame["canvas_frame"])
            assert local.exists()
            assert local.parent == out / "canvas", "the frame lives in the deliverable"
            assert anim.sha256_of(local) == frame["canvas_frame_sha256"]
            assert Path(
                frame["canvas_frame_in_the_source_run"]["as_written"]
            ).parent == source / "canvas"
        # and what is NOT copied is named, with where it lives
        assert "frames/" in block["what_is_still_external"]
        assert "derived_from.source_run" in block["what_is_still_external"]
        assert block["source_run"]["as_written"] == str(source)

    def test_a_copy_whose_bytes_do_not_match_is_refused(self, tmp_path, monkeypatch):
        """The copy is checked against the SOURCE RUN's hash, not against itself."""
        import shutil as _shutil

        source = self._source_run(tmp_path)
        original = _shutil.copyfile

        def _corrupting_copy(src, dst, *args, **kwargs):
            original(src, dst)
            Path(dst).write_bytes(Path(dst).read_bytes() + b"\x00")
            return dst

        monkeypatch.setattr(_shutil, "copy2", _corrupting_copy)
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(
                source, tmp_path / "out", keep=[17.0], grid_step_ka=0.5
            )
        assert "hashes to" in str(excinfo.value)
        assert "is NOT encoded" in str(excinfo.value)

    def test_a_count_over_the_sources_slots_is_recounted_for_this_file(self, tmp_path):
        """A "17" inside a 9-frame manifest describes a different file."""
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.5], grid_step_ka=0.5
        )
        assert manifest["sequence_limits_cover"]["n_slots_in_the_sequence"] == 2
        assert manifest["sequence_limits_cover"]["slots"] == [17.0, 16.5]
        assert manifest["sequence_limits_cover"]["slots_carrying_no_frame"] == []
        assert "recounted for this file" in (
            manifest["sequence_limits_cover"]["note_for_this_subsequence"]
        )
        resolution = manifest["raster_source_resolution"]
        assert sum(resolution["by_status"].values()) == 2
        assert "recounted from their own per-frame records" in (
            resolution["counted_over"]
        )

    def test_the_per_frame_blocks_are_carried_not_re_derived(self, tmp_path):
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[16.5], grid_step_ka=0.5
        )
        frame = manifest["frames"][0]
        assert frame["basin_levels"] == {"pontic": {"level_m": -18.0204}}
        assert frame["sill"][0]["verdict"] == "WITHIN_TOLERANCE"
        assert manifest["sill_verdicts"]["by_pair"] == {
            "caspian|pontic": {"WITHIN_TOLERANCE": 1}
        }
        assert manifest["level_curves"] == [{"basin": "pontic"}]

    def test_the_palette_is_re_derived_over_the_subset(self, tmp_path):
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.5], grid_step_ka=0.5
        )
        assert manifest["palette"]["gif_audit"][
            "n_frames_with_a_local_color_table"
        ] == 0
        assert manifest["palette"]["gif_pixel_audit"][
            "n_frames_decoded_and_compared"
        ] == 2

    def test_the_open_quantizer_question_is_re_measured_on_these_frames(
        self, tmp_path
    ):
        """Every arm is re-measured on the frames the decision is recorded beside."""
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.5, 16.0], grid_step_ka=0.5
        )
        comparison = manifest["palette"]["quantizer_comparison"]
        assert comparison["shipped"] == anim.PALETTE_METHOD
        assert comparison["n_frames"] == 3
        assert set(comparison["arms"]) == {"MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE"}
        for name, arm in comparison["arms"].items():
            assert arm["status"] == "MEASURED", f"{name}: {arm}"
            assert arm["mean_of_frame_means"] >= 0.0
            assert 0.0 <= arm["max_per_pixel"] <= 441.68
        assert comparison["lowest_mean_error"] in comparison["arms"]
        assert comparison["lowest_worst_pixel"] in comparison["arms"]

    def test_the_decision_is_taken_and_recorded_beside_the_arms(self, tmp_path):
        """An open decision in a shipped artifact is one the last reader takes."""
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.5, 16.0], grid_step_ka=0.5
        )
        comparison = manifest["palette"]["quantizer_comparison"]
        decision = comparison["decision"]
        assert "WORST per-pixel error decides" in decision["rule"]
        assert decision["shipped"] == anim.PALETTE_METHOD == "MAXCOVERAGE"
        # the selection is DERIVED from the arms in the same object
        arms = {
            name: arm for name, arm in comparison["arms"].items()
            if arm["status"] == "MEASURED"
        }
        assert decision["selected_by_the_rule"] == min(
            arms, key=lambda k: arms[k]["max_per_pixel"]
        )
        assert decision["verdict"] in (
            "AGREES_WITH_THE_SHIPPED_METHOD", "DISAGREES_WITH_THE_SHIPPED_METHOD"
        )
        assert decision["what_it_buys"]["arm"] == decision["selected_by_the_rule"]
        assert decision["what_it_costs"]["mean_per_channel_equivalent"] == (
            pytest.approx(
                decision["what_it_costs"]["mean_per_pixel_error"] / (3 ** 0.5)
            )
        )
        # the master really was built with the shipped arm
        assert manifest["palette"]["master"]["method"] == anim.PALETTE_METHOD

    def test_the_decision_contradicts_itself_out_loud_when_the_data_disagrees(self):
        """A constant the measurement no longer supports must not pass silently."""
        arms = {
            "arms": {
                "MEDIANCUT": {
                    "status": "MEASURED", "mean_of_frame_means": 1.0,
                    "max_per_pixel": 200.0,
                },
                "MAXCOVERAGE": {
                    "status": "MEASURED", "mean_of_frame_means": 2.0,
                    "max_per_pixel": 20.0,
                },
            },
            "n_frames": 2, "size": 256, "dither": "NONE",
        }
        agrees = anim.palette_method_decision(arms, shipped="MAXCOVERAGE")
        assert agrees["verdict"] == "AGREES_WITH_THE_SHIPPED_METHOD"
        assert agrees["what_it_buys"]["worst_pixel_tighter_by_factor"] == 10.0
        assert agrees["what_it_costs"]["extra_mean_per_pixel_error"] == 1.0

        disagrees = anim.palette_method_decision(arms, shipped="MEDIANCUT")
        assert disagrees["verdict"] == "DISAGREES_WITH_THE_SHIPPED_METHOD"
        assert disagrees["selected_by_the_rule"] == "MAXCOVERAGE"
        assert "not switched mid-run" in disagrees["when_this_disagrees"]

        assert anim.palette_method_decision({"arms": {}})["verdict"].startswith(
            "NOT_MEASURED"
        )

    def test_a_canvas_frame_that_changed_since_it_was_measured_is_refused(
        self, tmp_path
    ):
        source = self._source_run(tmp_path)
        target = json.loads(
            (source / "sequence_manifest.json").read_text()
        )["frames"][1]["canvas_frame"]
        _noisy_png(Path(target), size_px=(48, 36), seed=99)  # same size, other bytes
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(
                source, tmp_path / "out", keep=[16.5], grid_step_ka=0.5
            )
        assert "hashes to" in str(excinfo.value)
        assert "is NOT re-encoded" in str(excinfo.value)

    def test_a_slot_that_never_rendered_cannot_enter_an_all_imagery_sequence(
        self, tmp_path
    ):
        source = self._source_run(tmp_path, refuse=(16.5,))
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(
                source, tmp_path / "out", keep=[16.5], grid_step_ka=0.5
            )
        assert "produced no frame" in str(excinfo.value)

    def test_an_epoch_the_source_never_carried_is_refused_not_rendered(self, tmp_path):
        source = self._source_run(tmp_path)
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(
                source, tmp_path / "out", keep=[14.0], grid_step_ka=0.5
            )
        assert "carries no slot" in str(excinfo.value)
        assert "selects, it does not render" in str(excinfo.value)

    def test_a_subsequence_pointed_at_itself_is_refused(self, tmp_path):
        """It would copy each frame over itself and overwrite the manifest it reads."""
        source = self._source_run(tmp_path)
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(source, source, keep=[17.0], grid_step_ka=0.5)
        assert "same directory" in str(excinfo.value)

    def test_a_missing_source_manifest_is_refused(self, tmp_path):
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(tmp_path / "nothing", tmp_path / "out")
        assert "does not exist" in str(excinfo.value)

    def test_a_missing_canvas_file_is_refused(self, tmp_path):
        source = self._source_run(tmp_path)
        target = json.loads(
            (source / "sequence_manifest.json").read_text()
        )["frames"][0]["canvas_frame"]
        Path(target).unlink()
        with pytest.raises(anim.SubsequenceSourceError) as excinfo:
            anim.write_subsequence(
                source, tmp_path / "out", keep=[17.0], grid_step_ka=0.5
            )
        assert "does not exist" in str(excinfo.value)

    def test_the_subsequence_manifest_names_the_other_grid(self, tmp_path):
        source = self._source_run(tmp_path)
        manifest = anim.write_subsequence(
            source, tmp_path / "out", keep=[17.0, 16.0], grid_step_ka=0.5
        )
        assert manifest["grid"]["step_ka"] == 0.5
        assert manifest["grid"]["companion_sequence"].endswith(
            "sequence_manifest.json"
        )
        assert "EVERY ONE OF THEM REAL IMAGERY" in manifest["what"]
        assert manifest["derived_from"]["source_grid_step_ka"] == 0.25


# ---------------------------------------------------------------------------
@_needs_curves
class TestTheLevelsAreDerivedOrRefused:
    """Each of the three surfaces is a rule, and the Pontic's refuses eight times."""

    def test_the_mediterranean_is_minus_the_offset_and_says_which_contract_section(
        self,
    ):
        from scripts import bake_khvalynian as bk

        levels = anim.basin_levels_for(16.5)
        assert levels["mediterranean"].level_m == pytest.approx(
            -bk.sea_surface_offset(16.5), abs=1e-9
        )
        assert levels["mediterranean"].frame.startswith("F3")
        assert "section 2" in levels["mediterranean"].provenance
        assert "Marmara" in levels["mediterranean"].provenance  # the stated caveat

    def test_the_offset_matches_the_reference_table_to_its_own_last_digit(self):
        """Not to 0.005 m -- to 0.0005 m, which is what a 3-decimal figure means."""
        from scripts import bake_khvalynian as bk

        for epoch, printed in anim.OFFSET_REFERENCE:
            assert bk.sea_surface_offset(epoch) == pytest.approx(
                printed, abs=anim.OFFSET_REFERENCE_PRECISION_M
            )

    def test_the_mediterranean_level_is_the_negative_of_that_same_reference(self):
        """The cross-check lands on the sequence's own frames, not beside them."""
        for epoch, printed in anim.OFFSET_REFERENCE:
            if epoch not in anim.epoch_grid():
                continue
            levels = anim.basin_levels_for(epoch)
            assert levels["mediterranean"].level_m == pytest.approx(
                -printed, abs=anim.OFFSET_REFERENCE_PRECISION_M
            )

    def test_the_pontic_curve_has_no_row_at_the_eight_quarter_ka_epochs(self):
        refused = []
        for epoch in anim.epoch_grid():
            try:
                anim.basin_levels_for(epoch)
            except anim.BasinLevelUnavailableError as exc:
                assert "pontic" in str(exc)
                # the OTHER two resolved and the refusal carries them
                assert set(exc.resolved) == {"caspian", "mediterranean"}, (
                    "both other levels resolve at every epoch of this window and "
                    "must survive the pontic's refusal"
                )
                refused.append(epoch)
        assert refused == [16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25]

    def test_the_caspian_level_is_the_bakes_own_threshold(self):
        from scripts import bake_khvalynian as bk

        expected = bk.caspian_threshold(bk.load_curve(), 16.5, "defensible")
        assert anim.basin_levels_for(16.5)["caspian"].level_m == pytest.approx(
            expected.threshold_m, abs=1e-9
        )

    def test_the_manych_expectation_is_the_bakes_sill_path_f1(self):
        from scripts import bake_khvalynian as bk

        frame = "F0 present-day datum (H_modern + dz_solid)"
        got = anim.manych_expectation(16.5, frame)
        assert got.value_m == pytest.approx(27.701019287109375, abs=1e-6)
        assert got.value_m == pytest.approx(
            bk.sill_test(bk.load_curve(), 16.5, "defensible").sill_path_f1_m, abs=1e-9
        )
        assert got.frame == frame
        assert (got.lon, got.lat) == pytest.approx((44.102083, 45.706250), abs=1e-6)

    def test_the_bosphorus_dz_is_read_at_the_measured_crest_cell(self):
        assert anim.dz_at_bosphorus(16.5) == pytest.approx(10.818970, abs=1e-4)
        crest_f1 = anim.BOSPHORUS_CREST_F0_M + anim.dz_at_bosphorus(16.5)
        assert crest_f1 == pytest.approx(-9.181, abs=0.001)


# ---------------------------------------------------------------------------
class TestTheThirdStateSurvivesIntoTheManifest:
    """CONNECTED / UNOBSERVABLE / CLOSED, in the file a frame is audited FROM.

    The renderer distinguishes the three and writes all three into each frame's
    ``.basins.json``.  Until this revision :func:`build_manifest` copied only
    ``painted_water`` up out of that sidecar, so the manifest -- whose stated purpose
    is that any single frame can be audited from it without re-running anything --
    carried no way to tell a body this raster MEASURES to be closed from a body whose
    connectivity it cannot observe.  A state absent from the audit surface is
    indistinguishable from a state that does not exist, which is the substitution
    this phase exists to remove.
    """

    @staticmethod
    def _frame(epoch, *, connected, unobs, closed, flat=None, comps=None,
               drawn=None, classified=None, edges=("south",), png="f.png"):
        flat = sum((connected, unobs, closed)) if flat is None else flat
        comps = (1, 2, 3) if comps is None else comps
        return {
            "epoch_ka": epoch,
            "png": png,
            "spill": {
                "connectivity": {
                    "flat_mask_cells": flat,
                    "flat_components": sum(comps),
                    "connected": {"cells": connected, "components": comps[0],
                                  "km2": connected * 0.5},
                    "unobservable": {
                        "cells": unobs, "components": comps[1], "km2": unobs * 0.5,
                        "policy": "mark",
                        "by_edge": {e: {"cells": unobs // len(edges)} for e in edges},
                        "bodies": [
                            {"cells": unobs, "km2": unobs * 0.5,
                             "touched_edges": list(edges),
                             "surface_min_m": -77.9, "surface_max_m": -77.9,
                             "painted_at_basin": "mediterranean",
                             "lon": 34.5, "lat": 36.6}
                        ],
                    },
                    "closed": {"cells": closed, "components": comps[2],
                               "km2": closed * 0.5},
                },
                "unobservable_drawn": {
                    "cells": unobs if drawn is None else drawn,
                    "cells_classified": unobs if classified is None else classified,
                    "stripe_period_px": 30,
                },
            },
        }

    def test_the_three_states_partition_the_flat_mask(self):
        summary = anim.unobservable_summary(
            [self._frame(16.5, connected=8_436_006, unobs=64_117, closed=140_597)]
        )
        row = summary["per_epoch"][0]
        assert row["connected_cells"] + row["unobservable_cells"] + row[
            "closed_cells"
        ] == row["flat_mask_cells"]
        assert row["connected_components"] + row["unobservable_components"] + row[
            "closed_components"
        ] == row["flat_components"]
        assert row["partition_closes_on_cells"] is True
        assert row["partition_closes_on_components"] is True
        assert summary["partition_is_exact_on_every_frame"] is True
        assert summary["frames_where_the_partition_does_not_close"] == []

    def test_a_partition_that_does_not_close_is_named_not_averaged_away(self):
        broken = self._frame(16.0, connected=10, unobs=1, closed=1, flat=99)
        summary = anim.unobservable_summary(
            [self._frame(16.5, connected=10, unobs=1, closed=1), broken]
        )
        assert summary["partition_is_exact_on_every_frame"] is False
        named = summary["frames_where_the_partition_does_not_close"]
        assert [f["epoch_ka"] for f in named] == [16.0]
        assert named[0]["flat_mask_cells"] == 99
        assert named[0][
            "connected_plus_unobservable_plus_closed_cells"
        ] == 12

    def test_a_rendered_frame_with_no_connectivity_is_listed_not_skipped(self):
        blind = {"epoch_ka": 15.0, "png": "f.png", "spill": {}}
        refused = {"epoch_ka": 14.75, "png": None, "spill": {}}
        summary = anim.unobservable_summary(
            [self._frame(16.5, connected=10, unobs=1, closed=1), blind, refused]
        )
        assert summary["n_frames_with_a_measurement"] == 1
        assert summary["epochs_with_no_measurement"] == [15.0], (
            "a frame that rendered but reported nothing must be named; a summary "
            "over the frames that happened to report is a summary of another sequence"
        )

    def test_the_cells_are_never_added_across_epochs(self):
        """The same ground appears in every frame; a sum describes no surface."""
        summary = anim.unobservable_summary([
            self._frame(16.5, connected=10, unobs=100, closed=5),
            self._frame(16.0, connected=10, unobs=100, closed=5),
        ])
        assert summary["unobservable_cells_most"]["unobservable_cells"] == 100
        assert summary["unobservable_cells_fewest"]["unobservable_cells"] == 100
        assert summary["unobservable_cells_most"]["epoch_ka"] in (16.5, 16.0)
        assert "sum" not in json.dumps(
            {k: v for k, v in summary.items() if k.endswith(("_most", "_fewest"))}
        )

    def test_the_extremes_name_the_epoch_that_produced_them(self):
        summary = anim.unobservable_summary([
            self._frame(16.5, connected=10, unobs=64_117, closed=5),
            self._frame(13.0, connected=10, unobs=12, closed=5),
        ])
        assert summary["unobservable_cells_most"] == {
            "epoch_ka": 16.5, "unobservable_cells": 64_117
        }
        assert summary["unobservable_cells_fewest"] == {
            "epoch_ka": 13.0, "unobservable_cells": 12
        }

    def test_what_the_morphology_took_from_the_third_state_is_a_number(self):
        summary = anim.unobservable_summary([
            self._frame(16.5, connected=10, unobs=100, closed=5,
                        drawn=97, classified=100)
        ])
        row = summary["per_epoch"][0]
        assert row["cells_classified_as_unobservable"] == 100
        assert row["cells_drawn_as_unobservable"] == 97
        assert row["cells_the_morphology_took_from_the_third_state"] == 3

    def test_no_shipped_sentence_in_this_block_carries_a_digit(self):
        """Every number it ships is a FIELD, so a reader can check it in place."""
        summary = anim.unobservable_summary(
            [self._frame(16.5, connected=10, unobs=1, closed=1)]
        )
        for key in ("rule", "why_the_counts_are_not_summed",
                    "where_the_full_record_is"):
            assert not re.search(r"\d", summary[key]), (
                f"{key} embeds a digit; a sentence carrying a number the object "
                "beside it does not is exactly what this phase retracts"
            )
        for name in ("CONNECTIVITY_STATES_RULE", "CONNECTIVITY_NO_SUM_RULE"):
            assert not re.search(r"\d", getattr(anim, name)), name

    def test_the_rule_says_which_state_is_a_measurement_and_which_is_not(self):
        rule = anim.CONNECTIVITY_STATES_RULE
        assert "CLOSED is a MEASUREMENT" in rule
        assert "UNOBSERVABLE is the ABSENCE of one" in rule
        assert "neither water nor land" in rule
        assert "dropped" in rule, "the retired spelling must stay named"

    def test_the_manifest_frame_block_carries_both_new_keys(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        assert "connectivity_states" in manifest
        for frame in manifest["frames"]:
            assert "connectivity" in frame["spill"], (
                "build_manifest dropped the third state on the way up from the "
                "sidecar"
            )
            assert "unobservable_drawn" in frame["spill"]
        # nothing rendered here, so the summary must say so rather than report zero
        summary = manifest["connectivity_states"]
        assert summary["n_frames_with_a_measurement"] == 0
        assert summary["partition_is_exact_on_every_frame"] is False


#: Where the two shipped grids live.  Only the ``@_dataset`` class below reads it.
SHIPPED_ROOT = Path(
    os.environ.get("PALEO_SHIPPED_ANIM_ROOT")
    or r"Y:\paleo\renders\khvalynian_spill_animation"
)

_shipped = pytest.mark.skipif(
    not (SHIPPED_ROOT / "spill_0.5ka_9_epochs" / "sequence_manifest.json").exists(),
    reason=(
        "needs the shipped deliverable under "
        f"{SHIPPED_ROOT}; set PALEO_SHIPPED_ANIM_ROOT to point elsewhere"
    ),
)


# ---------------------------------------------------------------------------
@_shipped
class TestWhatWasActuallyShipped:
    """The acceptance, read off the artifacts rather than off this run's memory."""

    @staticmethod
    def _load(name):
        return json.loads(
            (SHIPPED_ROOT / name / "sequence_manifest.json").read_text(encoding="utf-8")
        )

    def test_the_deliverable_is_nine_epochs_of_imagery_and_no_placeholders(self):
        nine = self._load("spill_0.5ka_9_epochs")
        assert nine["grid"]["step_ka"] == 0.5
        assert nine["n_frames"] == 9
        assert nine["n_rendered"] == 9
        assert nine["n_refused"] == 0
        assert [f["epoch_ka"] for f in nine["frames"]] == [
            17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0
        ]
        assert all(f["refusal"] is None and f["png"] for f in nine["frames"])

    def test_the_seventeen_slot_grid_keeps_the_eight_that_cannot_resolve(self):
        seventeen = self._load("spill_0.25ka_17_slots")
        assert seventeen["grid"]["step_ka"] == 0.25
        assert seventeen["n_frames"] == 17
        assert seventeen["n_rendered"] == 9
        assert seventeen["n_refused"] == 8
        refused = [f["epoch_ka"] for f in seventeen["frames"] if f["refusal"]]
        assert refused == [16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25]
        for frame in seventeen["frames"]:
            if frame["refusal"]:
                assert frame["threshold"] is None
                assert "CurveAgeGapError" in frame["refusal"]
                assert Path(frame["canvas_frame"]).exists()

    def test_neither_grid_can_be_mistaken_for_the_other(self):
        nine = self._load("spill_0.5ka_9_epochs")
        seventeen = self._load("spill_0.25ka_17_slots")
        assert "0.5 ka grid" in nine["what"]
        assert nine["grid"]["do_not_call_this"] == "a 0.25 ka animation"
        assert nine["grid"]["quarter_ka_epochs_not_in_this_sequence"] == [
            16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25
        ]
        assert "CurveAgeGapError" in nine["grid"]["why_those_are_absent"]
        assert seventeen["grid"]["quarter_ka_epochs_not_in_this_sequence"] == []

    def test_the_two_grids_carry_the_same_frame_bytes(self):
        nine = self._load("spill_0.5ka_9_epochs")
        seventeen = self._load("spill_0.25ka_17_slots")
        by_epoch = {
            f["epoch_ka"]: f["canvas_frame_sha256"]
            for f in seventeen["frames"] if f["png"]
        }
        assert len(by_epoch) == 9
        for frame in nine["frames"]:
            assert frame["canvas_frame_sha256"] == by_epoch[frame["epoch_ka"]]
            assert frame["canvas_frame_sha256_reverified"] == by_epoch[
                frame["epoch_ka"]
            ]
            assert anim.sha256_of(frame["canvas_frame"]) == by_epoch[
                frame["epoch_ka"]
            ], "the shipped canvas PNG no longer hashes to what the manifest records"

    def test_the_manych_guard_ran_and_agreed_on_every_rendered_frame(self):
        nine = self._load("spill_0.5ka_9_epochs")
        verdicts = []
        for frame in nine["frames"]:
            entry = next(
                s for s in frame["sill"] if s["pair"] == ["caspian", "pontic"]
            )
            assert entry["measured_saddle_m"] is not None, (
                f"{frame['epoch_ka']} ka produced no saddle; an unrun guard is not a "
                "passed guard"
            )
            assert abs(
                entry["measured_saddle_m"] - entry["expected_m"]
            ) == pytest.approx(entry["gap_m"], abs=1e-9)
            verdicts.append(entry["verdict"])
        assert set(verdicts) == {"WITHIN_TOLERANCE"}, verdicts

    def test_the_terrain_verdict_and_the_terrain_grouping_agree_epoch_by_epoch(self):
        """THREE READINGS OF THE TERRAIN, checked against each other.  Not the picture.

        RENAMED 2026-08-04, and the rename is the fix.  This was
        ``test_the_caspian_stands_above_the_crest_exactly_where_it_spills`` with the
        docstring "The picture and the verdict must agree, epoch by epoch" -- and the
        data contradicts that sentence: ``painted_water.basins_joined`` is FALSE on
        all nine shipped frames while ``spilling`` is TRUE on five of them.  The
        assertion never went near the picture.  ``joined`` below is read from
        ``spill.connected_groups``, which is ``assign_basins``' own terrain flood, so
        what the test compared was the verdict against itself -- a title standing in
        for a guard, which is the defect class the 2026-08-04 recovery README names
        in as many words ("a test recovered by name only is a title, not a guard").

        The name now says what the assertion does.  What the PICTURE does is checked,
        with the assertion the old name implied, in
        :meth:`test_the_picture_never_joins_the_two_basins_and_that_is_the_terrain`.
        """
        nine = self._load("spill_0.5ka_9_epochs")
        for frame in nine["frames"]:
            saddle = next(
                s for s in frame["sill"] if s["pair"] == ["caspian", "pontic"]
            )["measured_saddle_m"]
            caspian = frame["basin_levels"]["caspian"]["level_m"]
            grouped = any(
                set(group) >= {"caspian", "pontic"}
                for group in frame["spill"]["connected_groups"]
            )
            assert (caspian > saddle) == bool(frame["spill"]["spilling"]) == grouped, (
                f"{frame['epoch_ka']} ka: caspian {caspian:+.4f} against saddle "
                f"{saddle:+.4f} but spilling={frame['spill']['spilling']} and "
                f"groups={frame['spill']['connected_groups']}"
            )

    def test_the_picture_never_joins_the_two_basins_and_that_is_the_terrain(self):
        """``basins_joined`` is FALSE at every epoch, INCLUDING the ones that spill.

        This is the correct answer, it is a property of the ground, and it must never
        be shipped as a defect or a near-miss.  The Caspian does clear the Manych
        crest; what is on the other side is not a receiving sea but ~450 km of bed
        standing ABOVE the Pontic surface, so the spill is a river across dry ground
        into a lake tens of metres lower.  No flat surface and no linear ramp between
        two flat surfaces can draw that as one water body, because it was not one.
        The number that says so on each frame is ``gap_cells``: cells of the measured
        route that the frame drew DRY.
        """
        nine = self._load("spill_0.5ka_9_epochs")
        joined, spilling, gaps = {}, {}, {}
        for frame in nine["frames"]:
            epoch = frame["epoch_ka"]
            painted = frame["spill"]["painted_water"]
            joined[epoch] = painted["basins_joined"]
            spilling[epoch] = bool(frame["spill"]["spilling"])
            gaps[epoch] = painted["gap_cells"]
        assert set(joined.values()) == {False}, (
            f"a frame drew the basins as one body: {joined}"
        )
        assert any(spilling.values()), (
            "no epoch spills at all, so this test is no longer about anything"
        )
        # the two questions are DIFFERENT questions, and the artifact keeps both
        assert any(spilling[e] and joined[e] is False for e in joined), (
            "spilling and basins_joined never disagree here, which would mean the "
            "picture had started to follow the verdict"
        )
        for epoch, spills in spilling.items():
            if spills:
                assert gaps[epoch] > 0, (
                    f"{epoch} ka spills and the route has no dry cell, which would "
                    "make basins_joined=false unexplained by this frame's own numbers"
                )

    def test_no_frame_breached_a_limit_that_was_actually_measured(self):
        for name in ("spill_0.5ka_9_epochs", "spill_0.25ka_17_slots"):
            manifest = self._load(name)
            for frame in manifest["frames"]:
                by_quantity = (
                    (frame["clipping"].get("out_of_range") or {}).get("by_quantity")
                    or {}
                )
                for quantity in anim.MEASURED_LIMIT_QUANTITIES:
                    assert int(by_quantity.get(quantity) or 0) == 0, (
                        f"{name} {frame['epoch_ka']} ka breached {quantity}"
                    )

    def test_every_marked_cell_has_its_excursion_measured_in_the_same_run(self):
        nine = self._load("spill_0.5ka_9_epochs")
        for frame in nine["frames"]:
            marked = int(
                ((frame["clipping"].get("out_of_range") or {}).get("by_quantity") or {})
                .get("shade_above_shade_max") or 0
            )
            excursion = frame["analytic_shade_ceiling"]
            assert excursion["status"] == "MEASURED"
            assert excursion["cells_above_shade_max"] >= marked
            assert excursion["excursion_above_ceiling"] <= anim.SHADE_EXCURSION_BUDGET
            assert excursion["clearance_below_floor"] > 0.0

    def test_the_limits_were_measured_over_exactly_the_epochs_that_were_drawn(self):
        nine = self._load("spill_0.5ka_9_epochs")
        seventeen = self._load("spill_0.25ka_17_slots")
        assert seventeen["sequence_limits_cover"]["verdict"].startswith("EXACT")
        assert seventeen["sequence_limits_cover"]["n_slots_in_the_sequence"] == 17
        assert seventeen["sequence_limits_cover"][
            "epochs_the_limits_were_computed_over"
        ] == seventeen["sequence_limits_cover"]["epochs_that_produced_a_frame"]
        assert len(seventeen["sequence_limits_cover"]["slots_carrying_no_frame"]) == 8
        assert nine["sequence_limits"]["epochs"] == [
            f["epoch_ka"] for f in nine["frames"]
        ]

    def test_the_canvas_is_measured_and_not_the_declared_fallback(self):
        for name in ("spill_0.5ka_9_epochs", "spill_0.25ka_17_slots"):
            manifest = self._load(name)
            derived = manifest["canvas"]["derived_from"]
            assert "element-wise maximum" in derived
            assert "fallback" not in derived
            assert manifest["canvas"]["size_px"] != list(anim.FALLBACK_CANVAS_PX)

    @staticmethod
    def _path(value):
        """A path key, in whichever spelling the manifest that wrote it used.

        ``animation.gif`` became a two-spelling record on 2026-08-04 (see
        ``anim.PATH_SPELLING_DECISION``).  The 17-slot tree on disk was written by
        the previous writer and still carries a bare string; it can only be rebuilt
        by a full native render, which is the next phase's work.
        """
        return Path(value["as_written"] if isinstance(value, dict) else value)

    def test_both_animations_carry_one_global_palette_and_verified_delays(self):
        for name in ("spill_0.5ka_9_epochs", "spill_0.25ka_17_slots"):
            manifest = self._load(name)
            animation = manifest["animation"]
            assert animation["gif_palette_verdict"] == "ONE_GLOBAL_PALETTE"
            assert set(
                animation["gif_frame_durations_ms_read_back_with_pil"]
            ) == {250}
            assert len(
                animation["gif_frame_durations_ms_read_back_with_pil"]
            ) == manifest["n_frames"]
            assert self._path(animation["gif"]).exists()
            assert manifest["palette"]["gif_audit"][
                "n_frames_with_a_local_color_table"
            ] == 0

    def test_the_deliverables_webp_is_the_24_bit_escape_it_is_advertised_as(self):
        """The finding: the named remedy carried 222 colours against a 256 table."""
        manifest = self._load("spill_0.5ka_9_epochs")
        animation = manifest["animation"]
        audit = manifest["palette"]["webp_audit"]
        assert animation["webp_verdict"] == "BIT_EXACT_24_BIT"
        assert audit["pixels_differing_from_the_source"] == 0
        assert audit["escapes_the_gif_ceiling"] is True
        assert audit["n_distinct_colours_decoded"] > audit["gif_colour_table_ceiling"]
        # the headroom sentence in the same file must state that measurement
        sentence = manifest["palette"]["headroom"]["if_256_is_not_enough"]
        assert f"{audit['n_distinct_colours_decoded']:,}" in sentence
        assert "IS available in this deliverable" in sentence
        # ... and the WebP holds every colour the sequence has, not a subset
        assert audit["n_distinct_colours_decoded"] == manifest["palette"]["master"][
            "n_distinct_in_sequence"
        ]
        # both delays asserted, not just the GIF's
        assert animation["webp_delay_asserted"] is True
        assert set(
            animation["webp_frame_durations_ms_read_back_with_pil"]
        ) == {250}

    def test_the_deliverable_is_self_contained_and_says_what_it_still_needs(self):
        manifest = self._load("spill_0.5ka_9_epochs")
        block = manifest["self_contained"]
        assert block["verdict"] == "SELF_CONTAINED_FOR_THE_ANIMATION"
        assert block["files_copied"] == manifest["n_frames"] == 9
        assert block["bytes_copied"] > 0
        root = SHIPPED_ROOT / "spill_0.5ka_9_epochs"
        for frame in manifest["frames"]:
            local = Path(frame["canvas_frame"])
            assert local.exists(), local
            assert local.parent == root / "canvas"
            assert anim.sha256_of(local) == frame["canvas_frame_sha256"]
        assert "frames/" in block["what_is_still_external"]

    def test_the_quantizer_decision_is_taken_and_derived_from_its_own_arms(self):
        manifest = self._load("spill_0.5ka_9_epochs")
        comparison = manifest["palette"]["quantizer_comparison"]
        decision = comparison["decision"]
        arms = {
            name: arm for name, arm in comparison["arms"].items()
            if arm["status"] == "MEASURED"
        }
        assert set(arms) == {"MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE"}
        assert decision["selected_by_the_rule"] == min(
            arms, key=lambda k: arms[k]["max_per_pixel"]
        )
        assert decision["verdict"] == "AGREES_WITH_THE_SHIPPED_METHOD"
        assert manifest["palette"]["master"]["method"] == decision["shipped"]
        # the trade the predecessor left open, now closed with both sides shown
        assert decision["what_it_buys"]["worst_pixel_tighter_by_factor"] > 1.0
        assert decision["what_it_costs"]["extra_mean_per_pixel_error"] > 0.0

    def test_the_deliverable_records_both_spellings_of_its_paths(self):
        manifest = self._load("spill_0.5ka_9_epochs")
        block = manifest["paths"]
        mapped = {e["drive"]: e for e in block["drive_map"]}
        assert "Y:" in mapped
        assert mapped["Y:"]["expands_to_a_unc_share"] is True
        assert mapped["Y:"]["unc_target"].startswith("\\\\")
        assert block["topo_dir"]["differ"] is True
        # both spellings really do appear across the file it describes
        spellings = block["spelling_by_key"]
        assert any(t["unc"] for t in spellings.values())
        assert any(t["drive_letter"] for t in spellings.values())

    def test_the_shipped_deliverable_carries_the_third_state(self):
        """The finding this round cleared, checked in the artifact and not the code."""
        for name in ("spill_0.5ka_9_epochs", "spill_0.25ka_17_slots"):
            manifest = self._load(name)
            summary = manifest["connectivity_states"]
            assert summary["partition_is_exact_on_every_frame"] is True, (
                f"{name}: {summary['frames_where_the_partition_does_not_close']}"
            )
            assert summary["epochs_with_no_measurement"] == []
            assert summary["policy_on_every_frame"] == ["mark"]
            rendered = [f for f in manifest["frames"] if f["png"]]
            assert summary["n_frames_with_a_measurement"] == len(rendered)
            for frame in rendered:
                conn = frame["spill"]["connectivity"]
                assert conn, f"{name} {frame['epoch_ka']} ka lost its connectivity"
                assert frame["spill"]["unobservable_drawn"] is not None
                assert set(conn) >= {"connected", "unobservable", "closed"}
                assert conn["connected"]["cells"] + conn["unobservable"]["cells"] + (
                    conn["closed"]["cells"]
                ) == conn["flat_mask_cells"]

    def test_the_summary_numbers_are_the_frames_own_numbers(self):
        """No sentence in the summary may carry a figure its frames contradict."""
        manifest = self._load("spill_0.5ka_9_epochs")
        summary = manifest["connectivity_states"]
        by_epoch = {
            f["epoch_ka"]: f["spill"]["connectivity"]
            for f in manifest["frames"] if f["png"]
        }
        for row in summary["per_epoch"]:
            conn = by_epoch[row["epoch_ka"]]
            assert row["unobservable_cells"] == conn["unobservable"]["cells"]
            assert row["unobservable_km2"] == conn["unobservable"]["km2"]
            assert row["closed_cells"] == conn["closed"]["cells"]
            assert row["connected_cells"] == conn["connected"]["cells"]
            assert row["flat_mask_cells"] == conn["flat_mask_cells"]
        most = summary["unobservable_cells_most"]
        assert most["unobservable_cells"] == max(
            r["unobservable_cells"] for r in summary["per_epoch"]
        )
        assert by_epoch[most["epoch_ka"]]["unobservable"]["cells"] == most[
            "unobservable_cells"
        ]

    def test_an_unobservable_body_is_never_drawn_as_water_or_as_land(self):
        """It must reach an OBSERVATION boundary, and carry the surface it sits under."""
        manifest = self._load("spill_0.5ka_9_epochs")
        seen = 0
        for frame in manifest["frames"]:
            block = frame["spill"]["connectivity"]["unobservable"]
            for body in block.get("bodies") or []:
                seen += 1
                assert body["touched_edges"], (
                    "a body with no touched edge is CLOSED, which is a measurement, "
                    "and must not be reported in the state that means 'undecidable'"
                )
                assert set(body["touched_edges"]) <= {
                    "north", "south", "west", "east", "nodata"
                }
                assert body["surface_min_m"] is not None
                assert body["cells"] > 0
        assert seen > 0, (
            "no unobservable body in the whole deliverable -- either the scene "
            "changed or the state stopped being recorded"
        )

    def test_every_closed_body_really_is_enclosed(self):
        manifest = self._load("spill_0.5ka_9_epochs")
        for frame in manifest["frames"]:
            for body in (
                frame["spill"]["connectivity"]["closed"].get("bodies") or []
            ):
                assert body["touched_edges"] == [], (
                    "a body touching the edge of the observed grid cannot be called "
                    "closed: this raster did not see its rim"
                )

    def test_every_raster_kept_a_canonical_source_path(self):
        for name in ("spill_0.5ka_9_epochs", "spill_0.25ka_17_slots"):
            manifest = self._load(name)
            resolution = manifest["raster_source_resolution"]
            assert resolution["epochs_with_no_source_path"] == []
            for frame in manifest["frames"]:
                source = frame["raster"]["source_path"]
                assert source and source.lower().startswith("y:\\paleo"), source
            assert all(
                curve.get("sha256") or curve.get("status")
                for curve in manifest["level_curves"]
            )


# ---------------------------------------------------------------------------
@_dataset
class TestOnTheRealRasters:

    def test_all_seventeen_rasters_exist(self):
        missing = []
        for epoch in anim.epoch_grid():
            try:
                anim.raster_for(epoch)
            except anim.MissingEpochRasterError:
                missing.append(epoch)
        assert not missing, f"no raster for {missing}"

    def test_what_the_registry_does_to_the_two_hand_written_seeds_this_repo_had(self):
        """One is reproduced to the cell; the other is CORRECTED by one cell.

        Pinning both halves, because "the registry reproduces the seeds" is the
        comfortable claim and it is only half true -- and the half that is false is
        a 67 m deeper cell, which is the kind of move that changes the painted
        corridor even where B1 measured the saddle to be invariant.
        """
        raster = anim.raster_for(16.5)
        seeds = {
            s.basin: s
            for s in anim.resolve_seeds(
                raster,
                {"caspian": 29.9444, "pontic": -18.0204, "mediterranean": -77.905823},
                epoch_ka=16.5,
            )
        }
        # the Caspian: the same cell the hand-written constant names
        assert (seeds["caspian"].lon, seeds["caspian"].lat) == pytest.approx(
            (50.743750, 38.952083), abs=1e-6
        )
        assert (seeds["caspian"].row, seeds["caspian"].col) == (3611, 6058)
        assert seeds["caspian"].elevation_m == pytest.approx(-1048.861, abs=0.01)

        # the Pontic: NOT the hand-written cell -- one cell away, and deeper
        assert (seeds["pontic"].lon, seeds["pontic"].lat) == pytest.approx(
            (35.281250, 42.197917), abs=1e-6
        )
        assert (seeds["pontic"].row, seeds["pontic"].col) == (2832, 2347)
        assert seeds["pontic"].elevation_m == pytest.approx(-2645.555, abs=0.01)
        hand_written = (35.28542, 42.20208)
        assert abs(seeds["pontic"].lon - hand_written[0]) == pytest.approx(
            0.0041667, abs=1e-5
        ), "the correction is exactly one cell in longitude"
        assert abs(seeds["pontic"].lat - hand_written[1]) == pytest.approx(
            0.0041667, abs=1e-5
        ), "and exactly one cell in latitude"

        assert seeds["mediterranean"].elevation_m == pytest.approx(-4450.646, abs=0.01)

    def test_nine_epochs_plan_and_eight_refuse_on_the_pontic_curve(self):
        plans = anim.plan_sequence()
        assert len(plans) == 17
        planned = [p.epoch_ka for p in plans if p.renderable]
        refused = [p.epoch_ka for p in plans if not p.renderable]
        assert planned == [17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0]
        assert refused == [16.75, 16.25, 15.75, 15.25, 14.75, 14.25, 13.75, 13.25]
        for plan in plans:
            if plan.renderable:
                assert set(plan.levels) == {"caspian", "pontic", "mediterranean"}
                assert len(plan.seeds) == 3
                assert plan.dem_frame == "F0 present-day datum (H_modern + dz_solid)"
            else:
                assert "CurveAgeGapError" in plan.refusal

    def test_the_pontic_never_stands_above_the_measured_bosphorus_crest(self):
        """So at most ONE pair is donor->receiver and 'ramp' is never refused."""
        for plan in anim.plan_sequence():
            if not plan.renderable:
                continue
            crest_f1 = anim.BOSPHORUS_CREST_F0_M + plan.dz_bosphorus_m
            assert plan.levels["pontic"].level_m < crest_f1, (
                f"at {plan.epoch_ka} ka the Pontic stands above the Bosphorus crest; "
                "two donor->receiver pairs would then be possible and CORRIDOR_NOTE "
                "is wrong"
            )

    def test_the_sequence_limits_are_computed_over_the_renderable_epochs_only(self):
        plans = anim.plan_sequence()
        limits = anim.sequence_limits_for(
            plans, config=RenderConfig(max_px=anim.NATIVE_MAX_PX)
        )
        assert limits is not None
        assert sorted(limits.epochs) == sorted(
            p.epoch_ka for p in plans if p.renderable
        )
        assert len(limits.rasters) == len(limits.epochs)
        assert all(len(r.sha256) == 64 for r in limits.rasters)
        for epoch, rule in limits.epoch_rules:
            assert "Pontic surface" in rule and "RECEIVING basin" in rule

    def test_every_pinned_extremum_is_attributed_to_an_epoch(self):
        """A pinned mapping whose owner cannot be named is one nobody can argue with."""
        plans = anim.plan_sequence()
        config = RenderConfig(max_px=anim.NATIVE_MAX_PX)
        limits = anim.sequence_limits_for(plans, config=config)
        extrema = anim.attribute_extrema(plans, limits, config=config)
        renderable = {p.epoch_ka for p in plans if p.renderable}
        for name in (
            "land_min_m", "land_max_m", "depth_max_m",
            "dem_data_min_m", "dem_data_max_m", "depth_data_max_m",
        ):
            owners = extrema[name]["produced_by_epoch_ka"]
            assert owners != "UNATTRIBUTED", f"{name} matches no single epoch"
            assert set(owners) <= renderable
            assert set(extrema[name]["per_epoch"]) == {f"{e:g}" for e in renderable}
        assert "shade_min" not in extrema and "shade_max" not in extrema


# ---------------------------------------------------------------------------
# THE INSET AND THE RAMP  (2026-08-04, B-RAMP)
#
# Two things this phase adds and one thing it must not do:
#   * corridor="ramp" DRAWS an interpolated surface, so the added disclosure that
#     says so is load-bearing and its template must carry no digit;
#   * the inset is a SECOND GRID -- its extent, its seeds and its sill guard are all
#     provenanced separately, and a coordinate typed into any of the three would be
#     indistinguishable downstream from a measured one;
#   * neither may quietly fall back: not ramp->step, not "inset refused"->"no inset
#     was asked for", not "two colour mappings that happen to agree"->"one mapping".
# ---------------------------------------------------------------------------

_F0 = "F0 present-day datum (H_modern + dz_solid)"

#: Deeps placed inside the INSET's own seed boxes, so the toy raster can resolve the
#: inset registry the same way the production one does.
_TOY_INSET_DEEPS = {
    "caspian_inset": (47.8, 44.8, -900.0),
    "pontic_inset": (37.0, 43.9, -1900.0),
}


def _toy_tagged(path: Path) -> Path:
    return _write_scene_raster(
        path,
        deeps={**_TOY_DEEPS, **_TOY_INSET_DEEPS},
        tags={"vertical_frame": _F0},
    )


def _toy_plan(raster: Path, epoch_ka: float = 16.5):
    """A minimal EpochPlan over the toy raster -- levels only, no curve reads."""
    return anim.EpochPlan(
        epoch_ka=float(epoch_ka),
        raster_path=str(raster),
        raster_sha256=anim.sha256_of(raster),
        dem_frame=_F0,
        levels={
            name: anim.BasinLevel(
                basin=name, level_m=level, frame=_F0,
                rule="toy fixture", provenance="toy fixture",
            )
            for name, level in _TOY_LEVELS.items()
        },
        dz_bosphorus_m=2.5,
    )


class TestTheInsetExtentIsARegistryEntryNotALayoutChoice:

    def test_the_registered_extent_contains_the_measured_crest(self):
        assert anim.MANYCH_INSET.contains(
            anim.MANYCH_CREST_MEASURED_LON, anim.MANYCH_CREST_MEASURED_LAT
        )
        assert anim.MANYCH_CREST_MEASURED_LON == pytest.approx(44.110417, abs=1e-6)
        assert anim.MANYCH_CREST_MEASURED_LAT == pytest.approx(45.735417, abs=1e-6)

    def test_an_extent_that_does_not_contain_its_required_point_raises(self):
        with pytest.raises(anim.InsetExtentError, match="does not contain"):
            anim.InsetExtent(
                name="too_far_west", lon_min=30.0, lat_min=43.0,
                lon_max=40.0, lat_max=47.0, why="deliberately short of the crest",
                seed_boxes=(
                    anim.SeedBox(
                        basin="pontic", name="inside_this_one", lon_min=31.0,
                        lat_min=44.0, lon_max=32.0, lat_max=45.0, why="fixture",
                    ),
                ),
                must_contain=(("the crest", 44.11, 45.73),),
            )

    def test_a_seed_box_that_leaves_the_extent_raises(self):
        with pytest.raises(anim.InsetExtentError, match="not wholly inside"):
            anim.InsetExtent(
                name="box_outside", lon_min=44.0, lat_min=45.0,
                lon_max=45.0, lat_max=46.0, why="one degree around the crest",
                seed_boxes=anim.MANYCH_INSET.seed_boxes,
            )

    def test_an_extent_with_no_seed_box_raises(self):
        with pytest.raises(ValueError, match="registers no seed box"):
            anim.InsetExtent(
                name="seedless", lon_min=44.0, lat_min=45.0, lon_max=45.0,
                lat_max=46.0, why="fixture", seed_boxes=(),
            )

    def test_the_inset_could_not_have_inherited_the_scenes_seed_boxes(self):
        """The reason a second registry exists, stated as an assertion.

        Both of the scene's relevant boxes lie OUTSIDE this extent, so "reuse the
        seeds we already have" was never available -- an inherited coordinate would
        have been clamped or refused by _seed_rowcol, and either way the cell drawn
        would not be the cell named.
        """
        by_basin = {b.basin: b for b in anim.SEED_BOXES}
        for basin in ("caspian", "pontic"):
            box = by_basin[basin]
            inside = (
                anim.MANYCH_INSET.lon_min <= box.lon_min
                and box.lon_max <= anim.MANYCH_INSET.lon_max
                and anim.MANYCH_INSET.lat_min <= box.lat_min
                and box.lat_max <= anim.MANYCH_INSET.lat_max
            )
            assert not inside, f"{box.name} is inside the inset after all"

    def test_every_bound_and_every_box_carries_a_reason(self):
        assert anim.MANYCH_INSET.why.strip()
        for bound in ("EAST", "WEST", "SOUTH", "NORTH"):
            assert bound in anim.MANYCH_INSET.why, f"no reason given for {bound}"
        for box in anim.MANYCH_INSET.seed_boxes:
            assert box.why.strip()
        assert anim.MANYCH_INSET.basins() == ("caspian", "pontic")
        assert anim.MANYCH_INSET.pairs() == (("caspian", "pontic"),)

    def test_the_pontic_margin_names_the_surface_and_the_epoch_it_belongs_to(self):
        """It shipped a margin against the HIGHEST surface labelled as the LOWEST.

        Re-derived here from the numbers the sentence itself names, so the test is a
        subtraction and not a second copy of the answer: the box's own cell on
        khv_16.5ka.tif is -2168.60 m, the window's lowest Pontic surface is -40.35 m
        at 17.0 ka (the module docstring's own range bottoms out at +2128.1 m), and
        the difference is 2128.26 m -- not the 2150.6 m that shipped, which is the
        margin against 16.5 ka's OWN surface of -18.02 m.
        """
        why = anim.MANYCH_INSET.why
        seed_elevation, lowest_surface, own_surface = -2168.60, -40.35, -18.02
        assert f"{lowest_surface - seed_elevation:.2f} m" == "2128.25 m"
        assert "2128.26 m below the LOWEST Pontic surface" in why, (
            "the margin against the lowest surface must be the one labelled so"
        )
        assert f"{own_surface - seed_elevation:.2f} m" == "2150.58 m"
        assert "2150.58 m below the surface of its OWN epoch" in why
        assert "2150.6 m below the lowest Pontic surface" not in why
        # and the per-epoch range the module docstring records is quoted intact
        assert "+2128.12 m (17.0 ka) to +2150.71 m (15.0 ka)" in why

    def test_the_south_bound_quotes_the_routes_own_clearance_not_the_cell_size(self):
        """'0.0021 deg, half a cell' was the CELL, 187 times smaller than the margin."""
        why = anim.MANYCH_INSET.why
        southernmost = 43.58958333333333          # route_lat_span[0], as measured
        clearance = southernmost - anim.MANYCH_INSET.lat_min
        assert clearance == pytest.approx(0.3896, abs=0.0001)
        assert clearance / (1 / 240) == pytest.approx(93.5, abs=0.05)
        assert "43.5896 N" in why
        assert "0.3896 deg" in why
        assert "93.5 cells" in why
        assert "cleared the edge by 0.0021 deg" not in why

    def test_the_magnification_is_the_same_in_both_axes(self):
        """2.48x / 3.91x described a stretch the composition cannot produce.

        Both panels come out of one ``imshow`` with ``aspect="equal"`` in one square
        figure, and both extents are wider than the axes box, so both are
        width-limited: the linear scale is the WIDTH ratio in BOTH axes.  Measured
        here on that same layout rather than argued from it.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfg = RenderConfig()
        scene = (25.5, 58.0, 36.0, 54.0)
        inset = (
            anim.MANYCH_INSET.lon_min, anim.MANYCH_INSET.lon_max,
            anim.MANYCH_INSET.lat_min, anim.MANYCH_INSET.lat_max,
        )

        def px_per_deg(extent):
            w, e, s, n = extent
            fig, ax = plt.subplots(
                figsize=(cfg.fig_size_in, cfg.fig_size_in), dpi=cfg.dpi
            )
            ax.imshow(
                np.zeros((8, 8)), extent=[w, e, s, n], origin="upper",
                aspect="equal", interpolation="bilinear",
            )
            fig.canvas.draw()
            box = ax.get_window_extent()
            plt.close(fig)
            return box.width / (e - w), box.height / (n - s)

        full_x, full_y = px_per_deg(scene)
        inset_x, inset_y = px_per_deg(inset)
        assert inset_x / full_x == pytest.approx(2.4809, abs=0.001)
        assert inset_y / full_y == pytest.approx(2.4809, abs=0.001)
        assert inset_x / full_x == pytest.approx(inset_y / full_y, rel=1e-9), (
            "one imshow with aspect='equal' cannot scale the two axes differently"
        )
        why = anim.MANYCH_INSET.why
        assert "MAGNIFIED 2.48x, IN BOTH AXES" in why
        assert "2.48x in longitude and 3.91x in latitude" not in why
        assert "not a magnification of anything drawn" in why


class TestTheInsetCropHoldsTheSameCells:

    def test_the_crop_is_cell_for_cell_the_source_window(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        record = anim.crop_to_extent(source, anim.MANYCH_INSET, tmp_path / "cut.tif")
        assert record["verdict"] == "CELL_FOR_CELL_IDENTICAL_TO_THE_SOURCE_WINDOW"
        assert record["cells_differing_after_write"] == 0
        window = record["window"]
        with rasterio.open(source) as src, rasterio.open(record["inset_path"]) as cut:
            expected = src.read(
                1,
                window=rasterio.windows.Window(
                    window["col_off"], window["row_off"],
                    window["width"], window["height"],
                ),
            )
            assert np.array_equal(cut.read(1), expected)
            assert cut.res == src.res, "the crop must not resample"
            assert cut.crs == src.crs
            assert cut.nodata == src.nodata

    def test_the_crop_carries_the_vertical_frame_tag(self, tmp_path):
        """Without it assign_basins cannot frame-check the inset's expectation.

        A crop that dropped the tag would render happily and UNGUARDED --
        SillFrameMismatchError only fires when a frame is declared, and
        declared_vertical_frame returns None for a silent raster.
        """
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        record = anim.crop_to_extent(source, anim.MANYCH_INSET, tmp_path / "cut.tif")
        with rasterio.open(record["inset_path"]) as cut:
            tags = cut.tags()
        assert tags["vertical_frame"] == _F0
        assert tags["inset_extent_name"] == anim.MANYCH_INSET.name
        assert tags["inset_source_sha256"] == anim.sha256_of(source)
        assert "inset_crop_rule" in tags

    def test_a_crop_that_leaves_the_source_raster_refuses(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        beyond = anim.InsetExtent(
            name="off_the_edge", lon_min=20.0, lat_min=43.2,
            lon_max=30.0, lat_max=47.8,
            why="deliberately west of the scene's own 25.5 E edge",
            seed_boxes=(
                anim.SeedBox(
                    basin="pontic", name="anywhere", lon_min=21.0, lat_min=44.0,
                    lon_max=22.0, lat_max=45.0, why="fixture",
                ),
            ),
        )
        with pytest.raises(anim.InsetExtentError, match="not inside"):
            anim.crop_to_extent(source, beyond, tmp_path / "cut.tif")

    def test_the_record_names_both_files_by_content(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        record = anim.crop_to_extent(source, anim.MANYCH_INSET, tmp_path / "cut.tif")
        assert record["source_sha256"] == anim.sha256_of(source)
        assert record["inset_sha256"] == anim.sha256_of(record["inset_path"])
        assert record["source_sha256"] != record["inset_sha256"]
        assert record["cells"] == record["window"]["width"] * record["window"]["height"]


class TestTheInsetSeedsAreResolvedNotTyped:

    def test_plan_inset_resolves_both_seeds_inside_the_extent(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        assert cut.refusal is None, cut.refusal
        assert cut.renderable
        assert {s.basin for s in cut.seeds} == {"caspian", "pontic"}
        for seed in cut.seeds:
            assert anim.MANYCH_INSET.contains(seed.lon, seed.lat)
            assert seed.elevation_m < seed.level_m
            assert seed.rule == anim.SEED_RULE
            assert seed.raster_sha256 == cut.raster_sha256
            assert seed.raster_path == str(Path(cut.raster_path).resolve())
        deepest = {s.basin: s.elevation_m for s in cut.seeds}
        assert deepest["caspian"] == pytest.approx(-900.0)
        assert deepest["pontic"] == pytest.approx(-1900.0)

    def test_an_inset_seed_that_is_not_under_its_own_water_refuses(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        plan = _toy_plan(source)
        plan.levels["caspian"] = anim.BasinLevel(
            basin="caspian", level_m=-2000.0, frame=_F0,
            rule="a surface below every cell of its own box",
            provenance="fixture",
        )
        cut = anim.plan_inset(plan, anim.MANYCH_INSET, tmp_path / "inset")
        assert cut.refusal is not None
        assert "SeedAboveItsLevelError" in cut.refusal
        assert cut.seeds == ()
        assert not cut.renderable

    def test_a_refused_full_frame_gives_a_refused_inset_that_says_why(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        plan = _toy_plan(source)
        plan.refusal = "CurveAgeGapError: no Pontic row at this epoch"
        cut = anim.plan_inset(plan, anim.MANYCH_INSET, tmp_path / "inset")
        assert cut.refusal is not None
        assert "nothing to inset into" in cut.refusal
        assert "CurveAgeGapError" in cut.refusal


class TestTheInsetIsGuardedOnItsOwnGrid:

    def test_the_expectation_covers_exactly_the_pairs_the_inset_declares(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        expectations = anim.inset_sill_expectations(cut)
        assert set(expectations) == {("caspian", "pontic")}
        full = anim.sill_expectations_for(16.5, _F0, 2.5)
        assert set(full) - set(expectations) == {
            ("mediterranean", "pontic"), ("caspian", "mediterranean")
        }

    def test_the_inset_expectation_is_the_same_quantity_as_the_full_frames(
        self, tmp_path
    ):
        """Same expectation, different measurement.  That is the whole design.

        The crest is a property of the terrain and the vertical frame; the SADDLE is
        a property of the grid.  Arming the inset with a softer expectation would
        make its guard weaker than the one the full frame is held to.
        """
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        mine = anim.inset_sill_expectations(cut)[("caspian", "pontic")]
        theirs = anim.sill_expectations_for(16.5, _F0, 2.5)[("caspian", "pontic")]
        assert isinstance(mine, SillExpectation)
        assert mine.value_m == theirs.value_m
        assert mine.frame == theirs.frame == _F0
        assert mine.source == theirs.source

    def test_a_crest_measured_outside_the_extent_refuses(self, tmp_path):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        sidecar = {
            "sills": [
                {
                    "pair": ["caspian", "pontic"],
                    "lon": 30.0, "lat": 45.0, "elevation_m": 28.0,
                    "route_stats": {"south_lat": 44.0, "north_lat": 46.0},
                }
            ]
        }
        with pytest.raises(anim.InsetExtentError, match="outside its own bounds"):
            anim.assert_route_inside_the_inset(cut, sidecar)

    def test_a_route_leaving_the_extent_in_latitude_is_reported_not_hidden(
        self, tmp_path
    ):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        sidecar = {
            "sills": [
                {
                    "pair": ["caspian", "pontic"],
                    "lon": anim.MANYCH_CREST_MEASURED_LON,
                    "lat": anim.MANYCH_CREST_MEASURED_LAT,
                    "elevation_m": 28.0,
                    "route_stats": {"south_lat": 40.0, "north_lat": 46.0},
                }
            ]
        }
        report = anim.assert_route_inside_the_inset(cut, sidecar)
        assert report["crest_inside"] is True
        assert report["route_lat_inside"] is False
        assert report["verdict"] == "ROUTE_LEAVES_THE_INSET_IN_LATITUDE"

    def test_the_longitude_half_of_the_route_check_says_it_did_not_run(self, tmp_path):
        """sill_route_stats carries no longitudes, and that is said, not implied."""
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        report = anim.assert_route_inside_the_inset(cut, None)
        assert report["checked"] is False
        assert "NOT RECORDED" in report["route_lon_span"]


class TestOneColourMappingForBothPanels:

    @staticmethod
    def _limits(**overrides):
        from paleoeurope.viz.paleocostline_render import (
            RasterFingerprint, SequenceLimits,
        )

        base = dict(
            depth_min_m=0.0, depth_max_m=2000.0, land_min_m=-30.0, land_max_m=2300.0,
            ice_max_m=None, shade_min=-0.7071067811865476, shade_max=1.0,
            dem_data_min_m=-4452.0, dem_data_max_m=5569.0, depth_data_max_m=4489.0,
            depth_percentile=95.0, land_percentile_lo=2.0, land_percentile_hi=98.0,
            vert_exag=8.0, azdeg=315.0, altdeg=45.0, epochs=(16.5,),
            rasters=(
                RasterFingerprint(
                    path="khv_16.5ka.tif", sha256="a" * 64,
                    size_bytes=1, epoch_ka=16.5,
                ),
            ),
            computed_from="fixture",
        )
        base.update(overrides)
        return SequenceLimits(**base)

    def test_the_same_object_is_reported_as_identity(self):
        limits = self._limits()
        full = RenderConfig(sequence_limits=limits, clip_policy="mark")
        inset = anim.dataclasses.replace(full, extra_figure_title="INSET")
        report = anim.assert_one_colour_mapping(full, inset)
        assert report["same_object"] is True
        assert report["verdict"] == "ONE_MAPPING_SAME_OBJECT"
        assert report["colour_fields_differing"] == 0

    def test_two_mappings_that_differ_in_one_colour_field_are_refused(self):
        full = RenderConfig(sequence_limits=self._limits(), clip_policy="mark")
        inset = RenderConfig(
            sequence_limits=self._limits(land_max_m=1800.0), clip_policy="mark"
        )
        with pytest.raises(anim.PaletteNotSharedError, match="land_max_m"):
            anim.assert_one_colour_mapping(full, inset)

    def test_an_unpinned_panel_is_refused(self):
        full = RenderConfig(sequence_limits=self._limits(), clip_policy="mark")
        with pytest.raises(anim.PaletteNotSharedError, match="no pinned SequenceLimits"):
            anim.assert_one_colour_mapping(full, RenderConfig())

    def test_a_different_clip_policy_is_refused(self):
        limits = self._limits()
        full = RenderConfig(sequence_limits=limits, clip_policy="mark")
        inset = RenderConfig(sequence_limits=limits, clip_policy="clip")
        with pytest.raises(anim.PaletteNotSharedError, match="clip policies differ"):
            anim.assert_one_colour_mapping(full, inset)

    def test_equal_by_value_is_reported_as_such_and_not_as_identity(self):
        """Two objects that agree are NOT one mapping, and the record says which."""
        full = RenderConfig(sequence_limits=self._limits(), clip_policy="mark")
        inset = RenderConfig(sequence_limits=self._limits(), clip_policy="mark")
        report = anim.assert_one_colour_mapping(full, inset)
        assert report["same_object"] is False
        assert report["verdict"] == "ONE_MAPPING_BY_VALUE"


class TestTheComposedCanvasNeverScalesAndNeverHides:

    def test_both_panels_survive_the_composition_pixel_for_pixel(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(300, 120), seed=1)
        inset = _noisy_png(tmp_path / "inset.png", size_px=(260, 90), seed=2)
        record = anim.compose_full_and_inset(
            full, inset, tmp_path / "composed.png", caption="a caption", font_px=14,
        )
        assert record["inset_present"] is True
        assert [p["role"] for p in record["panels"]] == ["full", "inset"]
        composed = np.asarray(Image.open(record["composed"]).convert("RGB"))
        for panel, source in zip(record["panels"], (full, inset)):
            assert panel["scaled"] is False and panel["cropped"] is False
            width, height = panel["size_px"]
            x, y = panel["at_px"]
            original = np.asarray(Image.open(source).convert("RGB"))
            assert np.array_equal(composed[y:y + height, x:x + width], original), (
                f"the {panel['role']} panel was altered by the composition"
            )
        assert record["size_px"][1] > 120 + 90, "no room was left for the caption"

    def test_a_missing_inset_is_stated_rather_than_silently_dropped(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(200, 80), seed=3)
        caption = anim.compose_caption(
            16.5, None, {"refusal": "SeedAboveItsLevelError: boom"}, None
        )
        record = anim.compose_full_and_inset(
            full, None, tmp_path / "composed.png", caption=caption, font_px=14,
        )
        assert record["inset_present"] is False
        assert len(record["panels"]) == 1
        assert "THE INSET REFUSED AND IS NOT DRAWN" in record["caption"]
        assert "SeedAboveItsLevelError" in record["caption"]

    def test_the_composition_records_the_bytes_it_was_given(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(120, 60), seed=4)
        inset = _noisy_png(tmp_path / "inset.png", size_px=(120, 40), seed=5)
        record = anim.compose_full_and_inset(
            full, inset, tmp_path / "composed.png", caption="c", font_px=12,
        )
        assert record["panels"][0]["source_sha256"] == anim.sha256_of(full)
        assert record["panels"][1]["source_sha256"] == anim.sha256_of(inset)
        assert record["composed_sha256"] == anim.sha256_of(record["composed"])
        assert "NO LOCATOR RECTANGLE" in record["rule"]

    def test_the_caption_names_the_extent_and_the_insets_own_saddle(self):
        caption = anim.compose_caption(
            16.5,
            None,
            {
                "basins_sidecar": {
                    "sills": [
                        {
                            "pair": ["caspian", "pontic"],
                            "elevation_m": 28.000944,
                            "expectation_status": "CHECKED: within tolerance",
                        }
                    ]
                }
            },
            anim.InsetPlan(epoch_ka=16.5, extent=anim.MANYCH_INSET),
        )
        assert anim.MANYCH_INSET.name in caption
        assert "+28.0009" in caption
        assert "GUARDED SEPARATELY" in caption


class TestRampLegalityIsProvedPerEpochNotInherited:

    @staticmethod
    def _frame(epoch, levels, saddles):
        return {
            "epoch_ka": epoch,
            "refusal": None,
            "basin_levels": {
                name: {"level_m": value} for name, value in levels.items()
            },
            "sill": [
                {"pair": list(pair), "measured_saddle_m": value}
                for pair, value in saddles.items()
            ],
        }

    def test_a_pair_whose_levels_are_below_its_saddle_cannot_merge(self):
        frame = self._frame(
            16.5,
            {"caspian": 29.94, "pontic": -18.02, "mediterranean": -77.91},
            {
                ("caspian", "pontic"): 28.00,
                ("mediterranean", "pontic"): 13.34,
                ("caspian", "mediterranean"): None,
            },
        )
        report = anim.ramp_pair_candidates(frame)
        assert report["n_could_be_donor_receiver"] == 1
        assert report["pairs_that_could_merge"] == [["caspian", "pontic"]]
        assert report["verdict"] == "RAMP_IS_DEFINED"
        by_pair = {tuple(r["pair"]): r for r in report["pairs"]}
        assert by_pair[("mediterranean", "pontic")]["state"] == "CANNOT_MERGE"
        assert by_pair[("caspian", "mediterranean")]["state"] == "CANNOT_MERGE"
        assert "never met" in by_pair[("caspian", "mediterranean")]["why"]

    def test_two_pairs_above_their_saddles_would_raise(self):
        frame = self._frame(
            16.5,
            {"caspian": 29.94, "pontic": 20.0, "mediterranean": -77.91},
            {
                ("caspian", "pontic"): 28.00,
                ("mediterranean", "pontic"): 13.34,
                ("caspian", "mediterranean"): 15.0,
            },
        )
        report = anim.ramp_pair_candidates(frame)
        assert report["n_could_be_donor_receiver"] == 3
        assert report["verdict"] == "RAMP_WOULD_RAISE"

    def test_a_missing_level_is_UNDECIDED_and_is_never_counted_as_safe(self):
        frame = self._frame(
            16.5,
            {"caspian": 29.94, "pontic": None, "mediterranean": -77.91},
            {("caspian", "pontic"): 28.00},
        )
        report = anim.ramp_pair_candidates(frame)
        assert report["verdict"] == "UNDECIDED"
        assert report["n_undecided"] == 1
        assert report["n_could_be_donor_receiver"] == 0
        legality = anim.ramp_legality([frame], "ramp")
        assert legality["verdict"].startswith("UNDECIDED_AT")
        with pytest.raises(anim.RampNotLegalError, match="not defined throughout"):
            anim.assert_ramp_is_legal(legality)

    def test_the_refusal_names_the_epoch_and_offers_no_fallback(self):
        bad = self._frame(
            15.0,
            {"caspian": 40.0, "pontic": 20.0, "mediterranean": 18.0},
            {
                ("caspian", "pontic"): 28.0,
                ("mediterranean", "pontic"): 13.0,
                ("caspian", "mediterranean"): 15.0,
            },
        )
        legality = anim.ramp_legality([bad], "ramp")
        assert legality["verdict"] == "RAMP_WOULD_RAISE_AT: 15 ka"
        with pytest.raises(anim.RampNotLegalError) as excinfo:
            anim.assert_ramp_is_legal(legality)
        message = str(excinfo.value)
        assert "15 ka" in message
        assert "Nothing here falls back" in message

    def test_what_raised_is_kept_separate_from_what_was_predicted(self):
        frame = self._frame(
            16.5, {"caspian": 29.94, "pontic": -18.02}, {("caspian", "pontic"): 28.0}
        )
        frame["refusal"] = "CorridorNotRampableError: 2 basin pairs are connected"
        legality = anim.ramp_legality([frame], "ramp")
        assert legality["epochs_that_raised_CorridorNotRampableError"] == [16.5]

    def test_the_rule_carries_no_digit(self):
        assert not any(c.isdigit() for c in anim.RAMP_LEGALITY_RULE)


class TestTheRenderersLineCountIsQuotedCorrectlyWhereThisModuleQuotesIt:
    """One number about a file this module may not edit, and it must still be right.

    ``tests/test_viz.py``'s incident stub says the renderer is "6 369 lines"; it is
    6,405, and the 2026-08-04 recovery README says 6 405 twice.  That file is outside
    this module's write scope, so the disagreement is REPORTED and not fixed -- and
    what is fixed is the only place this module states the same number.
    """

    def test_this_module_quotes_the_renderers_real_length(self):
        renderer = (
            Path(anim.__file__).resolve().parents[1]
            / "paleoeurope" / "viz" / "paleocostline_render.py"
        )
        lines = len(renderer.read_bytes().splitlines())
        import inspect

        source = inspect.getsource(anim)
        assert f"{lines:,} lines" in source, (
            f"the renderer is {lines:,} lines and this module says otherwise"
        )

    def test_the_stub_that_gets_it_wrong_is_named_and_left_alone(self):
        """A wrong number in a file this agent may not touch is still a finding."""
        stub = Path(anim.__file__).resolve().parents[1] / "tests" / "test_viz.py"
        if not stub.exists():
            pytest.skip("the incident stub is not in this tree")
        renderer = (
            Path(anim.__file__).resolve().parents[1]
            / "paleoeurope" / "viz" / "paleocostline_render.py"
        )
        lines = len(renderer.read_bytes().splitlines())
        text = stub.read_text(encoding="utf-8")
        assert "6 369 lines" in text, (
            "the stub's figure changed; re-check whether the disagreement this test "
            "records still exists"
        )
        assert lines == 6405, (
            "the renderer's length changed; the report that the stub is wrong by "
            f"{6405 - 6369} lines needs re-deriving against {lines}"
        )


class TestTheAddedDisclosureIsATemplateWithNoDigit:

    @staticmethod
    def _frame():
        return {
            "epoch_ka": 16.5,
            "basin_levels": {
                "caspian": {"level_m": 29.94438934326172},
                "pontic": {"level_m": -18.020370483398438},
            },
            "sill": [
                {
                    "pair": ["caspian", "pontic"],
                    "measured_saddle_m": 28.000944137573242,
                    "measured_lonlat": [44.110416666666666, 45.735416666666666],
                    "expected_m": 27.701019287109375,
                    "tolerance_m": 2.0,
                    "gap_m": 0.2999248504638672,
                    "verdict": "WITHIN_TOLERANCE",
                    "expectation_source": "MANYCH_SILL_PATH_M +25.00 m (F0 ...)",
                }
            ],
            # the 2026-08-04 ramp probe's own measured corridor
            "spill": {
                "connectivity": {
                    "corridor_if_painted": {
                        "cells": 15381, "km2": 2290.0, "components": 112,
                    }
                },
                "painted_water": {"basins_joined": False},
            },
        }

    def test_no_shipped_template_in_this_module_carries_a_numeric_literal(self):
        """STRUCTURAL.  It enumerates nothing, so a new template cannot escape it.

        THIS IS THE FIFTH TIME.  A sentence carrying a figure the object beside it
        does not have has now shipped five times on this project -- the corridor
        note's single-epoch saddle, the palette headroom's proxy frames, the offset
        cross-check's five contract rows, the renderer's own corridor cost, and the
        inset's "+8.30 m at 2000 px" -- and every fix so far was to delete the digits
        from ONE named string and add ONE test naming it.  The list of names is what
        keeps failing: it is complete only until the next template is written.

        So this asks the MODULE which strings are templates
        (:func:`~scripts.animate_khvalynian_spill.shipped_templates`, which finds
        them by inspection) and refuses a digit in any of them.  A template added
        tomorrow is in the set tomorrow, with nothing to remember.
        """
        templates = anim.shipped_templates()
        assert templates, "the discovery found nothing, so it is guarding nothing"
        offenders = {
            name: [c for c in text if c.isdigit()]
            for name, text in templates.items()
            if any(c.isdigit() for c in text)
        }
        assert offenders == {}, (
            "these shipped templates carry a numeric literal, which is the defect "
            "class this project has retracted five sentences for -- interpolate the "
            f"value from the artifact being written instead: {sorted(offenders)}"
        )

    def test_the_discovery_finds_the_templates_that_are_actually_used(self):
        """A guard over a set that misses the real templates guards nothing."""
        found = anim.shipped_templates()
        for name in (
            "RAMP_ADDENDUM_TEMPLATE",
            "STEP_ADDENDUM_TEMPLATE",
            "CORRIDOR_DRAWN_TEMPLATE",
            "RAMP_CLAIM_TEMPLATE",
            "NO_RAMP_CLAIM_TEMPLATE",
            "INSET_GUARD_TEMPLATE",
            "INSET_EROSION_TEMPLATE",
        ):
            assert name in found, f"{name} is a shipped template and was not found"
            assert found[name] is getattr(anim, name)
        # ... and prose that merely contains braces is NOT swept in as a template,
        # which is what would make the guard noisy enough to be switched off
        assert "CONNECTIVITY_STATES_RULE" not in found
        assert "{" in anim.CONNECTIVITY_STATES_RULE

    def test_a_template_with_a_digit_in_it_is_caught_by_the_discovery(self, monkeypatch):
        """The guard is proved on a planted offender, not just on a clean module."""
        monkeypatch.setattr(
            anim, "PLANTED_OFFENDER_TEMPLATE", "the saddle reads +8.30 m at {grid}",
            raising=False,
        )
        found = anim.shipped_templates()
        assert "PLANTED_OFFENDER_TEMPLATE" in found
        assert any(c.isdigit() for c in found["PLANTED_OFFENDER_TEMPLATE"])

    def test_the_sentence_never_asserts_a_ribbon_the_frame_does_not_have(self):
        """The probe falsified the first version of this sentence.  Pinned here.

        The tail read "read the CONNECTED RIBBON as ...".  On the 16.5 ka ramp probe
        the ramp drew 15,381 corridor cells in 112 DISJOINT pieces with
        basins_joined=false, so the phrase asserted a connectedness the picture does
        not have -- the same defect class as the four sentences this project has
        already retracted.

        The check used to be ``"connected ribbon" not in text``, which is CASE
        SENSITIVE and could not have caught the phrase it was written for: the
        retired tail was upper-case.  It is now case-insensitive, it covers the
        template as well as the filled text, and it asserts what the sentence says
        INSTEAD -- a guard, not a title.
        """
        text = anim.corridor_addendum(self._frame(), corridor_mode="ramp")["text"]
        for phrase in ("connected ribbon", "the ribbon", "one water body along"):
            assert phrase not in text.lower(), phrase
            assert phrase not in anim.RAMP_ADDENDUM_TEMPLATE.lower(), phrase
        # what it says instead, on the frame's own numbers
        assert "never as 'the two seas met'" in text
        assert "basins_joined" in text
        assert "15,381 corridor cell(s)" in text
        assert "112 disjoint piece(s)" in text
        assert "2,290 sq km" in text
        assert "basins did NOT visibly merge" in text

    def test_a_frame_whose_basins_did_merge_says_so(self):
        frame = self._frame()
        frame["spill"]["painted_water"]["basins_joined"] = True
        text = anim.corridor_addendum(frame, corridor_mode="ramp")["text"]
        assert "DID visibly merge into one water body" in text

    def test_an_uncounted_corridor_says_not_measured_rather_than_zero(self):
        frame = self._frame()
        frame["spill"] = {}
        text = anim.corridor_addendum(frame, corridor_mode="ramp")["text"]
        assert "not measured corridor cell(s)" in text
        assert "were not counted for merging" in text
        assert "0 corridor cell(s)" not in text

    def test_the_step_sentence_also_reports_what_was_drawn(self):
        text = anim.corridor_addendum(self._frame(), corridor_mode="step")["text"]
        assert "WHAT IT ACTUALLY DREW HERE" in text

    def test_the_ramp_sentence_separates_the_measured_crest_from_the_drawn_slope(self):
        added = anim.corridor_addendum(self._frame(), corridor_mode="ramp")
        text = added["text"]
        assert added["template_carries_no_digit"] is True
        assert "+28.001 m at 44.110417 E / 45.735417 N" in text
        assert "+27.701 m" in text and "gap +0.300 m" in text
        assert "NEITHER measured NOR modelled" in text
        assert "LINEARLY IN ALONG-CHANNEL GRAPH DISTANCE" in text
        assert "no hydraulic model produced it" in text
        assert "never as 'the water stood at this height here'" in text
        assert "+29.944 m" in text and "-18.020 m" in text
        assert "NOT measured on the raster it is guarding" in text

    def test_the_step_sentence_says_nothing_is_interpolated(self):
        added = anim.corridor_addendum(self._frame(), corridor_mode="step")
        assert "Nothing in this frame is interpolated" in added["text"]
        assert "NEITHER measured NOR modelled" not in added["text"]

    def test_an_unknown_corridor_mode_raises_rather_than_falling_back(self):
        with pytest.raises(KeyError):
            anim.corridor_addendum(self._frame(), corridor_mode="whatever")

    def test_a_frame_with_no_measured_crest_gets_no_sentence_at_all(self):
        frame = self._frame()
        frame["sill"][0]["measured_saddle_m"] = None
        assert anim.corridor_addendum(frame, corridor_mode="ramp") is None

    def test_the_sentence_moves_with_the_frame_it_is_written_for(self):
        """The defect this shape exists to prevent: one frame's number on nine."""
        first = anim.corridor_addendum(self._frame(), corridor_mode="ramp")["text"]
        other = self._frame()
        other["sill"][0]["measured_saddle_m"] = 31.668964385986328
        other["sill"][0]["expected_m"] = 31.36914825439453
        other["sill"][0]["gap_m"] = 0.2998161315917969
        other["basin_levels"]["caspian"]["level_m"] = 28.1917
        second = anim.corridor_addendum(other, corridor_mode="ramp")["text"]
        assert "+31.669 m" in second and "+31.669 m" not in first
        assert "+28.192 m" in second


class TestHowBigTheClaimTheRampMakesIs:
    """The one quantity the milestone was asked for: how much ground the ramp wets.

    It is measured per frame at write time, it goes in the manifest AND on the canvas,
    and it is named so it cannot be read off the one field that resembles it.
    """

    #: The 2026-08-04 ramp+inset probe's own full-frame numbers.
    @staticmethod
    def _spill(**overrides):
        spill = {
            "corridor": {"mode": "ramp", "cells": 15381, "painted": True},
            "connectivity": {
                "corridor_if_painted": {
                    "cells": 15381,
                    "km2": 2289.9162641270204,
                    "components": 112,
                    "bed_min_m": -18.02033042907715,
                    "bed_max_m": 29.819061279296875,
                    "donor": "caspian",
                    "donor_level_m": 29.94438934326172,
                    "receiver": "pontic",
                    "receiver_level_m": -18.020370483398438,
                }
            },
            "painted_water": {
                "cells": 8447753,
                "corridor_cells_restored_by_the_or": 243,
            },
            "morphology": {"cells_added": 3438, "cells_removed": 7315},
        }
        spill.update(overrides)
        return {"spill": spill}

    def test_the_claim_is_the_corridor_and_it_is_counted_in_cells_and_km2(self):
        claim = anim.ramp_new_wet_ground(self._spill())
        assert claim["corridor_is_painted"] is True
        assert claim["cells_the_ramp_wets_that_a_flat_surface_leaves_dry"] == 15381
        assert claim["km2_the_ramp_wets_that_a_flat_surface_leaves_dry"] == (
            pytest.approx(2289.916, abs=0.001)
        )
        assert claim["pieces"] == 112
        assert claim["share_of_this_frames_painted_water"] == pytest.approx(
            100.0 * 15381 / 8447753, abs=1e-4
        )

    def test_it_cannot_be_confused_with_the_or_restoration_counter(self):
        """The audit's finding: the only field that looks like it is 63x smaller."""
        claim = anim.ramp_new_wet_ground(self._spill())
        beside = claim["not_to_be_confused_with"]
        assert beside["field"] == "painted_water.corridor_cells_restored_by_the_or"
        assert beside["value"] == 243
        assert beside["this_quantity_is_larger_by"] == pytest.approx(63.3, abs=0.01)
        assert "de-speckle" in beside["what_that_field_counts"]
        # and the name of the quantity itself cannot be mistaken for that field
        assert "cells_the_ramp_wets_that_a_flat_surface_leaves_dry" in claim
        assert "restored_by_the_or" not in "".join(
            k for k in claim if k != "not_to_be_confused_with"
        )

    def test_the_flat_frame_is_ruled_out_by_a_check_and_not_by_an_argument(self):
        claim = anim.ramp_new_wet_ground(self._spill())
        assert claim["every_cell_stands_above_the_receivers_flat_surface"] is True
        assert claim["is_an_upper_bound"] is False
        assert claim["bed_min_above_receiver_level_by_m"] == pytest.approx(
            -18.02033042907715 - -18.020370483398438, abs=1e-12
        )
        assert "Its SIGN is the measurement" in claim[
            "the_margin_is_thin_by_construction"
        ]
        assert "THIN BY CONSTRUCTION" in anim.ramp_claim_clause(claim), (
            "the caption must not let a 4e-05 m margin read as a strong result"
        )

    def test_a_corridor_bed_under_the_receivers_surface_makes_it_an_upper_bound(self):
        """Then some of those cells are dry only for want of a route, and it says so."""
        spill = self._spill()
        spill["spill"]["connectivity"]["corridor_if_painted"]["bed_min_m"] = -30.0
        claim = anim.ramp_new_wet_ground(spill)
        assert claim["every_cell_stands_above_the_receivers_flat_surface"] is False
        assert claim["is_an_upper_bound"] is True
        assert "unreachable" in claim["why_it_would_be_an_upper_bound"]
        assert "UPPER BOUND" in anim.ramp_claim_clause(claim)

    def test_a_step_frame_claims_nothing_and_says_what_it_would_have_claimed(self):
        spill = self._spill(
            corridor={"mode": "step", "cells": 1713697, "painted": False}
        )
        claim = anim.ramp_new_wet_ground(spill)
        assert claim["corridor_is_painted"] is False
        assert claim["cells_the_ramp_wets_that_a_flat_surface_leaves_dry"] == 0
        assert claim["km2_the_ramp_wets_that_a_flat_surface_leaves_dry"] == 0.0
        assert claim["if_it_were_painted"]["cells"] == 15381
        text = anim.ramp_claim_clause(claim)
        assert "HOW BIG THE NEW CLAIM IS: none" in text
        assert "MARKED and not painted" in text

    def test_a_frame_that_says_nothing_about_its_corridor_says_where_that_came_from(
        self
    ):
        """No silent assumption: the field records which source answered."""
        bare = anim.ramp_new_wet_ground({"spill": {}}, corridor_mode="ramp")
        assert bare["corridor_is_painted"] is True
        assert "the mode this call was given" in bare["corridor_is_painted_read_from"]
        assert bare["cells_the_ramp_wets_that_a_flat_surface_leaves_dry"] is None
        assert "not measured" in anim.ramp_claim_clause(bare)

    def test_the_clause_is_drawn_on_the_canvas_and_carried_in_the_manifest(self):
        frame = {
            "epoch_ka": 16.5,
            "basin_levels": {
                "caspian": {"level_m": 29.94438934326172},
                "pontic": {"level_m": -18.020370483398438},
            },
            "sill": [
                {
                    "pair": ["caspian", "pontic"],
                    "measured_saddle_m": 28.000944137573242,
                    "measured_lonlat": [44.110416666666666, 45.735416666666666],
                    "expected_m": 27.701019287109375,
                    "tolerance_m": 2.0,
                    "gap_m": 0.2999248504638672,
                    "verdict": "WITHIN_TOLERANCE",
                }
            ],
            **self._spill(),
        }
        added = anim.corridor_addendum(frame, corridor_mode="ramp")
        assert "HOW BIG THE NEW CLAIM IS" in added["text"]
        assert "15,381 cell(s)" in added["text"]
        assert "2,290 sq km" in added["text"]
        assert "243 cell(s)" in added["text"]
        # the manifest carries the same object the sentence was built from
        assert added["ramp_new_wet_ground"][
            "cells_the_ramp_wets_that_a_flat_surface_leaves_dry"
        ] == 15381
        assert added["template_carries_no_digit"] is True

    def test_the_sequence_summary_never_sums_the_ground_across_epochs(self):
        frames = [
            {"epoch_ka": 16.5, "png": "a.png",
             "ramp_new_wet_ground": anim.ramp_new_wet_ground(self._spill())},
            {"epoch_ka": 16.0, "png": "b.png",
             "ramp_new_wet_ground": anim.ramp_new_wet_ground(
                 self._spill(
                     corridor={"mode": "step", "cells": 0, "painted": False}
                 )
             )},
        ]
        summary = anim.ramp_claim_summary(frames)
        assert [r["epoch_ka"] for r in summary["per_epoch"]] == [16.5, 16.0]
        assert summary["n_frames_with_a_painted_corridor"] == 1
        assert summary["epochs_that_claim_nothing"] == [16.0]
        assert summary["most_cells"]["epoch_ka"] == 16.5
        assert summary["fewest_cells"]["epoch_ka"] == 16.5
        # no FIELD adds the ground up; the prose may say why not
        assert not [
            key for key in summary
            if any(word in key for word in ("total", "sum", "all_epochs"))
            and not key.startswith("why")
        ]
        assert "corridor_cells_restored_by_the_or" in summary["do_not_confuse_it_with"]

    def test_the_manifest_carries_it_per_frame_and_once_for_the_sequence(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        assert "ramp_claim" in manifest
        for frame in manifest["frames"]:
            assert "ramp_new_wet_ground" in frame


class TestTheInsetPanelDoesNotDrawTheFullFramesShoreline:
    """The erosion the figure title used to deny by saying only "same cells"."""

    def test_the_operator_bite_is_measured_and_not_typed(self):
        bite = anim.measure_despeckle_edge_bite()
        assert bite["edge_bite_cells"] >= 1
        assert bite["cells_lost_of_the_probe"] > 0
        assert bite["closing_iterations"] >= 1 and bite["opening_iterations"] >= 1
        assert "despeckle_ocean_mask" in bite["measured_on"]

    def test_the_bite_follows_the_operator_rather_than_a_remembered_number(
        self, monkeypatch
    ):
        """Change the iterations and the measurement changes with them."""
        import paleoeurope.viz.paleocostline_render as render

        base = anim.measure_despeckle_edge_bite()["edge_bite_cells"]
        monkeypatch.setattr(render, "OCEAN_OPENING_ITERATIONS", 6)
        monkeypatch.setattr(render, "OCEAN_CLOSING_ITERATIONS", 6)
        assert anim.measure_despeckle_edge_bite()["edge_bite_cells"] > base

    def test_the_figure_title_no_longer_says_only_same_cells(self):
        """The one cue a reader with the PNG and no manifest gets."""
        title = anim.INSET_FIGURE_TITLE
        assert "RE-FLOODED" in title
        assert "eroded" in title
        assert "same cells," not in title, (
            "the bare phrase reads as 'the same water', which is what it is not"
        )

    def test_the_erosion_is_reported_with_the_two_panels_own_counts(self):
        report = anim.inset_edge_erosion(
            {
                "morphology": {"cells_removed": 7315, "cells_added": 3438},
                "grid": {"width": 7800, "height": 4320},
            },
            {"morphology": {"cells_removed": 8349, "cells_added": 773}},
            {"window": {"width": 3144, "height": 1104}},
        )
        assert report["crop_perimeter_cells"] == 2 * (3144 + 1104) - 4
        assert report["upper_bound_on_the_eroded_rim_cells"] == (
            report["crop_perimeter_cells"] * report["operator_bite"]["edge_bite_cells"]
        )
        assert report["inset_panel_despeckle"]["cells_removed"] == 8349
        assert report["full_panel_despeckle"]["cells_removed"] == 7315
        # the inset loses about one cell per border cell; the whole scene loses far
        # less per border cell, which is what makes the rim visible in the numbers
        assert report["cells_removed_per_perimeter_cell"] > (
            report["full_panel_cells_removed_per_perimeter_cell"]
        )
        assert "UPPER BOUND" in report["how_the_upper_bound_is_built"]

    def test_it_refuses_to_estimate_the_part_it_cannot_read(self):
        report = anim.inset_edge_erosion(None, None, None)
        assert report["crop_perimeter_cells"] is None
        assert report["upper_bound_on_the_eroded_rim_cells"] is None
        assert report["cells_removed_per_perimeter_cell"] is None
        assert report["inset_panel_despeckle"]["cells_removed"] is None

    def test_the_caption_says_it_where_a_png_only_reader_will_see_it(self):
        caption = anim.compose_caption(
            16.5,
            None,
            {
                "basins_sidecar": {
                    "sills": [
                        {
                            "pair": ["caspian", "pontic"],
                            "elevation_m": 28.000944,
                            "expectation_status": "CHECKED",
                        }
                    ]
                }
            },
            anim.InsetPlan(epoch_ka=16.5, extent=anim.MANYCH_INSET),
            edge_erosion=anim.inset_edge_erosion(
                {"morphology": {"cells_removed": 7315}},
                {"morphology": {"cells_removed": 8349}},
                {"window": {"width": 3144, "height": 1104}},
            ),
        )
        assert "DO NOT DRAW THE SAME SHORELINE" in caption
        assert "8,492 cell(s)" in caption          # the crop's own perimeter
        assert "8,349 cell(s)" in caption          # what its de-speckle removed
        assert not any(c.isdigit() for c in anim.INSET_EROSION_TEMPLATE)

    def test_a_frame_with_no_erosion_measurement_says_nothing_about_it(self):
        """Silence, not an implied zero."""
        caption = anim.compose_caption(
            16.5, None, {"basins_sidecar": {"sills": []}},
            anim.InsetPlan(epoch_ka=16.5, extent=anim.MANYCH_INSET),
        )
        assert "SAME SHORELINE" not in caption


class TestTheInsetGuardSentenceCarriesOnlyItsOwnObjectsFigures:

    def test_the_sentence_is_a_digit_free_template(self):
        assert not any(c.isdigit() for c in anim.INSET_GUARD_TEMPLATE)

    def test_the_retired_sentence_quoted_grids_the_object_does_not_carry(self):
        """+8.30 m at 2000 px and +5.24 m at 4000 px were in no field beside it."""
        assert "8.30" not in anim.INSET_GUARD_TEMPLATE
        assert "5.24" not in anim.INSET_GUARD_TEMPLATE
        assert "2000 px" not in anim.INSET_GUARD_TEMPLATE
        assert "4000 px" not in anim.INSET_GUARD_TEMPLATE

    def test_every_figure_in_the_filled_sentence_is_a_field_of_the_same_object(
        self, tmp_path
    ):
        source = _toy_tagged(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(_toy_plan(source), anim.MANYCH_INSET, tmp_path / "inset")
        block = anim._inset_frame_block(
            cut,
            {
                "basins_sidecar": {
                    "sills": [
                        {
                            "pair": ["caspian", "pontic"],
                            "elevation_m": 28.000944137573242,
                            "lon": anim.MANYCH_CREST_MEASURED_LON,
                            "lat": anim.MANYCH_CREST_MEASURED_LAT,
                            "measured_on": "dem_sm at max_px, 1104x3144 cells",
                        }
                    ]
                },
                "png": "x.png",
            },
            tolerance_m=2.0,
        )
        sill = block["sill"]
        sentence = sill["why_the_inset_is_guarded_separately"]
        assert f"{sill['measured_saddle_m']:+.4f} m" in sentence
        assert f"{sill['expected_m']:+.4f} m" in sentence
        assert f"{sill['gap_m']:+.4f} m" in sentence
        assert f"{sill['tolerance_m']:.2f} m" in sentence
        assert sill["measured_on"] in sentence


class TestTheCorridorModeIsCarriedNotAssumed:

    def test_corridor_note_reports_the_mode_it_was_given(self):
        note = anim.corridor_note([], "ramp")
        assert note["corridor"] == "ramp"
        assert anim.corridor_note([], "step")["corridor"] == "step"

    def test_the_verdict_sentence_does_not_call_a_ramp_build_a_step_build(self):
        """It shipped 'step is kept by choice' inside an object saying ramp."""
        ramp = anim.corridor_note([], "ramp")["what_the_verdict_means"]
        step = anim.corridor_note([], "step")["what_the_verdict_means"]
        assert "'ramp', what this run asked for" in ramp
        assert "'step', what this run asked for" in step
        assert "'step' is kept by choice" not in ramp

    def test_the_rule_no_longer_hard_codes_step(self):
        assert "corridor='step'." not in anim.CORRIDOR_RULE

    def test_build_animation_refuses_an_unknown_corridor(self, untagged_topo_dir,
                                                         tmp_path):
        with pytest.raises(ValueError, match="corridor="):
            anim.build_animation(
                out_root=tmp_path / "out", topo_dir=untagged_topo_dir,
                epochs=[16.5], corridor="slope", plan_only=True,
            )

    def test_the_cli_offers_both_modes_and_defaults_to_step(self, capsys):
        with pytest.raises(SystemExit):
            anim._cli(["--help"])
        text = capsys.readouterr().out
        assert "--corridor" in text and "--inset" in text


@_shipped
class TestRampWasVerifiedOnTheNineShippedEpochs:
    """The predecessor's claim, re-derived from the shipped measurements.

    The claim inherited into this phase was that ramp is legal because "the
    pontic<->mediterranean saddle sits at +13.345 m, above every Pontic surface in
    the window".  That figure is ONE epoch's -- 16.5 ka.  The sentence was true and
    the number in it was not a constant.  This re-runs the test per epoch, on the
    saddles each frame measured for itself.

    AND THE REPLACEMENT SENTENCE HAD ITS OWN VERSION OF THE SAME FAULT.  It read
    "measured across the nine the saddle RUNS +13.325 to +16.064 m", which reads as a
    series through time and is not one: those are the ASCENDING SORT's endpoints.  In
    epoch order from 17.0 to 13.0 ka the nine are +13.476 / +13.345 / +13.325 /
    +13.528 / +14.099 / +14.092 / +16.064 / +15.363 / +14.919 m -- the smallest at
    16.0 ka and the largest at 14.0 ka, with the series rising and falling twice.
    :meth:`test_the_saddle_series_is_not_monotone_so_the_sorted_pair_is_not_the_run`
    is the guard, and ``corridor_note`` now ships the series by epoch beside the
    sorted pair with each named for what it is.
    """

    @staticmethod
    def _frames():
        manifest = json.loads(
            (SHIPPED_ROOT / "spill_0.5ka_9_epochs" / "sequence_manifest.json")
            .read_text(encoding="utf-8")
        )
        return manifest["frames"]

    def test_at_most_one_pair_could_merge_at_every_one_of_the_nine(self):
        legality = anim.ramp_legality(self._frames(), "ramp")
        assert legality["n_epochs_examined"] == 9
        assert legality["verdict"] == "RAMP_IS_DEFINED_AT_EVERY_EPOCH_OF_THIS_RUN"
        assert legality["epochs_that_raised_CorridorNotRampableError"] == []
        for epoch in legality["per_epoch"]:
            assert epoch["n_undecided"] == 0, epoch
            assert epoch["n_could_be_donor_receiver"] <= 1, epoch
        anim.assert_ramp_is_legal(legality)

    def test_the_only_pair_that_can_ever_merge_is_caspian_pontic(self):
        for epoch in anim.ramp_legality(self._frames(), "ramp")["per_epoch"]:
            for pair in epoch["pairs_that_could_merge"]:
                assert pair == ["caspian", "pontic"], epoch["epoch_ka"]

    def test_the_pontic_mediterranean_saddle_is_not_the_constant_it_was_quoted_as(self):
        saddles = []
        for frame in self._frames():
            entry = next(
                s for s in frame["sill"]
                if sorted(s["pair"]) == ["mediterranean", "pontic"]
            )
            saddles.append(entry["measured_saddle_m"])
        assert len({round(v, 6) for v in saddles}) == 9, (
            "nine epochs, nine different saddles -- one of them is not the window"
        )
        assert min(saddles) == pytest.approx(13.325, abs=0.01)
        assert max(saddles) == pytest.approx(16.064, abs=0.01)
        for frame, saddle in zip(self._frames(), saddles):
            assert frame["basin_levels"]["pontic"]["level_m"] < saddle

    def test_the_saddle_series_is_not_monotone_so_the_sorted_pair_is_not_the_run(self):
        """A min..max pair may stand for a series ONLY if the series is monotone.

        It is not: the smallest of the nine is at the THIRD epoch and the largest at
        the SEVENTH, so "runs +13.325 to +16.064 m" describes the sort and not the
        window.  ``corridor_note`` therefore ships both, each named, and this pins
        the property that makes the distinction necessary rather than cosmetic.
        """
        frames = self._frames()
        note = anim.corridor_note(frames, "step")
        grid = note["on_the_render_grid"]
        by_epoch = grid["measured_saddle_by_epoch"]
        assert [row[0] for row in by_epoch] == [f["epoch_ka"] for f in frames], (
            "the series must be in EPOCH order, which is the order the frames are in"
        )
        assert grid["measured_saddle_is_monotone_in_time"] is False
        low, high = grid["measured_saddle_min_max_m_sorted_not_chronological"]
        assert low == min(row[1] for row in by_epoch)
        assert high == max(row[1] for row in by_epoch)
        assert (by_epoch[0][1], by_epoch[-1][1]) != (low, high), (
            "the first and last epoch's readings are not the sorted endpoints, "
            "which is exactly why the pair must not be read as a run through time"
        )
        assert grid["measured_saddle_lowest_at_epoch_ka"] == 16.0
        assert grid["measured_saddle_highest_at_epoch_ka"] == 14.0

    def test_caspian_and_mediterranean_never_touch_at_any_epoch(self):
        for frame in self._frames():
            entry = next(
                s for s in frame["sill"]
                if sorted(s["pair"]) == ["caspian", "mediterranean"]
            )
            assert entry["measured_saddle_m"] is None, frame["epoch_ka"]


@_dataset
class TestTheInsetOnTheRealRaster:

    def test_the_inset_resolves_the_same_saddle_cell_as_the_full_scene(self, tmp_path):
        """The measurement the whole inset rests on, re-run from the registry.

        Both grids are native, but they are different arrays: the inset's dem_sm is
        a Gaussian over a different domain and its priority flood runs over a
        different graph.  That they agree is a result, not a definition.
        """
        from scipy.ndimage import gaussian_filter
        from paleoeurope.viz.paleocostline_render import (
            MASK_SMOOTHING_SIGMA_PX, _measure_sills, _priority_flood_ownership,
        )

        source = anim.raster_for(16.5)
        record = anim.crop_to_extent(
            source, anim.MANYCH_INSET, tmp_path / "manych_inset.tif"
        )
        assert record["window"] == {
            "col_off": 2400, "row_off": 1488, "width": 3144, "height": 1104
        }
        levels = {"caspian": 29.94438934326172, "pontic": -18.020370483398438}
        seeds = anim.resolve_seeds(
            record["inset_path"], levels, epoch_ka=16.5,
            raster_sha256=record["inset_sha256"],
            boxes=anim.MANYCH_INSET.seed_boxes,
        )
        with rasterio.open(record["inset_path"]) as dataset:
            dem = dataset.read(1).astype("float32")
            dem = np.where(dem == dataset.nodata, np.nan, dem)
            transform = dataset.transform
        nodata = ~np.isfinite(dem)
        dem_sm = gaussian_filter(
            np.where(nodata, 0.0, dem), sigma=MASK_SMOOTHING_SIGMA_PX
        )
        dem_sm = np.where(nodata, np.nan, dem_sm).astype("float64")
        seed_rc = []
        for seed in sorted(seeds, key=lambda s: s.basin):
            col, row = ~transform * (seed.lon, seed.lat)
            seed_rc.append((int(row), int(col)))
        basin_id, bottleneck, _parent = _priority_flood_ownership(
            dem_sm, seed_rc, max(levels.values()) + 100.0
        )
        value, (row, col), _partner = _measure_sills(basin_id, bottleneck, 2)[(1, 2)]
        lon, lat = transform * (col + 0.5, row + 0.5)
        # the full scene at max_px=7800 reads +28.000944137573242 m at this cell
        assert value == pytest.approx(28.000944137573242, abs=1e-6)
        assert lon == pytest.approx(anim.MANYCH_CREST_MEASURED_LON, abs=1e-9)
        assert lat == pytest.approx(anim.MANYCH_CREST_MEASURED_LAT, abs=1e-9)
        assert anim.MANYCH_INSET.contains(lon, lat)

    @_needs_curves
    def test_both_inset_seed_boxes_are_wet_at_every_epoch_of_the_window(self, tmp_path):
        """The condition that FORCED the extent's bounds, checked on every raster."""
        floors = {"caspian": None, "pontic": None}
        for epoch in (17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0):
            plan = anim.plan_epoch(epoch)
            assert plan.renderable, plan.refusal
            cut = anim.plan_inset(plan, anim.MANYCH_INSET, tmp_path / "inset")
            assert cut.refusal is None, f"{epoch} ka: {cut.refusal}"
            for seed in cut.seeds:
                margin = seed.level_m - seed.elevation_m
                assert margin > 0.0, f"{epoch} ka {seed.basin} is dry"
                floors[seed.basin] = (
                    margin if floors[seed.basin] is None
                    else min(floors[seed.basin], margin)
                )
            Path(cut.raster_path).unlink()
        assert floors["caspian"] > 80.0
        assert floors["pontic"] > 2000.0


# ---------------------------------------------------------------------------
# THE TIGHT EXTENT  (2026-08-05, S-SCOPE)
#
# The owner asked for the corridor to BE the frame rather than an inset in a scene.
# How close the frame can get is a MEASUREMENT and not a layout choice, and this
# block is where the measurement is pinned:
#
#   * every bound of MANYCH_CORRIDOR_TIGHT is one step back from a cliff that was
#     measured at 16.5 ka on each candidate's own crop, own gaussian and own flood,
#     and then re-checked on all nine of the window's own rasters;
#   * one of the four cliffs is INVISIBLE to the sill guard -- the Kerch/Taman
#     gateway pair, 2.51 m apart and both far below the controlling col -- so it is
#     measured separately and registered in must_contain;
#   * MANYCH_INSET is NOT edited. It shipped; artifacts name it. A second extent is
#     a second registry entry, and the registry is asked for the list rather than
#     told it.
# ---------------------------------------------------------------------------

#: Deeps placed inside the TIGHT extent's own seed boxes.  A separate fixture from
#: ``_toy_tagged`` on purpose: adding cells to the shared toy raster would move
#: argmins the other classes assert on.
_TOY_TIGHT_DEEPS = {
    "caspian_tight": (45.6, 45.9, -700.0),
    "pontic_tight": (37.2, 44.7, -1500.0),
}


def _toy_tagged_tight(path: Path) -> Path:
    return _write_scene_raster(
        path,
        deeps={**_TOY_DEEPS, **_TOY_INSET_DEEPS, **_TOY_TIGHT_DEEPS},
        tags={"vertical_frame": _F0},
    )


def _px_per_deg(extent_wsen):
    """Pixels per degree the renderer's own figure setup gives this extent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = RenderConfig()
    w, s, e, n = extent_wsen
    fig, ax = plt.subplots(figsize=(cfg.fig_size_in, cfg.fig_size_in), dpi=cfg.dpi)
    ax.imshow(
        np.zeros((8, 8)), extent=[w, e, s, n], origin="upper",
        aspect="equal", interpolation="bilinear",
    )
    fig.canvas.draw()
    box = ax.get_window_extent()
    plt.close(fig)
    return box.width / (e - w), box.height / (n - s)


class TestTheTightExtentIsAMeasurementNotAPreference:

    def test_it_contains_both_of_the_points_it_is_required_to_contain(self):
        tight = anim.MANYCH_CORRIDOR_TIGHT
        required = {name: (lon, lat) for name, lon, lat in tight.must_contain}
        assert len(required) == 2
        assert tight.contains(
            anim.MANYCH_CREST_MEASURED_LON, anim.MANYCH_CREST_MEASURED_LAT
        )
        assert tight.contains(
            anim.AZOV_PONTIC_COL_KERCH_LON, anim.AZOV_PONTIC_COL_KERCH_LAT
        )
        assert any("crest" in name for name in required)
        assert any("Kerch col" in name for name in required)

    def test_an_extent_that_loses_the_gateway_refuses_instead_of_redrawing_it(self):
        """The whole point of registering the col: the sill guard would not notice.

        A west bound east of the Kerch col still measures the same +28.000944 m
        saddle at the same crest cell -- that was measured, and it is why the col is
        a ``must_contain`` point rather than a comment.
        """
        with pytest.raises(anim.InsetExtentError, match="does not contain"):
            anim.InsetExtent(
                name="lost_the_kerch_col",
                lon_min=36.7, lat_min=44.6, lon_max=46.0, lat_max=47.6,
                why="deliberately east of the measured Kerch col",
                seed_boxes=anim.MANYCH_CORRIDOR_TIGHT.seed_boxes,
                must_contain=anim.MANYCH_CORRIDOR_TIGHT.must_contain,
            )

    def test_the_two_gateway_cols_are_the_pair_the_reason_quotes(self):
        """A subtraction, not a second copy of the answer."""
        why = anim.MANYCH_CORRIDOR_TIGHT.why
        gap = anim.AZOV_PONTIC_COL_TAMAN_M - anim.AZOV_PONTIC_COL_KERCH_M
        east = anim.AZOV_PONTIC_COL_TAMAN_LON - anim.AZOV_PONTIC_COL_KERCH_LON
        assert f"{gap:.6f} m higher" == "2.505670 m higher"
        assert "2.505670 m higher" in why
        assert f"{east:.6f} deg further east" == "0.633333 deg further east"
        assert "0.633333 deg further east" in why
        assert anim.AZOV_PONTIC_COL_TAMAN_M > anim.AZOV_PONTIC_COL_KERCH_M, (
            "the Taman line is the one that survives cutting the Kerch line off, so "
            "it must be the HIGHER of the two or the pair is the wrong way round"
        )
        assert "connected-path minimax" in anim.AZOV_PONTIC_COL_RULE
        assert "dem_sm" in anim.AZOV_PONTIC_COL_RULE

    def test_the_west_bound_keeps_the_gateway_inside_by_the_stated_clearance(self):
        tight = anim.MANYCH_CORRIDOR_TIGHT
        clearance = anim.AZOV_PONTIC_COL_KERCH_LON - tight.lon_min
        assert clearance == pytest.approx(0.210417, abs=5e-7)
        assert clearance / (1 / 240) == pytest.approx(50.5, abs=0.05)
        assert "0.210417 deg" in tight.why
        assert "50.5 cells" in tight.why

    def test_every_bound_and_every_box_carries_a_reason(self):
        tight = anim.MANYCH_CORRIDOR_TIGHT
        assert tight.why.strip()
        for bound in ("WEST", "SOUTH", "EAST", "NORTH"):
            assert bound in tight.why, f"no reason given for {bound}"
        for box in tight.seed_boxes:
            assert box.why.strip()
        assert tight.basins() == ("caspian", "pontic")
        assert tight.pairs() == (("caspian", "pontic"),)

    def test_it_could_not_have_inherited_any_seed_box_this_module_already_has(self):
        """A third extent forced a third registry, stated as an assertion."""
        tight = anim.MANYCH_CORRIDOR_TIGHT
        for box in (*anim.SEED_BOXES, *anim.MANYCH_INSET.seed_boxes):
            if box.basin not in tight.basins():
                continue
            inside = (
                tight.lon_min <= box.lon_min and box.lon_max <= tight.lon_max
                and tight.lat_min <= box.lat_min and box.lat_max <= tight.lat_max
            )
            assert not inside, f"{box.name} is inside the tight extent after all"

    def test_it_is_a_strict_sub_extent_of_the_shipped_inset(self):
        """It cuts the shipped inset on all four sides; it does not move any of them
        outward, so every cell it holds is a cell that inset already held."""
        tight, shipped = anim.MANYCH_CORRIDOR_TIGHT, anim.MANYCH_INSET
        assert tight.lon_min > shipped.lon_min
        assert tight.lat_min > shipped.lat_min
        assert tight.lon_max < shipped.lon_max
        assert tight.lat_max < shipped.lat_max

    def test_the_shipped_inset_was_not_edited_to_get_here(self):
        """Its frames and manifests name it; an extent that moved under its own name
        would make every one of those artifacts unreproducible."""
        shipped = anim.MANYCH_INSET
        assert shipped.bounds() == (35.5, 43.2, 48.6, 47.8)
        assert shipped.name == "manych_corridor_and_both_receiving_seas"
        assert [b.name for b in shipped.seed_boxes] == [
            "caspian_mangyshlak_shelf_inset", "pontic_northeast_abyssal_inset"
        ]
        assert len(shipped.must_contain) == 1

    def test_the_frame_arithmetic_the_reason_quotes_is_recomputed_here(self):
        tight, shipped = anim.MANYCH_CORRIDOR_TIGHT, anim.MANYCH_INSET
        t_w, t_h = tight.lon_max - tight.lon_min, tight.lat_max - tight.lat_min
        s_w, s_h = shipped.lon_max - shipped.lon_min, shipped.lat_max - shipped.lat_min
        assert (round(t_w, 6), round(t_h, 6)) == (9.6, 3.0)
        assert f"{t_w * t_h:.2f}" == "28.80"
        assert f"{s_w * s_h:.2f}" == "60.26"
        assert f"{100 * (t_w * t_h) / (s_w * s_h):.2f} %" == "47.79 %"
        for text in ("9.6 x 3.0 deg", "28.80 of 60.26", "47.79 %"):
            assert text in tight.why

    def test_the_magnification_is_the_width_ratio_in_both_axes(self):
        """Measured on the renderer's own figure setup, exactly as MANYCH_INSET's is.

        Both extents are wider than the square axes box, so both are width-limited
        and the linear scale is the WIDTH ratio in longitude and latitude alike.
        """
        scene = (25.5, 36.0, 58.0, 54.0)
        full_x, full_y = _px_per_deg(scene)
        tight_x, tight_y = _px_per_deg(anim.MANYCH_CORRIDOR_TIGHT.bounds())
        inset_x, _inset_y = _px_per_deg(anim.MANYCH_INSET.bounds())
        assert tight_x / full_x == pytest.approx(3.3854, abs=0.001)
        assert tight_y / full_y == pytest.approx(3.3854, abs=0.001)
        assert tight_x / full_x == pytest.approx(tight_y / full_y, rel=1e-9)
        assert tight_x / inset_x == pytest.approx(1.3646, abs=0.001)
        why = anim.MANYCH_CORRIDOR_TIGHT.why
        assert "3.3854x the full panel" in why
        assert "2.4809x" in why
        assert "1.3646x more" in why

    def test_what_the_caution_costs_is_a_number_and_it_is_the_right_one(self):
        """The tightest extent that measured OK at all nine epochs was
        36.6 / 44.8 / 45.4 / 47.2; the registered one is wider on every side and the
        reason says by how much in linear scale rather than calling it 'safer'."""
        why = anim.MANYCH_CORRIDOR_TIGHT.why
        scene_width, registered, extreme = 32.5, 9.6, 8.8
        assert f"{scene_width / extreme:.4f}x" == "3.6932x"
        assert (
            f"{100 * ((scene_width / extreme) / (scene_width / registered) - 1):.2f} %"
            == "9.09 %"
        )
        assert "36.6 / 44.8 / 45.4 / 47.2" in why
        assert "8.8 x 2.4 deg" in why
        assert "3.6932x" in why
        assert "9.09 % more linear scale" in why

    def test_the_route_is_not_offered_as_a_margin(self):
        """The shipped inset quoted a route clearance as a bound's reason.  On this
        extent the route touches three edges, and the reason says why that is not a
        finding: below the controlling col the parent chain's path is a tie-break."""
        why = anim.MANYCH_CORRIDOR_TIGHT.why
        assert "ROUTE CLEARANCE IS NOT A MARGIN ON THIS EXTENT" in why
        assert "equivalence class" in why
        assert "flat-index tie-break" in why


class TestTheExtentRegistryIsAskedRatherThanTold:

    def test_the_registry_holds_both_extents_and_their_names_are_unique(self):
        names = [e.name for e in anim.INSET_EXTENTS]
        assert names == sorted(set(names), key=names.index), "duplicate extent name"
        assert anim.MANYCH_INSET in anim.INSET_EXTENTS
        assert anim.MANYCH_CORRIDOR_TIGHT in anim.INSET_EXTENTS

    def test_lookup_round_trips_and_refuses_an_unregistered_name(self):
        for extent in anim.INSET_EXTENTS:
            assert anim.inset_extent_by_name(extent.name) is extent
        with pytest.raises(anim.InsetExtentError, match="no InsetExtent named"):
            anim.inset_extent_by_name("manych_corridor_that_was_never_registered")

    def test_the_cli_offers_exactly_the_registry_and_nothing_else(self, capsys):
        """Read off the CLI's own ``--help``, not off a list copied beside it.

        ``--inset`` used to name ONE extent, spelled out twice in its own help text.
        Its choices now come from :data:`INSET_EXTENTS`, so a registered extent is
        offerable the moment it is registered and an unregistered name is refused by
        argparse before anything is planned.
        """
        with pytest.raises(SystemExit) as exit_info:
            anim._cli(["--help"])
        assert exit_info.value.code == 0
        help_text = capsys.readouterr().out
        for extent in anim.INSET_EXTENTS:
            assert extent.name in help_text, f"{extent.name} is not offered"

        with pytest.raises(SystemExit) as refused:
            anim._cli(["--inset", "not_a_registered_extent", "--plan-only"])
        assert refused.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


@_dataset
class TestTheTightExtentOnTheRealRasters:

    def test_the_crop_window_and_the_saddle_are_what_the_registry_claims(self, tmp_path):
        """The measurement the tight extent rests on, re-run from the registry.

        Same recipe as ``TestTheInsetOnTheRealRaster``: a different array, a
        different gaussian domain and a different flood graph, agreeing with the
        full scene's saddle to the last float64 bit is a RESULT, not a definition.
        """
        from scipy.ndimage import gaussian_filter
        from paleoeurope.viz.paleocostline_render import (
            MASK_SMOOTHING_SIGMA_PX, _measure_sills, _priority_flood_ownership,
        )

        tight = anim.MANYCH_CORRIDOR_TIGHT
        source = anim.raster_for(16.5)
        record = anim.crop_to_extent(source, tight, tmp_path / "tight.tif")
        assert record["window"] == {
            "col_off": 2616, "row_off": 1536, "width": 2304, "height": 720
        }
        assert record["verdict"] == "CELL_FOR_CELL_IDENTICAL_TO_THE_SOURCE_WINDOW"
        levels = {"caspian": 29.94438934326172, "pontic": -18.020370483398438}
        seeds = anim.resolve_seeds(
            record["inset_path"], levels, epoch_ka=16.5,
            raster_sha256=record["inset_sha256"], boxes=tight.seed_boxes,
        )
        by_basin = {s.basin: s for s in seeds}
        assert by_basin["caspian"].lon == pytest.approx(45.722917, abs=1e-6)
        assert by_basin["caspian"].lat == pytest.approx(46.093750, abs=1e-6)
        assert by_basin["caspian"].elevation_m == pytest.approx(-19.728, abs=1e-3)
        assert by_basin["pontic"].lon == pytest.approx(37.260417, abs=1e-6)
        assert by_basin["pontic"].lat == pytest.approx(44.622917, abs=1e-6)
        assert by_basin["pontic"].elevation_m == pytest.approx(-1219.323, abs=1e-3)

        with rasterio.open(record["inset_path"]) as dataset:
            dem = dataset.read(1).astype("float32")
            dem = np.where(dem == dataset.nodata, np.nan, dem)
            transform = dataset.transform
        nodata = ~np.isfinite(dem)
        dem_sm = gaussian_filter(
            np.where(nodata, 0.0, dem), sigma=MASK_SMOOTHING_SIGMA_PX
        )
        dem_sm = np.where(nodata, np.nan, dem_sm).astype("float64")
        seed_rc = []
        for seed in sorted(seeds, key=lambda s: s.basin):
            col, row = ~transform * (seed.lon, seed.lat)
            seed_rc.append((int(row), int(col)))
        basin_id, bottleneck, _parent = _priority_flood_ownership(
            dem_sm, seed_rc, max(levels.values()) + 100.0
        )
        value, (row, col), _partner = _measure_sills(basin_id, bottleneck, 2)[(1, 2)]
        lon, lat = transform * (col + 0.5, row + 0.5)
        assert value == pytest.approx(28.000944137573242, abs=1e-6)
        assert lon == pytest.approx(anim.MANYCH_CREST_MEASURED_LON, abs=1e-9)
        assert lat == pytest.approx(anim.MANYCH_CREST_MEASURED_LAT, abs=1e-9)
        assert tight.contains(lon, lat)

    def test_the_two_gateway_cols_reproduce_on_their_own_window(self):
        """Both constants, re-measured on the raster, on ``dem_sm``, from scratch."""
        from scipy.ndimage import gaussian_filter
        from paleoeurope.viz.paleocostline_render import (
            MASK_SMOOTHING_SIGMA_PX, _measure_sills, _priority_flood_ownership,
        )

        source = anim.raster_for(16.5)
        with rasterio.open(source) as dataset:
            raw = rasterio.windows.from_bounds(
                35.8, 43.8, 38.6, 46.6, dataset.transform
            )
            window = rasterio.windows.Window(
                int(round(raw.col_off)), int(round(raw.row_off)),
                int(round(raw.width)), int(round(raw.height)),
            )
            dem = dataset.read(1, window=window).astype("float32")
            dem = np.where(dem == dataset.nodata, np.nan, dem)
            transform = dataset.window_transform(window)
        nodata = ~np.isfinite(dem)
        dem_sm = gaussian_filter(
            np.where(nodata, 0.0, dem), sigma=MASK_SMOOTHING_SIGMA_PX
        )
        dem_sm = np.where(nodata, np.nan, dem_sm).astype("float64")

        def rc(lon, lat):
            col, row = ~transform * (lon, lat)
            return int(row), int(col)

        lowland = rc(37.20, 46.20)          # the palaeo-Azov lowland
        deep = rc(37.60, 44.20)             # the eastern Black Sea
        lons = np.array(
            [(transform * (c + 0.5, 0.5))[0] for c in range(dem.shape[1])]
        )

        def col_with_west_bound(west):
            work = np.where((lons >= west)[None, :], dem_sm, np.nan)
            basin_id, bottleneck, _p = _priority_flood_ownership(
                work, [lowland, deep], 900.0
            )
            value, (row, column), _partner = _measure_sills(basin_id, bottleneck, 2)[
                (1, 2)
            ]
            lon, lat = transform * (column + 0.5, row + 0.5)
            return float(value), float(lon), float(lat)

        kerch = col_with_west_bound(35.8)
        taman = col_with_west_bound(36.7)
        assert kerch[0] == pytest.approx(anim.AZOV_PONTIC_COL_KERCH_M, abs=1e-6)
        assert kerch[1] == pytest.approx(anim.AZOV_PONTIC_COL_KERCH_LON, abs=1e-9)
        assert kerch[2] == pytest.approx(anim.AZOV_PONTIC_COL_KERCH_LAT, abs=1e-9)
        assert taman[0] == pytest.approx(anim.AZOV_PONTIC_COL_TAMAN_M, abs=1e-6)
        assert taman[1] == pytest.approx(anim.AZOV_PONTIC_COL_TAMAN_LON, abs=1e-9)
        assert taman[2] == pytest.approx(anim.AZOV_PONTIC_COL_TAMAN_LAT, abs=1e-9)
        assert taman[0] > kerch[0]

    def test_the_south_bound_one_step_tighter_is_a_refusal_not_a_squeeze(self):
        """44.9 N works at 16.5 ka and is DRY at 17.0 and 13.0 ka.

        This is the failure mode the nine-epoch check exists for, reproduced on the
        rasters that produce it: the deepest Pontic water a 44.9 N extent could hold
        stands ABOVE two of the window's nine Pontic surfaces, so ``resolve_seed``
        raises rather than seeding a sea that is not there.
        """
        would_be = anim.SeedBox(
            basin="pontic", name="pontic_if_the_south_bound_were_44_9",
            lon_min=36.85, lat_min=44.91, lon_max=37.40, lat_max=44.96,
            why="the deepest Pontic water a south bound of 44.9 N would leave inside",
        )
        # 16.5 ka: it resolves, which is exactly why one epoch is not a check
        resolved = anim.resolve_seed(
            would_be, anim.raster_for(16.5),
            level_m=-18.020370483398438, epoch_ka=16.5,
        )
        assert resolved.elevation_m < -18.020370483398438
        for epoch, level in ((17.0, -40.346136), (13.0, -34.802301)):
            with pytest.raises(anim.SeedAboveItsLevelError, match="not below basin"):
                anim.resolve_seed(
                    would_be, anim.raster_for(epoch),
                    level_m=level, epoch_ka=epoch,
                )
        # and the registered bound does resolve on the same two rasters
        registered = next(
            b for b in anim.MANYCH_CORRIDOR_TIGHT.seed_boxes if b.basin == "pontic"
        )
        for epoch, level in ((17.0, -40.346136), (13.0, -34.802301)):
            seed = anim.resolve_seed(
                registered, anim.raster_for(epoch), level_m=level, epoch_ka=epoch
            )
            assert level - seed.elevation_m > 1000.0

    @_needs_curves
    def test_both_boxes_are_wet_and_the_saddle_holds_at_every_epoch(self, tmp_path):
        """The nine-epoch check, through the module's own machinery.

        An extent that works at 16.5 ka and fails at 13.0 ka is not usable for a
        sequence, so every epoch gets its own raster, its own crop, its own seeds
        and its own flood -- and the saddle it measures must be the one the guard
        expects, at the registered crest cell, on all nine.
        """
        from scipy.ndimage import gaussian_filter
        from paleoeurope.viz.paleocostline_render import (
            MASK_SMOOTHING_SIGMA_PX, SILL_TOLERANCE_M, _measure_sills,
            _priority_flood_ownership,
        )

        tight = anim.MANYCH_CORRIDOR_TIGHT
        floors = {"caspian": None, "pontic": None}
        for epoch in (17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0):
            plan = anim.plan_epoch(epoch)
            assert plan.renderable, plan.refusal
            cut = anim.plan_inset(plan, tight, tmp_path / "tight")
            assert cut.refusal is None, f"{epoch} ka: {cut.refusal}"
            levels = {b: plan.levels[b].level_m for b in tight.basins()}
            for seed in cut.seeds:
                margin = seed.level_m - seed.elevation_m
                assert margin > 0.0, f"{epoch} ka {seed.basin} is dry"
                floors[seed.basin] = (
                    margin if floors[seed.basin] is None
                    else min(floors[seed.basin], margin)
                )
            with rasterio.open(cut.raster_path) as dataset:
                dem = dataset.read(1).astype("float32")
                dem = np.where(dem == dataset.nodata, np.nan, dem)
                transform = dataset.transform
            nodata = ~np.isfinite(dem)
            dem_sm = gaussian_filter(
                np.where(nodata, 0.0, dem), sigma=MASK_SMOOTHING_SIGMA_PX
            )
            dem_sm = np.where(nodata, np.nan, dem_sm).astype("float64")
            seed_rc = []
            for seed in sorted(cut.seeds, key=lambda s: s.basin):
                col, row = ~transform * (seed.lon, seed.lat)
                seed_rc.append((int(row), int(col)))
            basin_id, bottleneck, _p = _priority_flood_ownership(
                dem_sm, seed_rc, max(levels.values()) + 100.0
            )
            measured, (row, col), _partner = _measure_sills(
                basin_id, bottleneck, 2
            )[(1, 2)]
            lon, lat = transform * (col + 0.5, row + 0.5)
            expected = anim.manych_expectation(epoch, cut.dem_frame).value_m
            assert abs(measured - expected) <= SILL_TOLERANCE_M, (
                f"{epoch} ka: measured {measured:+.6f} against expected "
                f"{expected:+.6f}"
            )
            assert lon == pytest.approx(anim.MANYCH_CREST_MEASURED_LON, abs=1e-9)
            assert lat == pytest.approx(anim.MANYCH_CREST_MEASURED_LAT, abs=1e-9)
            Path(cut.raster_path).unlink()
        assert floors["caspian"] > 38.0
        assert floors["pontic"] > 1178.0


class TestTheTightExtentPlansOnAToyRaster:
    """The parts that need no dataset: the crop, the seeds and the expectation."""

    def test_plan_inset_resolves_both_seeds_inside_the_tight_extent(self, tmp_path):
        source = _toy_tagged_tight(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(
            _toy_plan(source), anim.MANYCH_CORRIDOR_TIGHT, tmp_path / "inset"
        )
        assert cut.refusal is None, cut.refusal
        assert cut.renderable
        assert {s.basin for s in cut.seeds} == {"caspian", "pontic"}
        for seed in cut.seeds:
            assert anim.MANYCH_CORRIDOR_TIGHT.contains(seed.lon, seed.lat)
            assert seed.elevation_m < seed.level_m
            assert seed.rule == anim.SEED_RULE
        deepest = {s.basin: s.elevation_m for s in cut.seeds}
        assert deepest["caspian"] == pytest.approx(-700.0)
        assert deepest["pontic"] == pytest.approx(-1500.0)

    def test_the_expectation_covers_exactly_the_one_pair_it_declares(self, tmp_path):
        source = _toy_tagged_tight(tmp_path / "khv_16.5ka.tif")
        cut = anim.plan_inset(
            _toy_plan(source), anim.MANYCH_CORRIDOR_TIGHT, tmp_path / "inset"
        )
        expectations = anim.inset_sill_expectations(cut)
        assert set(expectations) == {("caspian", "pontic")}
        mine = expectations[("caspian", "pontic")]
        theirs = anim.sill_expectations_for(16.5, _F0, 2.5)[("caspian", "pontic")]
        assert isinstance(mine, SillExpectation)
        assert mine.value_m == theirs.value_m, (
            "the expectation is a property of the terrain and the vertical frame; a "
            "tighter extent may not be held to a softer one"
        )

    def test_the_crop_carries_the_extent_name_and_the_vertical_frame(self, tmp_path):
        source = _toy_tagged_tight(tmp_path / "khv_16.5ka.tif")
        record = anim.crop_to_extent(
            source, anim.MANYCH_CORRIDOR_TIGHT, tmp_path / "cut.tif"
        )
        with rasterio.open(record["inset_path"]) as cut:
            tags = cut.tags()
        assert tags["inset_extent_name"] == anim.MANYCH_CORRIDOR_TIGHT.name
        assert tags["vertical_frame"] == _F0
        assert record["verdict"] == "CELL_FOR_CELL_IDENTICAL_TO_THE_SOURCE_WINDOW"


# ---------------------------------------------------------------------------
# R-CLOSEUP: the figure is the only lever on the legend, and the close-up page
# ---------------------------------------------------------------------------


def _legend_probe_png(
    path: Path,
    *,
    map_px=(400, 200),
    legend_px=(120, 60),
    water_cols=60,
    legend_alpha=0.28,
    legend_pad_px=6,
    colourbar=False,
) -> Path:
    """A picture shaped like a rendered frame, with every quantity KNOWN.

    White margin, a one-pixel black spine box, green "land", a blue "water" strip
    down the right-hand side, and a TRANSLUCENT dark legend pasted into the
    bottom-right corner of the map -- translucent because that is what the real
    legend is, and a brightness test would report the water beneath it as gone.

    *legend_pad_px* is matplotlib's ``borderaxespad``: on the real 16.5 ka corridor
    panel the legend clears the right spine by 95 px and the bottom one by 12.
    ``0`` puts it flush against them, which is the case that broke the detector.

    *colourbar* adds the SECOND framed box every real frame has, to the right of the
    map and TALLER than it, filled with a blue ramp.  It is the case that broke the
    detector the second time: counting dark pixels per column made its two vertical
    edges look more like spines than the map's own, and the map rectangle grew to
    swallow a colour bar whose blue then counted as water.
    """
    margin = 30
    extra = 260 if colourbar else 0
    width, height = map_px[0] + 2 * margin + extra, map_px[1] + 2 * margin
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0 = margin, margin
    x1, y1 = margin + map_px[0] - 1, margin + map_px[1] - 1
    array[y0:y1 + 1, x0:x1 + 1] = (7, 175, 88)                        # land, green
    iy0, ix1, iy1 = y0 + 1, x1 - 1, y1 - 1                            # inside the spines
    wx0 = ix1 - water_cols + 1
    array[iy0:iy1 + 1, wx0:ix1 + 1] = (158, 202, 225)                 # water, blue
    lw, lh = legend_px
    lx1, ly1 = ix1 - legend_pad_px, iy1 - legend_pad_px
    lx0, ly0 = lx1 - lw + 1, ly1 - lh + 1
    patch = array[ly0:ly1 + 1, lx0:lx1 + 1].astype(float) * legend_alpha
    array[ly0:ly1 + 1, lx0:lx1 + 1] = patch.round().astype(np.uint8)
    array[y0, x0:x1 + 1] = 0
    array[y1, x0:x1 + 1] = 0
    array[y0:y1 + 1, x0] = 0
    array[y0:y1 + 1, x1] = 0
    if colourbar:
        # Narrow, and TALLER than the map -- so its vertical edges hold more dark
        # pixels than the map's spines do.  Same trick the real colour bar plays.
        cb_x0, cb_x1 = x1 + 150, x1 + 230
        cb_y0, cb_y1 = 4, height - 5
        ramp = np.linspace(255, 8, cb_y1 - cb_y0 + 1).astype(np.uint8)
        array[cb_y0:cb_y1 + 1, cb_x0:cb_x1 + 1, 0] = ramp[:, None]
        array[cb_y0:cb_y1 + 1, cb_x0:cb_x1 + 1, 1] = ramp[:, None]
        array[cb_y0:cb_y1 + 1, cb_x0:cb_x1 + 1, 2] = 255
        array[cb_y0, cb_x0:cb_x1 + 1] = 0
        array[cb_y1, cb_x0:cb_x1 + 1] = 0
        array[cb_y0:cb_y1 + 1, cb_x0] = 0
        array[cb_y0:cb_y1 + 1, cb_x1] = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path)
    return Path(path)


def _legend_probe_expected(
    *, map_px=(400, 200), legend_px=(120, 60), water_cols=60, legend_pad_px=6
) -> dict:
    """What ``_legend_probe_png`` put there, counted from its own construction."""
    map_w, map_h = map_px
    lw, lh = legend_px
    inner_w, inner_h = map_w - 2, map_h - 2
    water_total = water_cols * inner_h
    # The water strip is flush with the right-hand spine's inside edge; the legend
    # is inset from it by the pad, so they overlap in all but `legend_pad_px` columns.
    cols = min(water_cols, lw) - legend_pad_px
    rows = min(inner_h, lh)
    return {
        "water_total": water_total,
        "water_under": cols * rows,
        "legend_pct_of_map": 100.0 * lw * lh / (map_w * map_h),
        "inner_px": (inner_w, inner_h),
    }


class TestTheFigureIsTheOnlyLeverOnTheLegend:
    """``fig_size_in`` is measured, not asserted: the panel's own PNG is read back."""

    def test_the_fallback_canvas_follows_the_figure_it_was_given(self):
        assert anim.fallback_canvas_for(14.0, 150) == anim.FALLBACK_CANVAS_PX
        assert anim.fallback_canvas_for(anim.CLOSEUP_FIGURE_SIZE_IN, 150) == (3300, 3300)
        assert anim.fallback_canvas_for(22.0, 300) == (6600, 6600)

    def test_the_closeup_dpi_is_the_renderers_own_and_says_so_by_being_equal(self):
        """dpi is declared and NOT raised.  A constant that merely repeated the
        renderer's number would drift; this asserts they are the same number, so
        'the close-up did not touch dpi' is checked rather than remembered."""
        assert anim.CLOSEUP_FIGURE_DPI == RenderConfig().dpi
        assert anim.CLOSEUP_FIGURE_SIZE_IN != RenderConfig().fig_size_in, (
            "the figure IS raised -- that is the whole lever"
        )

    def test_a_raised_figure_does_not_keep_the_default_fallback(
        self, tmp_path, untagged_topo_dir
    ):
        """A run that raises the figure and declares the OLD box declares a bound
        that is not one.  Nothing renders here, so the fallback is what ships."""
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            duration_ms=250,
            fig_size_in=anim.CLOSEUP_FIGURE_SIZE_IN,
        )
        assert manifest["n_rendered"] == 0
        assert manifest["canvas"]["size_px"] == [3300, 3300]
        assert manifest["render_config"]["fig_size_in"] == anim.CLOSEUP_FIGURE_SIZE_IN
        assert manifest["render_config"]["untrimmed_figure_px"] == [3300, 3300]
        assert manifest["render_config"]["dpi"] == RenderConfig().dpi, (
            "dpi buys pixels and no relief on the legend; it is not raised silently"
        )

    def test_an_explicit_fallback_still_wins(self, tmp_path, untagged_topo_dir):
        manifest = anim.build_animation(
            out_root=tmp_path / "out",
            topo_dir=untagged_topo_dir,
            fig_size_in=22.0,
            fallback_canvas_px=(120, 90),
        )
        assert manifest["canvas"]["size_px"] == [120, 90]

    def test_the_measurement_finds_the_map_and_the_legend_it_was_given(self, tmp_path):
        png = _legend_probe_png(tmp_path / "probe.png")
        got = anim.measure_legend_occlusion(png, (36.4, 44.6, 46.0, 47.6))
        assert got["status"] == "MEASURED"
        assert got["map_px"] == [400, 200]
        assert got["map_rect_px"] == [30, 30, 429, 229]
        assert got["legend"]["found"] is True
        assert got["legend"]["size_px"] == [120, 60]
        assert got["legend"]["pct_of_map"] == pytest.approx(120 * 60 / 800.0, abs=1e-6)
        assert got["legend"]["fill_ratio"] > 0.9, "a legend is a filled box"
        assert got["px_per_deg_lon"] == pytest.approx(400 / 9.6, abs=1e-3)
        assert got["px_per_deg_lat"] == pytest.approx(200 / 3.0, abs=1e-3)

    def test_the_water_under_a_TRANSLUCENT_legend_is_counted_not_lost(self, tmp_path):
        """The defect a brightness test would have: the legend darkens, not deletes."""
        png = _legend_probe_png(tmp_path / "probe.png")
        got = anim.measure_legend_occlusion(png, (36.4, 44.6, 46.0, 47.6))
        want = _legend_probe_expected()
        assert got["water_px_in_map"] == want["water_total"]
        assert got["legend"]["water_px_under"] == want["water_under"]
        assert got["legend"]["pct_of_water_under"] == pytest.approx(
            100.0 * want["water_under"] / want["water_total"], abs=1e-3
        )

    def test_a_legend_flush_against_the_spines_is_not_the_whole_map(self, tmp_path):
        """The regression this detector shipped with for one run of the tests: the
        spine box is itself a dark connected component, so a legend touching it made
        the largest dark component the SPINES, whose bounding box is everything."""
        png = _legend_probe_png(tmp_path / "flush.png", legend_pad_px=0)
        got = anim.measure_legend_occlusion(png)
        assert got["legend"]["found"] is True
        assert got["legend"]["size_px"] != got["map_px"], "it found the spine box"
        # The two-pixel ring is excluded from the search, so a flush legend measures
        # exactly that much smaller on each side it touches -- under-, never over-.
        assert got["legend"]["size_px"] == [
            120 - anim.LEGEND_SPINE_MARGIN_PX + 1, 60 - anim.LEGEND_SPINE_MARGIN_PX + 1
        ]
        assert got["spine_margin_px"] == anim.LEGEND_SPINE_MARGIN_PX

    def test_a_bigger_figure_shrinks_the_legend_against_the_map(self, tmp_path):
        """The whole claim of the lever, on two pictures whose only difference is
        the map size -- the legend is fixed in points and does not move with it."""
        small = anim.measure_legend_occlusion(
            _legend_probe_png(tmp_path / "small.png", map_px=(400, 200)),
            (36.4, 44.6, 46.0, 47.6),
        )
        large = anim.measure_legend_occlusion(
            _legend_probe_png(tmp_path / "large.png", map_px=(628, 314),
                              water_cols=94),
            (36.4, 44.6, 46.0, 47.6),
        )
        assert large["legend"]["size_px"] == small["legend"]["size_px"]
        assert large["legend"]["pct_of_map"] < small["legend"]["pct_of_map"]
        assert large["px_per_deg_lon"] > small["px_per_deg_lon"]
        # 1/fig_size_in in area terms: (400*200)/(628*314) of the old fraction.
        assert large["legend"]["pct_of_map"] == pytest.approx(
            small["legend"]["pct_of_map"] * (400 * 200) / (628 * 314), rel=1e-3
        )

    def test_a_colour_bar_beside_the_map_is_not_part_of_the_map(self, tmp_path):
        """THE DEFECT THIS FUNCTION SHIPPED WITH FOR ONE RENDER, and the reason the
        rectangle is found by run LENGTH and not by dark-pixel COUNT.

        Measured on the real 16.5 ka corridor panel while that render was in flight:
        the colour bar's two vertical edges carried 1,540 dark pixels each against
        the map's own spines' 920, so a per-column count put the map's right edge at
        the colour bar's, reported a 2,940 px map as 3,078 px, and swept the colour
        bar's blue ramp into the water total.  Every figure downstream moved in the
        flattering direction, which is how it survived being looked at.
        """
        plain = anim.measure_legend_occlusion(
            _legend_probe_png(tmp_path / "plain.png"), (36.4, 44.6, 46.0, 47.6)
        )
        withbar = anim.measure_legend_occlusion(
            _legend_probe_png(tmp_path / "bar.png", colourbar=True),
            (36.4, 44.6, 46.0, 47.6),
        )
        assert withbar["map_px"] == plain["map_px"] == [400, 200]
        assert withbar["map_rect_px"] == plain["map_rect_px"]
        assert withbar["water_px_in_map"] == plain["water_px_in_map"], (
            "the colour bar's ramp was counted as water in the map"
        )
        assert withbar["px_per_deg_lon"] == plain["px_per_deg_lon"]
        assert withbar["legend"]["pct_of_map"] == plain["legend"]["pct_of_map"]
        assert withbar["spine_run_px"] == 400 and withbar["spine_rows_found"] == 2

    def test_a_wider_map_beside_the_same_colour_bar_is_still_read_exactly(
        self, tmp_path
    ):
        """The rectangle is the DRAWN map's, whatever else is on the page."""
        for map_px in ((400, 200), (640, 200), (400, 320)):
            got = anim.measure_legend_occlusion(
                _legend_probe_png(
                    tmp_path / f"p{map_px[0]}x{map_px[1]}.png",
                    map_px=map_px, colourbar=True,
                )
            )
            assert got["map_px"] == list(map_px)
            assert got["spine_run_px"] == map_px[0]

    def test_a_picture_with_no_spines_says_so_rather_than_inventing_a_map(
        self, tmp_path
    ):
        blank = tmp_path / "blank.png"
        Image.fromarray(
            np.full((40, 60, 3), 255, dtype=np.uint8), "RGB"
        ).save(blank)
        got = anim.measure_legend_occlusion(blank)
        assert got["status"] == "NO_MAP_RECTANGLE_FOUND"
        assert "no dark pixel" in got["why"]

    def test_the_extent_is_never_guessed_from_the_picture(self, tmp_path):
        got = anim.measure_legend_occlusion(_legend_probe_png(tmp_path / "p.png"))
        assert "extent_deg" not in got
        assert "px_per_deg_lon" not in got
        assert "window_deg" not in got["legend"]
        assert got["legend"]["size_px"] == [120, 60], "the pixels are still measured"


class TestTheCloseUpPageCarriesTheCorridorAlone:

    def test_only_the_corridor_panel_reaches_the_page_and_untouched(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(300, 120), seed=11)
        inset = _noisy_png(tmp_path / "inset.png", size_px=(260, 90), seed=12)
        record = anim.compose_full_and_inset(
            full, inset, tmp_path / "composed.png", caption="c", font_px=14,
            drop_full=True,
        )
        assert [p["role"] for p in record["panels"]] == ["inset"]
        assert record["inset_present"] is True
        panel = record["panels"][0]
        assert panel["scaled"] is False and panel["cropped"] is False
        composed = np.asarray(Image.open(record["composed"]).convert("RGB"))
        width, height = panel["size_px"]
        x, y = panel["at_px"]
        assert np.array_equal(
            composed[y:y + height, x:x + width],
            np.asarray(Image.open(inset).convert("RGB")),
        )
        assert record["size_px"][1] < 120 + 90, "the whole-scene panel is not on it"

    def test_the_omitted_panel_is_named_hashed_and_explained(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(300, 120), seed=13)
        inset = _noisy_png(tmp_path / "inset.png", size_px=(260, 90), seed=14)
        record = anim.compose_full_and_inset(
            full, inset, tmp_path / "composed.png", caption="c", font_px=14,
            drop_full=True,
        )
        omitted = record["full_panel_omitted"]
        assert omitted["source"] == str(full.resolve())
        assert omitted["source_sha256"] == anim.sha256_of(full)
        assert "NOT SHRUNK INTO A LOCATOR" in omitted["why"]
        assert record["rule"] == anim.CLOSEUP_COMPOSE_RULE
        assert "frames/full" in record["rule"]

    def test_the_two_panel_composition_is_untouched(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(300, 120), seed=15)
        inset = _noisy_png(tmp_path / "inset.png", size_px=(260, 90), seed=16)
        record = anim.compose_full_and_inset(
            full, inset, tmp_path / "composed.png", caption="c", font_px=14,
        )
        assert [p["role"] for p in record["panels"]] == ["full", "inset"]
        assert record["full_panel_omitted"] is None
        assert record["rule"] == anim.COMPOSE_RULE

    def test_dropping_the_full_panel_with_no_inset_is_refused(self, tmp_path):
        full = _noisy_png(tmp_path / "full.png", size_px=(120, 60), seed=17)
        with pytest.raises(ValueError) as excinfo:
            anim.compose_full_and_inset(
                full, None, tmp_path / "composed.png", caption="c", drop_full=True,
            )
        assert "NOTHING on the page" in str(excinfo.value)
        assert not (tmp_path / "composed.png").exists()

    def test_the_caption_says_the_page_is_the_corridor_alone(self):
        caption = anim.compose_caption(
            16.5, None, {}, anim.InsetPlan(
                epoch_ka=16.5, extent=anim.MANYCH_CORRIDOR_TIGHT
            ),
            closeup=True,
        )
        assert "the Manych corridor, and nothing else" in caption
        assert "full scene above" not in caption
        assert anim.MANYCH_CORRIDOR_TIGHT.name in caption

    def test_the_caption_quotes_the_measurement_it_was_handed(self, tmp_path):
        """PROSE SHIPPED IN A SIDECAR IS DATA: every figure in the sentence is
        recomputed here from the object the sentence was built from."""
        occlusion = anim.measure_legend_occlusion(
            _legend_probe_png(tmp_path / "p.png"), (36.4, 44.6, 46.0, 47.6)
        )
        caption = anim.compose_caption(
            16.5, None, {}, None, closeup=True, legend_occlusion=occlusion,
        )
        legend = occlusion["legend"]
        assert f"{legend['size_px'][0]:,} x {legend['size_px'][1]:,} px" in caption
        assert f"{occlusion['map_px'][0]:,} x {occlusion['map_px'][1]:,} px map" in caption
        assert f"{legend['pct_of_map']:.2f} % of it" in caption
        assert f"{legend['pct_of_water_under']:.2f} %" in caption
        assert (
            f"({legend['water_px_under']:,} of {occlusion['water_px_in_map']:,})"
            in caption
        )

    def test_a_caption_with_no_measurement_says_nothing_about_the_legend(self):
        caption = anim.compose_caption(16.5, None, {}, None, closeup=True)
        assert "THE LEGEND SITS INSIDE THE MAP" not in caption

    def test_the_two_panel_caption_did_not_change_wording(self):
        caption = anim.compose_caption(16.5, None, {}, None)
        assert caption.startswith("16.5 ka — full scene above, Manych inset below.")


class TestTheCloseUpBuildRefusesRatherThanShowingWhatItWasToldNotTo:

    def test_an_unknown_panels_value_is_refused(self, tmp_path, untagged_topo_dir):
        with pytest.raises(ValueError) as excinfo:
            anim.build_animation(
                out_root=tmp_path / "out", topo_dir=untagged_topo_dir,
                panels="corridor",
            )
        assert "panels='corridor'" in str(excinfo.value)
        assert "does not map an unknown value onto a default" in str(excinfo.value)

    def test_panels_inset_without_an_inset_is_refused(
        self, tmp_path, untagged_topo_dir
    ):
        with pytest.raises(ValueError) as excinfo:
            anim.build_animation(
                out_root=tmp_path / "out", topo_dir=untagged_topo_dir, panels="inset",
            )
        message = str(excinfo.value)
        assert "there is no inset to put on the page" in message
        assert anim.MANYCH_CORRIDOR_TIGHT.name in message
        assert not (tmp_path / "out" / "khvalynian_spill.gif").exists()

    def test_the_manifest_says_which_panels_it_composed(
        self, tmp_path, untagged_topo_dir
    ):
        manifest = anim.build_animation(
            out_root=tmp_path / "out", topo_dir=untagged_topo_dir,
            fallback_canvas_px=(120, 90),
        )
        assert manifest["panels_composed"]["panels"] == "both"
        assert manifest["panels_composed"]["rule"] == anim.COMPOSE_RULE
        assert manifest["render_config"]["figure_is_the_only_lever_on_the_legend"] == (
            anim.FIGURE_LEVER_RULE
        )


@_shipped
class TestTheMapRectangleIsReadOffRealFramesCorrectly:
    """The check that caught the colour-bar defect, on artifacts rather than fixtures.

    ``render_paleocostline_epoch`` draws with ``aspect='equal'``, so the axes' spine
    box has the extent's own width-over-height in degrees.  A detector that includes
    the colour bar does not: it reported 3078 x 920 for a 9.6 x 3.0 deg extent, an
    aspect of 3.35 against 3.20, and swept the colour bar's blue ramp into the water
    total.  Two shipped panels, two different extents, one invariant.
    """

    PANELS = (
        (
            "spill_0.5ka_9_epochs_RAMP_corridor_MANYCH_inset/frames/inset/"
            "paleo_render_16.5ka.png",
            (35.5, 43.2, 48.6, 47.8),
        ),
        (
            "spill_0.5ka_9_epochs_RAMP_corridor_MANYCH_inset/frames/full/"
            "paleo_render_16.5ka.png",
            (25.5, 36.0, 58.0, 54.0),
        ),
    )

    def test_the_rectangle_has_the_extents_own_aspect(self):
        for relative, extent in self.PANELS:
            png = SHIPPED_ROOT / relative
            if not png.exists():
                pytest.skip(f"{png} is not on this machine")
            got = anim.measure_legend_occlusion(png, extent)
            west, south, east, north = extent
            want = (east - west) / (north - south)
            width, height = got["map_px"]
            assert width / height == pytest.approx(want, rel=0.01), (
                f"{relative}: {width}x{height} is {width / height:.4f}, and the "
                f"extent is {want:.4f} -- the rectangle is not the map's"
            )
            assert got["px_per_deg_lon"] == pytest.approx(
                got["px_per_deg_lat"], rel=0.01
            ), "aspect='equal' means one scale, and this is two"
            # A box has two horizontal spines, and each is antialiased across one to
            # three rows in a saved figure -- so the count is small.  A LARGE one
            # would mean the run length matched something that is not a spine.
            assert 2 <= got["spine_rows_found"] <= 6, got["spine_rows_found"]


@_shipped
class TestTheManifestCarriesWhatItTellsAReaderToLookFor:
    """A pointer to a field that is not there is a defect in a file whose only job
    is that a frame can be audited from it.

    The first close-up build shipped exactly that: ``render_config.figure_is_the_
    only_lever_on_the_legend`` said the measurement was in
    ``frames[].inset.legend_occlusion``, the manifest's inset projection did not
    carry that key, and the numbers were reachable only through
    ``frames[].compose.caption`` -- which nothing told a reader to read.
    """

    BUILD = "spill_0.5ka_9_epochs_RAMP_CLOSEUP_corridor_only"

    def _manifest(self):
        path = SHIPPED_ROOT / self.BUILD / "sequence_manifest.json"
        if not path.exists():
            pytest.skip(f"{path} is not on this machine")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_field_the_render_config_rule_names_is_present(self):
        manifest = self._manifest()
        rule = manifest["render_config"]["figure_is_the_only_lever_on_the_legend"]
        assert "frames[].inset.legend_occlusion" in rule
        measured = 0
        for frame in manifest["frames"]:
            inset = frame.get("inset") or {}
            if not inset.get("png"):
                continue          # a refused panel has no picture to measure
            assert "legend_occlusion" in inset, (
                f"{frame['epoch_ka']} ka: the rule points at a field this frame "
                "does not carry"
            )
            occlusion = inset["legend_occlusion"]
            assert occlusion is not None, f"{frame['epoch_ka']} ka: recorded as null"
            assert occlusion["status"] == "MEASURED"
            assert occlusion["legend"]["found"] is True
            measured += 1
        assert measured == len(manifest["frames"]), "every frame drew a panel"

    def test_the_caption_and_the_record_quote_the_same_measurement(self):
        """Two places, one object -- checked by recomputing the sentence's own
        figures out of the record beside it."""
        manifest = self._manifest()
        for frame in manifest["frames"]:
            occlusion = frame["inset"]["legend_occlusion"]
            caption = frame["compose"]["caption"]
            legend = occlusion["legend"]
            assert f"{legend['pct_of_map']:.2f} % of it" in caption
            assert f"{legend['pct_of_water_under']:.2f} %" in caption
            assert (
                f"{legend['water_px_under']:,} of {occlusion['water_px_in_map']:,}"
                in caption
            )
