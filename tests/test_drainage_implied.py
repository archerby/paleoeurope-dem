"""Guards for ``drainage_implied`` as a pipeline step.

The heavy end of this is proved by ARTIFACT, not by a fixture: the step
reproduces the rivers probe's shipped product PIXEL-IDENTICALLY (sha256
78465bb5…) when pointed at the same inputs. That comparison lives in the WORK_LOG
and in the manifest, because it needs the 51 N tree, WhiteboxTools and about two
minutes. What is tested here is everything that can fail without any of that.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rivers_probe"))

import drainage_implied as D  # noqa: E402


# ---------------------------------------------------------------------------
# the sweep ladder
# ---------------------------------------------------------------------------

#: The eighteen points the rivers probe actually swept, read out of its own saved
#: sweep_16.5.json. Pinned here because the chosen threshold is ONE OF THEM: a
#: different ladder is a different answer, not a different rendering of the same
#: one.
PROBE_LADDER = [100.0, 177.8279, 316.2278, 562.3413, 1000.0, 1778.2794,
                3162.2777, 5623.4133, 10000.0, 17782.7941, 31622.7766,
                56234.1325, 100000.0, 177827.941, 316227.766, 562341.3252,
                1000000.0, 1778279.41]


def test_the_ladder_is_the_probes_ladder():
    """Sweeping powers of two instead put the threshold at 3200 against the
    probe's 3162.2777, and the two products then differed in 1437 cells -- all
    of them probe-only, the signature of a threshold higher by 1.2 %."""
    assert len(D.SWEEP_CELLS) == len(PROBE_LADDER)
    for got, want in zip(D.SWEEP_CELLS, PROBE_LADDER):
        assert abs(got - want) < 1e-3, (got, want)


def test_the_ladder_is_derived_not_typed():
    """Eighteen floats typed out can drift from the sentence describing them.

    Checked with ast, over CODE constants only. My first version grepped the
    file text for "3162.2777" and failed on the comment that explains why the
    ladder matters -- a check reading the prose it had just written, which is
    the fourth time that shape has turned up in this phase.
    """
    import ast
    tree = ast.parse((REPO / "scripts" / "drainage_implied.py")
                     .read_text(encoding="utf-8"))
    typed = [n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, float)
             and any(abs(n.value - t) < 1e-3 for t in PROBE_LADDER[2:])]
    assert not typed, "sweep points appear as literals in code: %r" % typed
    assert isinstance(D.SWEEP_CELLS, tuple) and len(D.SWEEP_CELLS) == D.SWEEP_N


# ---------------------------------------------------------------------------
# the threshold rule and its refusals
# ---------------------------------------------------------------------------

def _rows(dens, man=None):
    """(threshold, cells, length, density, manych_length) rows for a given
    density curve."""
    man = [1.0] * len(dens) if man is None else man
    return [(float(t), 0, 0.0, float(d), float(m))
            for t, d, m in zip(D.SWEEP_CELLS, dens, man)]


def test_it_picks_the_largest_threshold_still_on_the_plateau():
    """A pure power law: every point is on the plateau, so the rule takes the
    largest one below the Manych bound."""
    ts = np.array(D.SWEEP_CELLS)
    dens = 1.0 * ts ** -0.5
    chosen, plateau, upper, _ = D.choose_threshold(_rows(dens))
    assert chosen == pytest.approx(D.SWEEP_CELLS[-1])
    assert plateau == pytest.approx(-0.5, abs=1e-6)
    assert upper == float("inf")


def test_the_manych_bound_is_read_from_the_corridor_and_refuses(monkeypatch):
    """Planted: the corridor dies at a threshold BELOW the one the plateau would
    otherwise allow. The rule must refuse rather than erase the feature the
    scene exists to show."""
    ts = np.array(D.SWEEP_CELLS)
    dens = 1.0 * ts ** -0.5
    man = [1.0] * len(ts)
    for i in range(3, len(ts)):        # corridor carries nothing from point 3 on
        man[i] = 0.0
    with pytest.raises(D.DrainageRefusal) as exc:
        D.choose_threshold(_rows(dens, man))
    assert "erases the feature" in str(exc.value)


def test_the_retired_no_plateau_condition_could_never_fire():
    """The refusal this replaced was decorative, and the proof is arithmetic.

    It asked whether ANY slope lies within PLATEAU_TOL of
    ``median(slope[:5])``. With five values the median IS one of them, so that
    distance is exactly zero at that point -- for every input that has ever been
    or could be passed. A guard that cannot fail is not a guard.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        slope = rng.normal(0, 50, size=len(D.SWEEP_CELLS))
        plateau = float(np.median(slope[:5]))
        assert (np.abs(slope - plateau) <= D.PLATEAU_TOL).any()


def test_it_refuses_when_the_fine_end_is_not_flat():
    """The replacement CAN fire: a density curve whose five finest points spread
    far more in log-log slope than a plateau would."""
    ts = np.array(D.SWEEP_CELLS, float)
    dens = np.exp(-(np.log10(ts) ** 3))          # slope grows without settling
    with pytest.raises(D.DrainageRefusal) as exc:
        D.choose_threshold(_rows(dens))
    assert "no plateau at the fine end" in str(exc.value)


def test_the_flatness_bound_has_headroom_over_the_real_data():
    """Measured across the 51 N reproduction and all five 53 N spill epochs the
    five finest slopes spread 0.0070-0.0089. The bound must clear that, and must
    not be so wide that the test above stops firing."""
    assert D.PLATEAU_SPREAD_MAX > 0.0089 * 2
    assert D.PLATEAU_SPREAD_MAX < 0.1


# ---------------------------------------------------------------------------
# a requested (common) threshold
# ---------------------------------------------------------------------------

def test_a_requested_threshold_overrides_the_per_epoch_choice():
    ts = np.array(D.SWEEP_CELLS)
    dens = 1.0 * ts ** -0.5
    chosen, _, _, _ = D.choose_threshold(_rows(dens), requested=3162.277660168379)
    assert chosen == pytest.approx(3162.277660168379)


def test_a_requested_threshold_is_still_refused_at_the_manych_bound():
    """Measured on the 53 N frame: the per-epoch rule picks 3162 cells at four
    spill epochs and 31623 at 14.5 ka, so a series built per-epoch shows density
    falling 0.040 -> 0.0126 -- a 69 % collapse that is an artefact of the
    threshold. A common threshold is the fix, but one that erases the corridor at
    some epoch is worse than none."""
    ts = np.array(D.SWEEP_CELLS)
    dens = 1.0 * ts ** -0.5
    man = [1.0] * len(ts)
    for i in range(6, len(ts)):
        man[i] = 0.0                              # corridor dies at point 6
    with pytest.raises(D.DrainageRefusal) as exc:
        D.choose_threshold(_rows(dens, man), requested=D.SWEEP_CELLS[10])
    assert "requested" in str(exc.value)
    assert "erases the feature" in str(exc.value)


# ---------------------------------------------------------------------------
# locations refuse instead of defaulting
# ---------------------------------------------------------------------------

def test_fields_root_refuses_because_two_frames_exist(monkeypatch):
    """The probe read khvalynian_basin_fields as a literal. Two trees exist and
    they differ in 645 802 cells; a default picks a FRAME for the reader."""
    monkeypatch.delenv("PALEO_BASIN_FIELDS_ROOT", raising=False)
    with pytest.raises(D.DrainageRefusal) as exc:
        D.fields_root()
    assert "645 802" in str(exc.value) or "645802" in str(exc.value)


def test_fields_root_refuses_a_path_that_is_not_there(tmp_path, monkeypatch):
    monkeypatch.setenv("PALEO_BASIN_FIELDS_ROOT", str(tmp_path / "nope"))
    with pytest.raises(D.DrainageRefusal):
        D.fields_root()


# ---------------------------------------------------------------------------
# the WBT artifact rule
# ---------------------------------------------------------------------------

def test_a_wbt_call_that_writes_nothing_is_a_failure_even_at_return_code_zero(
        tmp_path):
    """WhiteboxTools panics inside Rust and its Python wrapper still returns 0 --
    a success with no output file. This project judges by artifacts; here that
    has to be applied to a third-party tool."""
    def silent_success(*a, callback=None, **k):
        return 0
    with pytest.raises(D.DrainageRefusal) as exc:
        D._run_wbt("FillDepressions", silent_success,
                   artifact=tmp_path / "never_written.tif")
    assert "produced no artifact" in str(exc.value)


def test_a_wbt_call_that_writes_its_artifact_passes(tmp_path):
    art = tmp_path / "written.tif"

    def writes(*a, callback=None, **k):
        art.write_bytes(b"x")
        if callback:
            callback("done")
    D._run_wbt("D8Pointer", writes, artifact=art)


def test_a_panic_in_the_output_is_a_failure_even_with_the_artifact(tmp_path):
    art = tmp_path / "written.tif"

    def panics(*a, callback=None, **k):
        art.write_bytes(b"x")
        callback("thread 'main' panicked at ...")
    with pytest.raises(D.DrainageRefusal):
        D._run_wbt("FillDepressions", panics, artifact=art)


# ---------------------------------------------------------------------------
# the cross-epoch question is MEASURED, not decided
# ---------------------------------------------------------------------------

def _record(epoch, plateau, slopes, upper, chosen):
    return {"epoch_ka": epoch, "threshold_cells": chosen,
            "threshold_plateau_exponent": plateau,
            "threshold_upper_bound_cells": upper,
            "sweep": [{"threshold_cells": t} for t in D.SWEEP_CELLS],
            "sweep_log_log_slopes": list(slopes)}


def test_a_common_threshold_is_reported_when_one_exists():
    flat = [-0.5] * len(D.SWEEP_CELLS)
    recs = [_record("16.5", -0.5, flat, float("inf"), D.SWEEP_CELLS[-1]),
            _record("16", -0.5, flat, float("inf"), D.SWEEP_CELLS[-1])]
    out = D.common_threshold(recs)
    assert out["exists"] is True
    assert out["largest_cells"] == pytest.approx(D.SWEEP_CELLS[-1])


def test_no_common_threshold_is_reported_as_none_not_as_a_pick():
    """Two epochs whose plateaus do not overlap. The step must say the series is
    not differenceable rather than choosing one epoch's threshold for both."""
    n = len(D.SWEEP_CELLS)
    early = [-0.5] * 4 + [9.0] * (n - 4)      # plateau only at the fine end
    late = [9.0] * (n - 4) + [-0.5] * 4       # plateau only at the coarse end
    recs = [_record("16.5", -0.5, early, float("inf"), D.SWEEP_CELLS[0]),
            _record("16", -0.5, late, float("inf"), D.SWEEP_CELLS[-1])]
    out = D.common_threshold(recs)
    assert out["exists"] is False
    assert out["largest_cells"] is None
    assert out["per_epoch_chosen_cells"] == {"16.5": D.SWEEP_CELLS[0],
                                             "16": D.SWEEP_CELLS[-1]}


def test_the_product_says_its_threshold_is_per_epoch():
    """A density computed at a different threshold each epoch is not a time
    series. The raster has to say so itself, because a reader differencing two
    of them will not have the manifest open."""
    src = (REPO / "scripts" / "drainage_implied.py").read_text(encoding="utf-8")
    assert "threshold_is_per_epoch" in src
    assert "must not be differenced across epochs" in src


def test_the_product_carries_what_it_is_not():
    src = (REPO / "scripts" / "drainage_implied.py").read_text(encoding="utf-8")
    assert "product_is_not" in src
    assert "No channel here was observed" in src
