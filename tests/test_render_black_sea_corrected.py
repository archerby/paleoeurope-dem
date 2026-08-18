"""Regression tests for the Pontic-curve reader in ``scripts/render_black_sea_corrected.py``.

WHY THIS FILE EXISTS
--------------------
On 2026-07-30 a ``#`` header block (datum annotation + marker-column
documentation) was prepended to
``Paleo-Europe_3D/docs/superpowers/research/2026-06-21-black-sea-level-curve.csv``.
``render_black_sea_corrected.py`` read that file with a bare
``pd.read_csv(BLACK_SEA_CSV)``.  pandas took the first ``#`` line as the header
row and then died on the first line with more than one field:

    pandas.errors.ParserError: Error tokenizing data.
    C error: Expected 1 fields in line 6, saw 2

The fix is ``comment="#"``.  The tests below pin it from three directions so a
future header addition — to either curve — cannot break a reader again:

* the shipped CSV is read **through the consumer's own loader**, not through a
  re-implementation of it in the test;
* the header block the loader has to survive is asserted to still be there, so
  the loader test cannot pass vacuously;
* every ``pd.read_csv`` call site in ``scripts/`` and ``paleoeurope/`` is
  AST-scanned for a ``comment=`` keyword, which is what makes this a guard
  against the NEXT reader rather than a fix for this one.

The Caspian curve (``2026-07-28-caspian-khvalynian-level-curve.csv``) carries a
``#`` header block too and fails identically under a bare read; it has no reader
in this repo today, and the AST scan is what keeps it that way.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pandas as pd
import pytest

import scripts.render_black_sea_corrected as rbsc

REPO_ROOT = Path(__file__).resolve().parents[1]

# The two curve files that carry a `#` header block.  Both live in the sibling
# viewer repo; both are optional from this repo's point of view, so every test
# that touches one skips rather than fails when it is absent.
#
# They used to be reached through ``rbsc.BLACK_SEA_CSV``, which is still a
# hardcoded ``D:\…`` literal inside the script (that constant is flagged STALE in
# ``render_black_sea_corrected.py`` and is deliberately not being re-pointed by
# this change).  Taking the path from the script meant BOTH parametrised tests
# skipped on every non-Windows box even where the CSVs are present -- so the
# location is resolved here through the same ``PALEO_DATA_ROOT`` indirection the
# rest of the repo uses, and the coupling to the script's own constant is kept
# alive by ``test_the_script_still_points_at_the_curve_this_module_checks``.
_DEFAULT_PALEO_DATA_ROOT = REPO_ROOT.parents[1] / "Paleo-Europe_3D"
_PALEO_DATA_ROOT = Path(
    os.environ.get("PALEO_DATA_ROOT") or _DEFAULT_PALEO_DATA_ROOT
).resolve()

#: Relative to the Paleo-Europe_3D tree; this is the tail ``rbsc.BLACK_SEA_CSV``
#: spells out absolutely.
_CURVE_DIR_PARTS = ("docs", "superpowers", "research")
PONTIC_CSV_NAME = "2026-06-21-black-sea-level-curve.csv"

_CURVE_DIR = _PALEO_DATA_ROOT.joinpath(*_CURVE_DIR_PARTS)
PONTIC_CSV = _CURVE_DIR / PONTIC_CSV_NAME
CASPIAN_CSV = (
    PONTIC_CSV.parent / "2026-07-28-caspian-khvalynian-level-curve.csv"
)


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(
            f"sibling-repo curve CSV not available here: {path} "
            "(set PALEO_DATA_ROOT to the Paleo-Europe_3D tree)"
        )
    return path


def test_the_script_still_points_at_the_curve_this_module_checks():
    """The tests below read a path they build themselves.  This is the one place
    that ties it back to ``rbsc.BLACK_SEA_CSV``.

    Without it, re-pointing the script at a different curve would leave every
    test in this file green while checking a file the script no longer opens.
    ``rbsc.BLACK_SEA_CSV`` is a Windows-spelled literal, so it is compared as
    text with both separators rather than as a ``Path``.
    """
    spelled = str(rbsc.BLACK_SEA_CSV).replace("\\", "/")
    assert spelled.endswith("/".join((*_CURVE_DIR_PARTS, PONTIC_CSV_NAME))), (
        f"render_black_sea_corrected.BLACK_SEA_CSV is {rbsc.BLACK_SEA_CSV!r}, "
        f"which is no longer …/{'/'.join(_CURVE_DIR_PARTS)}/{PONTIC_CSV_NAME} — "
        "this module is checking a file the script does not read"
    )
    assert str(rbsc.load_bs_curve.__defaults__[0]) == str(rbsc.BLACK_SEA_CSV)


# ---------------------------------------------------------------------------
# The precondition — without it every other test in this file is vacuous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "csv_path", [PONTIC_CSV, CASPIAN_CSV], ids=["pontic", "caspian"]
)
def test_the_shipped_curve_csvs_still_carry_a_comment_header_block(csv_path):
    """Both curves lead with `#` lines.  This is the thing readers must skip."""
    head = _require(csv_path).read_text(encoding="utf-8").splitlines()
    assert head[0].startswith("#"), (
        f"{csv_path.name} no longer starts with a `#` comment block — the "
        "loader tests below would then pass for the wrong reason"
    )
    assert sum(1 for ln in head if ln.startswith("#")) > 10


@pytest.mark.parametrize(
    "csv_path", [PONTIC_CSV, CASPIAN_CSV], ids=["pontic", "caspian"]
)
def test_a_bare_read_csv_still_fails_on_both_curves(csv_path):
    """The defect itself, pinned: this is why ``comment='#'`` is load-bearing.

    Not a test of pandas — a test that the header block is genuinely fatal to a
    naive reader, so that the ``comment="#"`` in the loader is understood as a
    requirement and not as decoration somebody may tidy away.
    """
    with pytest.raises(pd.errors.ParserError):
        pd.read_csv(_require(csv_path))


# ---------------------------------------------------------------------------
# The consumer, exercised on the SHIPPED file
# ---------------------------------------------------------------------------


def test_load_bs_curve_reads_the_shipped_pontic_csv():
    """REGRESSION: this raised ParserError from 2026-07-30 until the fix."""
    df = rbsc.load_bs_curve(_require(PONTIC_CSV))
    assert list(df.columns)[:3] == [
        "age_ka",
        "level_m_global_gradual",
        "level_m_global_catastrophic",
    ]
    assert len(df) == 57
    # The header block is gone, not merely tolerated: no row survived as text.
    assert df["age_ka"].dtype.kind == "f"
    assert df[rbsc.BS_COLUMN].dtype.kind == "f"


def test_the_consumer_can_read_every_epoch_it_renders():
    """`main()`'s whole per-epoch loop, minus the plotting."""
    df = rbsc.load_bs_curve(_require(PONTIC_CSV))
    for epoch in rbsc.EPOCHS_KA:
        level, state = rbsc.bs_level_for_epoch(df, float(epoch))
        assert -150.0 <= level <= 1.0, f"{epoch} ka level {level} out of range"
        assert state


def test_bs_level_for_epoch_returns_the_16_5_ka_hero_row():
    """A value check, so a silently-shifted parse cannot pass the shape checks."""
    df = rbsc.load_bs_curve(_require(PONTIC_CSV))
    level, state = rbsc.bs_level_for_epoch(df, 16.5)
    assert level == -30.0
    assert state == "lake_isolated"


def test_marker_columns_are_never_coerced_to_nan():
    """``NOT_APPLICABLE`` is spelled out precisely because pandas eats ``"NA"``.

    A NaN here would be indistinguishable from a missing measurement, which is
    the defect class the marker columns exist to remove.
    """
    df = rbsc.load_bs_curve(_require(PONTIC_CSV))
    for col in ("marker_lon", "marker_lat", "marker_source"):
        assert df[col].isna().sum() == 0, f"{col} has NaN — a marker went missing"


# ---------------------------------------------------------------------------
# The guard that covers the NEXT reader, not just this one
# ---------------------------------------------------------------------------


def _read_csv_calls(py: Path):
    """``(lineno, has_comment_kw)`` for every ``*.read_csv(...)`` call in *py*."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our files
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "read_csv":
            yield node.lineno, any(kw.arg == "comment" for kw in node.keywords)


def test_no_shipped_module_reads_a_csv_without_comment_handling():
    """Every ``read_csv`` in the shipped code must pass ``comment=``.

    Scoped to ``scripts/`` and ``paleoeurope/`` — the modules that ship.  Any
    CSV this project reads is one a human may annotate; the cheapest way to stop
    the 2026-07-30 breakage recurring is to require the keyword everywhere
    rather than to enumerate which files happen to read a curve today.
    """
    offenders = []
    for pkg in ("scripts", "paleoeurope"):
        for py in sorted((REPO_ROOT / pkg).rglob("*.py")):
            for lineno, has_comment in _read_csv_calls(py):
                if not has_comment:
                    offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    assert offenders == [], (
        "read_csv without comment='#': " + ", ".join(offenders)
    )
