"""--plan-only must not delete the record of a rendered sequence.

Written after doing exactly that to a shipped build on 2026-08-23. The flag is
documented "resolve levels, seeds and refusals; render nothing", and it renders
nothing -- but it writes the manifest anyway, so aiming it at an existing build
replaced a 1.24 MB record of nine rendered frames with a 232 KB one whose every
frame reads png=False. The PNGs survived; the file saying what they are did not.

The convention this protects already existed and nothing enforced it: the tree
carries ``renders/_ABORTED_2026-08-04_.../_planonly_scratch_rfinal/``, a scratch
directory someone made for this very reason.

Every case is planted, including the three where the guard must stay SILENT --
a guard that refuses too much gets switched off.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import animate_khvalynian_spill as anim  # noqa: E402


def _manifest(tmp_path, payload):
    p = tmp_path / "sequence_manifest.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.mark.parametrize("frame", [
    {"epoch_ka": 16.5, "png": "full/paleo_render_16.5ka.png"},
    {"epoch_ka": 16.5, "inset": {"png": "inset/paleo_render_16.5ka.png"}},
    {"epoch_ka": 16.5, "composed_png": "composed/16.5.png"},
])
def test_it_refuses_when_the_manifest_describes_a_rendered_frame(tmp_path, frame):
    """A frame counts as rendered through any of the three keys a picture can
    land in. Missing one of them would let the clobber through for a build
    composed differently -- and the close-up build, the one this happened to, is
    composed from the inset panel alone."""
    p = _manifest(tmp_path, {"frames": [frame]})
    with pytest.raises(anim.PlanOnlyWouldOverwriteARenderedManifest) as exc:
        anim._refuse_to_overwrite_a_rendered_manifest(p)
    msg = str(exc.value)
    assert "1 RENDERED frames" in msg
    assert "_planonly_scratch_" in msg, "the refusal must name the way out"


def test_the_refusal_lists_the_epochs_it_is_protecting(tmp_path):
    p = _manifest(tmp_path, {"frames": [
        {"epoch_ka": e, "inset": {"png": "x.png"}}
        for e in (17.0, 16.5, 16.0)]})
    with pytest.raises(anim.PlanOnlyWouldOverwriteARenderedManifest) as exc:
        anim._refuse_to_overwrite_a_rendered_manifest(p)
    assert "17 16.5 16" in str(exc.value)


def test_it_is_silent_on_a_plan_only_manifest(tmp_path):
    """Re-planning into a scratch directory that already holds a plan is the
    normal case and must not be blocked."""
    p = _manifest(tmp_path, {"frames": [{"epoch_ka": 17.0, "inset": {}},
                                        {"epoch_ka": 16.5}]})
    anim._refuse_to_overwrite_a_rendered_manifest(p)


def test_it_is_silent_when_there_is_no_manifest(tmp_path):
    anim._refuse_to_overwrite_a_rendered_manifest(tmp_path / "absent.json")


def test_it_is_silent_when_the_manifest_cannot_be_parsed(tmp_path):
    """Nothing legible to protect. Refusing here would let an unrelated broken
    file block a legitimate plan."""
    p = tmp_path / "sequence_manifest.json"
    p.write_text("{ not json", encoding="utf-8")
    anim._refuse_to_overwrite_a_rendered_manifest(p)


def test_the_guard_is_actually_wired_into_the_write():
    """The helper existing is not the same as it being called.

    Checked line by line against the source, because calling build_animation for
    real needs a full render. Note the shape of the check: my first version did
    ``src.index("_refuse_to_overwrite_a_rendered_manifest(manifest_path)")``,
    which matches the DEF line before it ever reaches the call -- a check reading
    what it produced instead of what it guards, which is the third time that
    shape has appeared in this phase.
    """
    lines = Path(anim.__file__).read_text(encoding="utf-8").splitlines()
    call_i = write_i = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "_refuse_to_overwrite_a_rendered_manifest(manifest_path)":
            call_i = i
        if s.startswith("manifest_path.write_text(json.dumps(manifest"):
            write_i = i
    assert call_i is not None, "the guard is defined but never called"
    assert write_i is not None, "the manifest write moved; re-point this test"
    assert call_i < write_i, "the guard must run BEFORE the write"
    assert lines[call_i - 1].strip() == "if plan_only:", (
        "the guard must be conditional on plan_only, not unconditional")
