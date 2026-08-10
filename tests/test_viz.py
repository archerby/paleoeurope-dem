"""THIS FILE WAS DESTROYED ON 2026-08-04 AND HAS NOT BEEN RESTORED.

It is a stub that FAILS ON PURPOSE.  Do not delete it and do not skip it: an empty
or absent ``tests/test_viz.py`` makes the suite report green with zero renderer
tests collected, which is a worse lie than a red one.

WHAT HAPPENED
-------------
The C-EDGE agent was rewriting this file with a whole-file write while ``D:`` had
212 KB free.  ``open(path, "w")`` truncated the file, the ``write()`` that followed
raised ``OSError: [Errno 28] No space left on device``, and the previous contents —
5 310 lines, of which only 1 563 are committed at ``2a38b33`` — were gone.  The
uncommitted 3 747 lines were the work of the phase's earlier agents (the sill
guard, the connectivity round, the P-PALETTE sequence-limits block) plus the C-EDGE
round in progress.  Nothing in ``git`` holds them: they were never staged, the two
dangling stashes date from 2026-07-01 and carry a 527-line version, and no other
copy exists in either repository.

WHAT IS NOT AFFECTED
--------------------
``paleoeurope/viz/paleocostline_render.py`` is intact and carries the completed
C-EDGE work (the UNOBSERVABLE connectivity state, and the per-frame rebuild of every
shipped sentence).  It imports cleanly.  The other four modified files in the tree
were never touched by this agent.

WHAT EXISTS TO RESTORE FROM, and what each source can and cannot give
---------------------------------------------------------------------
All of it is outside the repository, under the session scratchpad
``C:\\Users\\pavel\\AppData\\Local\\Temp\\claude\\D--pasha-repos-findus-infra\\
4249b1fb-5cee-4f40-911c-fd1e43fc5915\\scratchpad``:

``recover/test_viz.cpython-310.pyc``
    A plain (NOT pytest-rewritten) CPython 3.10 compile of this file as it stood at
    05:47, i.e. after most of the C-EDGE edits.  383 code objects, ``co_filename``
    pointing at this path, highest ``co_firstlineno`` 5310.  It carries **every
    docstring and every string and numeric constant verbatim**, plus every name and
    every first line number.  It does NOT carry statement text, so it cannot be
    turned back into source without a Python 3.10 decompiler, and none is installed.
    Installing one was not attempted: fetching and running a decompiler off the
    network is outside what this agent may do.
``recover/test_viz_inventory.json``
    That pyc, unpacked: one entry per code object, in line order, with its name,
    parent, first line number, docstring, string constants, numeric constants and
    referenced names.  This is the authoritative inventory of WHAT was in the file.
``test_viz_before.py``
    A verbatim snapshot of this file at 2026-08-03 20:01, 3 741 lines, left by an
    earlier agent of this phase.  It covers 231 of the 383 code objects.  The other
    83 named objects — the whole P-PALETTE block, the U-CONNECT block, and the
    C-EDGE additions — post-date it and are NOT in it.  It is an OLDER state, so
    several of the tests it does contain were subsequently rewritten; restoring it
    wholesale would put back superseded assertions.
``patch_tests_1.py``, ``patch_tests_2.py``, ``patch_tests_3.py``,
``patch_tests_new.py``, ``patch_tests_unobs.py``, ``patch_dataset.py``
    Every C-EDGE edit to this file, as replayable search-and-replace scripts.  They
    are exact, but they apply against the destroyed base, not against the snapshot.

WHY THIS STUB DOES NOT CONTAIN A RECONSTRUCTION
-----------------------------------------------
A reconstruction is possible for the 231 objects the snapshot holds and for the
C-EDGE additions.  It is NOT possible for the ~900 lines of P-PALETTE tests, which
this agent never read and which survive only as docstrings and constants.  Writing
a file that looks like the test suite, passes, and quietly omits or weakens guards
another agent wrote is exactly the substitution this phase exists to remove — a
plausible object standing in for a measured one, with nothing in the artifact to
tell a reader which is which.  So nothing is substituted.  The gap is stated, the
material to close it is named, and the suite stays red until somebody closes it.

Re-running the C-EDGE acceptance therefore did not happen: the Manych guard at
2000/4000/native, the byte-identity of the single-basin Pontic frame against
``3bcef60d...``, and the production-frame connectivity numbers are all unverified
since the last edit to the renderer.  They were green before the C-EDGE round began
and the renderer's own changes are described in its docstrings, but this file is
where they were checked, and it no longer checks anything.
"""

import pytest


def test_the_renderer_test_suite_was_destroyed_and_not_restored():
    """Fail loudly, with the whole account, until this file is rebuilt.

    Deleting this test to get a green suite would leave
    ``paleoeurope/viz/paleocostline_render.py`` — 6 369 lines, the module that
    decides what every shipped frame paints — with no tests at all and no sign that
    it ever had any.
    """
    pytest.fail(
        "tests/test_viz.py was truncated to 0 bytes on 2026-08-04 by a whole-file "
        "write against a full disk (D: had 212 KB free; the open() truncated and "
        "the write() raised ENOSPC). 5,310 lines were lost, of which 1,563 are "
        "recoverable from git at 2a38b33 and 3,747 were uncommitted work by this "
        "phase's earlier agents. paleoeurope/viz/paleocostline_render.py is "
        "unaffected and still carries the C-EDGE changes. Recovery material — a "
        "non-rewritten .pyc holding every docstring and constant, a JSON inventory "
        "of all 383 code objects, a verbatim 3,741-line snapshot from 2026-08-03 "
        "20:01, and replayable patch scripts for every C-EDGE edit — is listed in "
        "this module's docstring with the exact paths. NOTHING has been "
        "reconstructed in its place: a test file that passes while silently "
        "omitting guards another agent wrote is the substitution this phase exists "
        "to remove. Rebuild it, then delete this test."
    )
