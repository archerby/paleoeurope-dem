# INCIDENT 2026-08-04 06:07 — `tests/test_viz.py` was destroyed by a full disk

## What happened

An agent rewrote `tests/test_viz.py` with a whole-file `open(path, "w")` while **D: had 212 KB
free**. `open(..., "w")` truncated the file, then `write()` raised `OSError` errno 28 (no space left
on device). The old content was already gone and the new content was never written.

**5,310 lines were lost.** Only 1,563 of them exist in git at `2a38b33`; the other ~3,747 were
uncommitted work produced across phases 2–6 of the 2026-08-03/04 Khvalynian session.

Root cause is not the agent's editor choice alone — D: had been at 100 % for the whole session and
was allowed to keep falling (713 MB → 400 MB → 212 KB) while agents wrote to it. The lesson that
belongs in the operating rules: **never let a phase run with a working volume under ~2 GB, and never
whole-file-write a large source file onto a volume that could be full.**

## What was NOT lost

Everything else survived and is intact in the working tree:

| File | State |
|---|---|
| `paleoeurope/viz/paleocostline_render.py` | intact, 6 405 lines, carries all phase 1–6 work |
| `scripts/bake_black_sea.py`, `scripts/bake_khvalynian.py` | intact |
| `tests/test_bake_black_sea.py`, `tests/test_bake_khvalynian.py` | intact |
| `scripts/animate_khvalynian_spill.py` | intact, 6 405 lines (the equal line count is a coincidence — the files differ) |
| `tests/test_animate_khvalynian_spill.py` | intact, 2 798 lines |

The rendered artifacts on `Y:\paleo\renders\` and the 39 re-baked Pontic frames on D: are unaffected.

## Recovery material in this directory

| File | What it is | Tests |
|---|---|---|
| `git_2a38b33.py.txt` | the committed baseline | 1 563 lines / 79 tests |
| `snapshot_2026-08-03T2001.py.txt` | a real source snapshot taken mid-session | 3 741 lines / 149 tests |
| `snapshot_2026-08-04T1105.py.txt` | a near-identical later working copy | 3 737 lines / 149 tests |
| `test_viz.cpython-310.pyc` | **bytecode of the file as it stood just before destruction** | all 383 code objects |
| `test_viz_inventory.json` | an inventory extracted from that bytecode | — |

The `.pyc` is the only artifact that contains the *last* state. Names, docstrings, constants and
code objects are recoverable from it; the source text is not, without a Python 3.10 decompiler.

## Why the file was NOT auto-reconstructed

A test file that imports and passes while silently omitting guards another agent wrote is exactly
the substitution this whole phase existed to remove — a green suite asserting less than it claims.
`tests/test_viz.py` is therefore a stub whose single test FAILS with this account.

## What reconstruction requires

1. **Port the 2026-08-03T2001 snapshot to the current API.** It does not import today: it references
   `SILL_ROUTE_NOTE`, which C-EDGE replaced with `sill_route_stats` / `sill_route_note`. Expect other
   drift from phases 5–6 (`_seeded_water_components`, the three-state connectivity, the per-frame
   prose templates).
2. **Re-add the ~60+ tests that exist only in the bytecode** — P-PALETTE's 47 (`TestEveryColourLimitWasDerivedPerFrame`,
   `TestSequenceLimitsCarriesItsOwnProvenance`, `TestOutOfRangeIsVisibleCountedAndRefusable`,
   `TestPinningIsOptInAndCheap`), plus U-CONNECT's and C-EDGE's. Every one of them is named, with
   what it asserts, in the corresponding agent reports for that session.
3. **Verify each restored test against the code it guards** rather than against the inventory — a
   test recovered by name only is a title, not a guard.

Until that is done the repository's viz coverage is the stub plus whatever
`tests/test_animate_khvalynian_spill.py` exercises indirectly. Do not read a green suite as
covering the renderer.
