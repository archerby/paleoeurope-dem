"""
tests/conftest.py

Shared pytest fixtures for the paleoeurope test suite.

Primary data source (committed to git, no download needed):
  tests/fixtures/fabdem/N62E006_FABDEM_V1-2_fixture.tif  — real FABDEM 1024×1024 px
  tests/fixtures/gebco/GEBCO_2024_fixture.tif             — real GEBCO 116×116 px
  tests/fixtures/egm2008/egm2008_fixture.tif              — real EGM2008 55×55 px
  tests/fixtures/ice6g/ICE6G_fixture.nc                   — real ICE-6G 62×61 pts, 2 epochs

All files were cut from local real datasets for tile N62E006 (62-63°N, 6-7°E,
Sunnmøre / Ålesund, Norway) — 50 % land / 50 % ocean with max elevation 1058 m.
See tests/fixtures/README.md for full provenance.

Fallback: if fixtures are absent (unusual), calls
``scripts/generate_synthetic_data.py`` to produce calibrated synthetic data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Repository root
REPO_ROOT    = Path(__file__).parent.parent
# Real committed fixtures (primary — no download needed)
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
# Synthetic/downloaded fallback data (gitignored)
SYNTH_DIR    = REPO_ROOT / "tests" / "data"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_synthetic_data.py"

# Expected fixture files
_FIXTURE_FILES = [
    FIXTURES_DIR / "fabdem" / "N62E006_FABDEM_V1-2_fixture.tif",
    FIXTURES_DIR / "gebco"  / "GEBCO_2024_fixture.tif",
    FIXTURES_DIR / "egm2008" / "egm2008_fixture.tif",
    FIXTURES_DIR / "ice6g"  / "ICE6G_fixture.nc",
]


def _fixtures_present() -> bool:
    return all(f.exists() for f in _FIXTURE_FILES)


# ------------------------------------------------------------------
# Session-scoped: resolve data directory once
# ------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def synthetic_data_dir() -> Path:
    """Return the directory that holds the test rasters/NetCDF.

    Priority:
      1. ``tests/fixtures/``  (committed real data — preferred)
      2. ``tests/data/``      (downloaded / generated — fallback)

    If neither is ready, synthetic data is generated via
    ``scripts/generate_synthetic_data.py``.
    """
    if _fixtures_present():
        return FIXTURES_DIR

    # Fallback: generate synthetic data
    marker = SYNTH_DIR / ".generated"
    if not marker.exists():
        subprocess.check_call(
            [sys.executable, str(GENERATE_SCRIPT), "--output-dir", str(SYNTH_DIR)],
        )
    return SYNTH_DIR


# ------------------------------------------------------------------
# Per-file path fixtures  (resolve against whichever dir was chosen)
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def fabdem_dir(synthetic_data_dir: Path) -> Path:
    """Directory containing FABDEM tile(s)."""
    return synthetic_data_dir / "fabdem"


@pytest.fixture(scope="session")
def gebco_path(synthetic_data_dir: Path) -> Path:
    """Path to GEBCO 2024 sample GeoTIFF."""
    d = synthetic_data_dir / "gebco"
    # fixtures name vs legacy synthetic name
    for name in ("GEBCO_2024_fixture.tif", "GEBCO_2024_N51E000.tif"):
        p = d / name
        if p.exists():
            return p
    return d / "GEBCO_2024_fixture.tif"   # let test fail with a useful path


@pytest.fixture(scope="session")
def egm2008_path(synthetic_data_dir: Path) -> Path:
    """Path to EGM2008 undulation clip."""
    d = synthetic_data_dir / "egm2008"
    for name in ("egm2008_fixture.tif", "egm2008_N51E000.tif"):
        p = d / name
        if p.exists():
            return p
    return d / "egm2008_fixture.tif"


@pytest.fixture(scope="session")
def ice6g_path(synthetic_data_dir: Path) -> Path:
    """Path to ICE-6G_C VM5a NetCDF."""
    d = synthetic_data_dir / "ice6g"
    for name in ("ICE6G_fixture.nc", "ICE6G_sample.nc"):
        p = d / name
        if p.exists():
            return p
    return d / "ICE6G_fixture.nc"


@pytest.fixture(scope="session")
def ice7g_path(synthetic_data_dir: Path) -> Path:
    """Path to ICE-7G thickness NetCDF (fixture if available)."""
    d = synthetic_data_dir / "ice7g"
    for name in ("ICE7G_fixture.nc", "ICE7G_sample.nc"):
        p = d / name
        if p.exists():
            return p
    return d / "ICE7G_fixture.nc"


# ------------------------------------------------------------------
# Constant fixtures  (N62E006: 62-63°N, 6-7°E, Norwegian coast)
# ------------------------------------------------------------------


@pytest.fixture
def tile_bounds() -> tuple[float, float, float, float]:
    """Bounding box of the fixture tile (lon_min, lat_min, lon_max, lat_max)."""
    if _fixtures_present():
        # Full 1°×1° tile N62E006 (62-63°N, 6-7°E)
        return (6.0, 62.0, 7.0, 63.0)
    # Legacy synthetic tile N51E000
    return (0.0, 51.0, 1.0, 52.0)


@pytest.fixture
def tile_id() -> str:
    """FABDEM tile identifier for the test region."""
    return "N62E006" if _fixtures_present() else "N51E000"


# ------------------------------------------------------------------
# Tiny in-memory DataArrays for fast unit tests
# ------------------------------------------------------------------


@pytest.fixture
def simple_land_da() -> xr.DataArray:
    """20×20 DataArray: southern 10 rows are land (100 m), north is NaN."""
    rng = np.random.default_rng(42)
    arr = np.full((20, 20), np.nan, dtype=np.float32)
    arr[10:, :] = rng.uniform(50.0, 150.0, size=(10, 20)).astype(np.float32)
    lats = np.linspace(62.84, 62.58, 20)   # N62E006 latitude range
    lons = np.linspace(6.68, 6.96, 20)
    return xr.DataArray(arr, dims=["y", "x"], coords={"y": lats, "x": lons})


@pytest.fixture
def simple_ocean_da() -> xr.DataArray:
    """20×20 DataArray: uniform -50 m (ocean)."""
    arr = np.full((20, 20), -50.0, dtype=np.float32)
    lats = np.linspace(62.84, 62.58, 20)
    lons = np.linspace(6.68, 6.96, 20)
    return xr.DataArray(arr, dims=["y", "x"], coords={"y": lats, "x": lons})
