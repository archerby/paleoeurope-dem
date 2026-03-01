# ICE-7G Fixture (Local Generation)

The open-demo GIA pipeline mirrors production by using two matrices per epoch:

- ICE-6G for bedrock (GIA) correction
- ICE-7G for ice thickness (`stgit`)

For redistribution/size reasons, `ICE7G_fixture.nc` may be generated locally.

## Generate

1. Place the ICE-7G per-epoch NetCDF files in the path configured in:
   `scripts/prepare_test_fixtures.py` (`ICE7G_DIR`).
2. Run:

```bash
python scripts/prepare_test_fixtures.py
```

Expected input filenames:

- `I7G_NA.VM7_1deg.0.nc`
- `I7G_NA.VM7_1deg.8.nc`
- `I7G_NA.VM7_1deg.12.nc`
- `I7G_NA.VM7_1deg.21.nc`

Output:

- `tests/fixtures/ice7g/ICE7G_fixture.nc`
