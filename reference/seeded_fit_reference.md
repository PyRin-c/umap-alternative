# Public serial seeded dense fit

The Windows x86-64 fixture records 24 fresh public
`UMAP(..., compatibility_layout=True, n_jobs=1).fit_transform` runs against
`umap-learn==0.6.0`. Each checkpoint uses its own total epoch count. The
generator repeats each fit and checks the embedding and CSR bits before writing
the fixture. The 500-epoch runs also match the public estimator's default.

| Input and initialization | Seed | Neighbors | Components | Epochs |
| --- | ---: | ---: | ---: | --- |
| Weighted, explicit | 0 | 4 | 2 | 0, 1, 2, 3, 11, 30, 500 |
| Weighted, random | 4294967295 | 4 | 2 | 0, 1, 2, 3, 11, 30, 500 |
| Weighted, random | 42 | 7 | 3 | 0, 2, 3, 10, 30 |
| Three unit-weight pairs, random | 7 | 2 | 1 | 0, 1, 2, 3, 30 |

The fixture stores two input matrices and three distinct CSR graphs. Tests
compare every graph and all 374 coordinate values bit for bit through
`fit_dense_embedding_seeded`. All 24 checkpoints matched in the recorded
run. The max-seed 500-epoch checkpoint also distinguishes `f64` repulsion
strength `0.37` from the same value rounded to `f32`.

Run from the repository root in PowerShell.

```powershell
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'umap-seeded-fit-numba'
$env:CARGO_TARGET_DIR = Join-Path $env:TEMP 'umap-seeded-fit-target'
$fixture = 'reference/fixtures/umap-learn-0.6.0-seeded-fit-windows-x86_64.json'
uv run --project reference --locked python reference/generate_seeded_fit_fixture.py --output $fixture
cargo test -p umap-core --locked --test seeded_embedding_fit
```
