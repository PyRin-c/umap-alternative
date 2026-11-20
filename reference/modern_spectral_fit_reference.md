# UMAP 0.6.0 connected modern spectral fit

Seeded modern Adam and momentum support spectral initialization through
ordinary `fit` and `fit_transform`. The package's default
`init="spectral"` selects it for connected graphs. The public 0.6.0
wheel probes pass `init="spectral"` explicitly to select the same route.

## Connected solver and RNG

The eigensolver uses the sampled graph after dropping weak memberships.
The public `graph_` retains its original memberships. The pruning fixture
has a connected irregular tree with unit memberships and unilateral
memberships of `0.0625`. Its 12-epoch budget removes the unilateral edges
from the eigensolver and native optimizer.

For `k = n_components + 1 < n_rows`, SciPy's `eigsh` uses the float32
normalized adjacency `D^(-1/2) @ W @ D^(-1/2)`, the largest algebraic
eigenvalues, an all-ones start, `tol=1e-4`, and
`ncv=max(2*k + 1, floor(sqrt(n_rows)))`. The 83-row holdout exercises
the square-root term. The layout takes the eigenvectors in descending
eigenvalue order and drops the first. At `k == n_rows`, the tested
fits use a dense float64 normalized Laplacian and `scipy.linalg.eigh`
in ascending eigenvalue order, again dropping the first eigenvector.
The solver's signs are retained.

The eigensolver consumes no words from the caller's `RandomState`. The
selected coordinates are expanded to a maximum absolute coordinate
of 40 and rounded to float32. The caller then draws a single
`normal(scale=0.01, size=(n_rows, n_components))` block, including any
Gaussian already cached. The sampled noise is cast to float32 before the
addition. Rust receives the prepared coordinates and the advanced state
at the existing centering, rescaling and optimizer handoff. The public
fixtures record the cursor and Gaussian cache before and after jitter
and after fitting, along with the next four raw words.

When `n_components >= n_rows`, the estimator emits the observed rank
warning and draws a float64 uniform layout on `[-10, 10)`. It uses the
same amplitude and jitter stages as the spectral solve. Tests exercise
dimensions below, at, and above the limit, including the three-row
full-rank solve and a warm Gaussian cache in the fallback.

## Exact fixtures and residuals

The pinned public wheels were observed on Ubuntu and Windows x86-64
using CPython 3.13.12 and one thread. Every regular configuration was
repeated through both estimator entry points. The installed native wheel
matches every same-platform CSR, final embedding, checkpoint, and RNG
record for these configurations.

| Optimizer | Configurations per host | Epochs | Reported snapshots | Checkpoint coordinate bits | Final coordinate bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Momentum | 10 | 45 | 34 | 1,726 | 469 |
| Adam | 10 | 53 | 31 | 2,020 | 547 |

Fixture regeneration reproduced every regular case on both hosts.
The installed release wheels each passed 1,399 Python tests.
The locked Rust suite passed 183 tests with three existing ignored
probes on each host. Formatting and Clippy with warnings denied passed.

The two public wheels agreed on all graph and RNG records. The
`adam-rank-limit` fit has a platform-specific coordinate record. Its
configuration uses the default fitted curve and differs by one float32
step, `5.96e-8`, at one coordinate in each of the last two checkpoints.

The symmetric six-vertex cycle has a repeated eigenvalue. With the
jitter values zeroed through the caller's public `RandomState.normal`,
eight fixed-seed wheel fits returned eight distinct bases on each host.
The maximum pairwise coordinate differences were 2.63234 on Ubuntu and
2.54402 on Windows. After orthonormalizing the columns, the projectors
onto the selected subspace varied by at most `2.56e-7` and `1.52e-7`,
respectively. The test checks that subspace alongside the CSR and RNG
bits. Regenerating the fixture exercises this variability. The
`rank_degenerate.observations` arrays preserve the signed coordinate
bases after layout rescaling.

A separate weighted graph probe used an eight-row precomputed distance
matrix. Set its diagonal to zero, each other entry `[i, j]` to
`8 + i + j`, and each symmetric ring edge `(i, (i+1) % 8)` to
`1 + i/10` in float32. Fit with `metric="precomputed"`,
`n_neighbors=4`, `set_op_mix_ratio=0.1`,
`n_epochs=[12, 2, 0, 2, 5]`, `negative_selection_range=3`,
`exclude_graph_neighbors=True`, and seed 109 with momentum. Compared
with the Ubuntu wheel, 12 CSR weights differed by one float32 step.
The coordinate residual started at `2.09e-6` after initialization and
reached `2.15e-6` at the final checkpoint. That case exercises an
existing graph-arithmetic difference. The exact pruning fixture uses
binary weights to isolate the spectral and epoch handoff.

Disconnected spectral placement, recursive initialization and large
or approximate graphs remain separate coverage. The connected solve
retains SciPy solver errors and makes no sign or basis guarantees for
new spectra with tied eigenvalues.

## Reproduce

On Ubuntu or WSL from the repository root, use temporary and cache
directories under the WSL home:

```bash
export TMPDIR="$HOME/.cache/umap-alternative-verification"
mkdir -p "$TMPDIR"
export UV_PROJECT_ENVIRONMENT="$TMPDIR/reference-venv"
export NUMBA_CACHE_DIR="$TMPDIR/modern-sampling-numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python -IB reference/generate_modern_spectral_fit_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-spectral-fit-ubuntu-x86_64.json
export UV_PROJECT_ENVIRONMENT="$TMPDIR/venv-modern-positive"
export CARGO_TARGET_DIR="$TMPDIR/target-modern-spectral"
uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked
uv run --no-sync maturin build --release --locked --interpreter python --out "$TMPDIR/modern-spectral-wheels"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --reinstall "$TMPDIR/modern-spectral-wheels/"*.whl
uv run --no-sync python -m pytest -q
```

On Windows:

```powershell
$spectralChecks = Join-Path $env:LOCALAPPDATA 'Temp/umap-modern-spectral-checks'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $spectralChecks 'reference-venv'
$env:NUMBA_CACHE_DIR = Join-Path $spectralChecks 'numba'
$env:OMP_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = $env:MKL_NUM_THREADS = $env:NUMBA_NUM_THREADS = '1'
uv run --project reference --locked python -IB reference/generate_modern_spectral_fit_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-modern-spectral-fit-windows-x86_64.json
$env:UV_PROJECT_ENVIRONMENT = Join-Path $spectralChecks 'dev-venv'
$env:CARGO_TARGET_DIR = Join-Path $spectralChecks 'target'
uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked
$wheelDir = Join-Path $spectralChecks 'wheel'
uv run --no-sync maturin build --release --locked --interpreter python --out $wheelDir
$wheel = Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' -File | Select-Object -First 1
uv pip install --python "$env:UV_PROJECT_ENVIRONMENT/Scripts/python.exe" --no-deps --reinstall $wheel.FullName
uv run --no-sync python -m pytest -q
```
