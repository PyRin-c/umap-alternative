# UMAP 0.6.0 seeded dense modern fit

`UMAP.fit` and `fit_transform` run serial native Adam or momentum with
seeded random or explicit initialization. The Python estimators accept an
integer seed or a NumPy `RandomState`. A sorted, deduplicated epoch list
reports post-epoch positions and returns a separate final snapshot when
the list includes the epoch budget. The fitted training input still
returns `embedding_` on transform.

## Initialization and graph

Rust generates and centers the initial coordinates with the caller's
MT19937 state and Gaussian cache. NumPy scales the robust spans in
`isotropic` and `per_axis` modes. The target span is
`3 * log10(n_rows) * log2(1 + repulsion_strength) * init_scale_factor`.
The repulsion-strength factor affects both robust modes, including fits
with zero negative rate. The graph-edge initializer uses only the weights
retained by the sampling cutoff. In the weak-median probe, keeping the
weak leaf would move the median edge length from 1 to 5.

The public fitted graph retains weak memberships. A row with only pruned
edges remains finite in `embedding_`. A row with no edges in the fitted
graph is masked with NaN in the final output. Its initialized coordinates
still participate as possible repulsive targets, and the reported history
contains its finite coordinates.

## Optimizer and projection

The [serial sampling probes](modern_sampling_reference.md) identify the
positive and negative schedules and candidate order. Both optimizers read
the starting coordinates across an epoch. For each retained CSR edge they
process the attraction followed immediately by that edge's negative draws.
The distance uses the seeded blocked float32 reduction with a fused scalar
remainder. Scheduled force arithmetic promotes the float32 repulsion
strength to float64.

Adam accumulates each force into a float32 buffer, then applies the existing
moment update to the completed buffer. Momentum rounds the carried velocity
to float32 at the start of the epoch and rounds each addition of an
already learning-rate-scaled contribution to that same velocity. It adds
the completed velocity to the coordinate. The direct sample-list Rust
kernel retains its grouped replay semantics.

When the selection range is below the vertex count, Python prepares the
candidate order by normalizing the Gaussian direction in NumPy and sorting
the binary64 projections with `np.argsort`. Rust consumes that permutation
and continues to schedule edges and apply the optimizer. The 24-row
tied-projection fixtures use a caller-supplied direction to make the
observed tie order affect both embeddings. The full-range path dispatches
entirely in Rust and hands the final MT19937 state and Gaussian cache back
to the caller.

## Measured parity

The pinned public wheels recorded 33 configurations each on Windows and
Ubuntu x86-64. Every configuration was repeated through `fit_transform`
after `fit`. The Rust-backed estimator reproduced the same-platform CSR
offsets, indices and weight bits, final and checkpoint coordinate bits,
and caller RNG cursor, cache and next four raw words in every configuration.

| Optimizer | Configurations per platform | Epochs | Reported snapshots | Checkpoint coordinate bits | Final coordinate bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Momentum | 16 | 616 | 71 | 5,620 | 1,424 |
| Adam | 17 | 629 | 75 | 5,684 | 1,440 |

The two platforms' wheels differed in six configurations with fitted
default curve coefficients. The largest cross-platform coordinate
difference was `7.51019e-6` in the 500-epoch Adam case. Graph, RNG and
input records were identical. The Python tests select the same-platform
golden fixture and require exact coordinate bits in both entry points.
The locked workspace tests passed with 180 Rust tests and two existing
ignored probes. Both platforms passed all 1,240 Python tests, and the
workspace passed formatting and Clippy with warnings denied.

The [connected spectral study](modern_spectral_fit_reference.md) covers
the same fit path with spectral initialization.
Further coverage includes larger and approximate graphs, disconnected
spectral and recursive initialization, the hybrid compatibility optimizer,
density and nondefault negative scaling, and held-out transforms or
updates from a modern fit. The Rust-owned projection path still rejects
ambiguous ranking keys. The Python fit resolves these with its local
NumPy projection order.

## Reproduce

From the repository root on Ubuntu or WSL:

```bash
export TMPDIR="$HOME/.cache/umap-alternative-verification"
mkdir -p "$TMPDIR"
export UV_PROJECT_ENVIRONMENT="$TMPDIR/reference-venv"
export NUMBA_CACHE_DIR="$TMPDIR/modern-sampling-numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python -IB reference/generate_modern_fit_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-fit-ubuntu-x86_64.json
export UV_PROJECT_ENVIRONMENT="$TMPDIR/venv-modern-positive"
uv sync --locked
uv run --no-sync cargo test --workspace --locked
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
uv run --no-sync python -m pytest -q
```

From the repository root on Windows:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'Temp/umap-modern-fit-reference'
$env:NUMBA_CACHE_DIR = Join-Path $env:LOCALAPPDATA 'Temp/umap-modern-fit-numba'
$env:OMP_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = $env:MKL_NUM_THREADS = $env:NUMBA_NUM_THREADS = '1'
uv run --project reference --locked python -IB reference/generate_modern_fit_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-modern-fit-windows-x86_64.json
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA 'Temp/umap-modern-fit-venv'
uv sync --locked
uv run --no-sync python -m pytest -q
```
