# Disconnected modern spectral initialization

The pinned `umap-learn==0.6.0` wheels were observed through public
`UMAP.fit` and `UMAP.fit_transform` with `init="spectral"`, one job, and
the modern Adam or momentum layout. The generator records the fitted
CSR, the float32 coordinate bits, the initialization draws, and the
caller RandomState cursor, cached Gaussian, and next four uint32 words.
Each exact case runs through both fit methods.

## Placement and RNG

- Find components after applying the epoch-dependent sampling cutoff.
  The weak-bridge fixture keeps a connected public `graph_` and produces
  two 9-row components for initialization at 12 epochs.
- For at most twice the requested dimension in components, assign the
  positive axes followed by their negatives in component-label order.
  Above that count, build `exp(-distance**2)` between component centroids
  and embed it using scikit-learn with the caller RandomState. For dense
  precomputed distances, use the float64 mean of each directed off-diagonal
  block before the exponential. This also preserves infinite distances set
  by a cutoff. The directed-block fixture distinguishes averaging the
  two blocks before the exponential from passing their separate affinities
  to scikit-learn.
- Place rows within half the smallest positive distance to each center.
  Components below `max(2 * n_components, n_components + 2)` use a
  uniform block in component-label order. Larger ones use the same
  modern normalized-adjacency eigenvectors as the connected path and
  consume no local RNG. The width-three fixture has components of sizes
  8, 7, 6, 5, and 3, exercising the boundary between a solve and uniform
  fallback. Disconnected placement precedes the whole-graph rank check.
  Dimensions equal to or above the row count still use these per-component
  draws, with no connected rank warning.
- Scale the assembled float32 coordinates to a maximum absolute value
  of 40. Add one float32 Gaussian block with standard deviation `0.01`
  before handing the layout and caller RNG to the existing native path.
  An isolated row receives a singleton uniform block and participates in
  centering, rescaling, and optimization checkpoints. Its final embedding
  row is NaN. If all rows are isolated, both estimators raise a `ValueError`
  before advancing the supplied RNG.

## Installed-wheel comparisons

The Windows and Ubuntu x86-64 fixtures use CPython 3.13.12, NumPy 2.5.5,
SciPy 1.18.1, scikit-learn 1.9.1, Numba 0.68.1, and llvmlite 0.50.0,
with the numerical thread counts set to one. The project release wheels
match all 16 exact cases on each host, through both fit methods. The
checks include CSR structure and membership bits, initial and final
coordinate bits, intermediate epoch checkpoints, and caller RNG state
and next words. Exact cases cover natural disconnections, connections
removed by cutoff, epoch pruning, a zero negative rate, a small projection
range, and disconnected rank limits. The precomputed cutoff case also
reproduces the scikit-learn warning about its disconnected component affinity.
On both hosts the installed project wheel passed 1,440 Python tests.
The locked Rust workspace passed 184 tests with four ignored, and passed
the formatting and Clippy checks.

The six-cycle plus pair fixture exercises an intrinsically repeated
eigenvalue. It zeros the returned jitter while consuming the full Gaussian
block and stops at epoch zero. Compare the QR projector of the cycle's
centered coordinates with the analytic sine/cosine projector. Eight public
fits produced eight distinct coordinate arrays on each host. In the recorded
samples the largest coordinate difference between public repeats was 2.403
on Windows and 2.385 on Ubuntu. Both projector errors stayed below `1.2e-7`.
The regression test
checks the projector, the graph and the RNG. Solver basis and sign choices
can still affect coordinates and subsequent iterations for new symmetric
or near-degenerate graphs. Recursive initialization remains separate.

## Reproduce

Use the host-specific `windows` or `ubuntu` suffix. Run the public fixture
generator in the reference environment and use a separate environment to
build and install the project wheel. Set `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMBA_NUM_THREADS` to `1`.
Set `UV_PROJECT_ENVIRONMENT`, `NUMBA_CACHE_DIR`, `CARGO_TARGET_DIR` and
the wheel output directory under a C: temp directory or the Ubuntu home.

```sh
# In the reference environment
uv run --project reference --locked python -B reference/generate_modern_disconnected_spectral_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-disconnected-spectral-ubuntu-x86_64.json

# In the project environment
uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked --quiet
uv run --no-sync maturin build --release --locked --interpreter python --out "$wheel_dir"
uv pip install --python "$project_python" --no-deps --reinstall "$wheel_file"
uv run --no-sync python -B -m pytest -p no:cacheprovider -q
```
