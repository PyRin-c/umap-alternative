# Recursive contraction in UMAP 0.6.0

The generator records fresh observations from the pinned public wheel. It
uses `UMAP.fit`, `UMAP.fit_transform` and the public `label_propagation_init`
function. Direct placement probes supply the graph and anchor explicitly.
Fit probes record the CSR, every requested epoch checkpoint, the caller
RandomState cursor and Gaussian cache, and its next four uint32 words.
The six archived input recipes from the initial recursive study are rerun.
Their original fixtures and measurements remain available.

## Contraction and expansion

- The PCA base case applies through `base_init_threshold`, inclusive. For
  the first contraction, select the greater of the requested ratio's floor
  and the threshold capped at half the rows. Later contractions use the
  ratio's floor. Use at least two parts. The threshold-six fixtures fit
  five, six and seven rows, and the default-threshold fixture fits 1,025.
- Rank hubs by the layout graph's weighted degree. Keep the largest
  degrees in ascending order, with stable row ordering for equal degrees.
  A weighted round sums each known part's neighbor support. Assign the
  largest support at or above `1.0`, then start the next round with the new
  labels. The sparse multiplication's output order resolves equal support.
  The fixtures bracket `1.0` with pairs of adjacent float32 weights around
  `0.5` and permute both equal-support and equal-degree examples.
- Sweep weakly attached rows in row order and copy their first known
  neighbor in CSR order. A row still unassigned after these sweeps takes
  a signed Tausworthe draw modulo the number of parts. Its state comes from
  adding the row index to each of the three seed words drawn for the
  current level. This shares the existing Rust generator and consumes no
  extra caller RNG draws. An isolated row supplies zero neighbor average
  and still contributes to its part's mean and the intermediate layout.
- Transition contraction carries the raw sums of cross-part memberships.
  The layout graph normalizes those sums by outgoing weight and joins
  the two directions with fuzzy union. The layout weights determine hub
  degrees and expansion. Fuzzy-union mode combines cross-part memberships
  directly and honors `remove_coarse_diagonal`. The three-level transition
  probe distinguishes the weights used for scoring, hub ranking and
  expansion.
- The verified coarse budgets are zero to three for a single contraction
  and zero or one for multiple contractions. A positive coarse budget
  uses native Adam with the ordinary schedule, learning rate `0.5`, and one projected
  candidate refresh per epoch. The first coarse epoch advances the
  projection RNG before an edge can update. Outer Adam and momentum
  continue to use their recursive refinement schedules.

## Installed-wheel verification

The results use CPython 3.13.12 and the same pinned NumPy, SciPy,
scikit-learn, Numba and llvmlite versions as the initial recursive study.
Both hosts run with one numerical thread. Graph and RNG comparisons use
exact bits, including the 1,025-row graph against the wheel on its host.
The coordinate comparisons use absolute tolerances. The largest drift
occurs after positive outer updates in the two rank-one holdouts, where
several initialized rows are near coincident.

| Check | Windows | Ubuntu WSL |
| --- | --- | --- |
| Direct placements with exact RNG | 11 | 11 |
| Largest direct placement coordinate residual | `4.769e-7` | `4.769e-7` |
| Fits with exact CSR and caller RNG, through each fit method | 16 | 16 |
| Largest fit residual at epoch zero | `1.479e-5` | `1.479e-5` |
| Largest fit residual after outer updates | `2.289e-4` | `2.289e-4` |
| Installed-wheel Python tests passed | 1,573 | 1,573 |
| Locked Rust tests passed, ignored | 184, 4 | 184, 4 |

The two-coarse-epoch holdout with a nondefault curve has a `1.479e-5`
residual at outer epoch zero and `1.634e-5` after the last outer update.
All 16 fits and 11 direct placements are tolerance comparisons for their
coordinate arrays. Formatting and warning-denied Clippy passed on both hosts.

The third coarse epoch across multiple levels remains unresolved. The
unrestricted candidate for `holdout-three-coarse-epochs` matched graph
and RNG bits but differed by up to `0.290781` at outer epoch zero and
`0.691123` after the last outer update on both hosts. The supported API
rejects that budget before advancing the caller RNG. Longer and default
coarse budgets, alternate coarse optimizer settings, PCA subsampling
and other anchor methods also remain outside the verified path.

## Reproduce

Keep temporary files, caches, environments, Cargo targets and wheel outputs
on C: for Windows or under the Ubuntu home for WSL. Set
`CARGO_BUILD_JOBS=2` and set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `NUMBA_NUM_THREADS` to `1`. Generate the fixture
in the reference environment and use a separate project environment for
the build. Use the matching host suffix in the output filename.

```sh
uv run --project reference --locked python -B reference/generate_modern_recursive_contraction_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-contraction-ubuntu-x86_64.json

uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked --quiet
uv run --no-sync maturin build --release --locked --interpreter python --out "$wheel_dir"
uv pip install --python "$project_python" --no-deps --reinstall "$wheel_file"
uv run --no-sync python -B -m pytest -p no:cacheprovider -q
```
