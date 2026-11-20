# Seeded modern recursive initialization

The [contraction follow-up](modern_recursive_contraction_reference.md)
replays the six residual probes below and records new holdouts. The
measurements in this file describe the original implementation.

The reference generator uses `UMAP.fit` and `UMAP.fit_transform` from the
pinned `umap-learn==0.6.0` wheel. It records the CSR, every requested epoch
checkpoint, each initialization sketch and seed draw, and the caller
RandomState's cursor, cached Gaussian, and next four uint32 words. The
project fits the same data through its installed release wheel.

## Base case and hierarchy

- Modern `init="recursive"` and `init="auto"` use the same recursive path.
  The initial anchor is a randomized PCA projection with eight extra
  sketch columns, one power iteration, and float32 per-column range
  normalization. It draws the sketch once from the caller's RandomState.
  The default samples all rows through 16,384 and subsamples above that.
  This port covers the full-data sketch.
- At or below `base_init_threshold`, which defaults to 1,024, the base
  case uses the PCA anchor even for disconnected graphs. Center each
  coordinate and scale its 5th-to-95th percentile span to
  `3 * log10(n_rows) * log2(depth + 1)`. The root is at depth one.
  Degree-zero rows participate in the base and checkpoint layouts and
  receive NaNs in the final embedding.
- Above the threshold, the verified part count is the greater of
  `n_rows // recursive_coarsening_ratio` and
  `min(base_init_threshold, n_rows // 2)`. The supported graphs have a
  strict weighted-degree boundary between hubs and other rows. Each
  component of non-hub rows attaches to exactly one hub. The fixtures
  cover attached paths, connected and disconnected groups, ratios 2, 3,
  and 4, and two successive coarsenings. Each level averages the anchor
  within each part and consumes one three-word sampler seed, including
  at `n_embedding_epochs=0`.
- For the next level, all cross-part edges in these graphs join hubs.
  Transition weights normalize the off-diagonal weights by row, then
  combine the two directions with fuzzy union. Fuzzy-union weights
  retain the cross-part memberships and optionally retain coarse
  diagonal memberships. `coarsen_on_layout_weights=False` uses the
  fuzzy-union graph for the partition. Expansion averages each row's
  part coordinates with its graph-weighted neighbor average. It then
  centers the result and aligns it to the parent anchor with an
  orientation-preserving Procrustes rotation.

## Positive outer epochs

The prepared coordinates go through the same native initialization and
sample dispatch as spectral and explicit layouts. Recursive fits select
a refinement schedule. In the formulas `e` starts at zero, `N` is the
outer epoch budget, `r = 1 - e/N`, and
`p = 1 - max(e - 1, 0)/N`.

| Optimizer | Recursive schedule |
| --- | --- |
| Momentum | `alpha = learning_rate * r**2 / 8 + 0.0025`, `carry = 0.5 - 0.5 * p**2` |
| Adam | `k = min(N // 8, 100)`. Before `k`, `alpha = 0.1 + (0.5 * learning_rate - 0.1) * (1 - e/k)**2`. From `k`, `alpha = 0.2 - 0.15 * (e - k)/(N - k)`. |

Public momentum fits still move at zero learning rate with nonzero
forces. The Adam fixtures cross the `N // 8` phase boundary and verify
the handoff with a nondefault learning rate. Independent first-move
probes at budgets 800 and 1,600 observe the same capped early step.

## Verification and remaining cases

The Windows and Ubuntu WSL x86-64 records use CPython 3.13.12, NumPy
2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, Numba 0.68.1, and llvmlite
0.50.0 with one numerical thread. All 23 supported cases agree on CSR
and RNG bits on both hosts and through both fit methods. The largest
absolute coordinate residual was `3.815e-6` at epoch zero in the 1,024-row
base case and `1.193e-6` after positive epochs. Two repeated-singular-value
probes verify the analytic centered PCA projector. Changing the seed
changed their coordinate arrays by up to 1.06246. The installed project
wheel passed 1,534 Python tests on each host. The locked Rust workspace
passed 184 tests with four ignored, and formatting and Clippy passed.

The remaining public probes keep the actual layouts and RNG trace for
competing hubs, a three-level ratio-two case, a degree-zero row above
the base threshold, and a positive coarse epoch budget. On the linked
stars, increasing the coarse budget from zero to three changes the
outer epoch-zero coordinates by up to 0.20995 and adds three shuffles
of the three coarse rows. The 1,025-row probe confirms one additional
sampler seed above the default threshold.
Its large public CSR is stored as the SHA-256 of concatenated
little-endian uint32 `indptr`, `indices`, and membership bits. The
implementation rejects unsupported branches before advancing the
caller RNG. General label propagation, the above-threshold isolated-row
placement, positive coarse optimization, PCA subsampling, and other
anchor methods still need reconstruction. The existing spectral,
random, and explicit initializers keep their preparation and schedules.

## Reproduce

Use a temporary directory on C: for Windows or under the Ubuntu home
for WSL. Set `UV_PROJECT_ENVIRONMENT`, `NUMBA_CACHE_DIR`, and
`CARGO_TARGET_DIR` there. Use separate environments for the reference
wheel and the project wheel. Set `CARGO_BUILD_JOBS=2` and set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMBA_NUM_THREADS` to `1`. Use the corresponding host suffix below.

```sh
# Reference environment
uv run --project reference --locked python -B reference/generate_modern_recursive_fit_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-fit-ubuntu-x86_64.json

# Separate project environment
uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked --quiet
uv run --no-sync maturin build --release --locked --interpreter python --out "$wheel_dir"
uv pip install --python "$project_python" --no-deps --reinstall "$wheel_file"
uv run --no-sync python -B -m pytest -p no:cacheprovider -q
```
