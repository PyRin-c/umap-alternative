# UMAP 0.6.0 modern initialization

The fixture records public calls through `UMAP.fit`, `simplicial_set_embedding`
and `recursive_init`. It uses one thread and the pinned wheel environment in
`reference/uv.lock`. Windows and Ubuntu produced identical graph, coordinate,
RNG and validation records. The fixture records the operating system as
metadata.

Rust accepts an explicit float32 layout, random initialization, supplied
spectral vectors, or prepared coordinates. The initializer owns the
coordinate draws. The caller draws the optimizer's three seed words
afterward, including at zero epochs. The Python estimator passes both the
MT19937 state and the cached Gaussian through the native boundary. Duplicate
explicit rows trigger jitter scaled by the mean nearest-row distance.
All-coincident layouts still consume the Gaussian draws even when their
jitter scale is zero.

| Rescaling | Observed rule |
| --- | --- |
| `per_axis` | Center each column, then set its 5th-to-95th percentile span to `3 * log10(n_rows) * init_scale_factor`. A zero span uses a denominator of one. |
| `isotropic` | Use the geometric mean of the corrected spans and a shared multiplier. |
| `graph_edge` | Scale by the weighted median length of stored positive edges. A zero length uses a denominator of one. A graph containing only stored zeros leaves the centered coordinates unscaled. |

The Rust graph-edge and per-axis checkpoints compare coordinate bits,
including powers of ten outside the float32 normal range, ties around
half the edge weight, self-loops, zero weights, and distances whose squares
underflow. Native Rust isotropic checkpoints allow a relative error of
`1.5e-6` for float32 logarithm and exponential rounding. The Python
estimator uses NumPy's robust-span rescaling after Rust centers the
coordinates. The zero-epoch Python tests
check the exact graph, coordinates, RNG cursor, Gaussian cache and next
four words for selected cases. They exercise Adam and momentum selection
and the implicit optimizer with `compatibility_layout=False`. Invalid
shapes and scale settings are checked before the first coordinate draw.

The [estimator fit study](modern_fit_reference.md) covers how the sampler's
cutoff affects graph-edge initialization and how repulsion strength sets
the target for robust-span initialization. The Python estimator uses the
same initialization path for zero and positive epoch budgets.

Rust spectral handoff tests use captured eigenvectors and allow an absolute
error of `1e-5`. Supplying the recorded recursive coordinates to the
rescaling stage reproduces every final coordinate bit. The independent
recursive model reproduces the observed caller RNG checkpoints, with
coordinate errors up to 10.48 on a held-out graph. Direct calls to
`simplicial_set_embedding` without an estimator can use a different robust
target. The fixture records two such calls with a zero target span. The
Python estimator supports random and explicit modern layouts. Recursive
and modern spectral initialization and transform and update parity for
modern fits remain open.

## Reproduce

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python -I reference/generate_modern_initialization_fixture.py \
  --output /tmp/umap-modern-init.json
uv run --project reference --locked python -I reference/model_modern_initialization.py \
  /tmp/umap-modern-init.json
cargo test --locked -p umap-core --test modern_initialization_fixtures --test modern_layout_fixtures
```
