# Dense compatibility inverse transform

`generate_inverse_transform_fixture.py` records the public `inverse_transform`
of the locked `umap-learn==0.6.0` wheel. It uses the estimator constructor,
`fit`, `set_params`, and the public fitted `embedding_` attribute. After
fitting with zero epochs, it assigns a fixed embedding and changes the epoch
count, negative rate, and repulsion to isolate the updates. The two original
45-epoch comparisons reuse the fitted embedding and query coordinates from
the existing transform fixture. They pass the curve parameters recorded in
this repository. [Original transform checkpoints](transform_reference.md)

## Initialization and geometry

- The replay uses a joggled Delaunay triangulation with `QJ` and locates the
  whole query batch in that triangulation. A Rust caller supplies the simplex
  vertex indices and one simplex location per query. The first vertex of the
  located simplex seeds the inside-hull search. [SciPy Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
- Outside the hull, the first minimum squared distance at float32 precision
  seeds the same neighborhood search. In the precision probe, that choice
  selects vertex 11 after traversal. Seeding with a float64 sum would select
  vertex 6. The fixture records the exterior warning and includes a collinear
  embedding which joggling handles.
- With `k = min(original feature count, training row count)`, visit up to
  `k` vertices in breadth-first order. For each visit, discover its unseen
  Delaunay neighbors in ascending vertex order. Rank all discovered candidates
  by distance, including those discovered during the last visit, and keep
  the closest `k`.
- For ranking and the curve, subtract and square in float32, then sum and
  take the square root in float64. Evaluate `1 / (1 + a * d^(2 * b))` and
  normalize the memberships in float64. Accumulate the feature coordinates
  in ascending training row order, rounding each addition to float32.

## Positive epochs

- An explicit scalar fit setting gives `floor(n_epochs / 3)` inverse epochs.
  An epoch list uses its largest value. With the default setting, 9,999 and
  10,000 query rows matched an explicit fit setting of 300. A batch of
  10,001 rows matched 90. The fitted inputs were unchanged in this comparison.
- Use the maximum raw membership across the batch. Divide each raw membership
  by that maximum, multiply by the inverse epoch count, and divide the
  epoch count by the result. The arithmetic is float64. The targeted
  interval cases straddle epoch two before rounding to float32 and include a
  five-epoch run where `5 / (5 * (1 / 3))` is above three.
- Visit query rows in input order and their edges in distance order. Apply
  attraction and then consume the negatives due on each active edge. Start
  with one quarter of the fit learning rate and set the rate after each epoch
  to the initial inverse rate times `1 - epoch / inverse_epochs`. Zero and
  one inverse epoch preserve initialization.
- The feature-space gradient divides the coordinate difference by
  `distance + 1e-6`. For attraction the multiplier is
  `-1 / (raw_weight * sigma + 1e-6)`. For repulsion the sampled tail has
  membership `exp(-max(0, distance - rho) / (sigma + 1e-6))`. Its multiplier
  is `repulsion_strength * membership / ((1 - membership) * sigma + 1e-6)`.
  Clip each gradient to `[-4, 4]` before applying the learning rate. Only
  the query reconstruction moves.
- The seeded inverse path initializes one shared negative stream with the
  three MT19937 seed words. Duplicate queries in the same batch can draw
  different negatives. The replay fits the original-space Euclidean
  `rho` and `sigma` from exhaustive training neighbors and the existing
  smooth scale search, or accepts fixed scales explicitly.

## Public boundaries

| Input or state | Public result |
| --- | --- |
| Empty, wrong-width, or nonfinite queries | `ValueError` |
| Sparse query matrix | `TypeError` |
| Sparse fitted input, graph mode, densMAP, precomputed metric, or Hamming metric | `ValueError` |
| One-dimensional embedding | `ValueError` with `Need at least 2-D data` |
| One fitted row or fewer than `n_components + 2` points | `QhullError` in the tested geometries |
| Collinear two-dimensional embedding | Finite results and the exterior warning in the recorded probe |
| Exterior query whose squared distances overflow | The wheel warned and returned zero feature coordinates at zero epochs |
| Before fitting | `AttributeError` |

## Python estimator

`UMAP.inverse_transform` supports dense compatibility fits with Euclidean,
Manhattan, Chebyshev, cosine, or correlation distance and
an integer transform seed. It reads the current public `embedding_` and
locates the query batch with SciPy's `Delaunay(..., qhull_options="QJ")`.
Each call returns a fresh float32 array with the original feature width.
The original features, effective neighbor count, local connectivity, curve,
and learning rate come from the fit. The estimator reads `n_epochs`,
`transform_seed`, `repulsion_strength`, and `negative_sample_rate` at call
time. A list of epochs can be set after fitting. The seeded Python fit
accepts scalar epoch settings.

Graph fits and sparse training are unavailable for inverse transformation.
The inverse adapter derives scales from the fitted feature metric and
cutoff, or from a retained copy of fit-time supplied neighbor distances.
The supplied distances can be mutated after fitting without changing a
reconstruction. Approximate inverse fits and additional metrics remain open.

The Rust implementation keeps candidate discovery order on equal distances.
The larger circle and duplicate-grid probes exercise the different equal-key
order of the wheel's sorting path. High-dimensional exterior reductions can
also select a different seed. A query whose memberships all round to zero
returns zero coordinates at initialization and raises `ValueError` when
optimization is active. For extreme exterior queries, the public wheel also
emits NumPy's overflow `RuntimeWarning`. The Rust path emits the exterior warning.

## Verification

Windows and Ubuntu WSL produced identical reference records. The Rust replay
matched every coordinate bit in the regression fixture on both platforms.

| Checkpoint | Equal float32 coordinates |
| --- | ---: |
| Initialization and one-epoch runs | 2,365 / 2,365 |
| Positive epochs with attraction alone | 156 / 156 |
| Positive epochs with attraction and repulsion | 288 / 288 |
| Separate tied-distance diagnostics | 165 / 190 |

The optimized counts include 9/9 coordinates for each of the original
45-epoch interior and exterior checkpoints. The tied-distance differences
occur in 4 of 15 cases, where a tie changes which vertices fill the candidate
quota.

The Python tests replay the locked checkpoints through the estimator and
also reproduce both original 45-epoch results from a real fit and forward
transform. Separate public-wheel probes verify the initialization, first
active epoch, transform seeds 0 and `2**32 - 1`, fractional fitted local
connectivity, and a distinct fit with a truncated neighbor count. These
checks pass on fresh wheel installations for Windows and Ubuntu WSL.
An active inverse optimizer uses the original-space metric gradient. The
Manhattan gradient is the coordinate sign, and Chebyshev uses the sign
on the greatest-difference axis. The angular gradients round each
component to float32 before applying the scale and learning rate.
Cosine rounds coordinate products to float32 and sums in float64.
Correlation centers and multiplies in float64.

The metric and scale fixture records 50 public checkpoints, including
one-step and longer attraction and repulsion, zero norms, constant rows,
finite cutoffs, and supplied widths greater than the effective neighbor
count. Windows and Ubuntu WSL regenerate the same record. The port
matches 448 of 450 coordinate bits. Two coordinates in the correlation
attraction-only checkpoint at fit epoch 30 differ by one float32 ULP.

## C11 inverse reconstruction

`umap_capi_inverse_transform_dense` borrows the original features, fitted
embedding, queries, and geometry for one call. Provide a row-major Delaunay
simplex array with `n_components + 1` vertex indices per simplex and keep the
vertex order from the triangulator. Provide one simplex index per query in
the same coordinate space. Use `UMAP_CAPI_INVERSE_EXTERIOR` (`SIZE_MAX`) for
an outside location. The inverse starts those queries at the first nearest
fitted vertex under the float32 squared-distance rule and reports their count
in the owning result's view. The caller constructs the geometry and locates
the queries. The result owns the row-major reconstructed features. An empty
query batch returns `UMAP_CAPI_STATUS_INVALID_SHAPE`. [Public C header](../crates/umap-capi/include/umap_capi.h)

For an exhaustive fit, pass `NULL` for scales and supply the effective fit
neighbor count, fitted local connectivity, metric, and any fit cutoff. For
other fitted neighborhoods, pass one `(rho, sigma)` pair per training row
or the original ranked distances with their full supplied width. Keep the
fit-time cutoff decisions as positive infinity in the ranked array. The
smoothing target is the effective neighbor count, which can be smaller than
the supplied width.

The C11 test matched all 99 coordinate bits across 11 runs on Windows x64
with MSVC and Ubuntu 24.04 WSL x64 with GCC. It uses the
original Euclidean initialization and optimized interior and exterior
checkpoints, the Manhattan, Chebyshev, cosine, and correlation checkpoints
at 30 fit epochs with repulsion one, `supplied_width`, `supplied_cutoff`,
and `finite_cutoff` through both native derivation and the fitted-scale input.
The metric test geometry was obtained from the recorded `embedding` and
`queries` float32 arrays with public SciPy 1.18.1 `Delaunay` using `QJ` and
`find_simplex`. The C test preserves the returned simplex order and maps
SciPy's `-1` locations to `SIZE_MAX`. The ranked arrays reproduce the fixture
generator's scaled distances and inclusive cutoff mask. The direct scale
pairs were derived by the Rust scale fitter from the same `finite_cutoff`
case. [C11 checkpoints and inputs](../crates/umap-capi/tests/c_inverse_smoke.c)

The C path uses the same Rust candidate order, exterior distance reduction,
and metric gradients as the Rust replay. The tied-distance, high-dimensional
exterior, and one-ULP correlation differences described above apply to this
entry point with the same supplied geometry.

## Reproduction

Run from the repository root with the locked reference environment. Keep
the reference venv, build products, and Numba cache under LocalAppData on
Windows or the Linux user cache under WSL.

```powershell
$inverseReference = Join-Path $env:LOCALAPPDATA 'Temp\umap-inverse-reference'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $inverseReference 'venv'
$env:UV_CACHE_DIR = Join-Path $inverseReference 'uv'
$env:NUMBA_CACHE_DIR = Join-Path $inverseReference 'numba'
$env:CARGO_TARGET_DIR = Join-Path $inverseReference 'target'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
uv sync --project reference --locked --python 3.13.12
uv run --project reference --no-sync python -I reference/generate_inverse_transform_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-inverse-windows-x86_64.json
cargo test -p umap-core --test inverse_transform_fixture -- --nocapture
cargo test -p umap-core --test inverse_transform_fixture -- --ignored --nocapture
uv run --project reference --no-sync python -I reference/generate_dense_inverse_metrics_fixture.py --output reference/fixtures/umap-learn-0.6.0-inverse-metrics-x86_64.json --check
```
