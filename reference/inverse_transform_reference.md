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

The Rust module covers dense Euclidean data with externally supplied geometry.
Open work includes Python estimator wiring, the arithmetic of other input and
output metrics, high-dimensional exterior reductions, and positive updates
after every membership for a query rounds to zero. The Rust implementation
keeps candidate discovery order on equal distances. The larger circle and
duplicate-grid probes exercise the different equal-key order of the wheel's
sorting path.

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
```
