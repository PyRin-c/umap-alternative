# Public serial dense embedding transform

The Windows x86-64 probes use the locked `umap-learn==0.6.0` wheel,
`compatibility_layout=True`, one job, and separate integer fitting and
transform seeds. They use the public estimator constructor, `fit`,
`set_params`, `transform`, and the fitted `embedding_` attribute. Graph-mode
calls capture public COO coordinates for independent arithmetic replays.
Setting `n_epochs` after a zero-epoch fit isolates the transform schedule
from changes to the fitted coordinates.

## Schedule and updates

- An explicit fit epoch setting gives `fit_epochs / 3`, rounded down.
  Counts 0 through 5 return initialization, 6 through 8 give one active
  pass, and 9 adds a second active pass. With the default setting, repeated
  queries at 9,999 and 10,000 rows matched an explicit fit setting of 300
  epochs, and 10,001 rows matched 90 epochs. The batch size in this test
  changed while the fitted model stayed fixed.
- Initialize from the positive memberships before pruning. Schedule entries
  in stored COO order. Pruning uses the maximum weight across the batch and
  a floor of `maximum_weight / transform_epochs`. Zero weights never get a
  cursor. Edges below that floor cannot become due before the final epoch,
  so the output does not distinguish eager pruning from retaining those
  inactive cursors. The weak-query probe retains its initialization even
  though a stronger query in the same batch moves.
- Divide each membership by the maximum in float32, then multiply by the
  epoch count in float32. Divide the epoch count by the rounded count in
  float64 to obtain the positive interval. Eight targeted checkpoints
  distinguish this from directly computing the ratio of maximum to weight.
  The committed schedule probe places a weight just below 7/11 with 14
  transform epochs and isolates attraction by setting repulsion to zero.
- Visit zero-based epochs and apply attraction before draining that edge's
  negative cursor. Both intervals initially give the next due time. The
  negative interval is the positive interval divided by the negative rate.
  The replay uses the same cursors as CSR fitting. Start at one quarter of
  the fit learning rate. After each pass, set the next rate to this initial
  value times `1 - epoch / transform_epochs`. A sweep over rates 1, 2, 3,
  4, 5, and 8 on fixed fitted coordinates checked the timing and decay.
- Draw the three seed words from fresh MT19937 state initialized by
  `transform_seed`. Use the same per-head streams as
  [serial fitting](negative_stream_reference.md), seeded from the initial
  query coordinate in component zero. Every draw uses the nonnegative
  remainder of the signed word modulo the number of fitted rows. A query
  row and a fitted row can have the same numeric index. The replay permits
  that draw and validates it using exact-match and distinct-query rows.
- Both updates write only the query coordinates. The fit and transform
  share the float64 curve and clipped gradient formulas, with float32
  coordinate differences and float32 storage. Coincident pairs have zero
  displacement. Draws of disconnected fitted rows with NaN coordinates
  have zero displacement and still advance the sampler. Missing query
  rows retain their initial NaNs.
- The observed float32 squared-distance reduction uses fused accumulation.
  Complete blocks of 32 accumulate into 32 lanes. Group those lanes as
  four consecutive sets of eight, add the first pair of sets and the last
  pair separately, and add those results into eight lanes. Add the lower
  and upper four lanes, then reduce as `(a+c)+(b+d)`. Complete blocks of
  four in the remainder accumulate into four lanes with the preceding sum
  in lane zero. Reduce those four in the same order and accumulate the
  final one to three components serially. Independent one-update probes
  with 93 queries at each of 28 widths from 1 to 200 matched 127,038
  coordinate bits. The committed multiepoch checkpoints exercise one to
  65 components with both forces.
- A finite disconnection threshold retains neighbors strictly below the
  threshold after scale calibration over the selected distances. Both
  the observed threshold comparison and the Rust input use float32.
  The probes at distance 1 used cutoffs of 1 and the next float32 above
  1, then checked a mixed batch and a batch without any connected queries.

## Rust verification

`transform_dense_embedding` builds the existing exhaustive dense feature
graph, initializes the query coordinates, and runs the fixed-tail optimizer.
The regression fixture checks 41,040 transformed coordinate bits across
25 cases, including both default-epoch batch boundaries. Each Rust transform
is repeated against the same fitted slice. The older 45-fit-epoch public
fixture matches all 16 newly fitted coordinate bits and all 6 transformed
coordinate bits with its recorded default curve. The new seed and epoch
cases fit in Rust before transforming and match another 160 fitted bits.

Regenerate the checkpoint file in PowerShell from the repository root.

```powershell
$transformReferenceRoot = Join-Path $env:LOCALAPPDATA 'Temp/umap-transform-embedding-reference'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $transformReferenceRoot 'venv'
$env:UV_CACHE_DIR = Join-Path $transformReferenceRoot 'uv'
$env:NUMBA_CACHE_DIR = Join-Path $transformReferenceRoot 'numba'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:CARGO_TARGET_DIR = Join-Path $transformReferenceRoot 'target'
uv sync --project reference --locked --python 3.13.12
uv run --project reference --no-sync python -IB reference/generate_transform_embedding_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-transform-embedding-windows-x86_64.json
cargo test -p umap-core --locked --test transform_embedding_fixture
```

## Feature graph residuals and integration

- Equal-distance order can affect optimization even when initialization
  and membership bits match. In a five-neighbor Euclidean batch with
  training `[0, 0, 0, 0, .1, 10, 20, 21, 25, 30, 40, 44]` and queries
  `[0, 0, -.04, .11, .3, 19.1, 22.1, 4050]`, queries .11 and .3 give
  training columns `[4, 0, 2, 1, 3]` publicly and `[4, 0, 1, 2, 3]`
  with Rust's training-index tie break. All 24 initialization bits match.
  With 45 fit epochs, negative rate 5, and the three-component modular
  initialization from the initializer note, 18/24 optimized bits match
  and the maximum absolute error is 0.022767544. Replaying captured COO
  restores 24/24. The next probe should sweep training widths and
  permutations with ties inside and across the last selected rank.
- A generic five-feature Manhattan batch matches all 196 neighbor columns
  but 176/196 membership bits. Generate it like the Chebyshev holdout
  with RNG seed 8092, seven neighbors, fit seed 3, transform seed 3133,
  `a=0.81+0.07`, `b=1.12+0.03`, learning rate `0.64+0.01`,
  repulsion `0.37+0.05`, and negative rate 5. At the 1,200 fit-epoch
  setting, 101/112 initialization bits and 100/112 optimized bits match,
  with maximum transform error 0.017395973. The next probe should compare
  the public COO memberships with candidate Manhattan accumulation widths
  and compare the smoothing scale and normalization using those distances.
- The core fixture records exact, dense feature search with fit local
  connectivity one and Euclidean layout distances. Approximate-neighbor and
  sparse transforms need their own feature graph parity and end-to-end probes.

## Python estimator

Public `set_params` probes on one fitted model distinguished the settings used
by a held-out embedding call. The estimator passes the current `n_epochs`,
`transform_seed`, `repulsion_strength`, and `negative_sample_rate` to Rust.
The fitted effective neighbor count, feature metric, output width, resolved
curve, and initial learning rate are reused. Changes to `n_neighbors`,
`metric`, `n_components`, `random_state`, `a`, `b`, `spread`, `min_dist`, or
`learning_rate` take effect at the next successful fit. `transform_seed` accepts
integers from 0 to `2**32 - 1`, defaults to 42, and is validated on held-out
embedding calls. Each call begins with fresh transform RNG state.

The estimator copies dense training input at fit and returns the fitted
`embedding_` object if the normalized query bits match all fitted input rows.
Held-out queries use `embedding_` as fixed coordinates. Transform errors and
failed refits preserve the fitted graph, embedding, feature input, and fit-time
parameters. Graph-mode calls retain the fitted CSR shortcut and held-out COO
output.

The serial transform uses exact dense feature search and Euclidean layout
distances. The estimator uses the current `local_connectivity` to construct
the directed query graph. The graph-options fixture also checks positive
epochs with training connectivity 2.0. The Python estimator uses the
default disconnection distance. The public wheel captures
`disconnection_distance` at fit, and the native binding accepts a finite
cutoff with supplied fixed tails in the threshold and disconnected-row probes.

On Windows x86-64 and Ubuntu 24.04 x86-64, fresh public 0.6.0 captures matched
the same 41,040 held-out coordinate bits. The installed wheels matched all
41,040 through the native binding. Through `UMAP.fit` and `transform`, each
wheel matched 1,300 fitted bits and 41,014 held-out bits, including the original
45-epoch case. The end-to-end comparison covers random and distinct explicit
initialization with the default cutoff. It includes 10,000- and 10,001-query
batches, and repeated transforms matched all 41,008 repeated-query bits. The
live-parameter probes matched all 60 bits for the seed endpoints, epoch and
sampling changes, and fitted geometry settings. An additional spectral case
matched 16 fitted and 10 optimized coordinate bits on both platforms. The two
wheels reproduced the equal-distance COO and Manhattan differences above with
the same per-coordinate agreement and maximum absolute errors.
