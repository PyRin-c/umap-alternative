# Public UMAP update reference

`generate_update_fixture.py` records `UMAP.fit` and `UMAP.update` from the
published `umap-learn==0.6.0` wheel. It uses the public `graph_` and
`embedding_` attributes. Each case records inputs, parameter overrides, call
results, warning categories and messages, and the public state after each call.
All update calls use the default validation options.

The fixture stores IEEE-754 words as `bits_uint32` for float32 arrays and
`bits_uint64` for the singleton float64 embedding. A CSR graph preserves the
published `indptr`, `indices`, and `data_bits_uint32` arrays in storage order.
A null attribute denotes absence. All 34 cases were repeated with fresh inputs
and a fresh estimator. Separate Python processes with separate Numba caches
produced byte-identical fixture files.

## Update and graph behavior

| Cases | Public result |
| --- | --- |
| Dense and CSR success | `None`. The graph and embedding grow from eight rows to nine or ten. The checkpoints cover a single row, close and far pairs, a duplicate row, one feature, and successive calls. |
| Update compared with fresh combined fit | With the original requested neighbor count available at fit, the compared graphs match in shape, CSR order, and every weight bit. The paired and sequential examples end with the same graph and different embedding coordinates. |
| Requested neighbors equal to or above the eight training rows | Fit warns and caps the count at seven. Update graphs match fresh fits of the combined data with `n_neighbors=7`. Fresh fits using the original requests of eight or twelve produce different graphs. |
| `n_neighbors=1` | Fit raises `ValueError`. |
| One or two training rows | The one-row fit returns a float64 zero embedding of shape `(1, 2)` with no graph. Its update raises `AttributeError` and preserves that state. For two rows, fit warns about the neighbor cap and disconnected vertices, then raises `ValueError` with the empty `(2, 2)` graph already published. |
| Empty, wrong-width, zero-feature, one-dimensional, or NaN batch | Update raises `ValueError`. Every stored graph and embedding bit is preserved. |
| Supervised or precomputed fit | Update raises `ValueError` and preserves graph and embedding. |
| Graph-only fit | Update raises `AttributeError` after publishing the expanded graph. The new graph matches a fresh combined graph-mode fit, and the embedding is absent throughout. |
| Fresh estimator | Update raises `AttributeError` and leaves both attributes absent. |

## Independent initialization model

`analyze_update_fixture.py` predicts the layout using NumPy, the fixture's input
rows, and the public embedding before each update. Graph weights enter only the
weighted-centroid comparison. These steps reproduced every coordinate in all
17 zero- and one-epoch update checkpoints.

1. Stack old and new feature rows. For each new row, sort by squared Euclidean
   distance and take the first `min(n_neighbors, original_training_rows - 1)`
   indices, including the new row itself. The cases used for the model have
   distinct input rows.
2. Retain indices from before the current update and average their existing
   float32 coordinates in distance order using a serial float32 sum. An
   empty list uses the mean of the entire old embedding, accumulated and
   divided in float64 before rounding to float32. Each appended row uses the
   same pre-update embedding.
3. Append the centroids in input order. When the resulting layout has
   coincident coordinates, take the mean Euclidean distance to each row's
   closest other row, counting zero distances. Multiply the mean by 0.001
   for the Gaussian standard deviation. Create a NumPy `RandomState` from
   `transform_seed`, advance it by three uint32 words, draw one Gaussian per
   coordinate in row order, and round the noise to float32 before adding it. The
   seed controls vary `random_state` and `transform_seed` independently.
4. Apply `10 * (x - min) / (max - min)` in float32 separately to each column.
   This can change existing coordinates even when the layout required no
   jitter.

The single added row uses old indices `[2, 4, 1]`. Its appended bits are
`[1086032232, 1082285404]`, and 15 of 16 old coordinates retain their bits.
Both rows in the closest pair select old indices `[2, 4]`. Their identical
provisional coordinates become distinct after jitter, and 4 of 16 old coordinates retain
their bits. Reversing this pair gives the same positional embedding bits
because the two provisional coordinates coincide. The pair with distinct
anchors uses `[3, 4]` and `[3, 1]` respectively, and its outputs distinguish
this model from one that feeds the first appended coordinate into the second
centroid. At `n_neighbors=2`, the closest pair selects only itself and its
batch partner, so both provisional coordinates use the old embedding mean.

The analyzer also reports bit-match counts for alternative initializations.
For the closest pair, omitting jitter or using graph-weighted centroids
matches 4 of 20 coordinates. Ranking old rows alone or counting the batch
partner as a zero coordinate matches 16 of 20. For the distinct-anchor pair,
allowing the first appended row to influence the second matches 18 of 20.

The one-epoch single and paired checkpoints match their zero-epoch
counterparts. At two epochs, the paired embedding differs from the predicted
initial layout. The model covers initialization, with the later optimizer
checkpoints available as public observations.

## Rust replay

`prepare_dense_update` rebuilds the dense fuzzy graph and prepares the
normalized initial layout using the fitted neighbor count. The native
regression replays 20 dense zero- and one-epoch checkpoints, including
successive calls, a one-dimensional fallback mean over 127, 128, and 129 old
rows, nine layout dimensions, and both limits of the unsigned transform seed.
The rebuilt CSR and coordinate bits match the public snapshot on Windows and
Ubuntu WSL. The independent Python analyzer checks the same initialization
with the captured fitted coordinates on each host.

`update_dense_embedding` carries the initialization RNG into the
[serial optimizer](update_layout_reference.md) and returns the combined CSR
and new coordinates. Python `UMAP.update(X, ensure_all_finite=True)` uses that
path for unsupervised dense models fitted with computed neighbors and
Euclidean, Manhattan, Chebyshev, cosine, or correlation distance.
A successful call returns `None`, retains the
effective fitted neighbor count, and updates the exact-input transform
shortcut to include all appended rows. A graph-only fit publishes the
expanded graph and preserves the previous exact-input shortcut when the
missing embedding raises `AttributeError`.

In C, `umap_capi_update_dense_embedding` borrows a fitted embedding handle
and a contiguous matrix with the appended rows at the end. It returns a
second handle with the updated coordinates and CSR graph. Both handles use
the existing embedding view and release calls. The metric-selecting
`umap_capi_update_dense_embedding_with_metric` accepts the metric tag used
for the original fit. The C11 smoke test covers zero, one, and two Euclidean
update epochs and six-epoch cosine and correlation updates.

The public update sets weights below `max(weight) / sampling_epochs` to
positive zero while preserving the CSR positions. The duplicate-row case
stores two zeros for weights below the default sampling cutoff. In a separate
case with a weight near 0.058155, the edge is retained at 10 epochs, stored
as zero at 11 and 17, and retained again at 18. This supports the use of the
ordinary 500-epoch budget for small datasets through 10 epochs and the
specified budget above 10. Fit-time curve and learning-rate settings are
retained, and update reads the current epoch count, graph settings, repulsion,
negative sample rate and transform seed.

### Metric and cutoff controls

The expanded fixture includes successive batches, update epochs 0, 6, 9
and 11, an omitted epoch count, explicit connectivity options, and both
ends of the transform seed range. The updated graph uses the effective
fitted neighbor count even when the requested count changes after fit.
It rebuilds without the original distance cutoff. Antipodal cosine and
correlation controls gain entries that were cut during fit. Held-out
transforms still use the fitted cutoff after a successful update.

If fit masked some rows with NaNs, those values propagate through column
normalization. Duplicate provisional rows require jitter. With those
duplicates and masked fitted rows, the public update exposes the rebuilt
graph, raises `ValueError("Input contains NaN.")`, and preserves the former
embedding and exact-input shortcut. The Python path reproduces this state.
C callers retain the original handle and receive a nonfinite-input status
in the duplicate case.

Manhattan training and update round each coordinate difference to float32
before summing in float64. Replaying a dense Manhattan distance matrix of
these values through the public precomputed graph eliminates eight initial
and four updated weight differences on the metric control. Native dense
cutoff fits use the same distance and ranking kernels as the corresponding
uncut fits, and retain only a row of candidates and the selected ranks.

Sparse Euclidean feature updates are covered below. Additional sparse metrics,
externally supplied neighbors, and failures during the optimizer remain open.

## Sparse Euclidean updates

Python `UMAP.update` accepts finite feature batches after an unsupervised
sparse Euclidean compatibility fit with computed neighbors. The public
fixture exercises CSR, CSC, and COO fits, dense and CSR, CSC, and COO
appends, and appends from SciPy `csr_array`. The update retains the
effective neighbor count from the first fit, including truncation to one
or to one less than the original row count. Changing `n_neighbors` after
fit leaves that effective count in use.

The native search rebuilds the graph over all combined rows, using the
fitted metric and no distance cutoff. Equal distances rank by row index,
including when a tied row does not select itself. It returns only the
rebuilt CSR graph and the appended rows' neighbor columns. The shared
serial update prepares the appended centroids, normalizes the combined
layout, samples both old and new rows, and retains the positions of weak
memberships that become stored zeros. The fitted cutoff still applies to
subsequent held-out transforms.

Float32 CSR batches may store repeated columns and signed zeros. The search
coalesces repeated values in float64, then rounds the sum of squared
differences to float32 before the square root. On the `base_coo` case's
second append, summing repeated values in float32 changes 12 CSR weights.
The same precision rule also applies to ranked sparse Euclidean fits.
Input storage, previous graph objects, and previous embedding arrays
keep their original contents. Sparse features remain sparse through the
update, including a test with 200,001 columns.

An empty batch, the wrong feature width, a disallowed nonfinite batch, and
an invalid transform seed fail before graph publication. A failed two-row
fit can already expose an empty graph. Updating that model or a graph-only
model publishes the next graph before raising `AttributeError` for the
missing embedding. A failed layout caused by invalid epochs or duplicate
provisional coordinates with NaNs retains the prior embedding and
exact-input shortcut while publishing the expanded graph. A successful
append replaces both the graph and the embedding. A combined CSR snapshot
takes the fitted-input shortcut in the matching cases. The mixed dense
append exercised the held-out path for the reconstructed combined CSR.

The two pinned hosts generated the same fixture. Its 53 fit cases and
70 update calls include 55 successful updates. The comparisons cover
2,406 fitted CSR weights, 3,810 weights across the update checkpoints,
974 fitted coordinates, and 1,078 retained and 208 appended coordinates
after successful calls. Failure checkpoints check 198 retained coordinates.
The fixture fixes `a` and `b` and exercises omitted, zero, one, and positive
epoch counts, epoch lists, repeated calls, and both unsigned transform seed
endpoints.
[Generator](generate_sparse_update_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-sparse-update-x86_64.json)

```sh
uv run --project reference --locked python reference/generate_sparse_update_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-sparse-update-x86_64.json --check
```

Two residuals are reproducible from that fixture. In `nan_allowed`, a CSR
batch containing NaN with `ensure_all_finite=False` expands the public graph
from 9 to 10 rows, stores NaN weights, and returns a finite embedding. The
next finite append raises `ValueError`. The port raises `ValueError` before
publishing any change on the NaN call. In `cutoff_tie`, `negative_epochs`,
and `noninteger_epochs`, the exact old-input shortcut agrees after the failed
updates. For the newly expanded held-out inputs, the port detects the stale
layout size and raises `ValueError`. The public wheel returns a 10-by-2
result in `cutoff_tie` and raises `IndexError` in the four invalid-epoch probes.

## Finite-cutoff fit timing

The Windows benchmark uses dense Euclidean features from NumPy
`default_rng(20261121).standard_normal((n_rows, 8)).astype(np.float32)`,
15 neighbors, a distance cutoff of 2.75, and one thread. Each measurement
is the median of seven runs after warming both paths. The former path used
scikit-learn pairwise distances and NumPy's stable row sort before handing
the selected ranks to Rust. The native path includes the Python-to-Rust
conversion, distance computation, cutoff counting, and CSR construction.

| Rows | Former graph build | Native graph build | Former differing weights | Native differing weights |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 44.17 ms | 7.18 ms | 6,940 / 19,860 | 0 / 19,860 |
| 2,048 | 183.29 ms | 24.62 ms | 13,960 / 40,734 | 0 / 40,734 |

Both paths match the CSR structure of the pinned public wheel for these
inputs. The former distance and index matrices occupy 48 MiB combined at
2,048 rows. The native search retains the selected ranks and one row of
candidates at a time.

## Reproduction

Run from the repository root. The reference lock specifies CPython 3.13.12
and all package versions. The generator checks the Python version and the
seven main numerical package pins and sets the thread limits before imports.

```powershell
$updateReferenceRoot = Join-Path $env:LOCALAPPDATA 'Temp\umap-update-reference'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $updateReferenceRoot 'venv'
$env:UV_CACHE_DIR = Join-Path $updateReferenceRoot 'uv'
$env:NUMBA_CACHE_DIR = Join-Path $updateReferenceRoot 'numba'
uv sync --project reference --locked --python 3.13.12
$updateFixture = 'reference/fixtures/umap-learn-0.6.0-update-windows-x86_64.json'
uv run --project reference --no-sync python -I reference/generate_update_fixture.py --output $updateFixture
uv run --project reference --no-sync python -I reference/analyze_update_fixture.py $updateFixture --check
$holdouts = 'reference/fixtures/umap-learn-0.6.0-update-initialization-holdouts-x86_64.json'
uv run --project reference --no-sync python -I reference/generate_update_fixture.py --initialization-holdouts --output $holdouts
uv run --project reference --no-sync python -I reference/analyze_update_fixture.py $holdouts --check
$metrics = 'reference/fixtures/umap-learn-0.6.0-update-metrics-windows-x86_64.json'
uv run --project reference --no-sync python -I reference/generate_dense_update_metrics_fixture.py --output $metrics --check
```
