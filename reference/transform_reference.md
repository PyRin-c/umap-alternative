# Public UMAP transform and inverse transform reference

`generate_transform_fixture.py` captures a fixed-seed dense case from the
published `umap-learn==0.6.0` wheel through `UMAP.fit`, `transform`,
`inverse_transform`, and the public fitted attributes `embedding_` and `graph_`.
The input has eight distinct three-feature training rows and three held-out
feature rows. The estimator uses Euclidean distances, four neighbors, a
two-dimensional embedding, random initialization, 45 fit epochs,
`compatibility_layout=True`, `random_state=42`, `transform_seed=24`, and one
job. The JSON `parameters` object applies to both cases with the case's
`transform_mode` added.

| Checkpoint | Output in this fixture |
| --- | --- |
| `cases.embedding.training.embedding` | Fitted float32 array of shape `(8, 2)` |
| `cases.embedding.transform.output` | Float32 array of shape `(3, 2)` for the held-out feature rows |
| `cases.embedding.inverse_transform.output` | Float32 array of shape `(3, 3)` for the three fixed queries inside the fitted embedding hull |
| `cases.embedding.inverse_transform_roundtrip.output` | Float32 array of shape `(3, 3)` for the observed forward output. The public call warned that all three query points lie outside the embedding hull and that it used nearest-vertex extrapolation. |
| `cases.graph.training.graph` | Float32 CSR training graph of shape `(8, 8)` with 30 stored entries |
| `cases.graph.transform.output` | Float32 COO graph of shape `(3, 8)` with 12 stored entries, with queries on rows and training rows on columns |

The extrapolation warning was the only warning in these checkpoints. Each
case was repeated with a fresh estimator and input arrays. On the second
embedding fit, both inverse calls used the same recorded inputs and ran before
the forward transform. The output bits, sparse index order, and warning
records matched for every repeated checkpoint. Two independent Python processes
with separate Numba caches produced byte-identical JSON on the recorded
Windows x86-64 environment. The fixture covers this dataset and settings on
that locked package stack.

For Rust comparisons, decode `bits_uint32` and `data_bits_uint32` with
`f32::from_bits`. Compare fitting against `cases.embedding.training.embedding`,
then use that captured embedding with the recorded training and query rows as
fixed inputs for downstream comparisons. The direct inverse case reads
`inputs.inverse_queries`. The round-trip inverse case takes its fixed input
from `cases.embedding.transform.output`. The graph-mode checkpoint preserves
the public COO `row`, `col`, and data arrays in their original stored order.

## Graph transform boundary probes

Additional probes used the locked 0.6.0 wheel's public `fit` and `transform`
with graph mode and the fixture settings on Windows x86-64.

| Input or state | Observed public result |
| --- | --- |
| Fitted rows as a copy, a list, a float64 array, or a Fortran-order array | The fitted `graph_` CSR object by identity. Changing a positive zero to negative zero produces COO. |
| One held-out row | A `(1, 8)` COO matrix with four stored entries. |
| Zero rows, zero features, a 1-D vector, or a 3-D array | `ValueError` from input validation. |
| Transform before fitting | `AttributeError` with the message `'UMAP' object has no attribute '_raw_data'`, even if the query itself is invalid. |
| Fit with only one training row | `ValueError` with the message `Transform unavailable when model was fit with only a single data sample.` |
| NaN, infinity, or overflow during float32 conversion | `ValueError` from input validation. |
| Two identical rows querying the training column `[0, 0, 0, 0, 0.1, 10, 20, 21, 25, 30]` at zero with five neighbors | COO with columns `[0, 1, 2, 3, 4]` and weights `[1, 1, 1, 1, 0]` stored for each row. |
| Refit with NaN in the training rows | The rejected fit leaves the previous graph and held-out transform usable. |

The Python wheel uses the existing Rust transform for dense feature input
with `local_connectivity=1.0`. The wheel requires matching feature counts.
The public small-data probe returned COO for both fewer and more query features.
The Rust core rejects nonfinite selected distances. The same public probe
with a finite `float32` maximum coordinate returned COO with four stored zeros
at column zero. Python rejects sparse feature input.
For the held-out embedding path, see the [serial transform reference](transform_embedding_reference.md).

The Rust core uses exhaustive Euclidean, Manhattan, or Chebyshev distances.
It orders tied neighbors by training index. Tie order and metric arithmetic
can differ from the reference. The Python tests check exact COO
structure and float32 bits for the committed Euclidean graph fixture and the
duplicate-query probe.

## Precomputed query distances

Python `metric="precomputed"` accepts a rectangular matrix with one column
per fitted row. It returns directed COO memberships in graph mode and
fixed-tail coordinates in embedding mode. A fitted distance input that
matches by float32 bits returns the existing fitted object. For an external
matrix, the public estimator issues a warning and checks nonnegative
distances and the training column count. A width mismatch or negative
value raises `AssertionError`. Empty or nonfinite queries fail array
validation.

The Python adapter uses NumPy's `argsort` for the reference's tie order
and sends the selected distances and columns to Rust for scale fitting,
membership weights, and coordinate optimization. Native Rust and C APIs
accept an optional list of ranked training columns. Their built-in ranking
uses ascending column indices to resolve ties. In a mixed-zero public probe at width eight,
NumPy selected columns 2, 7, 6, and 4 in that order. The two hosts used the
same published fixture values for widths 8, 17, and 65.

With a finite `disconnection_distance`, Python fitting replaces cut
entries with infinity in the normalized input matrix. For a C-contiguous
float32 input this is a write to the caller's array. The original copy then
takes the ordinary transform path, and the changed array fails the
finite-query check. Rust and C borrow their buffers. A ranked query
distance exactly at the cutoff stores zero at COO coordinate (0, 0).
An entirely cut query produces NaN embedding coordinates.

The public captures agree across Windows and Ubuntu WSL. The regressions
check all 198 weights and their COO coordinates across 11 graph cases,
and all 132 transformed coordinate bits across 13 embedding cases. The
cases include zero-epoch warnings, asymmetric distances, local connectivity,
effective neighbor truncation, cutoff boundaries, and six input errors.
Sparse precomputed inputs and new data transforms under densMAP remain
outside this path.

On Windows x86-64 with an Intel Core Ultra 9 285K, ranking and constructing
the graph for 256 queries, 4,096 fitted rows, and 15 neighbors took a
median of 13.11 ms with the compact Python-to-Rust handoff. Passing the
whole distance matrix took 32.60 ms. Both builds produced the same hash
over the COO shape, ordered indices, and all 3,840 float32 weights. The
measurement used float32 uniform distances in [0.05, 3) from NumPy
`RandomState(73)`, three warmups, and nine timings for each build. It
includes NumPy ranking, conversion, and Rust COO construction.

Reproduce the fixture on either x86-64 host using the locked reference
environment. Use an OS-specific output path for the environment metadata.

```sh
uv run --project reference --locked python -I reference/generate_precomputed_transform_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-precomputed-transform-windows-x86_64.json --check
```

## Reproduce the fixture

Reproduce the Windows record from the repository root with the locked reference
environment and caches under LocalAppData.

```powershell
$transformReferenceRoot = Join-Path $env:LOCALAPPDATA 'Temp\umap-transform-reference'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $transformReferenceRoot 'venv'
$env:UV_CACHE_DIR = Join-Path $transformReferenceRoot 'uv'
$env:NUMBA_CACHE_DIR = Join-Path $transformReferenceRoot 'numba'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
uv sync --project reference --locked --python 3.13.12
uv run --project reference --no-sync python -I reference/generate_transform_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json
```
