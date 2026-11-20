# Supplied nearest-neighbor arrays

The Windows x86-64 fixture records `umap-learn==0.6.0` graph fits for
eight one-dimensional float32 rows with `n_neighbors=4`, plus graph
transforms of three held-out rows. It calls only public `fit_transform`
and `transform` methods. The supplied indices are int32, distances are
float32, and the search index in the `precomputed_knn` tuple is `None`.

The fixture stores the supplied input arrays, the fitted CSR and held-out
COO results as float32 bit patterns, and warning or exception details.

| Input | Observed behavior |
| --- | --- |
| Four columns | The unmodified arrays reproduce the computed 28-entry graph. Changing the first row's distances changes weights. |
| Three columns or seven rows | A warning reports that the supplied arrays are ignored. Both fitted and held-out graphs match the computed baseline. |
| Five columns | The fifth column contributes. The fitted graph has 38 entries. |
| Different shapes for indices and distances | The fit raises `ValueError`. |
| `-1` index paired with `+inf` distance | Missing arcs are accepted. Other rows can still supply reciprocal edges to a row with no non-self neighbor. |
| Self in the second column, no self, repeated index, or unsorted distances | The fit succeeds in each recorded case. |
| Threshold at 1.0 or the preceding float32 value | Both fitted graphs have six entries. |
| Threshold at the following float32 value | The fitted graph has ten entries. |
| Threshold at zero | The graph has no stored entries. |

All thirteen successful cases that used the supplied neighbors raised
`NotImplementedError` on held-out transform. In the four cutoff cases
the fit changed both arrays supplied by the caller, replacing cut indices
with `-1` and their distances with `+inf`. Windows and Ubuntu WSL
reproduced the same outputs and warnings for all 17 cases. The scope is
small dense graph fits with default local connectivity.

## Python estimator

Pass `precomputed_knn=(indices, distances, None)` for an unsupervised graph
or serial seeded dense embedding. A two-item tuple also works. The width
check uses the requested `n_neighbors` before dataset-size truncation. A
short width takes precedence over a row-count mismatch. Fallback bypasses
both caller-array writes and the supplied search index. The fixtures include
a fallback fit at cutoff `1.0`, including the held-out COO coordinates and
stored zeros at `(0, 0)` for cut entries.

The estimator evaluates `distances >= disconnection_distance` in the
array's dtype and assigns `-1` to those indices before writing `+inf` to
the distances. It attempts both assignments even for an empty mask. A
read-only index array prevents either write. A read-only distance array
can leave the indices changed while the previously fitted output remains
available for exact input. After an accepted tuple reaches the masking
stage, a write error still leaves held-out search subject to the supplied
search-index check. Shape validation and cutoff type validation precede
both writes. A shape error during refit preserves the previous graph and
exact-input match while held-out search sees the new, validated features.
A singleton validates the arrays and returns before assigning either one.
Native graph validation applies to the remaining index-distance pairs
after masking. These regression cases use in-range integral indices with
finite, non-negative distances or the paired `(-1, +inf)` marker.

The exact normalized input returns the stored `graph_` or `embedding_`
by identity. With `None` as the search index, held-out input is validated
before the public `NotImplementedError` is raised. A public-wheel
probe with a PyNNDescent `NNDescent` object called its public `query`
method for held-out features. External search adapters are outside the
verified estimator scope and raise `NotImplementedError` for held-out
input. Approximate or large-data paths also need separate compatibility
checks. An empty supplied graph is observable through `graph_` even when
layout fitting raises. A previous `embedding_` stays available.

The Python fixtures record seven layouts with random, explicit and
spectral initialization. They include connected, disconnected and isolated
rows. The installed Windows and Ubuntu 24.04 wheels reproduced all graph
bits and matched the coordinates of the respective public wheel. The
25-epoch explicit initialization differs across platforms by up to
2.58837 in a coordinate, so each platform has its own layout fixture. An
embedding uses the existing Rust layout over the supplied graph after
epoch-dependent pruning. Large duplicate weights can still exceed the
native layout's unit-interval limit.

## Rust graph stage

`umap_core::fit::fit_supplied_neighbors_graph` accepts immutable row-major
int32 indices and float32 distances, the row count, the actual column
count, the requested `n_neighbors`, `GraphOptions`, and a float32
disconnection distance. It builds CSR directly from the supplied rows.
`fit_precomputed_graph` ranks rows of a square distance matrix.

- The smooth-scale target is `log2(n_neighbors)`. All supplied columns
  contribute, even when the input width exceeds the requested count or
  the row count. In the public case with eight rows and nine requested
  neighbors, the accepted nine-column arrays use a target of `log2(9)`.
- Local connectivity counts positive finite distances in supplied order.
  The offset saturates at the maximum finite distance if the
  connectivity exceeds the number of positive finite distances.
  The scale search skips column zero. A non-self index in that column
  still emits a membership, while a self index anywhere contributes its
  distance to the scale calculation according to its column.
- A `(-1, +inf)` pair contributes neither an edge nor a distance sample.
  The minimum scale uses the mean of finite distances in the row when
  the local offset is positive and across all rows when it is zero.
  Missing neighbors leave the solve target unchanged.
- The cutoff removes every distance greater than or equal to the
  supplied float32 threshold before graph construction. Use `+inf`
  with already-masked arrays or when no finite cutoff applies. Results
  obtained from caller-preserved input arrays and from the public fit's
  post-cutoff arrays agree exactly in the recorded cases.
- Repeated directed coordinates are summed in input column order before
  the fuzzy set operation. A duplicate sum can exceed one and reciprocal
  duplicates can yield negative or zero union weights. CSR omits the
  diagonal and exact zeros. The native layout validates weights against
  the unit interval.

Callers own estimator validation and fallback, the choice and conversion
of the cutoff, caller-visible array mutation, and search-index adapters.
The Rust API rejects malformed shapes, out-of-range indices, and
index-distance pairs other than a valid finite distance or the paired
missing marker. The public fit cases with an undersized width or wrong
row count fall back before this stage.

The Rust tests compare CSR shape, offsets, columns, and every float32
weight bit for all 13 accepted cases above and 14 additional public-wheel
fits. The latter include separate requested counts on the same width,
width above the row count, adjacent float32 distances in one row, missing
distances in the row and batch scale floors, fractional connectivity,
and repeated coordinates with union, intersection, and mixed weights.
The 14 held-out cases reproduced across Windows and Ubuntu WSL.

## Wide duplicate rows

A Windows stress sweep of 24 public fits matched CSR support throughout.
Two width-71 inputs with repeated indices produced one asymmetric pair
each in the public graph. Rust returned the same weight in both
directions, matching the public reverse weight. The forward weights
differed by two and one float32 ULPs, respectively.

| Sweep iteration | Requested count | Local connectivity | Mix | Forward coordinate | Public forward bits | Rust bits in both directions |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 3 | 0.65 | 1 | `(2, 9)` | `0xc04009d6` | `0xc04009d4` |
| 11 | 5 | 5 | 0 | `(1, 3)` | `0x3edf82a2` | `0x3edf82a3` |

To reproduce the two inputs, use the generator's base fit settings and
`np.arange(12, dtype=np.float32).reshape(-1, 1)` for training. Generate
the arrays with the pinned NumPy version and fit iterations 7 and 11
using the table's overrides.

```python
rng = np.random.default_rng(914)
for i in range(12):
    shape = (12, (5, 11, 41, 71)[i % 4])
    indices = rng.integers(0, 12, shape, dtype=np.int32)
    distances = rng.uniform(0, 7, shape).astype(np.float32)
    # Capture indices and distances at iterations 7 and 11.
```

Verify both fixtures with the locked reference environment.

```powershell
uv sync --project reference --locked --python 3.13.12
& reference/.venv/Scripts/python.exe -I reference/generate_precomputed_neighbors_fixture.py --check
& reference/.venv/Scripts/python.exe -I reference/generate_precomputed_neighbors_fixture.py --rust-held-out --check
& reference/.venv/Scripts/python.exe -I reference/generate_precomputed_neighbors_fixture.py --python-estimator --check
```

On Ubuntu, run the last command with the locked environment's
`reference/.venv/bin/python` interpreter. The Python regression tests in
`tests/test_supplied_neighbors.py` use the platform's layout fixture.
