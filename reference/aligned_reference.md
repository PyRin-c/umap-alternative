# AlignedUMAP public reference

The fixtures call the public `fit`, `fit_transform`, and `update` methods of
`umap-learn==0.6.0`. They record `embeddings_`, the graphs exposed by
`mappers_`, and the fitted state available after validation errors. The
Windows and Ubuntu WSL runs use CPython 3.13.12 and the reference lockfile.
Each uses one numerical thread and a fresh Numba cache. Warning records
cover the estimator. Numba can also emit a type-safety notice on the first
compiled update.

## Initial alignment

An independent replay uses the public mapper graphs. It computes each
normalized Laplacian from float32 degree sums and sparse products, then
solves for the first nontrivial eigenvectors in float64. It scales each
unregistered layout to maximum absolute coordinate 10 and rounds once
to float32. The first slice uses these coordinates directly.

For each subsequent slice, invert the adjacent relation and sort by
destination row index. A repeated destination uses the last source in
relation insertion order. Form the float32 covariance `current.T @ previous`
over the matched rows, take its SVD with SciPy's `gesdd`, and apply the
orthogonal matrix `U @ Vt` to all rows of the current slice. This fit uses
zero translation and retains the scale of each unregistered slice.

The replay matched 345 coordinate bits on each platform. The cases cover
complete, identity, partial, and repeated-destination relations and a
separate three-component holdout. One epoch yields the same coordinates
as zero epochs in the base probe. Changing the mapper initialization
from random to spectral also leaves the zero-epoch alignment identical
on that input.

## First updates and incremental fit

At two epochs, changing alignment regularization from 0 to 0.3 changed
48 of 54 coordinate bits. At eight epochs, it changed all 54. Changing
the alignment window from two to one at eight epochs also changed all
54 bits. The public usage guide describes the window as the span of
neighboring slices considered and the coefficient as the strength of
the alignment objective. [AlignedUMAP usage guide](https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html)

The update probe appends a third slice to a two-slice fit. Both
existing layouts retain their exact bits at 0, 2, and 8 epochs. The appended
layout differs from the zero-epoch layout of the three-slice batch fit.
At zero epochs, linked new rows receive the coordinates of their
counterparts. For a new row that links indirectly via its fuzzy graph,
accumulate the weight times the linked neighbor's coordinates in
ascending stored-column order, rounding the product and sum to float32.
Sum the weights and divide in float64, then round the coordinates to
float32. The complete and partial initialization replays match 32 bits.
Separate Windows probes returned different coordinates across processes
for rows with neither a direct link nor a linked neighbor, at identical
seed settings and zero epochs.

The probe also covers missing relations, a relation list shorter or
longer than the number of adjacent slice pairs, and an empty set of
links. Missing relations raise `ValueError` before the mapper state is
published. An empty set of links raises `ValueError` after the mapper
graphs are published. Short and long lists raise `AssertionError`.

## Rust and Python compatibility stage

`umap_alternative.AlignedUMAP` accepts dense compatibility-mode datasets
through `fit`, `fit_transform`, and `update`.

```python
from umap_alternative import AlignedUMAP

model = AlignedUMAP(n_neighbors=4, n_epochs=8, random_state=42, compatibility_layout=True)
first_layout, second_layout = model.fit_transform([first, second], relations=[first_to_second])
model.update(third, relations=second_to_third)
third_layout = model.embeddings_[-1]
```

The initial layouts use the graph eigensolver in Python. Rust handles the
shared negative stream, edge timing, coordinate updates, relation chains,
and neighborhood weights. The isolated append probes distinguish the
float32 negative interval and cursor from the float64 positive interval.
At two epochs the maximum-weight edge receives its first positive update and
three negative draws. The aligned optimizer uses its own curve, step size,
repulsion strength, and negative rate. Varying the corresponding mapper
parameters left the positive-epoch checkpoints unchanged when their graphs
were identical.

An alignment weight is the Jaccard overlap of the two neighborhoods,
restricted to rows linked by the composed relation. The weight decays by
`exp(1 - abs(slice_offset))`. Changing stored membership values while
keeping the neighborhood support fixed left the isolated update unchanged.
The epoch shuffle uses a separate MT19937 stream seeded from the magnitude
of the first negative-stream lane. The batch negative stream starts after
the spectral starting blocks. The append stream starts directly from
`transform_seed`.

Append initialization copies linked coordinates from the preceding slice.
The last source in insertion order supplies a duplicated destination.
An unlinked row with linked neighbors receives their graph-weighted mean.
Rows without linked neighbors use uniform coordinates in `[-10, 10]`.
Single-epoch fits seed that fallback through the epoch shuffle. A zero-epoch
fit initializes a separate fallback stream for each estimator. Earlier
embedding arrays retain their identity and bits through append.

The installed Windows and Ubuntu WSL wheels matched all 345 initial
coordinate bits and the locked mapper graph bits on each platform. Appending
from the captured frozen layouts matched all new coordinate bits at 0, 2,
and 8 epochs. The independent optimizer fixtures test alternate seeds,
coordinate scales, neighborhood overlap, temporal decay, and both unlinked
initialization paths. The isolated append arithmetic matched to one float32
ULP. Partial and seeded unlinked append initialization matched exactly.

### Residuals

The positive-epoch joint fit is reproducible and responds to the window and
regularization. Its coordinate trajectory differs from the public wheel.
In the three-slice reference with regularization 0.01, the installed
wheels matched 6 of 54 coordinates at 2 epochs and 0 of 54 at 8 epochs.
Maximum absolute-coordinate differences were 3.1272917 and 5.0714192 on
both platforms. The isolated probes identify the shared seed and advance
counts, while the discrepancy in the joint updates remains unresolved.
A positive-epoch append inherits any differences already present in its
frozen batch layouts.

The public wheel returns a Numba typed list. This implementation uses a
Python list of float32 NumPy arrays. Its graph-mode mappers expose the fuzzy
CSR graphs. Connected dense compatibility layouts have coordinate fixtures.
Modern layouts, additional nearest-neighbor backends, and mapper-specific
embedding behavior are outside this stage.

## Reproduction

```sh
uv sync --project reference --locked --python 3.13.12
uv run --project reference --locked python -I reference/generate_aligned_fixture.py --check
uv run --project reference --locked python -I reference/generate_aligned_optimizer_fixture.py --check
```

The following commands run the estimator assertions and print the batch
residuals for the installed local wheel.

```sh
python -I -m pytest -q tests/test_aligned.py
python -I reference/compare_aligned_wheel.py
```
