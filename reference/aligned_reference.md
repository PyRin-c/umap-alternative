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
seed settings. The fallback RNG and positive-epoch optimizer remain open.

The probe also covers missing relations, a relation list shorter or
longer than the number of adjacent slice pairs, and an empty set of
links. Missing relations raise `ValueError` before the mapper state is
published. An empty set of links raises `ValueError` after the mapper
graphs are published. Short and long lists raise `AssertionError`.

## Reproduction

```sh
uv sync --project reference --locked --python 3.13.12
uv run --project reference --locked python -I reference/generate_aligned_fixture.py --check
```
