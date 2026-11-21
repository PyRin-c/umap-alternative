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

The overlap calculation uses the two neighborhoods restricted to rows
linked by the composed relation. Each mapped source neighbor contributes
to the intersection count when its destination appears in the other
neighborhood, including repeated destinations. Two empty restricted
neighborhoods have overlap 1. The weight decays by
`exp(1 - abs(slice_offset))` and is stored in float32. Changing stored
membership values with fixed neighborhood support left the isolated
update unchanged.

At each epoch, the optimizer shuffles the original slice order with a
separate MT19937 stream seeded from the magnitude of the current first
negative-stream lane. Each stored edge position is visited in that slice
order before advancing to the next position. Edges whose timers are due
advance the shared negative stream. The batch stream starts after the
spectral starting blocks. The append stream starts directly from
`transform_seed`.

The tail update in a joint fit reuses the attraction coefficient and
reads the head coordinate after its update. Relation terms also read
live coordinates, starting at the first due edge. The update clips the
base gradient and clips the sum again after adding the relation terms.
For a slice at index `s` with window `w`, the optimizer considers
earlier slices starting at `max(0, s - w)` and later slices ending
before `min(s + w, n_slices - 1)`. Window zero therefore has no
alignment term and window one uses the preceding slice.

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

### Reconstructed batch checkpoints

On both Windows and Ubuntu WSL, the three-slice fit at regularization
0.01 matches all 54 coordinate bits at each of 2 and 8 epochs. The
previous maxima were 3.1272917 and 5.0714192, with 6 and 0 exact
coordinates, respectively. The unweighted fits, windows zero and one,
and the partial and repeated-destination batch cases match 54/54 bits
per checkpoint.

The first due edge and the eight-epoch fit distinguish the following
alternatives in the independent replay. The reconstructed sequence
matches every bit of these checkpoints.

| Replay change | Checkpoint | Exact bits | Max absolute difference |
| --- | --- | ---: | ---: |
| Finish each slice before the next | 2 epochs, weight 0 | 6/54 | 4.0765376 |
| Interleave by due-edge rank | 2 epochs, weight 0 | 6/54 | 4.0078206 |
| Negate the head increment for the tail | 2 epochs, weight 0 | 6/54 | 4.2444468 |
| Sort the deliberately reversed CSR columns | 2 epochs, 5/6/7 rows, weight 0 | 24/36 | 0.0159810 |
| Keep one fixed permutation | 8 epochs, weight 0 | 0/54 | 3.5987597 |
| Advance a persistent shuffle RNG | 8 epochs, weight 0 | 0/54 | 3.1605763 |
| Shuffle the previous epoch's order | 8 epochs, weight 0 | 0/54 | 2.0574889 |
| Read epoch-start relation coordinates | 2 epochs, weight 0.01 | 6/54 | 0.0768521 |
| Delay alignment past the first due edges | 2 epochs, weight 0.01 | 6/54 | 0.2739663 |

The storage-order probe replaces the public mapper graphs through
the public `UMAP.fit` method. It keeps the sparse support and
memberships fixed while reversing the stored columns of each row.
The other alternatives use public fits and layout checkpoints with
the original graphs.

### Remaining differences

At regularization 0.3 the base fit matches 53/54 bits at 2 epochs,
with a maximum difference of one float32 ULP (2.3841858e-7), and
54/54 at 8 epochs. Selected alternate seeds and identity relations
also retain small arithmetic differences. These differences persist
in a replay using fused multiply-add for the final coordinate update.

### Relation extent at window two

The six-row fixture leaves row 5 beyond the largest relation index.
The public optimizer receives relation and weight arrays of shape
`(3, 5, 5)`. With the row dimension extended to six and filled with
`-1` and zero, it produces a different layout. Copying the values
at the original flattened offsets into an explicitly sized table
reproduces the unmodified public output. In particular, row 5 can
read row 0 of the next time offset. Four-slice controls confirm that
the relation table includes the forward boundary slot even though
the optimizer excludes that boundary from its update sweep.

Rust resolves these offsets with checked indices. It uses the weight
from the referenced slot and the temporal decay of the update's slice.
An offset beyond the available table or a row outside the destination
slice contributes no alignment force. The new tests compare with the
unmodified public fit. The padded diagnostic remains in the fixture.

| Time window two probe | Matching coordinates | Maximum absolute difference |
| --- | ---: | ---: |
| One trailing row, two epochs | 36 / 36 | 0 |
| One trailing row, eight epochs | 33 / 36 | `3.58e-7` |
| Two trailing rows, two epochs | 36 / 36 | 0 |
| Two trailing rows, eight epochs | 36 / 36 | 0 |
| Two trailing rows across four slices, eight epochs | 48 / 48 | 0 |

The previous maximum for the one-row, eight-epoch fixture was
`0.5040512`. The difference is within six float32 steps. The
window-one and linked-last-row controls still match exactly.

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
