# Public transform initialization

The pinned `umap-learn==0.6.0` wheel exposes `init_transform` for ranked
weights and `init_graph_transform` for sparse memberships. Both return
float32 coordinates. The corpus records 107 calls with 459 coordinate
bits and nine exceptions. Windows and Ubuntu WSL generate identical
fixture bytes. Warnings and their order are recorded separately.

`init_transform(indices, weights, embedding)` adds the weighted fitted
rows in rank order. Repeated ranks contribute repeatedly. Weights are used
as supplied, including negative values and zeros. Each addition rounds
to float32. A wider input product is added to the current coordinate
before rounding. In-range negative indices count backward from the end
of the fitted matrix. Empty ranked rows start at zero, and a zero weight
times a fitted NaN propagates that NaN. The port bounds all fitted indices
and validates the matrix shapes before entering the Rust kernel.

`init_graph_transform(graph, embedding)` leaves empty rows as NaN. A row
with memberships exactly equal to one copies the fitted coordinates at
the first matching storage position. Standard positive canonical CSR
rows run in Rust. Weighted rows use the original weight precision to add
the first weight after a pairwise reduction of the tail, take its
reciprocal, and scale each membership. The coordinate accumulation runs
in reverse column order and uses the promoted multiplication dtype.
Float32 coordinates are rounded once after a float64 sparse product.

Other sparse inputs use the SciPy sparse product to preserve duplicate
folding, format quirks, integer weight promotion, and NumPy warnings.
For square CSC, the direct-copy and empty-row decisions use its stored
column groups, while weighted rows use the corresponding matrix rows.
The observed rectangular CSC calls raise `IndexError`. COO raises
`AttributeError`. A stored zero in a row with nonzero mass can disappear
from the sparse product. A nonempty zero-mass row produces the observed
divide-by-zero warning, whereas an empty row produces NaNs without a
warning. Direct selections can succeed with a shorter fitted matrix
when all selected indices exist. Weighted rows require its row count to
match the graph column count.

The fixtures include graph and rank widths of 7, 8, 9, 127, 128, 129,
130 and 258, float16/32/64 coordinates, signed and unsigned inputs,
duplicate and unsorted storage, signed weights, NaNs, infinities,
the smallest float32 subnormal, and empty axes.

```sh
uv run --project reference --locked python -I reference/generate_public_transform_init_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-public-transform-init-x86_64.json --check
```
