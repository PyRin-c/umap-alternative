# Distinct-row UMAP fitting

The public `umap-learn==0.6.0` wheel with `unique=True` fits the graph
and the optimizer on distinct rows. Dense rows are sorted lexicographically.
Sparse rows use a key made from their stored columns followed by their stored
values. In both cases the first source row represents each group, including
its supervised label. The Rust CSR grouping keeps explicit zeros in the key
and treats either zero sign as equal. The Python `csr_unique` helper
coalesces repeated columns on a copy before grouping.

The fitted graph has one row and column per group. The fitted coordinates,
epoch snapshots, and density radii use the original row order, with equal
coordinates for rows in the same group. The effective neighbor count uses
the number of distinct rows. An input with over one row collapsing into a
single group produces a single zero coordinate and no fitted graph. Explicit
initial coordinates are indexed by group. Precomputed distances and supplied
neighbor distances are rejected by the public estimator in unique mode.

Held-out transforms and dense updates use the original training rows. A
held-out COO graph has one column per original row. The inverse optimizer uses
the scales in distinct-row order for the prefix of the original rows. Two
controlled 48-epoch inverse probes verify this indexing, with and without a
repeated original row. The port uses a row's representative scale for
original indices beyond that prefix. Exact parity for positive inverse
optimization that uses those extra indices remains open.

For the duplicate inverse case in the fixture, replace its queries with
`[[8.81, 8.73], [8.55, 8.51]]` to exercise the appended row in the supplied
layout. Two independent Windows reference processes at 48 epochs disagreed
on all six coordinate bits for these queries even with fixed seeds.

The fixture records distinct-row graph bits, final and checkpoint coordinates,
training and held-out transforms, the initialized inverse, and dense and
CSR updates. It covers dense, CSR, CSC and COO inputs, categorical and continuous
targets, the compatibility optimizer and Adam, and truncated neighbor
counts. Explicit curve coefficients are used in the common fixture since
inferred coefficients can differ between Windows and Ubuntu WSL.

On Windows x86-64, a 64,000-row CSR matrix with 512 columns and six stored
entries per row produced identical indices, inverse maps and counts.
The median of seven warmed calls was 217.9 ms for the public wheel helper
and 31.4 ms for the Rust-backed helper, including Python argument
conversion. The input used MT19937 seed 33, 32,000 randomly seeded rows
and a shuffled second copy of all rows. For each row, sorted columns were
six strides of 71 from a random start, reduced modulo 512. Values were
uniform integers from -20 through 20, cast to float32.

## Reproduction

Use the locked reference environment with caches and build products on
LocalAppData on Windows or under the Linux user cache on WSL.

```text
uv run --project reference --locked python -I reference/generate_unique_fixture.py --output reference/fixtures/umap-learn-0.6.0-unique-x86_64.json --check
python -I -m pytest -q tests/test_unique.py
```
