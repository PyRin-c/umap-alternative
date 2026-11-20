# Projection words and prepared query pivots

Fresh scalar, rejection, and depth-three tests are reported in the
[projection generator reference](projection_generator_reference.md).

The baseline is `dbff1b7`. The probe uses the public PyNNDescent 0.7.0
constructor, `neighbor_graph`, `prepare`, and `query` in the pinned
umap-learn 0.6.0 environment. A recording caller `RandomState` replaces
the scalar passed in the public constructor. Results use zero training
refinement rounds to expose projection leaves through connected components.

## Training pivot and leaf sequence

For a node with `n` rows in its current order, two unsigned 64-bit words
`a` and `b` produce the pivot positions

```text
first = a % n
compact = b % (n - 1)
second = compact + (compact >= first)
```

The probe recovers the unordered pivot pairs from the only compatible
Euclidean bisector for each connected split. A generalized Chinese
remainder calculation determines the first word and the compacted second
word. It requires the combined modulus for each word to span the unsigned
64-bit range. Across the five scalar values, 218 root witnesses and 517
child witnesses recover unique word pairs. The first six training words
are reproducible for these five scalars.

| Caller scalar | Root witnesses | Left child witnesses | Right child witnesses |
| ---: | ---: | ---: | ---: |
| 0 | 45 | 48 | 50 |
| 1 | 44 | 53 | 54 |
| 42 | 43 | 54 | 55 |
| 1819927849474927636 | 44 | 48 | 47 |
| 2033886794655244609 | 42 | 55 | 53 |

The parent scans from the start of its row array. A row nearer the second
pivot swaps with the last unclassified row and the replacement is examined
at the same position. Each child inherits the resulting array slice. Using
original row order within each child produces no common unsigned word pair
for any of the child streams. At 131 rows, scalar 1819927849474927636 gives
ordered training pivots `(117, 27)`. The root words for that scalar are
2911110826249513496 and 16896642894157928877.

When the left child stops at capacity, the right child still uses the third
pair. The tested parent and child slots therefore have stable positions
across capacity changes. The native `with_projection_words` method accepts
explicit pairs indexed by those positions and reuses the existing Euclidean
tree query and graph walk. It uses the left branch for exact projection ties
and keeps a leaf for a degenerate split. The caller still supplies the
words. The mapping from an arbitrary scalar to the raw word generator and
the wheel's scheduling below depth two remain open.

On 54 independent fit configurations, Rust reproduces all 76 ordered
pivot pairs and all ordered leaf arrays of the word model. It matches
39,832 of 39,878 public connected-component labels, with full agreement
in 53 cases. All eight 4,095- and 4,096-row cases agree completely. The
33-row fixture selects last row 32 as its second pivot. The 135-row
fixture selects first row 0 as its first pivot. Capacities immediately
below, at, and above a node size determine whether the node splits.

In the 131-row, seed-32 input, the root splits into 51 and 80 rows. At
capacity 51 the right pair is `(26, 23)` and all 131 labels agree. Using
the left slot's words for this right split leaves 92 of 131 labels in
agreement. Across the eight cases with a stopped left child and a split
right child, the slot model matches all 1,252 labels while shifting the
right words matches 864. The component residual is the 133-row case at
capacity 25. One leaf has five rows and the wheel exports 11 cross-leaf
offers at width seven, merging some graph components. At width four the
same pivots produce all 133 labels with zero crossing offers.

## Prepared geometry

The prepared probe uses 640 Gaussian queries at width one and epsilon zero.
At those settings the nearest entry determines the result. It finds the
unique root and up to two child bisectors compatible with these entries and
then scores nine different queries over the same public training graph.
The last two queries are the first and last training rows. The sweep uses
widths 1, 5, 8, and 13 at zero, the smallest positive float64, the smallest
positive float32, 0.12, and 0.24.

The 131-row, seven-feature calibration reproduces the earlier prepared
root `(30, 127)` when the training capacity is 130. Lowering that capacity
to 43 produces prepared root `(85, 116)` and children `(85, 127)` and
`(62, 116)`. The independent 163-row case produces prepared root
`(14, 117)` and children `(14, 30)` and `(98, 117)`. Tests encode these
prepared geometries separately from the recovered training word stream.
The prepared pivots therefore cannot be inferred by simply using the
training tree as the query tree. The public estimator keeps its current
routing while prepared-query parity remains open.

For scalar 2033886794655244609 with 131 rows and capacity 130, the
training word pair selects `(15, 9)` in both dimensions. Prepared queries
recover `(18, 30)` at seven features and `(9, 36)` at eleven. The scalar
and row count alone therefore leave the prepared geometry underdetermined.

## Ordered ranks and warm latency

All 63 independent rank-one entries at epsilon zero and all 8,505
graph-only rank and distance slots agree. With the recovered prepared
geometries, overall ordered rank agreement rises from 5,739 to 8,080 of
8,505. Using the recovered training words for query preparation agrees on
3,732 slots. All 8,391 shared finite distances match bit for bit. The
remaining support differences contain 114 wheel-only and 114 Rust-only
slots. At 131 rows and capacity 43, the first independent query at width
five and epsilon 0.24 has wheel-only IDs `[88, 29]` and Rust-only IDs
`[46, 130]`. At eleven features, capacity 43 and epsilon zero, query row
three at width five has wheel-only ID `34` and Rust-only ID `106`.

The 42 and 73 labels below refer to scalars originally generated from
those caller seeds. Each timing is a release median of 41 warm queries
for nine rows at width five and epsilon 0.12 on Windows x86-64, Intel Core
Ultra 9 285K. Times exclude fitting, preparation, and JIT compilation.

| Rows | Features | Scalar source | Training cap | Ordered agreement | Prepared Rust | Prepared wheel | Graph-only Rust | Graph-only wheel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131 | 7 | 42 | 130 | 1,215 / 1,215 | 7.6 µs | 19.4 µs | 2.8 µs | 9.4 µs |
| 131 | 7 | 42 | 43 | 1,116 / 1,215 | 4.2 µs | 12.9 µs | 2.8 µs | 9.6 µs |
| 163 | 7 | 42 | 54 | 1,215 / 1,215 | 5.0 µs | 14.5 µs | 2.5 µs | 9.1 µs |
| 131 | 11 | 73 | 43 | 1,136 / 1,215 | 5.0 µs | 17.4 µs | 3.2 µs | 12.2 µs |
| 131 | 7 | 73 | 130 | 1,177 / 1,215 | 6.1 µs | 16.9 µs | 2.9 µs | 9.7 µs |
| 131 | 11 | 73 | 130 | 1,139 / 1,215 | 6.7 µs | 17.8 µs | 3.2 µs | 9.4 µs |
| 197 | 11 | literal 1 | 65 | 1,082 / 1,215 | 5.6 µs | 18.4 µs | 3.3 µs | 12.5 µs |

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-projection-words'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$env:PROJECTION_WORD_OBSERVATIONS = Join-Path $probe 'projection-words.json'
uv run --project reference --locked --no-build python -I reference/probe_projection_words.py --output $env:PROJECTION_WORD_OBSERVATIONS
cargo test --release --workspace --locked -j 2
cargo test --release --locked -j 2 -p umap-core --lib compare_projection_words_with_wheel -- --ignored --nocapture
```
