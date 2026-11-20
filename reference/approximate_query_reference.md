# Native approximate queries over a fixed training graph

The follow-up reproduces the wide zero-epsilon case and measures prepared
tree queries. [Prepared approximate query reference](prepared_approximate_query_reference.md)

The baseline is `36fe8c6`. This experiment isolates held-out queries from
projection pivots by supplying the same directed training rows to the public
NNDescent constructor and to Rust's `DenseApproximateQueryIndex`. The wheel
uses zero refinement rounds and zero search trees. In the boundary cases the
training topology comes from an independent exact SciPy cKDTree at width five,
including self. The probe checks the exported training support before querying.
Python estimator routing continues to use its existing path.

## Entry points and traversal

- A caller `RandomState` control replaces each of the two signed three-word
  draws in turn. The second draw determines the query stream. Rust advances
  that state ten times, then samples `abs(signed_word) % training_rows`. Each
  batch starts from the same state. Each row consumes up to the training
  width in attempts, capped by the requested query width. A duplicate still
  consumes an attempt. Rust normalizes missing ranks to `(-1, +inf)` for the
  ranked query handoff.
- A two-hop chain reaches the same target when the supplied edge directions
  are reversed. Rust visits the symmetric union of the links in index order.
  Equal-distance result heaps select the left child on a tie and retain the
  incumbent on an equal offer. In a separate frontier test, the lower-index
  of two equal-distance bridge points wins even when it enters the queue
  second. That first expansion tightens the cutoff and blocks the other
  bridge.
- Let `worst` be the farthest retained squared distance and `min_edge` the
  minimum squared non-self edge distance. The cutoff follows
  `worst + epsilon * (worst - min_edge)`. A candidate at the cutoff does
  not expand. In the chain with squared entry distance 100 and bridge
  distance 110.25, adding a disconnected edge of weight 50 moves the
  epsilon crossing between 0.20 and 0.21. Weight 99 moves it between 10
  and 11. The edge's disconnected position distinguishes this behavior from
  filtering the active edge by its own weight.
- The native index can compute that minimum from features or accept explicit
  squared edge distances. It uses the corrected float32 Euclidean
  accumulation for candidate priorities and takes the square root to emit
  distances. Its graph uses the unpruned union of the supplied links.

The 786 ordered control slots agree across caller seeds 0, 42, 73, and
`2^32 - 1`, duplicate entry attempts, rank-width limits, equal distances and
the weighted bridge experiments.

## Boundary holdouts

Each case has four independent Gaussian queries, a repeat of the first row,
and two exact training rows. The sweep covers widths 1, 2, 3, 5, 8, and 13
at epsilons 0, 0.03, 0.12, 0.24, 0.5, and 1. The zero-minimum column uses
the same native walk with the edge term set to zero. Shared support compares
finite neighbors within each query regardless of rank. Distance-bit counts
are over those shared neighbors.

| Rows | Features | Caller seed | Ordered slots, zero minimum | Ordered slots, measured minimum | Shared support | Shared distance bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 7 | 42 | 1,207 / 1,344 | 1,229 / 1,344 | 1,234 / 1,311 | 1,234 / 1,234 |
| 4,096 | 7 | 42 | 1,235 / 1,344 | 1,253 / 1,344 | 1,253 / 1,316 | 1,253 / 1,253 |
| 4,095 | 11 | 0 | 1,217 / 1,344 | 1,248 / 1,344 | 1,249 / 1,321 | 1,249 / 1,249 |
| 4,096 | 11 | 73 | 1,208 / 1,344 | 1,248 / 1,344 | 1,252 / 1,318 | 1,252 / 1,252 |

The 4,096-row, seven-feature flat-weight control repeats the topology with
all non-self weights set to `1e-8`. It matches 1,253 ordered slots and all
1,253 shared distance bits. The native and wheel rank-one IDs agree on
976 of 1,008 sweep results in the four measured-minimum cases. At width five
and epsilon 0.12 they both find the independently computed nearest row for
22 of 28 queries, including all eight exact training rows. The two occurrences
of the first query can return different rank lists in the same batch. The
4,095-row seven-feature case begins `[1351, 888, 675, 1327, 3545]` and repeats
that query as `[1351, 675, 2078, 1662, 546]`. A fresh call with the same batch
replays the original ranks. Singleton and reversed-batch controls also agree.

For each shared graph, UMAP `fit_transform(X)` and `transform(X.copy())`
produce identical CSR output. The exact-input call bypasses the recorded
public `query` method. Reversing the training row order uses the query method
and returns a COO of 20,475 or 20,480 entries at 4,095 or 4,096 rows.
Rust replay of the supplied training ranks matches all 97,622 CSR coordinates
and weight bits. UMAP's held-out transform requests width five and epsilon
0.12. Native query ranks passed to `transform_ranked_neighbors_graph`
reproduce all 140 ordered COO triples across the four cases, as does direct
replay of the wheel ranks into that handoff.

### Remaining query differences

The largest gap is a wide query at epsilon zero. In the 4,096-row
seven-feature case, the first query at width 13 returns nine finite wheel
ranks. It starts `[1091, 3462, 327, 538]`. Setting epsilon to `1e-300`
returns 13 finite ranks starting `[1351, 888, 675, 1327]`. The native walk
keeps an infinite cutoff while the result buffer is underfilled and returns
13 ranks in both calls. All disagreements in the sweep are at epsilon zero
and widths greater than the training width. Rust matches 190 of those 588
slots and every one of the other 4,788 slots. These graphs exercise the
tree-free path. Search-tree preparation and default end-to-end estimator
parity remain open.

## Query-only timing

Windows x86-64, Intel Core Ultra 9 285K, one search thread. Release Rust
and PyNNDescent 0.7.0 (UMAP 0.6.0, Numba 0.68.1) medians are from 31 warmed
calls for a seven-row batch at width five and epsilon 0.12. They include
return-buffer allocation and exclude setup, wheel compilation and UMAP
membership construction.

| Rows | Features | Caller seed | Native | Wheel |
| ---: | ---: | ---: | ---: | ---: |
| 4,095 | 7 | 42 | 5.6 µs | 12.6 µs |
| 4,096 | 7 | 42 | 5.3 µs | 12.1 µs |
| 4,095 | 11 | 0 | 5.5 µs | 14.1 µs |
| 4,096 | 11 | 73 | 6.6 µs | 12.8 µs |

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-approximate-query'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$env:APPROXIMATE_QUERY_OBSERVATIONS = Join-Path $probe 'observations.json'
uv run --project reference --locked --no-build python -I reference/probe_approximate_query.py --output $env:APPROXIMATE_QUERY_OBSERVATIONS
cargo test --workspace --locked -j 2
cargo test --release --locked -j 2 -p umap-core --test approximate_query -- --ignored --nocapture
```
