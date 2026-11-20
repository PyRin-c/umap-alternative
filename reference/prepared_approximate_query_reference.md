# Prepared approximate Euclidean queries

The baseline is `a18f16b`. The probe uses PyNNDescent 0.7.0 from the
locked UMAP 0.6.0 environment and calls only the public constructor,
`neighbor_graph`, `prepare`, and `query`. It holds the training topology
fixed by constructing a tree-enabled wheel index, exporting its public
neighbor graph, then creating a second wheel index and the Rust indexes
from that graph. The probe checks that the two wheel training exports agree.

`init_graph` suppresses tree construction in the wheel. Adding `tree_init=True`,
`n_trees=8`, and `n_search_trees=8` on a supplied graph preserves the two
caller state draws and every query rank in the control. On the six
split-forest boundary cases, `n_search_trees=0`, `1`, `2`, and the full
construction count return equal ranks and bits in all 35 batches per case.
The graph-only constructor is the control for removing the prepared tree.

## Cutoff and underfilled results

The cutoff is rounded to float32 and frontier priorities must beat it
strictly. In the weighted bridge control, the squared entry and bridge
priorities are 100 and 121 and the minimum edge weight is 16. Epsilon 0.25
leaves the bridge exactly at the cutoff. A float64 step above that cutoff
and the midpoint to the next float32 value still leave rank one at the
entry. At the next float32 value the walk reaches the target.

At epsilon zero with an underfilled result, the wheel removes already seen
nodes from an expanded row before grouping fresh neighbors. It offers the
first two in each complete group of four, then the entire final group of
one to three. When the result fills, subsequent offers use the ordinary
strict cutoff. If the row offers candidates and leaves vacancies, the walk
stops after that row. The degree-one through degree-fourteen star controls
and controls with already seen edges distinguish these cases. The same
walk with even the smallest positive float64 epsilon continues to fill
vacancies when no tree is present.

Prepared queries have a separate zero boundary. At width 131 in the
one-split control, epsilon zero and one quarter of the smallest positive
float32 produce 679 finite ranks. At three quarters of that float32 value
and at the value itself, all 1,179 ranks are finite. The Rust prepared
path uses the zero walk when epsilon rounds to zero at float32 precision.

The tree-free Rust replay now matches all 5,376 ordered slots and 5,266
finite distance bits across the four original boundary cases. The 588
wide zero-epsilon slots rise from 190 to 588 matches. The new graph-only
cases below also match at all tested epsilons, including the smallest
positive float64 value.

## Prepared geometry

At width one and epsilon zero the first result is a direct entry witness
because traversal cannot improve it. For 131 rows, seven features and
caller seed 42, the probe evaluates all 8,515 unordered Euclidean bisectors
against 256 independent queries. The one-tree preparation yields a unique
matching pair of rows 30 and 127. The training projection on that input
uses rows 27 and 117. Using eight construction trees yields a different
prepared query pair, rows 18 and 30. Duplicating the batch or replacing
either of the signed caller draws preserves the one-tree results. These
observations leave the mapping from the caller's forest scalar to the
prepared pivots and the prepared tree selection rule open.

`DenseApproximateQueryIndex::with_projection_tree` uses the existing Rust
Euclidean projection builder with a caller-selected seed and leaf cap. It
scores the selected leaf as entry points, then uses the squared-distance
walk over the supplied graph. It builds its own tree from features. The
one-split calibration chooses a native seed that reproduces the observed
root geometry, then checks different queries from the ones used to recover
the pair. The boundary comparisons use the caller seed directly and retain
the native projection builder's pivot order. The estimator continues to
use its existing routing while prepared search parity is open.

On the nine independent one-split checks, Rust matches every rank-one
entry at epsilon zero and all 45 slots at width five and epsilon 0.12.
Across the complete sweep it matches 9,562 of 10,332 ordered slots, up
from 6,180 with no tree. All 8,809 shared finite distances match bit for
bit. Even with the calibrated split, the first Gaussian holdout from
seed 1701 at width five and epsilon zero returns `[68, 15, 125, 8, 35]`
in the wheel and `[15, 125, 8, 35, 74]` in Rust. Prepared graph traversal
can still differ after the same leaf has been selected.

## Boundary rank checks

The seven-feature cases reuse the training and query arrays from the
previous probe. The eleven-feature cases use independently seeded arrays
and queries. Each batch contains four Gaussian rows, one repeated query,
and two exact training rows. The sweep uses widths 1, 5, 6, 8 and 13, and
epsilons 0, the smallest positive float64, `1e-300`, 0.03, 0.12, 0.24,
and 0.5. The no-tree replay matches all 29,438 slots in the new capture,
including the controls.

The public one-leaf builds reproduce the exact cKDTree training graph at
both row counts. Rust matches all 6,468 ordered prepared slots and their
distance bits across the four one-leaf cases. The two seven-feature
sixteen-tree builds also reproduce that same cKDTree graph exactly. The
eight-tree builds share their exported training graph across prepared,
graph-only, and Rust runs.

| Rows | Features | Caller seed | Wheel forest | No-tree agreement | Native prepared agreement | Shared distance bits | Prepared at width 5, epsilon 0.12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 7 | 42 | 8 trees | 996 / 1,617 | 1,102 / 1,617 | 1,371 / 1,371 | 22 / 35 |
| 4,095 | 7 | 42 | 16 trees | 895 / 1,617 | 1,058 / 1,617 | 1,414 / 1,414 | 30 / 35 |
| 4,096 | 7 | 42 | 8 trees | 951 / 1,617 | 1,149 / 1,617 | 1,390 / 1,390 | 33 / 35 |
| 4,096 | 7 | 42 | 16 trees | 899 / 1,617 | 1,176 / 1,617 | 1,426 / 1,426 | 26 / 35 |
| 4,095 | 11 | 19 | 8 trees | 567 / 1,617 | 636 / 1,617 | 1,062 / 1,062 | 10 / 35 |
| 4,096 | 11 | 73 | 8 trees | 791 / 1,617 | 1,010 / 1,617 | 1,318 / 1,318 | 15 / 35 |

## Warm query latency

Windows x86-64, Intel Core Ultra 9 285K, one thread. Each number is the
median of 41 warm calls for seven rows with width five and epsilon 0.12.
Only the query call is timed. The measurements exclude training, graph
construction, tree preparation and JIT compilation.

| Rows | Features | Wheel forest | Native graph-only | Wheel graph-only | Native prepared | Wheel prepared |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 7 | 8 trees | 4.8 µs | 15.0 µs | 4.6 µs | 15.6 µs |
| 4,095 | 7 | 16 trees | 4.9 µs | 11.8 µs | 20.0 µs | 45.1 µs |
| 4,096 | 7 | 8 trees | 5.1 µs | 12.2 µs | 5.3 µs | 12.4 µs |
| 4,096 | 7 | 16 trees | 5.0 µs | 12.6 µs | 22.0 µs | 40.8 µs |
| 4,095 | 11 | 8 trees | 6.0 µs | 16.0 µs | 6.5 µs | 17.6 µs |
| 4,096 | 11 | 8 trees | 5.2 µs | 14.3 µs | 6.1 µs | 17.6 µs |

The one-leaf control scores every training row. It takes 130.8–131.9 µs
in Rust and 614.6–752.0 µs in the wheel across the four boundary cases.

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-prepared-query'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$env:PREPARED_QUERY_OBSERVATIONS = Join-Path $probe 'prepared.json'
$env:APPROXIMATE_QUERY_OBSERVATIONS = Join-Path $probe 'tree-free.json'
uv run --project reference --locked --no-build python -I reference/probe_prepared_approximate_query.py --output $env:PREPARED_QUERY_OBSERVATIONS
uv run --project reference --locked --no-build python -I reference/probe_approximate_query.py --output $env:APPROXIMATE_QUERY_OBSERVATIONS
cargo test --release --locked -j 2 -p umap-core --test approximate_query -- --ignored --nocapture
cargo test --release --workspace --locked -j 2
```
