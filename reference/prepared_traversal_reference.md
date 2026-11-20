# Prepared Euclidean query traversal with live training offers

The baseline is integrated main `8c2600b`. The probe uses the public
NNDescent constructor, `neighbor_graph`, `prepare()`, and `query()` from the
locked UMAP 0.6.0 environment (PyNNDescent 0.7.0, Numba 0.68.1). Training
uses one tree and zero refinement rounds. Preparation uses one search
tree, a leaf budget of one, degree multiplier 100, and zero
diversification and degree-prune aggressiveness.

## Live graph and visit order

A width-five fit exports self and four other training IDs per row. A
second fit at width six, with all other constructor inputs held fixed,
reproduces the five exported ranks and their distance bits and exposes
one more offer. Supplying the wider table as the symmetric query
adjacency accounts for the ordinary prepared-rank differences. Only
the width-five index supplies the wheel query results, so the two fits
do not conflate changes to query tree geometry with changes to offers.
The wider fit is an observation technique for these zero-round cases.
The native index accepts live offers directly and keeps the original
training width as the draw limit for a tree-free query.

The zero-epsilon queries at width `n` or `n+17` distinguish the
prepared visit order. After offering every row in the selected leaf,
the closest leaf row supplies the first graph expansion. That expansion
filters seen candidates, offers the first two of each complete group of
four, and offers every trailing candidate while the heap is underfilled.
If it is still underfilled, traversal stops even when the closest row
had no unseen links. With the smallest positive float32 epsilon, the
same wide query continues through the reachable component. The smallest
positive float64 epsilon rounds to zero in this prepared path.

The order of the remaining candidates is the left-to-right leaf order
of the prepared tree and original row order within each leaf. The
probe first fixes the geometric splits with rank-one, zero-epsilon
queries or reuses an archived geometry. It orients the unordered pivot
pairs with a separate 96-query, wide, zero-epsilon witness batch. This
distinguishes branch directions when graph links span multiple leaves.
For example, on the 179-row depth-two holdout the correct orientation
predicts all 96 support sets and the runner-up predicts 85. At depth
one, the off-leaf candidates all belong to the other leaf, so both root
directions agree on these witnesses. The timed and scored queries are
disjoint from the direction witnesses.

Thirty-six equal-frontier queries at depths one and two use exact
dyadic midpoints between two rows in the same leaf with different
off-leaf adjacency. The lower row wins an equal entry priority.
For caller zero, the midpoint of rows 0 and 8 gives their common
squared priority `2.74530029296875`. At
width one and epsilon zero, the wheel keeps row 0 even though its
neighbor 32 is closer. Epsilon 0.12 allows that expansion and makes
row 32 the winner. With an underfilled width, the off-leaf offers
identify row 0 as the expanded row. Separate one-leaf controls keep
equal-distance incumbent results and reproduce the order of the
max-heap's tied children and final sort. At width five, the all-tied
query yields `[2, 3, 0, 1, 4]`.

## Exactness and remaining differences

The exported baseline uses the same oriented geometry and traversal
with the five public ranks. Distance bits are counted in their emitted
slots, including vacancies. Support counts compare finite IDs
independently within each query.

| Sweep | Exported rank slots | Exported distance bits | Live rank and distance slots | Exported wheel-only / Rust-only | Live wheel-only / Rust-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary and equal heap | 33,387 / 34,730 | 33,463 / 34,730 | 34,730 / 34,730 | 407 / 407 | 0 / 0 |
| Wide | 322,049 / 337,425 | 322,377 / 337,425 | 337,425 / 337,425 | 508 / 284 | 0 / 0 |
| Graph-only | 25,320 / 25,320 | 25,320 / 25,320 | 25,320 / 25,320 | 0 / 0 | 0 / 0 |

On the 163-row, nine-feature case (scalar 2), query 3 at width five
and epsilon zero has wheel-only row 105 and exported-Rust-only row
111. On the 191-row depth-two holdout, query 0 at width 191 and epsilon
zero gives wheel-only `[137, 5, 1, 138]` and exported-Rust-only
`[18, 183, 166]`. Both residuals disappear with the live graph.
The two public captures agree on all recorded observations outside
the latency measurements. The older 4,095/4,096 fixed-graph
suite still agrees on all 6,720
tree-free slots, including the flat-weight control. The separate
prepared regression's ten boundary configurations still agree on all
16,170 tree-free slots.

Projection ties are a separate geometry residual. On 25 alternating
minus-one and plus-one rows with a one-split search tree, caller 0
returns `[0, 0, 1, 1]` for queries `[0, -1, 1, 0]` at width one
and epsilon zero. The identical zero queries reach different leaves.
The native word-replay geometry deterministically chooses the left
child on equal projections, so a fixed branch cannot match both
zero queries. The caller-scalar-to-pivot-word generator remains open,
as do native construction of the live offers outside these controls
and parity with default pruning, refinement, or multi-tree preparation.
The estimator retains its existing routing.

## Warm latency

Release medians of 41 warm calls at width five and epsilon 0.12 on
Windows x86-64 (Intel Core Ultra 9 285K), in microseconds per query
batch. Each timed call includes allocating the ranked result.
The initial fit, preparation, and wheel JIT are outside the timer.

| Case | Batch rows | Exported Rust | Live Rust | Prepared wheel | Graph-only Rust / wheel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 113 rows, depth zero | 19 | 11.2 | 10.9 | 48.3 | 6.2 / 19.1 |
| 137 rows, depth one | 19 | 17.3 | 17.6 | 34.5 | 7.0 / 17.4 |
| 179 rows, depth two | 19 | 11.8 | 12.1 | 27.7 | 8.2 / 21.2 |
| 191 rows, depth two | 19 | 19.5 | 15.4 | 33.2 | 10.7 / 20.3 |

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-prepared-traversal'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$archive = Join-Path $probe 'projection-words.json'
$generator = Join-Path $probe 'projection-generator.json'
$env:PREPARED_TRAVERSAL_OBSERVATIONS = Join-Path $probe 'observations.json'
uv run --project reference --locked --no-build python -I reference/probe_projection_words.py --output $archive
uv run --project reference --locked --no-build python -I reference/probe_projection_generator.py --archive $archive --output $generator
uv run --project reference --locked --no-build python -I reference/probe_prepared_traversal.py --archive $archive --archive $generator --output $env:PREPARED_TRAVERSAL_OBSERVATIONS
cargo test --release --locked -j 2 -p umap-core --test prepared_traversal -- --ignored --nocapture
cargo test --release --workspace --locked -j 2
```

The traversal probe also runs with no `--archive` arguments for the
independent holdouts and tie controls alone.
