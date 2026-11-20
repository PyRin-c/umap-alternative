# Prepared Euclidean projection ties

This stage starts from integrated main `dbc51b1`. It uses the public
NNDescent constructor, exported neighbor graph, `prepare()`, and
`query()` from the locked UMAP 0.6.0 environment (PyNNDescent 0.7.0,
Numba 0.68.1). Each index uses one search tree and one worker. The
rank sweeps condition on oriented split geometry and the wider
zero-round live-offer table from the earlier replay.

## Branch stream

The query stream starts from the second signed three-word constructor
draw. After ten advances, the next word's low bit selects the child
at a tied split. Zero selects the first pivot's side and one the
second. Strict splits consume no word. A tied child consumes the
next word only if the root sends the query there. Each query call
restarts the stream and advances it in batch and tree traversal
order. Rank width and epsilon affect the subsequent graph walk.

The initial probe uses 25 alternating minus-one and plus-one rows.
Sixty-four repeated zero queries expose 64 branch choices per call.
For each of eight caller seeds, including zero, one, and the two
largest uint32 seeds, four substituted forest scalars give the same
sequence. Replacing the first signed draw leaves the sequence
unchanged. Replacing the second changes it, and a zero second draw
gives 64 zeros. All 2,432 choices match the existing signed RNG with
the ten advances. The constructor records exactly three draws and
preparing and querying add no public caller draws.

The fixed-geometry holdouts check repeated rows, singleton and fresh
calls, adjacent pairs, reversed pairs, and mixed batch order. The
179-row depth-two split has ordered pivots `(144, 119)` at the root
and `(84, 172)` and `(148, 119)` at the children. Integer bisector
equations give float32 points on the root and one or both child
planes. With caller 73, repeating the first triple intersection at
width one and epsilon zero gives IDs
`[107, 85, 110, 108, 107, 107, 108, 85]` in the first eight
positions. These identify leaves `[3, 1, 0, 2, 3, 3, 2, 1]` in
left-to-right order. A fresh singleton returns 107.

## Margin arithmetic

Each split uses the float32 normal `a - b`. To form the intercept,
round each product of this normal and the float32 pivot midpoint to
float32, accumulate the negated products in float64, then convert the
sum to float32. The query dot product starts with the intercept in
lane zero. It uses fused multiply-add in 32-coordinate blocks,
four-coordinate blocks, then a scalar tail. The horizontal reduction
reuses the order measured for squared Euclidean distance.

The branch consumes a bit when the magnitude of the rounded margin
is at most `1e-8_f32`. A larger positive margin selects the first
pivot's side and a larger negative one the second. Centering the
137-row roots at their midpoints separates the margin from intercept
cancellation. The probes use signed zero, both smallest nonzero
float32 values, and adjacent float32 query coordinates around the
positive and negative band edges. For the seven-feature root at unit
scale, the last tied margin is `9.99999993922529e-9`, and the next
sampled margin is `1.000000171558213e-8`. Rescaling the data by
factors as small as 1/16 and as large as 64 preserves the margin
threshold. The nonzero bands and the uncentered nextafter probes
distinguish this arithmetic from comparing rounded Euclidean
distances to the pivots.

One alternative kept each intercept product in float64. That yielded
126 wheel-only and 126 Rust-only IDs over 822 uncentered nextafter
slots at nine and eleven features. Rounding each product before the
float64 sum removes those differences. A second alternative used only
four-coordinate dot blocks. At 32 features, caller 177 and repeated
query row 2, it returns ID 19 while the wheel returns 79 even though
the distance bits agree. With the 32-coordinate blocks, the same
fixture matches all 423 ranks and distance bits.

## Verification

| Control | Exact ordered ranks and distance slots |
| --- | ---: |
| Centered roots, float32 band edges, four caller and scale configurations | 124,124 / 124,124 |
| Depth-two intersections and batch schedules, four caller seeds | 181,612 / 181,612 |
| Uncentered nextafter at archived nine- and eleven-feature splits | 822 / 822 |
| Independent one-split float32 holdouts, 4 to 97 features | 6,174 / 6,174 |
| Total new rank slots | 312,732 / 312,732 |
| Existing fixed-geometry prepared traversal | 372,155 / 372,155 |
| Existing fixed-graph tree-free traversal | 25,320 / 25,320 |

The width sweep uses 1, 3, 5, 13 and seventeen more than the training
row count. It covers zero epsilon, the smallest positive float64
and float32 epsilons, and 0.12. Independent feature holdouts cover the
31/32/33 transition and repeated 32-coordinate blocks at 64, 65,
and 97 features. The final replay has zero wheel-only and zero
Rust-only IDs over the tested geometry. Two public captures produce
byte-identical results. An earlier graph-only boundary suite at
4,095 and 4,096 rows also passes.

The native tie stream runs when the caller supplies prepared
projection words. The fit-time assignment of training rows on tied
planes and caller-scalar-to-pivot-word generation remain open.
Default pruning, multi-tree preparation, and estimator routing retain
the previous path.

## Reproduce on Windows

Generate the archive using the prepared traversal probe. Use the full
archive from the earlier report to reproduce the historical rank totals.
[Prepared traversal controls](prepared_traversal_reference.md)

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-projection-ties'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$env:PREPARED_TRAVERSAL_OBSERVATIONS = Join-Path $probe 'prepared-traversal.json'
$env:PREPARED_PROJECTION_TIES = Join-Path $probe 'projection-ties.json'
uv run --project reference --locked --no-build python -I reference/probe_prepared_traversal.py --output $env:PREPARED_TRAVERSAL_OBSERVATIONS
uv run --project reference --locked --no-build python -I reference/probe_prepared_projection_ties.py --archive $env:PREPARED_TRAVERSAL_OBSERVATIONS --output $env:PREPARED_PROJECTION_TIES
cargo test --release --locked -j 2 -p umap-core --lib compare_prepared_projection_ties_with_wheel -- --ignored --nocapture
cargo test --release --locked -j 2 -p umap-core --test prepared_traversal -- --ignored --nocapture
cargo test --release --workspace --locked -j 2
```
