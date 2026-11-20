# Caller scalars and projection word tests

The baseline is `a02bb43`. The probe uses the public PyNNDescent 0.7.0
constructor, `neighbor_graph`, `prepare`, and `query` in the pinned
umap-learn 0.6.0 environment. It loads the earlier projection-word
capture as a separate control and obtains new witnesses from the wheel.
[Previous projection word capture](projection_words_reference.md)

## Caller state

The caller draws three signed words twice, then one scalar in `[0, 2^62)`.
The scalar takes the low 30 bits of the next raw MT19937 word as its high
word and uses the following raw word as its low word. The new public Rust
helper `draw_projection_scalar` reproduces the whole sequence, including
signed-word rejection before it reads the scalar. The probe exercises
caller seeds 0, 1, 2, 42, `2^32 - 2`, and `2^32 - 1`. A caller state
installed through public `RandomState.set_state` makes the signed draw
reject four raw words and makes the subsequent scalar reach `2^62 - 1`.
The next Rust draw is the sentinel word 15.

## Training words

New root witnesses use 55 sizes between 65 and 521 and nine features
generated with data seed 2771. They recover a unique pair for all 19
forced scalars, including zero, adjacent values around 1, 42, `2^31`,
and `2^32`, and the two largest values in the scalar range. All five
archived root pairs agree exactly on these new inputs. Independent
child captures recover four additional pairs at depth three for eight
scalars. Neither observation uses JIT or package internals.

The capture has 1,015 root witnesses, 1,317 child witnesses, and 4,093
grandchild witnesses. It contains 67 exact word pairs, including all
seven node slots for each of the eight deeper scalars. For the maximum
scalar, the root words are `0xf1c3e1794537a392` and
`0xc3925360ea19d47c`.

The probe checks the direct NumPy PCG64, PCG64DXSM, Philox and SFC64
streams, both MT19937 32-bit packing orders, Python `Random`, and
SplitMix64. It compares exact raw words first and then compares reduced
pivot positions on the public witnesses. Each candidate agrees on zero
of ten archived words and zero of 28 held-out words. PCG64 agrees on one
of 747 held-out pivot pairs after reduction. Every other candidate agrees
on zero, and all agree on zero of the 268 archive-scalar pivot witnesses.
The arbitrary scalar-to-word generator remains unidentified. Replay uses
the recovered words through the existing explicit Rust API. The estimator
continues to use its existing routing.

Thirty separate depth-three fits use new feature seeds, training widths
around 768 and 1,024 rows, and the 4,095 and 4,096 row cases. The Rust
replay agrees on all 190 ordered pivot pairs and ordered leaf arrays of
the positional model. It matches 44,063 of 44,067 public component labels.
The packed preorder alternative matches 14,218 labels. One full tree
visits slots `[0, 1, 3, 4, 2, 5, 6]` in recursive order. At 4,096 rows,
increasing capacity from 301 to 302 omits slot 6 and preserves the six
other slots. Both cases reproduce all 4,096 component labels. This fixes
node placement and leaves the generator's temporal draw order open.
The four component-label differences occur in the 1,024-row, eleven-feature
case at capacity 38. There are two cross-leaf graph offers and one leaf
with four rows.

## Prepared projection

The prepared query probes distinguish the effect of the graph from the
training word stream. Rank-one, epsilon-zero queries witness projection
entries before they can expand the walk. As in the prior capture, the
query geometry is recovered from separate queries and then replayed on
nine held-out rows, including the first and last training rows. The rank
sweeps compare epsilon zero with the smallest positive float64 and
float32 values, as well as 0.12 and 0.24.

A permutation which keeps both training pivots in position preserves all
655 training graph slots and all 640 prepared rank-one results after
relabeling. Reducing capacity from 130 to 80 preserves the same graph and
changes 106 results by adding a prepared child split. Changing training
rank width from five to seven changes 218 entries and replaces the
prepared root `(30, 127)` with `(14, 77)`. Three different-scalar pairs
have matching ordered training pivots and identical training graphs at
47, 50, and 85 rows. Their prepared queries agree on all 1,920 indices
and distance bits. Prepared geometry varies with rank width and capacity
at a fixed scalar.

The four held-out prepared replays match all 36 rank-one entries at
epsilon zero and all 4,860 graph-only slots and distance bits. Calibrated
prepared geometry matches 4,685 of 4,860 ordered slots, up from 4,020
with the original native tree seed. Reusing the training words for
prepared queries matches 2,223 slots. All 4,797 shared prepared distances
match bit for bit. Remaining support differences include 63 wheel-only
and 63 Rust-only slots. In the maximum-scalar case at width five and
epsilon zero, query row five has wheel-only IDs `[43, 79]` and Rust-only
IDs `[8, 112]`.

Warm timings are release medians of 41 nine-query batches at width five
and epsilon 0.12, on Windows x86-64 with an Intel Core Ultra 9 285K. The
timed call excludes fitting, preparation and JIT compilation. Latencies
are 5.1–7.4 µs for Rust and 16.7–22.4 µs for the wheel with a prepared
tree. Graph-only latencies are 2.9–3.2 µs and 9.5–11.9 µs respectively.

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-projection-generator'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$archive = Join-Path $probe 'projection-words.json'
$env:PROJECTION_WORD_OBSERVATIONS = Join-Path $probe 'projection-generator.json'
uv run --project reference --locked --no-build python -I reference/probe_projection_words.py --output $archive
uv run --project reference --locked --no-build python -I reference/probe_projection_generator.py --archive $archive --output $env:PROJECTION_WORD_OBSERVATIONS
cargo test --release --workspace --locked -j 2
cargo test --release --locked -j 2 -p umap-core --lib compare_projection_words_with_wheel -- --ignored --nocapture
```
