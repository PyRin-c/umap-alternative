# Training projection words and tied rows

This stage starts from integrated main `71e7301`. The locked public
NNDescent constructor and `neighbor_graph` expose the training leaves
through zero-round fits. The wheel environment is UMAP 0.6.0,
PyNNDescent 0.7.0 and Numba 0.68.1. Each fit uses one worker.
For the leaf tests, every fitted leaf has enough neighbors to produce
a separate connected component.

## Keys, pivots and tied rows

Let `mix(x)` be one SplitMix64 output, including the initial
golden-ratio increment. All arithmetic below wraps at 64 bits.

| Value | Derived from |
| --- | --- |
| Root key | `mix(caller_scalar XOR mix(tree_ordinal + 1))` |
| First pivot word | `mix(node_key)` |
| Second pivot word | `mix(node_key XOR 0x5851f42d4c957f2d)` |
| Child key | `mix((node_key << 1) OR side)` |
| Tied training side | `mix(node_key XOR original_row_id) AND 1` |

The tree ordinal starts at zero. The first word reduces modulo the
incoming row count. The second reduces modulo one less and skips the
first position. The first pivot lies toward the positive side of
the signed projection. Side zero goes to that branch. The native
stage preserves the in-place partition order so each child sees the
same incoming row positions.

Inverting the SplitMix64 output finalizer on archived pivot words
revealed the common node key and the salt for the second pivot.
All 67 archived word pairs match exactly, including all seven
slots for eight scalars. Four new caller scalars contribute another
four independently recovered root pairs. Those pairs use 207 unique
public split witnesses across different row counts, with the same
remainder reconstruction as the earlier archive.

Training ties use the original row ID before any in-place swap or
subdivision. Reversing tied feature rows while holding the pivots
fixed, fitting different row counts, exchanging either signed
constructor draw, and using caller seed endpoints preserve the
same keyed assignment at a common scalar and node. The paired
353-row depth-two control reshuffles strict root assignments
while preserving both child sizes. All 241 shared child tie choices
agree by original row ID. Aligning by incoming child position agrees
on 247 of 349 choices. Coincident pivots give a zero plane and
exercise the same keyed assignment.

Prepared queries use the existing signed draw stream. The training
key comes from the independent caller scalar and assigns rows without
consuming a stream as the splitter scans or recurses.

## Margin arithmetic

For pivot vectors `a` and `b`, form the float32 normal `a - b`.
The training intercept is the fused float32 dot product of that
normal with `-(a + b) / 2`. The signed margin uses the same
32-coordinate and four-coordinate reductions and scalar tail, with
the intercept in lane zero. Both signs at or within the float32
`1e-8` band use the keyed side. Axis-aligned controls exercise
signed subnormals and the float32 coordinates immediately below,
at, and above both band edges.

The prepared intercept instead sums individually rounded products
in float64. With that intercept, the first five angled training
controls match 995 of 1,179 public branch labels. For the
197-row, seven-feature case at scalar 8231, it sends row 17 to
pivot 163's branch while the wheel sends it to pivot 186's. The
intercept words are `0xc043ddba` with prepared arithmetic and
`0xc043ddbb` with training arithmetic. In the 32-feature case
at scalar 841, the prepared intercept sends row 4 to pivot 213's
branch while the wheel sends it to pivot 185's. The training
intercept restores both choices and matches all the angled holdouts.

## Verification and scope

| Control | Exact observations |
| --- | ---: |
| Archived and newly recovered unsigned pivot words | 142 / 142 |
| Crossed root controls, including 4,095 and 4,096 rows | 31,458 / 31,458 |
| Axis-aligned float32 band edges | 4,728 / 4,728 |
| Nested splits and paired child row controls | 7,324 / 7,324 |
| Angled planes at 4 to 97 features | 2,159 / 2,159 |
| Coincident pivot controls | 414 / 414 |
| Total public training leaf memberships | 46,083 / 46,083 |
| Ordered ranks and distance bits from unions of 1, 2, 3 and 5 tree offers | 48,092 / 48,092 |

Two complete public captures were byte-for-byte identical. The
public components witness membership. The Rust leaf arrays
also match the independently coded in-place model. Descendant
pivot positions test the inherited order through depth four,
including capacity changes that stop only some branches. The
forest rank control uses continuous inputs and rows with enough
co-leaf offers to avoid vacancy fill.

The new `training_projection_leaves` entry point reconstructs
training words from arbitrary caller scalars. It shares the
projection reductions with explicit prepared query replay.
The 312,732 prepared projection-tie slots, 372,155 prepared
traversal slots and 25,320 graph-only slots still match. Release
workspace tests and the older 4,095/4,096 query regressions pass.

Generation of the prepared tree geometry remains open. Training
offer ordering under exact distance ties, vacancy fill, default
refinement and prepared pruning remain separate stages.
Estimator routing retains its existing path.

## Reproduce on Windows

Build the training-word archive with the previous public probe.
[Caller scalar archive](projection_generator_reference.md)

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-training-projection'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-target'
$words = Join-Path $probe 'projection-words.json'
$archive = Join-Path $probe 'projection-generator.json'
$env:TRAINING_PROJECTION_OBSERVATIONS = Join-Path $probe 'training-projection.json'
uv run --project reference --locked --no-build python -I reference/probe_projection_words.py --output $words
uv run --project reference --locked --no-build python -I reference/probe_projection_generator.py --archive $words --output $archive
uv run --project reference --locked --no-build python -I reference/probe_training_projection_ties.py --archive $archive --output $env:TRAINING_PROJECTION_OBSERVATIONS
cargo test --release --locked -j 2 -p umap-core --lib compare_training_projection_with_wheel -- --ignored --nocapture
cargo test --release --workspace --locked -j 2
```
