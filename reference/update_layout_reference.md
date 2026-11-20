# Serial dense UMAP update layout

`generate_update_layout_fixture.py` records public `UMAP.fit`, `UMAP.update`,
`graph_`, and `embedding_` from the pinned 0.6.0 wheel. Each trial fits a
fresh estimator and applies its public parameter overrides after fit. Within
each case, the fitted embedding and the updated CSR graph are bit-identical
across all trials. The generator repeats the entire sweep. Its separate
`numpy_rng_model` record comes from public NumPy `RandomState` calls.

The Rust replay matches all 27 checkpoints (526 coordinates), including 20
positive-epoch checkpoints, on Windows x86-64 and Ubuntu x86-64 under WSL.
The regenerated fixtures differ only in platform and machine metadata. The
tests supply the public zero-epoch layout and combined graph directly to the
optimizer. The [initialization reference](update_reference.md) covers the
earlier feature and coordinate preparation.

## Optimization schedule

An update with `n_epochs=None` matches explicit zero epochs, including when
the estimator was fitted with the default epoch count. One epoch also
matches zero. At two epochs, the unit-weight edges first become due and use
the full fit-time learning rate. The tested three-, 30-, 200-, and 500-epoch
checkpoints match the existing seeded CSR schedule, with float32 weight
normalization and expected sample counts, float64 due times, and decay
applied after each epoch. Setting `learning_rate` to 0.5 after fitting with
0.13 gives exactly the 0.13 replay. Post-fit changes to
`repulsion_strength` and `negative_sample_rate` affect optimization.

Positive updates visit both directed memberships in ascending CSR order and
move both endpoints, including existing rows. The paired two-epoch
attraction-only trial moves seven of eight old rows and both appended rows.
The negative cursor is drained when its associated positive edge is due.
At rate one the two-epoch checkpoint matches attraction only. At rate two
each unit edge takes one negative draw. The negative update moves the head
and may sample any combined row, including the head. A self draw consumes
the RNG word and leaves the coordinates unchanged.

The published graph retains all 40 directed memberships in the paired case
at both two and 30 epochs. Twelve are unit-weight edges, and the smallest
weight is approximately 0.102897. Only the unit edges execute during the
two-epoch sweep. The 30-epoch attraction-only trial matches the scheduled
weighted sweep, using float32 squared distances and float64 force, clip,
and rate arithmetic before storing each coordinate as float32. The
optimizer keeps the supplied CSR graph and derives edge eligibility from
the cursors. Public checkpoints leave internal pruning unobservable for
edges whose first due time lies outside the run.

## RNG handoff

1. Start MT19937 with `transform_seed` and consume three accepted seed
   words before any Gaussian initialization noise.
2. Draw Gaussian jitter from that stream if provisional coordinates
   coincide. The paired layout consumes another 52 raw words with seed
   42 and 40 with seed 24. The single addition and distinct-anchor pair
   consume zero jitter words.
3. Draw three more accepted words for optimizer seeding. The existing
   [per-head sampler](negative_stream_reference.md) adds the head's first
   normalized coordinate's float64 bit pattern to each signed seed lane.
   Each head keeps its own stream across edges and epochs, with signed
   words mapped to indices by nonnegative remainder.

| Transform seed and case | First optimizer word position, zero-based | Raw optimizer seed words |
| --- | ---: | --- |
| 42, single | 3 | `[787846414, 3143890026, 3348747335]` |
| 42, close pair | 55 | `[4222944499, 2544401215, 2004731384]` |
| 24, distinct anchors | 3 | `[227008855, 4294397329, 1363601553]` |
| 24, close pair | 43 | `[2415635830, 2710853551, 2200320217]` |

An exhaustive search through negative targets for the paired two-epoch
checkpoint yields one bitwise replay. In CSR order the targets are
`[6, 6, 4, 0, 2, 2, 2, 6, 6, 8, 5, 2]`. This includes self draws for
heads 6 and 8. The per-head sampler seeded after jitter reproduces the
sequence. Using the initial triple for seeding disagrees even in the
no-jitter runs. The public update probes cover two integer transform seeds,
two layout dimensions and small dense Euclidean training sets. The seed
boundary cases for integer rejection and wider layout arithmetic have
separate fit fixtures.

## Rust integration

Call `update_layout::begin_update_layout_rng(transform_seed)` before
initialization. Use the returned `Mt19937` for any Gaussian draws and
pass it, the normalized row-major float32 coordinates, and the full
combined CSR graph to `update_layout::optimize_update_layout`. Set
`EmbeddingLayoutConfig.n_epochs` to zero when the estimator's public
epoch parameter is omitted, and use the fit-time rate and curve with
the update-time repulsion strength and negative sample rate.

## Reproduction

Use the locked reference environment and a host-local Numba cache. The
following Bash commands reproduce the fixture in Ubuntu under WSL.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/umap-update-layout/venv"
export UV_CACHE_DIR="$HOME/.cache/umap-update-layout/uv"
export NUMBA_CACHE_DIR="$HOME/.cache/umap-update-layout/numba"
export CARGO_TARGET_DIR="$HOME/.cache/umap-update-layout/target"
uv sync --project reference --locked --python 3.13.12
uv run --project reference --no-sync python -B -s reference/generate_update_layout_fixture.py \
  --output "$HOME/.cache/umap-update-layout/ubuntu.json"
cargo test -p umap-core --release --test update_layout_fixture
```

On Windows, use an equivalent cache directory under `$env:LOCALAPPDATA\Temp`
and run the same Python generator and Rust test from the repository root.
