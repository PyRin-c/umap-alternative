# Spectral RNG handoff

The Windows x86-64 fixture records public `umap-learn==0.6.0` compatibility
fits of one connected eight-row graph at seeds `0` and `4294967295`. The
generator reconstructs the raw two-component layout with public SciPy and
NumPy calls. It draws the `(8, 3)` and `(8, 2)` normal blocks and saves the
MT19937 words and cursor immediately after the second block. It verifies the
normalized coordinates against the public zero-epoch fit and the caller RNG
state against the public fit after three accepted sampler seed words.

The Rust test compares all 160 float32 output bits from the positive-epoch
checkpoints at 2, 3, 11, 30, and 100 epochs. It also compares the zero-epoch
coordinates, the post-fit RNG cursor, and the next eight words. Restarting
from the original seed changes the 30-epoch layout for both cases. The same
fixture records the NumPy seed and first twisted word arrays and mixed uniform
and integer continuations at cursors 0, 623, and 624.

## Rust call contract

After computing the raw coordinates, extract the word array and cursor from
the supplied Python `RandomState`.

```python
algorithm, words, cursor, has_gaussian, cached_gaussian = rng.get_state()
```

Import the MT19937 words and cursor, then pass the existing CSR graph, raw
row-major `f32` values, and layout settings to Rust.

```rust
let mut caller_rng = Mt19937::from_state(&words, cursor)?;
let config = EmbeddingLayoutConfig {
    n_components: 2,
    a: 1.7_f64,
    b: 0.9_f64,
    learning_rate: 0.13_f64,
    n_epochs: 30,
    repulsion_strength: 0.37_f64,
    negative_sample_rate: 5.0,
};
let embedding = fit_graph_embedding_with_rng(graph, &raw_coordinates, &config, &mut caller_rng)?;
let (next_words, next_cursor) = caller_rng.state();
```

The fit normalizes a copy of the coordinates and moves the supplied graph into
the result. The caller RNG advances only for sampler seeding, even when the
epoch count is zero. An unsuccessful fit preserves the RNG state. When
writing the exported stream back to Python, keep the original `has_gaussian`
and `cached_gaussian` fields in the `RandomState.set_state()` tuple.

## Wider public probes

For the 17 cases in `generate_spectral_fixture.py`, fresh public fits at 2,
11, 30, and 100 epochs matched public explicit-init replays from the raw
spectral coordinates and advanced RNG. The Rust handoff matched the final
MT19937 word array and cursor for every fit. Some public fits zeroed stored
memberships below `graph.max() / n_epochs`. Recreating the spectral coordinates
with each fit's observed CSR weights removed the resulting layout differences
on the 13-, 19-, 32-, 36-, 48-, 53-, and 83-row inputs. The Rust handoff uses
the weights and raw coordinates provided by the caller.

The following probes used the same `a`, `b`, learning rate, repulsion strength,
and negative sample rate as the compact fixture.

- The 59-row input at 100 epochs differed in all 118 coordinates, with maximum
  absolute error `0.30931425`. Its edge `(10, 36)` and reciprocal edge have
  weight `0.7118644118309021`. The current `f64` interval samples the edge at
  epoch 59. Computing `100 / float32(100 * weight)` shifts that sample to
  epoch 60. Testing that interval in a scratch optimizer made all 118
  coordinates match, which identifies a schedule rounding difference.
- The weighted input at seed 42 with six components and the nine-row input
  with eight components retained small differences even with the observed graph
  and matching RNG state. At 100 epochs they differed by at most 10 and 14 ULP,
  respectively. The alternative sampling interval left these differences intact.
  The nine-row input also had two one-ULP differences at two epochs with
  `a = b = 1` and zero repulsion, which reproduces a residual in the attraction
  arithmetic. The exact operation responsible is unresolved.

## Regeneration

Regenerate and verify from the repository root in PowerShell.

```powershell
$env:CARGO_TARGET_DIR = Join-Path $env:TEMP 'umap-spectral-rng-handoff-target'
$fixture = 'reference/fixtures/umap-learn-0.6.0-spectral-handoff-windows-x86_64.json'
uv run --project reference --locked python reference/generate_spectral_handoff_fixture.py --output $fixture
cargo test -p umap-core --locked --test spectral_handoff
```
