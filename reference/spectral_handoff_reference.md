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

## Seeded training schedule

Calling the published wheel's `make_epochs_per_sample` on explicit float32
weights gives this rule, with `N` the total epoch count and `m` the largest
membership in the supplied graph.
[UMAP schedule API](https://umap-learn.readthedocs.io/en/0.4dev/api.html#umap.umap_.make_epochs_per_sample)

```text
normalized = f32(weight / m)
expected_samples = f32(f32(N) * normalized)
positive_interval = f64(N) / f64(expected_samples)
negative_interval = positive_interval / negative_sample_rate
```

Zero expected samples give the public sentinel `-1.0` and no Rust cursor.
At `N = 0` this applies to all edges. At `N = 1` the unit interval of the
maximum-weight edges is first due at epoch one, so the only executed epoch,
zero, performs no updates. Both fits still draw the three accepted sampler
seed words. Each positive due time starts at its interval and advances by
that interval in float64. At each positive update the existing negative
cursor consumes its float64 backlog.

| Total epochs | Maximum weight bits | Edge weight bits, in input order | Public positive interval bits, in input order |
| ---: | --- | --- | --- |
| 1 | `3f800000` | `3effffff`, `3f000000`, `3f000001` | `4000000010000010`, `4000000000000000`, `3fffffffc0000080` |
| 3 | `3f800000` | `3eaaaaaa`, `3eaaaaab`, `3eaaaaac` | `4008000018000018`, `4008000000000000`, `4007ffffd0000060` |
| 7 | `3f400000` | `3e7fffff`, `3e800000`, `3e800001` | `400800000db6db76`, `4007ffffe4924944`, `4007ffffe4924944` |
| 59 | `3f35c28f` | `3c4529d3`, `3c4529d4`, `3c4529d5` | `404d800000000000`, `404d800000000000`, `404d7fffc5000076` |
| 60 | `3f35c28f` | `3c4529d3`, `3c4529d4`, `3c4529d5` | `404d7ffff4333338`, `404d7ffff4333338`, `404d7fffba2eef94` |
| 100 | `3f35c28f` | `3c4529d3`, `3c4529d4`, `3c4529d5` | `404d7ffff4333338`, `404d7ffff4333338`, `404d7fffd163d754` |
| 16,777,217 | `3f35c28f` | `3f35c28f`, `3f35c28f` | `3ff0000010000000`, `3ff0000010000000` |

The embedding-stage probe gives each weight its own directed edge. It uses
two tied maximum edges, three adjacent weights near `m / 59`, and three near
`m / 3`. Across 24 fits with maxima `1.0`, `0.71`, and `0.75` and initial
pairs `[0, 10]`, the schedule replay matches every coordinate. The pinned
nonunit case uses distinct pairs `[i / 16, 10 - i / 16]` and includes a stored
zero. With maximum bits `3f35c28f` and 60 epochs, all three edges near
`m / 59` update at epoch 59. Their head bits are `3e00e697`, `3e40e986`, and
`3e807644`. A direct float64 weight ratio schedules the lowest member of
the triplet at epoch 60.
On the 59-row spectral graph, the weight of edge `(10, 36)` and its reciprocal
has bits `3f363cbf`. The 100-epoch rule changes their interval from
`3ff679e79bccea86` to `3ff679e7a4a41e61` and moves one sample on each edge
from epoch 59 to 60.
[UMAP embedding-stage API](https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.simplicial_set_embedding)

## Wider squared distances

Single directed unit-edge probes hold the schedule and initial coordinates
fixed. At two epochs with `a = b = 1` and zero repulsion, there is one attraction
update. An independent replay matches its coordinates for 128 random pairs
at each of 39 tested widths between 1 and 160 components with this float32
reduction.

1. For each complete 32-component block, accumulate squared differences in
   32 lanes using fused multiply-add. For each `i` from 0 to 7, merge the
   lanes as `(lane[i] + lane[i+8]) + (lane[i+16] + lane[i+24])`. Add the high
   four merged lanes into the low four and reduce those as
   `(v[0] + v[2]) + (v[1] + v[3])`.
2. Accumulate any complete four-component blocks in four lanes using fused
   multiply-add, starting with the preceding sum in lane zero. Reduce the
   lanes in the same four-lane order.
3. Fold any final one to three components into the sum in component order,
   using fused multiply-add. Layout forces and coordinate additions continue
   to use float64, with float32 coordinate storage.

The same distance calculation matches all 400 isolated negative updates at
widths 6, 8, 37, and 68. The repulsive probe uses `a = 1e-30`, `b = 1`,
`gamma = 4`, rate 2, and seed 42. The first draw of the head at `x = 0`
targets row 4 of six. Its attraction to row 1 is negligible at the recorded
precision. The reduction is pinned to the behavior observed with the locked
Windows x86-64 wheel environment on an Intel Core Ultra 9 285K.

### Rust replay results

| Probe | Baseline changed coordinates | Baseline maximum error | Updated mismatches |
| --- | ---: | ---: | ---: |
| 59 rows, 100 epochs | 118/118 | `0.30931425` absolute | 0 |
| Weighted, six components, 100 epochs | 47/48 | 10 ULP | 0 |
| Nine rows, eight components, 100 epochs | 46/72 | 14 ULP | 0 |
| Weighted, six components, two epochs, attraction only | 1/48 | 1 ULP | 0 |
| Nine rows, eight components, two epochs, attraction only | 2/72 | 1 ULP | 0 |
| Nonunit boundary, 60 epochs | 2/18 | `0.00087963045` absolute | 0 |
| Isolated attraction, 68 components | 1/408 | 1 ULP | 0 |
| Isolated repulsion, 37 components | 1/222 | 1 ULP | 0 |
| Dense explicit fit with float32 scalars, six components, 100 epochs | 38/48 | 22 ULP | 0 |

The schedule change on its own fixes the 59-row case and leaves the six-
and eight-component residuals at their baseline counts. With the grouped
distance reduction, Rust matches all 3,272 coordinates and the exported RNG
state across the 68 wider spectral runs at 2, 11, 30, and 100 epochs. Those
runs supply the observed CSR for each epoch count. The regression fixture
pins 2,168 coordinates from 23 spectral, explicit dense, and isolated-stage
checkpoints, along with post-fit RNG cursors and next words. Two fresh public
embedding-stage replays reproduce every recorded checkpoint.

The two serial seeded fit entry points use the compatibility schedule and
distance reduction for both float32 and float64 parameter callers.
`optimize_csr` and `fit_dense_embedding` with an injected sampler retain the
direct float64 membership ratio, callback order, and sequential distance
reduction. The held-out transform optimizer can keep calling the existing
direct layout helpers and constructing its own explicit sampling cursors.
Those helpers keep their documented sequential fused distance calculation.

## Regeneration

Regenerate and verify from the repository root in PowerShell.

```powershell
$env:CARGO_TARGET_DIR = Join-Path $env:TEMP 'umap-spectral-rng-handoff-target'
$fixture = 'reference/fixtures/umap-learn-0.6.0-spectral-handoff-windows-x86_64.json'
uv run --project reference --locked python reference/generate_spectral_handoff_fixture.py --output $fixture
cargo test -p umap-core --locked --test spectral_handoff
cargo test -p umap-core --locked --test seeded_layout_precision
```

To replay the seeded-layout precision fixture with the public embedding-stage
API, reinterpret the stored weights and raw coordinates as float32 and rebuild
the CSR from `indptr` and `indices`. Use the recorded scalar settings with
`compatibility_layout=True` and `parallel=False`. A public NumPy `RandomState`
seeded with `seed` and advanced by `624 * block + 1` raw words yields the
MT19937 word array for the handoff. Set its position to `cursor` before passing
the explicit init to the stage. For the dense case, pass the fixture's
`dense.rows` and `dense.neighbors` to a fresh serial compatibility estimator
with the recorded explicit init and seed.
