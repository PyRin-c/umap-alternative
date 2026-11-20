# UMAP 0.6.0 seeded serial modern sampling

`ModernSampledLayout` in the Rust `sampling` module composes native serial
dispatch with the existing Adam and momentum epoch kernels. It takes the CSR
graph, initialized coordinates, and caller RNG state after initialization.
The caller preserves the Gaussian cache across epochs. Each step returns the
directed attractions, accepted non-self repulsions, and post-update coordinates.

## Wheel observations

`generate_modern_sampling_fixture.py` uses the published `umap-learn==0.6.0`
wheel, `UMAP.fit`, `graph_`, `embedding_list_`, and a caller-provided NumPy
`RandomState` subclass that records its public `randint`, `shuffle`, and
`randn` calls. The fixture uses Ubuntu x86-64 under WSL, CPython 3.13.12, and
one Numba thread. Each recorded fit was repeated with a fresh input and RNG.

The momentum probes start from affinely independent points. Each target's
force contributes along a distinguishable direction. The generator inverts
the observed momentum updates and uses least squares to recover attraction
activity from rate-zero controls. It uses that attraction schedule to recover
negative target multiplicities from the repulsive fits. The largest difference
between a fitted multiplicity and its rounded integer was `0.000539713`.
Self targets have zero displacement. The wheel coordinates identify the
non-self multiset; they do not uniquely identify the ordering of repeated
samples or individual self slots.

## Caller RNG and candidate order

- The caller RNG first consumes `randint(-2147483647, 2147483646, 3)`.
  This also happens with zero epochs. The full-range controls found no
  change in targets when the caller supplied different triplets with the
  same shuffle stream.
- Momentum shuffles its existing vertex permutation once per epoch. Adam
  shuffles two independent existing permutations with full-range selection.
  Its second permutation supplies the next epoch's candidates. Both
  permutations begin in vertex order. Each uses the same MT19937 stream and
  the masked rejection draw for each descending swap.
- With `negative_selection_range < n_rows`, both optimizers shuffle once
  and draw one standard normal per embedding component. The new candidate
  order sorts the vertices by their binary64 projection onto the normalized
  Gaussian direction, using the coordinates after the epoch update. Equal
  row count and selection range uses the full-range branch.
- The ordered pool prepared at epoch `e` supplies candidates at epoch
  `e + 1`. Shuffling and projection draws occur at rate zero too. Each
  fit consumes the RNG draws for its final epoch.

## Edge timing and dispatch

Weak memberships are removed from both the schedule and graph-neighbor
exclusion, while the fitted public CSR still contains them. The tested
small-graph cutoff is `max_weight / 500` for budgets up to 10 and
`max_weight / n_epochs` for larger budgets. The division is float32. An edge
exactly at the cutoff is retained. After pruning, retained CSR entries get
new consecutive edge ordinals. The 10- and 11-epoch fixtures test the
float32 value immediately below, equal to, and immediately above each cutoff.

For a retained membership `w`, the positive interval is
`N / f64(f32(N) * f32(w / max_weight))`. With a nonzero rate, the negative
interval is `(positive_interval / rate) * 1.5` in float64. Each cursor starts
at its interval, and negative backlog is consumed only when its positive
edge fires. Its count is `max(0, floor((epoch - next_negative) / interval))`,
and the due time advances by that count times the interval. On a unit edge,
rate one first emits a negative at epoch 3. Rate five emits 2, 3, 3, and 4
draws per edge at epochs 1 through 4. The membership `float32(0.2)` in a
seven-epoch fit separates the positive interval from a direct float64 weight
ratio. Membership `float32(1/3)` at 12 epochs distinguishes the negative
division-then-multiplication from folding the factor into the rate.

The first candidate position for retained edge ordinal `k` in epoch `e`
is `k * (e + 1) % n_rows`. Without exclusion, subsequent slots advance
by `k` modulo the pool length. A self target consumes its slot. With
exclusion, scan the pool by consecutive positions, wrapping as needed, and
skip the source and its retained CSR neighbors until the backlog is
satisfied. A fully excluded row produces no repulsive forces. Repeated
eligible targets are preserved, including when the count exceeds the
number of eligible vertices.

## Verification and limits

| Optimizer | Fits | Epochs | Attractions | Accepted negatives | Exact epoch coordinates | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Momentum | 23 | 183 | 2,450 | 6,457 | 8,123 / 11,632 | 1.79e-7 |
| Adam | 15 | 108 | 1,516 | 3,637 | 5,071 / 6,972 | 4.30e-6 |

Momentum's counts and target multiplicities match the independent decode
exactly. Adam's negative count is the native replay count, checked against
the wheel through its embeddings and RNG checkpoints. Across both
optimizers, the RNG test matches 38 seed handoffs, 378 in-place shuffles and
their raw-word cursors, and 506 Gaussian variates within two binary64 steps.
The suite also compares the post-fit cache and the next four raw words for
each fit. The held-outs include a non-unit graph maximum, a dense Euclidean
weighted graph, alternate seeds, and projected orders on 9 and 17 vertices.

There is an unresolved projection ordering case. With a 24-row simplex and
the controlled direction `arange(24) % 4`, a stable projection sort gives
62 incorrect head-target multiplicity entries at epoch 1. NumPy's default
sort matches this observation. The exact ordering of equal or numerically
indistinguishable projection keys across BLAS and NumPy builds is open.
The sampler reports `AmbiguousProjection` before an epoch whose negatives
need that order. The fixture generator reproduces the tie independently.

Coordinate accumulation and rounding still have the residuals above.
Further coverage is needed for disconnected layouts, nondefault
`negative_sample_scale` and its adaptation parameter, and large dense
graphs. The short-budget cutoff above 10,000 rows uses the existing
compatibility layout's 200-epoch threshold and lacks a modern wheel witness.
The native sampler exposes integer rates. Python fit integration remains
separate.

## Reproduction

From the repository root in WSL, with `uv` and the locked reference
environment available:

```bash
export TMPDIR="$HOME/.cache/umap-alternative-verification"
mkdir -p "$TMPDIR"
export UV_PROJECT_ENVIRONMENT="$TMPDIR/reference-venv"
export NUMBA_CACHE_DIR="$TMPDIR/modern-sampling-numba"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python -IB reference/generate_modern_sampling_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-sampling-ubuntu-x86_64.json
cargo test -p umap-core --lib --locked modern_rng_matches_public_seed_shuffle_and_normal_draws
cargo test -p umap-core --test modern_sampling_fixtures --locked -- --nocapture
```

Regeneration produced the fixture byte for byte. The locked workspace tests
passed with 175 tests and one existing ignored test. Workspace Clippy passed
with `-D warnings`, and `cargo fmt -p umap-core -- --check` passed. The
workspace-wide format check reports an existing import wrap in
`crates/umap-python/src/lib.rs` on the base commit.
