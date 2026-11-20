# UMAP 0.6.0 modern dense layout

`generate_modern_layout_fixture.py` exercises the published 0.6.0 wheel
through `UMAP.fit`, `graph_`, `embedding_`, and `embedding_list_`. It
records float32 bits, public parameter outcomes, and three fresh fits per
configuration. The wheel observations come from CPython 3.13.12 on Ubuntu
x86-64 under WSL with one Numba thread and `n_jobs=1`.
[UMAP optimizer guide](https://umap-learn.readthedocs.io/en/latest/optimizers.html)

Run from the repository root on the Windows-mounted drive in WSL.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/umap-alternative-verification/reference-venv"
export NUMBA_CACHE_DIR="$HOME/.cache/umap-alternative-verification/reference-numba"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python -I reference/generate_modern_layout_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-layout-ubuntu-x86_64.json
cargo test -p umap-core --test modern_layout_fixtures
```

## Starting coordinates and graph

All fits pass explicit, distinct coordinates and `init_rescale="graph_edge"`.
The zero-epoch pair witness passes `[[0,0],[1,0],[0,6],[3,10]]` with a scale
factor of 2 and receives `[[-2,-8],[0,-8],[-2,4],[4,12]]`. The result is
consistent with centering each column at its median and scaling the smaller
of the two equally weighted pair distances to 2. The Rust tests begin at the
recorded epoch-zero coordinates. Both a dense feature matrix and a dense
precomputed distance matrix produce the same unit-weight pair graph.
Rust fits the supplied inputs and compares the CSR offsets, targets, and
weight bits before replaying any epoch.
[UMAP estimator API](https://umap-learn.readthedocs.io/en/latest/api.html)

## Forces and timing

Let `d` be source minus target and `q` its squared Euclidean norm. The
single-edge attractive displacement direction is
`-2*a*b*q^(b-1)*d/(1+a*q^b)` for `q > 0`. The `a=100000` pair case
exceeds the compatibility component clipping threshold and follows the
unclipped result. The fit's symmetric pair graph contributes one such force
to each endpoint. The cycle and weighted graph probes support accumulation
from the same starting coordinates before each optimizer update.
[UMAP paper](https://arxiv.org/abs/1802.03426)

The pair negative witness uses `negative_sample_rate=1` and graph-neighbor
exclusion. Its first active negative epoch is 3 and a single cross-pair
sample per row explains the measured force. The two possible targets are
distinguished by the sign or magnitude of the coordinate changes. For seed
42, the first two rows' fitted displacement indicates target 2 in momentum
and target 3 in Adam. For both optimizers, the third and fourth rows are
replayed with target 0. The first pair coincides at the Adam draw, so targets
0 and 1 give the same force in those last two rows.

The inferred one-sided repulsive force for `q > 0` is
`gamma * 2 * tanh(2*b*sqrt(q)/((0.001+q)*(1+a*q^b))) * d/sqrt(q)`.
The four-cycle leaves only the opposite vertex eligible for each row. The
close two-coordinate opposite pair discriminates a vector magnitude limit
from per-component clipping. Changing `gamma` scales the limited force.
The cycle's observed sample counts per row at rate 5 are consistent with
`[0, 4, 6, 6, 8]` at epochs 0 through 4. Rate 3 gives two per row at epoch 1
in the two-epoch probes. These are inferred counts for these inputs.

Weight-one edges first contribute at epoch 1. The dense weighted graph also
has memberships approximately 0.82775 and 0.58496. Replaying with the
existing positive cursor at interval `1 / weight` reproduces both
optimizers' coordinate bits through the ten-epoch fit. With the maximum
weight equal to one, the two weaker memberships first fire at epoch 2.
The 0.82775 memberships then fire at epochs 3, 4, 5, 7, 8, and 9. The
0.58496 memberships then fire at epochs 4, 6, 7, and 9.

## Learning rates, state, and checkpoints

Epoch `e` starts at zero and runs while `e < N` for an `N`-epoch fit.
With the tested graphs, scalar epoch lengths 0 and 1 both return the
initialized coordinates. A request list sets `N` to its maximum and
reports snapshots after the indexed epoch, in increasing order with
duplicate requests collapsed. A request for `N` receives the final
coordinates again when `N-1` is also requested. The unsorted request
`[7, 2, 0, 2, 5]` records the snapshots for 0, 2, 5, and the final result.

For momentum, `r = 1 - e/N` and `r_prev = 1 - max(e-1, 0)/N`. The inferred
coordinate update uses a float32 velocity initialized to zero.

```text
alpha = (learning_rate / 8) * r^2
carry = 0.5 - r_prev^2 / 8
velocity = f32(carry * velocity + alpha * accumulated_force)
position = f32(position + velocity)
```

For Adam, the first phase runs up to `k = min(floor(N/2), 100)`. The fitted
schedule uses the supplied learning rate in the first phase and fixed
endpoints in the second. In particular, `learning_rate=0` freezes the
momentum fixture and still allows Adam to move.

```text
alpha = 0.1 + (2*learning_rate - 0.1)*(1-e/k)^2  when e < k
alpha = 0.2 - 0.15*(e-k)/(N-k)                  when e >= k
m = f32(beta1*m + (1-beta1)*accumulated_force)
v = f32(beta2*v + (1-beta2)*f32(accumulated_force^2))
position = f32(position + alpha * (m/(1-beta1^(e+1))) /
               (sqrt(v/(1-beta2^(e+1))) + 0.0002))
```

The moment coefficients are float32 0.9 and 0.99 before promotion for
arithmetic. The first active epoch uses bias-correction time 2. A
coordinate with zero net force keeps its position and both moments. The
axis-collision probe leaves one axis fixed while the other continues to
move. The delayed negative probe reactivates the collided x coordinates
after an intervening zero-force epoch and recovers the carried moments.
Momentum carries its previous velocity through the same zero-force
condition. [Adam paper](https://arxiv.org/abs/1412.6980)

On the same initialization and graph, explicit `compatibility_layout=False`
with the optimizer omitted matches explicit Adam. Either modern optimizer
opts into the modern layout when `compatibility_layout=None` and raises
`ValueError` with `compatibility_layout=True`. The wheel accepts zero
learning rate and zero negative sample rate. Negative values for those
parameters or for the repulsion strength, empty or negative epoch lists,
and a floating-point epoch count raise `ValueError` in the recorded probes.
[UMAP estimator API](https://umap-learn.readthedocs.io/en/latest/api.html)

## Verification and remaining work

The fixture replays 38 cases bit for bit. These cover isolated attraction,
individually identifiable repulsion, state at zero force, weighted
attraction, and repeated negative draws on the symmetric square for both
optimizers. They also cover Adam's phase change at epoch 100 in a
202-epoch run. The remaining seven cases measure residuals against the same
public graph and recorded checkpoints.

| Probe | Momentum maximum float32 steps | Adam maximum float32 steps |
| --- | ---: | ---: |
| 1D cycle attraction | 1 | 0 |
| 1D cycle with two negative draws per row | 64 | 2 |
| 1D cycle with repeated negatives | 1 | 0 |
| Symmetric square with two negative draws per row | 0 | 1 |
| Asymmetric 2D cycle with repeated negatives | 3 | 1 |

The 64-step discrepancy occurs near a coordinate of -0.000167, where
attraction and repulsion almost cancel. Its absolute difference is below
`1e-9`. The accumulation order and intermediate rounding for general
multi-neighbor forces remain open.

All three fresh fits in each recorded configuration agreed bit for bit
at fixed seed. In the cycle seed controls, changing the seed from 42 to
43 with graph-neighbor exclusion kept all checkpoint bits. Turning
exclusion off made the seeds diverge at epoch 1 for both optimizers.
These observations use one Numba thread.

A full modern fit still needs a verified general negative selection
stream, weighted-interval rounding and weak-edge handling across graphs
and epoch budgets, and recursive initialization.
The estimator can already produce random or explicit initial coordinates
at zero epochs. [Initialization checkpoints](modern_initialization_reference.md)
The learning-rate schedules here cover explicit initialization and
small dense Euclidean layouts. Hard-negative sampling, high component
counts, and the schedule with other initialization methods need separate
probes. `ModernLayout` takes initialized coordinates and accepted samples
as inputs and exposes the post-epoch coordinates to its caller.
