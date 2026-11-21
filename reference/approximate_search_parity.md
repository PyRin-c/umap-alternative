# Seeded dense approximate search

The reference is the pinned UMAP 0.6.0 wheel and its PyNNDescent 0.7.0
dependency. The capture uses one numerical thread, fixed seeds, dense
float32 features, and the public `nearest_neighbors`, `UMAP.fit`, and
`UMAP.transform` entry points. The crate implements the search independently.

## Training

The training forest uses the tree count selected by UMAP and stops at
`max(60, 5 * k)` candidates per leaf for the tested widths. Candidate
sweeps join newly added directed links, including their reverse links,
against both new and still-retained links. A sweep that changes at most
`floor(n_rows * k / 1000)` directed slots ends refinement. Explicit
supplied-rank replay retains its existing fixed-round behavior.

The regression covers ten Euclidean, six correlation, and twelve cosine
configurations. It compares every ordered training index, distance, CSR
pointer, column, and weight. The cases include rows 59, 60, and 61, and
automatic approximate routing at 4,096 rows with widths 5 and 15. The
estimator routing test also fits 4,095 distinct rows and checks that a
4,096-row input reduced by deduplication takes the exact path.
Separate platform captures preserve the reference's graph weight bits,
which differ between Windows and Ubuntu even for equal ranked distances.

## Held-out queries

The query tree uses the retained training graph, including the
extra non-self offer that is absent from the public ranks. At each
node, select the three rows with the most incoming links in the
full training graph, break degree ties by row index, and choose the
most balanced of their three pairwise hyperplanes. The default
leaf size is 30. Queries visit the selected leaf first and expand
through the symmetrized diversified links with squared priorities.
The degree cap retained `ceil(1.5 * (k + 1))` eligible links in the
measured controls.
The tree-free, supplied-graph, and supplied-tree entry points are
also available.

This observation follows the published description of the search tree
and forward diversification. [PyNNDescent 0.7 performance notes](https://pynndescent.readthedocs.io/en/latest/performance_0_7.html#index-preparation)

The training cases also compare 32 held-out queries at width 1 and at
the fitted width, plus the resulting estimator COO order and weight
bits. A separate repeated-coordinate case verifies the prepared
heap's ordering at equal priorities. The cosine tests cover a zero norm,
an underflowed query norm, and an overflowed query norm. The angular
planes use uncentered unit pivot directions. Correlation ranks and walks
using centered distances. Cosine uses normalized input and log-distance
priorities. The training norm uses float32 division and the query norm
uses float32 multiplication by a rounded reciprocal.

The caller MT19937 stream advances through both three-word states and the
projection scalar. The Rust index returns that state for compatibility
layout initialization. Six reference fits cover a fresh state, a cached
Gaussian, and a cursor that crosses the twist boundary, each at zero and
two epochs. The tests compare coordinates and the entire final RNG state.

## Remaining numerical differences

The four 31-, 32-, 33-, and 64-feature cosine cases use bounded numerical
checks for distances and weights, with exact ordered ranks, CSR support,
and COO order. The measured differences agree on Windows and Ubuntu.

| Array | Differing values | Maximum difference |
| --- | ---: | ---: |
| Training rank distances | 11 / 3,195 | 1 ULP |
| Held-out rank distances | 38 / 768 | 4 ULP |
| Fitted CSR weights | 26 / 3,254 | 16 ULP |
| Held-out COO weights | 18 / 640 | 1 ULP |

The fixture retains each reference float32 value for reproducing these
differences. Public one-coordinate probes matched the normalized training
rows involved in the residuals. The remaining differences appear when
ranking with a multi-coordinate cosine dot product. All other fixture
cases compare float32 bits directly.

The 127-row, 31-feature case isolates the training difference to the
reciprocal pair of rows 50 and 87 with zero-based indices. The fixture
uses NumPy RandomState seed 38 for these features and index seed 43.
The normalized rows match the one-coordinate probes, while the fitted
pair yields a different float32 distance after the multi-coordinate dot.
Projection ties in the training graph and other degree-cap configurations
remain outside the captured scope.

## Reproduction

```sh
uv run --project reference --locked python -I reference/generate_approximate_training_fixture.py --check
python -I -m pytest -q tests/test_approximate_training.py
cargo test --release --locked -p umap-core --test approximate_neighbors --test approximate_query
```
