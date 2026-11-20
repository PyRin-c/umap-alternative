# Public neighbor and graph stages

This fixture uses the public helpers of the pinned `umap-learn==0.6.0`
wheel. The port exposes the corresponding functions in
`umap_alternative.umap_` and `umap_alternative.utils`.

- `smooth_knn_dist` preserves input order and excludes slot zero from
  the scale search. It returns float32 scales and offsets with either
  float32 or float64 input. Finite positive distances determine local
  connectivity. The scale target is `log2(k) * bandwidth`, and the
  scale floors use finite distances. The fixture covers limited and zero
  iteration counts, fractional and oversized connectivity, missing
  values, signed distances, and empty axes.
- `compute_membership_strengths` preserves neighbor slot order and
  duplicate coordinates. A missing index of -1 leaves a zero placeholder.
  Self edges have zero weight in square graphs and are permitted in
  bipartite graphs. The float32 distance output is optional.
- `fuzzy_simplicial_set` accepts ranked indices and distances or a
  precomputed dense distance matrix. The directed output is COO.
  Fuzzy set operations produce CSR. A requested distance matrix is DOK
  and takes the maximum of the two directed distance sums. The function
  returns three values when `return_dists` is omitted and four when it
  is explicitly set.
- Precomputed `nearest_neighbors` uses stable row ranks and replaces
  selected positive infinities' indices with -1. `fast_knn_indices`
  and `submatrix` expose the corresponding sorting and gather stages.
  `average_nn_distance` returns float64 row means and warns for NaN
  means. `make_epochs_per_sample` preserves the sample count's
  intermediate precision before dividing in float64.
- `disconnected_vertices` tests the fitted graph's row sums, so a
  stored zero or cancelling pair can leave a row disconnected.
  With `unique=True`, the result uses the original fit's row map,
  including after an update expands the graph.

The 53-case fixture is shared by Windows and Ubuntu WSL x86-64. The
public helper tests also reuse 214 archived fuzzy graphs with their
offset and scale bits and the two host-specific curve fixtures.
Automatic search from raw features in the low-level helper remains
part of approximate-search verification.

## Reproduction

Run in the locked reference environment with Numba and build caches
under LocalAppData on Windows or the Linux user cache under WSL.

```text
uv run --project reference --locked python -I reference/generate_graph_primitives_fixture.py --output reference/fixtures/umap-learn-0.6.0-graph-primitives-x86_64.json --check
python -I -m pytest -q tests/test_graph_primitives.py tests/test_curve.py
```
