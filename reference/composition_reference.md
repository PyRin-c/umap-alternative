# Fitted UMAP graph composition

The public `umap-learn==0.6.0` wheel accepts `left + right`,
`left * right`, and `left - right` for fitted estimators with the same
number of vertices. The operators combine the fitted CSR graphs and fit a
new embedding over the shared vertex indices. They can combine estimators
trained on different feature matrices.

- Union and intersection walk the union of stored entries. An absent
  membership uses half the graph's minimum stored membership, with a floor
  of `1e-8`. Intersection caps the right fallback at `1e-4`. The probes
  distinguish an absent entry from an explicitly stored zero.
- Each nonempty row is divided by its maximum. Union and intersection
  then raise the positive weights to a common power targeting a row sum
  of `log2(15)`. Rows with fewer than four positive weights approach unit
  memberships. The last graph step applies a float32 fuzzy union with the
  transpose.
- Subtraction uses the support of the left graph, multiplies by the right
  complement, and applies the same maximum normalization and symmetrization
  without the power adjustment. The public operation writes the raw product
  into the left fitted graph before building the new graph. The port also
  preserves this observable mutation, including on an empty layout result.
- In the verified layout cases the compatibility optimizer uses seed 42,
  spectral initialization for connected graphs, and random initialization
  with a warning for disconnected graphs. It uses the smaller embedding
  width and learning rate, the largest supplied epoch count, and averages
  the fitted curve coefficients, repulsion strengths, and negative rates.
  Weak layout memberships can remain as stored zeros in the new graph.

The fixture records 27 CSR outputs and 566 stored weight bits, left and right
graph state after each operation, and 144 coordinate bits across nine zero-
and positive-epoch checkpoints. It includes asymmetric memberships, short
rows, values near the weak membership cutoff, disjoint support, and stored
zeros. The eleven-epoch boundary case removes a bridge after the combined
graph is tested for connectedness. The common fixture is reproducible on
Windows and Ubuntu WSL x86-64 with the pinned reference environment. The
bridge coordinates differ by host and use separate exact fixtures.

The output represents a combined graph and its embedding. It has no original
feature dataset for forward or inverse transforms. Broad checks for cases where
layout pruning changes the component count, very large connected graphs,
alternate user-supplied graph storage, and non-repeating spectral bases remain
open.

## Reproduction

Use the locked `reference` environment and place build products, environments,
and Numba caches under LocalAppData on Windows or the Linux user cache on WSL.
Use the matching bridge fixture name for the host.

```text
uv run --project reference --locked python -I reference/generate_composition_fixture.py --output reference/fixtures/umap-learn-0.6.0-composition-x86_64.json --bridge-output reference/fixtures/umap-learn-0.6.0-composition-bridge-windows-x86_64.json --check
cargo test --locked -p umap-core --test graph_composition
python -I -m pytest -q tests/test_composition.py
```
