# Public UMAP sparse graph reference

The fixture records `umap-learn==0.6.0` through the public
`UMAP(transform_mode="graph").fit_transform(X)` API. The generator constructs
inputs with SciPy and captures the returned CSR matrix's public arrays.
[Generator](generate_sparse_graph_fixture.py) · [JSON fixture](fixtures/umap-learn-0.6.0-sparse-graph-windows-x86_64.json)

The JSON stores three base inputs as raw float32 CSR. Each case identifies its
base input, input variant, metric, and any parameter override. `graph` holds
the exact shape, `indptr`, `indices`, and unsigned float32 weight bits in CSR
storage order. When a case's CSR graph is identical to another case's graph,
`same_graph_as` names that case. For example, a raw CSR record can be rebuilt
with `csr_matrix((np.array(record["data_bits_uint32"], dtype=np.uint32).view(np.float32),
record["indices"], record["indptr"]), shape=record["shape"])`. The variant
constructors in the generator define the conversions and storage edits.

## Feature metrics and formats

The 12-by-8 feature input has 66 stored values. Row 2 is empty and rows 1 and
10 are identical. For the canonical CSR cosine graph, row 2 has columns
`[0, 1, 3]`, all with unit weight. The edges `(1, 10)` and `(10, 1)` also have
unit weight.

| Metric | CSR entries | Dense and CSR structure | Different weight bits |
| --- | ---: | --- | ---: |
| Euclidean | 50 | Identical | 10 |
| Manhattan | 48 | Identical | 0 |
| Chebyshev | 52 | Identical | 0 |
| Cosine | 50 | Identical | 24 |

Manhattan and Chebyshev have different neighbor support on this input. The
generator independently sums the absolute coordinate differences for Manhattan
and takes their maximum for Chebyshev. Feeding those float32 distances back
through dense `metric="precomputed"` matched all CSR arrays and weight bits
for each metric on this fixture.

| Sparse input | Construction | Observed graph relative to canonical CSR |
| --- | --- | --- |
| CSC | `csr.tocsc()` | Exact match for all four metrics |
| COO with duplicate coordinates | Store each feature twice at half its value | Exact match for all four metrics. SciPy's `split_coo.tocsr()` also reproduced the base CSR arrays. |
| CSR with reversed rows | Reverse the stored columns and values within each row | Exact match for all four metrics. The cosine fit sorted the supplied CSR in place, restoring the exact canonical CSR arrays. |
| CSR with duplicate indices | Store two consecutive half values for each column using the raw CSR constructor | Euclidean, Manhattan, and Chebyshev matched. Cosine had the same structure and 30 different weight bits. The supplied CSR still had all 132 stored entries after the cosine fit. |

For example, cosine edge `(0, 4)` has weight bits `0x3eee5192` on canonical CSR
and `0x35f27c70` on split-entry CSR. Public scikit-learn `cosine_distances` on each
of the two CSR inputs, followed by dense precomputed graph mode, reproduced
its corresponding graph exactly.

## Independent canonical CSR Euclidean replay

Merge the two sorted CSR rows with two pointers in increasing column order.
For a column present in both rows, subtract the float32 inputs after widening
them to float64. For a column present on one side only, use zero for the missing
coordinate. Square each difference in float64 and add it to a float64
accumulator. Round the squared sum to float32, then take a float32 square
root. The generator supplies the resulting dense distance matrix to public
UMAP in precomputed graph mode and compares the full CSR arrays and bits with
the direct sparse Euclidean fit.

| Input | Nearest distance bits matching public scikit-learn CSR | Graph weight bits matching direct sparse fit | Different distance / graph bits if the square root is computed in float64 first |
| --- | ---: | ---: | ---: |
| Feature probe | 48/48 | 50/50 | 3 / 10 |
| Smallest coordinator holdout | 48/48 | 52/52 | 4 / 8 |

The holdout is a separate 12-by-8 input generated with `RandomState(374)` and a
60% masking probability, with varying feature support across rows. The nearest
distance comparison uses the first four entries per row selected by a stable
sort of public `sklearn.metrics.pairwise.euclidean_distances(csr)` output,
including the diagonal among the zero-distance candidates. The diagnostic sort
defines which distances to check. UMAP selects its own neighbors in both graph
fits. On the feature probe, entry `(3, 8)` has distance bits `1082362016` with
the float32 square root and `1082362017` with the float64 square root followed
by a cast. Both square-root variants produced the same CSR structure on each
input. These checks establish exact replay for the recorded canonical inputs.

## Sparse precomputed distances

The 6-by-6 symmetric base CSR stores four positive distances per row and omits
the diagonal and one off-diagonal entry per row. These cases use `n_neighbors=3`.
Storing the zero diagonals explicitly changes the graph, and converting the
same base CSR to dense fills every absent entry with zero. The two asymmetric
variants distinguish changing the stored pattern from changing one stored value.

| Case | Edit to base input | Public result |
| --- | --- | --- |
| Sparse baseline | None | 20 graph entries |
| Dense | `csr.toarray()` | 16 graph entries |
| Explicit zero diagonal | Store six zero diagonal entries | 12 graph entries |
| Exactly three stored distances in some rows | Remove `(0, 2)` and `(2, 0)` | `ValueError` with `kth(=3) out of bounds (3)` |
| Too few stored distances | Also remove `(0, 4)` and `(4, 0)` | `ValueError` with `Some rows contain fewer than n_neighbors distances!` |
| Asymmetric support | Remove `(0, 2)` only | `ValueError` with `Sparse precomputed distance matrices should be symmetrical!` |
| Asymmetric values | Change `(0, 1)` from 1 to 3, leaving `(1, 0)` at 1 | 20 graph entries with the original CSR structure and changed weight bits |

Each precomputed case emitted the `UserWarning` about `inverse_transform`
being unavailable. Successes and errors are observations for these inputs under
the pinned versions.

## Reproducibility

Run from the repository root with the existing locked reference environment.

```powershell
uv run --project reference --locked python -I -B reference/generate_sparse_graph_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-sparse-graph-windows-x86_64.json
```

Both independent runs used CPython 3.13.12 on Windows 11 AMD64 (build 26200)
and verified all 13 installed distributions against `reference/uv.lock`.
The generator sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `NUMBA_NUM_THREADS` to `1` before numerical imports,
sets `n_jobs=1`, and gives each process a fresh temporary Numba cache.
The two processes produced byte-identical JSON.

SHA-256 `7f34ad6a771e765faa46b2861bca0168f1dcb194d2b4f16ba3a85fd83d6d94a7`
