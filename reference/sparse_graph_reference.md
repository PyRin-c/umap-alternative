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

## Python estimator coverage

`UMAP(transform_mode="graph").fit` and `fit_transform` use the exhaustive Rust
CSR feature path for unsupervised Euclidean, Manhattan, Chebyshev, and cosine fits.
The Python tests match the fixture's CSR shape, pointer and column order, and
every float32 weight bit for 20 feature cases and two additional holdouts.

CSR, CSC, and COO matrices and arrays are validated before conversion and
converted without changing the original values, coordinates, pointers, or
storage order. For Euclidean, Manhattan, and Chebyshev, columns are sorted
and duplicates are summed. Compressed inputs are cast to float32 before
summation. COO conversion
sums in the original dtype before casting to float32. A public API probe using
float64 duplicate values `[1e8, 1, -1e8]` at the same coordinate produced the
same graph as a single zero there for CSR and CSC, and as a single one for COO.
Malformed storage and nonfinite normalized values raise `ValueError`.

Cosine preserves repeated entries in compressed rows when sorting. Each
stored value contributes its float32 square to a float64 norm sum. After
the float64 square root and division, the normalized values round to
float32. The dot product accumulates float32 products over the common
columns, including each pair of entries when a column repeats. Off-diagonal
distances are `1 - dot`, clipped to `[0, 2]`, and the diagonal is zero. The
Rust tests compare canonical CSR, repeated CSR, and a second canonical
input. The Python tests exercise CSR, CSC, and COO conversion.

Empty rows are allowed. A singleton `fit` has no `graph_`, and its
`fit_transform` raises `AttributeError`. The Rust search ranks ties by
ascending row index and requires each row's selected neighbors to include
itself. An all-zero 4-by-2 matrix with `n_neighbors=2` therefore raises
`ValueError`. Public UMAP 0.6.0 returned a graph with 10 unit-weight entries
on this input. Three identical rows among four with `n_neighbors=2` also
raise `ValueError` when the third row loses its self neighbor.

Sparse precomputed distances use a separate path. [Precomputed distance comparison](transform_reference.md#csr-precomputed-distances)

## Sparse Euclidean compatibility layout

Seeded Euclidean embeddings accept CSR, CSC, and COO features. The estimator
uses the existing CSR normalization on a copy and shares the Rust row
distance calculation, fuzzy graph construction, and seeded optimizer.
Random and explicit initialization match from epochs 0 and 1 through epochs
2, 3, 6, 11, and 30 on the twelve-row fixture. Both also match the
`[0, 2, 6]` checkpoints. Graph and coordinate tests cover an empty training
row, a stored value at feature index 4096, all-zero rows with and without
stored zeros, and duplicate coordinates. The float64 cancellation case
keeps the conversion order already used for CSR, CSC, and COO graph fits.

The embedding graph uses the first effective number of ranks, including
zero-distance ties that omit self. The two-row all-zero case truncates to
one neighbor and each row has one edge in the resulting symmetric graph.
Graph-only fits retain the self-neighbor validation described above.
The cutoff is inclusive and the warning counts cut distances over all
ordered row pairs. At cutoff 0.5 the eight-row fixture cuts 44 distances
and row 7 has NaN coordinates. At cutoff zero all 64 distances are cut
and the public fit raises `ValueError`.

The fitted-input shortcut compares the sorted, validated CSR storage
captured at fit time. Explicit zeros and float32 duplicates count toward
this comparison. Dense queries for sparse-trained models are converted
to CSR. Float64 fitted-input queries and `fit_transform` calls return
float64 coordinates and leave the stored fit at float32 for multisample
inputs. Public UMAP sorts unsorted float32 CSR training input in place.
This estimator sorts a copy and preserves caller storage.

The existing spectral initialization matches the connected fixture
at epochs zero and six and the two-component cutoff fixture at epoch six.
The path with six components uses the same centroid calculation with
sparse row means flattened to one feature vector per component. Its
coordinate bits vary by host and are recorded separately. All other
fixture data matched byte for byte on Windows and Ubuntu WSL x86-64 with
CPython 3.13.12 and the locked reference dependencies.

Each host checks 42 fits. The comparisons cover 40 graphs with 1,530
weights, 814 stored coordinates, 814 returned coordinates, and 144
checkpoint coordinates, bit for bit. The fixture has 56 fitted-input probes. Of those, 32
return the fitted coordinates, including 22 that return the same object.
The other 24 take the public held-out path.

Both implementations raise `ValueError` for inverse transform with sparse
features. Sparse density embeddings remain outside this path. The public base
input fits densMAP and produces a cosine embedding. Public CSR-array training raised `TypingError` in the
probe. This estimator continues to normalize SciPy sparse arrays using
the same CSR matrix path. Approximate neighbor search, large tie sets,
modern layouts, and supervised sparse embeddings were outside the
verified scope.

Regenerate the common fixture and the spectral record for the current
host with the locked reference environment. Use `windows` or `linux` for
`<host>`.

```sh
uv run --project reference --locked python -I reference/generate_sparse_euclidean_embedding_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-sparse-euclidean-embedding-x86_64.json \
  --spectral-output reference/fixtures/umap-learn-0.6.0-sparse-euclidean-spectral-<host>-x86_64.json --check
```

## Held-out sparse Euclidean transform

After a sparse Euclidean fit, dense, CSR, CSC, and COO queries normalize to
CSR. The estimator compares the validated storage with the fitted snapshot
before merging duplicates. Exact matches return the fitted graph or
embedding. Float64 embedding queries return float64 coordinates. Compressed
and COO duplicates follow the same dtype conversion order as fitting.
The fixture includes empty rows, a zero-row input error, unsorted storage,
and cancellation among repeated float64 coordinates.

Rust merges the stored feature coordinates and selects the nearest ranks.
It keeps one distance row of scratch space and passes only one column index
and distance per selected rank to graph construction. The NumPy tie callback
sees a distance row only when the selected order or boundary involves a tie.
Query memberships use all selected distances before applying the inclusive
cutoff. Each cut rank retains a zero COO entry at (0, 0). The fixture uses
both the exact distance 0.5 and the next larger float32 cutoff. The cutoff
comes from the fit and survives changes to the constructor parameter.

The shared serial optimizer keeps fitted coordinates fixed. Transform
budgets resulting from fit epoch settings 0 through 5 return initialization.
Settings 0 through 2 emit the public divide warning, with an invalid
division warning if the query graph has zero weight. The fixture checks the
default budget and positive budgets with endpoint seeds 0 and 4294967295.
An entirely cut query has NaN coordinates. A query that selects
an isolated fitted point with unit membership also propagates NaNs.

For a narrower query, public results matched training restricted to the
leading query-width columns with dense metric rounding. The native merge
shares the dense float32 product calculation for that case. The
`residuals.wider_query` record preserves a public 3-to-4-column reproduction
whose graph differed from both zero padding and dropping the extra column.
This implementation raises `ValueError` for that width. The
`residuals.graph_only_ties` record preserves a legacy graph-only fit with
four fitted weight differences on a 17-row tie-heavy input. Its held-out COO
order and weights match exactly. Approximate sparse search remains outside
the verified scope.

The locked fixture regenerates byte for byte on Windows and Ubuntu WSL
x86-64 with CPython 3.13.12. The 84 transform calls check 403 COO entries,
400 coordinates, 14 fitted-input returns, and four validation errors.
The graph-only tie case also checks 12 held-out COO entries. [Generator](generate_sparse_euclidean_transform_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-sparse-euclidean-transform-x86_64.json)

```sh
uv run --project reference --locked python -I reference/generate_sparse_euclidean_transform_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-sparse-euclidean-transform-x86_64.json --check
```

## Sparse Manhattan and Chebyshev

The compatibility layout and held-out transform accept sparse Manhattan and
Chebyshev features. They share the CSR merge, ranked graph construction,
fitted-input shortcut, and seeded layout with the sparse Euclidean path.
The fixture compares fitted CSR graphs, graph-mode COO order and weights,
and embedding coordinates from fresh random and spectral fits. It exercises
dense and CSR, CSC, and COO queries, split and cancelling duplicates, empty
and zero rows, ties, and finite disconnection cutoffs.

The precision probes start with sparse float32 buffers. Let `e = 2**-25`.

- Manhattan rounds each coordinate subtraction to float32 and sums the
  absolute differences in float64. For the `terms` probe against a query
  of `[-e, 0, 0, 0]`, the first difference rounds to `1` and the remaining
  terms give `1 + 2*e`. The sparse fit keeps the edge at cutoff
  `1 + 2.5*e`. Widening the subtraction would give `1 + 3*e`.
- Manhattan fit selection and its inclusive cutoff use the wide sum. The
  `summed_fit` probe cuts an edge of distance `1 + 3*e` at equality and
  keeps it at `1 + 3.5*e`. The same sum rounds to `1 + 4*e` in transform,
  so the query edge at that cutoff is masked. The `collision` probe checks
  that matching-width ranks are selected before that rounding. The held-out
  order starts with columns 1 and 0 even though their membership distances
  both round to 1. The narrower query ranks in float32 and returns 0 and 1.
- Chebyshev takes the maximum absolute coordinate difference. In `maxed`,
  the first feature dominates. A difference of `1 + 3*e` rounds to
  `1 + 4*e` and is cut there. The half-ULP difference `1 + 2*e` rounds to
  1 and keeps its COO entry. The fit cutoff also uses the rounded maximum.

Public Manhattan fitting coalesced the float32 CSR with stored duplicates
in place. The mutated object returned the fitted embedding on transform.
Ordinary copies of the original or coalesced buffers took the held-out path.
This implementation preserves the caller's buffers and takes the held-out
path for those duplicate fits. Chebyshev preserves validated stored
duplicates for the fitted-input check. Both metrics use canonical CSR
distance processing for held-out queries.
Sparse cosine keeps graph-only support and its separate rule for repeated
compressed entries. Wider queries and approximate sparse search have the
same scope limits as the sparse Euclidean transform.

The pinned public fixture regenerates byte for byte on Windows and Ubuntu
WSL x86-64. It checks 37
fits with 1,132 CSR weights and 420 fitted coordinates, plus 78 transform
calls with 478 ordered COO entries, 382 returned coordinates, and five
fitted-input returns.
[Generator](generate_sparse_lp_transform_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-sparse-lp-transform-x86_64.json)

```sh
uv run --project reference --locked python -I reference/generate_sparse_lp_transform_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-sparse-lp-transform-x86_64.json --check
```

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

SHA-256 `c7341365017d89bc442943b402038963bc4e2fd1414a54bac736cee35c6a2417`
