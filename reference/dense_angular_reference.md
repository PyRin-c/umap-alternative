# Dense cosine and correlation

The reference harness calls the public `UMAP` estimator from the locked
`umap-learn==0.6.0` wheel. It records graph-mode `fit_transform` and held-out
`transform` using `compatibility_layout=True`, random initialization,
`random_state=42`, `n_jobs=1`, and 30 epochs. Fits are repeated with fresh
inputs. API failures are recorded as exception type and message. Fit records
include shape, CSR row pointers and columns, and float32 weight bits. Query
records include shape, ordered COO coordinates, and float32 weight bits.

The independent calculator supplies distances back through the public
`metric="precomputed"` graph fit. Across the ordinary cases, the graph bits
support these distance formulas.

- Cosine rounds individual coordinate products to float32 and sums in float64.
  Correlation computes the row means, centered products, and sums in float64.
  Both divide the dot product by the square root of the product of the two
  squared norms, subtract from one, and convert to float32.
- Two zero norms yield distance zero. One zero norm yields distance one. For
  correlation a constant row has zero centered norm, including all one-feature
  rows. Using a product of separate square roots changed the graph weight bits
  in the constant and repeated-row probe.
- Near parallel cosine rows can round to a negative distance. With cutoff
  zero the signed-distance probe retains the two reciprocal CSR entries,
  while the public precomputed fit on the signed matrix raises
  `KeyError("'precomputed'")`. The native distance keeps this rounding.
- The default cutoff is inclusive at two. In the sign-reversal probe, four
  fit entries remain under the default and ten under explicit infinity.
  A held-out graph uses the selected distances for scale estimation and
  emits zero at COO coordinate `(0, 0)` for a cut entry.
- The search ranks ties by ascending row index. A training row can fall
  outside its own zero-distance top-k, so memberships skip the diagonal by
  identity and the scale search skips the first ranked position. This covers
  the repeated-row and one-feature cases at `k=2`.

## Comparison with the port

On each platform, 57 fit cases match CSR shape, row pointers, columns, and all
2,510 weight bits. The reduction-width cases span 3 through 129 features and
bracket 4, 8, 16, 32, 64, and 128. The public Windows and Ubuntu fixtures
differ at two reciprocal cosine weights by one ULP at width 127. The port
matches the respective platform's values.

| Held-out case | Metric | Weight bits equal | COO difference using row-index tie order |
| --- | --- | ---: | --- |
| Generic, custom cutoff, graph options, sign reversal, signed rounding, correlation overflow probe, and width queries | Both where applicable | 254 of 254 | Exact |
| Zero, constant, and repeated rows | Cosine | 16 of 16 | Two tied columns swap places and one tie at the cutoff selects column 0 instead of column 1. |
| Zero, constant, and repeated rows | Correlation | 16 of 16 | Two tied pairs swap places. |
| Held-out ties | Cosine and correlation | 16 of 16 | One tied pair swaps places for each metric. |

Finite input values can overflow float32 cosine products. The extreme-value
probe returns 12 CSR and 6 COO NaN weights in the public model. The native
graph APIs reject selected non-finite distances.

## Embedding and held-out ranking

On Windows and Ubuntu WSL, 20 fixed-seed compatibility fits reproduce the
CSR graph and all 280 fitted and 140 transformed coordinate bits of the
host's public wheel. The cases cover zero and positive epochs, random,
explicit and spectral initialization, changing graph connectivity and mix,
and the two extreme transform seeds. Antipodal rows at distance 2 and
isolated rows exercise the inclusive cutoff and NaN coordinates. The
reference files differ between hosts for two tied fits and one connectivity
fit, so the regression uses the respective host's fixture. The standalone
cosine query also exercises the batch-dependent maximum membership used
for scheduling.

Python resolves ties within the selected ranks and at the selection
boundary using NumPy's ordering on the original float32 distance row.
The Rust search computes one row at a time and calls this ranking step only
when a tie can change the selected order. All angular graph-mode fixture
coordinates and weights now match in Python. A repeated-row sweep on
Windows with 7, 17, 33, 65 and 129 training rows agreed with NumPy argsort
for all 4,968 recorded query columns. The default native order used by C
still uses the training index for equal distances. The C11 smoke client
replays the explicit-layout public fit and a held-out query without ties
for both metrics.

Angular inverse transform, update and density are outside the current
embedding path. In a public probe with 96 training rows, 4 query
rows, 7 features and `k=5`, forcing approximate search kept the training CSR
support for both metrics but changed 324 of 486 cosine weight bits.
Held-out support differed at 6 cosine and 4 correlation coordinates.
Compatibility for that path requires reproducing the approximate training
search and the corresponding query index. The probe used standard-normal
float32 inputs from NumPy generators seeded with 57 for training and 62 for
queries.

## Reproduce

Run the generator through the locked reference environment on each platform.
Use the Windows or Ubuntu fixture suffix for the host being tested.

```sh
uv run --project reference --locked python -I reference/generate_dense_angular_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-dense-angular-windows-x86_64.json
uv run --project reference --locked python -I reference/generate_dense_angular_embedding_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-dense-angular-embedding-windows-x86_64.json --check
```
