# Dense approximate neighbors

The locked `umap-learn==0.6.0` wheel was exercised through public `UMAP`
`fit_transform` and `transform` calls in graph mode. The probes use
`compatibility_layout=True`, `init="random"`, 30 epochs, `n_jobs=1`,
`random_state=42`, `transform_seed=42`, and one thread for the numerical
libraries. Gaussian inputs have seven float32 features and four held-out rows.
The NumPy training and query seeds are 57 and 62, as in the
[dense angular probe](dense_angular_reference.md). Larger Gaussian inputs
extend the same training prefix. Each mode and handoff was repeated on two
fresh fits on Windows and Ubuntu WSL, with agreement on both hosts outside
the forced `k=2` repeated-vector case. That case uses 40 fresh fits. The
harness records API exception types and the first message line if a call
fails.

The fixtures save full ordered held-out COO coordinates and float32 weight
bits. For training CSR they save shape, stored count, isolated rows, leading
rows, and SHA-256 fingerprints. The support fingerprint concatenates shape,
row pointers and column indices as little-endian `uint32`. The weights are
hashed separately in CSR order as little-endian float32 bit patterns. The
comparison counts weight-bit differences and the largest finite absolute
and ULP differences on common nonzero coordinates. Held-out comparisons
also count ordered and multiset coordinate differences, including stored zeros.

## Automatic routing

With `k=5`, `force_approximation_algorithm=False` and `True` first agree at
4,096 rows for all three tested metrics. Both fit and held-out outputs differ
at 4,095. The Euclidean controls at 3,999 and 4,000 also differ, and the
4,097 control agrees throughout. At 4,096 rows an independent exhaustive
rank calculation finds 8 of 20 Euclidean, 4 of 20 cosine, and 3 of 20
correlation held-out columns beyond the exact fifth-neighbor distance in both
modes. The matching modes above the switch therefore still show approximate
query behavior.

| Training input | Metric | Automatic-only and forced-only CSR coordinates | Changed bits on common CSR coordinates, Windows / WSL | Automatic-only and forced-only nonzero COO coordinates |
| --- | --- | ---: | ---: | ---: |
| 3,999 | Euclidean | 416 / 398 | 7,768 / 7,766 of 22,654 | 5 / 5 |
| 4,000 | Euclidean | 392 / 358 | 7,714 / 7,712 of 22,690 | 4 / 4 |
| 4,095 | Euclidean | 454 / 436 | 8,066 / 8,064 of 23,160 | 7 / 7 |
| 4,095 | Cosine | 238 / 254 | 14,992 / 14,990 of 20,214 | 3 / 3 |
| 4,095 | Correlation | 700 / 826 | 1,744 / 1,744 of 19,478 | 5 / 5 |
| 4,096 | All three | 0 / 0 | 0 / 0 | 0 / 0 |
| 4,097 | Euclidean | 0 / 0 | 0 / 0 | 0 / 0 |

On common coordinates at 4,095 rows, the largest CSR weight differences were
0.555001 for Euclidean, 0.560422 for cosine, and 0.813469 for correlation.
The corresponding COO differences were 0.038798, 0.038276, and 0.137900.
At 4,097 input rows with `unique=True`, the routing follows the number of
distinct rows. The 4,095-distinct case has 376 automatic-only and 356
forced-only training coordinates. The 4,096-distinct case has identical
CSR and ordered COO in both modes. CSR shape uses the distinct count and
COO shape uses the original input count. The fixtures compare COO as returned
for these cases because the column mapping after deduplication is unverified.

## Small fits and ranking boundaries

| 96 × 7 Gaussian input | Changed CSR bits / common coordinates | Automatic-only and forced-only nonzero COO coordinates | Changed COO bits / common coordinates |
| --- | ---: | ---: | ---: |
| Euclidean, k=2 | 0 / 146 | 3 / 3 | 0 / 5 |
| Euclidean, k=5 | 136 / 578 | 0 / 0 | 2 / 20 |
| Cosine, k=2 | 0 / 136 | 0 / 0 | 0 / 8 |
| Cosine, k=5 | 324 / 486 | 3 / 3 | 10 / 17 |
| Correlation, k=2 | 0 / 128 | 2 / 2 | 0 / 6 |
| Correlation, k=5 | 0 / 470 | 2 / 2 | 3 / 18 |

Training support is identical within each Gaussian pair at 96 rows. The
largest shared training weight difference at `k=5` is 35 ULPs for Euclidean
and 312 ULPs for cosine. Cosine held-out weights differ by up to 0.126229 on
common coordinates. At `k=2`, Euclidean and correlation query differences
include columns strictly beyond the exhaustive rank boundary.

- With repeated training vectors at `k=2`, forcing approximation drops six
  directed CSR coordinates, all within two groups of three identical rows.
  The first group's automatic graph is the complete triangle. The forced
  graph joins rows 0 and 1 through row 2. The first held-out query chooses
  `[0, 1]` automatically and `[0, 2]` when forced. At `k=5`, ten automatic-only
  and six forced-only CSR coordinates remain. Three held-out columns are
  exchanged in each direction at both values of `k`. All retained query
  columns are within the exhaustive rank boundary, and every common
  membership bit agrees. The order of the three zero-distance candidates
  of query zero at `k=5` changes from `[0, 1, 2]` to `[1, 2, 0]`.
  In the forced `k=2` repeat probe, 11 of 40 Windows fits produced NaNs at
  `(0, 0)` and `(0, 2)` in place of `1.0`. All 40 WSL fits returned `1.0`
  at both zero-distance coordinates.
  The other six query weights, COO coordinates, and all CSR bits agree.
  The fixture retains the observed variants, uses the first finite output
  for the force-mode comparison, and checks both varying slots for either
  the recorded finite bits or two NaNs during reproduction.
- The one-axis cutoff case contains training values immediately below, at,
  and immediately above float32 `1.0`. At `k=7` and cutoff `1.0`, the first
  query keeps the predecessor of `1.0` and cuts the two other values. Both
  modes return 36 CSR entries, 88 isolated training rows, and seven nonzero
  query memberships plus seven stored `(0, 0, +0)` entries. All surviving
  query bits match their uncut versions. With the cutoff disabled, CSR has
  six automatic-only and four forced-only coordinates. The queries keep the
  same support and weight bits, with one equal-distance pair swapping order.
- On the five-row one-axis input `[0, 1, 4, 10, 18]`, the automatic graph and
  held-out COO match across requests for 4, 5, and 8 neighbors, using four
  query slots. The forced training graphs match public supplied-neighbor
  replays with four ranked training distances per row, including self.
  The replays pad with `(-1, +inf)` to reach the requested count and use
  that count for fuzzy scaling. All retain 18 CSR entries. Forced queries
  use the requested width of 4, 5, or 8. The width-eight case stores five
  memberships and three missing slots per query. For all three requests,
  the sum of query weights after the first rank is within `8e-6` of
  `log2(4)`. Training and query scale targets are separate quantities.
  The additional control with five supplied ranks and a request for four
  raised `TypingError` on both reference hosts.

## Rust handoff

The controlled fixture passes a public `precomputed_knn` tuple with a public
`NNDescent` subclass that returns specified arrays from its `query` method.
The test double lives only in the reference generator. The estimator passes
`k` to that method and uses `epsilon=0.12` with the Euclidean index and
`epsilon=0.24` with the cosine index in these cases. Both reference hosts
returned identical results for all controlled inputs.

`umap_core::transform::transform_ranked_neighbors_graph` consumes the search's
flat int32 column and float32 distance arrays in nondecreasing distance
order within each row and returns the directed COO. The caller supplies the
query width, training and query counts, effective neighbor count for the
query scale, cutoff, and training graph options. Equal-distance candidates
retain their rank order. The solve skips the first rank and targets the
base-two logarithm of the effective count. The nine-slot replay on eight
training rows sums to `log2(7)` within `7e-6` after the first rank. Missing
distances are omitted from scale floors.
Finite distances at or above the cutoff participate in scaling before their
slots become stored zeros at `(0, 0)`. Missing slots use the same placeholder.
Rust replays the controlled shapes, coordinates and all 131 weight bits, including
fractional connectivity, ties at `k=2`, an all-missing row, and a width greater
than the training row count. The existing exact transform uses the same
membership stage after selecting exhaustive ranks.

Native search feeds its training ranks to
`fit_supplied_neighbors_graph` and held-out ranks to
`transform_ranked_neighbors_graph`. The paired fingerprints let it check
training support before investigating weight arithmetic. The held-out COO
and exhaustive rank residuals distinguish a different candidate set from a
tie-order change or a weight difference. The
[native approximate stage](approximate_search_native_reference.md) records
the candidate controls and compares a standalone Rust index to these fixtures.

## Reproduce

Use the pinned reference environment to verify the fixture for the current
host. Omit `--check` to regenerate it.

```sh
uv run --project reference --locked python -I reference/generate_approximate_neighbors_fixture.py --check
uv run --project reference --locked python -I reference/generate_approximate_neighbors_fixture.py --handoff --check
cargo test --locked -j 2 -p umap-core --test ranked_query_graph
```
