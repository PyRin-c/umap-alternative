# Public boolean feature metrics

The fixture records `umap-learn==0.6.0` on the locked x86-64 Windows 11 and
Ubuntu 24.04.2 WSL stacks with CPython 3.13.12. It uses public `UMAP.fit_transform`,
`UMAP.transform`, the public `graph_` attribute, and named distance calls.
The generator records exception types and warning categories.
[Generator](generate_boolean_metric_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-boolean-metrics-x86_64.json) · [Model residuals](fixtures/umap-learn-0.6.0-boolean-model-residuals-x86_64.json)

## Distances and support

Let `a` count shared nonzero features, `b` and `c` count features nonzero on
only one side, `d` count features zero on both sides, and `n = a + b + c + d`.
Let `m = b + c`. These independent formulas replay the recorded estimator
graphs and queries on dense and canonical CSR features, where supported.

| Estimator metric | Observed distance | Tested default cutoff |
| --- | --- | --- |
| `hamming` | Number of numerically unequal coordinates divided by `n` | Matches uncut |
| `matching` | `m / n` | Matches uncut |
| `jaccard` | `m / (a + m)`, with zero for two empty supports | 1 |
| `dice` | `m / (2*a + m)`, with zero for two empty supports | 1 |
| `russellrao` | Zero for equal supports, otherwise `(n - a) / n` | Matches uncut |
| `rogerstanimoto` | `2*m / (n + m)` | Matches uncut |
| `sokalmichener` | `2*m / (n + m)` | Matches uncut |
| `sokalsneath` | `2*m / (a + 2*m)`, with zero for two empty supports | Matches uncut |
| `yule` | `2*b*c / (a*d + b*c)`, with zero when `b*c` is zero | Matches uncut up to 2 |
| `kulsinski` | Zero for equal supports, otherwise `(n + m - a) / (n + m)` | Matches uncut |

Signed nonzero values count as present. In the four-feature pair
`[-1, 2, 0, 0]` and `[1, -2, 0, 0]`, Hamming is `0.5` and matching is
`0`. The support-based distances in the table all return zero for that
same-support pair. Jaccard and Dice return `1` for disjoint nonempty
supports. Yule returns `2` when both sides have exclusive nonzero features
and there are no shared nonzeros. An empty support paired with a nonempty
support has Yule distance zero.

The pinned public names under `umap.distances` are the estimator names in
the table, with `rogers_tanimoto`, `sokal_michener`, and `sokal_sneath`
using underscores. Under `umap.sparse`, the corresponding `sparse_` names
take `(indices_a, data_a, indices_b, data_b)`. Hamming, matching,
Russell–Rao, Rogers–Tanimoto, Sokal–Michener, and Kulsinski also require
the feature width. No public `sparse_yule` name was found. The named
functions match the table on the tested dense arrays and CSR buffers that
store nonzero values only.

The stored-zero probe uses sorted CSR with unique columns. For the zero
row stored at column 0 and the row `[-1, 2, 0, 0]`, the named sparse
matching, Jaccard, and Dice calls return `0.25`, `0.5`, and `1/3`.
The estimator graphs replay using `0.5`, `1`, and `1` from logical
support. Eight named sparse metrics diverge from their estimator on this
fixture. Hamming continues to replay from numerical equality. For named
sparse Russell–Rao, even the pure stored-support formula leaves pair
differences. The estimator replays using logical support.

All nine supported stored-CSR fits preserve caller buffers. A copy with
the same stored entries returns `graph_` itself. Eliminating explicit zeros
from the query copy takes the held-out path.

The fixture also records SciPy's public pair functions for both numeric
and boolean arguments. For example, boolean SciPy Russell–Rao gives `0.5`
on the equal two-feature support above, and SciPy Dice on two zero rows
gives `nan`. UMAP reports zero in both cases.

## Ranking, cutoff, and layout

For square fit comparisons, the generator computes a candidate distance
matrix, selects ranks under each tested sort order, applies the cutoff,
and feeds the compact rows to public UMAP via `precomputed_knn`. For
queries it compares the ordered COO indices to the candidate order and
compares full COO records with the public precomputed transform. This
separates evidence about rank order from evidence about membership weights.

The equal-distance fixture distinguishes ascending training-index ties
from NumPy quicksort ties. Stable row ranking replays the fit CSR. NumPy
quicksort replays the held-out COO order in the pinned environment.
The `same_support` cases place a proper superset ahead of equal-support
rows and include two repeated empty rows. These cases distinguish the
Russell–Rao and Kulsinski equal-support shortcut from always evaluating
their quotient during fit as well as during transform.

The cutoff is inclusive. A cut query rank remains a stored COO entry with
zero weight and row and column zero. The `next64` and `next32` cases use
the first float64 and float32 values above the boundary. Hamming,
Jaccard, and Dice fit and query outputs still cut distance `1` when the
threshold is `1.0000000000000002`. Distance `1` survives at
`1.0000001192092896`. The Yule axis cases make the same distinction at
distance `2` in fit. This is evidence for a float32 cutoff comparison in
these cases. A cutoff of zero removes even zero-distance ranks.

The layout snapshots use random initialization with fixed curve, fit seed,
and transform seed. They include zero and nine epochs. The `isolated`
cases retain a single zero-support training row, which Jaccard and Dice
isolate under their default cutoff. The fixture records the NaN coordinate
bits and the held-out outputs. The alternate-format graph queries all match
the same-format COO records, including CSR queries after a dense Yule fit.

## Reading and reproducing the artifacts

Each `cases` entry supplies a `spec` and a `public` key into `observations`.
`inputs` gives the training matrix followed by the queries. Values are
stored as unsigned IEEE-754 words with a dtype and shape. CSR preserves
`indptr` and `indices`, and COO preserves `row` and `col` in returned order.
`models` lists the hypotheses with exact fit-graph, query-order, and
query-weight comparisons. Multiple matches leave that difference
underdetermined on that input. The residual file uses the same inputs and
parameters. Its `named_disagreements` entries refer to supported estimator
cases in the common fixture and identify which named-function comparisons
diverge.

The two hosts regenerate the same artifacts byte for byte. The common
fixture contains 107 graph cases and 42 embedding cases. Its records
cover 8,726 fitted CSR weights, 3,058 ordered query COO entries, 684
alternate-format COO entries, 832 fitted coordinates, and 336 held-out
coordinates. All 58 paired dense and CSR cases have identical output bits
in this comparison.

The residual observations include CSR Yule, which raises `ValueError`,
and `kulczynski1`, which raises `ValueError` in either format on the tested
features. The explicit-zero cutoff on a Jaccard embedding also raises
`ValueError` after exposing an empty CSR graph. Packed-byte bit metrics
lie outside this feature fixture.

Run from the repository root on either host with the locked reference
environment.

```sh
uv run --project reference --locked python reference/generate_boolean_metric_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-boolean-metrics-x86_64.json \
  --residual-output reference/fixtures/umap-learn-0.6.0-boolean-model-residuals-x86_64.json --check
```
