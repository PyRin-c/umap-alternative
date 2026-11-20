# Dense supervised embedding against public UMAP 0.6.0

The reference generator uses the installed `umap-learn==0.6.0` estimator's
public `fit`, `fit_transform`, `transform` and `update` methods. It records
CSR offsets, columns and float32 membership bits, coordinate bits, ordered
warnings, exception types, and the state of a caller-supplied NumPy
`RandomState`. Windows x86-64 and Ubuntu x86-64 on WSL both used CPython
3.13.12, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1 and Numba 0.68.1.
[Generator](generate_supervised_embedding_fixture.py)

## Categorical and partially labeled targets

The estimator fits the [categorical graph](supervised_graph_reference.md),
then uses the existing dense layout. At 30 epochs, it zeros memberships
below the largest weight divided by 30. Those entries remain in `graph_`.
Random, spectral and explicit initialization feed the same Rust serial
optimizer. Spectral initialization uses the positive sampled graph to find
connected components. Isolated rows participate in initialization and have
NaN coordinates in the fitted embedding.

| Eight-row case at 30 epochs | CSR entries | Stored zeros |
| --- | ---: | ---: |
| Plain fit | 28 | 0 |
| Categorical, weight `0`, `nextafter(0, 1)` or `0.000001` | 28 | 0 |
| Categorical, weight `0.5` or `0.95` | 28 | 18 |
| Categorical, weight `0.98`, `nextafter(1, 0)` or `1` | 10 | 0 |
| Partial labels, weight `0.5` | 28 | 4 |
| Partial labels, weight `0.25` | 28 | 0 |
| Partial labels, weight `nextafter(1, 0)` or `1` | 20 | 0 |
| All labels `-1`, weight `0.5` | 28 | 0 |

At weight zero, known cross-class memberships are still attenuated. At
weight `0.95`, cross-class weights survive categorical graph construction
and become stored zeros at layout pruning. At `0.98` they underflow during
graph construction. The partial-label graph has 28 positive memberships
at zero epochs and 20 positive memberships in the 11-epoch spectral fit.

Across 37 cases per host, both project fit methods matched all 36 CSR
graphs and all 728 stored membership bits. The singleton produced the
recorded float64 zero `embedding_` and float32 `fit_transform` output.
Thirty-one repeatable embeddings matched all 462 coordinate bits on both
hosts, including the explicit initialization, four epoch checkpoints and
categorical output-density diagnostics. Against the first public capture,
32 of 34 embeddings matched on each host, comprising 478/497 coordinates on
Windows and 465/497 on Ubuntu. The three spectral cases below varied across
six public fits while their CSR bits and RNG state stayed identical.
[Windows fixture](fixtures/umap-learn-0.6.0-supervised-embedding-windows-x86_64.json)
[Ubuntu fixture](fixtures/umap-learn-0.6.0-supervised-embedding-ubuntu-x86_64.json)

| Spectral case | Distinct public layouts, Windows / Ubuntu | Changed coordinates against first capture, Windows / Ubuntu |
| --- | ---: | ---: |
| Full labels, weight `0.5` | 3 / 3 | 0/16 / 16/16 |
| Full labels, weight `1` | 2 / 2 | 16/16 / 16/16 |
| Three rows, partial labels, one component | 2 / 2 | 3/3 / 0/3 |

The project matched the RNG checkpoint at the layout handoff for seeded
embedding cases that exercised it. The public categorical output-density
probe advanced its cursor to 43. The project layout stopped at 35 and
matched both radius arrays. An additional categorical densMAP probe gave
matching original radii with an embedded log-radius difference up to
approximately `1.22e-4`.

## Continuous targets

Independent feature-only fits and target-weight controls supported a
directed target graph with local connectivity one. The blended graph uses
the union of the feature and directed target supports. A missing feature
membership uses half the smallest positive feature membership, floored at
`1e-8`. A missing target membership uses `1e-4`. On this directed support,
below `0.5` the product is `feature * target**(weight / (1 - weight))`.
From `0.5` onward it is `feature**((1 - weight) / weight) * target`.
Calculate the product in float64, round to float32, divide each row by its
maximum, and take the fuzzy union with the transpose. Both weight
endpoints retain the union support. The Euclidean scalar probe contributes
24 directed target edges and produces 38 stored entries from a feature CSR
with 28. Setting `target_n_neighbors=2` produces 30 stored entries, and
using all eight rows produces 56. A continuous target of `-1` is an
ordinary numeric value.

The Rust stage accepts Euclidean, Manhattan and Chebyshev targets. The
fixtures cover scalar and vector labels, weight endpoints, values on both
sides of `0.5`, different feature and target neighborhood widths, a small
feature membership and an independent 11-row holdout. Both fit methods
matched the structure of all 36 emitted CSR graphs on each host. They
matched membership bits in 34 graphs and 1,388 of 1,420 stored entries.
Eight of nine embeddings and 116 of 138 coordinate bits matched on each
host, including active random and spectral layouts at 19 or 30 epochs.
The tests use bitwise checks for the repeatable cases and explicit graph
tolerances for the two holdouts below.
[Windows fixture](fixtures/umap-learn-0.6.0-supervised-continuous-windows-x86_64.json)
[Ubuntu fixture](fixtures/umap-learn-0.6.0-supervised-continuous-ubuntu-x86_64.json)

| Holdout | Changed stored weights | Max weight difference | Changed coordinates | Max coordinate difference |
| --- | ---: | ---: | ---: | ---: |
| Euclidean target, spectral | 16/92 | `1.102686e-6` | 22/22 | `4.098988` |
| Manhattan target, random | 0/86 | `0` | 0/22 | `0` |
| Chebyshev target, random | 16/86 | `6.529689e-5` | 0/22 | `0` |

Feature-only and weight-zero controls for those holdouts matched exactly.
A separate wide Euclidean distance reconstruction reduced the graph-mode
holdout difference to two stored weights at the tested weight `0.37`. The
remaining distance and layout arithmetic is unresolved.

The scalar cosine controls matched a candidate graph reconstruction.
Vector cosine weight `0.5` and `1` differed in 20/38 and 14/38 stored weights,
with maximum absolute differences of `1.490116e-6` and `8.344650e-7`.
Scalar ordinal returned 28, 28 and zero entries at weights `0`, `0.5` and
`1`. Vector ordinal raised `TypingError` with JIT and `ValueError` in the
JIT-disabled Windows diagnostic. Cosine and ordinal targets remain
unsupported. Continuous density fitting also raises an explicit error
because the wheel's original-space radius uses sparse feature-distance
support and the existing density path uses dense distances.

## Warnings and estimator state

- The categorical run matched all seven warnings in six cases on each
  host, including the seeded job override and neighbor truncation. The
  continuous run matched five warnings in three cases within the tested
  scope. A two-row feature graph warns that both vertices are disconnected,
  then raises `ValueError` with an empty `graph_` on a fresh estimator.
- Categorical attenuation to an empty graph raises `ValueError` during
  embedding without a disconnection warning from the target graph. Partial
  isolation produces NaN coordinates and leaves the other rows usable.
- The single-row fit bypasses target length validation. The project clears
  the previous graph when fitting a singleton and keeps a previous
  successful fit intact after a failed supervised refit. The public wheel
  retained the previous graph on singleton refit and exposed a new empty
  graph alongside the old embedding after a failed refit.
- Exact-input `transform` returns the stored fitted output, and `update`
  raises `ValueError` after successful supervised embedding fits. A
  successful plain refit clears the supervision flag and ignores target
  settings. For a continuous target, the project rejects a target neighbor
  count above the number of rows with `ValueError`. The wheel probe raised
  `SystemError` and exposed its feature graph.

Rust workspace tests passed on both hosts with 171 passed and one ignored.
Both installed project wheels passed all 1,060 Python tests. The tests
used local caches and two Cargo build jobs on each host.

## Regeneration

Use the locked reference environment and choose the fixture filename for
the current platform.

```sh
uv run --project reference --locked python -B reference/generate_supervised_embedding_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-supervised-embedding-ubuntu-x86_64.json
uv run --project reference --locked python -B reference/generate_supervised_embedding_fixture.py \
  --continuous-only --output reference/fixtures/umap-learn-0.6.0-supervised-continuous-ubuntu-x86_64.json
```
