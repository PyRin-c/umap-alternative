# Public trustworthiness curves

`umap_alternative.validation` implements the three public callables from
`umap.validation` in the pinned 0.6.0 wheel. The graph and layout code do not
depend on these diagnostics. The reference generator calls the public wheel
and records float64 scores and exception and warning categories.

## Ranking and normalization

`trustworthiness_vector(source, embedding, max_k, metric="euclidean")`
queries a scikit-learn KDTree for `max_k + 1` results per embedded row and
drops its first result. Both spaces use the requested metric. The accepted
metric names on the pinned stack are `euclidean`, `l2`, `minkowski` (with
power two), `manhattan`, `l1`, and `chebyshev`. In the source space, ties
retain input order up to 15 samples and are affected by a median-pivot
partition at 16 or more samples. The source distance row is stored in
float64. With float32 source values, Euclidean differences and squares and
Manhattan differences round to float32 before accumulation into float64.
Unsigned differences wrap before the Manhattan and Chebyshev reductions.

`make_trustworthiness_calculator(metric)` takes a callable and returns a
scorer that accepts the source coordinates and already ranked embedded
neighbors. It retains double-precision differences returned by a callable.
Both coordinate entry points return `1.0` at index zero.

`trustworthiness_vector_bulk(indices_source, indices_embedded, max_k)`
starts with `0.0`. The first occurrence of an embedded identifier supplies
its zero-based source rank. Its contribution at neighborhood size `k` is
`max(source_rank - k, 0)` provided it appears among the first `k` embedded
neighbors. The score is one minus the summed penalty times
`2 / (n_samples * k * (2 * n_samples - 3 * k - 1))`. The Rust implementation
adds each penalty across its contributing interval, then recovers all
scores with a single prefix pass. It streams source distance rows through
the ranker, using linear distance and sorting scratch space.

The public normalization can produce scores greater than one for large
valid neighbor counts and raises `ZeroDivisionError` at zero denominators.
Invalid ranks that would access storage out of bounds are checked by the
native implementation. The Python factory accepts ordinary Python distance
functions. It uses `TypeError` for unsupported types in place of errors
from the public wheel's Numba typing stage.

## Reproduction

The fixed corpus has 85 calls. The installed wheels match all 436 scores
from 70 successful calls and all 15 exception cases on both pinned hosts.
The corpus covers tied and repeated points, the
15/16/17 sample and 80/81 sample boundaries, zero and maximum query widths,
float32 and float64 source distances, unsigned and signed integers, zero
features, nonfinite values, and signed identifiers in the ranked API. The
six 81- and 127-row cases use platform-specific outputs. In the pinned
environment the query orders differ for those tied inputs across Windows
and Ubuntu. The common fixture uses the same source and embedding bytes
for both platforms.

Run from the repository root, setting `platform` to `windows` or `ubuntu`
on the corresponding host. Use the locked CPython 3.13.12 reference
environment and set `NUMBA_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`OMP_NUM_THREADS` to `1`.

```sh
uv run --project reference --locked python reference/generate_trustworthiness_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-trustworthiness-x86_64.json \
  --platform-output reference/fixtures/umap-learn-0.6.0-trustworthiness-${platform}-x86_64.json \
  --check
```

## Windows timings

Measured on an Intel Core Ultra 9 285K with numerical thread counts set to
one. The inputs were float32, with 12 source features, two embedded
features, and `max_k=20`. Both spaces were generated with consecutive
normal draws from `RandomState(147)`. The medians are over seven calls
after warmup, including conversion and queries in both vector columns.

| Samples | Public vector | Public prebuilt calculator | Rust-backed vector |
| ---: | ---: | ---: | ---: |
| 512 | 734.341 ms | 11.382 ms | 9.567 ms |
| 2,048 | 885.197 ms | 216.794 ms | 167.727 ms |

The prebuilt column calls the same public calculator repeatedly with
precomputed embedded ranks. It isolates the reusable reference scoring
path from the overhead observed in the public vector calls. All score
bits match across the three columns at both sizes.
