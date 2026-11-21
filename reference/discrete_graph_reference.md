# Discrete target distances and graph attenuation

The pinned `umap-learn==0.6.0` wheel exposes `fast_intersection`,
`fast_metric_intersection`, and `discrete_metric_simplicial_set_intersection`.
The corpus captures the returned sparse matrix, the caller's storage after
the call, direct edge values, distances, and inferred parameters. The fixture
has 47 graph calls, 18 raw edge calls, 211 distance calls, and 8 parameter
calls. The two hosts generate the same common fixture.

The default graph operation treats a label of `-1` at either endpoint as
unknown and applies `exp(-unknown_dist)`. Distinct known labels use
`exp(-far_dist)` and matching labels keep their weight. The explicit
`"categorical"` metric compares `-1` as an ordinary label. The named route
uses the values of `metric_kws` in insertion order as positional arguments.
It scales the distance before computing the exponential. Two float32
operands use a float32 exponential. The default float64 parameters use a
float64 exponential and round the final result into the input weight dtype.
The raw metric helper accepts a callable. If it raises, the weights for
earlier edges have already been updated.

The graph helper operates on the input's COO conversion, eliminates zeros,
then calls `reset_local_connectivity`. COO inputs retain their stored
order and repeated coordinates before reset. With CSR, the COO conversion
can share the data buffer, so the caller observes the adjusted values
while its stored zeros and structure remain. CSC conversions use separate
data. The rectangular control captures mutation followed by a shape error
during reset.

The six discrete names are `categorical`, `hierarchical_categorical`,
`ordinal`, `count`, `string`, and `myers`. The graph helper also accepts the
implemented Euclidean, Manhattan, Chebyshev and Hamming names and aliases.
The wider continuous distance registry remains part of the compatibility
work. `get_discrete_params` reproduces the tested ordinal range, Poisson
mean and normalization, and string length estimate. Public Numba typing
errors from invalid argument signatures correspond to Python `TypeError`.

The count distance folds an ordered Poisson recurrence. The initial
logarithmic sum uses the small integer rule below ten and the observed
Stirling expression from ten onward. The byte fixture tests large counts,
nonintegral counts, signed values, and zero normalization. Eight cases use
host-specific observations. Windows and Ubuntu differ by one ULP in a
finite float64 count-weighted graph and by the quiet-NaN sign in invalid
logarithms. The negative float32 logarithm agrees across the two hosts.

The edit functions count Unicode code points. `string` stops when every
prefix in a row exceeds the bound. `myers` caps the short bit-vector result
and takes the row algorithm when either string reaches 64 code points.
The corpus checks 63/64/65, both input orders, supplementary Unicode,
combining characters, empty strings, and a bound of zero.

Check from the repository root in the locked reference environment. Set
`host` to `windows` or `ubuntu` on the corresponding system.

```sh
uv run --project reference --locked python -I reference/generate_discrete_graph_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-discrete-graph-x86_64.json \
  --host-output reference/fixtures/umap-learn-0.6.0-discrete-graph-${host}-x86_64.json --check
```
