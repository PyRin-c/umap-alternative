# Public supervision with packed features

The fixture records `fit` and `fit_transform` for packed Hamming and
Jaccard on the locked x86-64 `umap-learn==0.6.0` wheel. Windows and
Ubuntu WSL generated the same SHA-256
`253ecaa257a4e6e1006bc484b41fdebd61bf1d07f3baf158ee13faf4bf2f2525`.
The 154 cases include 116 successful fits, 22 `ValueError` results,
10 `KeyError` results and six `TypeError` results. The fixture records
152 CSR graphs, 210 ordered held-out COO graphs, and 12 fitted
embeddings. Warnings are recorded by category and numeric values
as IEEE-754 words.
[Generator](generate_packed_supervised_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-packed-supervised-x86_64.json)

The packed feature graph is built from `uint8` directly. It is
published with the feature cutoff applied, before labels are
validated or a target graph is built. Disconnection warnings use
that feature graph. A continuous target can reconnect a row that
the feature cutoff isolated. A fully empty feature graph produces
an empty categorical graph and a `ValueError` for continuous
supervision. Embedding mode also raises `ValueError` when
categorical supervision leaves an empty graph.

## Categorical targets

Labels are compared by equality. Exactly `-1` represents an unknown
label, so `-2` is an ordinary class. The fixtures include all-unknown
labels, fractional classes, single-column arrays, and unsigned
classes above `2**53`. Known cross-class edges receive
`exp(-2.5 / (1 - target_weight))` below weight one. Edges incident
to an unknown label receive `exp(-1)`. Weight one, the first
float64 value above one, and NaN use full cross-class attenuation
in the tested cases. A negative weight still runs the below-one
formula. The tests include the adjacent float64 values around zero,
one half, and one.

## Continuous targets

Scalar and vector Euclidean, Manhattan, Chebyshev, cosine, Hamming
and Jaccard targets, vector correlation, and a square precomputed
target are verified. The fixtures also cover the `l2` and
`cityblock` aliases, target weights on both sides of one half,
and target rank counts from two through the full sample count.
The target dtype is retained through scikit-learn's public
pairwise distance calculation. Stable ranking precedes the
conversion of the selected distances to float32. Only compact
target ranks go to the existing Rust directed graph composition.
Packed feature bytes stay in the separate native feature path.

The equal, next-float64, and next-float32 feature cutoffs preserve
the packed fit and held-out cutoff distinction under supervision.
Fitted-input transforms return the graph or embedding shortcut.
Held-out queries still rank the packed feature distances and use
the supervised layout in embedding mode. Supervised update raises
`ValueError` in the tested cases.

## Failures and remaining differences

Before a cold fit succeeds, an invalid target leaves the published
feature graph visible and later transforms raise `AttributeError`.
After a failed refit, a held-out graph query sees the replacement
feature rows. The former exact input still takes its fitted-input
shortcut. The graph can therefore have the replacement size when
returned through that shortcut. An invalid target neighbor count
below two fails before replacing the graph.

Packed distance names used as target metrics and rectangular
precomputed targets raise `KeyError` after the feature graph has
been published. On the ten-row probe a target rank count of 11
raised `SystemError` in the public wheel. The port bounds the
rank count and raises `ValueError`. A scalar correlation target
and a nonfinite continuous target weight raise `ValueError` on
both paths. The public wheel can publish a partially composed
graph in those cases, while the port retains the feature graph.
Embedding queries after a failed refit to a different row count
can also differ when the old embedding and new graph lengths
disagree. Callable targets, other target metrics, and packed
density are outside these fixtures.

Recheck the fixture from the repository root.

```sh
uv run --project reference --locked python reference/generate_packed_supervised_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-packed-supervised-x86_64.json --check
```
