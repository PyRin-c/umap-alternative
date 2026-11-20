# Held-out embedding initialization

The Windows x86-64 probes ran the locked reference environment with
`umap-learn==0.6.0`, a fixed random and transform seed, and one job. They
used public `fit`, `transform`, and `set_params` calls and read the
fitted `embedding_` attribute. Each arithmetic comparison used the
captured float32 fitted coordinates and the query-to-training graph.

## Observed arithmetic

1. Discard zero memberships and order the remaining entries by increasing
   training column. A row with no entries has NaN in every component.
2. If a membership is exactly `1.0_f32`, copy the fitted coordinates at
   the smallest training column with that membership. In the near-match
   fixture, the query at zero has distance zero to column 2, but the
   membership for column 0 also rounds to one. The public output copies
   column 0. The query at `3e-8` also copies column 0.
3. Sum the weights after the first one, then add the first weight. All
   additions use float32. Tails of length below eight use a left-to-right
   sum. Longer tails accumulate complete groups of eight into eight
   lanes, reduce the lanes as `((a+b)+(c+d))+((e+f)+(g+h))`, and add any
   remaining weights one at a time. Above length 128, split the tail at
   the midpoint rounded down to a multiple of eight and add the two
   recursive results.
4. Take the float32 reciprocal of that sum and multiply each weight by it.
   In descending training-column order, multiply the rounded normalized
   weight by the fitted coordinate and add the rounded product to the
   output. The accumulator starts at positive zero.

One-hot fitted columns exposed the coefficients independently of the
coordinate accumulator. The first existing fixture row requires sum bits
`1076653113`. A left-to-right sum in ascending column order gives
`1076653114`. The fitted checkpoints at zero and five epochs and the
multi-component holdouts distinguish the coordinate summation order and
separate product rounding.

The Rust helper matched 18,823 coordinate bits across 51 public probe cases
using captured COO weights. These included neighborhood widths between 2 and
512, batched and repeated queries, duplicate training rows, near-unit weights,
underflow zeros, sparse rows after applying a disconnection threshold, and
fitted NaNs. Keeping the underflow zeros in the reduction affected the
normalized values, even when every contributing fitted coordinate was finite.
Public graph mode can also store zero placeholders at `(0, 0)` between
positive entries from other query rows. The helper skips those placeholders
before grouping rows.

The integration tests exercise the existing Rust graph builder with the
fixed fitted checkpoints and the independent grids with neighborhood
widths 9, 18, 130, and 258. The last two cross the recursive sum boundary.
They also exercise the near-match and underflow fixtures with Rust graphs.

## Public transform boundary

With the existing dense transform fixture, fit epoch counts zero through
five produce the initialization bits. At six fit epochs, the exact-match
query begins at fitted row 0 with bits `[1088652650, 1087811058]` and the
public transform returns `[1090644921, 1087830069]`. This is a fixture-specific
observation about the onset of updates.

Passing all fitted input rows in their original order returns the fitted
`embedding_` object by identity. The Rust helper starts after that estimator
decision and supplies coordinates for held-out layout optimization.

## Reproduce the integration checkpoints

Use the locked environment setup in [the transform reference](transform_reference.md#reproduce-the-fixture),
then run this script from the repository root. The initializer used for the
one-dimensional probes is normalized by the public zero-epoch fit before
recording the checkpoint.

```python
import json
import warnings
from pathlib import Path

import numpy as np
from umap import UMAP, __version__

assert __version__ == "0.6.0"
fixture = json.loads(
    Path("reference/fixtures/umap-learn-0.6.0-transform-windows-x86_64.json").read_text()
)
params = dict(fixture["parameters"], a=1.7, b=0.9)


def as_float(matrix):
    return np.asarray(matrix["bits_uint32"], dtype=np.uint32).view(np.float32)


def bits(matrix):
    return matrix.view(np.uint32).tolist()


def capture(x, q, **overrides):
    config = dict(params, **overrides)
    with warnings.catch_warnings(record=True):
        model = UMAP(**config).fit(x)
        coordinates = model.transform(q)
    print("fitted:", bits(model.embedding_))
    print("held out:", bits(coordinates))
    assert model.transform(x.copy()) is model.embedding_
    model.set_params(transform_mode="graph")
    graph = model.transform(q).tocoo()
    print("COO:", graph.row.tolist(), graph.col.tolist(), bits(graph.data))


x = as_float(fixture["inputs"]["training"])
q = as_float(fixture["inputs"]["transform_queries"])
queries = np.concatenate([q, x[[0]], q[[0]]])
for epochs in range(7):
    print("fit epochs", epochs)
    capture(x, queries, n_epochs=epochs)

for i in range(len(x)):
    basis = np.column_stack((np.arange(len(x)) == i, np.arange(len(x)))).astype(np.float32)
    capture(x, q, n_epochs=0, init=basis)


def capture_1d(training, queries, k, **overrides):
    x = np.asarray(training, dtype=np.float32).reshape(-1, 1)
    q = np.asarray(queries, dtype=np.float32).reshape(-1, 1)
    i = np.arange(len(x))
    init = np.column_stack(((i * 17) % 31, (i * 23) % 37, (i * 41) % 47)).astype(np.float32)
    capture(x, q, n_neighbors=k, n_components=3, n_epochs=0, init=init, **overrides)


for n, k in ((11, 9), (30, 18), (140, 130), (271, 258)):
    i = np.arange(n)
    training = (15 * i + 6) % n if n == 11 else np.argsort((19 * i) % (n + 3))
    capture_1d(training, [1.25, n / 2 + 0.125, n - 2.1875], k)

capture_1d([2e-8, 1, 0, 2, 4, 5, 6, 8], [0, 2e-8, 1e-8, 3e-8, 2e-7], 4)
capture_1d(
    [0.00011, 0.3, 0.00015, 0.00017, 0.4, 0.00019, 0.00023, 0.00029,
     0.00031, 0.5, 0.00041, 0.00043, 1, 2, 3, 4],
    [-0.00018, -0.000234, -0.000345, -0.00019, -0.0002, -0.00025, -0.00026, -0.00029],
    12,
)
capture_1d([0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7, 1.8], [-1.09, 15], 4,
           disconnection_distance=1.1)
```
