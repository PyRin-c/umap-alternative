# Raw recursive placement rounding in UMAP 0.6.0

The [fixture generator](generate_modern_recursive_rounding_fixture.py)
uses the pinned public `label_propagation_init`, `recursive_init` and
`UMAP.fit` calls. It records layout bits before outer normalization,
input CSR bits and caller RNG checkpoints. Its failure diagnostics
contain the exception class only.

## Percentile interpolation and transition weights

The base layout centers coordinates in float32. The ranks for the fifth
and ninety-fifth percentiles are computed in float64. Interpolation
starts at the nearer endpoint. The distance from that endpoint rounds
to float32 after it is computed, so the upper branch rounds
`1 - offset` directly. Computing `1 - float32(offset)` gives a
different result on the signed mixed-magnitude controls. The gap,
interpolation and span round in float32. The scale is applied in float64
before storing the coordinates.

All 26 committed base layouts now match every bit, up from 12. The
controls contain signed and offset coordinates, one through seven
axes, constant axes, subnormal widths and a one-ULP width. In a broader
175-case probe, nearest-endpoint interpolation matched every array.
Forming the upper distance after the float32 cast matched 169.

The supplied contractions separate raw transition sums from stored
layout weights. At an edge weight of `2**24`, individual cross-part
memberships can disappear when added in float32. These sums need
float64 through contraction and normalization. Even casting the
completed sums before normalization misses a positive-update control.
The layout memberships themselves round to float32. The one-level
probes hold labels and anchors fixed and distinguish the first
attraction by using budgets zero, two and three. Two-level held-outs
exercise the normalized graph at the following placement stage.

All 13 contraction arrays now match bit for bit, up from six. The
largest old error was `0.109083` and the caller RNG traces and input
CSR records match in both versions. The base and transition corrections
run with feature centering in float32 before the float64 PCA sketch
and a float64 application of the existing Procrustes rotation. The
combined path preserves the established fit tolerances on both hosts.

## Earlier archived star measurements

Before the full-data centering correction below, the first observable discrepancy occurred in the PCA-only
checkpoint. It uses threshold 27 and zero coarse epochs, so no
contraction, transition normalization or coarse optimizer has run.
With zero-based indices, row 1, axis 0 is `0.4871694744` in the port and
`0.4871698320` in the wheel. Only four of the 54 coordinate words
differ at this checkpoint, down from 47. The separately supplied
base controls match exactly, leaving the sampled PCA output and its
handoff as the remaining precision boundary.

| Stage | Before | After | Exact words after |
| --- | ---: | ---: | ---: |
| PCA-only root | `4.769e-7` | `3.577e-7` | 50 / 54 |
| One contraction, budget 0 | `7.153e-7` | `7.153e-7` | 13 / 54 |
| One contraction, budget 11 | `3.279e-7` | `3.279e-7` | 6 / 54 |
| Two contractions, budget 0 | `4.769e-7` | `4.769e-7` | 3 / 54 |
| Two contractions, budget 11 | `4.296e-7` | `1.512e-7` | 2 / 54 |
| Fitted budget-eleven star | `2.337e-5` | `4.769e-6` | 0 / 54 |

Graph and RNG bits match at each checkpoint. The fitted star now uses
the suite's default `5e-6` bound. The frozen Adam checkpoint assertions
from the previous study were bit exact. At that point the fitted
trajectory still differed.

An exploratory extreme mixed-scale PCA case still differs substantially
on its weakest axis. Use 129 rows and three columns from
`RandomState(742).normal`, cast to float32, multiply columns by
`exp2(linspace(-12, 16, 3, dtype=float32))`, and multiply every third
row by `float32(-2**-8)`. With a path CSR, three components, seed 742,
threshold 129 and zero epochs, the weakest axis differs by `11.646` on
Windows and the caller RNG matches. Using seed 489 for both the data
and sketch changes the weakest-axis error from `8.452` to `16.877`.
These two inputs are now included in the fixture below. The exact base
and transition claims apply to the supplied placement controls.

### Full-data PCA centering

An observation of the matrix passed to scikit-learn's randomized SVD
matches float32 column sums followed by float64 division and subtraction.
The loss of a few bits in the previous centering order affected the
lowest-variance column substantially. All 54 coordinates in the archived
PCA-only star and all 387 coordinates in each of the two mixed-scale cases
now match, along with the caller RNG state. The archived star after the
two-contraction, eleven-epoch coarse path still differs by up to
`1.49e-7` before outer normalization on Windows.

The more accurate SVD input slightly changes three integrated
coarse-placement cases. The two rank-one initialization bounds
are `8e-6` and `3.5e-6` respectively. The shared four-level, two-epoch
case uses `5.1e-6` in the coarse and transition replays. These bounds
apply to the affected placements and preserve the exact PCA checks.

The two hosts generated byte-identical fixtures. Windows and Ubuntu
WSL each passed 2,993 installed-wheel tests with zero expected failures,
198 locked Rust tests with nine ignored, formatting and warning-denied
Clippy.

## Exact contraction, expansion, and fitted checkpoints

Observing the layouts at successive calls to the public
`label_propagation_init` isolates the first difference in the archived
star. The 27-row PCA anchor matches all 54 values. After contraction,
the nine group anchors match 17 of 18 values with a sum-then-divide
mean. Applying the reciprocal group size to each member in float64
before summing in row order reproduces all 18 values.

At the expansion boundary the arrays passed to public `procrustes_align`
confirm that the graph-neighbor averages and wide centering match.
The rotation centers both operands and scales each to unit RMS.
Column accumulation is float32, mean division is float64, and the
centered arrays, ordered sum of squares, covariance, and SVD are
float32. The SVD uses SciPy's `gesdd`. Applying the rotation in float32
matches every coordinate in the captured one- and two-level layouts,
with zero and eleven coarse epochs.

Fuzzy-union contraction takes membership complements in float64.
Rounding `1 - weight` to float32 previously changed 34 entries on
the first contraction of the four-level control. The outer fit also
uses float32 interpolation from the nearer endpoint for its robust
percentiles, matching the seven-row threshold control.

The recursive tests compare the recorded coordinates by float32
bits at base, coarse, expanded, and fitted checkpoints. This includes
the archived star, its positive coarse updates, the rank-one
holdouts, and the four-level transition and fuzzy-union cases.
The repeated-singular-value controls compare the PCA subspace.

## Reproduce

Use separate locked reference and project environments with one
numerical thread and at most two Cargo jobs. Keep build products on C:
or under the WSL home. Replace `ubuntu` with `windows` for the Windows
fixture. See the [wheel build commands](modern_recursive_transition_reference.md#reproduce).

```sh
uv run --project reference --locked python -B -W ignore reference/generate_modern_recursive_rounding_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-rounding-ubuntu-x86_64.json
uv run --no-sync python -B -m pytest -p no:cacheprovider -q \
  tests/test_modern_recursive_rounding.py tests/test_modern_recursive_coarse.py tests/test_modern_recursive_star.py
```
