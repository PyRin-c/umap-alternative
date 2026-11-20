# Public serial spectral fit

`UMAP.fit` and `fit_transform` support the default `init="spectral"` for
dense unsupervised compatibility fits with an integer seed in
`[0, 2**32 - 1]`. Set `compatibility_layout=True` to select the serial
compatibility layout explicitly. Omitting it emits the existing
`FutureWarning`. The seeded `n_jobs` override and the neighbor truncation
warning follow the regular dense fit.

The estimator builds the fuzzy graph once. Before solving the Laplacian, it
zeros memberships strictly below the largest membership divided by the
sampling budget. For scalar epoch counts greater than ten the budget is the
requested epoch count. Through ten epochs the cutoff uses the default budget
of 500 for small inputs. The fit retains explicit zeros in `graph_` and uses
the positive edges for connectivity and the Laplacian. Rust consumes the
same pruned CSR and the raw spectral coordinates, normalizes the columns,
and continues the MT19937 stream at its post-initialization cursor. It
consumes the sampler seed even at zero epochs. The Python state restored
after the Rust call keeps both NumPy Gaussian cache fields.

## Installed-wheel comparisons

The Windows 11 and Ubuntu 24.04 x86-64 fixtures come from the public
`umap-learn==0.6.0` wheel. Both used CPython 3.13.12, NumPy 2.5.5, SciPy
1.18.1, scikit-learn 1.9.1, and Numba 0.68.1. Each fixture records both fit
methods and stores the CSR row offsets, column indices, membership bits,
float32 embedding bits, and the post-fit RNG cursor, Gaussian cache, and
next four words. The generator initializes the public estimator with a
NumPy `RandomState` to observe its final state. The project wheel tests use
integer seeds and inspect the stream passed through the initializer.

Both installed project wheels matched 24 graph and embedding checkpoints
through each fit method, including 12 from the Rust handoff and 12 new
checkpoints. The cases cover seeds 0 and 4294967295, epochs 0 and 1,
sampled positive epochs, default epochs and curve, a one-component output,
and the full-spectrum dimension boundary. The supplemental cases cover
Euclidean, Manhattan, and Chebyshev graphs and the surviving Gaussian
sample on an odd-row input. Passing the training input to `transform`
returns `embedding_`. [Handoff checkpoints](spectral_handoff_reference.md)

On the 13-row pruning probe, the graph-only fit stores all 66 positive
memberships. The spectral fits at 0, 1, and 10 epochs zero the same two
memberships. At 11 epochs the cutoff zeroes two additional memberships and
all 26 coordinate bits differ from the 10-epoch result with
`learning_rate=0`. This isolates the effect of pruning on initialization.
The positive-learning-rate 11-epoch fit also matches the wheel exactly.

The default-curve 500-epoch fit matches the reference on each host. Its
outputs differ across hosts. Replaying each host's recorded curve
coefficients on Windows reproduced its embedding exactly.
[Curve coefficient tests](../tests/test_curve.py)

## Boundaries and residuals

| Case | Estimator result |
| --- | --- |
| One row | Zero layout. `embedding_` is float64 and `fit_transform` returns float32. |
| Two rows | Truncation and disconnected-vertex warnings, then `ValueError` for the empty graph. |
| Connected graph with `n_components == n_rows - 1` | Dense full-spectrum solve. |
| `n_components >= n_rows` | `ValueError`. The public wheel emits a warning and uses a random fallback in this case. |
| Multiple components before or after pruning | `NotImplementedError`. A 20-row probe connected at ten epochs loses its bridge at eleven epochs. The public wheel returns a layout for the resulting components. |

A failed fit preserves the previous fitted graph, embedding, and training
data. The spectral helper retains SciPy's eigenvector signs and propagates
solver errors. Large-input spectral compatibility and alternate
initializers are outside the verified domain.
[Initialization reference](spectral_reference.md)

The Rust layout replay now matches the 59-row schedule boundary and the
six- and eight-component attraction boundaries when given the observed CSR.
The broader estimator regression still covers the recorded datasets above.
[Handoff report](spectral_handoff_reference.md)

On the weighted eight-row Manhattan probe, two reciprocal graph
memberships differ by one ULP. Three coordinate bits differ at zero
epochs and all 16 differ at 11 epochs, with maximum absolute error
`0.512455` at 11 epochs. Replaying the public CSR through the project
spectral path reproduced the reference bits at 0, 11, 30, and 100 epochs.

## Regeneration

Run the generator from the repository root on the respective platform.
Use `ubuntu` in the output filename for the Ubuntu run.

```sh
uv run --project reference --locked python reference/generate_spectral_fit_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-spectral-fit-windows-x86_64.json
```
