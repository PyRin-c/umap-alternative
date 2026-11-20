# Canberra and Bray–Curtis feature distances

The reference is the pinned public `umap-learn==0.6.0` wheel on
Windows x86-64 and Ubuntu WSL. The estimator accepts `canberra` and
`braycurtis` as feature metrics. Both metric names work for dense and
CSR inputs with the compatibility layout.

Canberra forms each absolute difference, the sum of the two absolute
coordinates, and their quotient in float32. The row distance is a
float64 sum of these terms. A coordinate pair with denominator zero
contributes zero. Bray–Curtis sums the float32 absolute differences in
float64 and divides by the float64 sum of the float32 absolute pair sums.
A zero denominator gives zero distance, including opposite signed rows.
CSR distance calculation merges the stored coordinates and supplies
implicit zeros.

The model built from the public named distance functions matches the
fixture graphs after casting the ranked distances to float32. On the
42-row signed and nonnegative inputs, replacing these functions with
SciPy's pairwise distance implementation changes 46 Canberra and 228
Bray–Curtis CSR weights on shared coordinates. The same comparison on
held-out queries changes 9 Canberra and 21 Bray–Curtis weights.

The regression fixture tests dense and CSR fits and forward transforms,
dense inverse optimization and updates, positive epochs, repeated zero
rows, opposing rows, and inclusive cutoffs of 1, 0.5 and 0. Rust uses the
fitted feature metrics for inverse gradients. Canberra differentiates the
per-coordinate ratio in float32. The Bray–Curtis gradient uses
`(sign(x-y)-distance)/denominator`, with zero gradient when the
denominator is zero. A separate wide-precision probe still finds
float64 distance and gradient rounding residuals for Bray–Curtis before
the outputs are narrowed to float32.

The C ABI exposes `UMAP_CAPI_METRIC_CANBERRA` and
`UMAP_CAPI_METRIC_BRAYCURTIS` with unchanged ownership conventions.

For the nonnegative wide Bray–Curtis held-out graph, one ordered weight
differs by one float32 ULP between Windows and Ubuntu WSL. The fixture
stores those held-out weights separately for each host so both
regressions can compare exact bits. Sparse duplicate-column precision,
larger approximate graphs, and the wider public metric surface remain
open.

## Reproduction

Use the locked reference environment and put build artifacts and Numba
caches under LocalAppData on Windows or the Linux user's cache under WSL.
In the fixture check, select the matching host's supplemental file.

```text
uv run --project reference --locked python -I reference/generate_ratio_metric_fixture.py --output reference/fixtures/umap-learn-0.6.0-ratio-metrics-x86_64.json --platform-output reference/fixtures/umap-learn-0.6.0-ratio-metrics-query-windows-x86_64.json --check
python -I -m pytest -q tests/test_ratio_metrics.py
```
