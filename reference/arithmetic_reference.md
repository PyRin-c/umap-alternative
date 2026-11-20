# Public UMAP optimizer arithmetic probes

`generate_arithmetic_probe.py` records two tiny, clean-room probes against the
locked public `umap-learn==0.6.0` wheel.  The generator uses only
`UMAP(...)`, `fit_transform`, and public `graph_` CSR fields; it does not read
UMAP source, private attributes, compiled implementation details, or tests.

Reproduce the Windows fixtures from the repository root:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
uv run --project reference --locked python reference/generate_arithmetic_probe.py `
  --output-dir reference/fixtures
```

The probe uses four distinct float32 input points, dense Euclidean distances,
`n_neighbors=3`, one output component, seed `42`, and `n_jobs=1`.  The graph
has ten nonzero CSR entries and includes nontrivial float32 edge weights.  An
explicit, unequal-spacing float32 initialization makes the compatibility
normalization observable.  Each checkpoint is a fresh public
`fit_transform` call, with `repulsion_strength=0`, and records exact output and
CSR float32 bit patterns for `n_epochs=0` and `n_epochs=2`.

The base case sets `(a, b) = (1.0, 1.0)`.  The second case changes only those
curve parameters to `(1.7, 0.9)`, providing a force-curve discriminator while
holding input, graph, seed, layout, and schedule parameters fixed.  Two fresh
calls for every checkpoint were bit-identical when the fixtures were recorded
on CPython 3.13.12 / Windows AMD64.

Fixture files:

- `umap-learn-0.6.0-arithmetic-4x1-base-windows-x86_64.json`
- `umap-learn-0.6.0-arithmetic-4x1-ab17-09-windows-x86_64.json`
