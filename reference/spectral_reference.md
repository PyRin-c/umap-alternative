# Connected spectral initialization reference

`generate_spectral_fixture.py` calls public `UMAP(...).fit_transform` with
`init="spectral"`, `n_epochs=0`, `compatibility_layout=True`, and `n_jobs=1`.
It fits each case twice and records the public `graph_` CSR, the normalized
float32 embedding bits, and four words drawn from the supplied `RandomState`
after the fit. The locked Windows and Ubuntu x86-64 runs used CPython 3.13.12,
NumPy 2.5.5, and SciPy 1.18.1 and produced the same fixture. The 17 cases cover
3 to 83 rows, 1 to 8 dimensions, and minimum, maximum, and interior seeds.

The helper computes the symmetric normalized Laplacian
`I - D**(-1/2) @ W @ D**(-1/2)` using the float32 graph degrees and sparse
product. The float64 identity promotes the difference. For `k = n_components + 1`
smaller than the row count, SciPy's `eigsh` uses smallest magnitudes, an all-ones
starting vector, tolerance `1e-4`, and `ncv = max(2*k + 1, floor(sqrt(n_rows)))`.
The 36-, 48-, 53-, and 83-row held-outs distinguish the square-root term and its
rounding. When `k == n_rows`, a dense `scipy.linalg.eigh` solve supplies all
eigenpairs. The helper sorts by eigenvalue and drops the first eigenvector.

It scales the remaining vectors to a global maximum absolute coordinate of
10 and casts them to float32. The caller's RNG first consumes a normal block
of shape `(n_rows, k)`. The next `(n_rows, n_components)` normal block is
multiplied by `1e-4` and cast to float32 before addition. The helper returns
the raw row-major coordinates. The optimizer applies its existing per-column
`10 * (x - low) / (high - low)` normalization. All 818 normalized coordinates
match the public fixture bit for bit. The test also finds the observed RNG
checkpoint exactly three words past the helper's final RNG state in every case.

The helper retains the solver's eigenvector signs. The public fit and the
helper can each return either orientation on a symmetric three-vertex path in
one dimension, even with a constant public random seed. An all-ones starting
vector is orthogonal to that path's antisymmetric eigenvector. Exact parity
is established for the recorded graphs and SciPy builds. Another 29-row
public probe produced two different CSR membership bits across the hosts, so
its spectral coordinates also differed between hosts.

The supported spectral domain is a connected graph with
`1 <= n_components < n_rows`. On the recorded public probes, fitting with
`n_components >= n_rows` warned and returned a fallback layout. The helper
raises `ValueError` for that dimension range and `NotImplementedError` for a
disconnected graph. Very large graph compatibility is outside the verified
domain. The helper uses SciPy's default iteration limit and propagates solver
failures such as `ArpackNoConvergence` before consuming the caller's RNG.
The dense full-spectrum case uses quadratic storage. Alternate large-graph
solvers and random or component-wise fallback layouts are outside this helper.

Regenerate the fixture and run the tests from the repository root.

```sh
uv run --project reference --locked python reference/generate_spectral_fixture.py --output reference/fixtures/umap-learn-0.6.0-spectral-x86_64.json
uv run --locked pytest tests/test_spectral.py
```
