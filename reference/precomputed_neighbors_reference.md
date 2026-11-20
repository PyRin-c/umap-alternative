# Supplied nearest-neighbor arrays

The Windows x86-64 fixture records `umap-learn==0.6.0` graph fits for
eight one-dimensional float32 rows with `n_neighbors=4`, plus graph
transforms of three held-out rows. It calls only public `fit_transform`
and `transform` methods. The supplied indices are int32, distances are
float32, and the search index in the `precomputed_knn` tuple is `None`.

The fixture stores the supplied input arrays, the fitted CSR and held-out
COO results as float32 bit patterns, and warning or exception details.

| Input | Observed behavior |
| --- | --- |
| Four columns | The unmodified arrays reproduce the computed 28-entry graph. Changing the first row's distances changes weights. |
| Three columns or seven rows | A warning reports that the supplied arrays are ignored. Both fitted and held-out graphs match the computed baseline. |
| Five columns | The fifth column contributes. The fitted graph has 38 entries. |
| Different shapes for indices and distances | The fit raises `ValueError`. |
| `-1` index paired with `+inf` distance | Missing arcs are accepted. Other rows can still supply reciprocal edges to a row with no non-self neighbor. |
| Self in the second column, no self, repeated index, or unsorted distances | The fit succeeds in each recorded case. |
| Threshold at 1.0 or the preceding float32 value | Both fitted graphs have six entries. |
| Threshold at the following float32 value | The fitted graph has ten entries. |
| Threshold at zero | The graph has no stored entries. |

All thirteen successful cases that used the supplied neighbors raised
`NotImplementedError` on held-out transform. In the four cutoff cases
the fit changed both arrays supplied by the caller, replacing cut indices
with `-1` and their distances with `+inf`. Windows and Ubuntu WSL
reproduced the same outputs and warnings for all 17 cases. This fixture
covers the small dense graph path. Large data, approximate search,
supplied search indexes, and the interaction with embedding fit and
nondefault local connectivity need separate checks.

Regenerate the fixture with the locked reference environment.

```powershell
uv sync --project reference --locked --python 3.13.12
& reference/.venv/Scripts/python.exe -I reference/generate_precomputed_neighbors_fixture.py --check
```
