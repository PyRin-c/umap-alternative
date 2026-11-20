# Public UMAP supervised graph reference

The generator fits the public `UMAP(transform_mode="graph").fit_transform(X, y=y)`
API on the same eight dense float32 rows in every case. It verifies that the
returned graph is the estimator's public `graph_` and records the CSR shape,
row pointers, column indices, and exact float32 weight bits in storage order.
[Generator](generate_supervised_graph_fixture.py) · [JSON fixture](fixtures/umap-learn-0.6.0-supervised-dense-graph-windows-x86_64.json)

The fixture pairs the unsupervised `y=None` baseline with integer classes
`10` and `20`, partial labels with `-1` at rows 3 and 4, and a vector of
eight `-1` values. At `target_weight=1`, the number of stored CSR entries
falls from the baseline's 28 to 10 with full labels and 20 with partial labels.
Edge `(0, 1)` joins different known labels and is absent in both full-weight
cases. Edge `(0, 3)` is absent with full labels and has bits `0x3e8bb62b`
with partial labels. With all rows unlabeled, edge `(0, 3)` has bits
`0x3ec1b05e` compared with `0x3e58989a` in the unsupervised baseline.

Run from the repository root with the locked environment.

```powershell
uv run --project reference --locked python -I -B reference/generate_supervised_graph_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-supervised-dense-graph-windows-x86_64.json
```

An existing locked `reference/.venv/Scripts/python.exe` can run the same
script with `-I -B`. The generator sets single-thread limits before import
and uses a fresh temporary Numba cache for each process. Verification used the
main worktree's reference interpreter on Windows 11 (build 26200), AMD64,
with CPython 3.13.12. All 13 installed distributions matched `reference/uv.lock`.
Two fresh fits per case matched exactly, and two independent generator processes
produced identical UTF-8 JSON bytes with LF line endings.

SHA-256 `8b0e5586b09ed86718a5001c88514f15614bed5360f627d31c863dd99ccf86c5`
