# Public UMAP layout reference

`generate_layout_fixture.py` records only public calls against the locked
`umap-learn==0.6.0` environment. It runs `UMAP(...).fit_transform` afresh for
each `n_epochs` value and stores the returned float32 embedding as uint32 bit
patterns. No UMAP source, private attributes, compiled implementation details,
or tests are read.

Reproduce the Windows fixture from the repository root:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
uv run --project reference --locked python reference/generate_layout_fixture.py `
  --output reference/fixtures/umap-learn-0.6.0-layout-windows-x86_64.json
```

The same command was also run under Ubuntu on WSL2 with an isolated temporary
environment (so the Windows `reference/.venv` was untouched):

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/umap-layout-reference-venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run --project reference --locked python reference/generate_layout_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-layout-ubuntu-x86_64.json
```

The probe uses eight distinct float32 points, dense Euclidean distance,
`n_neighbors=4`, `random_state=42`, `n_jobs=1`, and
`compatibility_layout=True`. It records checkpoints `n_epochs=0, 1, 2, 3, 5,
10` for both `init="random"` and a supplied float32 array with unequal column
ranges.

Observed on the recorded Windows host (CPython 3.13.12, NumPy 2.5.5,
umap-learn 0.6.0):

* `n_epochs=0` and `n_epochs=1` are bit-identical for both supplied cases;
  `n_epochs=2` is the first changed result in this small probe. This is an
  observation of this graph and schedule, not a claim that every UMAP case has
  the same no-op first epoch.
* With `compatibility_layout=True`, the `n_epochs=0` explicit result is
  independently min-max normalized per coordinate to `[0, 10]` and returned as
  float32. The unequal explicit ranges make this column-wise behavior
  observable.
* For this exact-neighbor case, the `init="random"`, `n_epochs=0` bits exactly
  match column-wise `[0,10]` normalization of the public call
  `np.random.RandomState(42).uniform(-10, 10, size=(8, 2)).astype(np.float32)`.
* `n_epochs` is a total run length for a fresh `fit_transform` call; it is not a
  public API for resuming an optimizer or reading intermediate state from one
  call. The fixture's checkpoints therefore compare separate final outputs.

Both Windows 11 x86_64 and Ubuntu x86_64 (WSL2 kernel 5.15.167.4) were tested.
For every recorded checkpoint in this small case, the Windows and Ubuntu
float32 bit patterns are identical. The two JSON files still retain separate
platform/package/thread metadata; this is evidence for this case, not a general
cross-platform guarantee.

