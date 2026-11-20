# ParametricUMAP Windows CPU reference

The fixture records `umap-learn==0.6.0` with TensorFlow 2.21.0 and Keras
3.15.1 on 64-bit Windows and CPython 3.13.12. The generator uses the
public estimator methods in two cases with explicit seeded Keras encoders.
The second case also supplies a seeded decoder and enables reconstruction.
It sets single-thread CPU execution, enables deterministic TensorFlow
operations, and disables oneDNN optimizations before importing TensorFlow.

Create a separate reference environment using PowerShell from the
repository root.

```powershell
$parametricEnv = Join-Path $env:TEMP 'umap-parametric-reference'
uv venv --python 3.13.12 $parametricEnv
$parametricPython = Join-Path $parametricEnv 'Scripts/python.exe'
uv pip sync --only-binary ':all:' --python $parametricPython reference/parametric-windows-x86_64.txt
$env:NUMBA_CACHE_DIR = Join-Path $parametricEnv 'numba-cache'
& $parametricPython -I reference/generate_parametric_fixture.py --check
```

Per case, the fixture stores 64 fitted coordinate bits, six held-out
coordinate bits, two bits for the single-row transform, and nine bits
for inverse-transformed coordinates. The zero-row transform raised
`UnboundLocalError` in both cases. Saving with `exclude_raw_data=True`
raised `NotImplementedError` in this TensorFlow/Keras environment, so
reload did not run. The fixture covers these two configurations.
Default encoder behavior, training updates, landmark examples, and
successful saving and loading require further comparison cases.
