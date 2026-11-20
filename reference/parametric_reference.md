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
& $parametricPython -I reference/generate_parametric_stages_fixture.py --check
```

Per case, the fixture stores 64 fitted coordinate bits, six held-out
coordinate bits, two bits for the single-row transform, and nine bits
for inverse-transformed coordinates. The zero-row transform raised
`UnboundLocalError` in both cases. Saving with `exclude_raw_data=True`
raised `NotImplementedError` in both cases. Those two configurations check
162 float32 coordinate bits across fitted, held-out, singleton and inverse
outputs.

The [stage fixture](fixtures/umap-learn-0.6.0-parametric-stages-windows-x86_64.json)
uses the same pinned environment. The probes call the public estimator methods,
record the inputs to supplied Keras models, and use a Keras callback to stop
training after the first batch. The no-op cases set the learning rate to zero
through the callback. Exception records contain the type and first message line.

## Positive edges and batches

Each positive endpoint appeared in the supplied encoder's inputs. To measure
the finite edge pool, one probe bypasses the public TensorFlow
`Dataset.shuffle` method for a dataset of scalar integer pairs. The intercepted
dataset is already infinite and requests a shuffle buffer of 10,000. With that
shuffle bypassed, two consecutive periods were identical in each case. The
remaining probes ran with the normal shuffle.

| Input rows | Graph epochs | Graph entries | Directed edges in pool | Total copies |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 30 | 10 | 10 |
| 8 | 2 | 30 | 22 | 32 |
| 8 | 5 | 30 | 30 | 98 |
| 32 | 10 | 128 | 128 | 848 |
| 8 | 3 | 30 | 18 | 22 |
| 8 | `None` | 30 | 30 | 10,876 |
| 8 | 4,176 | 30 | 30 | 51,502 |

The counts match `floor(float32(epochs) * weight)` on the public fuzzy graph
from `UMAP(transform_mode="graph")`. The three-epoch case uses
`local_connectivity=0, set_op_mix_ratio=0.25`. Its maximum membership is
0.7904693484, and normalizing by that maximum would give 30 copies. The
4,176-epoch case uses the same graph. Each direction of the maximum-weight
edge contributed 3,301 copies. Truncating the float64 product would give
3,300. The eight-row default-epoch result matches 500 graph epochs.

The native Rust function `umap_core::parametric::prepare_parametric_edges`
takes a CSR graph and an explicit epoch count. It emits directed edge counts
in CSR order and the expanded length. The Windows tests reproduce all
308/308 input graph weight bits, 268 retained directed counts, and 63,388
copies across the seven cases, including when the graph comes from native
dense Euclidean fitting.

With 848 copies, the observed Keras steps per epoch were 84 at batch size 1,
2 at size 32, 1 at size 84, and `None` at sizes 85, 848, and 849. This matches
`floor(total_copies / batch_size / 10)` with a zero result passed to Keras
as `None` in these probes. Batch size 849 still yielded a full first batch.
Explicit `batch_size=None` took the input row count at both 8 and 32 rows.

## Layout and reconstruction

With `a=1.5` and `b=0.75`, the positive-only loss was 0.737054706.
Using the logit `-log1p(a * distance ** (2 * b))`, mean sigmoid cross entropy
for positive labels reproduced it within 5.96e-8. With two negatives per
positive, the observed loss was 0.630794108. For the same batch, a permutation
of repeated tail embeddings paired with repeated heads reproduced the mean
over positive and negative labels within 1.19e-7. That comparison checked
2,520 candidate tail permutations. Changing `repulsion_strength` from 1 to
2.5 gave the same observed batch and loss.

The no-op cases preserved all 64 fitted bits. One step changed all 64 bits
with a maximum absolute change of 0.00378448 for the positive-only case and
0.00816640 with negatives. Within each pair, the first batch and reported
loss were identical.

With reconstruction enabled, all 8/8 decoder input bits matched the encoded
heads. The reconstruction term matched binary cross entropy against the
original head features. The observed loss increased by 0.690435886 at
reconstruction weight 1 and by one quarter of that at weight 0.25. At the
default `autoencoder_loss=False`, the one-step encoder matched the
layout-only encoder at 64/64 bits, and the decoder changed at 9/9 held-out
coordinates. Setting `autoencoder_loss=True` changed all 64 encoder
coordinate bits relative to the reconstruction step with the default,
with a maximum absolute difference of 0.00498337.

## Frozen inference and persistence

With the original seeded linear encoder and decoder frozen, `fit` returned
the estimator and matched direct encoder predictions at 64/64 fitted bits
in each 32-row case. Across 1, 15, 16 and 17 held-out rows, explicit matching
batch sizes gave exact encoder output. For 17 queries, `transform` with
batch size 0 or `None` matched the estimator's batch size of 16 at 34/34
bits. Direct Keras default batching matched at 33/34 bits with a maximum
absolute difference of 2.98e-8. Decoder-backed `inverse_transform` matched
3/3 and 9/9 bits for one and three latent rows.

| Public operation | Observed result |
| --- | --- |
| Fit with zero input rows | `ValueError` for zero samples |
| Fit with one input row | A `float64` zero embedding of shape `(1, 2)` and no training callback |
| Fit with two input rows | `ValueError` from an empty maximum reduction |
| Fit at zero graph epochs | `IndexError` for index array dtype |
| Fit with batch size 0 or -1 | `InvalidArgumentError` and `ValueError`, respectively |
| Transform with zero rows | `UnboundLocalError` |
| Transform three rows with batch size -1 | `ValueError` for shape -3 |
| Inverse with zero rows | `ValueError` without reconstruction and `UnboundLocalError` with it |
| Inverse after singleton fit | `QhullError` without reconstruction and direct decoder output with it |

For singleton fits, estimator save and reload succeeded with both
`exclude_raw_data` values. The loaded transform matched 6/6 bits in each
case, and the loaded inverse matched 9/9 bits when a decoder was present.
For 32-row fits, both save settings raised `NotImplementedError` and left
Keras artifacts without `model.pkl`. Estimator reload then raised
`FileNotFoundError`. Keras could load the standalone encoder and decoder
artifacts and reproduce the same 6/6 and 9/9 bits.

Remaining scope includes a Python estimator or backend adapter, shuffled
batch and negative RNG replay, optimizer parity across multiple steps,
default architectures, landmark losses, large-data defaults, and full
estimator persistence after multi-row training.
