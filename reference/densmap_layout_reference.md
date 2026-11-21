# Serial densMAP layout

The [Windows capture](fixtures/umap-learn-0.6.0-densmap-layout-windows-x86_64.json)
and [Ubuntu capture](fixtures/umap-learn-0.6.0-densmap-layout-ubuntu-x86_64.json)
use the pinned umap-learn 0.6.0 wheel, CPython 3.13.12, one numerical thread,
explicit initialization, and `compatibility_layout=True`. The generator
uses the published `fit_transform` return values, `graph_`, `embedding_`,
and `embedding_list_`. It records float32 bits under their epoch keys, the
final result, aligned float32 input distances, and public exception types.
The two captures agree on all case data, checkpoint bits, and errors.
[UMAP API guide](https://umap-learn.readthedocs.io/en/latest/api.html)

The Rust entry point is `umap_core::densmap::optimize_densmap_csr`. It takes
normalized coordinates, the training CSR graph after its cutoff, and one
unsquared input-space distance per CSR entry. The fixture starts from the
wheel's graph and known distances to isolate layout from graph construction.
Rust normalizes the supplied initial coordinates, seeds the established
serial sampler, and runs the native layout against every captured epoch
key. A scalar zero-epoch run reports key 0. During a run of length `N`,
keys `0..N-1` are post-epoch positions and key `N` repeats the final
coordinates.

## Activation and edge order

Epochs start at zero. Density is eligible when `lambda > 0` and
`(e + 1) / N > 1 - dens_frac`, with the comparison evaluated in float64.
The positive sample cursor is initially due at epoch 1 for maximum-weight
edges, so scalar zero and one epoch return the initial layout even for
`dens_frac=1`. The smallest positive float64 fraction leaves
`1 - dens_frac` equal to one and never activates the objective in the
probes. At `N=4`, both half and its predecessor first change the coordinates
at key 2. The float64 successor of half first changes them at key 1.
The default fraction first changes key 7 at `N=10`. The predecessor of one
matches one, and `lambda=0` and `dens_frac=0` match plain UMAP.

The weighted probes distinguish the epoch's density reduction from the
positive sample sweep. For the reduction, visit every positive training
membership, including edges with an interval beyond the current epoch.
In the weak-edge probe the 3-epoch fit retains 34 memberships. The
16-epoch fit leaves eight of those entries at zero, and they are excluded
from the reduction and the sample sweep. The radius sweep adds each directed
entry to both endpoints. The optimizer then visits due edges in ascending
CSR order, updates both endpoints, and processes that edge's negative
samples using its updated head. Negatives carry the ordinary repulsion
term. The compatibility cursor, per-head sampler, and post-epoch
learning-rate decay are shared with the existing UMAP optimizer.

## Radius and correlation terms

The input log radii use the existing [original diagnostic reduction](densmap_reference.md#radius-arithmetic).
Let `R_i` be those radii centered and divided by their population standard
deviation. For squared embedding distance `q = ||y_i - y_j||²`, use
`p(q) = 1 / (1 + a q^b)`. Let `S_i` and `T_i` be the sums of `p(q)`
and `p(q) q` at vertex `i`. The embedding log radius is
`r_i = ln(T_i / S_i + 1e-8)`. This uses the input edge support, with
affinities derived only from embedding coordinates.
[Narayan, Berger & Cho, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8195812/)

At the start of each active epoch, freeze `S_i`, `r_i`, and the global
correlation terms. With `m` the mean embedding log radius and `n` the
vertex count, the pair probes at shifts 0, 0.1, 1 and 4 support

```text
v = sum((r_i - m)^2) / n + dens_var_shift
c = sum(r_i * R_i) / (n - 1)
w_i = R_i - c * (r_i - m) / v
```

The shift changes the variance term and the standard deviation
`sqrt(v)`. It leaves the covariance term unchanged. The pair endpoints
allow the shift, clipping, and covariance normalization to be varied
before any effective negative update. Dividing the covariance by `n`
or the variance by `n-1` moved first-active coordinates by up to about
0.02 and 0.05, respectively, in the unclipped pair probes.

Differentiating the weighted mean and then its logarithm gives the
per-edge radius response below. The approximation uses the frozen log
radius in the mean term, dropping the offset there. Use the current
edge distance for `p` and `h`; the second directed sample of a pair sees
the coordinates changed by the first. Freezing the edge distance as well
failed the first-active replay.

```text
h(q) = a b q^(b-1) / (1 + a q^b)
dr_i/dq = (p(q) / S_i) * ((1 - b * (1 - p(q))) / exp(r_i) + h(q))
Z = sum of positive directed graph memberships
density scale = 2 * lambda * Z * (w_i * dr_i/dq + w_j * dr_j/dq)
                / (membership_ij * n * sqrt(v))
```

The scale multiplies `y_i - y_j` and that direction is reversed at the
other endpoint. Division by the sampled graph membership compensates
for its update frequency. Both this component and the ordinary attractive
component are clipped independently to `[-4, 4]` before addition and
learning-rate scaling. The large-`lambda` pair fixture exposes the
separate clipping. [Narayan, Berger & Cho, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8195812/)

The embedding sweep stores both sums in float32. For the affinity sum,
add the wide affinity to the stored value and round once. For the distance
sum, round the weighted product before adding it. Divide the two sums in
float32, widen the quotient for the offset and logarithm, then round the log
radius to float32. The mean and population variance use float32 pairwise
reductions. The variance shift and square root also round to float32.
The covariance sums wide products of the float32 embedding log radii
before mean subtraction and the standardized input radii. The
per-vertex weights still use the centered embedding radii. The sum
of positive memberships is half the float32 sum of the incident
weights from the original radius pass.

The first-active probes include nine disjoint reciprocal pairs with distinct
radii and three variance shifts. They isolate the log and covariance
reductions from the weighted graph and repulsion. The balanced pair probe
uses equal embedding radii and distinct input radii. Its two forces almost
cancel at a coordinate initialized to zero. The wheel moves that coordinate
by about -3.08e-12. Using a float64 square root or exponential changes the result at that
coordinate by several tenths of the measured displacement. The replay
uses a float32 exponential and adds the clipped forces in float64 before
scaling and rounding the coordinate write.

## Degeneracy and replay limits

Equal input log radii have zero standard deviation. The wheel returns
NaN coordinates at the first active update in the equal-input probe and
finite coordinates at zero effective density updates. Equal embedding log
radii with distinct input radii produce finite updates for shifts 0.1,
1e-12, and 1e-40. At shift zero or 1e-50 the wheel reports
`ZeroDivisionError`. The latter shift rounds to zero in float32. Rust
reports a nonfinite density-gradient error before writing the affected edge
in these undefined cases. Checkpoints for earlier completed epochs remain
available. The native test also supplies exactly coincident endpoints with
zero variance and verifies that the zero-distance positives do not request
a gradient.

The current replay matches 2,285 of 2,437 finite coordinate values
in the Windows capture and 2,283 of 2,437 in the Ubuntu capture.
This counts each captured epoch key once, including the final key.
All inactive checkpoints, plain controls and paired first-active
cases agree bit for bit.

| Active probe | Previous max absolute difference | Current max absolute difference | Current max float32 steps |
| --- | ---: | ---: | ---: |
| Weighted, density active at key 1 | 9.54e-7 | 9.54e-7 | 2 |
| Weighted, half or the predecessor of half | 2.38e-7 | 2.38e-7 | 1 |
| Weighted with effective repulsion | 2.38e-7 | 9.54e-7 | 2 |
| Weighted with `dens_lambda=0.2` | 3.73e-9 | 3.73e-9 | 1 |
| Default density fraction | 0 | 5.96e-8 | 1 |
| Weak edges, retained in the 3-epoch run | 1.19e-7 | 1.86e-9 | 5 |
| Weak edges, pruned in the 16-epoch run | 3.34e-6 | 9.54e-7 | 10 |
| Independent 1D run | 7.45e-9 | 7.45e-9 | 1 |
| Independent 3D first-active run with no effective negatives | 1.91e-6 | 1.19e-7 | 1 |
| Independent 3D first-active run with shift 1e-12 | 4.77e-7 | 0 | 0 |
| Independent 3D run with effective negatives | 6.44e-5 | 4.77e-7 | 2 |

The 3D run with effective negatives matches through epoch 2 before density
activates. At epoch 3, the maximum difference decreased from 4.53e-5 to
5.96e-8. Over all keys of that run, the exact count increased from 76 to
147 out of 189. Its plain control matches for all eight epochs.

For the dense Python fit with `dens_frac=1` and 30 epochs, the first
difference is one coordinate at epoch 2 (one float32 step). The final
maximum absolute difference increased from `1.91e-6` to `8.11e-6`.
Adding the wide weighted distance to the stored radius sum before rounding
reduced the 30-epoch error to `1.43e-6` and increased the 16-epoch weak-edge error to
`6.06e-5`. Applying the float32 polynomial to the embedding radii caused
differences in the previously exact zero-shift and tiny-shift pair controls.
The implementation retains the existing embedding-radius calculation and
the Python tests record separate bounds for the three affected cases.

## Seeded fitting

`UMAP(densmap=True)` uses the serial Rust optimizer for dense features and
dense precomputed distances. It uses the same random, explicit, and
spectral initialization as compatibility UMAP. The Python defaults are
`dens_lambda=2.0`, `dens_frac=0.3`, and `dens_var_shift=0.1`. For small
datasets, the default density fit runs 700 epochs. An explicit count
through ten still uses 700 as the sampling budget for the graph cutoff.
The eight-row cutoff probe retains two memberships of about 0.00174360
at two epochs. The plain UMAP control stores zero at those positions.

The Python list, tuple, and one-dimensional integer-array forms of
`n_epochs` fit to the maximum requested epoch and store the distinct
checkpoints in ascending order in `embedding_list_`. A checkpoint
contains the layout after that epoch's updates. The maximum key denotes
the final result. `output_dens=True` adds the two diagnostic radii.
Transforming the original fitting input returns the fitted embedding.
Held-out and inverse transforms have the public densMAP errors.

For dense features, Rust callers can use
`umap_core::densmap::fit_dense_densmap_seeded`. The C11 function
`umap_capi_fit_dense_densmap_seeded` takes the seeded fit configuration
and three density parameters, then returns the standard owning graph and
embedding handle. The dense pair smoke test matches the public CSR and
coordinate bits at zero, one, and two epochs. The Python adapter matches
the precomputed training graph in the existing finite replay cases. Active
coordinates retain the arithmetic residuals reported above. Inputs with
undefined density statistics still produce a native arithmetic error.
The Python adapter currently requires internally computed neighbors and an
infinite distance cutoff. Density-preserving incremental updates remain
open. A public Windows probe that fitted eight rows at four epochs and
appended two rows with `densmap=True, n_epochs=[2, 4, 6]` exited with an
access violation. The corresponding plain fit updated to the same
coordinates as a scalar six-epoch update.

## First-active gradient isolation

The public optimizer call exposes standardized input radii and graph
memberships. An independent replay of the input reduction matched all
47 standardized radii from the 3D, weighted, weak-edge, paired and 1D
controls. It accumulated each weighted squared distance in float64,
stored each update in float32, and applied the logarithm to the
float32 mean with the float32 offset. One log argument in the 3D run is
`0x40b7c6bf`. The observed float32 log is `0x3fdfbded`, one step below
the wide logarithm rounded to float32. That difference changes the
standardized input radius used by the optimizer. Rust now applies
the pinned NumPy float32 rational polynomial to the optimizer inputs.
A mean of one plus the offset rounds to one and yields a zero log radius.
[NumPy float32 logarithm](https://github.com/numpy/numpy/blob/v2.5.5/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src#L487-L498),
[Polynomial coefficients](https://github.com/numpy/numpy/blob/v2.5.5/numpy/_core/src/umath/npy_simd_data.h#L104-L118)

To isolate the per-edge gradient, run the public optimizer for two
epochs with only the selected edge's interval set to `1`. Set the other
intervals to `100` and keep all edges in the density sweep. At
`a=.65, b=1.2, dens_lambda=1.3, dens_var_shift=.03` and learning rate
`.31`, the maximum first-update error for the independent replay was
`1.34e-5` in the 3D case and `1.49e-8` in the weighted case.
Passing unit vectors in place of the standardized input radii exposed
the covariance independently of the endpoint terms. Changing the radius
at a third vertex followed the raw embedding log radius at that vertex.
The uncentered cross product reduced the isolated 3D first-update error
to `9.54e-7`.

## Reproduction

Run from the repository root. Pass the corresponding capture path via
`--output` and append `--check` to verify the committed capture.

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\uv\umap-densmap-layout-reference"
$env:NUMBA_CACHE_DIR = "$env:LOCALAPPDATA\uv\umap-densmap-layout-numba"
uv run --project reference --locked python -I reference/generate_densmap_layout_fixture.py --output reference/fixtures/umap-learn-0.6.0-densmap-layout-windows-x86_64.json --check
cargo test -j 2 -p umap-core --test densmap_layout_fixtures
```

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/umap-alternative-verification/reference-venv"
export NUMBA_CACHE_DIR="$HOME/.cache/umap-alternative-verification/densmap-layout-numba"
uv run --project reference --locked python -I reference/generate_densmap_layout_fixture.py --output reference/fixtures/umap-learn-0.6.0-densmap-layout-ubuntu-x86_64.json --check
cargo test -j 2 -p umap-core --test densmap_layout_fixtures
```
