# Disconnected spectral fit

The probes use only the installed `umap-learn==0.6.0` estimator's public
`fit`, `fit_transform`, `set_params`, and fitted attributes. A supplied NumPy
`RandomState` exposes its post-fit cursor, cached Gaussian, and next four
32-bit words. Each case stores float32 embedding bits and CSR structure and
membership bits. The project tests use integer seeds and capture the
handoff stream. Explicit neighbor probes are replayed from their fitted CSR
at the project's graph construction boundary.

## Initialization before optimization

- Detect the components from the positive sampled memberships. If their
  count is at most twice the output dimension, assign the first positive
  coordinate axes and then their negatives, using
  `ceil(component_count / 2)` axes. This placement consumes no RNG words.
- Otherwise, average the training rows in each component, retaining
  float32 means in a float64 centroid matrix. Apply the selected feature
  metric to the centroids, build the affinity `exp(-distance**2)`, and
  fit a spectral embedding of that affinity with the shared RNG. Rescale
  the centers by their maximum. [scikit-learn SpectralEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html)
- In component-label order, use half the smallest positive distance to
  the embedded centers as the local radius. Below
  `max(2 * dimensions, dimensions + 2)` vertices, draw uniform coordinates
  within that radius. Larger components use the nontrivial eigenvectors
  of their normalized Laplacian, scaled together to that radius. A local
  solve consumes a normal block of shape `(size, dimensions + 1)`.
- Assemble the float32 coordinates, scale by the global maximum absolute
  coordinate to a magnitude of 10, and add a single `1e-4` Gaussian block.
  Rust normalizes each column and draws its sampler seed, even at zero
  epochs. Isolated rows participate in both steps. Their final coordinates
  are NaN.

## Installed-wheel comparisons

Windows and Ubuntu on WSL each recorded 19 cases with CPython 3.13.12,
NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, and Numba 0.68.1. The
project wheel matched the CSR and post-fit RNG on all 19 cases on both
hosts. It matched embedding bits in 16 cases on both hosts through both
fit methods. The fixtures use new centers and staggered, shuffled normal
quantiles for the held-out graph cases. The comparisons include Euclidean,
Manhattan, and Chebyshev, unequal component sizes, and isolated vertices.
Zero and one epoch agree bitwise in both repeated cases. Two epochs give
the first effective update and also match the wheel in both cases.

The 20-row bridge probe has one component at ten epochs. At eleven epochs
the two reciprocal bridge memberships become zero and split the sampled
graph into components of 16 and 4 vertices. The stored zeros stay in
`graph_`. At eleven epochs, the project matches both the static
`learning_rate=0` and optimized results.

## Remaining coordinate differences

| Checkpoint | Windows | Ubuntu on WSL |
| --- | --- | --- |
| `chebyshev-d2-e0-s24` (Euclidean fit), 7 pairs | 27/28 coordinates differ, max absolute error 1.49827 | 14/28 differ, max error 1.15307 |
| `mixed-d2-e0-s7`, sizes 5, 3, 4, 2, 1 | 8/30 differ, max error 0.307109 | 15/30 differ, max error 0.431604 |
| `wide-d2-e0-s0`, 5 widely separated pairs | Public repeat changes embedding bits | Public repeat changes embedding bits |

At a single-center distance query, floating-point rounding can report a
small positive distance from a center to itself. The minimum positive
distance then shrinks that component's local layout to approximately
the jitter scale. In the two stable Windows residuals, substituting a
near-zero radius for the affected components reproduced all coordinate
bits. The exact rounding path behind those radii remains unresolved.
In the wide case, the centroid affinity is disconnected and the
spectral solver warns. Two public fits with identical inputs and seeds
produce different embeddings while leaving identical RNG checkpoints.
Both outputs are recorded in the fixtures.

## Regeneration

Run the generator on the corresponding host. Use `windows` in the file
name when generating on Windows.

```sh
uv run --project reference --locked python reference/generate_disconnected_spectral_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-disconnected-spectral-ubuntu-x86_64.json
```
