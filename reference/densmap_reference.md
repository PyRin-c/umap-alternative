# densMAP and `output_dens` reference

The [fixture](fixtures/umap-learn-0.6.0-densmap-windows-x86_64.json) captures
public `UMAP.fit_transform`, `UMAP.fit`, `graph_`, `embedding_list_`, and
`transform` observations on umap-learn 0.6.0. The eight-point dataset, explicit
initialization, and 42 scalar-epoch fits reproduce the supplied public probe.
The generator adds same-run epoch checkpoints and arithmetic controls.
[UMAP 0.6.0 API guide](https://umap-learn.readthedocs.io/en/latest/api.html)

```powershell
uv run --project reference --locked python reference/generate_densmap_fixture.py --check
```

Omit `--check` to regenerate the fixture or pass `--output` for a separate
file. The captured environment was Windows x86-64 with CPython 3.13.12 and
[`reference/uv.lock`](uv.lock) enforced by `uv --locked`. The JSON records
the LF-normalized lockfile SHA-256 and the package versions. The generator
sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMBA_NUM_THREADS` to `1` before NumPy and UMAP imports. The fits use
`n_jobs=1`, `random_state=42`, and `compatibility_layout=True`.

## Public API observations

- `output_dens=False` returns an `(8, 2)` float32 embedding. With
  `output_dens=True`, the result is `(embedding, r_orig, r_emb)` and both
  radius arrays have shape `(8,)` and dtype float32. Output-only mode,
  `densmap=True, dens_lambda=0`, and `densmap=True, dens_frac=0` preserve all
  16 plain UMAP embedding bit patterns at every scalar checkpoint
  (`n_epochs=0, 1, 2, 3, 10, 30`). Their two radius outputs also agree bit
  for bit at each checkpoint. The public input graph is identical across all
  42 fits, and all 36 diagnostic tuples share the same original radius.
- `dens_frac=0` and `dens_frac=1` are accepted. The immediately adjacent
  float64 values outside the interval (`-5e-324`, `1.0000000000000002`)
  raise `ValueError`. A shift of `dens_var_shift=0` is accepted and changes
  the embedding bits starting with the scalar two-epoch fit, compared
  with the default shift of `0.1`. The float64 immediately below zero
  raises `ValueError` for either `dens_var_shift` or `dens_lambda`.
- Fitting with `densmap=True` and calling `transform` with two unseen rows
  raises `NotImplementedError` with the message `Transforming data into an
  existing embedding not supported for densMAP.`

## First visible density update

Each run below requests `n_epochs=list(range(N + 1))` and compares the
documented `embedding_list_` against a plain UMAP run of the same length.
Index `0` equals the scalar zero-epoch initialization in these runs, and the
plain embedding first moves at index `1`. The last list entry repeats its
predecessor and equals the corresponding scalar `n_epochs=N` final result.
`dens_lambda=0` and `dens_frac=0` match the control at every recorded index.

| Total epochs `N` | First differing index, `dens_frac=0.3` | First differing index, `dens_frac=1` |
| ---: | ---: | ---: |
| 2 | 1 | 1 |
| 3 | 2 | 1 |
| 10 | 7 | 1 |
| 30 | 21 | 1 |

All 16 coordinates have different float32 patterns at each first differing
index. For the default fraction, only the last update differs in the
three-epoch run. These checkpoints measure the first effect visible in
public coordinates for this graph and optimizer configuration.

## Radius arithmetic

Both outputs follow `log(weighted_mean + 1e-8)` with a natural logarithm.
The close-pair control resolves the offset where the squared input
distance is about `1.13e-8` and the embedding distance is about `1.06e-4`.
The `1e-8` candidate matches 6/6 float32 patterns in each space. A zero
offset matches 2/6 in each space. The densMAP paper defines its
density-preservation metric with weighted squared distances in both spaces.
The table describes the arithmetic observed for the public `output_dens`
diagnostics.
[Narayan, Berger & Cho, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8195812/)

- **Original radius.** Traverse the documented input `graph_` in ascending
  row and column order. Reconstruct Euclidean distances using float64
  arithmetic and round each distance to float32. For each directed entry
  of weight `w`, add `w` and `w * d²` to the denominator and numerator of
  both endpoints. Evaluate `w * d²` and its addition to the numerator in
  float64, rounding the result of each update to float32. The denominator
  updates, division, offset addition, and log all use float32. Seven public
  `metric="precomputed"` fits supply known float32 distances from the same
  data at scales from `1e-3` to `100` and distinguish the rounding choices.
- **Embedding radius.** Rebuild the graph on the returned embedding with
  the same `n_neighbors` (four, including self). In two dimensions, the
  distance candidate rounds both coordinate differences to float32, rounds
  `dx²`, rounds `dy² + rounded(dx²)` together, and takes a float32 square
  root. Supply these independently ranked neighbors and distances through
  public `precomputed_knn` and read the zero-epoch refit's `graph_`. Apply
  the same reduction using `w * d`. The independently solved theory graph
  uses nearest-neighbor offsets, a scale for directed membership sum
  `log2(k)`, and the fuzzy union `p + q - p*q`.
  [McInnes, Healy & Melville, UMAP](https://arxiv.org/abs/1802.03426)

Counts below measure exact equality of complete float32 patterns. Embedding
counts include each of the 16 distinct final embeddings once.

| Candidate | Matching patterns |
| --- | ---: |
| Original graph, squared distances with wide product and addition | 64/64 |
| Original graph, squared distances with separate float32 roundings | 61/64 |
| Original graph, squared distances added only to row endpoint | 56/64 |
| Refit embedding graph, unsquared distances with wide product and addition | 128/128 |
| Refit embedding graph, unsquared distances with a separately rounded product | 108/128 |
| Refit embedding graph, unsquared distances with separately rounded coordinate squares | 115/128 |
| Refit embedding graph, log of mean squared distance | 0/128 |
| Refit embedding graph, half of log of mean squared distance | 1/128 |
| Input graph with embedding distances | 0/128 |

The results characterize rounding and graph use on these probes. The
bit-exact embedding replay uses public UMAP to turn known neighbors into
weights. The independent fuzzy graph solver reproduces the edge support of
all 16 refit graphs and differs in weight by at most `6.36e-6`. Its radius
candidate matches 8/128 patterns with maximum absolute error about `2.74e-6`.
The precise weight solver and tie-breaking of the internal diagnostic graph,
the density gradient and its variance normalization during optimization,
and exact parity on other platforms or graph regimes remain unverified.

## Fixture layout

The epoch keys in `cases` have the form `epoch-mode`. Each case uses
`common_options`, its `mode_options` entry, and the explicit `init_float32`.
`graph` and `outputs` index the shared CSR graph and array pools,
respectively. `outputs` stores one ID for a scalar result and three IDs in
public tuple order for diagnostic results. Arrays store dtype, shape and
flattened row-major float32 bit patterns as unsigned integers. The
additional `metric="precomputed"` fits use the specified distance scale,
`n_epochs=0`, and `output_dens=True` with the common configuration.
