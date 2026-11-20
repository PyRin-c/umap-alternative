# Native dense approximate neighbors

`DenseApproximateIndex` returns training ranks for `fit_supplied_neighbors_graph`
and held-out ranks for `transform_ranked_neighbors_graph`. The caller chooses
the search route. In the public dense controls, automatic search is exhaustive
below 4,096 distinct training rows and approximate at 4,096. Forcing search
selects the approximate branch for multi-row fits. A caller that falls back
from unusable supplied neighbors can make the route decision after resolving
that fallback. The Python estimator is outside this native stage.

The Rust training buffer pads to the requested width, which also determines
the training scale target. The held-out buffer uses the requested width and
the query scale target uses `min(width, n_rows - 1)`. The existing handoffs own
cutoff behavior and missing-slot encoding. The five-row controls return 18 CSR
entries at each width, and the width-eight query stores three missing entries
per row.

## Public observations

The Windows probe called the public `NNDescent` constructor,
`neighbor_graph` and `query`, and UMAP with
public `precomputed_knn`. The tested constructor settings use `n_jobs=1`,
`random_state=42`, `max_candidates=60`, `min(64, 5 + round(sqrt(n) / 20))`
trees, and `max(5, round(log2(n)))` iterations. Supplied-index replays with
those settings matched all Gaussian training CSR and ordered held-out COO
bits. The repeated-vector `k=2` query can produce the previously recorded
two NaNs at duplicate coordinates. Wheel exceptions are reported by type.
[Original wheel fixtures](approximate_neighbors_reference.md)

With zero refinement on 32 squared integers and three ranks per row, the
no-tree control matched 36 of 96 aligned exhaustive ranks. One tree with
`leaf_size=4` matched 63, and one tree with `leaf_size=32` matched all 96.
For example, row 3 returned `[3, 8, 9]`, `[3, 4, 9]` and `[3, 2, 4]`,
respectively. The first two cases show the effects of random fill and
leaf-local candidates before refinement.

With a fixed seed graph, row 0 starts with only self, with incoming links
from rows 1 and 2. The first round joins 1 and 2, and row 0 still has an
empty rank. The second round fills it with row 1. Both rounds replay every
rank and distance bit in Rust on this control. With `max_candidates=1` the
wheel retains the seed ranks through two rounds. A cap of two reproduces the
cap-60 result. The same Rust model starting with
the wheel's zero-round columns reproduces every final training column on all
96-row Gaussian, repeat and cutoff controls. Recomputing distances with the
existing Rust metrics changes 136 of 578 Euclidean and 324 of 486 cosine CSR
bits in the Gaussian `k=5` controls. Correlation `k=5` matches all 470 bits.

At 4,096 rows, the wheel's first round inserts 741 Euclidean, 403 cosine, and
3,014 correlation directed ranks. Of these, 120, 63, and 465 respectively
have no common pivot in the zero-round forward/reverse rank graph. Its maximum
incident degree is 12, 11, and 11 respectively. A model that only joins pairs
from those frozen lists cannot reproduce the entire first round. Restarting
the wheel with the zero-round columns as an explicit `init_graph` adds 636,
343, and 2,633 directed ranks, all with a shared seed pivot. Rust reproduces
all 20,480 first-round columns on each restarted graph and matches 16,819,
4,837, and 20,480 aligned distance bits respectively. After 12 restarted
rounds, the matching column counts fall to 20,322, 20,398, and 19,939. The
frozen ranks leave candidate generation during full construction and later
rounds unresolved.

At 4,096, public `query(epsilon=0)` still selects 9 Euclidean, 9 cosine, and
6 correlation columns outside the exhaustive fifth-neighbor radius, out of
20 each. With the estimator's observed epsilon (0.12 for Euclidean and
0.24 for cosine and correlation), those counts are 8, 4, and 3. Cosine
`neighbor_graph` and `query` return float64 distances. Euclidean and
correlation return float32. After casting the public distances to float32,
the Rust handoffs match every Gaussian CSR weight bit and every stored
ordered COO triple when supplied the wheel's final ranks.

## Native comparison

The native search uses seed 42 for MT19937 projection pivots and tied splits.
Each tree compares points against two pivots, splitting by the closer metric
distance. A skewed split bisects the input order for bounded depth. Training
seeds pair candidates within each leaf, with a leaf size of at least ten and
random fill for insufficient ranks. Refinement joins fresh forward and reverse
candidates with one another and with the previous round's candidates, capping
each list at 60 in index order. Equal distances retain the first offered index
at the boundary. Held-out search starts from the first tree's leaf, fills
small leaves with a separate MT19937 stream restarted at seed 42 for each
query, and walks the undirected refined graph using the metric-specific
epsilon. The pivots, cap order, graph walk and tie behavior are native choices.

The table compares native approximate output against forced wheel output.
CSR support and nonzero COO support differences are native-only / wheel-only.
Changed CSR bits count common coordinates. Ordered COO counts give equal
coordinate slots / equal coordinate-and-weight triples, including stored zeros.

| Case | CSR stored native / wheel | CSR support difference | Changed CSR bits / common | COO support difference | Ordered COO coordinates / triples |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 Euclidean k2 | 142 / 146 | 8 / 12 | 0 / 134 | 4 / 4 | 4 / 4 of 8 |
| 96 Euclidean k5 | 578 / 578 | 12 / 12 | 138 / 566 | 0 / 0 | 20 / 18 of 20 |
| 96 cosine k2 | 134 / 136 | 10 / 12 | 0 / 124 | 2 / 2 | 6 / 6 of 8 |
| 96 cosine k5 | 486 / 486 | 14 / 14 | 316 / 472 | 4 / 4 | 14 / 4 of 20 |
| 96 correlation k2 | 124 / 128 | 0 / 4 | 0 / 124 | 3 / 3 | 4 / 4 of 8 |
| 96 correlation k5 | 472 / 470 | 12 / 10 | 18 / 460 | 2 / 2 | 17 / 15 of 20 |
| 4,095 Euclidean | 23,416 / 23,596 | 2,354 / 2,534 | 9,374 / 21,062 | 8 / 8 | 8 / 0 of 20 |
| 4,096 Euclidean | 23,436 / 23,560 | 2,354 / 2,478 | 9,360 / 21,082 | 12 / 12 | 7 / 4 of 20 |
| 4,095 cosine | 20,542 / 20,468 | 1,728 / 1,654 | 13,894 / 18,814 | 2 / 2 | 15 / 0 of 20 |
| 4,096 cosine | 20,496 / 20,484 | 1,712 / 1,700 | 13,866 / 18,784 | 2 / 2 | 16 / 1 of 20 |
| 4,095 correlation | 20,224 / 20,304 | 1,394 / 1,474 | 3,104 / 18,830 | 3 / 3 | 15 / 15 of 20 |
| 4,096 correlation | 20,262 / 20,300 | 1,470 / 1,508 | 3,400 / 18,792 | 1 / 1 | 15 / 15 of 20 |
| Repeated k2 | 156 / 182 | 6 / 32 | 0 / 150 | 3 / 3 | 4 / 4 of 8 |
| Repeated k5 | 396 / 396 | 10 / 10 | 0 / 386 | 3 / 3 | 6 / 6 of 20 |
| Adjacent cutoff disabled | 596 / 594 | 6 / 4 | 2 / 590 | 0 / 0 | 12 / 12 of 14 |
| Adjacent cutoff 1.0 | 36 / 36 | 0 / 0 | 0 / 36 | 0 / 0 | 14 / 14 of 14 |
| Five rows k4 | 18 / 18 | 0 / 0 | 0 / 18 | 0 / 0 | 14 / 14 of 16 |
| Five rows k5 | 18 / 18 | 0 / 0 | 0 / 18 | 0 / 0 | 18 / 18 of 20 |
| Five rows k8 | 18 / 18 | 0 / 0 | 0 / 18 | 0 / 0 | 30 / 30 of 32 |

At 4,096, the native search selects 8 Euclidean, 5 cosine, and 4 correlation
columns outside the exhaustive ranks, out of 20 each. The native and wheel
Euclidean searches both select eight such columns and share only eight
nonzero COO coordinates overall.

The native exhaustive path matched automatic output at 4,095 on all three
metrics, including all 23,614 Euclidean, 20,452 cosine, and 20,178 correlation
CSR bits and all 60 ordered query triples. At 4,096, the automatic branch
uses the approximate comparison above. The measured differences leave the
wheel's projection state, first-round candidate scheduling and search-index
construction open for further reconstruction.

## Training rank timing

Windows x86-64 release measurements used a single Rust thread, eight
Euclidean features, and width 15 including self. MT19937 seed 2026 supplied
row-major uniform features cast to float32. The approximate index used
seed 42. Each median uses five runs after one warm-up. The exact timing
includes allocating and ranking the result. The approximate timing also
includes retaining the index for queries. Recall counts approximate ranks
present in the exact row's best 15. These measurements cover the standalone
Rust searches.

| Rows | Exact | Approximate | Ranks present |
| ---: | ---: | ---: | ---: |
| 4,095 | 75.165 ms | 41.008 ms | 61,354 / 61,425 |
| 4,096 | 75.856 ms | 39.718 ms | 61,366 / 61,440 |
| 8,192 | 300.582 ms | 87.002 ms | 122,700 / 122,880 |

## Reproduce on Windows

```powershell
$probe = Join-Path $env:LOCALAPPDATA 'Temp/umap-approximate-probe'
New-Item -ItemType Directory -Force $probe | Out-Null
$env:UV_CACHE_DIR = Join-Path $probe 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $probe 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $probe 'python'
$env:NUMBA_CACHE_DIR = Join-Path $probe 'numba-cache'
$env:CARGO_HOME = Join-Path $probe 'cargo-home'
$env:CARGO_TARGET_DIR = Join-Path $probe 'cargo-isolated'
$env:APPROXIMATE_SEARCH_OBSERVATIONS = Join-Path $probe 'observations.json'
uv run --project reference --locked --no-build python -I reference/probe_approximate_search.py --output $env:APPROXIMATE_SEARCH_OBSERVATIONS
cargo test --locked -j 2 -p umap-core --test approximate_neighbors
cargo test --locked -j 2 -p umap-core --test approximate_neighbors -- --ignored --nocapture
```
