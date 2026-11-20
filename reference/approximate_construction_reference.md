# Dense approximate construction at 4,095 and 4,096 rows

This follow-up compares Rust against `e29f1c6` using the pinned Windows wheel
and the same measurement harness in both revisions. The reference cases use
seven Gaussian features, data seed 57 and search seed 42. The holdout uses
eleven features, data seed 88 and search seed 73. Both use five ranks and
four held-out queries. The public calls are `NNDescent` construction,
`neighbor_graph`, `query`, and UMAP fit/transform. The probe prints only the
exception type on failure. [Baseline reference](approximate_search_native_reference.md)

## Projection, fill, and the first refinement round

- A single tree with the leaf cap one less than the row count exposes its
  root partition through ranked edges. Across the three metrics, the
  recovered unordered pivots are `(612, 2584)` in the seven-feature input
  and `(139, 1895)` in the eleven-feature holdout. Separate geometry controls
  change the dimension and the search seed one at a time. Euclidean rank pairs
  have zero crossings of the Euclidean bisector. Cosine and correlation
  pairs have zero crossings of the uncentered cosine split. Centering that
  split for correlation creates 3,133 and 1,359 crossings respectively.
  A one-index pivot perturbation also introduces crossings in both inputs.
- With eight trees, an explicit leaf size of 60 matches all 20,480
  zero-round columns from the default leaf size on each 4,096-row case.
  Caps 59 and 61 each change some columns for every metric. Rust uses a
  minimum leaf cap of 60 and the observed raw angular split for both angular
  metrics. It retains the baseline's seeded pivot generator and depth bound.
- A public `RandomState` subclass records two signed three-word draws and
  a separate scalar in `[0, 2^62)` with trees enabled. For seed 42, the
  first signed state is `[-538846105, 1273642420, 1935803229]` and the scalar
  is `1819927849474927636`. Holding that scalar fixed on a 131-row control
  preserves all 917 one-tree columns when switching seed 42 to 73. Switching
  the entire seed preserves 430 columns, and the no-tree builds share 145
  aligned columns across seeds. The no-tree model uses the first state and
  draws absolute signed Tausworthe words modulo the row count. Self-draws
  retry, and duplicate other draws consume one attempt. Rust uses that
  stream for shortages and caps attempts at twice the vacancy count plus
  32 so a degenerate state terminates. A two-leaf control with 61 rows and
  width 50 matches all 3,050 columns when fill draws only for the vacancies.
- In the fixed two-rank control, `[1, 2]` and `[0, 1]` export the same
  zero-round row `[0, 1]`. Only the former offers `1 ↔ 2` on the first
  refinement round. Rust retains the fifth non-self candidate in a width-five
  build and adds self when writing the ranked output. An explicitly supplied
  self still occupies a seed slot. The original frozen-rank checkpoint,
  candidate cap, iteration limit, and output width are covered by the existing
  tests.

At 4,096, the first five zero-round ranks of a width-six build match the
width-five export in all six cases. Dropping self from each width-six row
leaves the five live seed offers. Passing those five indices back as public
`init_graph` reproduces all 20,480 normal wheel columns after both one and
twelve rounds in every case. With the same live seed arrays, Rust matches
every first-round column. These are the later-round residuals with that
controlled starting graph. Distance pairs require both an equal index and
an equal float32 bit pattern.

| Features and metric | Frozen width-five restart, first columns | Live-seed Rust first distance pairs | Live-seed Rust round 12 columns / distance pairs |
| --- | ---: | ---: | ---: |
| 7 Euclidean | 20,228 | 16,811 | 20,473 / 16,808 |
| 7 cosine | 20,354 | 4,835 | 20,467 / 4,838 |
| 7 correlation | 19,412 | 20,480 | 20,213 / 20,213 |
| 11 Euclidean | 19,588 | 16,459 | 20,378 / 16,387 |
| 11 cosine | 19,935 | 5,790 | 20,449 / 5,773 |
| 11 correlation | 19,369 | 20,480 | 20,170 / 20,170 |

With trees disabled, Rust matches all 20,480 zero-round columns for
Euclidean and correlation in both inputs and all first-round columns for
all three metrics. Cosine zero-round columns match 16,257 and 16,310.
Cosine zero-round ordering and distance arithmetic remain open, as does the
wheel's pivot stream after the observed scalar draw. Setting wheel `delta=0`
changes at most 30 final columns across the live-seed controls. The largest
remaining Rust live-seed discrepancy is 310 final columns.

## Candidate quality and exact ranks

Training recall counts non-self directed columns shared with exhaustive
top-five search, independent of their slot positions. There are 16,380
possible hits at 4,095 and 16,384 at 4,096. Exact columns and distance pairs
compare the final native ranks to wheel ranks in order, including self.

| Rows, features, metric | Recall baseline → update / wheel | Exact columns baseline → update | Exact distance pairs baseline → update |
| --- | ---: | ---: | ---: |
| 4,095, 7, Euclidean | 14,922 → 16,107 / 16,105 | 17,699 → 19,549 | 14,675 → 16,103 |
| 4,096, 7, Euclidean | 14,951 → 16,004 / 16,132 | 17,722 → 19,344 | 14,671 → 15,950 |
| 4,095, 7, cosine | 15,315 → 16,241 / 16,216 | 18,401 → 19,949 | 4,730 → 4,797 |
| 4,096, 7, cosine | 15,267 → 16,209 / 16,206 | 18,338 → 19,860 | 4,723 → 4,792 |
| 4,095, 7, correlation | 15,790 → 15,961 / 15,864 | 18,753 → 19,248 | 18,753 → 19,248 |
| 4,096, 7, correlation | 15,746 → 15,920 / 15,799 | 18,594 → 19,224 | 18,594 → 19,224 |
| 4,095, 11, Euclidean | 13,168 → 15,407 / 15,384 | 14,774 → 17,666 | 12,138 → 14,300 |
| 4,096, 11, Euclidean | 13,016 → 15,348 / 15,363 | 14,351 → 17,438 | 11,802 → 14,157 |
| 4,095, 11, cosine | 12,834 → 15,387 / 15,302 | 14,255 → 17,241 | 5,076 → 5,414 |
| 4,096, 11, cosine | 12,848 → 15,351 / 15,280 | 14,201 → 17,230 | 5,076 → 5,405 |
| 4,095, 11, correlation | 13,393 → 14,926 / 14,822 | 14,485 → 16,589 | 14,485 → 16,589 |
| 4,096, 11, correlation | 13,410 → 14,871 / 14,842 | 14,712 → 16,451 | 14,712 → 16,451 |

## Graph support and bits

CSR support counts directed coordinates present on only one side. Weight
differences count exact float32 bit changes on common CSR coordinates. The
COO column counts include all stored ranks. Each coordinate and triple count
is out of 20. The harness also prints the held-out support and distance
matches and compares held-out support against exhaustive ranks.

| Rows, features, metric | CSR support differences baseline → update | Updated changed CSR bits / common | Ordered COO coordinates baseline → update | Ordered COO triples baseline → update |
| --- | ---: | ---: | ---: | ---: |
| 4,095, 7, Euclidean | 4,888 → 1,632 | 8,322 / 22,778 | 8 → 4 | 0 → 0 |
| 4,096, 7, Euclidean | 4,832 → 1,862 | 8,498 / 22,644 | 7 → 10 | 4 → 4 |
| 4,095, 7, cosine | 3,382 → 828 | 14,860 / 20,052 | 15 → 17 | 0 → 0 |
| 4,096, 7, cosine | 3,412 → 942 | 14,830 / 20,002 | 16 → 11 | 1 → 1 |
| 4,095, 7, correlation | 2,868 → 1,950 | 2,154 / 19,312 | 15 → 10 | 15 → 10 |
| 4,096, 7, correlation | 2,978 → 1,810 | 2,236 / 19,398 | 15 → 15 | 15 → 15 |
| 4,095, 11, Euclidean | 11,044 → 5,120 | 10,150 / 22,462 | 7 → 9 | 0 → 3 |
| 4,096, 11, Euclidean | 11,872 → 5,506 | 10,244 / 22,302 | 7 → 12 | 3 → 8 |
| 4,095, 11, cosine | 10,412 → 4,714 | 13,174 / 18,302 | 5 → 3 | 0 → 0 |
| 4,096, 11, cosine | 10,532 → 4,940 | 13,136 / 18,228 | 2 → 4 | 0 → 0 |
| 4,095, 11, correlation | 9,680 → 5,784 | 5,922 / 17,824 | 9 → 8 | 5 → 0 |
| 4,096, 11, correlation | 9,354 → 6,010 | 6,068 / 17,746 | 6 → 10 | 0 → 5 |

All six 4,095-row automatic cases match the exact native graph and all
20 ordered query triples. At 4,096, automatic and forced wheel graphs are
identical for the reference and holdout inputs. The approximate results above
still have sizeable CSR support and ordered COO residuals. The Python
estimator keeps its existing routing.

Across the 240 forced held-out coordinates, the baseline selected 67 outside
the exhaustive ranks and the update selected 33. The wheel selected 73.
This measures candidate quality against the exhaustive path. The ordered
match to the wheel varies across cases, as shown above.

## Native timing

Release measurements use the median of nine warm fits. The query median
uses 128 four-query batches per fit and excludes graph assembly. Timings on
Windows on a Core Ultra 9 285K varied by case. The table gives the 4,096-row
medians in milliseconds for construction and microseconds for each four-query
batch.

| Features and metric | Build baseline → update, ms | Query baseline → update, µs |
| --- | ---: | ---: |
| 7 Euclidean | 15.82 → 17.07 | 5.84 → 5.03 |
| 7 cosine | 17.28 → 17.94 | 7.36 → 5.13 |
| 7 correlation | 20.99 → 27.46 | 6.43 → 7.07 |
| 11 Euclidean | 25.91 → 19.87 | 6.39 → 7.28 |
| 11 cosine | 21.29 → 27.05 | 6.30 → 7.02 |
| 11 correlation | 25.89 → 28.85 | 6.41 → 8.37 |

## Reproduction on Windows

```powershell
$scratch = Join-Path $PWD 'target/approximate-followup'
New-Item -ItemType Directory -Force $scratch | Out-Null
$env:UV_CACHE_DIR = Join-Path $scratch 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $scratch 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $scratch 'python'
$env:NUMBA_CACHE_DIR = Join-Path $scratch 'numba-cache'
$env:CARGO_HOME = Join-Path $scratch 'cargo-home'
$env:APPROXIMATE_SEARCH_OBSERVATIONS = Join-Path $scratch 'observations.json'
uv run --project reference --locked --no-build python -I reference/probe_approximate_construction.py --output $env:APPROXIMATE_SEARCH_OBSERVATIONS

$baseline = Join-Path $scratch 'baseline'
New-Item -ItemType Directory -Force $baseline | Out-Null
$archive = Join-Path $scratch 'baseline.tar'
git archive --format=tar --output=$archive e29f1c6
tar -xf $archive -C $baseline
Copy-Item -LiteralPath 'crates/umap-core/tests/approximate_neighbors.rs' -Destination (Join-Path $baseline 'crates/umap-core/tests/approximate_neighbors.rs')
$env:APPROXIMATE_SEARCH_TIMINGS = '1'
$env:CARGO_TARGET_DIR = Join-Path $scratch 'cargo-baseline'
cargo test --manifest-path (Join-Path $baseline 'Cargo.toml') --locked --release -j 2 -p umap-core --test approximate_neighbors -- --ignored --nocapture
$env:CARGO_TARGET_DIR = Join-Path $scratch 'cargo-updated'
cargo test --locked --release -j 2 -p umap-core --test approximate_neighbors -- --ignored --nocapture
cargo test --locked -j 2 -p umap-core --lib replay_initial_offers -- --ignored --nocapture
```
