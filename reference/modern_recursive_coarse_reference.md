# Recursive coarse optimization in UMAP 0.6.0

The [fixture generator](generate_modern_recursive_coarse_fixture.py) runs the
pinned wheel through public `UMAP.fit` and `fit_transform`. Both methods agree
on every observed CSR graph, RNG checkpoint and coordinate bit on each host.
Fits with `n_epochs=[0]` isolate recursive initialization. Positive outer epoch
lists measure the handoff to full-range Adam, projected Adam and momentum.

## Schedule

- The default first-level budget is 50. Each child budget is
  `floor(parent_budget * sqrt(2))`. For first-level budgets 0, 1 and 2 the
  deeper budgets stay unchanged. A budget of 3 gives 4, 5 and 7 at successive
  levels. A budget of 48 gives 67 and 94 over the first two descents. The
  default gives 70 and 98.
- The caller RNG draws the PCA sketch and one `randint(-2147483647,
  2147483646, 3)` per contraction while descending. Optimization runs on
  unwind, from the smallest contracted graph to the largest. For example,
  the four-level three-epoch case runs 7 epochs on 2 rows, 5 on 3 rows, 4
  on 7 rows, and 3 on 14 rows. After expansion the outer fit draws its own
  seed triplet. The fixture compares the trace through coarsening as well
  as the final cursor, cached Gaussian and next four uint32 words.
- Coarse epochs use Adam, learning rate `0.5`, repulsion `4`, a negative
  sample rate of `1`, and allow graph neighbors as negative candidates.
  `recursive_repulsion_strength` and `recursive_negative_sample_rate`
  override the coarse values. Outer optimizer settings apply after the
  last expansion.
- The default `strong_to_one` curve uses the fourth root of both
  coefficients at each contraction. After `d` contractions the exponent
  is `0.25 ** d`. The `original` schedule keeps the supplied coefficients.
  Under `good_initialization_policy="heuristic"`, the refinement schedule
  was observed with two or more completed lower levels. The `never` policy
  uses ordinary epochs. `always` and `available` select refinement on the
  tested PCA anchors.
- By default each epoch shuffles once and draws a projection direction.
  `scaled_coarse` selects this branch for scales below one. At scale one
  or above, or with `coarse_n` or `fixed_200k` on these graph sizes, Adam
  advances two permutations per epoch and dispatches with the second.
  Both branches advance their RNG at negative rate zero.

## Verification and residuals

Results on Windows and Ubuntu WSL agree. Each host ran CPython 3.13.12
with the pinned reference libraries and one numerical thread. Across 43
fit recipes and each public method, all CSR graphs and caller RNG traces
match exactly. Of the 101 final and checkpoint coordinate arrays per
method, 89 are within the asserted tolerances and none match every bit.
The largest bounded error is `8.8692e-5` at outer epoch zero on the
three-level six-epoch probe. The largest bounded error after a positive
outer epoch is `3.0995e-6`.

The archived `holdout-three-coarse-epochs` now matches within `4.7684e-7`
at outer epoch zero and `3.0995e-6` at its last epoch. Its old residuals
were `0.290781` and `0.691123`. The archived nonunit curve case
`holdout-two-coarse-epochs` now matches within `1.073e-6` throughout.

| Fixture case with an open coordinate difference | Maximum absolute error on both hosts |
| --- | ---: |
| `ratio-three-stars-budget-4` | `0.337257` |
| `ratio-three-stars-full-range` | `0.213777` |
| `ratio-three-stars-budget-11` | `2.576397` |
| `ratio-three-stars-budget-None` | `3.663600` |
| `ratio-three-stars-positive-only-eleven` | `0.006819` |
| `four-level-budget-3` | `4.828363` |

The star hierarchy is within `1.431e-6` with budgets 0 and 3, and
within `7.153e-7` with budget 5 at coarse rate zero. The four-level
case is within `3.100e-6` through budget 2 and `9.925e-6` with budget 3
at coarse rate zero. These controls locate the first large differences
at the first effective coarse repulsion. The star difference also
occurs in the full-range branch. The 11-epoch star attraction-only
control has a separate, smaller difference. The evidence leaves edge
timing, weighting and negative target selection in these hierarchies
open. Other layouts can already diverge before optimization. For example,
with 31 rows and four features from `RandomState(1061).uniform(-5, 7)`,
`n_neighbors=5`, ratio 3, threshold 3, fit seed 42, `a=b=1`, and zero
inner and outer epochs, the maximum error was `3.534005` on Windows.

Both hosts passed the format check, warning-denied Clippy and 187 locked
Rust tests, with 6 ignored. The installed-wheel suite passed 2,009 tests
with 12 expected failures for the six coordinate residuals across the
two fit methods.

## Reproduce

Use an external build directory, set `CARGO_BUILD_JOBS=2` and set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMBA_NUM_THREADS` to `1`. Use separate reference and project
environments and choose the fixture suffix for the host.

```sh
uv run --project reference --locked python -B reference/generate_modern_recursive_coarse_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-coarse-ubuntu-x86_64.json
uv sync --locked
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked --quiet
uv run --no-sync maturin build --release --locked --interpreter python --out "$wheel_dir"
uv pip install --python "$project_python" --no-deps --reinstall "$wheel_file"
uv run --no-sync python -B -m pytest -p no:cacheprovider -q
```

Run the coordinate checks with `--runxfail` to print the six open
residuals for each fit method. The graph and RNG tests run independently.

```sh
uv run --no-sync python -B -m pytest -p no:cacheprovider -q --runxfail \
  tests/test_modern_recursive_coarse.py -k 'coordinates and (stars or four-level)'
```
