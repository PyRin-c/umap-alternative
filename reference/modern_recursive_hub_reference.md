# Tied recursive hubs in UMAP 0.6.0

The [coarse transition follow-up](modern_recursive_transition_reference.md)
resolves the four-level repulsive mismatch and isolates the remaining
attraction-only residual.

The [fixture generator](generate_modern_recursive_hub_fixture.py) uses the
pinned public `label_propagation_init`, `UMAP.fit` and `UMAP.fit_transform`
interfaces. The one-contraction controls supply a fixed CSR, anchor and
caller RandomState. They record the input graph, every seed and shuffle
checkpoint, the post-call RNG state and the expanded coordinates. These
controls cover a tied degree order and one membership one float32 step
below the tie, with projected and full-range sampling, zero and positive
coarse budgets, row permutations, and transition and fuzzy-union weights.

## Hub ordering

Hub labels follow the last entries of an ascending degree sort. The
wheel's order on ties differs from both NumPy's stable sort and its
default array sort on these hosts. The direct cycles distinguish an
insertion-sorted partition of 15 rows from the first quicksort split at
16 rows. The longer probes agree with a middle/first/last median pivot,
two-ended partition, and insertion sort for partitions below 16 rows.
Rust now computes this order before assigning cluster labels.

With stable ties the synthetic star's negative updates differed by up to
`0.168007` in the direct probes even though the graph and RNG matched.
At zero epochs the same graph was already within `4.769e-7`. Permuting
hub labels also permutes the coarse CSR rows and the ordinals that pick
negative targets. This explains how the placement check could pass before
the first repulsive update. All 13 direct cases now match input CSR and
RNG bits and stay within `8.345e-7` on both hosts. A separate replay of
the public `simplicial_set_embedding` on frozen 2-, 3-, 7-, 9- and 14-row
stage inputs matched the native Adam coordinates bit for bit when both
started from the same float32 layout and RNG. This control supplied zero
explicit-layout jitter and retained the resulting coincident rows.

## Fit results

Windows and Ubuntu WSL agree. All 43 archived fit recipes and the four
additional controls match CSR and caller RNG bits through both fit
methods. Four prior expected coordinate failures now pass.

Both hosts passed formatting, warning-denied Clippy and 192 locked Rust
tests with 7 ignored. Each installed-wheel suite passed 2,110 tests with
4 expected failures for the two remaining recipes through both methods.
The Windows and Ubuntu reference fixtures contain identical bytes.

| Archived case | Old error | Maximum absolute error on each host |
| --- | ---: | ---: |
| `ratio-three-stars-budget-4` | `0.337257` | `2.027e-6` |
| `ratio-three-stars-budget-11` | `2.576397` | `2.169e-6` |
| `ratio-three-stars-budget-None` | `3.663600` | `1.324e-5` |
| `ratio-three-stars-full-range` | `0.213777` | `8.345e-7` |
| `ratio-three-stars-positive-only-eleven` | `0.006819` | `0.006738` |
| `four-level-budget-3` | `4.828363` | `4.828363` |

The eleven-epoch attraction-only star is sensitive to coarse transition
arithmetic near half-weight ties. Promoting normalization and union to
float64 reduced its error to `6.009e-5` in a diagnostic replay and
exceeded the existing bound on the 37-row three-level fit. A candidate
that divided directly in float32 and promoted only the union exceeded
the epoch-zero bound of the archived 21-row transition case. The patch
keeps the existing transition arithmetic and the attraction-only xfail.

Stopping the four-level recipe after 1, 2 or 3 contractions at zero
coarse epochs reproduces the corresponding placement. With positive
coarse budget 3, the transition fit already differs by `0.010235` when
the smallest graph has 7 rows and by `0.366991` when it has 3. The
four-level fuzzy-union control matches within `1.550e-6`, and the archived
four-level transition fit with rate zero matches within `9.925e-6`. The
cause of the transition-specific divergence under repulsion remains open.
To reproduce the shallower probes, set the budget to 3 in the generator's
threshold-7 and threshold-3 transition controls and rerun the fit checks.

## Reproduce

Use the external storage and one-thread numerical settings in the
[previous verification commands](modern_recursive_coarse_reference.md#reproduce).
Build and install the locked project wheel in a separate environment.
Replace `ubuntu` with `windows` for the Windows fixture.

```sh
uv run --project reference --locked python -B reference/generate_modern_recursive_hub_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-hubs-ubuntu-x86_64.json
uv run --no-sync python -B -m pytest -p no:cacheprovider -q tests/test_modern_recursive_hubs.py
uv run --no-sync python -B -m pytest -p no:cacheprovider -q --runxfail \
  tests/test_modern_recursive_coarse.py -k 'coordinates and (stars or four-level)'
```
