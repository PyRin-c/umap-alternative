# Recursive coarse edge positions in UMAP 0.6.0

The [star placement follow-up](modern_recursive_star_reference.md) brings
the attraction-only recipe below `3e-5` maximum coordinate error.

The [fixture generator](generate_modern_recursive_transition_fixture.py)
records public `label_propagation_init`, `UMAP.fit` and `fit_transform`
observations. The supplied-graph controls fix the anchor, graph and seed.
They introduce a weak cross-part edge, break hub ties by changing leaf
weights, and cover projected and full-range sampling, rates 0, 1 and 5,
and one- and two-level fits. The estimator controls stop the dense
hierarchy at different levels and bracket the eleven-epoch star. An
exploratory invocation emitted ordinary wheel warning lines. The probe
was rerun with warnings suppressed, and the reported rule uses the
coordinate and RNG outputs.

## Inactive memberships occupy coarse edge positions

At rate one, the first negative updates occur in epoch 3 with zero-based
numbering. In the dense fixture stopped after one contraction, the first
coordinate difference appears at coarse budget 4. A budget of 3 and a
rate-zero budget of 4 both agree. With two contractions, root budget 3
already runs four epochs on the seven-row child and exposes the same
difference.

Using the public `recursive_init` function, the active negative targets
were recovered by overriding the caller RNG's shuffle result with the
identity and three cyclic shifts. The overrides still consumed the
ordinary shuffle draws. Both affected edges matched the original coarse
CSR ordinals on every shift. For the 14-row graph, the wheel used slots
`(12, 10)` while the compacted CSR gave `(12, 6)`. For the seven-row
graph it used `(0, 3)` while the compacted CSR gave `(3, 2)`. Supplied
orders also reproduced the wheel's unforced runs.

The recursive sampler now saves each retained edge's original CSR
position. It uses that position for both the initial candidate slot and
the stride. The weak edges remain inactive in the accumulation schedule.
The outer fit still uses compacted CSR positions. The native regression
checks two draws per edge and verifies that both modes have the same
positive schedule.

## Before and after

Coordinates are compared by maximum absolute error on Windows and
Ubuntu WSL. Both hosts produce the same measurements. All 12 new direct
controls match the CSR and full traced RNG checkpoints bit for bit.
All 15 new estimator recipes match the CSR and caller RNG bits through
both fit methods, as do the archived coarse recipes.

| Case | Before | After |
| --- | ---: | ---: |
| One contraction, budget 4, rate 0 | `2.384e-7` | `2.384e-7` |
| One contraction, budget 4, rate 1 | `0.00639057` | `7.153e-7` |
| Two contractions, budget 3, rate 1 | `0.0102345` | `4.769e-7` |
| Four contractions, budget 3, rate 1 | `4.82836` | `2.677e-5` |
| Four contractions, budget 4, rate 1 | `1.09637` | `4.769e-7` |
| Tied direct stage, full range, budget 4, rate 1 | `0.134474` | `3.577e-7` |
| Untied two-level direct stage, full range, budget 6, rate 5 | `1.75107` | `2.623e-5` |
| Attraction-only star, two contractions, budget 11 | `0.00673771` | `0.00673771` |

The four-level fit is now checked at `3e-5`. Its earlier budgets and
the other verified coordinate bounds continue to pass. The largest
error among the 12 direct controls is `2.623e-5` in the longer,
rate-five case. The remaining 11 are within `1.193e-6`.

The reference fixtures contain identical bytes on both hosts. Each
installed-wheel suite went from 2,443 passing tests and four expected
failures at `dd24c32` to 2,487 passing tests and two expected failures.
Formatting and warning-denied Clippy passed on both hosts. The locked
Rust suites passed 195 tests each, with seven ignored.

## Attraction-only star

With one contraction, budgets 10, 11 and 12 all agree within `9.537e-7`.
With two contractions, budgets 10 and 12 agree within `8.286e-6` and
`2.385e-6`. The budget-11 residual persists with the same fixed anchor
supplied through the public `label_propagation_init` API, where the raw
layout differs by `7.553e-5` before outer rescaling. A separate random
anchor on the same graph matches within `1.267e-7` at budget 11.

At budget 11 the three-row optimizer runs 15 epochs. Replaying the
nine-row optimizer once with the reconstructed placement and once with
the placement from the pinned public child call starts with a maximum
difference of `2.236e-7`. The first effective attractive update raises
it to `2.291e-4`. At the most affected coordinate the summed attractive
forces are `-4.256e-5` and `-4.273e-5`. Reusing the observed child
placement brings the raw parent layout error down to `2.784e-6`.
These probes identify amplification of near-coincident coordinates at
the child-to-parent boundary. They leave the exact placement rounding
open. Arithmetic variants that improved this recipe also exceeded
existing tolerances, so its coordinate xfail remains.

## Reproduce

Use separate pinned reference and project environments, one numerical
thread and at most two Cargo build jobs. Keep Cargo targets and wheel
output in external storage. Replace `ubuntu` with `windows` on Windows.

```sh
uv run --project reference --locked python -B -W ignore reference/generate_modern_recursive_transition_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-modern-recursive-transition-ubuntu-x86_64.json
cargo fmt --all -- --check
uv run --no-sync cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --no-sync cargo test --workspace --locked --quiet
uv run --no-sync maturin build --release --locked --interpreter python --out "$wheel_dir"
uv pip install --python "$project_python" --no-deps --reinstall "$wheel_file"
uv run --no-sync python -B -m pytest -p no:cacheprovider -q
uv run --no-sync python -B -m pytest -p no:cacheprovider -q --runxfail \
  tests/test_modern_recursive_coarse.py -k 'coordinates and positive-only-eleven'
```
