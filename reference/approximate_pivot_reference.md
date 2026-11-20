# Approximate projection probes and Euclidean distance bits

The follow-up recovers the unsigned root and child words and replays the
in-place partition order. [Projection word reference](projection_words_reference.md)

The baseline is integrated `bd6794e`. The Windows probe reuses the
construction inputs for caller seeds 42 and 73 and adds seeds 0 and 19 at
both 4,095 and 4,096 rows. It calls the published `NNDescent` constructor,
`neighbor_graph`, `query`, and UMAP fit and transform. A recording
`RandomState` supplies controlled draws through the constructor's public
`random_state` parameter. The probe reports wheel exceptions by type.
[Construction fixture and baseline](approximate_construction_reference.md)

## What the projection controls distinguish

**Caller state.** The recorded calls consume two signed three-word draws
followed by one scalar from `[0, 2^62)` when trees are enabled. Rust replay
reproduces all three draws for caller seeds 0, 1, 19, 42, 73, and `2^32-1`.
The scalar uses the low 30 bits of the next MT19937 word as its high word,
then the full following word. Sending scalar 1 and `2^32+1` into one tree
at 131 rows changes the unique recovered pair from `(17, 21)` to
`(110, 121)`.

**Pivot counterexample.** When a single tree of depth one produces two
connected components, the probe tests every Euclidean bisector and records
compatible unordered pivot pairs. With caller seed 42's
scalar `1819927849474927636`, the wheel pair is `(27, 117)` at 131 rows and
`(63, 79)` at 163 rows. These disjoint pairs rule out drawing an 8-bit
masked first index with rejection from the same stream over original row
order. The scaled index intervals are also disjoint, which rules out scaling
one common uniform first draw directly by the row count. At 131 rows the
native MT19937 indices are `(49, 104)` and direct bounded PCG64 draws are
`(52, 95)`. The order of the wheel's two pivots and the mapping from the
scalar remain open. Rust retains its existing pivot generator.

After swapping the two pivot rows and relabeling the result, all 917 rank
slots agree at depth one for each of three scalar seeds. At depth two the
same swaps change 575, 479, and 462 slots for scalar 0, caller 42's scalar,
and caller 73's scalar respectively. This exposes deeper order sensitivity
through the public API. The observations include leaf fill and leave the
initial ordered pair ambiguous.

**Tree count and ties.** The boundary probe checks the last rank and
retention of earlier offers when increasing the tree count from 1 to 2,
4, and 8. With a leaf cap of 60, all 12 cases have full ranks. None of their
last-rank distances worsens and no earlier offer strictly within the updated
radius disappears. This is consistent with adding trees to the same prefix.
In the equal-projection controls, swapping the two pivot vectors and reversing
the feature values of tied rows preserve assignments relative to the pivot
index. Moving the first tied row off the plane leaves all other assignments
unchanged across the 131-, 197-, and 251-row controls. The draw sequence
therefore does not advance only on ties. One draw per row and a keyed
assignment per row both fit the result.

**Vacancies.** Replacing the first signed draw changes 772 of 917 slots
with trees disabled. With one tree and complete width-seven leaf offers,
it changes zero. With one tree and width 95, it changes 7,766 of 12,445
slots. Replacing only the second signed draw changes zero in all three
construction controls.

**Later rounds.** On the two 4,096-row correlation holdouts, feeding the
live five non-self offers obtained from a width-six export reproduces every
normal wheel column after rounds 1, 2, 3, and 12 with `delta=0`. Restarting
from the visible width-five export differs by 961 and 1,052 positions after
one round and by 384 and 1,092 after twelve. These are public-wheel control
replays. The native seed measurements below build their own trees and ranks
from features and the caller seed.

## Independent Euclidean improvement

The approximate Euclidean path rounds each coordinate difference to
float32. It uses fused square accumulation over groups of 32 and then 4,
followed by scalar fused additions and the float32 square root. The
32-coordinate stage combines four eight-lane accumulators in adjacent pairs
and reduces across the lane halves. The resulting sum enters lane zero of
the four-coordinate stage. The four lanes reduce as `(0+2)+(1+3)`. Rust
uses the same ordered arithmetic via a portable path and via hardware FMA
when AVX and FMA are available on x86-64.

The all-pairs leaf control checks 841 bits in each of 20 dimensions from 1
through 127. Exact distances rise from 13,182 to all 16,820 values. The portable and
hardware paths agree on this host. The existing precise Euclidean metric
continues to use its own accumulation order.

### Native seed ranks

Shared support below counts directed non-self pairs against the wheel's
zero-round export. Bit counts compare the two arithmetic paths on exactly
those shared coordinates. Each width-five output has at most 16,380 non-self
slots at 4,095 rows and 16,384 at 4,096. Neither path receives wheel seed
ranks to generate native offers.

| Rows | Caller seed | Features | Shared support | Equal rank positions | Equal distance bits baseline → update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 42 | 7 | 14,425 | 12,809 | 11,234 → 14,425 |
| 4,096 | 42 | 7 | 13,918 | 11,771 | 10,821 → 13,918 |
| 4,095 | 73 | 11 | 11,495 | 8,283 | 8,667 → 11,495 |
| 4,096 | 73 | 11 | 11,235 | 7,951 | 8,463 → 11,235 |
| 4,095 | 0 | 7 | 14,229 | 12,486 | 11,034 → 14,229 |
| 4,096 | 0 | 7 | 14,383 | 12,757 | 11,177 → 14,383 |
| 4,095 | 19 | 11 | 11,350 | 8,097 | 8,594 → 11,350 |
| 4,096 | 19 | 11 | 11,543 | 8,356 | 8,695 → 11,543 |

With seed 0 at 4,096 rows, the native first round shares 15,728 non-self
coordinates and agrees on all their distance bits. With seed 19 it shares and
matches 14,069. The seed support gaps and the aligned rank counts measure
the remaining construction differences independently of arithmetic.

### Final ranks and graphs

The table reports approximate training ranks and graphs for Euclidean
inputs. The baseline and update have the same measured CSR support difference
counts and ordered column match counts. Every aligned final Euclidean rank
agrees in distance bits after the update. CSR weight changes are counted on
common coordinates and COO triples include both the ordered coordinate and
its weight bits. Each case uses four held-out queries and has 20 COO slots.

| Rows | Caller seed | CSR support differences | Aligned distance bits baseline → update | Changed CSR bits baseline → update / common | Ordered COO triples baseline → update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 42 | 1,632 | 16,103 → 19,549 | 8,322 → 1,866 / 22,778 | 0 → 0 |
| 4,096 | 42 | 1,862 | 15,950 → 19,344 | 8,498 → 2,218 / 22,644 | 4 → 5 |
| 4,095 | 73 | 5,120 | 14,300 → 17,666 | 10,150 → 5,002 / 22,462 | 3 → 5 |
| 4,096 | 73 | 5,506 | 14,157 → 17,438 | 10,244 → 5,232 / 22,302 | 8 → 10 |
| 4,095 | 0 | 1,604 | 16,078 → 19,569 | 8,354 → 1,832 / 22,762 | 4 → 5 |
| 4,096 | 0 | 1,566 | 16,114 → 19,552 | 8,394 → 1,852 / 22,806 | 9 → 10 |
| 4,095 | 19 | 5,438 | 14,161 → 17,417 | 10,212 → 5,238 / 22,338 | 7 → 10 |
| 4,096 | 19 | 5,138 | 14,315 → 17,606 | 10,056 → 4,996 / 22,488 | 0 → 0 |

The exact automatic path at 4,095 agrees with the wheel on all measured CSR
and COO coordinates and bits. At 4,096, automatic and forced wheel outputs
agree. The native approximate graphs still have support and ordered COO
residuals. The cosine and correlation comparison counts reproduce the
baseline. The Python estimator continues to use its existing routing.

## Release timing

These Windows timings use a Core Ultra 9 285K with the AVX/FMA path selected.
The harness takes the median of nine fits and times 128 four-query batches
per fit. Graph assembly and public-wheel calls are outside the timed regions.
Construction varies by case and all eight query samples took longer in this
run.

| Rows | Caller seed | Build baseline → update, ms | Four-query batch baseline → update, µs |
| ---: | ---: | ---: | ---: |
| 4,095 | 42 | 18.02 → 22.82 | 5.00 → 6.52 |
| 4,096 | 42 | 16.99 → 21.47 | 4.96 → 5.92 |
| 4,095 | 73 | 20.17 → 22.07 | 6.71 → 7.49 |
| 4,096 | 73 | 23.19 → 23.04 | 7.64 → 8.30 |
| 4,095 | 0 | 17.04 → 18.90 | 4.94 → 5.65 |
| 4,096 | 0 | 20.68 → 19.27 | 5.12 → 5.73 |
| 4,095 | 19 | 19.83 → 21.42 | 7.24 → 8.18 |
| 4,096 | 19 | 19.64 → 20.56 | 6.58 → 7.08 |

## Reproduction on Windows

```powershell
$scratch = Join-Path $env:LOCALAPPDATA 'Temp/umap-pivot-stream'
New-Item -ItemType Directory -Force $scratch | Out-Null
$env:UV_CACHE_DIR = Join-Path $scratch 'uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $scratch 'wheel-venv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $scratch 'python'
$env:NUMBA_CACHE_DIR = Join-Path $scratch 'numba-cache'
$env:CARGO_HOME = Join-Path $scratch 'cargo-home'
$original = Join-Path $scratch 'construction.json'
$holdout = Join-Path $scratch 'pivots.json'
uv run --project reference --locked --no-build python -I reference/probe_approximate_construction.py --output $original
uv run --project reference --locked --no-build python -I reference/probe_approximate_pivots.py --output $holdout

$baseline = Join-Path $scratch 'baseline'
New-Item -ItemType Directory -Force $baseline | Out-Null
$archive = Join-Path $scratch 'baseline.tar'
git archive --format=tar --output=$archive bd6794e
tar -xf $archive -C $baseline
$env:APPROXIMATE_SEARCH_TIMINGS = '1'
foreach ($fixture in @($original, $holdout)) {
    $env:APPROXIMATE_SEARCH_OBSERVATIONS = $fixture
    $env:CARGO_TARGET_DIR = Join-Path $scratch 'cargo-baseline'
    cargo test --manifest-path (Join-Path $baseline 'Cargo.toml') --locked --release -j 2 -p umap-core --test approximate_neighbors -- --ignored --nocapture
    $env:CARGO_TARGET_DIR = Join-Path $scratch 'cargo-updated'
    cargo test --locked --release -j 2 -p umap-core --test approximate_neighbors -- --ignored --nocapture
    cargo test --locked --release -j 2 -p umap-core --lib measure_native_seed_ranks -- --ignored --nocapture
}
$env:APPROXIMATE_SEARCH_OBSERVATIONS = $original
cargo test --locked --release -j 2 -p umap-core --lib replay_initial_offers -- --ignored --nocapture
```
