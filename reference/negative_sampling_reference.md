# Public UMAP negative sampling reference

`generate_negative_sampling_probe.py` uses `UMAP(...).fit_transform`, the
fitted `graph_`, and the public state of the caller-supplied NumPy
`RandomState`. The fixture records exact float32 coordinate and graph bits from
the official `umap-learn==0.6.0` wheel on Windows x86_64. Every fit is repeated
with fresh input arrays and a fresh RNG in the same initial state.

Reproduce from the repository root using the existing pinned interpreter. The
recorded run used CPython 3.13.12 and NumPy 2.5.5.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'umap-negative-observation-numba'
& '<main-checkout>/reference/.venv/Scripts/python.exe' -B `
  reference/generate_negative_sampling_probe.py `
  --output reference/fixtures/umap-learn-0.6.0-negative-sampling-windows-x86_64.json
```

## Reading indices from coordinates

The main precomputed input has six rows and three isolated mutual pairs. Each
retained positive edge has weight 1. At `n_epochs=0` and `1` the base public
coordinates are bit-identical. The `n_epochs=2`, `repulsion_strength=0`
control agrees with a single attractive pass at learning rate `0.1`. For each
positive edge, the replay applies that attraction and then tries all six rows
as the destination of each negative sample. Negatives update the head only.
After a component's last edge, its public final coordinates constrain the
choices before the replay advances to the next component. Row order comes
from the public CSR graph. The replay uses float64 force arithmetic and
rounds each write to float32, with an absolute coordinate tolerance of
`2e-6`. The JSON preserves the fit bits and the replay residual separately.

At `n_epochs=2`, rate 2 exposes one draw per edge. Its decode is unique across
the six candidates. Rate 3 exposes two draws per edge. For seed 42, 16 ordered
replays fit the base rate-3 coordinates within tolerance. Exactly one begins
with the independently decoded rate-2 draws. The same cross-rate prefix
rule produces a unique match in each rate-3 case. The base rate-2 replay
matches every coordinate exactly. The prefix-constrained base rate-3 replay
has a maximum residual of `1.862645149230957e-9`.

| Head | Seed 42, rate 2 | Seed 42, rate 3 with prefix | Seed 7, rate 3 with prefix |
| ---: | ---: | :--- | :--- |
| 0 | 4 | 4, 4 | 4, 5 |
| 1 | 0 | 0, 0 | 2, 1 |
| 2 | 0 | 0, 5 | 2, 4 |
| 3 | 4 | 4, 1 | 4, 0 |
| 4 | 0 | 0, 2 | 4, 5 |
| 5 | 2 | 2, 1 | 4, 0 |

- **Edge order.** Reversing all CSR edges and doing all attractions first
  each yielded zero matches for the seed-42 base graph and for its last pair
  alone, at rates 2 and 3.
- **Draws are independent of earlier heads' work in these cases.** Setting
  `disconnection_distance=1.5` removes heads 0 through 3 from the edge
  schedule. The retained heads still decode to `[0, 2]` and `[2, 1]` at
  rate 3 for seed 42, and `[4, 5]` and `[4, 0]` for seed 7. For seed 42 a
  global sequential stream advanced only by executed draws would give
  the trimmed rate-2 run the base run's first two indices `[4, 0]`. The
  trimmed run gives `[0, 2]`. Disconnected targets remain eligible in the
  decode even though their returned coordinates are NaN. The seed-7 head 4
  sequence includes a self-target, and the seed-42 head 0 sequence repeats a
  target.
- **Draw position follows the head across edges.** In the three-vertex
  component, seed-42 head 2 draws `0` on `2 -> 0` and then `5` on `2 -> 1`
  at rate 2. Removing `2 -> 0` leaves `2 -> 1` with draw `0`, the head's
  first choice also seen in the isolated-pair runs. An independently seeded
  stream for a fixed ordered edge would preserve its first choice after
  the other edge is removed.
- **Draw position carries across epochs.** At rate 1, `n_epochs=3` exposes
  the same first indices as the one-draw rate-2 run despite the intervening
  attractive pass. At rate 2, extending the run to three epochs exposes
  two more draws per head. Of 32 candidates for that extra pass, one starts
  with the second draw inferred from rate 3. In head order its extra draws
  are `[4, 4]`, `[0, 4]`, `[5, 1]`, `[1, 3]`, `[2, 1]`, and `[1, 5]`.
  Zero candidates start every head over at its first draw.
- **The first initial coordinate affects choices.** With seed 42, changing
  head 4's normalized `x` from `6` to `1.75` changes its first two choices
  from `[0, 2]` to `[4, 5]` and preserves the other heads' indices. Changing
  only head 4's `y` from `3` to `4.75` preserves the choices. Giving heads 4
  and 5 identical initial `x=10` and different `y` yields `[2, 1]` for both.
  Applying `2 * init + [5, -6]` preserves the normalized zero-epoch bits
  and the rate-2 embedding bits.

## Index reduction

The count sweep keeps heads 0 and 1 at `[0, 0]` and `[1, 4]`, fixes their
unit-weight mutual pair, and fills the remaining rows with distinct
disconnected coordinates. The sorted pair is the whole positive schedule.
Across the nine coprime sizes `5, 7, 11, 13, 17, 19, 23, 29, 31`, a common
integer whose nonnegative remainder at each size equals the detected index
would be fixed modulo `33426748355`. This modulus exceeds `2**32`, so there
can be at most one signed 32-bit representative per seed and head.

| Seed | Head 0, first representative | Head 1, first representative |
| ---: | ---: | ---: |
| 42 | 1512524284 | -1574483460 |
| 7 | -1475731520 | -1609949248 |

All four representatives predict the independently measured choices at held-out
sizes `6, 8, 10, 34`. The negative representatives have no congruent unsigned
32-bit value over the nine coprime sizes. This rules out a size-independent
unsigned 32-bit value reduced by ordinary modulo for those cases. The
signed-remainder model leaves room for other algorithms with the same observed
residues.

## Caller RNG initialization

For distinct explicit init, the full `RandomState` after the fit matches a
copy advanced by three raw MT19937 words in ordinary cases. This also holds at
zero epochs, for the trimmed graph, and across the recorded epoch and rate
changes. Seed 42 consumes `[1608637542, 3421126067, 4083286876]`. Seed 7
consumes `[327741615, 976413892, 3349725721]`. The caller RNG is therefore
insufficient as a direct sequential source of all optimizer negatives.

Injecting raw `0xfffffffc` at cursor 1 advances to cursor 4 and produces
different embedding bits from explicitly consuming the injected word before
the fit. Each of `0xfffffffd`, `0xfffffffe`, and `0xffffffff` advances to
cursor 5 and produces exactly the same embedding bits as its skip control.
The resulting RNG state in every boundary case also matches the public NumPy
call `randint(-2147483647, 2147483646, size=3)` on a copy of the same initial
state. This identifies a compatible rejection width of `4294967293` raw
values, while the numerical offset of the accepted values remains open.

With `init="random"`, seed 42 advances by 27 raw words. An explicit init
made with `RandomState(42).uniform(-10, 10, (6, 2)).astype(float32)` starts
at the same normalized coordinates. At rate 2, passing that explicit array
and an RNG advanced by the same uniform call reproduces the random-init
embedding bits and the final RNG state. The same array with an unadvanced
RNG gives first indices `[4, 2, 2, 4, 4, 4]`, versus `[2, 2, 0, 4, 4, 4]`
for random init and its advanced control. The evidence puts the three accepted
words after the 24 raw words used by the public uniform initialization in
this case.

The exact seed-word transformation, incorporation of the initial coordinate,
and generator recurrence remain open. The probes establish a behavioral
stream per head in the tested schedule, with no observation of its storage
or of initialization paths such as spectral or duplicate-row jitter.
