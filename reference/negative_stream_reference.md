# Public UMAP negative stream reconstruction

A signed-wide adaptation of the three-lane Tausworthe recurrence matches all
48 signed words reconstructed from public `umap-learn==0.6.0` coordinates. It
also predicts all 288 indices at six held-out vertex counts and all 166 decoded
indices in the earlier varied-graph and epoch probes. The candidate takes the
shift parameters from L'Ecuyer's published QuickTaus construction and tests
integer-width and seed-mapping variations against the wheel observations.
[L'Ecuyer, *Maximally Equidistributed Combined Tausworthe Generators*, 1996](https://www.accellera.org/images/eda/sv-ec/att-0986/01-PierreMaximalRandomPaper.pdf)

## Reproduce the measurements and comparison

From the repository root, use the existing pinned Windows interpreter. The
recorded run used CPython 3.13.12 and NumPy 2.5.5.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP 'umap-negative-observation-numba'
$probePython = '<main-checkout>/reference/.venv/Scripts/python.exe'
$fixture = 'reference/fixtures/umap-learn-0.6.0-negative-stream-windows-x86_64.json'
& $probePython -B reference/generate_negative_stream_fixture.py --output $fixture
& $probePython -B reference/analyze_negative_stream.py $fixture --previous-fixture `
  reference/fixtures/umap-learn-0.6.0-negative-sampling-windows-x86_64.json
```

The generator records public `fit_transform` coordinate bits, verifies the
public `graph_`, and records the caller's public NumPy MT19937 state advance.
It repeats every fit with fresh input and an identically initialized caller
RNG. The raw JSON contains these recordings and the inputs. The analyzer
decodes indices and reconstructs signed words before evaluating any RNG
candidate.

Each run has a single unit-weight mutual pair and a set of disconnected
target rows. Explicit normalized coordinates and seed state stay fixed within
a sweep. The two active rows' coordinates at `n_epochs=0` verify the input
normalization. The `n_epochs=2` runs with rates 2, 3, and 4 expose one, two, and
three negatives per edge in one active pass. For each rate, the analyzer
enumerates the two new indices in CSR order using the earlier decoded values
as a cross-rate prefix. All 360 rate-and-size checkpoints have one replay
within the established `2e-6` absolute tolerance given that prefix. The
largest coordinate residual is `2.384185791015625e-7`.

The analyzer reconstructs a candidate constant word by its nonnegative
remainders over the vertex counts `5, 7, 11, 13, 17, 19, 23, 29, 31`. Their
product is `33426748355`, so each reconstruction has at most one signed
32-bit representative. The separate counts `12, 16, 21, 32, 35, 37` are
reserved for prediction, including the independent replay of second and
third draws at each count. Earlier seed-42 and seed-7 first-word results and
six-vertex index sequences motivated the candidate. The new sweeps add full
later words and new seed, coordinate, and caller-state combinations.

## Matching rule

1. For the caller RNG's three seed values, accept raw MT19937 words below
   `0xfffffffd` and subtract `2147483647` from each accepted word. This agrees
   with the earlier experiment at the rejection boundary. With random
   initialization, the earlier fixture places these draws after the uniform
   initialization draws.
2. For each head, promote the first normalized float32 coordinate to float64,
   interpret the resulting IEEE-754 bits as a signed 64-bit integer, and add
   that integer to each of the three signed seed values. The head retains its
   own three lane values and advances them at each scheduled negative draw.
3. For a lane `z`, let `U32` reduce modulo `2**32`. Update the lane using the
   expression below. The right shift is arithmetic and the updated lane
   retains the signed wide result. The low-bit masks and shifts are listed in
   the table.

   ```text
   advance = U32((z & mask) << s)
   feedback = (U32(z << q) XOR z) >> r
   next = advance XOR feedback
   ```

   | Lane | Mask | Left feedback shift `q` | Left advance shift `s` | Right shift `r` |
   | ---: | :--- | ---: | ---: | ---: |
   | 0 | `0xfffffffe` | 13 | 12 | 19 |
   | 1 | `0xfffffff8` | 2 | 4 | 25 |
   | 2 | `0xfffffff0` | 3 | 17 | 11 |

4. XOR the updated lanes, interpret the low 32 bits as signed, and take the
   nonnegative remainder modulo the full vertex count to obtain the target.

The masks and shifts come from the published construction. In the matching
model, the XOR's unshifted feedback operand and the saved lane both keep
their wide signed values. The 32-bit left-shift results and the final signed
32-bit output are separate narrowing operations.

## Discriminators

| Candidate variant | Matches on draw 1 / 2 / 3, out of 16 each | Held-out indices, out of 288 |
| :--- | :--- | ---: |
| Signed-wide feedback and binary64 addition | 16 / 16 / 16 | 288 |
| Canonical unsigned 32-bit feedback | 1 / 1 / 1 | 75 |
| Unsigned 64-bit feedback | 16 / 12 / 12 | 257 |
| Keep the wide left advance before the XOR | 16 / 2 / 1 | 160 |
| Truncate the saved lane to `int32` | 16 / 1 / 1 | 155 |
| Add binary32 coordinate bits | 5 / 5 / 5 | 100 |
| Add the integer part of the coordinate | 5 / 5 / 5 | 121 |
| Seed values alone | 5 / 5 / 5 | 127 |
| XOR the binary64 coordinate bits | 12 / 6 / 6 | 187 |
| Seed offset `-2147483648` | 10 / 8 / 8 | 192 |
| Seed offset `-2147483646` | 6 / 6 / 6 | 170 |
| Seed offset `0` | 0 / 0 / 0 | 56 |

The following changes hold the caller state fixed or replace it with
calibrated public MT19937 inputs. All values listed are signed full words
reconstructed from coordinate observations.

- For seed 42, changing only the second head's first coordinate from `1.0`
  to `1.0000001192092896` changes its stream from
  `[-1574483460, 797748276, 517968520]` to
  `[-1573171796, 801945252, 568246940]`. Head 0 stays at
  `[1512524284, -1869814856, -86469464]`.
- Three raw `0x7fffffff` words give the `x=0` head `[0, 0, 0]`. The `x=2.5`
  head gives `[-2113929216, 822083589, 554893360]` from the same caller
  state. This tests the seed offset and an exact zero lane initialization.
- A caller state beginning with `[0xfffffffd, 0x7fffffff, 0, 0xfffffffc]`
  advances by four raw words. Its `x=0` head gives `[1048512, 3456, -469776497]`
  and its `x=8.5` head gives `[-812646464, -2097191384, 1269448883]`, as
  predicted after rejecting the first raw word.

The results validate this rule for the first three draws per head on the
recorded serial compatibility-layout path. The earlier fixture broadens that
check to real draw positions across multiple outgoing edges and epochs, and
to explicit and random initialization. The evidence leaves later stream
positions, other layout paths, and behavior across other wheel builds to
further validation.
