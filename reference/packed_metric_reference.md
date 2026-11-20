# Public packed-byte feature metrics

The fixtures use the locked x86-64 `umap-learn==0.6.0` estimator in the
unsupervised compatibility layout and the named
`pynndescent.distances.bit_hamming` and `bit_jaccard` calls from the
locked dependency. Neither name is exposed as a function in
`umap.distances`. The estimator accepts the exact lowercase names.
Exception classes and warning categories are recorded separately from
numeric data. [Generator](generate_packed_metric_fixture.py) · [Common fixture](fixtures/umap-learn-0.6.0-packed-metrics-x86_64.json)

Packed Hamming counts the set bits in the bytewise XOR and returns that
count as a distance. For example, 127 and 128 have distance 8. Packed
Jaccard divides the intersection population count by the union
population count in float32, then takes the negative float32 logarithm.
Two empty supports give positive zero, equal nonempty supports give
negative zero, and nonempty disjoint supports give positive infinity.
For example, 3 and 255 give `-ln(2/8)`. The named functions raise
`TypingError` on float32 arrays. Estimator input is validated and
converted to `uint8`, including integer wrapping and truncation of
fractional values toward zero. Sparse input raises `TypeError` and
nonfinite input raises `ValueError`. Embedding results preserve the input
floating dtype, including float16 and float64. Integral inputs return
float32 coordinates.

Fit ranks equal distances by training index. Held-out COO uses the
pinned NumPy quicksort tie order. Public supplied-neighbor fits and
public smoothing and membership calls reproduced those choices on
both ordinary and tied rows. A cut held-out rank occupies a stored
zero at row and column zero. Jaccard uses a default cutoff of 1.
Hamming keeps every finite distance by default. A NaN threshold
preserves the original coordinates of disjoint query ranks with zero
membership.

The explicit cutoff is inclusive. The fit compares the threshold in
float64 against the float32 distance widened to float64. Held-out
queries compare in float32. For Hamming at distance 1 and Jaccard at
the float32 value of `ln(2)`, a threshold one float64 ULP above the
distance admits the fit edge and leaves the query rank cut. The first
float32 value above the distance admits both.

The common fixture contains 58 estimator cases and 14 named input pairs.
It records 1,540 fitted CSR weights, 1,026 ordered COO entries, 352
fitted coordinate values, and 432 held-out coordinate values. Each host
fixture adds two connected spectral cases and three named input pairs at
which the platforms differ by one float32 ULP. Outputs record IEEE-754
words, including the NaN coordinates of isolated rows.
[Windows fixture](fixtures/umap-learn-0.6.0-packed-metrics-windows-x86_64.json) · [Ubuntu fixture](fixtures/umap-learn-0.6.0-packed-metrics-ubuntu-x86_64.json)

The named Jaccard grid covers all 4,950 positive intersection-to-union
ratios with union counts up to 99. Windows and Ubuntu differ by one
float32 ULP on 30 ratios. The native path matches all 4,950 named
outputs on each host. Each host fixture includes the grid digest.

## Independent holdouts and timings

On both hosts, 128-, 512-, and 1,024-row holdouts matched all 37,066 fit
CSR weights and 648 held-out COO weights, including their indices.
For each row count `n`, use `np.random.RandomState(788 + n)` to draw
`uint8` training bytes of shape `(n, 16)` and then query bytes of shape
`(12, 16)` with `randint(0, 256, size=..., dtype=np.uint8)`. Fit with
`n_neighbors=9`, `n_jobs=1`, `random_state=42`, `transform_mode="graph"`,
`compatibility_layout=True`, and `disconnection_distance=np.inf`.

The table records median milliseconds for seven fresh estimator fits
after one warmup. Each caller used one numerical thread on the Intel Core
Ultra 9 285K workstation. The timers cover estimator creation and fit.

| Rows | Metric | Windows public | Windows Rust | Ubuntu WSL public | Ubuntu WSL Rust |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Hamming | 7.720 | 0.432 | 4.958 | 0.408 |
| 128 | Jaccard | 8.485 | 0.635 | 5.046 | 0.621 |
| 512 | Hamming | 113.509 | 3.460 | 66.824 | 3.434 |
| 512 | Jaccard | 114.734 | 6.710 | 70.377 | 6.440 |
| 1,024 | Hamming | 459.102 | 11.934 | 267.006 | 11.773 |
| 1,024 | Jaccard | 473.795 | 24.164 | 282.453 | 24.161 |

## Remaining differences

Narrow queries compare their prefix against the training rows. With a
four-byte query against three-byte fit rows, the public output can
depend on the backing array beyond the logical end of the fitted matrix.
To reproduce, load the fixture's `bytes` array and options with
`transform_mode="graph"`. Put its bytes in the first 30 elements of a
46-byte `np.uint8` array. Fit on the first 30 elements reshaped to
`(10, 3)` and leave the final 16 bytes filled with either 0 or 255.
For Hamming, transform `[[0, 0, 0, 0]]`. The stored columns change
from `[9, 0, 2, 1]` to `[0, 1, 2, 3]`. For Jaccard, transform
`[[5, 0, 128, 0]]`. The stored columns change from `[9, 4, 0, 0]`
to `[4, 0, 0, 0]`. The Rust path raises `ValueError` for a wider
query and bounds every read to the fitted matrix.

The pinned public wheel also accepted supervised fit, density outputs,
and update on the ordinary byte sample. On that sample the port raises
`NotImplementedError` for supervision and density and `ValueError`
for update. The disconnected Jaccard spectral probe on `axis` raised
`TypingError` in the public wheel and `InvalidParameterError` in the
port. The fixtures cover connected spectral initialization and
exhaustive search. Approximate packed neighbor search remains unverified.

Run from the repository root. Use `windows` or `ubuntu` for the host
fixture when checking the host-specific observations.

```sh
uv run --project reference --locked python reference/generate_packed_metric_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-packed-metrics-x86_64.json \
  --host-output reference/fixtures/umap-learn-0.6.0-packed-metrics-windows-x86_64.json --check
```
