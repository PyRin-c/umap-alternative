# Public boolean-feature updates

The fixture records unsupervised compatibility-layout updates from the locked
`umap-learn==0.6.0` wheel. It uses public `fit`, `update`, `transform`
and `set_params` calls and observes `graph_` and `embedding_`. Exceptions
are recorded by class and warnings by category. Windows and Ubuntu WSL
produced the same fixture with CPython 3.13.12. The SHA-256 is
`ed4215ee8891f3ad892ab9e79d4ddadc79002a20489d480a4ef5f594140d4c72`.
[Generator](generate_boolean_update_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-boolean-update-x86_64.json)

The 79 trajectories contain 156 update calls. There are 137 successful
updates, 15 `ValueError` results, and four `AttributeError` results in graph
mode. The fixture records 234 CSR snapshots with 21,978 weights and 228
embedding snapshots with 7,000 coordinates. The transforms contribute 162
ordered COO snapshots with 1,169 entries and 289 coordinate arrays with
4,042 values. All numeric values are stored as IEEE-754 words.
The first dense Dice update's original-input transform has stable
coordinate bits with an intermittent `RuntimeWarning`. The fixture
records both warning outcomes for that call.

The dense and canonical CSR cases cover numeric Hamming, matching, Jaccard,
Dice, Russell–Rao, Rogers–Tanimoto, Sokal–Michener, Sokal–Sneath, and
Kulsinski. Yule runs on dense features, and its CSR fit raises `ValueError`.
The data include signed values, empty supports, repeated rows, tied ranks
in matrices of both 12 and 24 rows, and explicit signed and unsigned CSR
zeros in fit and append inputs. The fitted neighbor cap persists when the
sample count grows and when the requested count changes. The trajectories
include zero, one, six, and nine update epochs and both endpoint
`transform_seed` values.

The update rebuild uses uncut neighbor ranks. For Hamming on the axes
input, the initial fit graph changes between cutoff 1 and the next
float32 value above 1. All three cutoff metrics in the fixture have the
same updated graph at both thresholds. Their subsequent held-out COO
records still distinguish the thresholds using the cutoff retained by
the original fit.

Dense fits reject CSR, CSC, and COO batches with `ValueError` before
replacing fitted outputs. CSR fits accept CSC, COO, and dense batches
through sparse stacking. The tested empty, wrong-width, and nonfinite
batches preserve both fitted objects. An invalid transform seed also
preserves both. Graph-mode update publishes the expanded graph and then
raises `AttributeError`. A negative epoch count publishes the graph and
keeps the previous embedding, as does a later attempt to append to that
partial state. The previous fit input still uses its shortcut. Later graph
queries see the expanded training width, and embedding initialization
raises `IndexError` if it needs an appended row that has no coordinates.

Positive-epoch transforms against a partial fit can vary in the public
wheel. To reproduce, load `base_dense` and `last_dense` from the fixture,
fit Hamming with the fixture options, set `n_epochs=-1`, and update with
`last_dense`. After the `ValueError`, set `n_epochs=9` and transform
`base_dense[:1]`. Windows probes returned coordinate words
`[1066191818, 1051727103]` and `[1063097107, 1049912367]` across fit
trajectories and storage formats. The WSL probe returned
`[1066191818, 1051727103]` in both formats. The Rust optimizer bounds
accesses to the retained coordinate buffer and gives
`[1063097107, 1049912367]` in this case. The stable single-row probe
with `n_epochs=6` is included in the fixture.

Run from the repository root on either locked host.

```sh
uv run --project reference --locked python reference/generate_boolean_update_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-boolean-update-x86_64.json --check
```
