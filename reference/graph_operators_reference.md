# Public directed graph operators

The locked `umap-learn==0.6.0` wheel exposes `general_simplicial_set_union`,
`general_simplicial_set_intersection`, and `reset_local_connectivity` in
`umap.umap_`. The two `umap.sparse` entry points write the same union and
intersection weights directly to existing COO value buffers. The port
exposes these five entry points with a shared native membership kernel.

Union and weighted intersection use the stored support of the sum of the
inputs. SciPy storage class and format determine the order of the resulting
COO coordinates. When the right graph is complemented, intersection uses
the left COO support as-is. The returned COO object shares the left value
buffer for COO and CSR inputs in the pinned environment. For a CSC input the
values are separate. The fixture checks returned identity and both operand
states, including when both operands are the same object.

An absent membership uses half the minimum stored weight with a floor of
`1e-8`. Intersection also caps the right fallback at `1e-4` and measures
it after complementing the right weights when applicable. A stored zero
is an explicit membership. When an operand CSR row contains repeated
columns, the last matching stored entry is used. If both weights are at
or below their fallbacks, intersection preserves the current value in
the output buffer. A shared operand reads the values already written to
that same buffer. The port validates CSR offsets and requires a full
shared value buffer for overlapping operand and result storage.

For mixing weight `w < 0.5`, intersection multiplies the left weight by
the right weight raised to `w / (1 - w)`. For `w >= 0.5`, it multiplies
the right weight by the left weight raised to `(1 - w) / w`. The fixture
brackets `0.5` with the adjacent float64 values and covers both endpoints,
out-of-range weights, mixed input precision, and integer result storage.

`reset_local_connectivity` applies sparse maximum-absolute row normalization
and takes a fuzzy union with the transpose. The optional reweighting
searches up to 32 times for a row exponent targeting `log2(15)` total
membership. In float64, a row of fewer than four supported memberships can
finish with the exponent `2**-32`. The fixture includes the three- and
four-edge cases, nonfinite input errors, and retained negative values
when recalibration is disabled.

The common fixture contains 180 matrix calls and six buffer calls. The
matrix results cover 1,443 stored output weights and eight exceptions.
Both pinned x86-64 hosts regenerate the same fixture.

To regenerate or check from the project root, use the locked reference
environment with its caches on the native host filesystem.

```sh
uv run --project reference --locked python -I reference/generate_graph_operators_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-graph-operators-x86_64.json --check
```
