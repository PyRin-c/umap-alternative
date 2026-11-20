# Public packed-feature updates

The fixture captures `UMAP.update` for `bit_hamming` and `bit_jaccard`
from the locked public `umap-learn==0.6.0` wheel. It uses seeded
unsupervised compatibility layouts and records only public method
results, `graph_`, and `embedding_`. The stable fixture has 34
trajectories with 94 update calls. Of those, 60 succeed, 22 raise
`ValueError`, six raise `TypeError`, and six raise `AttributeError`.
It includes 124 fitted CSR snapshots, 122 embedding snapshots, and
102 ordered held-out COO snapshots. Their numeric values are IEEE
bit patterns. Both pinned hosts regenerate SHA-256
`6c52fd9ee4106d082e2e7c8a1c5531c0353c2c49c9f71d648562f0d5dfdc0ae9`.
[Generator](generate_packed_update_fixture.py) · [Fixture](fixtures/umap-learn-0.6.0-packed-update-x86_64.json)

Append batches convert directly to `uint8` before concatenation.
The signed integer and floating probes include negative values,
fractions, and values above 255. The unsigned probe includes values
above `2**53`, where conversion through float64 would lose low
bits. The graph, embedding, and queries agree with direct byte
coercion. A transformed copy of all current rows uses the embedding
shortcut after byte coercion. Floating inputs still request their
own output dtype, including float16 and float64. The suffix bytes of
the valid-width backing arrays can be 0 or 255 with identical fitted,
updated, and held-out outputs.

The update keeps the effective neighbor cap from the original fit
as rows are added and when the requested count changes. The packed
ranker rebuilds the graph without the original fit cutoff. Held-out
queries retain that cutoff, even while the current public parameter
has been changed to 10. At distance 1 for Hamming and float32
`ln(2)` for Jaccard, equal and next-float64 thresholds produce the
same held-out COO, while the next-float32 threshold retains the edge.
All three thresholds give the same rebuilt graph in these probes.

The update uses the retained fitted coordinates and the ranked
neighbors for appended rows. An infinite Jaccard rank contributes
no coordinate to the average. An appended row with no finite
retained neighbors starts at the mean of the retained embedding.
This matters for the zero-row append to `base`, where all old supports
are nonempty. The fixtures cover zero, one, and positive epochs,
omitted and list epochs, repeated appends, and updates after
`unique=True` fitting.

Invalid dimensions, empty batches, sparse formats, nonfinite values,
and an invalid seed leave the old fitted objects in place. Graph
mode publishes the expanded graph and then raises `AttributeError`
because it has no embedding. Negative and floating update budgets
publish the expanded graph and prune weak weights before failing on
the layout budget. The old embedding persists. A later transform
that needs a missing coordinate raises `IndexError`. A transform
that can initialize from retained rows can still return coordinates,
including when the failed update left a small nonnegative floating
budget.

## Remaining public differences

For the first zero-row Jaccard append in `bit_jaccard_epochs`, an
isolated query of that single appended row returned coordinate
words `[1082743547, 1082896763]` on Windows and both those words and
`[2147479552, 2147479552]` on separate WSL runs. The multi-row
query remains stable and agrees with the port. The port returns
the appended row's finite coordinates for the single-row query.

Longer transforms against a partial fit can vary in the public
wheel. Fit Jaccard on `base` with the fixture parameters and infinite
cutoff, set `n_epochs=-1`, and append `first`. After the update
raises `ValueError`, set `n_epochs=9` and transform `base[:1]`.
The Windows Jaccard probe returned `[1071934328, 1066934770]`, and
WSL returned `[1070813122, 1066402616]`. Both public Hamming probes
returned `[1071934328, 1066934770]`. The port bounds reads to the
retained coordinate array and returns `[1070813122, 1066402616]`
for both metrics in this probe. The fixture checks the stable zero-budget
queries and the missing-coordinate error after partial publication.
A string budget such as `"bad"` also leaves the expanded graph and
old embedding, with `UFuncTypeError` from the public transform
and `ValueError` from the port when it queries retained rows.

Run from the repository root on either locked host.

```sh
uv run --project reference --locked python reference/generate_packed_update_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-packed-update-x86_64.json --check
```
