import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pytest

from umap_alternative import UMAP


PLATFORM = "ubuntu" if platform.system() == "Linux" else "windows"
FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / f"reference/fixtures/umap-learn-0.6.0-dense-angular-embedding-{PLATFORM}-x86_64.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("metric", ["cosine", "correlation"])
@pytest.mark.parametrize("case", FIXTURE)
def test_dense_angular_embedding_matches_public_wheel(case, metric):
    spec = FIXTURE[case]
    x = np.asarray(spec["training"], dtype=np.float32)
    q = np.asarray(spec["queries"], dtype=np.float32)
    options = dict(
        n_neighbors=4, random_state=42, n_jobs=1, compatibility_layout=True, init="random"
    ) | spec["options"]
    if isinstance(options["init"], list):
        options["init"] = np.asarray(options["init"], dtype=np.float32)
    model = UMAP(metric=metric, **options)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        fitted = model.fit_transform(x)
        assert model.transform(x.copy()) is model.embedding_
        held_out = model.transform(q)

    expected = spec["metrics"][metric]
    assert [warning.category.__name__ for warning in observed] == expected["warnings"]
    for field in ("indptr", "indices"):
        np.testing.assert_array_equal(getattr(model.graph_, field), expected[field])
    np.testing.assert_array_equal(model.graph_.data.view(np.uint32), expected["weights"])
    assert fitted.dtype == held_out.dtype == np.float32
    np.testing.assert_array_equal(fitted.view(np.uint32), expected["fitted"])
    np.testing.assert_array_equal(held_out.view(np.uint32), expected["transformed"])
