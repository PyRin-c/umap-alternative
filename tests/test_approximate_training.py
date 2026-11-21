import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from umap_alternative import UMAP
from umap_alternative._feature_search import ApproximateFeatureIndex
from umap_alternative._umap import _fit_neighbor_arrays


FIXTURE_DIR = Path(__file__).parents[1] / "reference/fixtures"
SUFFIX = "-ubuntu" if sys.platform.startswith("linux") else ""
FIXTURE = FIXTURE_DIR / f"umap-learn-0.6.0-approximate-training{SUFFIX}.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


@pytest.mark.parametrize(
    "case", REFERENCE["cases"], ids=lambda c: f"{c['metric']}-{c['rows']}-{c['features']}-k{c['width']}"
)
def test_approximate_training_and_transform(case):
    rows, features, width, seed = (case[name] for name in ("rows", "features", "width", "seed"))
    data = np.random.RandomState(case["data_seed"]).normal(size=(rows, features)).astype(np.float32)
    model = UMAP(
        n_neighbors=width, metric=case["metric"], n_jobs=1, random_state=seed, transform_mode="graph",
        force_approximation_algorithm=rows < 4096,
    ).fit(data)

    index = model._feature_search_index
    assert isinstance(index, ApproximateFeatureIndex)
    columns, distances = index.training_ranks()
    assert digest(columns, "<i4") == case["columns"]
    assert digest(distances, "<f4") == case["distances"]
    graph = model.graph_.tocsr()
    assert digest(graph.indptr, "<i8") == case["indptr"]
    assert digest(graph.indices, "<i4") == case["indices"]
    assert digest(graph.data, "<f4") == case["weights"]

    queries = np.random.RandomState(5).normal(size=(32, features)).astype(np.float32)
    for query_width, expected in case["query"].items():
        indices, distances = index.query(queries, k=int(query_width))
        assert digest(indices, "<i4") == expected["indices"]
        assert digest(distances, "<f4") == expected["distances"]
    transformed = model.transform(queries).tocoo()
    assert digest(transformed.row, "<i4") == case["transform"]["rows"]
    assert digest(transformed.col, "<i4") == case["transform"]["columns"]
    assert digest(transformed.data, "<f4") == case["transform"]["weights"]


@pytest.mark.parametrize(
    "case", REFERENCE["caller_rng"], ids=lambda c: f"{c['prior']}-{c['epochs']}"
)
def test_approximate_search_hands_caller_rng_to_embedding(case):
    data = np.random.RandomState(42).normal(size=(96, 4)).astype(np.float32)
    rng = np.random.RandomState(37)
    if case["prior"] == "gaussian":
        rng.normal(size=13)
    elif case["prior"] == "twist":
        rng.randint(2**16, size=623)
    model = UMAP(
        n_neighbors=5, n_epochs=case["epochs"], init="random", n_jobs=1,
        random_state=rng, compatibility_layout=True, force_approximation_algorithm=True,
    )
    assert digest(model.fit_transform(data), "<f4") == case["layout"]
    _, words, cursor, has_gaussian, gaussian = rng.get_state()
    assert digest(words, "<u4") == case["words"]
    assert cursor == case["cursor"]
    assert has_gaussian == case["has_gaussian"]
    assert digest(gaussian, "<f8") == case["gaussian"]


def test_approximate_euclidean_leaf_matches_tied_public_query_order():
    data = np.asarray([-1, 1] * 4, dtype=np.float32).reshape(-1, 1)
    search = ApproximateFeatureIndex(data, 5, "euclidean", 42)
    queries = np.asarray([[0], [-1], [1]], dtype=np.float32)
    indices, distances = search.query(queries, 5)
    np.testing.assert_array_equal(indices, [[2, 3, 0, 1, 4], [4, 6, 0, 2, 3], [7, 3, 5, 1, 0]])
    np.testing.assert_array_equal(distances, [[1] * 5, [0] * 4 + [2], [0] * 4 + [2]])


def test_native_training_graph_keeps_nondefault_smoothing_and_rank_bits():
    data = np.random.RandomState(71).normal(size=(96, 4)).astype(np.float32)
    model = UMAP(
        n_neighbors=7, n_jobs=1, random_state=42, transform_mode="graph",
        local_connectivity=1.5, set_op_mix_ratio=0.375, force_approximation_algorithm=True,
    ).fit(data)
    index = model._feature_search_index
    indices, distances = index.training_ranks()
    shape, indptr, columns, weights = _fit_neighbor_arrays(indices, distances, 7, (1.5, 0.375))
    assert model.graph_.shape == shape
    np.testing.assert_array_equal(model.graph_.indptr, indptr)
    np.testing.assert_array_equal(model.graph_.indices, columns)
    assert digest(model.graph_.data, "<f4") == digest(weights, "<f4")
    assert all(buffer.flags.writeable for buffer in (
        model.graph_.indptr, model.graph_.indices, model.graph_.data,
    ))


@pytest.mark.parametrize("width", [1, 2, 5])
def test_approximate_cosine_zero_query_matches_public_empty_heap(width):
    data = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [0, 0], [0, 0], [2, 0]],
        dtype=np.float32,
    )
    search = ApproximateFeatureIndex(data, 5, "cosine", 42)
    queries = np.array([[0.0, -0.0], [np.finfo(np.float32).smallest_subnormal, 0]], dtype=np.float32)
    indices, distances = search.query(queries, width)
    np.testing.assert_array_equal(indices, np.full((2, width), 8, dtype=np.int32))
    np.testing.assert_array_equal(distances, np.ones((2, width), dtype=np.float32))


def test_approximate_cosine_overflowed_query_norm_keeps_tied_candidates():
    data = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [0, 0], [0, 0], [2, 0]],
        dtype=np.float32,
    )
    search = ApproximateFeatureIndex(data, 5, "cosine", 42)
    indices, distances = search.query(np.array([[3e38, 3e38]], dtype=np.float32), 5)
    np.testing.assert_array_equal(indices, [[2, 3, 0, 1, 4]])
    np.testing.assert_array_equal(distances, np.ones((1, 5), dtype=np.float32))
