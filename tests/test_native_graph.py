import pytest

from umap_alternative import _native


TIES_DATA = [
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]


def test_native_dense_graph_returns_ties_fixture_csr():
    shape, indptr, indices, data = _native._fit_dense_graph(
        TIES_DATA, 8, 2, 4, "euclidean"
    )

    assert shape == (8, 8)
    assert indptr == [0, 5, 10, 15, 20, 23, 26, 29, 32]
    assert indices == [
        1,
        2,
        3,
        4,
        5,
        0,
        2,
        3,
        4,
        5,
        0,
        1,
        3,
        6,
        7,
        0,
        1,
        2,
        6,
        7,
        0,
        1,
        5,
        0,
        1,
        4,
        2,
        3,
        7,
        2,
        3,
        6,
    ]
    assert data == [1.0] * 32


@pytest.mark.parametrize("metric", ["Euclidean", "l2", "Manhattan", "l1", "Chebyshev", "linfty"])
def test_native_dense_graph_accepts_metric_aliases(metric):
    graph = _native._fit_dense_graph([0.0, 0.0, 1.0, 0.0], 2, 2, 2, metric)

    assert graph[0] == (2, 2)
    assert graph[1] == [0, 1, 2]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([0.0, 1.0], 2, 2, 1, "euclidean"), "does not match 2 values"),
        (([0.0, 1.0], 2, 1, 3, "euclidean"), "outside the valid range 1..=2"),
        (([0.0, float("nan")], 1, 2, 1, "euclidean"), "not finite"),
    ],
)
def test_native_dense_graph_translates_core_validation_errors(args, message):
    with pytest.raises(ValueError, match=message):
        _native._fit_dense_graph(*args)


def test_native_dense_graph_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unsupported metric"):
        _native._fit_dense_graph([0.0, 0.0], 1, 2, 1, "cosine")


def test_native_dense_categorical_graph_translates_label_count_error():
    with pytest.raises(ValueError, match="expected 2 categorical labels, got 1"):
        _native._fit_dense_categorical_graph([0.0, 1.0], 2, 1, 2, "l2", [-1], 0.5)
