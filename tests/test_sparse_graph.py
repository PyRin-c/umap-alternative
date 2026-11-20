import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_array, coo_matrix, csc_array, csc_matrix, csr_array, csr_matrix

from umap_alternative import UMAP, _native


FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "reference/fixtures/umap-learn-0.6.0-sparse-graph-windows-x86_64.json"
    ).read_text(encoding="utf-8")
)
SUPPORTED_CASES = [
    name
    for name, case in FIXTURE["cases"].items()
    if case["metric"] != "precomputed" and case["variant"] != "dense"
]


def graph_model(**overrides):
    parameters = FIXTURE["base_parameters"].copy()
    parameters.pop("force_approximation_algorithm")
    return UMAP(**(parameters | overrides))


def fixture_input(name, variant):
    record = FIXTURE["inputs"][name]
    base = csr_matrix(
        (
            np.array(record["data_bits_uint32"], dtype=np.uint32).view(np.float32),
            record["indices"],
            record["indptr"],
        ),
        shape=record["shape"],
    )
    if variant == "csr":
        return base
    if variant == "csc":
        return base.tocsc()
    if variant == "split_csr":
        return csr_matrix(
            (np.repeat(base.data * 0.5, 2), np.repeat(base.indices, 2), base.indptr * 2),
            shape=base.shape,
        )
    if variant == "split_coo":
        coo = base.tocoo()
        return coo_matrix(
            (np.tile(coo.data * 0.5, 2), (np.tile(coo.row, 2), np.tile(coo.col, 2))),
            shape=base.shape,
        )
    assert variant == "reverse_rows_csr"
    for start, stop in zip(base.indptr, base.indptr[1:]):
        base.data[start:stop] = base.data[start:stop][::-1]
        base.indices[start:stop] = base.indices[start:stop][::-1]
    return base


def storage(matrix):
    indices = (matrix.row, matrix.col) if matrix.format == "coo" else (matrix.indptr, matrix.indices)
    return (matrix.data, *indices)


def assert_graph(graph, shape, indptr, indices, bits):
    assert isinstance(graph, csr_matrix)
    assert graph.dtype == np.float32
    assert graph.shape == tuple(shape)
    np.testing.assert_array_equal(graph.indptr, indptr)
    np.testing.assert_array_equal(graph.indices, indices)
    np.testing.assert_array_equal(graph.data.view(np.uint32), bits)


@pytest.mark.parametrize("name", SUPPORTED_CASES)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_sparse_feature_fit_matches_public_csr_shape_order_and_weight_bits(name, method):
    case = FIXTURE["cases"][name]
    matrix = fixture_input(case["input"], case["variant"])
    before = tuple(array.copy() for array in storage(matrix))
    canonical = matrix.has_canonical_format
    expected = FIXTURE["cases"][case.get("same_graph_as", name)]["graph"]
    model = graph_model(metric=case["metric"])

    returned = getattr(model, method)(matrix)

    assert returned is (model if method == "fit" else model.graph_)
    assert model.n_features_in_ == matrix.shape[1]
    assert model.n_neighbors_ == FIXTURE["base_parameters"]["n_neighbors"]
    assert_graph(
        model.graph_, expected["shape"], expected["indptr"], expected["indices"],
        expected["data_bits_uint32"],
    )
    for after, original in zip(storage(matrix), before):
        np.testing.assert_array_equal(after, original)
    assert matrix.has_canonical_format == canonical


@pytest.mark.parametrize(
    ("constructor", "alias", "metric"),
    [
        (csr_array, "l2", "euclidean"),
        (csc_array, "cityblock", "manhattan"),
        (coo_array, "linfty", "chebyshev"),
    ],
)
def test_sparse_arrays_and_existing_metric_aliases(constructor, alias, metric):
    matrix = constructor(fixture_input("features", "csr"))
    expected = FIXTURE["cases"][f"features_csr_{metric}"]["graph"]
    assert_graph(
        graph_model(metric=alias).fit_transform(matrix),
        expected["shape"], expected["indptr"], expected["indices"], expected["data_bits_uint32"],
    )


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_duplicate_sum_uses_observed_public_dtype_order(format):
    data = np.array([1e8, 1, -1e8, 2, 4, 7], dtype=np.float64)
    if format == "coo":
        matrix = coo_matrix((data, ([0, 0, 0, 1, 2, 3], [0] * 6)), shape=(4, 1))
    elif format == "csc":
        matrix = csc_matrix((data, [0, 0, 0, 1, 2, 3], [0, 6]), shape=(4, 1))
    else:
        matrix = csr_matrix((data, [0] * 6, [0, 3, 4, 5, 6]), shape=(4, 1))
    graph = graph_model(n_neighbors=3).fit_transform(matrix)
    first = 1 if format == "coo" else 0
    expected = graph_model(n_neighbors=3).fit_transform(csr_matrix([[first], [2], [4], [7]]))
    assert_graph(
        graph, expected.shape, expected.indptr, expected.indices, expected.data.view(np.uint32)
    )


def test_normalization_uses_storage_when_the_callers_sorted_flags_are_stale():
    matrix = fixture_input("features", "reverse_rows_csr")
    matrix.has_canonical_format = True
    expected = FIXTURE["cases"]["features_csr_euclidean"]["graph"]

    output = graph_model().fit_transform(matrix)

    assert_graph(
        output, expected["shape"], expected["indptr"], expected["indices"],
        expected["data_bits_uint32"],
    )
    assert matrix.has_canonical_format
    assert matrix.indices[0] > matrix.indices[1]


def test_two_zero_distance_neighbors_fit_when_each_row_keeps_itself():
    graph = graph_model(n_neighbors=2).fit_transform(csr_matrix([[1], [1], [2], [3]]))
    assert_graph(graph, (4, 4), [0, 2, 3, 5, 6], [1, 2, 0, 0, 3, 2], [0x3F800000] * 6)


@pytest.mark.parametrize(
    "matrix",
    [
        csr_matrix((4, 2)),
        csr_matrix(([-0.0, 0.0], [1, 0], [0, 1, 1, 2, 2]), shape=(4, 2)),
        csr_matrix([[1], [1], [1], [3]]),
    ],
    ids=["implicit-zeros", "stored-signed-zeros", "duplicate-feature-rows"],
)
def test_zero_distance_ties_excluding_self_raise_a_core_error(matrix):
    with pytest.raises(ValueError, match="neighbor row 2 does not include itself"):
        graph_model(n_neighbors=2).fit_transform(matrix)


def test_two_sparse_rows_use_the_truncated_neighbor_count_and_empty_graph_contract():
    model = graph_model()
    with pytest.warns(UserWarning) as caught:
        graph = model.fit_transform(csr_matrix([[0, 0], [1, 0]]))

    assert model.n_neighbors_ == 1
    assert_graph(graph, (2, 2), [0, 0, 0], [], [])
    assert [str(item.message).splitlines()[0] for item in caught] == [
        "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1",
        "A large number of your vertices were disconnected from the manifold.",
    ]


def test_singleton_sparse_fit_clears_existing_graph_and_leaves_no_graph_to_return():
    model = graph_model().fit(fixture_input("features", "csr"))
    matrix = csr_matrix((1, 2))

    assert model.fit(matrix) is model
    assert model.n_features_in_ == 2
    assert not hasattr(model, "graph_")
    assert not hasattr(model, "n_neighbors_")
    with pytest.raises(AttributeError, match="'UMAP' object has no attribute 'graph_'"):
        model.fit_transform(matrix)
    with pytest.raises(ValueError, match="Transform unavailable .* only a single data sample"):
        model.transform(matrix)


@pytest.mark.parametrize(
    ("format", "field", "values", "message"),
    [
        ("csr", "indptr", [0, 1, 2], "sparse indptr"),
        ("csr", "indptr", [1, 1, 2, 3, 4], "sparse indptr"),
        ("csr", "indptr", [0, 1, 2, 3, 3], "sparse indptr"),
        ("csr", "indptr", [0, 1, 3, 2, 4], "sparse indptr"),
        ("csr", "indices", [0, 0, 0], "one integer per stored value"),
        ("csr", "indices", [0, -1, 0, 0], "out of bounds"),
        ("csr", "indices", [0, 3, 0, 0], "out of bounds"),
        ("csr", "indices", [0.0] * 4, "one integer per stored value"),
        ("csc", "indptr", [0, 5, 4, 4], "sparse indptr"),
        ("csc", "indices", [0, 1, 2, 4], "out of bounds"),
        ("coo", "row", [0, 1, -1, 3], "out of bounds"),
        ("coo", "col", [0, 0, 0, 3], "out of bounds"),
    ],
)
def test_malformed_storage_fails_before_conversion(format, field, values, message):
    matrix = csr_matrix([[1, 0, 0], [2, 0, 0], [4, 0, 0], [7, 0, 0]]).asformat(format)
    setattr(matrix, field, np.asarray(values))

    with pytest.raises(ValueError, match=message):
        graph_model().fit(matrix)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_nonfinite_data_is_rejected_even_in_a_singleton(value, format):
    with pytest.raises(ValueError, match="Input contains (NaN|infinity)"):
        graph_model().fit(csr_matrix([[value]]).asformat(format))


def test_nonfinite_duplicate_sum_is_rejected():
    large = np.finfo(np.float32).max
    matrix = csr_matrix((np.array([large, large]), [0, 0], [0, 2]), shape=(1, 1))
    with pytest.raises(ValueError, match="Input contains infinity"):
        graph_model().fit(matrix)


def test_failed_sparse_refits_preserve_fitted_graph_and_transform_state():
    training = fixture_input("features", "csr").toarray()
    for original in (training, csr_matrix(training)):
        model = graph_model().fit(original)
        graph = model.graph_
        malformed = csr_matrix([[2]])
        malformed.indices[0] = -1
        model.set_params(n_neighbors=2)

        for matrix, message in (
            (malformed, "out of bounds"),
            (csr_matrix([[np.nan]]), "Input contains NaN"),
            (csr_matrix((4, 2)), "neighbor row 2 does not include itself"),
        ):
            with pytest.raises(ValueError, match=message):
                model.fit_transform(matrix)
            assert model.graph_ is graph
            assert model.n_features_in_ == training.shape[1]
            assert model.n_neighbors_ == 4
            assert model.transform(training) is graph

    model.set_params(n_neighbors=4).fit(training)
    assert model.transform(training) is model.graph_


def test_unimplemented_sparse_paths_fail_explicitly():
    matrix = fixture_input("features", "csr")
    with pytest.raises(TypeError, match="CSR, CSC, or COO"):
        graph_model().fit(matrix.tolil())
    model = graph_model(metric="cosine").fit(matrix)
    with pytest.raises(NotImplementedError, match="after fitting sparse features"):
        model.transform(matrix[1:])


def test_native_sparse_binding_reports_core_validation_errors():
    with pytest.raises(ValueError, match="CSR row 0, entry 1 has column 0, expected greater"):
        _native._fit_sparse_graph([0, 2], [0, 0], [1.0, 2.0], (1, 2), 1, "euclidean")
