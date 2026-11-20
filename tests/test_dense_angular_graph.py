import json
import platform
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from umap_alternative import UMAP


PLATFORM = "ubuntu" if platform.system() == "Linux" else "windows"
FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / f"reference/fixtures/umap-learn-0.6.0-dense-angular-{PLATFORM}-x86_64.json"
    ).read_text(encoding="utf-8")
)
METRICS = ("cosine", "correlation")
pytestmark = pytest.mark.filterwarnings("ignore:A large number of your vertices were disconnected")


def features(spec):
    if "training" in spec:
        training = np.asarray(spec["training"], dtype=np.float32)
    else:
        rng = np.random.default_rng(spec["seed"])
        if "scaled_rows" in spec:
            training = rng.uniform(0.1, 2, spec["random_shape"]).astype(np.float32)
            for destination, source, factor in spec["scaled_rows"]:
                training[destination] = factor * training[source]
        else:
            training = rng.standard_normal(spec["random_shape"]).astype(np.float32)
    queries = None
    if "queries" in spec:
        queries = np.asarray(spec["queries"], dtype=np.float32)
    elif "query_prefix" in spec:
        queries = training[:spec["query_prefix"]].copy()
    return training, queries


def assert_graph(graph, expected):
    assert graph.format == expected["format"]
    assert graph.dtype.name == expected["dtype"]
    assert graph.shape == tuple(expected["shape"])
    fields = ("indptr", "indices") if graph.format == "csr" else ("row", "col")
    for field in fields:
        np.testing.assert_array_equal(getattr(graph, field), expected[field])
    np.testing.assert_array_equal(graph.data.view(np.uint32), expected["data_bits"])


def query_with_index_ties(case, metric, expected):
    columns = expected["col"].copy()
    if case == "held_out_ties":
        columns[2:4] = [4, 5]
    elif case == "zero_constant_repeated":
        if metric == "cosine":
            columns[7] = 0
        else:
            columns[9:11] = [4, 6]
        columns[12:14] = [4, 6]
    return expected | {"col": columns}


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_dense_angular_graph_mode(case, metric):
    spec = FIXTURE["cases"][case]["input"]
    expected = FIXTURE["cases"][case]["metrics"][metric]
    training, queries = features(spec)
    cutoff = float(spec["cutoff"]) if "cutoff" in spec else None
    model = UMAP(
        transform_mode="graph", compatibility_layout=True, metric=metric, n_neighbors=spec["k"],
        disconnection_distance=cutoff, **spec.get("options", {}),
    )
    if case == "product_overflow" and metric == "cosine":
        with pytest.raises(ValueError, match="invalid distance NaN"):
            model.fit(training)
        return

    assert_graph(model.fit_transform(training), expected["fit"])
    assert model.transform(training.copy()) is model.graph_
    if queries is not None:
        assert_graph(model.transform(queries), query_with_index_ties(case, metric, expected["transform"]))


@pytest.mark.parametrize("metric", METRICS)
def test_dense_angular_python_scope_is_graph_mode(metric):
    with pytest.raises(NotImplementedError, match="requires transform_mode='graph'"):
        UMAP(metric=metric).fit([[1, 2], [-1, -2], [3, 4]])


def test_sparse_correlation_is_outside_the_dense_angular_path():
    with pytest.raises(ValueError, match="recognised string"):
        UMAP(transform_mode="graph", metric="correlation").fit(csr_matrix([[1, 0], [-1, 2]]))
