import pickle

import numpy as np
import pytest
from numba.typed import List
from scipy.sparse import csr_matrix

from umap_alternative import AlignedUMAP, UMAP


def assert_coordinates(values, shape):
    assert type(values) is np.ndarray
    assert values.dtype == np.float32
    assert values.shape == shape
    assert np.isfinite(values).all()


@pytest.mark.parametrize("as_input", [np.asarray, csr_matrix], ids=["dense", "csr"])
def test_installed_estimator_fits_transforms_and_updates(as_input):
    data = np.random.RandomState(42).normal(size=(15, 3)).astype(np.float32)
    model = UMAP(n_neighbors=4, random_state=42, n_jobs=1, n_epochs=6, compatibility_layout=True)
    training = as_input(data[:12])
    queries = as_input(data[12:14])

    embedding = model.fit_transform(training)
    assert_coordinates(embedding, (12, 2))
    assert embedding is model.embedding_
    assert type(model.graph_) is csr_matrix
    assert model.graph_.dtype == np.float32
    assert model.transform(training.copy()) is embedding
    assert_coordinates(model.transform(queries), (2, 2))
    if as_input is np.asarray:
        assert_coordinates(model.inverse_transform(embedding[:2]), (2, 3))

    model.update(as_input(data[14:]))
    assert_coordinates(model.embedding_, (13, 2))
    assert model.graph_.shape == (13, 13)


def test_aligned_returns_and_restores_typed_embeddings():
    data = np.random.RandomState(13).normal(size=(2, 12, 3)).astype(np.float32)
    relations = [dict(enumerate(range(data.shape[1])))]
    model = AlignedUMAP(n_neighbors=4, random_state=42, n_epochs=2, compatibility_layout=True)
    with pytest.warns(UserWarning, match="n_jobs value -1 overridden"):
        result = model.fit_transform(list(data), relations=relations)

    assert result is model.embeddings_
    assert type(result) is List
    restored = pickle.loads(pickle.dumps(model)).embeddings_
    assert type(restored) is List
    for before, after in zip(result, restored, strict=True):
        assert_coordinates(before, (12, 2))
        assert_coordinates(after, (12, 2))
        assert before.flags.c_contiguous and after.flags.c_contiguous
        np.testing.assert_array_equal(after, before)
