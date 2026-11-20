"""Record public UMAP row deduplication and sparse row keys."""

import argparse
import json
import warnings
from pathlib import Path


def capture():
    import numpy as np
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse
    from umap import UMAP
    from umap.utils import csr_unique

    features = [[7, 3], [1, 0], [7, 3], [0, 0], [2, 4], [1, 0], [4, 3], [0, 0],
                [3, 7], [3, 5]]
    queries = [[1.5, 2], [0, 0], [7, 3]]
    x = np.asarray(features, dtype=np.float32)
    q = np.asarray(queries, dtype=np.float32)
    compatibility = dict(
        n_neighbors=3, n_epochs=9, init="random", random_state=42, n_jobs=1,
        compatibility_layout=True, unique=True, a=1.2, b=0.8,
    )

    def encode(value):
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        if issparse(value):
            coordinates = (
                [value.indptr.tolist(), value.indices.tolist()] if value.format == "csr"
                else [value.row.tolist(), value.col.tolist()]
            )
            return {
                "format": value.format, "shape": list(value.shape),
                "coordinates": coordinates, "bits": value.data.view(np.uint32).tolist(),
            }
        return value.view(np.uint32).tolist()

    cases = []
    for name, options, labels in (
        ("dense", {}, None),
        ("dense_graph", dict(transform_mode="graph"), None),
        ("csr", {}, None),
        ("csr_graph", dict(transform_mode="graph"), None),
        ("csc", {}, None),
        ("coo", {}, None),
        ("categorical", {}, [1, 0, 1, 2, 0, 1, 1, 0, 2, 0]),
        ("continuous", dict(target_metric="euclidean"), list(range(len(features)))),
        ("checkpoints", dict(n_epochs=[0, 2, 9]), None),
        ("adam", dict(n_epochs=3, compatibility_layout=False, optimizer="adam"), None),
        ("density", dict(n_epochs=0, output_dens=True), None),
        ("explicit", dict(n_epochs=0, init=[[1, 2], [2, 6], [3, 4], [4, 5],
                                           [5, 1], [6, 3], [7, 0]]), None),
        ("truncated", dict(n_neighbors=7, n_epochs=0), None),
    ):
        convert = (
            csr_matrix if name.startswith("csr") else
            csc_matrix if name == "csc" else coo_matrix if name == "coo" else np.asarray
        )
        options = compatibility | options
        estimator_options = options.copy()
        if name == "explicit":
            estimator_options["init"] = np.asarray(options["init"], dtype=np.float32)
        model = UMAP(**estimator_options)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = model.fit_transform(convert(x), None if labels is None else np.asarray(labels))
        encoded = {
            "name": name, "options": options, "labels": labels, "result": encode(result),
            "graph": encode(model.graph_), "warnings": [w.category.__name__ for w in seen],
        }
        if name == "checkpoints":
            encoded["coordinates_by_epoch"] = [encode(a) for a in model.embedding_list_]
        if name in {"dense", "dense_graph", "csr", "csr_graph", "csc", "coo", "truncated"}:
            encoded["training_transform"] = encode(model.transform(convert(x)))
            encoded["query_transform"] = encode(model.transform(convert(q)))
        if name == "dense":
            model.set_params(n_epochs=0)
            encoded["inverse"] = encode(model.inverse_transform(model.embedding_[[1, 4]]))
            model.update(q)
            encoded["updated_graph"] = encode(model.graph_)
            encoded["updated_embedding"] = encode(model.embedding_)
        cases.append(encoded)

    utility_inputs = [
        ([0, 0, 0], [], []),
        ([0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
         [0, -0.0, 0, 0.5, 0.5, 1, 2, 3, 3, 0, 1, 1, 0, 0]),
        ([0, 1, 2, 3, 4, 6, 6], [1, 0, 1, 1, 0, 1], [9, 9, 5, 9, 1, 2]),
    ]
    utilities = []
    for offsets, indices, data in utility_inputs:
        matrix = csr_matrix(
            (np.asarray(data, dtype=np.float32), indices, offsets),
            shape=(len(offsets) - 1, 3),
        )
        utilities.append({
            "indptr": offsets, "indices": indices, "values": data,
            "groups": [part.tolist() for part in csr_unique(matrix)],
        })

    inverse_layout = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [0, 2], [1, 2], [9, 9]]
    inverse_features = [[7, 3, 2], [1, 0, 2], [0, 0, 0], [2, 4, 8], [4, 3, 5],
                        [3, 7, 0], [3, 5, 2], [8, 8, 8]]
    inverse_queries = [[0.21, 0.11], [0.48, 0.24], [1.07, 0.62]]
    inverse = []
    for duplicate, repulsion in ((False, 1.0), (True, 0.0)):
        input_values = inverse_features + ([inverse_features[1]] if duplicate else [])
        settings = compatibility | dict(
            n_neighbors=4, n_epochs=0, a=1.0, b=1.0, negative_sample_rate=4,
            repulsion_strength=repulsion,
        )
        model = UMAP(**settings).fit(np.asarray(input_values, np.float32))
        model.embedding_ = np.asarray(inverse_layout[:len(input_values)], np.float32)
        model.n_epochs = 48
        inverse.append({
            "features": input_values, "options": settings, "layout": model.embedding_.tolist(),
            "queries": inverse_queries,
            "result": encode(model.inverse_transform(np.asarray(inverse_queries, np.float32))),
        })
    return {
        "features": features, "queries": queries, "cases": cases, "utility": utilities,
        "inverse": inverse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = capture()
    document = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, separators=(",", ":"))}'
        for key, value in result.items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != document:
            parser.exit(1, f"{args.output} differs from regenerated fixture\n")
    else:
        args.output.write_text(document, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Report exception types because JIT diagnostics can include source text.
        raise SystemExit(type(error).__name__) from None
