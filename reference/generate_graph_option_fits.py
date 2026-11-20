"""Capture public graph and seeded embedding fits with nondefault graph options."""

import argparse
import json
import os
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path


FIXTURE = Path(__file__).with_name("fixtures") / "umap-learn-0.6.0-estimator-graph-options-x86_64.json"


def capture():
    import numpy as np
    from scipy.sparse import csr_matrix
    from umap import UMAP

    assert version("umap-learn") == "0.6.0"
    X = np.array(
        [[0, 0], [0.25, 0.75], [0.5, 1.5], [1, 0.125],
         [1.75, 1.25], [2.5, 0.25], [3.125, 1], [3.5, 0.125]],
        dtype=np.float32,
    )
    delta = X.astype(np.float64)[:, None, :] - X.astype(np.float64)[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32)
    queries = np.array([[0.125, 0.5], [1.25, 1.125], [2.125, 0.625]], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1, -1, 1])
    initial = np.array(
        [[0, 6], [1, 3], [4, 0], [2, 1], [5, 4], [6, 5], [3, 2], [7, 7]],
        dtype=np.float32,
    )
    options = dict(
        n_neighbors=4, n_components=2, a=1.7, b=0.9, learning_rate=0.13,
        random_state=42, n_jobs=1, compatibility_layout=True, init="random",
        transform_mode="graph", n_epochs=0,
    )

    def record(kind, **settings):
        estimator_settings = settings.copy()
        data = X
        y = None
        if kind == "distances":
            data = distances
            estimator_settings["metric"] = "precomputed"
        elif kind == "sparse":
            data = csr_matrix(X)
        elif kind == "categorical":
            y = labels
        if estimator_settings.get("init") == "explicit":
            estimator_settings["init"] = initial.copy()
        model = UMAP(**(options | estimator_settings))
        with warnings.catch_warnings(record=True):
            model.fit(data, y=y)
        graph = model.graph_
        records = {
            "kind": kind,
            "settings": settings,
            "indptr": graph.indptr.tolist(),
            "indices": graph.indices.tolist(),
            "weight_bits": graph.data.view(np.uint32).tolist(),
        }
        if hasattr(model, "embedding_"):
            records["coordinates_bits"] = model.embedding_.view(np.uint32).tolist()
            if settings["local_connectivity"] > 1:
                records["transformed_bits"] = model.transform(queries).view(np.uint32).tolist()
        elif kind == "dense":
            query_graph = model.transform(queries)
            records["query_graph"] = {
                "row": query_graph.row.tolist(),
                "col": query_graph.col.tolist(),
                "weight_bits": query_graph.data.view(np.uint32).tolist(),
            }
        return records

    cases = [
        record("dense", local_connectivity=0.0, set_op_mix_ratio=1.0),
        record("dense", local_connectivity=1.0, set_op_mix_ratio=0.0),
        record("dense", local_connectivity=1.25, set_op_mix_ratio=0.375),
        record("dense", local_connectivity=2.0, set_op_mix_ratio=1.0),
        record("dense", local_connectivity=4.0, set_op_mix_ratio=0.125),
        record("dense", local_connectivity=0.5, set_op_mix_ratio=float(np.nextafter(1.0, 0.0))),
        record("dense", local_connectivity=0.5, set_op_mix_ratio=float(np.nextafter(0.0, 1.0))),
        record("sparse", local_connectivity=0.625, set_op_mix_ratio=0.5),
        record("sparse", local_connectivity=1.125, set_op_mix_ratio=0.75, metric="cosine"),
        record("categorical", local_connectivity=1.25, set_op_mix_ratio=0.375),
        record("distances", local_connectivity=1.25, set_op_mix_ratio=0.375),
    ]
    for init in ("explicit", "random", "spectral"):
        for epochs in (0, 2, 11):
            cases.append(record(
                "dense", init=init, transform_mode="embedding", n_epochs=epochs,
                local_connectivity=0.75, set_op_mix_ratio=0.375,
            ))
    cases.append(record(
        "dense", transform_mode="embedding", n_epochs=18,
        local_connectivity=2.0, set_op_mix_ratio=0.875,
    ))
    return {
        "options": options,
        "X": X.tolist(),
        "distances": distances.tolist(),
        "queries": queries.tolist(),
        "labels": labels.tolist(),
        "initial": initial.tolist(),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    with tempfile.TemporaryDirectory(prefix="umap-graph-option-fits-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        fixture = capture()
    content = json.dumps(fixture, allow_nan=False, separators=(",", ":")) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"public fixture differs from {args.output}")
    else:
        args.output.write_bytes(content.encode("utf-8"))
