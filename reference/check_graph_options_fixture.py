"""Replay the public graph-options fixture against the installed UMAP wheel."""

import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path


def check():
    import numpy as np
    from umap.umap_ import fuzzy_simplicial_set

    root = Path(__file__).resolve().parents[1]
    path = root / "crates/umap-core/tests/fixtures/graph_options.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    environment = fixture["reference"]
    for package in ("umap_learn", "numpy", "scipy", "numba"):
        if version(package) != environment[package]:
            raise RuntimeError(f"{package} must be {environment[package]}")
    host = platform.system()
    if (
        platform.python_version() != environment["python"]
        or host not in ("Linux", "Windows")
        or platform.machine().lower() not in ("x86_64", "amd64")
    ):
        raise RuntimeError("Use the pinned Python on Windows or Linux x86-64")

    graph_count = 0
    parameter_count = 0
    for dataset in fixture["datasets"]:
        indices = np.asarray(dataset["neighbors"], dtype=np.int32)
        distances = np.asarray(dataset["distance_bits"], dtype=np.uint32).view(np.float32)
        data = np.zeros((len(indices), 1), dtype=np.float32)
        for profile in dataset["profiles"]:
            parameter_count += len(indices)
            connectivity = profile["connectivity"]
            for expected in profile["graphs"]:
                mix = expected["mix"]
                graph, sigma, rho = fuzzy_simplicial_set(
                    data,
                    indices.shape[1],
                    np.random.RandomState(42),
                    "euclidean",
                    knn_indices=indices.copy(),
                    knn_dists=distances.copy(),
                    local_connectivity=connectivity,
                    set_op_mix_ratio=mix,
                )
                case = (dataset["name"], connectivity, mix)
                assert sigma.view(np.uint32).tolist() == profile["sigma_bits"], (case, "sigma")
                assert rho.view(np.uint32).tolist() == profile["rho_bits"], (case, "rho")
                graph = graph.tocsr()
                assert graph.indptr.tolist() == expected["indptr"], (case, "indptr")
                assert graph.indices.tolist() == expected["indices"], (case, "indices")
                bits = expected["data_bits"]
                if host == "Linux":
                    bits = expected.get("linux_x86_64_data_bits", bits)
                assert graph.data.view(np.uint32).tolist() == bits, (case, "weight bits")
                graph_count += 1
    print(f"{graph_count} public graphs and {parameter_count} offset/scale pairs matched on {host}")


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with tempfile.TemporaryDirectory(prefix="umap-graph-options-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        check()
