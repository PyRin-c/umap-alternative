"""Capture public AlignedUMAP and replay its initial alignment from fitted graphs."""

import argparse
import json
import os
import platform
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path


def fixture_path():
    platform_label = "windows" if platform.system() == "Windows" else "ubuntu"
    return (
        Path(__file__).with_name("fixtures")
        / f"umap-learn-0.6.0-aligned-{platform_label}-x86_64.json"
    )


def eigen_layout(record, n_components):
    from math import isqrt

    import numpy as np
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import eigsh

    n_rows = len(record["indptr"]) - 1
    n_eigenpairs = n_components + 1
    weights = np.asarray(record["bits"], dtype=np.uint32).view(np.float32)
    graph = csr_matrix(
        (weights, record["indices"], record["indptr"]), shape=(n_rows, n_rows)
    )
    inverse_degree = diags(1.0 / np.sqrt(np.asarray(graph.sum(axis=0)).ravel()))
    laplacian = eye(n_rows, dtype=np.float64, format="csr") - inverse_degree @ graph @ inverse_degree
    values, vectors = eigsh(
        laplacian,
        k=n_eigenpairs,
        which="SM",
        ncv=max(2 * n_eigenpairs + 1, isqrt(n_rows)),
        v0=np.ones(n_rows, dtype=np.float64),
        tol=1e-4,
    )
    layout = vectors[:, np.argsort(values)[1:n_eigenpairs]]
    return (layout * (10 / np.abs(layout).max())).astype(np.float32)


def align_initial_layout(graphs, relations, n_components):
    import numpy as np
    from scipy.linalg import svd

    layout = [eigen_layout(graphs[0], n_components)]
    for graph, links in zip(graphs[1:], relations, strict=True):
        current = eigen_layout(graph, n_components)
        backward = {destination: source for source, destination in links.items()}
        destination = sorted(backward)
        source = [backward[index] for index in destination]
        covariance = current[destination].T @ layout[-1][source]
        u, _, vt = svd(covariance, lapack_driver="gesdd")
        layout.append(current @ (u @ vt))
    return layout


def update_initial_layout(previous, graph, links):
    import numpy as np

    inverse = {destination: source for source, destination in links.items()}
    weights = np.asarray(graph["bits"], dtype=np.uint32).view(np.float32)
    output = np.empty((len(graph["indptr"]) - 1, previous.shape[1]), dtype=np.float32)
    for row, (begin, end) in enumerate(zip(graph["indptr"], graph["indptr"][1:])):
        if row in inverse:
            output[row] = previous[inverse[row]]
            continue
        total = 0.0
        accum = np.zeros(previous.shape[1], dtype=np.float32)
        for entry in range(begin, end):
            column = graph["indices"][entry]
            weight = weights[entry]
            if column in inverse and weight > 0:
                accum += weight * previous[inverse[column]]
                total += float(weight)
        if total == 0.0:
            raise ValueError("update replay requires at least one linked neighbor per unlinked row")
        output[row] = accum.astype(np.float64) / total
    return output


def capture():
    import numpy as np
    from umap import AlignedUMAP

    def family(seed, size, width, extra, drop):
        rng = np.random.RandomState(seed)
        first = rng.normal(size=(size, width)).astype(np.float32)
        b_perm = rng.permutation(size)
        second = np.vstack((
            first[b_perm] * np.linspace(1.3, 0.8, width) + 2,
            rng.normal(size=(extra, width)) + 1.5,
        )).astype(np.float32)
        c_perm = rng.permutation(len(second))[drop:]
        third = (second[c_perm] * np.linspace(0.7, 1.4, width) - 1).astype(np.float32)
        return [first, second, third], [
            {int(origin): target for target, origin in enumerate(b_perm)},
            {int(origin): target for target, origin in enumerate(c_perm)},
        ]

    first = np.random.RandomState(837).normal(size=(9, 3)).astype(np.float32)
    b_perm = [5, 2, 0, 8, 1, 7, 4, 3, 6]
    second = np.vstack((
        first[b_perm] * [1.3, 0.8, 1.1] + [2, -3, 1],
        [[0.5, -2.3, 0.75]],
    )).astype(np.float32)
    c_perm = [8, 7, 2, 4, 6, 9, 0, 5]
    third = (second[c_perm] * [0.71, 1.4, 0.87] + [-0.5, 1.5, 2]).astype(np.float32)
    r_ab, r_bc = [
        {origin: target for target, origin in enumerate(permutation)}
        for permutation in (b_perm, c_perm)
    ]
    data = {"base": [first, second, third]}
    data["holdout"], holdout_groups = family(491, 14, 4, 3, 5)
    relation_groups = {
        "complete": [r_ab, r_bc],
        "identity": [{i: i for i in range(9)}, {i: i for i in range(8)}],
        "partial": [{i: r_ab[i] for i in (0, 2, 4, 6, 8)}, {i: r_bc[i] for i in (0, 2, 4)}],
        "empty": [{}, {}],
        "short": [r_ab],
        "long": [r_ab, r_bc, {}],
        "duplicate_target": [r_ab | {4: r_ab[2]}, r_bc],
        "holdout": holdout_groups,
    }
    base_settings = dict(
        n_neighbors=4, n_components=2, a=1.7, b=0.9, random_state=42,
        compatibility_layout=True, init="random", n_epochs=0,
        alignment_window_size=2, alignment_regularisation=0.01,
    )
    graphs = {}
    cases = {}

    def bits(array):
        assert array.dtype == np.float32
        return np.ascontiguousarray(array).view(np.uint32).tolist()

    def observe(model, call):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                result = call()
            except Exception as error:
                record = {"error": type(error).__name__, "message": str(error)}
            else:
                record = {
                    "returns": "self" if result is model else type(result).__name__,
                    "returns_embeddings": result is getattr(model, "embeddings_", None)
                    and result is not None,
                }
        record["warnings"] = [
            [item.category.__name__, str(item.message)] for item in captured
            if item.category.__name__ != "NumbaTypeSafetyWarning"
        ]
        if hasattr(model, "embeddings_"):
            record["embeddings"] = [bits(layout) for layout in model.embeddings_]
        if hasattr(model, "n_models_"):
            record["n_models"] = int(model.n_models_)
        if hasattr(model, "mappers_"):
            ids = []
            for mapper in model.mappers_:
                graph = mapper.graph_
                csr = {
                    "indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(),
                    "bits": bits(graph.data),
                }
                graph_id = next((id for id, other in graphs.items() if csr == other), None)
                if graph_id is None:
                    graph_id = str(len(graphs))
                    graphs[graph_id] = csr
                ids.append(graph_id)
            record["graphs"] = ids
        return record

    def fit(name, group="complete", family_name="base", method="fit", **overrides):
        np.random.seed(839)
        settings = base_settings | overrides
        model = AlignedUMAP(**settings)
        rows = data[family_name]
        relations = {} if group is None else {"relations": relation_groups[group]}
        record = observe(model, lambda: getattr(model, method)([x.copy() for x in rows], **relations))
        cases[name] = {
            "family": family_name, "relation_group": group, "method": method,
            "overrides": overrides, **record,
        }

    for epochs in (0, 1, 2, 8):
        for coefficient in (0.0, 0.01, 0.3):
            fit(f"epoch-{epochs}-weight-{coefficient}", n_epochs=epochs, alignment_regularisation=coefficient)
    for mode in ("identity", "partial", "empty", "short", "long", "duplicate_target"):
        fit(mode, group=mode)
    fit("missing", group=None)
    fit("single-window", n_epochs=8, alignment_window_size=1)
    fit("spectral-mappers", init="spectral")
    fit("fit-transform", n_epochs=8, method="fit_transform")
    fit("holdout", group="holdout", family_name="holdout", n_components=3, random_state=7)

    update_links = {src: dst for src, dst in r_bc.items() if dst in (0, 2, 4, 6)}
    initial_update_bits = {}
    updates = [(f"update-{epochs}", epochs, r_bc) for epochs in (0, 2, 8)]
    updates.append(("update-partial-0", 0, update_links))
    for name, epochs, links in updates:
        np.random.seed(839)
        model = AlignedUMAP(**(base_settings | {"n_epochs": epochs}))
        before = observe(model, lambda: model.fit([first.copy(), second.copy()], relations=[r_ab]))
        after = observe(model, lambda: model.update(third.copy(), relations=links))
        assert before["embeddings"] == after["embeddings"][:2]
        cases[name] = {
            "overrides": {"n_epochs": epochs}, "relations": list(links.items()),
            "before": before, "after": after,
        }
        if epochs == 0:
            previous = np.asarray(before["embeddings"][-1], dtype=np.uint32).view(np.float32)
            replay = update_initial_layout(previous, graphs[after["graphs"][-1]], links)
            expected = after["embeddings"][-1]
            assert bits(replay) == expected, name
            initial_update_bits[name] = replay.size

    # Replay from the captured adjacency, independently of the estimator.
    counts = {}
    for name in ("epoch-0-weight-0.01", "identity", "partial", "duplicate_target", "holdout"):
        case = cases[name]
        actual = align_initial_layout(
            [graphs[id] for id in case["graphs"]], relation_groups[case["relation_group"]],
            (base_settings | case["overrides"])["n_components"],
        )
        counts[name] = [
            int(np.count_nonzero(np.asarray(bits(layout)) == np.asarray(expected)))
            for layout, expected in zip(actual, case["embeddings"], strict=True)
        ]

    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "packages": {
            package: version(package) for package in ("umap-learn", "numpy", "scipy", "numba", "scikit-learn")
        },
        "settings": base_settings,
        "data": {name: [matrix.tolist() for matrix in rows] for name, rows in data.items()},
        "relations": {
            name: [list(mapping.items()) for mapping in group] for name, group in relation_groups.items()
        },
        "graphs": graphs,
        "cases": cases,
        "initial_replay_matching_bits": counts,
        "initial_update_matching_bits": initial_update_bits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=fixture_path())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if version("umap-learn") != "0.6.0" or platform.python_version() != "3.13.12":
        raise SystemExit("Use the locked reference environment in reference/")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with tempfile.TemporaryDirectory(prefix="umap-aligned-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        result = capture()
    content = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"public fixture differs from {args.output}")
    else:
        args.output.write_bytes(content.encode("utf-8"))
    print(result["initial_replay_matching_bits"])
