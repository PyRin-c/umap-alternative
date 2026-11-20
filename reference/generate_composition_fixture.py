"""Record fuzzy set operators through the public umap-learn estimators."""

import argparse
import json
import operator
import warnings
from pathlib import Path


def record():
    import numpy as np
    from scipy.sparse import csr_matrix
    from umap import UMAP

    left_features = [[0, 0], [2, 0], [0, 3], [3, 2], [4, 4], [5, 0], [5, 2], [4, 5]]
    right_features = [[0, 0], [0, 2], [2, 1], [4, 2], [3, 4], [5, 1], [7, 0], [2, 5]]
    common = {"n_neighbors": 4, "n_jobs": 1, "init": "random", "compatibility_layout": True}
    case_options = [
        ("fitted_zero", {}, {}, None),
        ("fitted_positive", {"n_epochs": 8, "a": 1.2, "b": 0.8, "learning_rate": 1.7,
                             "repulsion_strength": 0.6, "negative_sample_rate": 2},
         {"n_epochs": 4, "a": 1.6, "b": 1.4, "learning_rate": 0.4,
          "repulsion_strength": 1.3, "negative_sample_rate": 5}, None),
        ("varied_memberships", {}, {}, (
            [1.0, 0.95, 0.8, 0.5, 0.2, 0.09, 0.02],
            [0.3, 0.8, 0.01, None, 0.6, 0.8, 0.00012],
        )),
        ("four_memberships", {}, {}, ([1.0, 0.2, 0.02, 0.004], [])),
        ("short_row", {}, {}, ([1.0, 0.5, 1.0e-20], [])),
        ("one_sided", {}, {}, ([1.0], [0.5] * 7)),
        ("stored_zeros", {}, {}, (
            [1.0, 0.75, 0.5, 0.0, 0.25, None, 0.0625],
            [0.5, None, 0.1, 0.0, 0.2, None, 0.0],
        )),
        ("disjoint", {}, {}, ([1.0, 0.25], [None, None, 0.75, 0.5])),
        ("pruned_bridge", {"n_epochs": 11}, {}, "bridge"),
    ]

    def graph(weights):
        edges = [(0, i + 1, w) for i, w in enumerate(weights) if w is not None]
        edges.extend((i, i % 7 + 1, 1.0) for i in range(1, 8))
        rows, columns, data = zip(*edges)
        return csr_matrix((np.asarray(data, dtype=np.float32), (rows, columns)), shape=(8, 8))

    def csr_bits(csr):
        return [csr.indptr.tolist(), csr.indices.tolist(), csr.data.view(np.uint32).tolist()]

    cases = []
    for name, left_overrides, right_overrides, weights in case_options:
        left_options = common | {"n_components": 2, "n_epochs": 0, "random_state": 34}
        right_options = common | {"n_components": 3, "n_epochs": 0, "random_state": 88}
        left_options.update(left_overrides)
        right_options.update(right_overrides)
        left = UMAP(**left_options).fit(np.asarray(left_features, dtype=np.float32))
        right = UMAP(**right_options).fit(np.asarray(right_features, dtype=np.float32))
        if weights == "bridge":
            chain = csr_matrix(
                (np.asarray([1.0, 0.04, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                 (np.arange(7), np.arange(1, 8))), shape=(8, 8)
            )
            left.graph_ = chain + chain.T
            right.graph_ = csr_matrix(([1.0], ([7], [7])), shape=(8, 8), dtype=np.float32)
        elif weights is not None:
            left.graph_, right.graph_ = map(graph, weights)
        operands = [csr_bits(left.graph_), csr_bits(right.graph_)]
        outputs = {}
        for label, op in (("union", operator.add), ("intersection", operator.mul),
                          ("difference", operator.sub)):
            left_input, right_input = left.graph_.copy(), right.graph_.copy()
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                result = op(left, right)
            remaining = [csr_bits(left.graph_), csr_bits(right.graph_)]
            left.graph_, right.graph_ = left_input.copy(), right_input.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                repeated = op(left, right)
            left.graph_, right.graph_ = left_input, right_input
            bits = result.embedding_.view(np.uint32)
            outputs[label] = {
                "graph": csr_bits(result.graph_),
                "embedding": bits.tolist() if np.array_equal(
                    bits, repeated.embedding_.view(np.uint32)
                ) and weights in (None, "bridge") else None,
                "warnings": [notice.category.__name__ for notice in seen],
                "operands_after": remaining,
            }
        cases.append({
            "name": name, "options": [left_options, right_options],
            "operands": operands, "outputs": outputs,
            "override_graphs": weights is not None,
        })
    return {"features": [left_features, right_features], "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixture = record()
    bridge = next(case for case in fixture["cases"] if case["name"] == "pruned_bridge")
    bridge_layouts = {}
    for name, output in bridge["outputs"].items():
        bridge_layouts[name] = output["embedding"]
        output["embedding"] = None
    for path, content in ((args.output, fixture), (args.bridge_output, bridge_layouts)):
        serialized = "{\n" + ",\n".join(
            f'  "{key}": {json.dumps(value, separators=(",", ":"))}'
            for key, value in content.items()
        ) + "\n}\n"
        if args.check:
            if path.read_text(encoding="utf-8") != serialized:
                parser.exit(1, f"{path} differs from the regenerated fixture\n")
        else:
            path.write_text(serialized, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream JIT diagnostics can include source text.
        raise SystemExit(type(error).__name__) from None
