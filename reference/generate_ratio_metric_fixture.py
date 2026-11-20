"""Capture Canberra and Bray–Curtis through the public UMAP estimator."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse
from umap import UMAP


def encode(value):
    if issparse(value):
        coordinates = ([value.indptr.tolist(), value.indices.tolist()] if value.format == "csr"
                       else [value.row.tolist(), value.col.tolist()])
        return {
            "format": value.format, "shape": list(value.shape), "coordinates": coordinates,
            "bits": value.data.view(np.uint32).tolist(),
        }
    return value.view(np.uint32).tolist()


def capture():
    base = np.asarray([
        [1., 0., 0., 0.], [.3, .7, 0., 0.], [.5, 0., 1., 0.], [.8, 0., 0., 1.],
        [0., 1.4, 1., 0.], [0., .7, 0., 1.], [0., 0., .2, 1.], [1.1, .5, .7, 0.],
    ], dtype=np.float32)
    query = np.asarray([[-.2, .7, 0., .5], [0., 0., 1., .4], [.1, 0., -.5, 1.]], dtype=np.float32)
    ties = np.asarray([[0., 0.], [0., 0.], [1., .5], [-1., -.5], [0., 1.], [.3, .25]],
                      dtype=np.float32)
    rng = np.random.RandomState(123)
    scales = [1., 1e-3, 1e2, 1e-6]
    wide = (rng.randn(42, 23) * rng.choice(scales, size=(42, 23))).astype(np.float32)
    wide[rng.uniform(size=wide.shape) < .29] = 0
    wide_query = (rng.randn(11, 23) * rng.choice(scales, size=(11, 23))).astype(np.float32)
    data = {
        "base": (base, query), "ties": (ties, None),
        "wide_signed": (wide, wide_query), "wide_positive": (abs(wide), abs(wide_query)),
    }
    common = dict(
        n_neighbors=4, n_epochs=9, init="random", random_state=42, n_jobs=1,
        compatibility_layout=True, a=1.2, b=0.8,
    )
    cases = []
    for metric in ("canberra", "braycurtis"):
        for sparse in (False, True):
            convert = csr_matrix if sparse else np.asarray
            scenarios = [
                ("base", dict(transform_mode="graph"), True),
                ("base", {}, True),
                ("wide_signed", dict(n_neighbors=7, n_epochs=0, transform_mode="graph"), True),
                ("wide_positive", dict(n_neighbors=7, n_epochs=0, transform_mode="graph"), True),
            ]
            scenarios.extend(
                ("ties", dict(n_neighbors=3, n_epochs=0, transform_mode="graph",
                              disconnection_distance=cutoff), False)
                for cutoff in (None, 1.0, 0.5, 0.0)
            )
            for key, overrides, transform in scenarios:
                options = common | {"metric": metric} | overrides
                x, q = data[key]
                model = UMAP(**options)
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    output = model.fit_transform(convert(x))
                case = {
                    "dataset": key, "sparse": sparse, "options": options,
                    "graph": encode(model.graph_), "fit_warnings": [w.category.__name__ for w in seen],
                }
                if not issparse(output):
                    case["embedding"] = encode(output)
                if transform:
                    case["transform"] = encode(model.transform(convert(q)))
                if key == "base" and "embedding" in case:
                    if not sparse:
                        model.set_params(n_epochs=48)
                        points = model.embedding_[[1, 3]]
                        case["inverse_points"] = encode(points)
                        case["inverse"] = encode(model.inverse_transform(points))
                    model.set_params(n_epochs=2)
                    model.update(convert(q))
                    case["updated_graph"] = encode(model.graph_)
                    case["updated_embedding"] = encode(model.embedding_)
                cases.append(case)
    return {
        "data": {k: [x.tolist(), None if q is None else q.tolist()] for k, (x, q) in data.items()},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform-output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixture = capture()
    platform_weights = {}
    for case in fixture["cases"]:
        if case["dataset"] == "wide_positive" and case["options"]["metric"] == "braycurtis":
            key = "csr" if case["sparse"] else "dense"
            platform_weights[key] = case["transform"].pop("bits")
            case["transform"]["platform_weights"] = key
    for output, data in ((args.output, fixture), (args.platform_output, platform_weights)):
        text = "{\n" + ",\n".join(
            f'  "{key}": {json.dumps(value, separators=(",", ":"), allow_nan=False)}'
            for key, value in data.items()
        ) + "\n}\n"
        if args.check:
            if output.read_text(encoding="utf-8") != text:
                parser.exit(1, f"{output} differs from the regenerated fixture\n")
        else:
            output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream JIT diagnostics can contain source text.
        raise SystemExit(type(error).__name__) from None
