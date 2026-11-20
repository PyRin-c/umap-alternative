"""Capture public sparse Manhattan and Chebyshev fits and transforms."""

import argparse
import json
import os
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1"
))

import numpy as np
from scipy.sparse import csr_matrix, issparse
from umap import UMAP


def read_fixture(name):
    folder = Path(__file__).with_name("fixtures")
    return json.loads((folder / f"umap-learn-0.6.0-{name}-x86_64.json").read_bytes())


def decode_csr(record):
    data = np.asarray(record["data"]["bits"], dtype=np.uint32).view(np.float32)
    return csr_matrix((data, record["indices"], record["indptr"]), shape=record["shape"])


def encode(values):
    if not issparse(values):
        return {"shape": list(values.shape), "dtype": values.dtype.name,
                "bits": values.view(f"uint{values.itemsize * 8}").ravel().tolist()}
    result = {"shape": list(values.shape), "format": values.format, "data": encode(values.data)}
    if values.format == "coo":
        return result | {"row": values.row.tolist(), "col": values.col.tolist()}
    return result | {"indices": values.indices.tolist(), "indptr": values.indptr.tolist()}


def warning_records(caught):
    return [[item.category.__name__, str(item.message)] for item in caught]


def build():
    base = read_fixture("sparse-euclidean-transform")
    options = {key: value for key, value in base["options"].items() if key != "metric"}
    options["transform_seed"] = 0
    training = {name: decode_csr(base["training"][name])
                for name in ("csr", "wide_ties", "cutoff", "cancellation")}
    X = training["csr"]
    training["split_csr"] = csr_matrix(
        (np.repeat(X.data / 2, 2), np.repeat(X.indices, 2), X.indptr * 2), shape=X.shape,
    )
    batch = decode_csr(base["queries"]["csr"])
    split = decode_csr(base["queries"]["split_csr"])
    eps = np.float32(2**-25)
    upper_cutoff = float(np.nextafter(np.float32(1), np.float32(2)))
    terms = np.array([[1, eps, eps, 0], [1.05, 0, eps, 0], [1.18, eps, 0, .05],
                      [1.25, 0, eps, .1], [1.4, .05, -.01, 0], [1.5, 0, .1, 0]], np.float32)
    summed = terms.copy()
    summed[0, 3] = eps
    maxed = np.array([[upper_cutoff, .125, -.3, 0], [1.05, 0, .3, 0], [1.16, 0, .25, 0],
                      [1.23, 0, .125, .125], [1.35, .1, .2, 0], [1.41, .06, .15, .09]],
                      np.float32)
    collision = np.array([[1, eps, 0], [1, 0, 0], [1, 2*eps, 0], [1.06, 0, 0],
                          [1.125, 0, 0], [1.18, 0, .05], [1.35, .01, 0], [1.4, 0, 0]],
                          np.float32)
    training.update({
        "terms": csr_matrix(terms),
        "terms_fit": csr_matrix(np.vstack((np.array([[-eps, 0, 0, 0]], np.float32), terms))),
        "summed": csr_matrix(summed),
        "summed_fit": csr_matrix(np.vstack((np.zeros((1, 4), np.float32), summed))),
        "maxed": csr_matrix(maxed),
        "maxed_fit": csr_matrix(np.vstack((np.array([[eps, 0, .2, 0]], np.float32), maxed))),
        "collision": csr_matrix(collision),
        "collision_fit": csr_matrix(np.vstack((np.zeros((1, 3), np.float32), collision))),
        "zeros": csr_matrix((8, 5), dtype=np.float32),
    })
    cancel = csr_matrix((
        np.array([1e8, 1, -1e8, 2, 4], dtype=np.float64), [0] * 5, [0, 3, 4, 5],
    ), shape=(3, 1))
    queries = {
        "dense": batch.toarray(), "csr": batch, "csc": batch.tocsc(), "coo": split.tocoo(),
        "split": split, "narrow": batch[:, :-1], "f64": batch.astype(np.float64),
        "empty": decode_csr(base["queries"]["empty_rows"]), "fit_input": X.copy(),
        "fit_input_split": training["split_csr"].copy(), "tie": np.array([[0, 0], [0, 1], [0, -1]], np.float32),
        "cutoff": decode_csr(base["queries"]["cutoff"]),
        "cancel_csr": cancel, "cancel_csc": cancel.tocsc(), "cancel_coo": cancel.tocoo(),
        "terms": csr_matrix(np.array([[-eps, 0, 0, 0], [0, 0, 0, 0], [-3*eps, 0, 0, 0]], np.float32)),
        "summed": csr_matrix(np.array([[0, 0, 0, 0], [-eps, 0, 0, 0], [0, 0, 0, eps]], np.float32)),
        "maxed": csr_matrix(np.array([[eps, 0, .2, 0], [2*eps, 0, .2, 0], [3*eps, 0, .2, 0]], np.float32)),
        "collision": csr_matrix((1, 3), dtype=np.float32),
        "narrow_collision": csr_matrix((1, 2), dtype=np.float32),
        "zeros": csr_matrix((2, 5), dtype=np.float32),
    }
    cases = {}

    def capture(name, metric, source, mode, query_names, *, extra=(), **overrides):
        parameters = overrides | {"metric": metric}
        model = UMAP(**(options | parameters), transform_mode=mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(training[source].copy())
        fitted = model.graph_ if mode == "graph" else model.embedding_
        result = {
            "training": source, "mode": mode, "fit_mode": mode, "options": parameters,
            "fitted": encode(fitted), "fit_warnings": warning_records(caught), "transforms": {},
        }
        if mode == "embedding":
            result["fitted_graph"] = encode(model.graph_)
        calls = [(label, label, {}) for label in query_names] + list(extra)
        for label, query, changes in calls:
            original = {key: getattr(model, key) for key in changes}
            model.set_params(**changes)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                output = model.transform(queries[query].copy())
            call = {"identity": output is fitted, "warnings": warning_records(messages),
                    "query": query, "parameters": changes}
            if output is not fitted:
                call["result"] = encode(output)
            result["transforms"][label] = call
            model.set_params(**original)
        cases[name] = result

    for metric in ("manhattan", "chebyshev"):
        for mode in ("graph", "embedding"):
            capture(f"{metric}_{mode}_formats", metric, "csr", mode,
                    ("dense", "csr", "csc", "coo", "split", "narrow", "f64", "empty", "fit_input"))
            capture(f"{metric}_{mode}_ties", metric, "wide_ties", mode, ("tie",), n_epochs=0)
            capture(f"{metric}_{mode}_cutoff", metric, "cutoff", mode, ("cutoff",),
                    disconnection_distance=.5)
            capture(f"{metric}_{mode}_cancellation", metric, "cancellation", mode,
                    ("cancel_csr", "cancel_csc", "cancel_coo"), n_neighbors=3)
            capture(f"{metric}_{mode}_zero_features", metric, "zeros", mode, ("zeros",),
                    n_neighbors=2)
        capture(f"{metric}_duplicate_fit", metric, "split_csr", "embedding",
                ("fit_input_split", "fit_input", "csr"), n_epochs=0,
                extra=(("positive_budget", "csr", {"n_epochs": 30, "transform_seed": 2**32 - 1}),))
        capture(f"{metric}_spectral", metric, "csr", "embedding", ("csr",), init="spectral")

    for name in ("terms", "summed", "maxed"):
        metric = "chebyshev" if name == "maxed" else "manhattan"
        for mode in ("graph", "embedding"):
            capture(f"{name}_{mode}_query", metric, name, mode, (name,),
                    disconnection_distance=upper_cutoff)
    for factor in (3., 3.5, 4.):
        capture(f"sum_fit_{factor}", "manhattan", "summed_fit", "embedding", (),
                disconnection_distance=1 + factor * float(eps))
    capture("terms_fit", "manhattan", "terms_fit", "embedding", (),
            disconnection_distance=1 + 2.5 * float(eps))
    capture("max_fit", "chebyshev", "maxed_fit", "embedding", (), disconnection_distance=upper_cutoff)
    capture("wide_rank_query", "manhattan", "collision", "graph", ("collision", "narrow_collision"),
            n_neighbors=2)
    capture("wide_rank_fit", "manhattan", "collision_fit", "embedding", (), n_neighbors=2)
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                      "scikit-learn", "pynndescent")},
        "options": options,
        "training": {name: encode(data) for name, data in training.items()},
        "queries": {name: encode(data) for name, data in queries.items()},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in build().items()
    ) + "\n}\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, f"{args.output} differs from the regenerated fixture")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream JIT exception text can include code excerpts.
        raise SystemExit(type(error).__name__) from None
