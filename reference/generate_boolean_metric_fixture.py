"""Capture public boolean feature metrics and independently replay their ranks."""

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
from scipy.spatial import distance as scipy_distance
from umap import UMAP, distances, sparse

from generate_sparse_lp_transform_fixture import encode


NAMES = {
    "hamming": "hamming", "matching": "matching", "jaccard": "jaccard", "dice": "dice",
    "russellrao": "russellrao", "rogerstanimoto": "rogers_tanimoto",
    "sokalmichener": "sokal_michener", "sokalsneath": "sokal_sneath", "yule": "yule",
    "kulsinski": "kulsinski",
}
NEEDS_WIDTH = ("hamming", "matching", "russellrao", "rogerstanimoto", "sokalmichener", "kulsinski")
OPTIONS = {
    "n_components": 2, "random_state": 19, "transform_seed": 0, "n_epochs": 0,
    "init": "random", "n_jobs": 1, "compatibility_layout": True,
    "a": 1.576943460, "b": .895060879,
}


def stored_zeros(matrix):
    present = matrix != 0
    present[np.arange(len(matrix)), np.arange(len(matrix)) % matrix.shape[1]] = True
    rows, columns = np.nonzero(present)
    return csr_matrix((matrix[rows, columns], (rows, columns)), shape=matrix.shape)


def datasets():
    signed = np.array([[0, 0, 0, 0], [-1, 2, 0, 0], [1, -2, 0, 0], [0, 0, -1, -2],
                       [1, 0, -2, 0], [-1, 2, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0],
                       [-0., 0, 0, 0], [1, 1, 1, 1]], np.float32)
    queries = np.array([[-0., 0, 0, 0], [-2, 7, 0, 0], [0, 0, 5, 7], [0, -1, 2, 0]], np.float32)
    eye = np.eye(16, dtype=np.float32) * np.where(np.arange(16) % 2 == 0, 1, -2)[:, None]
    tie_queries = np.vstack((np.zeros((1, 16), np.float32), 3 * eye[[7, 12]])).astype(np.float32)
    same_support = np.array([[1, 1, 1, 1, 1], [-1, 0, 2, 0, 0], [0, 1, 2, 0, 0],
                              [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [2, 0, -1, 0, 0],
                              [0, 0, 0, 0, 0], [0, 0, 1, 0, 1]], np.float32)
    equality_queries = np.array([[2, 0, 2, 0, 0], [0, 0, 0, 0, 0]], np.float32)
    return {
        "signed": (signed, queries), "ties": (eye.astype(np.float32), tie_queries),
        "stored": (stored_zeros(signed), stored_zeros(queries)),
        "same_support": (same_support, equality_queries),
        "isolated": (np.delete(signed, 8, axis=0), queries),
    }


def canonical(matrix, form):
    if form == "csr":
        return csr_matrix(matrix, copy=True)
    return matrix.toarray() if issparse(matrix) else matrix.copy()


def named_call(metric, left, right, *, csr):
    name = NAMES[metric]
    function = getattr(sparse, f"sparse_{name}", None) if csr else getattr(distances, name, None)
    if function is None:
        return "unavailable"
    if csr:
        args = (left.indices, left.data, right.indices, right.data)
        if metric in NEEDS_WIDTH:
            args += (left.shape[1],)
    else:
        args = (left, right)
    return scalar(function, *args)


def scalar(function, *args):
    if function is None:
        return "unavailable"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = float(function(*args))
        except Exception as error:
            return type(error).__name__
    return value if np.isfinite(value) else str(value)


def pair_observations(signed, stored):
    indices = [(0, 8), (0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (6, 7), (7, 9)]
    csr = csr_matrix(signed)
    observed = {}
    for metric, name in NAMES.items():
        scipy = getattr(scipy_distance, metric, None)
        observed[metric] = {
            "dense_name": name, "csr_name": f"sparse_{name}" if metric != "yule" else None,
            "csr_width_argument": metric in NEEDS_WIDTH,
            "named_dense": [named_call(metric, signed[i], signed[j], csr=False) for i, j in indices],
            "named_csr": [named_call(metric, csr[i], csr[j], csr=True) for i, j in indices],
            "stored_csr": [named_call(metric, stored[i], stored[j], csr=True) for i, j in indices],
            "scipy_numeric": [scalar(scipy, signed[i], signed[j]) for i, j in indices],
            "scipy_boolean": [scalar(scipy, signed[i] != 0, signed[j] != 0) for i, j in indices],
        }
    return {"pairs_of_signed_rows": indices, "calls": observed}


def support(matrix, *, stored):
    if not stored:
        values = matrix.toarray() if issparse(matrix) else matrix
        return values != 0
    mask = matrix.copy()
    mask.data = np.ones_like(mask.data)
    return mask.toarray() != 0


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=np.asarray(denominator) != 0)


def formula(metric, queries, training, model):
    n = training.shape[1]
    if metric == "hamming" and model != "support_hamming":
        q = queries.toarray() if issparse(queries) else queries
        t = training.toarray() if issparse(training) else training
        return np.count_nonzero(q[:, None] != t[None, :], axis=2) / n
    q = support(queries, stored=model == "stored_support")[:, None, :]
    t = support(training, stored=model == "stored_support")[None, :, :]
    both = np.count_nonzero(q & t, axis=2)
    left = np.count_nonzero(q & ~t, axis=2)
    right = np.count_nonzero(~q & t, axis=2)
    mismatch = left + right
    neither = n - both - mismatch
    if metric in ("matching", "hamming"):
        return mismatch / n
    if metric == "jaccard":
        return ratio(mismatch, mismatch + both)
    if metric == "dice":
        return ratio(mismatch, mismatch + 2 * both)
    if metric == "russellrao":
        value = (n - both) / n
        return value if model == "unconditional" else np.where(mismatch == 0, 0., value)
    if metric in ("rogerstanimoto", "sokalmichener"):
        return 2 * mismatch / (n + mismatch)
    if metric == "sokalsneath":
        return ratio(2 * mismatch, both + 2 * mismatch)
    if metric == "yule":
        return ratio(2 * left * right, both * neither + left * right)
    if metric == "kulsinski":
        value = (n + mismatch - both) / (n + mismatch)
        return value if model == "unconditional" else np.where(mismatch == 0, 0., value)
    raise ValueError(metric)


def named_matrices(metric, training, queries, *, csr):
    def matrix(left):
        rows = [[named_call(metric, x, y, csr=csr) for y in training] for x in left]
        if any(isinstance(value, str) for row in rows for value in row):
            return None
        return np.asarray(rows, dtype=np.float64)

    fit = matrix(training)
    query = matrix(queries)
    return None if fit is None or query is None else (fit, query)


def candidates(metric, training, queries, input_name):
    choices = ["formula"]
    if input_name == "stored" and metric != "hamming":
        choices.append("stored_support")
    if metric in ("russellrao", "kulsinski"):
        choices.append("unconditional")
    if metric == "hamming":
        choices.append("support_hamming")
    models = {name: (formula(metric, training, training, name), formula(metric, queries, training, name))
              for name in choices}
    named = named_matrices(metric, training, queries, csr=issparse(training))
    aliases = {}
    if named is not None:
        for name, pair in models.items():
            if all(np.array_equal(left, right) for left, right in zip(pair, named, strict=True)):
                aliases["named"] = name
                break
        else:
            models["named"] = named
    return models, aliases


def encoded_result(action):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = action()
        except Exception as error:
            result = {"error": type(error).__name__}
        else:
            result = {"value": encode(value)}
    result["warnings"] = [warning.category.__name__ for warning in caught]
    return result


def public_case(spec, training, queries):
    cutoff = decode_cutoff(spec["cutoff"])
    mode = spec["mode"]
    model = UMAP(**(OPTIONS | {
        "metric": spec["metric"], "n_neighbors": spec["k"], "transform_mode": mode,
        "disconnection_distance": cutoff, "n_epochs": spec["epochs"],
    }))
    fit_input = training.copy()
    fitted = encoded_result(lambda: model.fit_transform(fit_input))
    result = {"fit": fitted}
    if "error" in fitted:
        if hasattr(model, "graph_"):
            result["fit_graph"] = encode(model.graph_)
        return result
    result["query"] = encoded_result(lambda: model.transform(queries.copy()))
    if mode == "embedding":
        result["fit_graph"] = encode(model.graph_)
    if spec["input"] == "signed" and cutoff is None and mode == "graph":
        alternate = canonical(queries, "dense" if issparse(queries) else "csr")
        result["mixed_query"] = encoded_result(lambda: model.transform(alternate))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result["fit_input_identity"] = model.transform(training.copy()) is model.graph_
    if spec["input"] == "stored":
        without_zeros = training.copy()
        without_zeros.eliminate_zeros()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result["fit_input_identity"] = model.transform(training.copy()) is model.graph_
            result["without_zeros_identity"] = model.transform(without_zeros) is model.graph_
        result["fit_storage_unchanged"] = encode(fit_input) == encode(training)
    return result


def decode_cutoff(value):
    return float("inf") if value == "inf" else value


def supplied_graph(training, distances, k, cutoff, order, precision, *, inclusive):
    ranked = np.argsort(distances, axis=1, kind=order)[:, :k].astype(np.int32)
    wide = np.take_along_axis(distances, ranked, axis=1)
    narrow = wide.astype(np.float32)
    compared = wide if precision == "float64" else narrow
    cut = compared >= cutoff if inclusive else compared > cutoff
    ranked[cut] = -1
    narrow[cut] = np.inf
    # Explicit ranks expose the graph-building phase through a public parameter.
    model = UMAP(**OPTIONS, metric="euclidean", transform_mode="graph", n_neighbors=k,
                 disconnection_distance=np.inf, precomputed_knn=(ranked, narrow, None))
    return encoded_result(lambda: model.fit_transform(training.copy()))


def query_order(distances, k, cutoff, order, precision, *, inclusive):
    compared = distances.astype(precision)
    columns = np.argsort(compared, axis=1, kind=order)[:, :k].copy()
    selected = np.take_along_axis(compared, columns, axis=1)
    rows = np.broadcast_to(np.arange(len(distances))[:, None], columns.shape).copy()
    cut = selected >= cutoff if inclusive else selected > cutoff
    rows[cut] = columns[cut] = 0
    return rows.ravel().tolist(), columns.ravel().tolist()


def replay(spec, observed, models):
    k = spec["k"]
    given = decode_cutoff(spec["cutoff"])
    cutoff = given if given is not None else (1. if spec["metric"] in ("jaccard", "dice") else np.inf)
    output = {"fit": [], "query_order": [], "query_weights": []}
    for name, (training, queries) in models.items():
        # The unconditional candidate includes the SciPy-style zero square diagonal.
        training = training.copy()
        if name == "unconditional":
            np.fill_diagonal(training, 0.)
        graph = observed["fit"]["value"]
        query = observed["query"].get("value")
        dummy = np.zeros((len(training), 1), np.float32)
        choices = [(order, "float64", True) for order in ("stable", "quicksort")]
        if given is not None and np.isfinite(given):
            choices += [("stable", "float32", True), ("stable", "float64", False)]
        for order, precision, inclusive in choices:
            result = supplied_graph(dummy, training, k, cutoff, order, precision, inclusive=inclusive)
            if result.get("value") == graph:
                output["fit"].append([name, order, precision, ">=" if inclusive else ">"])
        if query is None:
            continue
        choices = [(order, precision, True) for order in ("stable", "quicksort")
                   for precision in ("float32", "float64")]
        if given is not None and np.isfinite(given):
            choices += [("quicksort", "float32", False)]
        for order, precision, inclusive in choices:
            rows, cols = query_order(queries, k, cutoff, order, precision, inclusive=inclusive)
            if query["row"] == rows and query["col"] == cols:
                output["query_order"].append([name, order, precision, ">=" if inclusive else ">"])
        model = UMAP(**OPTIONS, metric="precomputed", transform_mode="graph", n_neighbors=k,
                     disconnection_distance=cutoff)

        def precomputed_query():
            model.fit(training.astype(np.float32))
            return model.transform(queries.astype(np.float32))

        if encoded_result(precomputed_query).get("value") == query:
            output["query_weights"].append(name)
    return output


def specifications():
    for metric in NAMES:
        for form in ("dense", "csr"):
            common = {"metric": metric, "form": form, "mode": "graph", "epochs": 0}
            for label, cutoff in (("default", None), ("cut_one", 1.), ("uncut", "inf")):
                yield f"{metric}_{form}_{label}", common | {"input": "signed", "k": 9, "cutoff": cutoff}
            yield f"{metric}_{form}_ties", common | {"input": "ties", "k": 3, "cutoff": "inf"}
            for epochs in (0, 9):
                yield f"{metric}_{form}_embedding_{epochs}", common | {
                    "input": "signed", "mode": "embedding", "epochs": epochs, "k": 5, "cutoff": None,
                }
        yield f"{metric}_stored", {"metric": metric, "form": "csr", "mode": "graph", "epochs": 0,
                                  "input": "stored", "k": 7, "cutoff": "inf"}
    for metric in ("hamming", "matching", "jaccard", "dice", "yule"):
        threshold = 2. if metric == "yule" else 1.
        for label, cutoff in (("next64", np.nextafter(threshold, np.inf)),
                              ("next32", float(np.nextafter(np.float32(threshold), np.float32(np.inf)))),
                              ("zero", 0.)):
            yield f"{metric}_{label}", {"metric": metric, "form": "dense", "mode": "graph", "epochs": 0,
                                       "input": "signed", "k": 9, "cutoff": float(cutoff)}
    for metric in ("russellrao", "kulsinski"):
        for form in ("dense", "csr"):
            yield f"{metric}_{form}_same_support", {"metric": metric, "form": form, "mode": "graph",
                                                     "epochs": 0, "input": "same_support", "k": 2,
                                                     "cutoff": "inf"}
    for label, cutoff in (("at_two", 2.), ("over_two", float(np.nextafter(2., np.inf))),
                          ("over_two32", float(np.nextafter(np.float32(2.), np.float32(np.inf))))):
        yield f"yule_ties_{label}", {"metric": "yule", "form": "dense", "mode": "graph", "epochs": 0,
                                     "input": "ties", "k": 3, "cutoff": cutoff}
    for metric in ("jaccard", "dice"):
        for form in ("dense", "csr"):
            yield f"{metric}_{form}_isolated", {"metric": metric, "form": form, "mode": "embedding",
                                                "epochs": 9, "input": "isolated", "k": 5, "cutoff": None}
    yield "jaccard_zero_embedding", {"metric": "jaccard", "form": "dense", "mode": "embedding",
                                     "epochs": 0, "input": "signed", "k": 5, "cutoff": 0.}


def compact_observations(cases):
    keys = {}
    observations = {}
    linked_cases = {}
    for name, case in cases.items():
        observed = case["public"]
        key = keys.setdefault(json.dumps(observed, sort_keys=True, allow_nan=False), name)
        observations.setdefault(key, observed)
        linked_cases[name] = case | {"public": key}
    return {"cases": linked_cases, "observations": observations}


def run():
    inputs = datasets()
    main = {}
    residuals = {}
    named_disagreements = {}
    cache = {}
    for name, spec in specifications():
        x, q = (canonical(value, spec["form"]) for value in inputs[spec["input"]])
        observed = public_case(spec, x, q)
        record = {"spec": spec, "public": observed}
        model_failed = "error" in observed["fit"] or "error" in observed.get("query", {})
        if not model_failed and spec["mode"] == "graph":
            key = spec["metric"], spec["input"], spec["form"]
            if key not in cache:
                cache[key] = candidates(spec["metric"], x, q, spec["input"])
            matrices, aliases = cache[key]
            fitted = replay(spec, observed, matrices)
            fitted["aliases"] = aliases
            record["models"] = fitted
            model_failed = not all(fitted[stage] for stage in ("fit", "query_order", "query_weights"))
            named = aliases.get("named", "named" if "named" in matrices else None)
            matches = {
                "fit": any(row[0] == named for row in fitted["fit"]),
                "query_order": any(row[0] == named for row in fitted["query_order"]),
                "query_weights": named in fitted["query_weights"],
            }
            if named is not None and not all(matches.values()):
                named_disagreements[name] = {"model": named, "matches": matches}
        (residuals if model_failed else main)[name] = record
    for form in ("dense", "csr"):
        x, q = (canonical(value, form) for value in inputs["signed"])
        spec = {"metric": "kulczynski1", "form": form, "mode": "graph", "epochs": 0,
                "input": "signed", "k": 5, "cutoff": None}
        residuals[f"kulczynski1_{form}"] = {"spec": spec, "public": public_case(spec, x, q)}
    return {
        "packages": {name: version(name) for name in ("umap-learn", "numpy", "scipy", "numba",
                                                       "scikit-learn", "pynndescent")},
        "options": OPTIONS,
        "inputs": {name: [encode(x), encode(q)] for name, (x, q) in inputs.items()},
        "named_distances": pair_observations(inputs["signed"][0], inputs["stored"][0]),
    } | compact_observations(main), compact_observations(residuals) | {
        "named_disagreements": named_disagreements,
    }


def emit(path, artifact, *, check):
    content = "{\n" + ",\n".join(
        f'  "{key}": {json.dumps(value, allow_nan=False)}' for key, value in artifact.items()
    ) + "\n}\n"
    if check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path.name} differs from the generated reference")
    else:
        path.write_text(content, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    common, residuals = run()
    emit(args.output, common, check=args.check)
    emit(args.residual_output, residuals, check=args.check)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream JIT diagnostics can contain executable source text.
        raise SystemExit(type(error).__name__) from None
