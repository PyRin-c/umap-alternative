"""Record the trustworthiness API on the pinned public umap-learn wheel."""

import argparse
from base64 import b64encode
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from numba import njit
from umap import validation


@njit
def right_first_value(left, right):
    return right[0]


def array_record(array):
    array = np.asarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": array.shape,
        "bytes": b64encode(array.tobytes()).decode("ascii"),
    }


def observe(action):
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            result = {"output": array_record(action())}
        except Exception as error:
            result = {"error": type(error).__name__}
    result["warnings"] = [type(item.message).__name__ for item in observed]
    return result


def generate():
    assert version("umap-learn") == "0.6.0"
    cases = []
    inputs = {}

    def record(name, mode, input_name, max_k, metric, action):
        cases.append({
            "name": name, "mode": mode, "input": input_name, "k": max_k,
            "metric": metric, "observed": observe(action),
        })

    def add_input(name, source, embedded):
        inputs[name] = [array_record(source), array_record(embedded)]

    def vector(label, source, embedded, settings):
        add_input(label, source, embedded)
        for k, metric in settings:
            record(f"{label}-{metric}-{k}", "vector", label, k, metric,
                   lambda: validation.trustworthiness_vector(source, embedded, k, metric))

    rng = np.random.RandomState(1519)
    source = rng.uniform(-3, 3, (23, 7)).astype(np.float32)
    embedded = rng.uniform(-2, 2, (23, 2)).astype(np.float64)
    vector("ordinary", source, embedded, [(-1, "euclidean"), (0, "euclidean"),
           (np.int32(1), "euclidean"), (5, "euclidean"), (5, "minkowski"), (5, "l2"),
           (5, "manhattan"), (5, "l1"), (5, "chebyshev"), (1, "p"), (1, "infinity"),
           (1, "cityblock"), (1, "cosine"), (1, "precomputed"), (1.75, "euclidean")])

    for n in (15, 16, 17, 40, 80, 81, 127):
        source = rng.randint(-2, 4, (n, 6)).astype(np.float32)
        embedded = rng.randint(-2, 3, (n, 2)).astype(np.float64)
        vector(f"ties-{n}", source, embedded, [(7, metric) for metric in
               ("euclidean", "manhattan", "chebyshev")])

    source = rng.randint(-6, 7, (27, 4)).astype(np.float32)
    embedded = rng.normal(size=(27, 3)).astype(np.float32)
    vector("wide-f32", source * np.array([2.0**21, 1.0, 1.0, 1.0], np.float32), embedded,
           [(6, "euclidean"), (6, "minkowski"), (6, "manhattan")])
    vector("close-f64", np.float64(1.0) + source.astype(np.float64) * 2.0**-29, embedded,
           [(5, metric) for metric in ("euclidean", "manhattan", "chebyshev")])
    vector("signed-i64", source.astype(np.int64) + 2**60, embedded,
           [(5, metric) for metric in ("euclidean", "manhattan", "chebyshev")])
    vector("narrow-unsigned", source.astype(np.int64).astype(np.uint16), embedded,
           [(5, metric) for metric in ("euclidean", "manhattan", "chebyshev")])
    vector("unsigned-i64", (source.astype(np.int64) + 7).astype(np.uint64), embedded,
           [(5, metric) for metric in ("euclidean", "manhattan", "chebyshev")])
    vector("bool", source > 0, embedded,
           [(5, metric) for metric in ("euclidean", "manhattan", "chebyshev")])

    nonfinite = source.copy()
    nonfinite[1, 0] = np.nan
    nonfinite[2, 2] = np.inf
    nonfinite[-1, 0] = -np.inf
    vector("nonfinite-source", nonfinite, embedded,
           [(4, metric) for metric in ("euclidean", "manhattan", "chebyshev")])
    vector("nonfinite-embedding", embedded, nonfinite, [(4, "euclidean")])
    vector("zero-features-source", source[:, :0], embedded, [(4, "manhattan")])
    vector("zero-features-embedding", embedded, source[:, :0], [(4, "euclidean")])
    for n, settings in ((0, [0]), (1, [0, 1]), (2, [1]), (5, [2, 3, 4]), (7, [4, 5, 6, 7, -2])):
        vector(f"neighbors-{n}", source[:n], embedded[:n], [(k, "euclidean") for k in settings])

    ranked_source = np.stack([rng.permutation(9) for _ in range(9)]).astype(np.int64)
    ranked_embedded = np.stack([rng.permutation(9) for _ in range(9)]).astype(np.int64)
    for label, src, query, counts in (
        ("ranked-full", ranked_source, ranked_embedded, [0, 1, 4, 5, 6, 8, 9]),
        ("ranked-strided", ranked_source[:, ::-1], ranked_embedded[:, ::-1], [4]),
        ("ranked-signed-identifiers", ranked_source - 4, ranked_embedded - 4, [4]),
        ("ranked-first-occurrence", ranked_source.repeat(2, axis=1), ranked_embedded, [4]),
    ):
        add_input(label, src, query)
        for k in counts:
            record(f"{label}-{k}", "bulk", label, k, None,
                   lambda: validation.trustworthiness_vector_bulk(src, query, k))

    calculator = validation.make_trustworthiness_calculator(right_first_value)
    for label, src in (
        ("factory-wide", (1.0 + np.arange(16, dtype=np.float64)[::-1] * 2.0**-30)[:, None]),
        ("factory-ties", source[:16, :1]),
        ("factory-nonfinite", nonfinite[:16, :1]),
    ):
        query = np.stack([rng.permutation(len(src))[:6] for _ in src]).astype(np.int32)
        add_input(label, src, query)
        record(label, "factory", label, 6, "right_first_value", lambda: calculator(src, query, 6))

    platform_cases = {}
    for case in cases:
        if isinstance(case["k"], np.integer):
            case["k"] = int(case["k"])
        if case["input"] in ("ties-81", "ties-127"):
            platform_cases[case["name"]] = case.pop("observed")
    return {"umap-learn": "0.6.0", "inputs": inputs, "cases": cases}, platform_cases


def write_or_check(path, result, check):
    contents = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if check:
        if path.read_text(encoding="utf-8") != contents:
            raise SystemExit(f"trustworthiness fixture differs: {path}")
    else:
        path.write_bytes(contents.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    common, platform = generate()
    write_or_check(args.output, common, args.check)
    write_or_check(args.platform_output, platform, args.check)


if __name__ == "__main__":
    main()
