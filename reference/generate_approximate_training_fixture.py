"""Capture ordered ranks and graphs from the public seeded approximate fit."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from threadpoolctl import threadpool_limits


# Rows, features, neighbor count, input seed, and search seed.
CASES = (
    (59, 4, 5, 52, 37),
    (60, 4, 5, 52, 37),
    (61, 4, 5, 52, 37),
    (96, 4, 5, 42, 42),
    (100, 4, 5, 75, 13),
    (257, 8, 5, 28, 13),
    (4096, 4, 5, 42, 42),
    (4096, 7, 5, 57, 42),
    (4096, 9, 15, 89, 2),
    (4096, 11, 5, 88, 73),
)
SUFFIX = "-ubuntu" if sys.platform.startswith("linux") else ""
FIXTURE = Path(__file__).with_name("fixtures") / f"umap-learn-0.6.0-approximate-training{SUFFIX}.json"


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def numerical(values, keep_bits):
    if keep_bits:
        return {"float32_le": np.asarray(values, dtype="<f4").tobytes(order="C").hex()}
    return digest(values, "<f4")


def capture():
    from umap import UMAP
    from umap.umap_ import nearest_neighbors

    result = []
    cases = [("euclidean", *case) for case in CASES]
    angular_cases = CASES[3:4] + CASES[5:]
    cases.extend(("correlation", *case) for case in angular_cases)
    cases.extend(("cosine", *case) for case in angular_cases + (
        (30, 3, 2, 31, 74),
        (31, 3, 2, 31, 74),
        (127, 31, 5, 38, 43),
        (128, 32, 5, 39, 43),
        (129, 33, 5, 40, 43),
        (255, 64, 5, 13, 11),
    ))
    for metric, rows, features, width, data_seed, seed in cases:
        data = np.random.RandomState(data_seed).normal(size=(rows, features)).astype(np.float32)
        angular = metric != "euclidean"
        keep_bits = metric == "cosine" and features >= 31
        columns, distances, search = nearest_neighbors(
            data, width, metric, {}, angular, np.random.RandomState(seed), n_jobs=1,
        )
        model = UMAP(
            n_neighbors=width, metric=metric, n_jobs=1, random_state=seed, transform_mode="graph",
            force_approximation_algorithm=rows < 4096,
        ).fit(data)
        graph = model.graph_.tocsr()
        queries = np.random.RandomState(5).normal(size=(32, features)).astype(np.float32)
        query_ranks = {}
        for query_width in (1, width):
            query_indices, query_distances = search.query(
                queries, k=query_width, epsilon=0.24 if angular else 0.12,
            )
            query_ranks[query_width] = {
                "indices": digest(query_indices, "<i4"),
                "distances": numerical(query_distances, keep_bits),
            }
        transformed = model.transform(queries).tocoo()
        result.append(dict(
            metric=metric, rows=rows, features=features, width=width, data_seed=data_seed, seed=seed,
            columns=digest(columns, "<i4"), distances=numerical(distances, keep_bits),
            indptr=digest(graph.indptr, "<i8"), indices=digest(graph.indices, "<i4"),
            weights=numerical(graph.data, keep_bits),
            query=query_ranks, transform=dict(
                rows=digest(transformed.row, "<i4"), columns=digest(transformed.col, "<i4"),
                weights=numerical(transformed.data, keep_bits),
            ),
        ))
    caller_rng = []
    data = np.random.RandomState(42).normal(size=(96, 4)).astype(np.float32)
    for prior in ("fresh", "gaussian", "twist"):
        for epochs in (0, 2):
            rng = np.random.RandomState(37)
            if prior == "gaussian":
                rng.normal(size=13)
            elif prior == "twist":
                rng.randint(2**16, size=623)
            model = UMAP(
                n_neighbors=5, n_epochs=epochs, init="random", n_jobs=1,
                random_state=rng, compatibility_layout=True, force_approximation_algorithm=True,
            )
            layout = model.fit_transform(data)
            _, words, cursor, has_gaussian, gaussian = rng.get_state()
            caller_rng.append(dict(
                prior=prior, epochs=epochs, layout=digest(layout, "<f4"),
                words=digest(words, "<u4"), cursor=cursor,
                has_gaussian=has_gaussian, gaussian=digest(gaussian, "<f8"),
            ))
    return {"cases": result, "caller_rng": caller_rng}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            records = capture()
        except Exception as error:
            raise SystemExit(type(error).__name__) from None
    output = json.dumps(records, indent=2) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != output:
            raise SystemExit("public approximate training fixture differs")
    else:
        args.output.write_text(output, encoding="utf-8", newline="\n")
