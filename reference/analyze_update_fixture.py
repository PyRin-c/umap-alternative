"""Compare independent update initialization models with public UMAP snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode(record: dict) -> np.ndarray:
    width = np.dtype(record["dtype"]).itemsize * 8
    return (
        np.asarray(record[f"bits_uint{width}"], dtype=f"uint{width}")
        .view(record["dtype"])
        .reshape(record["shape"])
    )


def squared_distances(queries: np.ndarray, rows: np.ndarray) -> np.ndarray:
    delta = queries.astype(np.float64)[:, None, :] - rows.astype(np.float64)
    return np.sum(delta * delta, axis=2)


def normalize(initial: np.ndarray) -> np.ndarray:
    minimum = initial.min(axis=0)
    return 10 * (initial - minimum) / (initial.max(axis=0) - minimum)


def prepare_layout(
    initial: np.ndarray, seed: int, *, discard_rng_words: int = 3
) -> tuple[np.ndarray, float | None]:
    if len(np.unique(initial, axis=0)) == len(initial):
        return normalize(initial), None

    # Calibrated from public layouts with coincident provisional coordinates.
    distances = squared_distances(initial, initial)
    np.fill_diagonal(distances, np.inf)
    scale = 0.001 * np.sqrt(distances.min(axis=1)).mean()
    rng = np.random.RandomState(seed)
    rng.randint(0, 2**32, size=discard_rng_words, dtype=np.uint32)
    noise = rng.normal(0.0, scale, size=initial.shape).astype(np.float32)
    perturbed = initial + noise
    return normalize(perturbed), float(scale)


def summarize_prediction(predicted: np.ndarray, actual: np.ndarray, old_count: int) -> dict:
    matches = predicted.view(np.uint32) == actual.view(np.uint32)
    difference = np.abs(predicted.astype(np.float64) - actual)
    return {
        "equal_bits": int(matches.sum()),
        "total_coordinates": int(matches.size),
        "appended_equal_bits": int(matches[old_count:].sum()),
        "max_abs_error": float(difference.max()),
    }


def graph_centroids(graph: dict, old: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    assert graph["format"] == "csr" and graph["dtype"] == "float32"
    indices = np.asarray(graph["indices"], dtype=np.intp)
    values = np.asarray(graph["data_bits_uint32"], dtype=np.uint32).view(np.float32)
    means = []
    for first, last in zip(
        graph["indptr"][len(old):-1], graph["indptr"][len(old) + 1:]
    ):
        row = indices[first:last]
        previous = row < len(old)
        weights = values[first:last][previous]
        if weights.size and weights.sum() > 0:
            means.append(np.sum(old[row[previous]] * (weights / weights.sum())[:, None], axis=0))
        else:
            means.append(fallback)
    return np.asarray(means)


def analyze_step(
    training: np.ndarray,
    additions: np.ndarray,
    old: np.ndarray,
    actual: np.ndarray,
    graph: dict,
    neighbor_count: int,
    transform_seed: int,
    fit_seed: int,
) -> dict:
    new_count = len(additions)
    old_count = len(training)
    all_rows = np.concatenate((training, additions))
    distances = squared_distances(additions, all_rows)
    ranked = np.argsort(distances, axis=1, kind="stable")
    nearest = ranked[:, :neighbor_count]
    previous_neighbors = [row[row < old_count] for row in nearest]
    fallback = old.mean(axis=0, dtype=np.float64).astype(np.float32)

    def ordered_mean(rows):
        total = np.zeros(old.shape[1], dtype=np.float32)
        for row in rows:
            total += row
        return total / len(rows)

    centroids = np.array(
        [ordered_mean(old[row]) if row.size else fallback for row in previous_neighbors]
    )
    initial = np.concatenate((old, centroids))
    predicted, jitter_scale = prepare_layout(initial, transform_seed)

    # Alternatives isolate feature rank, graph weights, batch dependencies, and RNG state.
    previous_rank = np.argsort(distances[:, :old_count], axis=1, kind="stable")
    if neighbor_count > 1:
        previous_only = old[previous_rank[:, :neighbor_count - 1]].mean(axis=1)
    else:
        previous_only = np.repeat(fallback[None], new_count, axis=0)
    chained = initial.copy()
    new_as_zero = centroids.copy()
    for offset, row in enumerate(nearest):
        available = row[row < old_count + offset]
        chained[old_count + offset] = (
            chained[available].mean(axis=0) if available.size else fallback
        )
        if neighbor_count > 1:
            new_as_zero[offset] = (
                old[previous_neighbors[offset]].sum(axis=0) / (neighbor_count - 1)
            )

    alternatives = {
        "raw_centroids": initial,
        "without_jitter": normalize(initial),
        "no_rng_words_discarded": prepare_layout(
            initial, transform_seed, discard_rng_words=0
        )[0],
        "nearest_previous_rows_alone": prepare_layout(
            np.vstack((old, previous_only)), transform_seed
        )[0],
        "earlier_appended_coordinates_available": prepare_layout(chained, transform_seed)[0],
        "appended_neighbors_count_as_zero": prepare_layout(
            np.vstack((old, new_as_zero)), transform_seed
        )[0],
        "graph_weighted_previous_neighbors": prepare_layout(
            np.vstack((old, graph_centroids(graph, old, fallback))), transform_seed
        )[0],
    }
    if fit_seed != transform_seed:
        alternatives["fit_random_state"] = prepare_layout(initial, fit_seed)[0]

    return {
        "neighbor_count_retained_from_fit": neighbor_count,
        "nearest_combined_row_indices": nearest.tolist(),
        "previous_row_indices_used": [row.tolist() for row in previous_neighbors],
        "provisional_appended_bits_uint32": centroids.view(np.uint32).tolist(),
        "observed_appended_bits_uint32": actual[old_count:].view(np.uint32).tolist(),
        "predicted_appended_bits_uint32": predicted[old_count:].view(np.uint32).tolist(),
        "previous_coordinates_unchanged": int(
            (old.view(np.uint32) == actual[:old_count].view(np.uint32)).sum()
        ),
        "gaussian_jitter_scale": jitter_scale,
        "initialization_model": summarize_prediction(predicted, actual, old_count),
        "alternative_models": {
            name: summarize_prediction(values, actual, old_count)
            for name, values in alternatives.items()
        },
    }


def analyze(fixture: dict) -> dict:
    inputs = {name: decode(record) for name, record in fixture["inputs"].items()}
    analyses = {}
    for name, case in fixture["cases"].items():
        parameters = fixture["base_parameters"] | case["parameter_overrides"]
        if parameters["n_epochs"] not in (0, 1, 2) or not case["updates"]:
            continue
        if case["fit"] is None or case["fit"]["state"]["embedding"] is None:
            continue
        training = inputs[case["fit_input"]]
        neighbor_count = min(parameters["n_neighbors"], len(training) - 1)
        old = decode(case["fit"]["state"]["embedding"])
        steps = []
        for update in case["updates"]:
            if update["status"] != "ok":
                break
            additions = inputs[update["input"]]
            actual = decode(update["state"]["embedding"])
            steps.append(
                analyze_step(
                    training,
                    additions,
                    old,
                    actual,
                    update["state"]["graph"],
                    neighbor_count,
                    parameters["transform_seed"],
                    parameters["random_state"],
                )
            )
            old = actual
            training = np.concatenate((training, additions))
        if steps:
            analyses[name] = {"n_epochs": parameters["n_epochs"], "steps": steps}
    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="require exact zero- and one-epoch predictions"
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    analyses = analyze(fixture)
    print(json.dumps(analyses, indent=2, allow_nan=False))
    if args.check:
        for name, case in analyses.items():
            if case["n_epochs"] == 2:
                continue
            for index, step in enumerate(case["steps"]):
                match = step["initialization_model"]
                if match["equal_bits"] != match["total_coordinates"]:
                    raise SystemExit(f"{name} update {index + 1}: initialization bits differ")


if __name__ == "__main__":
    main()
