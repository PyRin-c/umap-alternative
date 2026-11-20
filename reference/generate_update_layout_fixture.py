"""Capture controlled layout checkpoints through the public UMAP estimator."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

from generate_update_fixture import (
    BASE_PARAMETERS,
    INPUTS,
    PACKAGE_VERSIONS,
    PYTHON_VERSION,
    THREAD_LIMITS,
    UMAP,
    array_record,
    graph_record,
)

import numpy as np


def seed_boundary(seed: int, shape: list[int], jitter: bool) -> dict:
    rng = np.random.RandomState(seed)
    first_words = rng.randint(0, 0xFFFFFFFD, size=3, dtype=np.uint32)
    before_jitter = rng.get_state()[2]
    if jitter:
        rng.standard_normal(size=shape)
    jitter_raw_words = rng.get_state()[2] - before_jitter
    optimizer_words = rng.randint(0, 0xFFFFFFFD, size=3, dtype=np.uint32)
    return {
        "initial_raw_words": first_words.tolist(),
        "jitter_raw_word_count": jitter_raw_words,
        "optimizer_raw_words": optimizer_words.tolist(),
    }


def capture(batch: str, fit_overrides: dict, trials: dict, jitter: bool) -> dict:
    fit_parameters = BASE_PARAMETERS | fit_overrides
    results = {}
    fit_embedding = None
    updated_graph = None
    for name, overrides in trials.items():
        model = UMAP(**fit_parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(INPUTS["training"].copy())
                current_fit = array_record(model.embedding_)
                for key, value in overrides.items():
                    setattr(model, key, value)
                result = model.update(INPUTS[batch].copy())
            except Exception as error:
                raise RuntimeError(f"{name}: {type(error).__name__}: {error}") from None
        assert result is None
        fit_embedding = current_fit if fit_embedding is None else fit_embedding
        assert current_fit == fit_embedding, name
        current_graph = graph_record(model.graph_)
        updated_graph = current_graph if updated_graph is None else updated_graph
        assert current_graph == updated_graph, name
        results[name] = {
            "update_overrides": overrides,
            "embedding": array_record(model.embedding_),
        }
    return {
        "update_input": batch,
        "fit_overrides": fit_overrides,
        "fit_embedding": fit_embedding,
        "updated_graph": updated_graph,
        "numpy_rng_model": seed_boundary(
            fit_parameters["transform_seed"], results["0"]["embedding"]["shape"], jitter
        ),
        "trials": results,
    }


def make_fixture() -> dict:
    packages = {name: version(name) for name in PACKAGE_VERSIONS}
    if platform.python_version() != PYTHON_VERSION or packages != PACKAGE_VERSIONS:
        raise RuntimeError("run with the locked reference environment")

    cases = {
        "single": (
            "near_one",
            {},
            {str(epochs): {"n_epochs": epochs} for epochs in [0, 1, 2, 30]},
            False,
        ),
        "pair": (
            "near_pair",
            {},
            {
                **{str(epochs): {"n_epochs": epochs} for epochs in [0, 1, 2, 3, 30, 200, 500]},
                "default": {"n_epochs": None},
                "2_attraction_only": {"n_epochs": 2, "repulsion_strength": 0.0},
                "2_half_repulsion": {"n_epochs": 2, "repulsion_strength": 0.5},
                "30_attraction_only": {"n_epochs": 30, "repulsion_strength": 0.0},
                "2_rate_one": {"n_epochs": 2, "negative_sample_rate": 1},
                "2_rate_five": {"n_epochs": 2, "negative_sample_rate": 5},
                "2_other_update_learning_rate": {"n_epochs": 2, "learning_rate": 0.5},
            },
            True,
        ),
        "default_fit": (
            "near_one",
            {"n_epochs": None},
            {"0": {"n_epochs": 0}, "default": {}, "2": {"n_epochs": 2}},
            False,
        ),
        "distinct_anchors": (
            "near_pair_distinct_anchors",
            {"n_epochs": 0, "learning_rate": 0.5, "random_state": 17, "transform_seed": 24},
            {str(epochs): {"n_epochs": epochs} for epochs in [0, 2, 30]},
            False,
        ),
        "pair_other_transform_seed": (
            "near_pair",
            {"transform_seed": 24},
            {str(epochs): {"n_epochs": epochs} for epochs in [0, 2, 30]},
            True,
        ),
    }
    return {
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": packages,
            "thread_environment": THREAD_LIMITS,
        },
        "fit_parameters": BASE_PARAMETERS,
        "cases": {name: capture(*case) for name, case in cases.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_fixture()
    if make_fixture() != fixture:
        raise SystemExit("a fresh pass produced different public checkpoints")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
