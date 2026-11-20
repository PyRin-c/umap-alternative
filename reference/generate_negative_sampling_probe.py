"""Infer early negative targets from public UMAP 0.6.0 layout checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
from umap import UMAP

from generate_isolated_edge_arithmetic import float32_bits, graph_record, package_versions


INIT = np.array([[0, 0], [1, 4], [2.5, 1], [5, 8], [6, 3], [10, 10]], dtype=np.float32)
PAIR_DISTANCES = (3, 2, 1)
LEARNING_RATE = 0.1
TOLERANCE = 2e-6
PARAMETERS = {
    "n_components": 2,
    "n_neighbors": 2,
    "metric": "precomputed",
    "a": 1.0,
    "b": 1.0,
    "learning_rate": LEARNING_RATE,
    "n_jobs": 1,
    "compatibility_layout": True,
}

type Edge = tuple[int, int]
type Block = tuple[Edge, ...]
type Targets = dict[Edge, tuple[int, ...]]
type RNGFactory = Callable[[], np.random.RandomState]
type Match = tuple[Targets, float]


def make_distances() -> np.ndarray:
    distances = np.full((len(INIT), len(INIT)), 100, dtype=np.float32)
    np.fill_diagonal(distances, 0)
    for head, distance in zip(range(0, len(INIT), 2), PAIR_DISTANCES, strict=True):
        distances[head, head + 1] = distances[head + 1, head] = distance
    return distances


def words(rng: np.random.RandomState, count: int) -> np.ndarray:
    return rng.randint(0, 2**32, size=count, dtype=np.uint32)


def same_rng_state(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def fit(
    distances: np.ndarray,
    init: str | np.ndarray,
    threshold: float,
    make_rng: RNGFactory,
    epochs: int,
    rate: int,
    gamma: float = 1.0,
) -> tuple[np.ndarray, dict]:
    def run() -> tuple[np.ndarray, dict, tuple, tuple]:
        rng = make_rng()
        before = rng.get_state()
        model = UMAP(
            **PARAMETERS,
            init=init if isinstance(init, str) else init.copy(),
            n_epochs=epochs,
            negative_sample_rate=rate,
            repulsion_strength=gamma,
            disconnection_distance=threshold,
            random_state=rng,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            embedding = model.fit_transform(distances.copy())
        assert embedding.dtype == np.float32
        assert embedding.shape == (len(distances), PARAMETERS["n_components"])
        return embedding, graph_record(model.graph_), before, rng.get_state()

    embedding, graph, before, after = run()
    repeated, repeated_graph, repeated_before, repeated_after = run()
    bits = float32_bits(embedding)
    assert bits == float32_bits(repeated) and graph == repeated_graph
    assert same_rng_state(before, repeated_before) and same_rng_state(after, repeated_after)

    comparison = np.random.RandomState(0)
    comparison.set_state(before)
    consumed = []
    while not same_rng_state(comparison.get_state(), after):
        assert len(consumed) < 100
        consumed.append(int(words(comparison, 1)[0]))

    candidate = np.random.RandomState(0)
    candidate.set_state(before)
    if isinstance(init, str):
        candidate.uniform(-10, 10, size=embedding.shape)
    candidate.randint(-2_147_483_647, 2_147_483_646, size=3)
    return embedding, {
        "embedding_bits_uint32": bits,
        "graph": graph,
        "random_state": {
            "mt19937_cursor_before": before[2],
            "mt19937_cursor_after": after[2],
            "matched_raw_uint32_words": consumed,
            "matches_candidate_randint_state": same_rng_state(candidate.get_state(), after),
        },
    }


def attractive(layouts: np.ndarray, head: int, tail: int, alpha: float) -> None:
    delta = layouts[:, head].astype(np.float64) - layouts[:, tail]
    squared = np.sum(delta * delta, axis=1, keepdims=True)
    step = np.clip(-2 * delta / (1 + squared), -4, 4) * alpha
    layouts[:, head] += step
    layouts[:, tail] -= step


def repulsive(layouts: np.ndarray, head: int, targets: np.ndarray, alpha: float) -> None:
    delta = layouts[:, head].astype(np.float64) - layouts[np.arange(len(layouts)), targets]
    squared = np.sum(delta * delta, axis=1, keepdims=True)
    gradient = 2 * delta / ((0.001 + squared) * (1 + squared))
    layouts[:, head] += np.clip(gradient, -4, 4) * alpha


def replay_matches(
    init: np.ndarray,
    observed: np.ndarray,
    blocks: tuple[Block, ...],
    draws: int,
    *,
    alpha: float = LEARNING_RATE,
    order: str = "interleaved",
) -> list[Match]:
    """Enumerate ordered targets and reject by each component's final coordinates."""
    if order == "reverse":
        blocks = tuple(tuple(reversed(edges)) for edges in reversed(blocks))
    start = init[None, ...].copy()
    if order == "positives_first":
        for head, tail in itertools.chain.from_iterable(blocks):
            attractive(start, head, tail, alpha)
    branches = [(start[0], {}, 0.0)]
    for edges in blocks:
        options = np.array(list(itertools.product(range(len(init)), repeat=len(edges) * draws)))
        vertices = sorted({vertex for edge in edges for vertex in edge})
        surviving = []
        for layout, earlier, earlier_error in branches:
            layouts = np.repeat(layout[None, ...], len(options), axis=0)
            for position, (head, tail) in enumerate(edges):
                if order != "positives_first":
                    attractive(layouts, head, tail, alpha)
                for draw in range(draws):
                    repulsive(layouts, head, options[:, position * draws + draw], alpha)
            errors = np.max(np.abs(layouts[:, vertices] - observed[vertices]), axis=(1, 2))
            for index in np.flatnonzero(errors <= TOLERANCE):
                rows = options[index].reshape(len(edges), draws).tolist()
                choices = {edge: tuple(row) for edge, row in zip(edges, rows, strict=True)}
                surviving.append(
                    (layouts[index], earlier | choices, max(earlier_error, float(errors[index])))
                )
        branches = surviving
    return [(choices, error) for _, choices, error in branches]


def with_prefix(matches: list[Match], prefix: Targets) -> list[Match]:
    return [
        (choices, error)
        for choices, error in matches
        if all(choices[edge][:len(row)] == row for edge, row in prefix.items())
    ]


def decode(matches: list[Match], prefix: Targets | None = None) -> tuple[dict, Targets]:
    result = {"unconstrained_match_count": len(matches)}
    if prefix is not None:
        matches = with_prefix(matches, prefix)
        result["prefix_match_count"] = len(matches)
    assert len(matches) == 1
    choices, error = matches[0]
    result["targets_by_edge"] = {f"{head}->{tail}": row for (head, tail), row in choices.items()}
    result["max_absolute_coordinate_error"] = error
    return result, choices


def assert_graph(graph: dict, blocks: tuple[Block, ...]) -> None:
    edges = tuple(itertools.chain.from_iterable(blocks))
    assert graph["indices"] == [tail for _, tail in edges]
    assert graph["indptr"] == [
        sum(head < row for head, _ in edges) for row in range(graph["shape"][0] + 1)
    ]
    assert graph["data_bits_uint32"] == [0x3F800000] * len(edges)


def checkpoints(
    distances: np.ndarray,
    init: str | np.ndarray,
    normalized: np.ndarray,
    threshold: float,
    blocks: tuple[Block, ...],
    make_rng: RNGFactory,
    rates: tuple[int, ...],
    *,
    controls: bool = False,
    third_epoch: bool = False,
) -> dict:
    zero, zero_record = fit(distances, init, threshold, make_rng, 0, 2)
    heads = sorted({head for block in blocks for head, _ in block})
    np.testing.assert_array_equal(zero[heads], normalized[heads])
    assert np.isnan(zero[list(set(range(len(INIT))) - set(heads))]).all()
    graph = zero_record.pop("graph")
    assert_graph(graph, blocks)
    rng_state = zero_record["random_state"]

    result = {
        "init": init if isinstance(init, str) else init.tolist(),
        "disconnection_distance": threshold,
        "graph": graph,
        "zero_epoch": zero_record,
        "epoch_two_by_rate": {},
    }
    if len(heads) < len(INIT):
        result["all_rows_normalized_init_bits_uint32"] = float32_bits(normalized)

    previous = None
    for rate in rates:
        observed, record = fit(distances, init, threshold, make_rng, 2, rate)
        assert record.pop("graph") == graph and record["random_state"] == rng_state
        matches = replay_matches(normalized, observed, blocks, draws=rate - 1)
        record["interleaved_decode"], previous = decode(matches, previous)
        if controls:
            record["alternative_match_counts"] = {
                order: len(replay_matches(normalized, observed, blocks, rate - 1, order=order))
                for order in ("reverse", "positives_first")
            }
        result["epoch_two_by_rate"][rate] = record

    if controls:
        one, record = fit(distances, init, threshold, make_rng, 1, 2)
        assert record.pop("graph") == graph and record["random_state"] == rng_state
        assert float32_bits(one) == float32_bits(zero)
        result["one_epoch"] = record

        no_repulsion, record = fit(distances, init, threshold, make_rng, 2, 2, gamma=0.0)
        assert record.pop("graph") == graph and record["random_state"] == rng_state
        matches = replay_matches(normalized, no_repulsion, blocks, draws=0)
        assert len(matches) == 1
        record["max_absolute_coordinate_error"] = matches[0][1]
        result["two_epochs_no_repulsion"] = record

    if third_epoch:
        assert previous is not None and rates == (2, 3) and len(heads) == len(INIT)
        first_draw = {edge: row[:1] for edge, row in previous.items()}
        second_draw = {edge: row[1:] for edge, row in previous.items()}
        result["epoch_three_by_rate"] = {}
        for rate in (1, 2):
            two, two_record = fit(distances, init, threshold, make_rng, 2, rate)
            three, record = fit(distances, init, threshold, make_rng, 3, rate)
            assert two_record.pop("graph") == record.pop("graph") == graph
            assert two_record["random_state"] == record["random_state"] == rng_state
            alpha = LEARNING_RATE * (1 - 1 / 3)
            matches = replay_matches(two, three, blocks, draws=rate, alpha=alpha)
            if rate == 1:
                assert len(matches) == 1 and matches[0][0] == first_draw
            else:
                record["epoch_restart_match_count"] = len(with_prefix(matches, first_draw))
            record["last_epoch_decode"], _ = decode(matches, second_draw if rate == 2 else None)
            result["epoch_three_by_rate"][rate] = {"epoch_two_baseline": two_record, **record}
    return result


def probe_pairs(
    init: str | np.ndarray,
    seed: int,
    threshold: float = 10,
    rates: tuple[int, ...] = (2, 3),
    *,
    preadvance: bool = False,
    controls: bool = False,
    third_epoch: bool = False,
) -> dict:
    def make_rng() -> np.random.RandomState:
        rng = np.random.RandomState(seed)
        if preadvance:
            rng.uniform(-10, 10, size=INIT.shape)
        return rng

    distances = make_distances()
    normalized, _ = fit(distances, init, 10, make_rng, 0, 2)
    blocks = tuple(
        ((head, head + 1), (head + 1, head))
        for head, distance in zip(range(0, len(INIT), 2), PAIR_DISTANCES, strict=True)
        if distance < threshold
    )
    return {
        "seed": seed,
        "random_state_preadvanced_by_uniform_init": preadvance,
        **checkpoints(
            distances, init, normalized, threshold, blocks, make_rng, rates,
            controls=controls, third_epoch=third_epoch,
        ),
    }


def probe_two_edges_per_head() -> dict:
    distances = np.full_like(make_distances(), 100)
    np.fill_diagonal(distances, 0)
    distances[0, 2] = distances[2, 0] = 1.5
    distances[1, 2] = distances[2, 1] = 1
    distances[4, 5] = distances[5, 4] = 1
    return {
        "seed": 42,
        "distance_matrix_float32": distances.tolist(),
        "both_edges": checkpoints(
            distances, INIT, INIT, 10,
            (((0, 2), (1, 2), (2, 0), (2, 1)), ((4, 5), (5, 4))),
            lambda: np.random.RandomState(42), (2,),
        ),
        "later_edge_only": checkpoints(
            distances, INIT, INIT, 1.25,
            (((1, 2), (2, 1)), ((4, 5), (5, 4))),
            lambda: np.random.RandomState(42), (2,),
        ),
    }


def probe_seed_rejection() -> dict:
    # Calibrated MT19937 key entries. Check the resulting raw words with
    # NumPy itself before passing each state through the public estimator.
    boundary_keys = {
        0xFFFFFFFC: 0x02FFDF78,
        0xFFFFFFFD: 0x12DD9BB1,
        0xFFFFFFFE: 0x02FFDF7A,
        0xFFFFFFFF: 0x12DD9BB3,
    }
    measured = {}
    for raw, injected_key in boundary_keys.items():
        def make_rng() -> np.random.RandomState:
            rng = np.random.RandomState(42)
            words(rng, 1)
            name, key, cursor, gauss, cache = rng.get_state()
            key[cursor] = injected_key
            rng.set_state((name, key, cursor, gauss, cache))
            return rng

        def skip_injected_word() -> np.random.RandomState:
            rng = make_rng()
            assert int(words(rng, 1)[0]) == raw
            return rng

        _, natural = fit(make_distances(), INIT, 10, make_rng, 2, 3)
        _, skipped = fit(make_distances(), INIT, 10, skip_injected_word, 2, 3)
        measured[hex(raw)] = {
            "injected_key_uint32": injected_key,
            "injected_word_observed": natural,
            "skipped_word_control": skipped,
            "embedding_bits_equal": natural["embedding_bits_uint32"] == skipped["embedding_bits_uint32"],
        }
    return measured


def probe_index_reduction() -> dict:
    """Measure the same heads at coprime vertex counts and test held-out counts."""
    atlas = np.array(
        [[0, 0], [1, 4], [10, 10]]
        + [[i / 2, j / 2 + 0.125] for i in range(1, 8) for j in range(1, 7)],
        dtype=np.float32,
    )
    coprime_sizes = (5, 7, 11, 13, 17, 19, 23, 29, 31)
    held_out_sizes = (6, 8, 10, 34)
    edges = ((0, 1), (1, 0))
    results = {}
    for seed in (42, 7):
        def make_rng() -> np.random.RandomState:
            return np.random.RandomState(seed)

        records = {}
        for size in (*coprime_sizes, *held_out_sizes):
            distances = np.full((size, size), 100, dtype=np.float32)
            np.fill_diagonal(distances, 0)
            distances[0, 1] = distances[1, 0] = 1
            init = atlas[:size]
            zero, zero_record = fit(distances, init, 10, make_rng, 0, 2)
            observed, record = fit(distances, init, 10, make_rng, 2, 2)
            assert zero_record["graph"] == record["graph"]
            assert_graph(record["graph"], (edges,))
            assert zero_record["random_state"] == record["random_state"]
            assert np.isnan(zero[2:]).all() and np.isnan(observed[2:]).all()
            np.testing.assert_array_equal(zero[:2], init[:2])
            matches = replay_matches(init, observed, (edges,), draws=1)
            decoded, targets = decode(matches)
            records[size] = {
                "negative_indices_in_head_order": [targets[edge][0] for edge in edges],
                "active_embedding_bits_uint32": float32_bits(observed[:2]),
                "max_absolute_coordinate_error": decoded["max_absolute_coordinate_error"],
            }

        # Extend the congruence a mod m with each observed index b mod size.
        modulus, residues = 1, [0, 0]
        for size in coprime_sizes:
            indices = records[size]["negative_indices_in_head_order"]
            inverse = pow(modulus, -1, size)
            residues = [
                a + modulus * (((b - a) * inverse) % size)
                for a, b in zip(residues, indices, strict=True)
            ]
            modulus *= size
        signed = [a if a < modulus // 2 else a - modulus for a in residues]
        assert modulus > 2**32 and all(-2**31 <= a < 2**31 for a in signed)
        for size in held_out_sizes:
            assert records[size]["negative_indices_in_head_order"] == [a % size for a in signed]
        results[seed] = {
            "signed_int32_representatives_in_head_order": signed,
            "observations_by_vertex_count": records,
        }
    return {
        "init_prefix_atlas_float32": atlas[:max(*coprime_sizes, *held_out_sizes)].tolist(),
        "coprime_counts": coprime_sizes,
        "coprime_counts_product": modulus,
        "held_out_counts": held_out_sizes,
        "seeds": results,
    }


def make_fixture() -> dict:
    packages = package_versions()
    assert packages["umap-learn"] == "0.6.0"
    varied_x = INIT.copy()
    varied_x[4, 0] = 1.75
    varied_y = INIT.copy()
    varied_y[4, 1] = 4.75
    matching_x = INIT.copy()
    matching_x[4, 0] = matching_x[5, 0]
    affine = INIT * 2 + np.array([5, -6], dtype=np.float32)
    random_values = np.random.RandomState(42).uniform(-10, 10, INIT.shape).astype(np.float32)
    scenarios = {
        "base": probe_pairs(INIT, 42, controls=True, third_epoch=True),
        "last_pair_only": probe_pairs(INIT, 42, 1.5, controls=True),
        "seed_7": probe_pairs(INIT, 7),
        "seed_7_last_pair_only": probe_pairs(INIT, 7, 1.5),
        "varied_head_x": probe_pairs(varied_x, 42),
        "varied_head_y": probe_pairs(varied_y, 42),
        "matching_x_for_heads_4_and_5": probe_pairs(matching_x, 42),
        "affine_equivalent_init": probe_pairs(affine, 42),
        "random_init": probe_pairs("random", 42, rates=(2,)),
        "explicit_random_values_advanced": probe_pairs(random_values, 42, rates=(2,), preadvance=True),
        "explicit_random_values_unadvanced": probe_pairs(random_values, 42, rates=(2,)),
    }
    return {
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": packages,
        },
        "parameters": PARAMETERS,
        "distance_matrix_float32": make_distances().tolist(),
        "replay_absolute_tolerance": TOLERANCE,
        "isolated_pair_scenarios": scenarios,
        "two_edges_per_head": probe_two_edges_per_head(),
        "seed_rejection_boundary": probe_seed_rejection(),
        "index_reduction": probe_index_reduction(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.write_text(json.dumps(make_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
