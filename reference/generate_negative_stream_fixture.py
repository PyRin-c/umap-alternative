"""Record public layout checkpoints for reconstructing three negative draws."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from generate_negative_sampling_probe import (
    PARAMETERS,
    assert_graph,
    fit,
    float32_bits,
    package_versions,
    words,
)


COPRIME_COUNTS = (5, 7, 11, 13, 17, 19, 23, 29, 31)
HELD_OUT_COUNTS = (12, 16, 21, 32, 35, 37)
EDGES = ((0, 1), (1, 0))
ATLAS = [[0, 0], [1, 4], [10, 10], [0, 8]] + [
    [i / 2, j / 2 + 0.125] for i in range(1, 8) for j in range(1, 7)
]
CALIBRATED_MT_WORDS = {
    0: 0,
    0x02FFDF78: 0xFFFFFFFC,
    0x12DD9BB1: 0xFFFFFFFD,
    0x92CDB9B7: 0x7FFFFFFF,
}
# Fields are name, caller seed, the two head x coordinates, and optional
# calibrated MT key entries to replace after consuming one raw MT19937 word.
CASES = (
    ("seed42_reference", 42, (0, 1), ()),
    ("seed7_reference", 7, (0, 1), ()),
    ("seed42_adjacent_x", 42, (0, 1 + 2**-23), ()),
    ("seed12345_mixed_x", 12345, (0.625, 3.1415927410125732), ()),
    ("seed20261121_mixed_x", 20261121, (4.125, 9.75), ()),
    ("seed42_high_accepted_word", 42, (0.3125, 6), (0x02FFDF78,)),
    ("seed42_repeated_midpoint_words", 42, (0, 2.5), (0x92CDB9B7,) * 3),
    ("seed42_mixed_rejection_boundary", 42, (0, 8.5), (0x12DD9BB1, 0x92CDB9B7, 0, 0x02FFDF78)),
)


def record_case(seed: int, head_x: tuple[float, float], injected_keys: tuple[int, ...]) -> dict:
    def make_rng() -> np.random.RandomState:
        rng = np.random.RandomState(seed)
        if injected_keys:
            words(rng, 1)
            name, key, cursor, gauss, cache = rng.get_state()
            key[cursor:cursor + len(injected_keys)] = injected_keys
            rng.set_state((name, key, cursor, gauss, cache))
        return rng

    counts = {}
    common_rng_state = None
    for size in (*COPRIME_COUNTS, *HELD_OUT_COUNTS):
        init = np.array(ATLAS[:size], dtype=np.float32)
        init[:2, 0] = head_x
        distances = np.full((size, size), 100, dtype=np.float32)
        np.fill_diagonal(distances, 0)
        distances[0, 1] = distances[1, 0] = 1
        coordinates = {}
        graph = None
        for epochs, rate in ((0, 2), (2, 2), (2, 3), (2, 4)):
            observed, record = fit(distances, init, 10, make_rng, epochs, rate)
            assert np.isfinite(observed[:2]).all() and np.isnan(observed[2:]).all()
            if epochs == 0:
                np.testing.assert_array_equal(observed[:2], init[:2])
                graph = record["graph"]
                assert_graph(graph, (EDGES,))
            assert graph == record["graph"]
            state = record["random_state"]
            del state["matches_candidate_randint_state"]
            if common_rng_state is None:
                common_rng_state = state
            assert state == common_rng_state
            coordinates[f"epochs_{epochs}_rate_{rate}"] = float32_bits(observed[:2])
        counts[size] = coordinates

    assert common_rng_state is not None
    if injected_keys:
        assert common_rng_state["matched_raw_uint32_words"][:len(injected_keys)] == [
            CALIBRATED_MT_WORDS[key] for key in injected_keys
        ]
    return {
        "caller_seed": seed,
        "head_x_normalized_float32": list(head_x),
        "injected_mt_keys_uint32": injected_keys,
        "caller_random_state": common_rng_state,
        "fit_active_coordinate_bits_uint32_by_vertex_count": counts,
    }


def make_fixture() -> dict:
    packages = package_versions()
    assert packages["umap-learn"] == "0.6.0"
    return {
        "environment": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": packages,
        },
        "parameters": {
            **PARAMETERS,
            "disconnection_distance": 10,
            "repulsion_strength": 1.0,
        },
        "input": {
            "explicit_init_prefix_atlas_float32": ATLAS[:max(*COPRIME_COUNTS, *HELD_OUT_COUNTS)],
            "pair_distance_float32": 1,
            "other_nondiagonal_distance_float32": 100,
        },
        "graph_observation": {
            "outgoing_edges_in_csr_order": EDGES,
            "weight_bits_uint32": [0x3F800000, 0x3F800000],
            "remaining_rows": "empty graph rows with NaN coordinates at every checkpoint",
        },
        "coprime_counts": COPRIME_COUNTS,
        "held_out_counts": HELD_OUT_COUNTS,
        "cases": {name: record_case(seed, head_x, key) for name, seed, head_x, key in CASES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    output.write_text(json.dumps(make_fixture(), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
