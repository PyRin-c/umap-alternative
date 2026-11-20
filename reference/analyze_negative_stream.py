"""Decode a public negative-stream fixture, then compare independent RNG models."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from generate_negative_sampling_probe import TOLERANCE, attractive, repulsive


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
# L'Ecuyer (1996), three-component QuickTaus parameters. Each row gives
# feedback shift, stride, right shift, and the number of low bits cleared.
TAUS88 = ((13, 12, 19, 1), (2, 4, 25, 3), (3, 17, 11, 4))


def int32(value: int) -> int:
    return ((value + (1 << 31)) & MASK32) - (1 << 31)


def fit_coordinates(bits: list[list[int]]) -> np.ndarray:
    return np.array(bits, dtype=np.uint32).view(np.float32)


def decode_indices(init: np.ndarray, checkpoints: dict, alpha: float) -> tuple[list, float]:
    """Enumerate the new pair of indices at each rate with a measured prefix."""
    size = len(init)
    candidates = np.indices((size, size)).reshape(2, -1)
    choices: list[list[int]] = [[], []]
    max_error = 0.0
    for rate in (2, 3, 4):
        target = fit_coordinates(checkpoints[f"epochs_2_rate_{rate}"])
        layouts = np.repeat(init[None, ...], size**2, axis=0)
        for head, draws in enumerate(choices):
            attractive(layouts, head, 1 - head, alpha)
            for earlier in draws:
                repulsive(layouts, head, np.full(size**2, earlier), alpha)
            repulsive(layouts, head, candidates[head], alpha)
        errors = np.max(np.abs(layouts[:, :2] - target), axis=(1, 2))
        matches = np.flatnonzero(errors <= TOLERANCE)
        assert len(matches) == 1, (size, rate, int(len(matches)), float(errors.min()))
        for head in (0, 1):
            choices[head].append(int(candidates[head, matches[0]]))
        max_error = max(max_error, float(errors[matches[0]]))
    return choices, max_error


def signed_representative(counts: list[int], indices: list[int]) -> int:
    residue, modulus = 0, 1
    for size, index in zip(counts, indices, strict=True):
        residue += modulus * (((index - residue) * pow(modulus, -1, size)) % size)
        modulus *= size
    assert modulus > 2**32
    word = residue if residue <= modulus // 2 else residue - modulus
    assert -(1 << 31) <= word < (1 << 31)
    return word


def reconstruct(fixture: dict) -> dict:
    counts = fixture["coprime_counts"]
    holdouts = fixture["held_out_counts"]
    alpha = fixture["parameters"]["learning_rate"]
    cases = {}
    for name, case in fixture["cases"].items():
        head_x = case["head_x_normalized_float32"]
        indices = {}
        max_error = 0.0
        for size in (*counts, *holdouts):
            init = np.array(fixture["input"]["explicit_init_prefix_atlas_float32"][:size], dtype=np.float32)
            init[:2, 0] = head_x
            checkpoints = case["fit_active_coordinate_bits_uint32_by_vertex_count"][str(size)]
            assert np.array_equal(init[:2], fit_coordinates(checkpoints["epochs_0_rate_2"]))
            indices[size], error = decode_indices(init, checkpoints, alpha)
            max_error = max(max_error, error)

        signed_words = [
            [signed_representative(counts, [indices[size][head][draw] for size in counts])
             for draw in range(3)]
            for head in (0, 1)
        ]
        for size in holdouts:
            assert indices[size] == [[word % size for word in stream] for stream in signed_words]
        cases[name] = {
            "head_x_normalized_float32": head_x,
            "caller_raw_uint32_words": case["caller_random_state"]["matched_raw_uint32_words"],
            "signed_words_by_head": signed_words,
            "negative_indices_by_vertex_count": indices,
            "max_absolute_coordinate_replay_error": max_error,
        }
    return cases


@dataclass(frozen=True)
class Candidate:
    coordinate: str = "binary64"
    combine: str = "add"
    feedback_width: str = "signed64"
    stride_width: int = 32
    saved_width: int = 64
    seed_offset: int = -(1 << 31) + 1


def coordinate_token(x: float, encoding: str) -> int:
    if encoding == "binary64":
        return int.from_bytes(struct.pack("<d", x), "little", signed=True)
    if encoding == "binary32":
        return int.from_bytes(struct.pack("<f", x), "little", signed=True)
    if encoding == "integer":
        return int(x)
    assert encoding == "none"
    return 0


def candidate_words(candidate: Candidate, raw: list[int], x: float) -> list[int]:
    # The earlier rejection probe is the separate evidence for this cutoff.
    accepted = [word for word in raw if word < 0xFFFFFFFD]
    assert len(accepted) == 3
    token = coordinate_token(x, candidate.coordinate)
    lanes = [word + candidate.seed_offset for word in accepted]
    if candidate.combine == "add":
        lanes = [lane + token for lane in lanes]
    else:
        assert candidate.combine == "xor"
        lanes = [lane ^ token for lane in lanes]

    output = []
    for _ in range(3):
        following = []
        for lane, (feedback, stride, right, clear) in zip(lanes, TAUS88, strict=True):
            low_mask = MASK32 ^ ((1 << clear) - 1)
            advanced = ((lane & low_mask) << stride) & ((1 << candidate.stride_width) - 1)
            mixed = ((lane << feedback) & MASK32) ^ lane
            if candidate.feedback_width == "unsigned32":
                mixed &= MASK32
            elif candidate.feedback_width == "unsigned64":
                mixed &= MASK64
            value = advanced ^ (mixed >> right)
            following.append(int32(value) if candidate.saved_width == 32 else value)
        lanes = following
        output.append(int32(lanes[0] ^ lanes[1] ^ lanes[2]))
    return output


def score_candidates(observations: dict, held_out_counts: list[int]) -> dict:
    wide = Candidate()
    models = {
        "signed_wide_binary64_add": wide,
        "canonical_unsigned32_feedback": replace(wide, feedback_width="unsigned32", saved_width=32),
        "unsigned64_feedback": replace(wide, feedback_width="unsigned64"),
        "unmasked_32bit_stride": replace(wide, stride_width=64),
        "truncate_saved_lanes_to_int32": replace(wide, saved_width=32),
        "binary32_add": replace(wide, coordinate="binary32"),
        "integer_coordinate_add": replace(wide, coordinate="integer"),
        "no_coordinate": replace(wide, coordinate="none"),
        "binary64_xor": replace(wide, combine="xor"),
        "seed_offset_int32_min": replace(wide, seed_offset=-(1 << 31)),
        "seed_offset_int32_min_plus_two": replace(wide, seed_offset=-(1 << 31) + 2),
        "seed_offset_zero": replace(wide, seed_offset=0),
    }

    candidate_scores = {}
    for label, model in models.items():
        matched_words_by_draw = [0, 0, 0]
        held_out_indices = 0
        for case in observations.values():
            for head, x in enumerate(case["head_x_normalized_float32"]):
                measured = case["signed_words_by_head"][head]
                predicted = candidate_words(model, case["caller_raw_uint32_words"], x)
                for draw, (expected, observed) in enumerate(zip(predicted, measured, strict=True)):
                    matched_words_by_draw[draw] += expected == observed
                for size in held_out_counts:
                    indices = case["negative_indices_by_vertex_count"][size][head]
                    held_out_indices += sum(a % size == b for a, b in zip(predicted, indices, strict=True))
        candidate_scores[label] = {
            "matched_words_by_draw": matched_words_by_draw,
            "total_words_per_draw": len(observations) * 2,
            "matched_held_out_indices": held_out_indices,
            "total_held_out_indices": len(observations) * 2 * 3 * len(held_out_counts),
        }
    return candidate_scores


def compare_previous_fixture(fixture: dict) -> dict:
    """Compare the candidate with the earlier varied-graph and epoch probes."""
    matched, total = 0, 0

    def compare(case: dict, checkpoint: dict, decode: str, draws_already_taken: int = 0) -> None:
        nonlocal matched, total
        normalized = fit_coordinates(case.get(
            "all_rows_normalized_init_bits_uint32", case["zero_epoch"]["embedding_bits_uint32"]
        ))
        raw = checkpoint["random_state"]["matched_raw_uint32_words"]
        if case["init"] == "random":
            raw = raw[2 * normalized.size:]
        cursors = {}
        for edge, indices in checkpoint[decode]["targets_by_edge"].items():
            head = int(edge.split("->")[0])
            start = cursors.get(head, draws_already_taken)
            stop = start + len(indices)
            predicted = candidate_words(Candidate(), raw, float(normalized[head, 0]))
            matched += sum(word % len(normalized) == index for word, index in
                           zip(predicted[start:stop], indices, strict=True))
            total += len(indices)
            cursors[head] = stop

    for case in fixture["isolated_pair_scenarios"].values():
        for checkpoint in case["epoch_two_by_rate"].values():
            compare(case, checkpoint, "interleaved_decode")
        for rate, checkpoint in case.get("epoch_three_by_rate", {}).items():
            compare(case, checkpoint, "last_epoch_decode", draws_already_taken=int(rate) - 1)
    for label in ("both_edges", "later_edge_only"):
        case = fixture["two_edges_per_head"][label]
        compare(case, case["epoch_two_by_rate"]["2"], "interleaved_decode")
    return {"matched_indices": matched, "total_indices": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--previous-fixture", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    observations = reconstruct(fixture)
    result = {
        "observations": observations,
        "candidate_scores": score_candidates(observations, fixture["held_out_counts"]),
    }
    if args.previous_fixture is not None:
        previous = json.loads(args.previous_fixture.read_text(encoding="utf-8"))
        result["previous_fixture_score"] = compare_previous_fixture(previous)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
