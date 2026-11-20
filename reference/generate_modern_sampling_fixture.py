"""Decode serial sample dispatch from public UMAP fits and RandomState calls."""

import argparse
import json
import os
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from umap import UMAP


def ring(n, chord=False):
    data = np.full((n, n), 8, dtype=np.float32)
    for i in range(n):
        data[i, i] = 0
        data[i, (i + 1) % n] = data[(i + 1) % n, i] = 1
    if chord:
        data[0, 2] = data[2, 0] = 0.75
    return data


class ObservedRandomState(np.random.RandomState):
    def __init__(self, seed, direction=None):
        super().__init__(seed)
        self.seed_draw = None
        self.epochs = []
        self.direction = direction

    def randint(self, *args, **kwargs):
        assert self.seed_draw is None and args == (-2147483647, 2147483646, 3) and not kwargs
        values = super().randint(*args, **kwargs)
        self.seed_draw = [*values.tolist(), self.get_state()[2]]
        return values

    def shuffle(self, x):
        super().shuffle(x)
        self.epochs.append({"permutation": x.tolist(), "shuffle_cursor": self.get_state()[2]})

    def randn(self, *args):
        values = super().randn(*args)
        if self.direction is not None:
            values[:] = self.direction
        self.epochs[-1]["normal_bits"] = values.view(np.uint64).tolist()
        self.epochs[-1]["normal_cursor"] = self.get_state()[2]
        return values

    def continuation(self):
        _, _, position, has_spare, spare = self.get_state()
        copy = np.random.RandomState()
        copy.set_state(self.get_state())
        return {
            "cursor": position,
            "gaussian_bits": int(np.float64(spare).view(np.uint64)) if has_spare else None,
            "next_words": copy.randint(0, 2**32, size=4, dtype=np.uint32).tolist(),
        }


def fit(data, init, n_epochs, seed, rate, exclude, cap, optimizer="momentum",
        curve=(1.0, 1.0), mix=1.0, neighbors=3, local=2.0, metric="precomputed",
        direction=None):
    state = ObservedRandomState(seed, direction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = UMAP(
            metric=metric, optimizer=optimizer, compatibility_layout=False,
            init=init.copy(), init_rescale="graph_edge", init_scale_factor=1.0,
            n_neighbors=neighbors, local_connectivity=local, set_op_mix_ratio=mix,
            n_components=init.shape[1], learning_rate=0.35, a=curve[0], b=curve[1],
            n_epochs=list(range(n_epochs + 1)), repulsion_strength=1.0,
            negative_sample_rate=rate, exclude_graph_neighbors=exclude,
            negative_selection_range=cap, n_jobs=1, random_state=state,
        ).fit(data.copy())
    return estimator, state


def recovered_forces(estimator):
    """Undo the public momentum update using observed float32 snapshots."""
    epochs = estimator.get_params()["n_epochs"][-1]
    alpha = estimator.get_params()["learning_rate"] / 8.0
    snapshots = estimator.embedding_list_
    step = np.zeros_like(snapshots[0], dtype=np.float64)
    for e in range(epochs):
        x = snapshots[max(e - 1, 0)].astype(np.float64)
        new_step = snapshots[e].astype(np.float64) - x
        carry = 0.5 - (1.0 - max(e - 1, 0) / epochs) ** 2 / 8.0
        yield x, (new_step - carry * step) / (alpha * (1.0 - e / epochs) ** 2)
        step = new_step


def force_basis(x, a, b):
    # Float32 coordinate subtraction is enough to decode target multiplicity
    # from these well-conditioned inputs. Avoid the diagonal singularity.
    delta = (x.astype(np.float32)[:, None] - x.astype(np.float32)[None, :]).astype(np.float64)
    distance_sq = (delta**2).sum(axis=2)
    np.fill_diagonal(distance_sq, 1.0)
    curve = 1.0 + a * distance_sq**b
    attractive = -2.0 * a * b * distance_sq**(b - 1.0) / curve
    limit = 2.0 * b * np.sqrt(distance_sq) / ((0.001 + distance_sq) * curve)
    repulsive = 2.0 * np.tanh(limit) / np.sqrt(distance_sq)
    return attractive[:, :, None] * delta, repulsive[:, :, None] * delta


def decode_counts(basis, force, maximum=None):
    n = len(force)
    count = np.zeros((n, n), dtype=np.int64)
    residual = 0.0
    for source in range(n):
        target = np.arange(n) != source
        matrix = basis[source, target].T
        assert np.linalg.cond(matrix) < 80.0
        fraction, *_ = np.linalg.lstsq(matrix, force[source], rcond=None)
        integer = np.rint(fraction)
        residual = max(residual, float(np.max(np.abs(fraction - integer))))
        assert np.all(integer >= 0)
        if maximum is not None:
            assert np.all(integer <= maximum)
        count[source, target] = integer
    assert residual < 0.02, residual
    return count, residual


def decode_positives(control, curve):
    positives = []
    residual = 0.0
    for x, force in recovered_forces(control):
        attractive, _ = force_basis(x, *curve)
        due, error = decode_counts(attractive, force, maximum=1)
        positives.append(due)
        residual = max(residual, error)
    return positives, residual


def decode_negatives(estimator, positives, curve):
    negatives = []
    residual = 0.0
    for (x, force), due in zip(recovered_forces(estimator), positives, strict=True):
        attractive, repulsive = force_basis(x, *curve)
        attraction = (attractive * due[:, :, None]).sum(axis=1)
        counts, error = decode_counts(repulsive, force - attraction)
        negatives.append(counts)
        residual = max(residual, error)
    return negatives, residual


def pairs(counts):
    """Encode (source, target) as source*n + target, repeating multiplicities."""
    return np.repeat(np.arange(counts.size), counts.ravel()).tolist()


def bits(snapshot):
    return np.ascontiguousarray(snapshot).view(np.uint32).ravel().tolist()


def record(name, data, init, n_epochs, seed, rate, exclude=False, cap=200_000,
           *, adam=False, **graph_options):
    options = dict(n_epochs=n_epochs, seed=seed, exclude=exclude, cap=cap, **graph_options)
    curve = graph_options.get("curve", (1.0, 1.0))
    control, control_state = fit(data, init, rate=0, **options)
    observed, state = fit(data, init, rate=rate, **options) if rate else (control, control_state)
    repeated, repeat_state = fit(data, init, rate=rate, **options)
    snapshots = [bits(x) for x in observed.embedding_list_]
    assert snapshots == [bits(x) for x in repeated.embedding_list_]
    assert state.epochs == repeat_state.epochs and state.continuation() == repeat_state.continuation()
    assert bits(control.embedding_list_[0]) == snapshots[0]
    positives, positive_error = decode_positives(control, curve)
    negatives, negative_error = decode_negatives(observed, positives, curve)
    graph = observed.graph_
    assert (graph != control.graph_).nnz == 0
    assert len(state.epochs) == n_epochs
    assert np.count_nonzero(graph.data) > 0
    assert connected_components(graph, return_labels=False) == 1
    result = {
        "name": name,
        "seed": seed,
        "n_rows": len(init),
        "n_components": init.shape[1],
        "n_epochs": n_epochs,
        "negative_sample_rate": rate,
        "exclude_graph_neighbors": exclude,
        "negative_selection_range": cap,
        "curve": curve,
        "graph": {
            "indptr": graph.indptr.tolist(), "indices": graph.indices.tolist(),
            "weights_bits": bits(graph.data),
        },
        "momentum_bits": snapshots,
        "positive": list(map(pairs, positives)),
        "negative": list(map(pairs, negatives)),
        "decode_error": max(positive_error, negative_error),
        "seed_draw": state.seed_draw,
        "random_epochs": state.epochs,
        "continuation": state.continuation(),
    }
    if adam:
        other, adam_state = fit(data, init, rate=rate, optimizer="adam", **options)
        again, again_state = fit(data, init, rate=rate, optimizer="adam", **options)
        result["adam_bits"] = [bits(x) for x in other.embedding_list_]
        assert result["adam_bits"] == [bits(x) for x in again.embedding_list_]
        assert adam_state.seed_draw == state.seed_draw
        assert adam_state.epochs == again_state.epochs
        assert adam_state.continuation() == again_state.continuation()
        result["adam_random_epochs"] = adam_state.epochs
        result["adam_continuation"] = adam_state.continuation()
    return result


def projection_tie_probe():
    """Reproduce the unresolved ordering of equal binary64 projection keys."""
    n = 24
    init = np.eye(n, dtype=np.float32)
    direction = np.arange(n) % 4.0
    options = dict(n_epochs=2, seed=42, exclude=False, cap=1, direction=direction)
    control, _ = fit(ring(n), init, rate=0, **options)
    sampled, _ = fit(ring(n), init, rate=5, **options)
    positive, _ = decode_positives(control, (1.0, 1.0))
    negative, _ = decode_negatives(sampled, positive, (1.0, 1.0))
    project = sampled.embedding_list_[0].astype(np.float64) @ (direction / np.linalg.norm(direction))

    def mismatch(order):
        predicted = np.zeros((n, n), dtype=int)
        for edge, (head, _) in enumerate(zip(*sampled.graph_.nonzero(), strict=True)):
            for multiple in (2, 3):
                target = order[edge * multiple % n]
                if target != head:
                    predicted[head, target] += 1
        return int(np.count_nonzero(predicted != negative[1]))

    result = {"n_rows": n, "direction": direction.tolist(),
              "stable_mismatches": mismatch(project.argsort(kind="stable")),
              "default_mismatches": mismatch(project.argsort())}
    assert result["stable_mismatches"] > 0
    assert result["default_mismatches"] == 0
    return result


def make_fixture():
    cases = []

    def add(name, n, epochs, seed, rate, exclude=False, cap=200_000, *, chord=False,
            dense=None, tilted=False, **options):
        data = ring(n, chord) if dense is None else np.asarray(dense, dtype=np.float32)
        init = np.eye(n, dtype=np.float32)
        if tilted:
            init = np.column_stack((np.sin(np.arange(n, dtype=np.float32) * 1.75), init))
        cases.append(record(name, data, init, epochs, seed, rate, exclude, cap, **options))

    add("zero-budget", 6, 0, 42, 5, adam=True)
    add("attraction-only", 6, 7, 42, 0, adam=True)
    add("first-eligible", 6, 2, 42, 5, adam=True)
    add("delayed-rate-one", 6, 7, 42, 1, True, adam=True)
    add("repeated-draws", 6, 9, 42, 5, adam=True)
    add("excluded-neighbors", 7, 9, 7, 5, True, adam=True, tilted=True)
    add("all-neighbors-excluded", 6, 7, 42, 5, True, neighbors=6, local=5.0, adam=True)
    add("one-eligible-neighbor", 4, 7, 169, 5, True, adam=True)
    add("positive-f32-rounding", 8, 7, 169, 5, True, chord=True, mix=0.2)
    add("negative-f64-rounding", 8, 12, 169, 5, True, chord=True, mix=1.0 / 3.0, adam=True)
    for epochs, cutoff in ((10, np.float32(1 / 500)), (11, np.float32(1 / 11))):
        for tag, weight in (("below", np.nextafter(cutoff, np.float32(0))),
                            ("equal", cutoff), ("above", np.nextafter(cutoff, np.float32(1)))):
            add(f"cutoff-{epochs}-{tag}", 8, epochs, 169, 5, True, chord=True, mix=float(weight))
    add("holdout-weighted-features", 7, 9, 229, 5, True, tilted=True,
        dense=[[0, 0], [1.2, 0.13], [0.35, 2.3], [3.45, 0.78], [2.8, 3.1], [2.1, 2.5], [0.1, 1.9]],
        metric="euclidean", local=1.0, neighbors=4, curve=(1.7, 0.9), adam=True)
    add("holdout-nonunit-maximum", 7, 13, 11, 5, tilted=True, chord=True,
        local=0.0, neighbors=4, mix=0.7, adam=True)
    add("hard-at-cap", 6, 5, 42, 5, cap=6, adam=True)
    add("hard-below-cap", 6, 7, 42, 5, cap=5, adam=True)
    add("hard-rate-zero", 6, 5, 42, 0, cap=5)
    add("hard-holdout-excluded", 9, 7, 81, 5, True, 2, tilted=True, adam=True)
    add("hard-holdout-larger", 17, 7, 229, 5, False, 1, tilted=True, adam=True)
    return {
        "operation": "UMAP.fit",
        "environment": {
            "platform": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {package: version(package) for package in (
                "umap-learn", "numpy", "scipy", "numba", "llvmlite", "scikit-learn"
            )},
            "threads": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"
            )},
        },
        "learning_rate": 0.35,
        "projection_tie": projection_tie_probe(),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = make_fixture()
    # Keep float32 checkpoint arrays on one line per case.
    cases = fixture.pop("cases")
    header = json.dumps(fixture, indent=2)[:-2]
    encoded = ",\n".join("    " + json.dumps(case, separators=(",", ":")) for case in cases)
    args.output.write_text(f'{header},\n  "cases": [\n{encoded}\n  ]\n}}\n', encoding="utf-8", newline="\n")
