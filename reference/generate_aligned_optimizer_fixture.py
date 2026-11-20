"""Capture public AlignedUMAP checkpoints that isolate joint and append updates."""

import argparse
import json
import os
import platform
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch


def capture():
    import numpy as np
    from scipy.sparse import csr_matrix
    from umap import AlignedUMAP

    def bits(x):
        return np.asarray(x, dtype=np.float32).view(np.uint32).tolist()

    settings = dict(
        n_neighbors=3, n_components=2, init="random", compatibility_layout=True, n_epochs=0,
        random_state=42, alignment_regularisation=0.0, local_connectivity=0.0,
    )
    x = np.random.RandomState(83).normal(size=(6, 3)).astype(np.float32)
    links = dict(enumerate(range(len(x))))
    fixed = np.array([[1, 2], [3, -4], [-2, -1], [0.25, -2], [-8, 1], [1, 4]], dtype=np.float32)
    cases = {}

    # Only the symmetric 2↔4 edge is due at two epochs. The remaining
    # rows supply known candidate coordinates for negative sampling.
    for seed in (0, 1, 37, 42, 83):
        for epochs in (2, 8):
            for scale in (0.125, 1.0, 16.0):
                config = settings | {"transform_seed": seed}
                m = AlignedUMAP(**config).fit([x, x], relations=[links])
                m.embeddings_[-1][:] = fixed * scale
                m.update(x, relations=links, n_epochs=epochs)
                cases[f"isolated-{seed}-{epochs}-{scale}"] = dict(
                    epochs=epochs, scale=scale, config=config, layout=bits(m.embeddings_[-1]),
                )

    # Changing one old graph row separates neighborhood overlap and graph
    # membership size. Moving an older anchor tests temporal decay.
    for neighbors in ([1, 4], [1], [5], [1, 4, 5]):
        for membership in (0.1, 1.0):
            for window in (1, 2):
                config = settings | {
                    "alignment_window_size": window, "alignment_regularisation": 0.1,
                }
                m = AlignedUMAP(**config).fit([x, x], relations=[links])
                m.embeddings_[-1][:] = fixed
                m.embeddings_[0][:] = fixed
                m.embeddings_[0][2, 0] += 4.0
                graph = m.mappers_[-1].graph_.toarray()
                graph[2] = 0.0
                graph[2, neighbors] = membership
                m.mappers_[-1].graph_ = csr_matrix(graph)
                m.update(x, relations=links, n_epochs=2)
                cases[f"weights-{neighbors}-{membership}-{window}"] = dict(
                    epochs=2, config=config, neighbors=neighbors, membership=membership,
                    layout=bits(m.embeddings_[-1]),
                )

    # Every unlinked row in this append has at least one linked neighbor.
    # Two previous rows link to row 0, so the last link supplies its coordinate.
    previous = np.random.RandomState(153).normal(size=(6, 3)).astype(np.float32)
    append_links = {1: 0, 0: 2, 5: 4, 3: 0}
    m = AlignedUMAP(**settings).fit([x, x], relations=[links])
    m.embeddings_[-1][:] = fixed
    m.update(previous, relations=append_links)
    graph = m.mappers_[-1].graph_
    links_for_new_rows = set(append_links.values())
    assert all(
        row in links_for_new_rows
        or any(neighbor in links_for_new_rows for neighbor in graph[row].indices)
        for row in range(len(previous))
    )
    unlinked = {}
    for seed in (37, 42, 95):
        config = settings | {"n_epochs": 1, "transform_seed": seed}
        model = AlignedUMAP(**config).fit([x, x], relations=[links])
        model.update(x, relations={0: 60}, n_epochs=0)
        unlinked[str(seed)] = dict(config=config, layout=bits(model.embeddings_[-1]))

    # Observe the arrays returned by NumPy's public RNG and dtype conversion.
    # With a distinct transform seed, the trace isolates the batch stream.
    random_state = np.random.RandomState
    seed_changes = []

    class SeedArray(np.ndarray):
        def astype(self, *args, **kwargs):
            array = np.asarray(self).astype(*args, **kwargs)
            seed_changes.append((array.tolist(), array))
            return array

    class ObservedRandomState(random_state):
        def __init__(self, seed=None):
            super().__init__(seed)
            self.observe = seed == 37

        def randint(self, low, high=None, size=None, dtype=int):
            result = super().randint(low, high, size, dtype)
            return result.view(SeedArray) if self.observe and size == 3 else result

    joint = {}
    with patch.object(np.random, "RandomState", ObservedRandomState):
        for epochs in (0, 1, 2, 8):
            seed_changes.clear()
            model = AlignedUMAP(**(settings | {"n_epochs": epochs, "transform_seed": 37})).fit(
                [x, x], relations=[links]
            )
            initial, final = seed_changes.pop()
            joint[str(epochs)] = dict(
                seed_before=initial, seed_after=final.tolist(),
                layouts=[bits(layout) for layout in model.embeddings_],
            )
    return dict(
        data=x.tolist(), fixed=fixed.tolist(), cases=cases, unlinked=unlinked, joint=joint,
        partial=dict(
            config=settings, data=previous.tolist(), relations=list(append_links.items()),
            previous=bits(fixed), layout=bits(m.embeddings_[-1]),
        ),
    )


if __name__ == "__main__":
    name = "windows" if platform.system() == "Windows" else "ubuntu"
    output = Path(__file__).with_name("fixtures") / f"umap-learn-0.6.0-aligned-optimizer-{name}-x86_64.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if version("umap-learn") != "0.6.0" or platform.python_version() != "3.13.12":
        raise SystemExit("Use the locked reference environment in reference/")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with tempfile.TemporaryDirectory(prefix="umap-aligned-optimizer-") as cache:
        os.environ["NUMBA_CACHE_DIR"] = cache
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = capture()
    content = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"public fixture differs from {args.output}")
    else:
        args.output.write_bytes(content.encode("utf-8"))
    print(f"Captured {len(result['cases']) + len(result['unlinked']) + len(result['joint']) + 1} aligned optimizer cases")
