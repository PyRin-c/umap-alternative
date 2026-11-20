"""Independent NumPy/SciPy model for the modern initialization observations."""

import argparse
import json
import sys
from math import isqrt
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import KDTree
from sklearn.utils.extmath import randomized_svd


def rescale(coordinates, graph, method, scale_factor, *, robust_span):
    centered = np.array(coordinates, dtype=np.float32, order="C", copy=True)
    centered -= centered.mean(axis=0)
    if method == "graph_edge":
        edges = graph.tocoo()
        keep = edges.data > 0
        if not keep.any():
            return centered
        lengths = np.linalg.norm(centered[edges.row[keep]] - centered[edges.col[keep]], axis=1)
        by_length = lengths.argsort()
        cumulative = np.cumsum(edges.data[keep][by_length], dtype=np.float64)
        selected = np.searchsorted(cumulative, cumulative[-1] / 2, side="left")
        median = lengths[by_length[selected]]
        denominator = median if median > 0 else np.float32(1)
        return centered * (np.float32(scale_factor) / denominator)

    lower, upper = np.percentile(centered, (5, 95), axis=0).astype(np.float32)
    widths = upper - lower
    widths[widths == 0] = 1.0
    if method == "isotropic":
        widths = np.exp(np.log(widths).mean())
    multiplier = np.asarray(np.float64(robust_span * scale_factor) / widths, dtype=np.float32)
    return centered * multiplier


def explicit_coordinates(initial, rng):
    values = np.ascontiguousarray(initial, dtype=np.float32)
    if len(np.unique(values, axis=0)) == len(values):
        return values
    distances, _ = KDTree(values).query(values, k=2)
    noise = rng.normal(0, 0.001 * distances[:, 1].mean(), values.shape)
    return (values + noise).astype(np.float32)


def random_coordinates(shape, rng):
    return rng.uniform(-10, 10, shape).astype(np.float32)


def spectral_vectors(graph, n_components):
    n_rows = graph.shape[0]
    k = n_components + 1
    if k >= n_rows or connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("the model requires a connected graph with more rows than eigenpairs")
    degrees = np.asarray(graph.sum(axis=0)).ravel()
    inv_degree = diags(1.0 / np.sqrt(degrees))
    adjacency = inv_degree @ graph @ inv_degree
    eigenvalues, eigenvectors = eigsh(
        adjacency, k=k, which="LA", v0=np.ones(n_rows, dtype=np.float32),
        ncv=min(n_rows, max(2 * k + 1, isqrt(n_rows))), tol=1e-4,
    )
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[1:]]


def spectral_coordinates(vectors, rng):
    coords = vectors * (4.0 / np.abs(vectors).max())
    coords += rng.normal(0, 0.001, coords.shape).astype(np.float32)
    return coords


def recursive_anchor(data, n_components, rng):
    # The public base-case handoff is consistent with eight oversamples and
    # a full-column projection. The float64 solve limits PCA rounding drift.
    centered = data.astype(np.float64)
    centered -= centered.mean(axis=0)
    rank = min(n_components, *centered.shape)
    left, singular, _ = randomized_svd(
        centered, n_components=rank, n_oversamples=8, n_iter=2, random_state=rng,
    )
    scores = (left * singular).astype(np.float32)
    span = np.ptp(scores, axis=0)
    scores = np.divide(scores, span, out=scores, where=span > 0)
    return np.pad(scores, ((0, 0), (0, n_components - rank)))


def recursive_base(anchor, depth):
    target = 3 * np.log10(len(anchor)) * np.log2(depth + 1)
    return rescale(anchor, None, "per_axis", 1.0, robust_span=target)


def coarsen(graph, requested, minimum_share):
    """Construct a partition by propagating labels from weighted-degree hubs."""
    n_rows = graph.shape[0]
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    hubs = np.argsort(degrees, kind="stable")[-requested:]
    n_connected, components = connected_components(graph, directed=False)
    for component in range(n_connected):
        rows = np.flatnonzero(components == component)
        if not np.isin(hubs, rows).any():
            hubs = np.append(hubs, rows[np.argmax(degrees[rows])])

    labels = np.full(n_rows, -1)
    labels[hubs] = np.arange(len(hubs))
    for _ in range(n_rows):
        missing = np.flatnonzero(labels < 0)
        next_labels = labels.copy()
        for row in missing:
            start, end = graph.indptr[row:row + 2]
            neighbors = labels[graph.indices[start:end]]
            labeled = neighbors >= 0
            if labeled.any():
                counts = np.bincount(neighbors[labeled], weights=graph.data[start:end][labeled])
                accepted = np.flatnonzero((counts > 0) & (counts >= minimum_share * degrees[row]))
                if len(accepted):
                    next_labels[row] = accepted[0]
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
    # A disconnected component or an unreached vertex remains its own part.
    return np.unique(np.where(labels >= 0, labels, len(hubs) + np.arange(n_rows)), return_inverse=True)[1]


def coarse_graph(graph, labels, mode):
    edges = graph.tocoo()
    count = int(labels.max()) + 1
    source, target = labels[edges.row], labels[edges.col]
    if mode == "fuzzy_union":
        complement = np.ones((count, count), dtype=np.float64)
        np.multiply.at(complement, (source, target), 1.0 - edges.data)
        return csr_matrix((1.0 - complement).astype(np.float32))

    sums = np.zeros((count, count), dtype=np.float64)
    np.add.at(sums, (source, target), edges.data)
    totals = sums.sum(axis=1, keepdims=True)
    transition = np.divide(sums, totals, out=np.zeros_like(sums), where=totals > 0)
    return csr_matrix((transition + transition.T - transition * transition.T).astype(np.float32))


def draw_layout_seed(rng):
    # The public zero-epoch embedding still consumes this three-word draw.
    rng.randint(-2**31 + 1, 2**31 - 1, size=3)


def recursive_zero_epoch(graph, anchor, rng, *, threshold, ratio, weights,
                         coarsen_on_layout=True, depth=1, parts=None):
    """Compare centroid-and-barycenter coarsening on zero coarse epochs."""
    n_rows = graph.shape[0]
    if n_rows <= threshold:
        return recursive_base(anchor, depth)

    requested = parts or max(threshold, n_rows // ratio)
    labels = coarsen(graph, requested, minimum_share=0.1 if weights == "transition" else 0)
    counts = np.bincount(labels)
    if len(counts) >= n_rows:
        return recursive_base(anchor, depth)

    centers = np.zeros((len(counts), anchor.shape[1]), dtype=np.float64)
    np.add.at(centers, labels, anchor)
    centers /= counts[:, None]
    coarse = coarse_graph(graph, labels, weights if coarsen_on_layout else "fuzzy_union")
    layout = recursive_zero_epoch(
        coarse, centers.astype(np.float32), rng,
        threshold=threshold, ratio=ratio, weights=weights, coarsen_on_layout=coarsen_on_layout,
        depth=depth + 1,
    )
    draw_layout_seed(rng)
    expanded = layout[labels]
    degree = np.asarray(graph.sum(axis=1))
    neighbors = np.divide(graph @ expanded, degree, out=expanded.copy(), where=degree > 0)
    expanded = 0.5 * (expanded + neighbors)
    expanded -= expanded.mean(axis=0)
    reference = anchor - anchor.mean(axis=0)
    rotation, _ = orthogonal_procrustes(expanded, reference)
    return (expanded @ rotation).astype(np.float32)


def words_to_floats(values, width):
    return np.array(values, dtype=np.uint32).view(np.float32).reshape(-1, width)


def rng_matches(rng, checkpoint):
    _, _, position, have_gaussian, gaussian = rng.get_state()
    return (
        position == checkpoint["position"]
        and have_gaussian == (checkpoint["gaussian"] is not None)
        and (not have_gaussian or gaussian == checkpoint["gaussian"])
        and rng.randint(2**32, size=4, dtype=np.uint32).tolist() == checkpoint["next_words"]
    )


def residual(actual, expected_bits):
    expected = words_to_floats(expected_bits, actual.shape[1])
    exact = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finite = np.isfinite(actual) & np.isfinite(expected)
    special = np.isfinite(actual) == np.isfinite(expected)
    if not special.all():
        return exact, np.inf
    return exact, np.max(np.abs(actual[finite] - expected[finite]), initial=0.0)


def compare(fixture):
    graphs = {
        key: csr_matrix((words_to_floats(g["weights_bits"], 1).ravel(), g["indices"], g["indptr"]),
                        shape=(g["n_rows"], g["n_rows"]))
        for key, g in fixture["graphs"].items()
    }
    with np.errstate(all="ignore"):
        for case in fixture["cases"] + fixture["spectral"]:
            graph = graphs[case["graph"]]
            components = case["n_components"]
            rng = np.random.RandomState(case["seed"])
            if "rng_start" in case:
                rng.standard_normal()
            if case["init"] == "explicit":
                values = explicit_coordinates(words_to_floats(case["initial_bits"], components), rng)
            elif case["init"] == "random":
                values = random_coordinates((graph.shape[0], components), rng)
            else:
                vectors = spectral_vectors(graph, components)
                if not residual(vectors, case["initial_bits"])[0]:
                    print(case["name"], "eigenvector residual", residual(vectors, case["initial_bits"]))
                values = spectral_coordinates(vectors, rng)
            target = 0 if case["name"].startswith("direct-") else 3 * np.log10(len(values))
            coords = rescale(values, graph, case["method"], case["factor"], robust_span=target)
            draw_layout_seed(rng)
            print(case["name"], *residual(coords, case["embedding_bits"]),
                  "rng", rng_matches(rng, case["rng"]))

    for case in fixture["recursive"]:
        if case["epochs"]:
            continue
        graph = graphs[case["graph"]]
        data = np.array(fixture["recursive_data"][case["graph"]], dtype=np.float32)
        rng = np.random.RandomState(case["seed"])
        anchor = recursive_anchor(data, 2, rng)
        raw = recursive_zero_epoch(graph, anchor, rng, threshold=case["threshold"],
                                   ratio=case["ratio"], weights=case["weights"],
                                   coarsen_on_layout=case["coarsen_on_layout"])
        raw_state = np.random.RandomState()
        raw_state.set_state(rng.get_state())
        coords = rescale(raw, graph, "graph_edge", 1.0, robust_span=0)
        draw_layout_seed(rng)
        print("recursive", case["graph"], case["seed"], case["threshold"], case["weights"],
              case["coarsen_on_layout"],
              "raw", residual(raw, case["raw_bits"]), "rng", rng_matches(raw_state, case["raw_rng"]),
              "initialized", residual(coords, case["embedding_bits"]), "rng", rng_matches(rng, case["rng"]))

    anchor = np.array(fixture["coarse_anchor"], dtype=np.float32)
    for case in fixture["coarse"]:
        rng = np.random.RandomState(42)
        raw = recursive_zero_epoch(graphs["coarse"], anchor, rng, threshold=case["threshold"],
                                   ratio=4, weights="transition", depth=case["depth"], parts=3)
        print("controlled coarse", case["threshold"], case["depth"],
              *residual(raw, case["embedding_bits"]), "rng", rng_matches(rng, case["rng"]))


if __name__ == "__main__":
    sys.tracebacklimit = 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    compare(json.loads(args.fixture.read_text(encoding="utf-8")))
