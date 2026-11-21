"""Dense PCA anchors and verified graph coarsening for modern layouts."""

import numbers
from math import sqrt

import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from . import _native
from ._curve import _finite_float


_BASE_INIT_THRESHOLD = 1024
_ANCHOR_SAMPLE_SIZE = 16384
_COARSE_EPOCHS = 50


def _robust_spans(coordinates):
    ranks = (len(coordinates) - 1) * np.array([0.05, 0.95], dtype=np.float64)
    floor = np.floor(ranks).astype(np.intp)
    ceiling = np.ceil(ranks).astype(np.intp)
    partitioned = np.partition(coordinates, np.r_[floor, ceiling], axis=0)
    left, right = partitioned[floor], partitioned[ceiling]
    gap = right - left
    offset = (ranks - floor)[:, None]
    # Interpolate from the nearer endpoint in float32. Compute the distance
    # from either endpoint before rounding it to float32.
    lower, upper = np.where(
        offset < 0.5, left + gap * offset.astype(np.float32), right - gap * (1.0 - offset).astype(np.float32),
    )
    spans = upper - lower
    spans[spans == 0] = 1.0
    return spans


def _recursive_base(coordinates, depth):
    coordinates = np.array(coordinates, dtype=np.float32, copy=True, order="C")
    coordinates -= coordinates.mean(axis=0)
    spans = _robust_spans(coordinates)
    target = 3.0 * np.log10(len(coordinates)) * np.log2(depth + 1.0)
    return (coordinates * (target / spans.astype(np.float64))).astype(np.float32)


def _sampled_pca_anchor(features, n_components, rng, *, graph=None, sample_size=None):
    # The projection draws one normal sketch with eight extra columns.
    sample_size = max(_ANCHOR_SAMPLE_SIZE if sample_size is None else sample_size, n_components)
    sampled = len(features) > sample_size
    anchors = rng.choice(len(features), size=sample_size, replace=False) if sampled else None
    sample = features[anchors] if sampled else features
    # Accumulate column sums in float32, then divide and center in float64.
    sample = np.ascontiguousarray(sample, dtype=np.float32)
    center = sample.sum(axis=0, dtype=np.float32).astype(np.float64) / len(sample)
    left, values, _ = randomized_svd(
        sample.astype(np.float64) - center,
        n_components=n_components, n_oversamples=8, n_iter=1, random_state=rng,
    )
    scores = np.ascontiguousarray(left * values, dtype=np.float32)
    spans = np.ptp(scores, axis=0)
    spans[spans == 0] = 1.0
    scores -= scores.min(axis=0)
    scores /= spans
    scores *= 10.0
    if sampled:
        links = graph[:, anchors]
        masses = np.asarray(links.sum(axis=1)).ravel()
        placed = np.broadcast_to(scores.mean(axis=0), (len(features), n_components)).copy()
        np.divide(links @ scores, masses[:, None], out=placed, where=masses[:, None] > 0)
        placed[anchors] = scores
        return placed
    return scores


def _validate_recursive_initialization(ratio, weights, kwds):
    if not isinstance(ratio, numbers.Integral) or ratio not in (2, 3, 4):
        raise ValueError("recursive_coarsening_ratio must be one of 2, 3, or 4")
    if weights not in ("transition", "fuzzy_union"):
        raise ValueError("coarse_layout_weights must be 'transition' or 'fuzzy_union'")
    if kwds is None:
        kwds = {}
    if not isinstance(kwds, dict):
        raise ValueError("recursive_init_kwds must be a dictionary or None")
    supported = {
        "base_init_threshold", "n_embedding_epochs", "anchor_method", "remove_coarse_diagonal",
        "recursive_negative_sample_rate", "recursive_repulsion_strength", "good_initialization_policy",
        "curve_schedule", "recursive_negative_selection_range_mode", "recursive_negative_selection_range_scale",
        "anchor_sample_size",
    }
    unknown = kwds.keys() - supported
    if unknown:
        raise NotImplementedError(f"unsupported recursive_init_kwds {sorted(unknown)!r}")
    if kwds.get("anchor_method", "sampled_pca") != "sampled_pca":
        raise NotImplementedError("recursive initialization supports the sampled_pca anchor")
    threshold = kwds.get("base_init_threshold", _BASE_INIT_THRESHOLD)
    if not isinstance(threshold, numbers.Integral) or threshold < 2:
        raise ValueError("base_init_threshold must be an integer of at least two")
    sample_size = kwds.get("anchor_sample_size", _ANCHOR_SAMPLE_SIZE)
    if not isinstance(sample_size, numbers.Integral) or sample_size <= 0:
        raise ValueError("anchor_sample_size must be a positive integer")
    validated = kwds.copy()
    budget = kwds.get("n_embedding_epochs")
    if budget is not None and (not isinstance(budget, numbers.Integral) or budget < 0):
        raise ValueError("n_embedding_epochs must be a nonnegative integer or None")
    validated["n_embedding_epochs"] = _COARSE_EPOCHS if budget is None else int(budget)
    for name in (
        "recursive_negative_sample_rate", "recursive_repulsion_strength", "recursive_negative_selection_range_scale",
    ):
        if name in kwds:
            value = _finite_float(kwds[name], name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            validated[name] = value
    if kwds.get("good_initialization_policy", "heuristic") not in ("heuristic", "never", "available", "always"):
        raise ValueError("invalid recursive good_initialization_policy")
    if kwds.get("curve_schedule", "strong_to_one") not in ("strong_to_one", "original"):
        raise NotImplementedError("recursive curve_schedule supports 'strong_to_one' and 'original'")
    if kwds.get("recursive_negative_selection_range_mode", "scaled_coarse") not in (
        "scaled_coarse", "coarse_n", "fixed_200k",
    ):
        raise ValueError("invalid recursive_negative_selection_range_mode")
    return validated


def _recursive_partition(graph, n_parts, seed, *, layout_graph=None):
    degree_graph = graph if layout_graph is None else layout_graph
    degrees = np.asarray(degree_graph.sum(axis=1)).ravel()
    hubs = np.asarray(_native._recursive_hub_order(degrees.tolist())[-n_parts:], dtype=np.intp)
    labels = np.full(len(degrees), -1, dtype=np.intp)
    labels[hubs] = np.arange(n_parts)
    while True:
        known = np.flatnonzero(labels >= 0)
        pending = np.flatnonzero(labels < 0)
        if not len(pending):
            break
        indicator = csr_matrix(
            (np.ones(len(known), dtype=np.float32), (known, labels[known])),
            shape=(len(labels), n_parts),
        )
        # Score the whole frontier before assigning any of its rows. Sparse
        # product order supplies the tie order when two parts have equal support.
        support = graph[pending] @ indicator
        for local, row in enumerate(pending):
            start, stop = support.indptr[local:local + 2]
            if start != stop:
                best = start + np.argmax(support.data[start:stop])
                if support.data[best] >= 1.0:
                    labels[row] = support.indices[best]
        if np.all(labels[pending] < 0):
            break

    # A weakly attached row can copy the first known neighbor in CSR order.
    # Sweep in row order while any of the pending rows can still acquire labels.
    while np.any(labels[pending] < 0):
        previous = np.count_nonzero(labels[pending] < 0)
        for row in pending:
            start, stop = graph.indptr[row:row + 2]
            neighbors = labels[graph.indices[start:stop]]
            known = neighbors[neighbors >= 0]
            if len(known):
                labels[row] = known[0]
        if np.count_nonzero(labels[pending] < 0) == previous:
            break

    missing = np.flatnonzero(labels < 0)
    if len(missing):
        labels[missing] = _native._recursive_part_labels(seed.tolist(), missing.tolist(), n_parts)
    return labels


def _coarse_graphs(graph, labels, n_parts, weights, remove_diagonal):
    edges = graph.tocoo()
    source, target = labels[edges.row], labels[edges.col]
    keep = source != target if weights == "transition" or remove_diagonal else slice(None)
    source, target, values = source[keep], target[keep], edges.data[keep]
    shape = (n_parts, n_parts)
    if weights == "transition":
        # The next contraction uses raw support and ranks hubs by the
        # normalized layout degrees. Expansion also uses the layout weights.
        # Accumulate duplicate cross-part weights before rounding the layout.
        coarse = csr_matrix((values.astype(np.float64), (source, target)), shape=shape)
        totals = np.asarray(coarse.sum(axis=1)).ravel()
        inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
        transition = coarse.multiply(inverse[:, None]).tocsr()
        layout = (transition + transition.T - transition.multiply(transition.T)).astype(np.float32)
        layout.eliminate_zeros()
        return coarse, layout
    elif len(values):
        order = np.lexsort((target, source))
        source, target, values = source[order], target[order], values[order]
        starts = np.r_[0, np.flatnonzero((source[1:] != source[:-1]) | (target[1:] != target[:-1])) + 1]
        complements = np.multiply.reduceat(1.0 - values.astype(np.float64), starts)
        coarse = csr_matrix((1.0 - complements, (source[starts], target[starts])), shape=shape)
    else:
        coarse = csr_matrix(shape, dtype=np.float32)
    coarse.eliminate_zeros()
    coarse.sort_indices()
    coarse = coarse.astype(np.float32)
    return coarse, coarse


def _aggregate_anchors(anchors, labels):
    counts = np.bincount(labels)
    means = np.zeros((len(counts), anchors.shape[1]), dtype=np.float64)
    # Each member contributes its share of the part's mean before summing.
    inverse_counts = 1.0 / counts
    np.add.at(means, labels, anchors * inverse_counts[labels, None])
    return means.astype(np.float32)


def _procrustes_unit_rms(coordinates):
    mean = coordinates.sum(axis=0, dtype=np.float32).astype(np.float64) / len(coordinates)
    centered = (coordinates - mean).astype(np.float32)
    squares = np.square(centered).ravel()
    # The reference sums the squared entries in row order in float32.
    np.add.accumulate(squares, out=squares)
    rms = np.sqrt(squares[-1] / np.float32(squares.size))
    return centered / rms if rms != 0 else np.zeros_like(centered)


def _expand_layout(graph, labels, child, anchor):
    rows = graph.tocoo().row
    n_rows = len(labels)
    degrees = np.bincount(rows, graph.data, minlength=n_rows)
    expanded = child[labels]
    # Each edge product rounds in float32 before it contributes to a
    # float64 row sum. Keep that precision through centering.
    averages = np.empty_like(expanded, dtype=np.float64)
    for axis, values in enumerate(expanded.T):
        products = graph.data * values[graph.indices]
        averages[:, axis] = np.bincount(rows, products, minlength=n_rows)
    np.divide(averages, degrees[:, None], out=averages, where=degrees[:, None] > 0)
    expanded = 0.5 * (expanded + averages)
    expanded -= expanded.mean(axis=0)
    expanded = expanded.astype(np.float32)
    left, _, right = svd(
        _procrustes_unit_rms(expanded).T @ _procrustes_unit_rms(anchor), lapack_driver="gesdd",
    )
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return expanded @ (left @ right)


def _modern_recursive_initialization(graph, features, n_components, rng, options, ratio, weights, refine):
    """Return prepared coordinates for the base case or a contracted hierarchy."""
    threshold = options.get("base_init_threshold", _BASE_INIT_THRESHOLD)
    if n_components > min(features.shape):
        raise NotImplementedError("recursive initialization requires enough rows and features for n_components")
    if not np.isfinite(features).all():
        raise ValueError("recursive PCA requires finite distances")
    coarse_epochs = options["n_embedding_epochs"]
    levels = []
    contraction_graph = graph
    sample_size = options.get("anchor_sample_size", _ANCHOR_SAMPLE_SIZE)
    anchor = (
        _sampled_pca_anchor(features, n_components, rng, graph=graph, sample_size=sample_size)
        if len(features) > max(sample_size, n_components)
        else _sampled_pca_anchor(features, n_components, rng)
    )
    anchors = [anchor]
    while graph.shape[0] > threshold:
        size = graph.shape[0]
        target = min(threshold, size // 2) if not levels else 2
        n_parts = max(2, size // ratio, target)
        seed = rng.randint(-2**31 + 1, 2**31 - 2, 3)
        labels = _recursive_partition(contraction_graph, n_parts, seed, layout_graph=graph)
        anchors.append(_aggregate_anchors(anchors[-1], labels))
        child = None
        if n_parts > threshold or coarse_epochs:
            contraction_graph, child = _coarse_graphs(
                contraction_graph, labels, n_parts, weights, options.get("remove_coarse_diagonal", False),
            )
        levels.append((graph, labels, child, coarse_epochs))
        if n_parts <= threshold:
            break
        graph = child
        coarse_epochs = int(coarse_epochs * sqrt(2))
    layout = _recursive_base(anchors[-1], depth=len(anchors))
    policy = options.get("good_initialization_policy", "heuristic")
    for level in range(len(levels) - 1, -1, -1):
        parent, labels, child, budget = levels[level]
        if budget:
            descendants = len(levels) - level - 1
            use_refinement = policy in ("always", "available") or (policy == "heuristic" and descendants >= 2)
            layout = refine(child, layout, budget, rng, depth=level + 1, use_refinement=use_refinement)
        layout = _expand_layout(parent, labels, layout, anchors[level])
    return layout
