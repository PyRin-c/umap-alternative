"""Dense PCA anchors and verified graph coarsening for modern layouts."""

import numbers

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from . import _native


_BASE_INIT_THRESHOLD = 1024
_ANCHOR_SAMPLE_SIZE = 16384


def _recursive_base(coordinates, depth):
    coordinates = np.array(coordinates, dtype=np.float32, copy=True, order="C")
    coordinates -= coordinates.mean(axis=0)
    lower, upper = np.percentile(coordinates, (5, 95), axis=0).astype(np.float32)
    spans = upper - lower
    spans[spans == 0] = 1.0
    target = 3.0 * np.log10(len(coordinates)) * np.log2(depth + 1.0)
    coordinates *= np.asarray(target / spans.astype(np.float64), dtype=np.float32)
    return coordinates


def _sampled_pca_anchor(features, n_components, rng):
    # The projection draws one normal sketch with eight extra columns.
    centered = features.astype(np.float64)
    centered -= centered.mean(axis=0)
    left, values, _ = randomized_svd(
        centered, n_components=n_components, n_oversamples=8, n_iter=1, random_state=rng,
    )
    scores = np.ascontiguousarray(left * values, dtype=np.float32)
    spans = np.ptp(scores, axis=0)
    spans[spans == 0] = 1.0
    scores -= scores.min(axis=0)
    scores /= spans
    scores *= 10.0
    return scores


def _validate_recursive_initialization(ratio, weights, kwds):
    if not isinstance(ratio, numbers.Integral) or ratio not in (2, 3, 4):
        raise ValueError("recursive_coarsening_ratio must be one of 2, 3, or 4")
    if weights not in ("transition", "fuzzy_union"):
        raise ValueError("coarse_layout_weights must be 'transition' or 'fuzzy_union'")
    if kwds is None:
        return {}
    if not isinstance(kwds, dict):
        raise ValueError("recursive_init_kwds must be a dictionary or None")
    supported = {"base_init_threshold", "n_embedding_epochs", "anchor_method", "remove_coarse_diagonal"}
    unknown = kwds.keys() - supported
    if unknown:
        raise NotImplementedError(f"unsupported recursive_init_kwds {sorted(unknown)!r}")
    if kwds.get("anchor_method", "sampled_pca") != "sampled_pca":
        raise NotImplementedError("recursive initialization supports the sampled_pca anchor")
    threshold = kwds.get("base_init_threshold", _BASE_INIT_THRESHOLD)
    if not isinstance(threshold, numbers.Integral) or threshold < 2:
        raise ValueError("base_init_threshold must be an integer of at least two")
    return kwds


def _recursive_partition(graph, n_parts, seed, *, layout_graph=None):
    degree_graph = graph if layout_graph is None else layout_graph
    degrees = np.asarray(degree_graph.sum(axis=1)).ravel()
    hubs = np.argsort(degrees, kind="stable")[-n_parts:]
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
        coarse = csr_matrix((values, (source, target)), shape=shape)
        totals = np.asarray(coarse.sum(axis=1)).ravel()
        inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
        transition = coarse.multiply(inverse[:, None]).tocsr()
        layout = transition + transition.T - transition.multiply(transition.T)
        layout.eliminate_zeros()
        return coarse, layout
    elif len(values):
        order = np.lexsort((target, source))
        source, target, values = source[order], target[order], values[order]
        starts = np.r_[0, np.flatnonzero((source[1:] != source[:-1]) | (target[1:] != target[:-1])) + 1]
        complements = np.multiply.reduceat((1.0 - values).astype(np.float64), starts)
        coarse = csr_matrix((1.0 - complements, (source[starts], target[starts])), shape=shape)
    else:
        coarse = csr_matrix(shape, dtype=np.float32)
    coarse.eliminate_zeros()
    coarse.sort_indices()
    coarse = coarse.astype(np.float32)
    return coarse, coarse


def _expand_layout(graph, labels, child, anchor):
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    expanded = child[labels]
    averages = np.divide(graph @ expanded, degrees[:, None], out=np.zeros_like(expanded), where=degrees[:, None] > 0)
    expanded = 0.5 * (expanded + averages)
    expanded -= expanded.mean(axis=0)
    left, _, right = np.linalg.svd(expanded.T @ (anchor - anchor.mean(axis=0)))
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return expanded @ (left @ right)


def _modern_recursive_initialization(graph, features, n_components, rng, options, ratio, weights, refine):
    """Return prepared coordinates for the base case or a contracted hierarchy."""
    threshold = options.get("base_init_threshold", _BASE_INIT_THRESHOLD)
    if n_components > min(features.shape):
        raise NotImplementedError("recursive initialization requires enough rows and features for n_components")
    if len(features) > _ANCHOR_SAMPLE_SIZE:
        raise NotImplementedError("recursive PCA anchor subsampling is not implemented")
    if not np.isfinite(features).all():
        raise ValueError("recursive PCA requires finite distances")
    coarse_epochs = options.get("n_embedding_epochs")
    if graph.shape[0] > threshold and coarse_epochs not in (0, 1, 2, 3):
        raise NotImplementedError("recursive coarsening supports zero to three explicit coarse epochs")
    if graph.shape[0] // ratio > threshold and coarse_epochs in (2, 3):
        raise NotImplementedError("multiple recursive levels support zero or one coarse epoch")
    levels = []
    contraction_graph = graph
    anchors = [_sampled_pca_anchor(features, n_components, rng)]
    while graph.shape[0] > threshold:
        size = graph.shape[0]
        target = min(threshold, size // 2) if not levels else 2
        n_parts = max(2, size // ratio, target)
        seed = rng.randint(-2**31 + 1, 2**31 - 2, 3)
        labels = _recursive_partition(contraction_graph, n_parts, seed, layout_graph=graph)
        counts = np.bincount(labels)
        means = np.zeros((len(counts), n_components), dtype=np.float32)
        np.add.at(means, labels, anchors[-1])
        means /= counts[:, None]
        anchors.append(means)
        child = None
        if n_parts > threshold or coarse_epochs:
            contraction_graph, child = _coarse_graphs(
                contraction_graph, labels, n_parts, weights, options.get("remove_coarse_diagonal", False),
            )
        levels.append((graph, labels, child))
        if n_parts <= threshold:
            break
        graph = child
    layout = _recursive_base(anchors[-1], depth=len(anchors))
    for (parent, labels, child), anchor in zip(reversed(levels), reversed(anchors[:-1]), strict=True):
        if coarse_epochs:
            layout = refine(child, layout, coarse_epochs, rng)
        layout = _expand_layout(parent, labels, layout, anchor)
    return layout
