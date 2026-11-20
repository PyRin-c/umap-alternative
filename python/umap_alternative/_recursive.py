"""Dense PCA anchors and verified graph coarsening for modern layouts."""

import numbers

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.utils.extmath import randomized_svd


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


def _recursive_partition(graph, n_parts):
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    order = np.argsort(degrees, kind="stable")
    rows, hubs = order[:-n_parts], order[-n_parts:]
    if degrees[rows].min() == 0 or degrees[rows[-1]] == degrees[hubs[0]]:
        raise NotImplementedError("recursive coarsening requires positive degrees and unambiguous hubs")

    # Each component of the non-hub rows must attach to a single hub.
    # Propagation order cannot change the labels in this case.
    count, groups = connected_components(graph[rows][:, rows], directed=False)
    attached = graph[rows][:, hubs].tocoo()
    first = np.full(count, n_parts, dtype=np.intp)
    last = np.full(count, -1, dtype=np.intp)
    np.minimum.at(first, groups[attached.row], attached.col)
    np.maximum.at(last, groups[attached.row], attached.col)
    if np.any(first != last):
        raise NotImplementedError("recursive coarsening with competing or missing hubs is not implemented")
    labels = np.empty(len(degrees), dtype=np.intp)
    labels[hubs] = np.arange(n_parts)
    labels[rows] = first[groups]
    return labels, hubs, degrees


def _coarse_graph(graph, labels, hubs, weights, remove_diagonal):
    # Cross-part edges join hubs because every non-hub row has a single
    # possible label. The hub subgraph therefore stores all cross-part weights.
    coarse = graph[hubs][:, hubs].astype(np.float64)
    coarse.setdiag(0.0)
    if weights == "transition":
        totals = np.asarray(coarse.sum(axis=1)).ravel()
        inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
        transition = coarse.multiply(inverse[:, None]).tocsr()
        coarse = transition + transition.T - transition.multiply(transition.T)
    elif not remove_diagonal:
        edges = graph.tocoo()
        internal = labels[edges.row] == labels[edges.col]
        complements = np.ones(len(hubs), dtype=np.float64)
        np.multiply.at(complements, labels[edges.row[internal]], 1.0 - edges.data[internal])
        coarse.setdiag(1.0 - complements)
    coarse.eliminate_zeros()
    return coarse.astype(np.float32)


def _recursive_levels(graph, threshold, ratio, weights, options):
    levels = []
    while graph.shape[0] > threshold:
        if options.get("n_embedding_epochs") != 0:
            raise NotImplementedError("recursive coarsening requires n_embedding_epochs=0")
        size = graph.shape[0]
        n_parts = max(size // ratio, min(threshold, size // 2))
        labels, hubs, degrees = _recursive_partition(graph, n_parts)
        levels.append((graph, labels, degrees))
        if n_parts <= threshold:
            break
        graph = _coarse_graph(graph, labels, hubs, weights, options.get("remove_coarse_diagonal", False))
    return levels


def _expand_layout(graph, labels, degrees, child, anchor):
    expanded = child[labels]
    expanded = 0.5 * (expanded + (graph @ expanded) / degrees[:, None])
    expanded -= expanded.mean(axis=0)
    left, _, right = np.linalg.svd(expanded.T @ (anchor - anchor.mean(axis=0)))
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return expanded @ (left @ right)


def _modern_recursive_initialization(graph, features, n_components, rng, options, ratio, weights):
    """Return prepared coordinates for the base case or an unambiguous hierarchy."""
    threshold = options.get("base_init_threshold", _BASE_INIT_THRESHOLD)
    if n_components > min(features.shape):
        raise NotImplementedError("recursive initialization requires enough rows and features for n_components")
    if len(features) > _ANCHOR_SAMPLE_SIZE:
        raise NotImplementedError("recursive PCA anchor subsampling is not implemented")
    if not np.isfinite(features).all():
        raise ValueError("recursive PCA requires finite distances")
    levels = _recursive_levels(graph, threshold, ratio, weights, options)
    anchors = [_sampled_pca_anchor(features, n_components, rng)]
    for _, labels, _ in levels:
        rng.randint(-2**31 + 1, 2**31 - 2, 3)
        counts = np.bincount(labels)
        means = np.zeros((len(counts), n_components), dtype=np.float32)
        np.add.at(means, labels, anchors[-1])
        means /= counts[:, None]
        anchors.append(means)
    layout = _recursive_base(anchors[-1], depth=len(anchors))
    for (graph, labels, degrees), anchor in zip(reversed(levels), reversed(anchors[:-1]), strict=True):
        layout = _expand_layout(graph, labels, degrees, layout, anchor)
    return layout
