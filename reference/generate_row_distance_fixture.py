"""Capture scalar feature distances from the pinned public wheel."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
from umap import distances


NAMES = (
    "canberra", "cosine", "correlation", "braycurtis", "hamming", "jaccard",
    "dice", "matching", "kulsinski", "rogerstanimoto", "russellrao",
    "sokalsneath", "sokalmichener", "yule",
)


def pairs():
    for dtype in (np.float32, np.float64):
        info = np.finfo(dtype)
        next_one = np.nextafter(dtype(1), dtype(2))
        tiny = info.smallest_subnormal
        for name, left, right in (
            ("empty", [], []),
            ("zero", [0, -0., 0], [-0., 0, -0.]),
            ("constants", [1, 1, 1], [2, 2, 2]),
            ("orthogonal", [1, 0, -1], [1, -2, 1]),
            ("antipodes", [1, -2], [-1, 2]),
            ("same-support", [-0., 4, -1, 0, 8], [0, -3, 9, 0, .2]),
            ("both-mismatches", [-.63, .01, 9.5, 0, -.3, 2], [.12, 0, 3.6, 4.01, .03, 0]),
            ("longer-right", [.12, 1., 4.], [-.23, .4, 8.5, 100.]),
            ("near-parallel", [1, next_one, 1.3], [next_one, 1, 1.3]),
            ("subnormal", [tiny, -tiny, 0], [tiny, tiny, tiny]),
            ("maximum", [info.max, -info.max, 1.], [info.max, 0, 0]),
            ("nan", [np.nan, 2, 0], [1, 1, 0]),
            ("negative-nan", [-np.nan, 1], [np.nan, 1]),
            ("nan-zero", [np.nan, 1], [0, 0]),
            ("infinite", [np.inf, 1], [np.inf, 1]),
            ("opposite-infinite", [np.inf, 1], [-np.inf, 0]),
        ):
            yield f"{dtype.__name__}-{name}", np.array(left, dtype), np.array(right, dtype)

        rng = np.random.default_rng(20182)
        for width in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
            left, right = (rng.normal(size=width).astype(dtype) for _ in range(2))
            left[::5] = 0
            right[::7] = 0
            yield f"{dtype.__name__}-width-{width}", left, right

    positive_nan, negative_nan = np.array([0x7ff8000000000021, 0xfff800000000004b], np.uint64).view(np.float64)
    yield "float64-payload-left", np.array([negative_nan, 1.]), np.array([positive_nan, 1.])
    yield "float64-payload-order", np.array([0., positive_nan]), np.array([negative_nan, 1.])

    for name, left, left_dtype, right, right_dtype in (
        ("bool", [0, 1, 0, 1], "bool", [1, 1, 0, 0], "bool"),
        ("signed16", [7, -11, 2, 0], "int16", [4, 11, 0, -8], "int16"),
        ("signed64", [2**63 - 1, 3], "int64", [2**63 - 2, 1], "int64"),
        ("signed64-min", [-2**63, 3], "int64", [2**63 - 1, 1], "int64"),
        ("signed64-min-zero", [-2**63], "int64", [0], "int64"),
        ("signed64-min-one", [-2**63], "int64", [1], "int64"),
        ("signed64-product", [2**32, 2**32 + 1, 0], "int64", [0, 2**32, 1], "int64"),
        ("unsigned8", [2, 3], "uint8", [4, 1], "uint8"),
        ("unsigned32", [2**32 - 1, 3], "uint32", [2**32 - 2, 1], "uint32"),
        ("unsigned64", [2**64 - 1, 3], "uint64", [2**64 - 2, 1], "uint64"),
        ("unsigned64-high-bit", [2**63, 3], "uint64", [1, 4], "uint64"),
        ("mixed-floats", [.123, 1., .1], "float32", [.73, 3., .01], "float64"),
        ("mixed-float32-int16", [.123, 1., .1], "float32", [2, 3, 4], "int16"),
        ("mixed-float32-int32", [.123, 1., .1], "float32", [2, 3, 4], "int32"),
        ("mixed-float64-high-uint", [3., 0, 4], "float64", [2**63 + 1, 1, 3], "uint64"),
        ("mixed-signed-unsigned", [-1, 3, 0], "int16", [4, 1, 0], "uint32"),
        ("mixed-bool-unsigned", [1, 1, 0], "bool", [4, 1, 0], "uint8"),
        ("mixed-signed-unsigned-high", [-1, 1], "int64", [2**64 - 1, 2**63 + 1], "uint64"),
        ("mixed-unsigned-signed-high", [2**64 - 1, 2**63 + 1], "uint64", [-1, 1], "int64"),
        ("mixed-signed-unsigned-wrap", [2**63 - 2], "int64", [2**63 - 1], "uint64"),
        ("mixed-signed-unsigned-round", [2**53 + 1], "int64", [2**53], "uint64"),
        ("float16", [0, .2], "float16", [.9, .3], "float16"),
    ):
        yield name, np.array(left, left_dtype), np.array(right, right_dtype)


def observe(function, left, right):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            value = np.float64(function(left, right)).view(np.uint64).item()
            result = {"bits": f"{value:016x}"}
        except Exception as error:
            result = {"error": type(error).__name__}
    result["warnings"] = [type(item.message).__name__ for item in seen]
    return result


def encode(row):
    return [row.dtype.str, row.tobytes().hex()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert version("umap-learn") == "0.6.0"
    rows = []
    for name, left, right in pairs():
        observed = {metric: observe(distances.named_distances[metric], left, right) for metric in NAMES}
        rows.append({"name": name, "left": encode(left), "right": encode(right), "observed": observed})
    data = ("[\n" + ",\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n]\n").encode()
    if args.check:
        if args.output.read_bytes() != data:
            raise SystemExit(f"fixture differs from {args.output}")
    else:
        args.output.write_bytes(data)


if __name__ == "__main__":
    main()
