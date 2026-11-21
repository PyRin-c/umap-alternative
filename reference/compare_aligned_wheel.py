"""Compare an installed umap-alternative wheel with the public AlignedUMAP fixture."""

import json
import platform
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from umap_alternative import AlignedUMAP


def main():
    system = "windows" if platform.system() == "Windows" else "ubuntu"
    path = Path(__file__).with_name("fixtures") / f"umap-learn-0.6.0-aligned-{system}-x86_64.json"
    public = json.loads(path.read_text(encoding="utf-8"))
    for name, case in public["cases"].items():
        if not name.startswith(("epoch-2", "epoch-8", "zero-window", "single-window", "batch-")):
            continue
        if name.endswith("-padded"):
            continue
        data = [np.asarray(matrix, dtype=np.float32) for matrix in public["data"][case["family"]]]
        links = [dict(items) for items in public["relations"][case["relation_group"]]]
        model = AlignedUMAP(**(public["settings"] | case["overrides"]))
        result = np.concatenate(model.fit_transform(data, relations=links))
        expected = np.asarray(sum(case["embeddings"], []), dtype=np.uint32).view(np.float32)
        matches = np.count_nonzero(result.view(np.uint32) == expected.view(np.uint32))
        print(f"{name:24} {matches:2}/{expected.size} exact, max abs {np.max(np.abs(result - expected)):.7f}")


if __name__ == "__main__":
    import warnings

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value -1 overridden", category=UserWarning)
        main()
