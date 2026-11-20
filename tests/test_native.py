from importlib.metadata import version

import umap_alternative


def test_native_version_matches_distribution():
    assert umap_alternative.__version__ == version("umap-alternative")
