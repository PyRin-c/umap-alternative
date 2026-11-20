"""Python bindings for the independent Rust implementation of UMAP."""

from ._native import __version__ as __version__
from ._umap import UMAP

__all__ = ["UMAP", "__version__"]
