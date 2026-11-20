"""Python bindings for the independent Rust implementation of UMAP."""

from ._native import __version__ as __version__
from ._aligned import AlignedUMAP
from ._umap import UMAP

__all__ = ["AlignedUMAP", "UMAP", "__version__"]
