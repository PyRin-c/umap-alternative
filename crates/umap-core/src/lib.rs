//! Core crate for the independent Rust implementation of UMAP.

pub mod neighbors;

/// Version of the Rust implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
