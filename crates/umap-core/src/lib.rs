//! Core crate for the independent Rust implementation of UMAP.

pub mod fuzzy_graph;
pub mod initialization;
pub mod layout;
pub mod neighbors;
pub mod rng;

/// Version of the Rust implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
