//! Core crate for the independent Rust implementation of UMAP.

pub mod density;
pub mod embedding;
pub mod fit;
pub mod fuzzy_graph;
pub mod initialization;
pub mod inverse;
pub mod layout;
pub mod neighbors;
pub mod optimize;
pub mod rng;
pub mod sampling;
pub mod sparse;
pub mod transform;
pub mod update;

/// Version of the Rust implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
