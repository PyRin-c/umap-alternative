//! Core crate for the independent Rust implementation of UMAP.

pub mod aligned;
pub mod approximate_neighbors;
pub mod density;
pub mod densmap;
pub mod embedding;
pub mod fit;
pub mod fuzzy_graph;
pub mod initialization;
pub mod inverse;
pub mod layout;
pub mod modern_initialization;
pub mod modern_layout;
pub mod neighbors;
mod numeric;
pub mod optimize;
pub mod packed;
pub mod parametric;
pub mod rng;
pub mod sampling;
pub mod sparse;
pub mod transform;
pub mod update;
pub mod update_layout;
pub mod validation;

/// Version of the Rust implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
