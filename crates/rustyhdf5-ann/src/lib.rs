//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
//!
//! This crate implements an HNSW index that can be serialized to/from HDF5 format
//! using the rustyhdf5 stack. The index supports cosine similarity and L2 distance.

mod hnsw;

pub use hnsw::{DistanceMetric, HnswIndex};
