//! Pure-Rust HDF5 binary format parsing.
//!
//! This crate provides low-level parsing of HDF5 file format structures.
//! It supports `no_std` environments with the `alloc` crate.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod attribute;
pub mod attribute_info;
pub mod chunked_read;
pub mod chunked_write;
pub mod file_writer;
pub mod object_header_writer;
pub mod type_builders;
pub mod btree_v1;
pub mod checksum;
pub mod btree_v2;
pub mod fractal_heap;
pub mod group_info;
pub mod group_v2;
pub mod link_info;
pub mod link_message;
pub mod data_layout;
pub mod data_read;
pub mod filter_pipeline;
pub mod extensible_array;
pub mod fixed_array;
pub mod filters;
pub mod dataspace;
pub mod datatype;
pub mod error;
pub mod global_heap;
pub mod group_v1;
pub mod local_heap;
pub mod message_type;
pub mod object_header;
pub mod shared_message;
pub mod signature;
pub mod superblock;
pub mod symbol_table;
pub mod vl_data;

#[cfg(feature = "provenance")]
pub mod provenance;
