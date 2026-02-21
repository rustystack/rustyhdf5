//! Pure-Rust HDF5 binary format parsing and writing.
//!
//! `purehdf5-format` is a zero-dependency (no C libhdf5) crate for reading and
//! writing HDF5 files.  It supports `no_std` environments with the `alloc` crate.
//!
//! # Writing files
//!
//! Use [`file_writer::FileWriter`] to create HDF5 files:
//!
//! ```rust
//! use purehdf5_format::file_writer::{FileWriter, AttrValue};
//!
//! let mut fw = FileWriter::new();
//! fw.create_dataset("data")
//!     .with_f64_data(&[1.0, 2.0, 3.0])
//!     .with_shape(&[3])
//!     .set_attr("unit", AttrValue::String("m/s".into()));
//! let bytes = fw.finish().unwrap();
//! ```
//!
//! # Reading files
//!
//! Parsing follows the HDF5 object model: superblock → object header → messages.
//!
//! ```rust,no_run
//! use purehdf5_format::{signature, superblock, object_header, group_v2,
//!     datatype, dataspace, data_layout, data_read, message_type::MessageType};
//!
//! let file_data = std::fs::read("output.h5").unwrap();
//! let sig = signature::find_signature(&file_data).unwrap();
//! let sb  = superblock::Superblock::parse(&file_data, sig).unwrap();
//! let addr = group_v2::resolve_path_any(&file_data, &sb, "data").unwrap();
//! let hdr = object_header::ObjectHeader::parse(
//!     &file_data, addr as usize, sb.offset_size, sb.length_size).unwrap();
//! ```
//!
//! # Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | yes | Standard library support |
//! | `checksum` | yes | Jenkins lookup3 checksum validation |
//! | `deflate` | yes | Deflate (gzip) compression via `flate2` |
//! | `provenance` | yes | SHINES provenance — SHA-256 hashing & verification |

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod attribute;
pub mod attribute_info;
pub mod chunk_cache;
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
#[cfg(feature = "parallel")]
pub mod lane_partition;
#[cfg(feature = "parallel")]
pub mod parallel_read;
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
