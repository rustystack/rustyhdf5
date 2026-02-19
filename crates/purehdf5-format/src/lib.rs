//! Pure-Rust HDF5 binary format parsing.
//!
//! This crate provides low-level parsing of HDF5 file format structures.
//! It supports `no_std` environments with the `alloc` crate.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod data_layout;
pub mod data_read;
pub mod dataspace;
pub mod datatype;
pub mod error;
pub mod message_type;
pub mod object_header;
pub mod signature;
pub mod superblock;
