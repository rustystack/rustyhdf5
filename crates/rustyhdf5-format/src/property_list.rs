//! Property list types for configuring dataset, file access, and file creation.
//!
//! Property lists group related configuration into reusable bundles.
//! They provide an alternative to individual setter methods on builders.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::type_builders::FillTime;

/// Dataset creation properties.
///
/// Controls storage layout, compression, fill values, and alignment
/// for a new dataset.
#[derive(Debug, Clone)]
pub struct DatasetCreateProps {
    /// Chunk dimensions (enables chunked storage).
    pub chunk_dims: Option<Vec<u64>>,
    /// Deflate compression level (0-9).
    pub deflate_level: Option<u32>,
    /// Shuffle filter before compression.
    pub shuffle: bool,
    /// Fletcher32 checksum.
    pub fletcher32: bool,
    /// LZ4 compression.
    pub lz4: bool,
    /// Zstandard compression level (1-22).
    pub zstd_level: Option<u32>,
    /// Fill value write time.
    pub fill_time: FillTime,
    /// Use compact (inline) storage.
    pub compact: bool,
    /// Per-dataset alignment in bytes.
    pub alignment: usize,
}

impl Default for DatasetCreateProps {
    fn default() -> Self {
        Self {
            chunk_dims: None,
            deflate_level: None,
            shuffle: false,
            fletcher32: false,
            lz4: false,
            zstd_level: None,
            fill_time: FillTime::Alloc,
            compact: false,
            alignment: 0,
        }
    }
}

impl DatasetCreateProps {
    /// Create default dataset creation properties.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk dimensions.
    pub fn chunk(mut self, dims: &[u64]) -> Self {
        self.chunk_dims = Some(dims.to_vec());
        self
    }

    /// Set deflate compression level (0-9).
    pub fn deflate(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Enable shuffle filter.
    pub fn shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    /// Enable fletcher32 checksum.
    pub fn fletcher32(mut self) -> Self {
        self.fletcher32 = true;
        self
    }

    /// Enable LZ4 compression.
    pub fn lz4(mut self) -> Self {
        self.lz4 = true;
        self
    }

    /// Set Zstandard compression level (1-22).
    pub fn zstd(mut self, level: u32) -> Self {
        self.zstd_level = Some(level);
        self
    }

    /// Set fill time policy.
    pub fn fill_time(mut self, ft: FillTime) -> Self {
        self.fill_time = ft;
        self
    }

    /// Use compact (inline) storage.
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Set per-dataset alignment in bytes.
    pub fn align(mut self, bytes: usize) -> Self {
        self.alignment = bytes;
        self
    }
}

/// File access properties.
///
/// Controls caching, alignment, and I/O behavior when reading files.
#[derive(Debug, Clone)]
pub struct FileAccessProps {
    /// Maximum bytes for the chunk cache.
    pub chunk_cache_bytes: usize,
    /// Maximum slots in the chunk cache.
    pub chunk_cache_slots: usize,
    /// Maximum bytes for the metadata cache.
    pub metadata_cache_bytes: usize,
    /// Global alignment threshold.
    pub alignment_threshold: usize,
    /// Global alignment bytes.
    pub alignment_bytes: usize,
    /// Sieve buffer size for raw data I/O (default: 64 KiB).
    ///
    /// The sieve buffer aggregates small contiguous reads/writes into
    /// fewer, larger I/O operations.
    pub sieve_buffer_size: usize,
    /// Metadata block allocation size (default: 2 KiB).
    ///
    /// Metadata is allocated in blocks of this size to reduce fragmentation.
    pub metadata_block_size: usize,
    /// Library version bounds: (low, high).
    ///
    /// Controls which HDF5 format features are used.
    /// - 0 = earliest (most compatible)
    /// - 1 = v18 (HDF5 1.8+)
    /// - 2 = v110 (HDF5 1.10+)
    /// - 3 = v112 (HDF5 1.12+)
    /// - 4 = latest (newest features)
    pub lib_version_bounds: (u8, u8),
}

/// Library version bound constants.
pub mod lib_version {
    /// Earliest version — maximum compatibility.
    pub const EARLIEST: u8 = 0;
    /// HDF5 1.8 features.
    pub const V18: u8 = 1;
    /// HDF5 1.10 features (SWMR, paged aggregation).
    pub const V110: u8 = 2;
    /// HDF5 1.12 features.
    pub const V112: u8 = 3;
    /// Latest version — all features.
    pub const LATEST: u8 = 4;
}

impl Default for FileAccessProps {
    fn default() -> Self {
        Self {
            chunk_cache_bytes: crate::chunk_cache::DEFAULT_CACHE_BYTES,
            chunk_cache_slots: crate::chunk_cache::DEFAULT_MAX_SLOTS,
            metadata_cache_bytes: 2 * 1024 * 1024, // 2 MiB default
            alignment_threshold: 0,
            alignment_bytes: 0,
            sieve_buffer_size: 64 * 1024, // 64 KiB
            metadata_block_size: 2048, // 2 KiB
            lib_version_bounds: (lib_version::EARLIEST, lib_version::LATEST),
        }
    }
}

impl FileAccessProps {
    /// Create default file access properties.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk cache size and slot count.
    pub fn chunk_cache(mut self, bytes: usize, slots: usize) -> Self {
        self.chunk_cache_bytes = bytes;
        self.chunk_cache_slots = slots;
        self
    }

    /// Set metadata cache size.
    pub fn metadata_cache(mut self, bytes: usize) -> Self {
        self.metadata_cache_bytes = bytes;
        self
    }

    /// Set global alignment.
    pub fn alignment(mut self, threshold: usize, bytes: usize) -> Self {
        self.alignment_threshold = threshold;
        self.alignment_bytes = bytes;
        self
    }

    /// Set sieve buffer size for raw data I/O aggregation.
    pub fn sieve_buffer(mut self, bytes: usize) -> Self {
        self.sieve_buffer_size = bytes;
        self
    }

    /// Set metadata block allocation size.
    pub fn metadata_block(mut self, bytes: usize) -> Self {
        self.metadata_block_size = bytes;
        self
    }

    /// Set library version bounds (low, high).
    ///
    /// Use constants from [`lib_version`] module.
    pub fn version_bounds(mut self, low: u8, high: u8) -> Self {
        self.lib_version_bounds = (low, high);
        self
    }
}

/// File creation properties.
///
/// Controls file-level format parameters.
#[derive(Debug, Clone)]
pub struct FileCreateProps {
    /// Superblock version (default: 3).
    pub superblock_version: u8,
    /// Size of offsets in bytes (default: 8).
    pub offset_size: u8,
    /// Size of lengths in bytes (default: 8).
    pub length_size: u8,
    /// B-tree v1 group leaf node K.
    pub group_leaf_node_k: u16,
    /// B-tree v1 group internal node K.
    pub group_internal_node_k: u16,
    /// B-tree v1 indexed storage internal node K.
    pub indexed_storage_internal_node_k: u16,
}

impl Default for FileCreateProps {
    fn default() -> Self {
        Self {
            superblock_version: 3,
            offset_size: 8,
            length_size: 8,
            group_leaf_node_k: 4,
            group_internal_node_k: 16,
            indexed_storage_internal_node_k: 32,
        }
    }
}

impl FileCreateProps {
    /// Create default file creation properties.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set superblock version.
    pub fn superblock_version(mut self, v: u8) -> Self {
        self.superblock_version = v;
        self
    }

    /// Set offset and length sizes.
    pub fn sizes(mut self, offset_size: u8, length_size: u8) -> Self {
        self.offset_size = offset_size;
        self.length_size = length_size;
        self
    }

    /// Set B-tree v1 group leaf node K.
    pub fn group_leaf_k(mut self, k: u16) -> Self {
        self.group_leaf_node_k = k;
        self
    }

    /// Set B-tree v1 group internal node K.
    pub fn group_internal_k(mut self, k: u16) -> Self {
        self.group_internal_node_k = k;
        self
    }

    /// Set B-tree v1 indexed storage internal node K.
    pub fn indexed_storage_k(mut self, k: u16) -> Self {
        self.indexed_storage_internal_node_k = k;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcpl_defaults() {
        let dcpl = DatasetCreateProps::new();
        assert!(dcpl.chunk_dims.is_none());
        assert_eq!(dcpl.fill_time, FillTime::Alloc);
        assert!(!dcpl.compact);
    }

    #[test]
    fn dcpl_builder_chain() {
        let dcpl = DatasetCreateProps::new()
            .chunk(&[10, 10])
            .deflate(6)
            .shuffle()
            .fill_time(FillTime::Never)
            .align(4096);
        assert_eq!(dcpl.chunk_dims, Some(vec![10, 10]));
        assert_eq!(dcpl.deflate_level, Some(6));
        assert!(dcpl.shuffle);
        assert_eq!(dcpl.fill_time, FillTime::Never);
        assert_eq!(dcpl.alignment, 4096);
    }

    #[test]
    fn fapl_defaults() {
        let fapl = FileAccessProps::new();
        assert_eq!(fapl.chunk_cache_bytes, 16 * 1024 * 1024);
        assert_eq!(fapl.chunk_cache_slots, 521);
        assert_eq!(fapl.sieve_buffer_size, 64 * 1024);
        assert_eq!(fapl.metadata_block_size, 2048);
        assert_eq!(fapl.lib_version_bounds, (lib_version::EARLIEST, lib_version::LATEST));
    }

    #[test]
    fn fapl_builder_chain() {
        let fapl = FileAccessProps::new()
            .chunk_cache(4 * 1024 * 1024, 127)
            .sieve_buffer(128 * 1024)
            .metadata_block(4096)
            .version_bounds(lib_version::V110, lib_version::LATEST);
        assert_eq!(fapl.chunk_cache_bytes, 4 * 1024 * 1024);
        assert_eq!(fapl.chunk_cache_slots, 127);
        assert_eq!(fapl.sieve_buffer_size, 128 * 1024);
        assert_eq!(fapl.metadata_block_size, 4096);
        assert_eq!(fapl.lib_version_bounds, (lib_version::V110, lib_version::LATEST));
    }

    #[test]
    fn fcpl_defaults() {
        let fcpl = FileCreateProps::new();
        assert_eq!(fcpl.superblock_version, 3);
        assert_eq!(fcpl.offset_size, 8);
        assert_eq!(fcpl.group_leaf_node_k, 4);
        assert_eq!(fcpl.group_internal_node_k, 16);
        assert_eq!(fcpl.indexed_storage_internal_node_k, 32);
    }

    #[test]
    fn fcpl_builder_chain() {
        let fcpl = FileCreateProps::new()
            .superblock_version(2)
            .sizes(4, 4)
            .group_leaf_k(8)
            .group_internal_k(32)
            .indexed_storage_k(64);
        assert_eq!(fcpl.superblock_version, 2);
        assert_eq!(fcpl.offset_size, 4);
        assert_eq!(fcpl.length_size, 4);
        assert_eq!(fcpl.group_leaf_node_k, 8);
        assert_eq!(fcpl.group_internal_node_k, 32);
        assert_eq!(fcpl.indexed_storage_internal_node_k, 64);
    }
}
