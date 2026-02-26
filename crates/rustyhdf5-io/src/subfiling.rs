//! Sub-filing support for extreme-scale parallel I/O.
//!
//! Sub-filing splits a single logical HDF5 file into multiple physical
//! files (sub-files), one per node or I/O aggregator. This avoids
//! contention on a single file system object and enables each node to
//! write independently.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────┐
//! │  Logical HDF5 File           │
//! │  ┌────────┬────────┬───────┐ │
//! │  │ Sub 0  │ Sub 1  │ Sub 2 │ │  ← stripe_count = 3
//! │  │ node-0 │ node-1 │ node-2│ │
//! │  └────────┴────────┴───────┘ │
//! │  stripe_size = 32 MiB        │
//! └──────────────────────────────┘
//! ```
//!
//! Data is distributed across sub-files in a round-robin fashion with
//! a configurable stripe size.
//!
//! # Status
//!
//! This module provides the configuration types and address mapping logic.
//! Full integration with the file writer requires additional work to
//! coordinate metadata across sub-files.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// Default stripe size: 32 MiB.
pub const DEFAULT_STRIPE_SIZE: u64 = 32 * 1024 * 1024;

/// Configuration for sub-filing.
#[derive(Debug, Clone)]
pub struct SubfileConfig {
    /// Number of sub-files (typically one per I/O aggregator or node).
    pub stripe_count: u32,
    /// Size of each stripe in bytes (data is round-robin'd across sub-files).
    pub stripe_size: u64,
    /// Base path for sub-files. Sub-files are named `{base}.subfile.{N}`.
    pub base_path: PathBuf,
}

impl SubfileConfig {
    /// Create a new sub-file configuration.
    pub fn new(base_path: impl AsRef<Path>, stripe_count: u32) -> Self {
        Self {
            stripe_count,
            stripe_size: DEFAULT_STRIPE_SIZE,
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Set the stripe size.
    pub fn with_stripe_size(mut self, size: u64) -> Self {
        self.stripe_size = size;
        self
    }

    /// Generate the path for a specific sub-file index.
    pub fn subfile_path(&self, index: u32) -> PathBuf {
        let base = self.base_path.display();
        PathBuf::from(format!("{base}.subfile.{index}"))
    }

    /// Generate paths for all sub-files.
    pub fn all_subfile_paths(&self) -> Vec<PathBuf> {
        (0..self.stripe_count)
            .map(|i| self.subfile_path(i))
            .collect()
    }

    /// Map a logical file offset to a (subfile_index, local_offset) pair.
    ///
    /// The mapping is round-robin: offset 0..stripe_size goes to sub-file 0,
    /// stripe_size..2*stripe_size goes to sub-file 1, etc.
    pub fn map_offset(&self, logical_offset: u64) -> (u32, u64) {
        let stripe_index = logical_offset / self.stripe_size;
        let subfile_index = (stripe_index % self.stripe_count as u64) as u32;
        let stripe_in_subfile = stripe_index / self.stripe_count as u64;
        let offset_in_stripe = logical_offset % self.stripe_size;
        let local_offset = stripe_in_subfile * self.stripe_size + offset_in_stripe;
        (subfile_index, local_offset)
    }

    /// Map a logical byte range to a set of (subfile_index, local_offset, length) operations.
    ///
    /// A single logical range may span multiple sub-files, so this returns
    /// a vector of I/O operations, each targeting a specific sub-file.
    pub fn map_range(&self, logical_offset: u64, length: u64) -> Vec<SubfileIo> {
        let mut ops = Vec::new();
        let mut remaining = length;
        let mut offset = logical_offset;

        while remaining > 0 {
            let (subfile_idx, local_off) = self.map_offset(offset);
            // How many bytes until the end of the current stripe?
            let offset_in_stripe = offset % self.stripe_size;
            let bytes_in_stripe = (self.stripe_size - offset_in_stripe).min(remaining);

            ops.push(SubfileIo {
                subfile_index: subfile_idx,
                local_offset: local_off,
                length: bytes_in_stripe,
            });

            offset += bytes_in_stripe;
            remaining -= bytes_in_stripe;
        }

        ops
    }
}

/// A single I/O operation targeting a specific sub-file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubfileIo {
    /// Index of the sub-file (0-based).
    pub subfile_index: u32,
    /// Offset within the sub-file.
    pub local_offset: u64,
    /// Number of bytes for this operation.
    pub length: u64,
}

impl fmt::Display for SubfileIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubfileIo(file={}, offset={}, len={})",
            self.subfile_index, self.local_offset, self.length
        )
    }
}

/// Manager for sub-file I/O operations.
///
/// Coordinates reads and writes across multiple sub-files.
#[derive(Debug)]
pub struct SubfileManager {
    config: SubfileConfig,
}

impl SubfileManager {
    /// Create a new sub-file manager with the given configuration.
    pub fn new(config: SubfileConfig) -> Self {
        Self { config }
    }

    /// Access the configuration.
    pub fn config(&self) -> &SubfileConfig {
        &self.config
    }

    /// Create all sub-files on disk (empty files).
    pub fn create_subfiles(&self) -> io::Result<()> {
        for path in self.config.all_subfile_paths() {
            std::fs::File::create(path)?;
        }
        Ok(())
    }

    /// Remove all sub-files from disk.
    pub fn remove_subfiles(&self) -> io::Result<()> {
        for path in self.config.all_subfile_paths() {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    /// Check if all sub-files exist.
    pub fn all_subfiles_exist(&self) -> bool {
        self.config.all_subfile_paths().iter().all(|p| p.exists())
    }

    /// Write data at a logical offset, distributing across sub-files.
    pub fn write_at(&self, logical_offset: u64, data: &[u8]) -> io::Result<()> {
        use std::io::Write;

        let ops = self.config.map_range(logical_offset, data.len() as u64);
        let mut data_offset = 0usize;

        for op in &ops {
            let path = self.config.subfile_path(op.subfile_index);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(path)?;
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(op.local_offset))?;
            let end = data_offset + op.length as usize;
            file.write_all(&data[data_offset..end])?;
            data_offset = end;
        }

        Ok(())
    }

    /// Read data from a logical offset, gathering from sub-files.
    pub fn read_at(&self, logical_offset: u64, length: u64) -> io::Result<Vec<u8>> {
        use std::io::Read;

        let ops = self.config.map_range(logical_offset, length);
        let mut result = Vec::with_capacity(length as usize);

        for op in &ops {
            let path = self.config.subfile_path(op.subfile_index);
            let mut file = std::fs::File::open(path)?;
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(op.local_offset))?;
            let mut buf = vec![0u8; op.length as usize];
            file.read_exact(&mut buf)?;
            result.extend_from_slice(&buf);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subfile_path_generation() {
        let config = SubfileConfig::new("/tmp/data.h5", 3);
        assert_eq!(
            config.subfile_path(0),
            PathBuf::from("/tmp/data.h5.subfile.0")
        );
        assert_eq!(
            config.subfile_path(2),
            PathBuf::from("/tmp/data.h5.subfile.2")
        );
    }

    #[test]
    fn all_subfile_paths() {
        let config = SubfileConfig::new("/tmp/data.h5", 2);
        let paths = config.all_subfile_paths();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], PathBuf::from("/tmp/data.h5.subfile.0"));
        assert_eq!(paths[1], PathBuf::from("/tmp/data.h5.subfile.1"));
    }

    #[test]
    fn map_offset_single_stripe() {
        let config = SubfileConfig::new("/tmp/data.h5", 3).with_stripe_size(1024);

        // Offset 0 -> sub-file 0, local offset 0
        assert_eq!(config.map_offset(0), (0, 0));

        // Offset 512 -> sub-file 0, local offset 512
        assert_eq!(config.map_offset(512), (0, 512));

        // Offset 1024 -> sub-file 1, local offset 0
        assert_eq!(config.map_offset(1024), (1, 0));

        // Offset 2048 -> sub-file 2, local offset 0
        assert_eq!(config.map_offset(2048), (2, 0));

        // Offset 3072 -> sub-file 0 again (wrap), local offset 1024
        assert_eq!(config.map_offset(3072), (0, 1024));
    }

    #[test]
    fn map_range_within_single_stripe() {
        let config = SubfileConfig::new("/tmp/data.h5", 2).with_stripe_size(1024);

        let ops = config.map_range(0, 512);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].subfile_index, 0);
        assert_eq!(ops[0].local_offset, 0);
        assert_eq!(ops[0].length, 512);
    }

    #[test]
    fn map_range_spanning_stripes() {
        let config = SubfileConfig::new("/tmp/data.h5", 2).with_stripe_size(1024);

        // Reading 1536 bytes starting at offset 512 should span two stripes
        let ops = config.map_range(512, 1536);
        assert_eq!(ops.len(), 2);
        // First: sub-file 0, offset 512, length 512 (fills rest of stripe 0)
        assert_eq!(ops[0].subfile_index, 0);
        assert_eq!(ops[0].local_offset, 512);
        assert_eq!(ops[0].length, 512);
        // Second: sub-file 1, offset 0, length 1024 (entire stripe 1)
        assert_eq!(ops[1].subfile_index, 1);
        assert_eq!(ops[1].local_offset, 0);
        assert_eq!(ops[1].length, 1024);
    }

    #[test]
    fn map_range_empty() {
        let config = SubfileConfig::new("/tmp/data.h5", 2).with_stripe_size(1024);
        let ops = config.map_range(0, 0);
        assert!(ops.is_empty());
    }

    #[test]
    fn subfile_io_display() {
        let io = SubfileIo {
            subfile_index: 1,
            local_offset: 4096,
            length: 1024,
        };
        let s = io.to_string();
        assert!(s.contains("file=1"));
        assert!(s.contains("offset=4096"));
        assert!(s.contains("len=1024"));
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = std::env::temp_dir();
        let base = dir.join("rustyhdf5_subfiling_test");

        let config = SubfileConfig::new(&base, 2).with_stripe_size(16);
        let mgr = SubfileManager::new(config);

        // Create sub-files
        mgr.create_subfiles().unwrap();
        assert!(mgr.all_subfiles_exist());

        // Write data that spans multiple sub-files
        let data: Vec<u8> = (0..64).collect();
        mgr.write_at(0, &data).unwrap();

        // Read it back
        let read_back = mgr.read_at(0, 64).unwrap();
        assert_eq!(read_back, data);

        // Read a partial range
        let partial = mgr.read_at(8, 16).unwrap();
        assert_eq!(partial, &data[8..24]);

        // Cleanup
        mgr.remove_subfiles().unwrap();
    }

    #[test]
    fn default_stripe_size() {
        let config = SubfileConfig::new("/tmp/data.h5", 4);
        assert_eq!(config.stripe_size, 32 * 1024 * 1024);
    }
}
