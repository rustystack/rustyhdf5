//! Virtual Object Layer (VOL) abstraction for pluggable HDF5 backends.
//!
//! The VOL trait provides a backend-agnostic interface for HDF5 file I/O,
//! enabling seamless integration with different storage systems such as
//! local filesystems, S3/object stores, DAOS, and HSDS.
//!
//! # Architecture
//!
//! The VOL separates the HDF5 API from the underlying storage:
//!
//! ```text
//! ┌───────────────────────────┐
//! │ rustyhdf5 High-level API  │
//! ├───────────────────────────┤
//! │   VirtualObjectLayer      │  ← trait defined here
//! ├───────┬───────┬───────────┤
//! │ Local │  S3   │   DAOS    │  ← pluggable backends
//! └───────┴───────┴───────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use rustyhdf5_io::vol::{VirtualObjectLayer, NativeVol};
//!
//! let vol = NativeVol::open("data.h5").unwrap();
//! let data = vol.read_dataset("sensors/temperature").unwrap();
//! ```

use std::collections::HashMap;
use std::fmt;
use std::io;

/// Capabilities that a VOL connector supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolCapability {
    /// Can read existing datasets.
    ReadData,
    /// Can write new datasets.
    WriteData,
    /// Can list groups and datasets.
    ListObjects,
    /// Can read/write attributes.
    Attributes,
    /// Supports chunked storage.
    ChunkedStorage,
    /// Supports parallel I/O.
    ParallelIO,
    /// Supports partial/selection reads.
    SelectionRead,
    /// Supports SWMR mode.
    Swmr,
}

/// Error type for VOL operations.
#[derive(Debug)]
pub enum VolError {
    /// I/O error from the underlying storage.
    Io(io::Error),
    /// The requested operation is not supported by this VOL connector.
    Unsupported(String),
    /// Object (dataset, group, attribute) not found.
    NotFound(String),
    /// Data format or parsing error.
    DataError(String),
}

impl fmt::Display for VolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VolError::Io(e) => write!(f, "VOL I/O error: {e}"),
            VolError::Unsupported(op) => write!(f, "VOL operation not supported: {op}"),
            VolError::NotFound(name) => write!(f, "VOL object not found: {name}"),
            VolError::DataError(msg) => write!(f, "VOL data error: {msg}"),
        }
    }
}

impl std::error::Error for VolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VolError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for VolError {
    fn from(e: io::Error) -> Self {
        VolError::Io(e)
    }
}

/// Information about a dataset within a VOL connector.
#[derive(Debug, Clone)]
pub struct VolDatasetInfo {
    /// Dataset path within the file.
    pub path: String,
    /// Shape (dimensions).
    pub shape: Vec<u64>,
    /// Element type description (e.g., "f64", "i32", "compound{...}").
    pub dtype: String,
    /// Total data size in bytes.
    pub size_bytes: u64,
}

/// Information about a group within a VOL connector.
#[derive(Debug, Clone)]
pub struct VolGroupInfo {
    /// Group path.
    pub path: String,
    /// Number of child datasets.
    pub num_datasets: usize,
    /// Number of child groups.
    pub num_groups: usize,
}

/// Virtual Object Layer trait for pluggable storage backends.
///
/// Implement this trait to add support for a new storage backend.
/// Each method has a default implementation that returns `Unsupported`,
/// so connectors only need to implement the methods they support.
pub trait VirtualObjectLayer: Send + Sync {
    /// Returns the connector name (e.g., "native", "s3", "daos").
    fn name(&self) -> &str;

    /// Returns the set of capabilities this connector supports.
    fn capabilities(&self) -> Vec<VolCapability>;

    /// Check if a specific capability is supported.
    fn supports(&self, cap: VolCapability) -> bool {
        self.capabilities().contains(&cap)
    }

    /// Open a file/container at the given location.
    ///
    /// The `location` interpretation depends on the backend:
    /// - Native: filesystem path
    /// - S3: `s3://bucket/key`
    /// - HSDS: `http://host:port/path`
    fn open(&mut self, location: &str) -> Result<(), VolError>;

    /// Close the current file/container.
    fn close(&mut self) -> Result<(), VolError> {
        Ok(())
    }

    /// Read raw bytes for a dataset at the given path.
    fn read_dataset(&self, path: &str) -> Result<Vec<u8>, VolError> {
        Err(VolError::Unsupported(format!(
            "{}: read_dataset not supported",
            self.name()
        )))
    }

    /// Get information about a dataset.
    fn dataset_info(&self, path: &str) -> Result<VolDatasetInfo, VolError> {
        Err(VolError::Unsupported(format!(
            "{}: dataset_info not supported",
            self.name()
        )))
    }

    /// List datasets in a group.
    fn list_datasets(&self, group_path: &str) -> Result<Vec<String>, VolError> {
        Err(VolError::Unsupported(format!(
            "{}: list_datasets not supported",
            self.name()
        )))
    }

    /// List subgroups in a group.
    fn list_groups(&self, group_path: &str) -> Result<Vec<String>, VolError> {
        Err(VolError::Unsupported(format!(
            "{}: list_groups not supported",
            self.name()
        )))
    }

    /// Read attributes for an object (dataset or group).
    fn read_attributes(&self, path: &str) -> Result<HashMap<String, Vec<u8>>, VolError> {
        Err(VolError::Unsupported(format!(
            "{}: read_attributes not supported",
            self.name()
        )))
    }

    /// Write a dataset at the given path.
    fn write_dataset(
        &mut self,
        path: &str,
        data: &[u8],
        shape: &[u64],
        dtype: &str,
    ) -> Result<(), VolError> {
        Err(VolError::Unsupported(format!(
            "{}: write_dataset not supported",
            self.name()
        )))
    }
}

/// Native (local filesystem) VOL connector.
///
/// Reads and writes HDF5 files on the local filesystem using the
/// rustyhdf5-format library for parsing.
#[derive(Debug)]
pub struct NativeVol {
    data: Option<Vec<u8>>,
    location: Option<String>,
}

impl NativeVol {
    /// Create a new native VOL connector.
    pub fn new() -> Self {
        Self {
            data: None,
            location: None,
        }
    }

    /// Create and open a native VOL connector from a file path.
    pub fn open_path(path: &str) -> Result<Self, VolError> {
        let mut vol = Self::new();
        vol.open(path)?;
        Ok(vol)
    }

    /// Create a native VOL connector from bytes already in memory.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            data: Some(data),
            location: Some("<memory>".into()),
        }
    }

    /// Access the raw file bytes, if loaded.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }
}

impl Default for NativeVol {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualObjectLayer for NativeVol {
    fn name(&self) -> &str {
        "native"
    }

    fn capabilities(&self) -> Vec<VolCapability> {
        vec![
            VolCapability::ReadData,
            VolCapability::WriteData,
            VolCapability::ListObjects,
            VolCapability::Attributes,
            VolCapability::ChunkedStorage,
            VolCapability::SelectionRead,
        ]
    }

    fn open(&mut self, location: &str) -> Result<(), VolError> {
        let data = std::fs::read(location)?;
        self.data = Some(data);
        self.location = Some(location.to_string());
        Ok(())
    }

    fn close(&mut self) -> Result<(), VolError> {
        self.data = None;
        self.location = None;
        Ok(())
    }

    fn read_dataset(&self, path: &str) -> Result<Vec<u8>, VolError> {
        let data = self.data.as_ref().ok_or_else(|| {
            VolError::Io(io::Error::new(io::ErrorKind::NotConnected, "file not open"))
        })?;

        use rustyhdf5_format::{
            data_layout::DataLayout, data_read::read_raw_data_full, dataspace::Dataspace,
            datatype::Datatype, filter_pipeline::FilterPipeline, group_v2::resolve_path_any,
            message_type::MessageType, object_header::ObjectHeader, signature::find_signature,
            superblock::Superblock,
        };

        let sig = find_signature(data).map_err(|e| VolError::DataError(e.to_string()))?;
        let sb = Superblock::parse(data, sig).map_err(|e| VolError::DataError(e.to_string()))?;
        let addr = resolve_path_any(data, &sb, path)
            .map_err(|e| VolError::NotFound(format!("{path}: {e}")))?;

        let header = ObjectHeader::parse(data, addr as usize, sb.offset_size, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;

        let dt_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or_else(|| VolError::DataError("missing datatype".into()))?;
        let (datatype, _) =
            Datatype::parse(&dt_msg.data).map_err(|e| VolError::DataError(e.to_string()))?;

        let ds_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .ok_or_else(|| VolError::DataError("missing dataspace".into()))?;
        let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;

        let dl_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .ok_or_else(|| VolError::DataError("missing data layout".into()))?;
        let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size)
            .map_err(|e| VolError::DataError(e.to_string()))?;

        let pipeline = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

        read_raw_data_full(
            data,
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            sb.offset_size,
            sb.length_size,
        )
        .map_err(|e| VolError::DataError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_vol_new() {
        let vol = NativeVol::new();
        assert_eq!(vol.name(), "native");
        assert!(vol.supports(VolCapability::ReadData));
        assert!(!vol.supports(VolCapability::Swmr));
    }

    #[test]
    fn native_vol_from_bytes() {
        use rustyhdf5_format::file_writer::FileWriter as FmtWriter;

        let mut fw = FmtWriter::new();
        fw.create_dataset("test").with_f64_data(&[1.0, 2.0, 3.0]);
        let bytes = fw.finish().unwrap();

        let vol = NativeVol::from_bytes(bytes);
        assert!(vol.as_bytes().is_some());

        let raw = vol.read_dataset("test").unwrap();
        assert_eq!(raw.len(), 3 * 8); // 3 f64 values
    }

    #[test]
    fn native_vol_not_open_error() {
        let vol = NativeVol::new();
        let result = vol.read_dataset("test");
        assert!(result.is_err());
    }

    #[test]
    fn native_vol_close() {
        use rustyhdf5_format::file_writer::FileWriter as FmtWriter;

        let mut fw = FmtWriter::new();
        fw.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = fw.finish().unwrap();

        let mut vol = NativeVol::from_bytes(bytes);
        assert!(vol.as_bytes().is_some());
        vol.close().unwrap();
        assert!(vol.as_bytes().is_none());
    }

    #[test]
    fn vol_error_display() {
        let err = VolError::Unsupported("read_dataset".into());
        assert!(err.to_string().contains("not supported"));

        let err = VolError::NotFound("missing".into());
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn vol_capabilities() {
        let vol = NativeVol::new();
        let caps = vol.capabilities();
        assert!(caps.contains(&VolCapability::ReadData));
        assert!(caps.contains(&VolCapability::WriteData));
        assert!(caps.contains(&VolCapability::ListObjects));
    }
}
