//! Writing API: FileBuilder and GroupBuilder for creating HDF5 files.

use rustyhdf5_format::file_writer::FileWriter as FormatWriter;
use rustyhdf5_format::type_builders::{
    AttrValue, DatasetBuilder as FormatDatasetBuilder, FinishedGroup,
    GroupBuilder as FormatGroupBuilder,
};

use crate::error::Error;

#[cfg(feature = "parallel")]
use rustyhdf5_format::chunked_write::ChunkOptions;
#[cfg(feature = "parallel")]
use rustyhdf5_format::datatype::Datatype;
#[cfg(feature = "parallel")]
use rustyhdf5_format::file_writer::IndependentDatasetBuilder;
#[cfg(feature = "parallel")]
use rustyhdf5_format::metadata_index::{build_dataset_metadata, MetadataBlock};

/// Builder for creating a new HDF5 file.
///
/// # Example
///
/// ```no_run
/// use rustyhdf5::FileBuilder;
/// use rustyhdf5::AttrValue;
///
/// let mut builder = FileBuilder::new();
/// builder.create_dataset("data").with_f64_data(&[1.0, 2.0, 3.0]);
/// builder.set_attr("version", AttrValue::I64(1));
/// builder.write("output.h5").unwrap();
/// ```
pub struct FileBuilder {
    writer: FormatWriter,
}

impl FileBuilder {
    /// Create a new file builder.
    pub fn new() -> Self {
        Self {
            writer: FormatWriter::new(),
        }
    }

    /// Create a dataset at the root level. Returns a mutable reference to
    /// a `DatasetBuilder` for configuring data, shape, and attributes.
    pub fn create_dataset(&mut self, name: &str) -> &mut FormatDatasetBuilder {
        self.writer.create_dataset(name)
    }

    /// Create a group builder. Call `.finish()` on the returned builder
    /// to complete it, then pass to `add_group()`.
    pub fn create_group(&mut self, name: &str) -> FormatGroupBuilder {
        self.writer.create_group(name)
    }

    /// Add a finished group to the file.
    pub fn add_group(&mut self, group: FinishedGroup) {
        self.writer.add_group(group);
    }

    /// Set an attribute on the root group.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        self.writer.set_root_attr(name, value);
    }

    /// Serialize the file to bytes in memory.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.writer.finish()?)
    }

    /// Serialize and write the file to the given path.
    pub fn write<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), Error> {
        let bytes = self.finish()?;
        std::fs::write(path, bytes).map_err(Error::Io)
    }
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Parallel dataset creation API ----

/// Specification for a dataset to be created in parallel.
///
/// Used with [`create_datasets_parallel`] to create many datasets concurrently
/// using independent metadata blocks and rayon.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct DatasetSpec {
    /// Dataset name (used as the root-level link name).
    pub name: String,
    /// HDF5 datatype for the dataset.
    pub datatype: Datatype,
    /// Shape of the dataset.
    pub shape: Vec<u64>,
    /// Raw data bytes (little-endian).
    pub raw_data: Vec<u8>,
    /// Chunk options (leave default for contiguous storage).
    pub chunk_options: ChunkOptions,
    /// Optional maximum shape for resizable datasets.
    pub maxshape: Option<Vec<u64>>,
    /// Attributes to attach to the dataset.
    pub attrs: Vec<(String, AttrValue)>,
}

#[cfg(feature = "parallel")]
impl DatasetSpec {
    /// Create a simple f64 dataset spec.
    pub fn f64(name: &str, data: &[f64]) -> Self {
        let mut raw = Vec::with_capacity(data.len() * 8);
        for &v in data {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.to_string(),
            datatype: rustyhdf5_format::type_builders::make_f64_type(),
            shape: vec![data.len() as u64],
            raw_data: raw,
            chunk_options: ChunkOptions::default(),
            maxshape: None,
            attrs: Vec::new(),
        }
    }

    /// Create a simple i32 dataset spec.
    pub fn i32(name: &str, data: &[i32]) -> Self {
        let mut raw = Vec::with_capacity(data.len() * 4);
        for &v in data {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.to_string(),
            datatype: rustyhdf5_format::type_builders::make_i32_type(),
            shape: vec![data.len() as u64],
            raw_data: raw,
            chunk_options: ChunkOptions::default(),
            maxshape: None,
            attrs: Vec::new(),
        }
    }

    /// Add an attribute to this spec.
    pub fn with_attr(mut self, name: &str, value: AttrValue) -> Self {
        self.attrs.push((name.to_string(), value));
        self
    }
}

/// Create multiple datasets in parallel using independent metadata blocks.
///
/// Uses rayon to build metadata blocks concurrently, then merges them into a
/// single valid HDF5 file. Each dataset gets its own independent metadata block,
/// avoiding contention on the file header.
///
/// Returns the serialized HDF5 file bytes.
#[cfg(feature = "parallel")]
pub fn create_datasets_parallel(specs: Vec<DatasetSpec>) -> Result<Vec<u8>, Error> {
    use rayon::prelude::*;

    // Phase 1: Build metadata blocks in parallel (one per dataset).
    // Each spec gets its own IndependentDatasetBuilder on its own "creator".
    let blocks: Vec<MetadataBlock> = specs
        .into_par_iter()
        .enumerate()
        .map(|(i, spec)| {
            let mut builder = IndependentDatasetBuilder::new(i as u32);
            let meta = build_dataset_metadata(
                &spec.name,
                spec.datatype,
                spec.shape,
                spec.raw_data,
                spec.chunk_options,
                spec.maxshape,
                spec.attrs,
            );
            builder.add_dataset(meta);
            builder.finish()
        })
        .collect();

    // Phase 2: Merge blocks and finalize into an HDF5 file.
    let bytes = rustyhdf5_format::file_writer::finalize_parallel(blocks)?;
    Ok(bytes)
}
