//! Reading API: File, Dataset, and Group handles for reading HDF5 files.
//!
//! When the `mmap` feature is enabled (default), [`File::open`] uses
//! memory-mapped I/O for zero-copy access.  [`File::open_buffered`] provides
//! the traditional read-into-`Vec<u8>` fallback.  [`File::from_bytes`] remains
//! available for in-memory usage (tests, etc.).

use std::collections::HashMap;

use purehdf5_format::attribute::extract_attributes_full;
use purehdf5_format::chunk_cache::ChunkCache;
use purehdf5_format::data_layout::DataLayout;
use purehdf5_format::data_read;
use purehdf5_format::dataspace::Dataspace;
use purehdf5_format::datatype::Datatype;
use purehdf5_format::error::FormatError;
use purehdf5_format::filter_pipeline::FilterPipeline;
use purehdf5_format::group_v1::{self, GroupEntry};
use purehdf5_format::group_v2;
use purehdf5_format::message_type::MessageType;
use purehdf5_format::object_header::ObjectHeader;
use purehdf5_format::signature;
use purehdf5_format::superblock::Superblock;
use purehdf5_format::symbol_table::SymbolTableMessage;

use crate::error::Error;
use crate::types::{attrs_to_map, classify_datatype, AttrValue, DType};

// ---------------------------------------------------------------------------
// FileData — internal storage for either owned bytes or an mmap
// ---------------------------------------------------------------------------

/// Internal storage: either an owned `Vec<u8>` or a memory-mapped region.
enum FileData {
    Owned(Vec<u8>),
    #[cfg(feature = "mmap")]
    Mmap(purehdf5_io::MmapReader),
}

impl FileData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            FileData::Owned(v) => v,
            #[cfg(feature = "mmap")]
            FileData::Mmap(r) => r.as_bytes(),
        }
    }

    fn len(&self) -> usize {
        self.as_bytes().len()
    }
}

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// An open HDF5 file for reading.
///
/// When the `mmap` feature is enabled (the default), [`File::open`] uses
/// memory-mapped I/O so the OS page-cache serves reads with zero copies.
/// Use [`File::open_buffered`] to get the old read-into-`Vec<u8>` behaviour,
/// or [`File::from_bytes`] for fully in-memory operation.
pub struct File {
    data: FileData,
    superblock: Superblock,
    /// Per-file chunk cache shared across all dataset reads.
    chunk_cache: ChunkCache,
}

impl File {
    /// Open an HDF5 file from a filesystem path.
    ///
    /// When the `mmap` feature is enabled (default), this uses memory-mapped
    /// I/O.  Otherwise it reads the entire file into a `Vec<u8>`.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        #[cfg(feature = "mmap")]
        {
            let reader = purehdf5_io::MmapReader::open(path).map_err(Error::Io)?;
            let data_ref = reader.as_bytes();
            let sig_offset = signature::find_signature(data_ref)?;
            let superblock = Superblock::parse(data_ref, sig_offset)?;
            Ok(Self {
                data: FileData::Mmap(reader),
                superblock,
                chunk_cache: ChunkCache::new(),
            })
        }
        #[cfg(not(feature = "mmap"))]
        {
            let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
            Self::from_bytes(bytes)
        }
    }

    /// Open an HDF5 file by reading it entirely into memory.
    ///
    /// This is the pre-mmap behaviour and is useful when memory-mapping is
    /// undesirable (e.g. network filesystems, very small files, etc.).
    pub fn open_buffered<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        Self::from_bytes(bytes)
    }

    /// Open an HDF5 file from an in-memory byte vector.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        let sig_offset = signature::find_signature(&data)?;
        let superblock = Superblock::parse(&data, sig_offset)?;
        Ok(Self {
            data: FileData::Owned(data),
            superblock,
            chunk_cache: ChunkCache::new(),
        })
    }

    /// Returns a handle to the root group.
    pub fn root(&self) -> Group<'_> {
        Group {
            file: self,
            address: self.superblock.root_group_address,
        }
    }

    /// Resolve a path and return a `Dataset` handle.
    ///
    /// The path uses `/` separators (e.g., `"group1/values"`).
    pub fn dataset(&self, path: &str) -> Result<Dataset<'_>, Error> {
        let data = self.data.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        let hdr = self.parse_header(addr)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(path.to_string()));
        }
        Ok(Dataset {
            file: self,
            header: hdr,
        })
    }

    /// Resolve a path and return a `Group` handle.
    ///
    /// The path uses `/` separators (e.g., `"sensors"`).
    /// Use `"/"` or `""` for the root group.
    pub fn group(&self, path: &str) -> Result<Group<'_>, Error> {
        let data = self.data.as_bytes();
        let addr = group_v2::resolve_path_any(data, &self.superblock, path)?;
        Ok(Group {
            file: self,
            address: addr,
        })
    }

    /// Returns the raw file bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// Returns a reference to the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Returns `true` when the file is backed by memory-mapped I/O.
    pub fn is_mmap(&self) -> bool {
        match &self.data {
            FileData::Owned(_) => false,
            #[cfg(feature = "mmap")]
            FileData::Mmap(_) => true,
        }
    }

    fn parse_header(&self, address: u64) -> Result<ObjectHeader, FormatError> {
        ObjectHeader::parse(
            self.data.as_bytes(),
            address as usize,
            self.superblock.offset_size,
            self.superblock.length_size,
        )
    }

    fn offset_size(&self) -> u8 {
        self.superblock.offset_size
    }

    fn length_size(&self) -> u8 {
        self.superblock.length_size
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("File")
            .field("size", &self.data.len())
            .field("superblock_version", &self.superblock.version)
            .field("mmap", &self.is_mmap())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Group handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 group.
pub struct Group<'f> {
    file: &'f File,
    address: u64,
}

impl<'f> Group<'f> {
    /// List the names of datasets in this group.
    pub fn datasets(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if has_message(&hdr, MessageType::DataLayout) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// List the names of subgroups in this group.
    pub fn groups(&self) -> Result<Vec<String>, Error> {
        let entries = self.children()?;
        let mut names = Vec::new();
        for entry in &entries {
            let hdr = self.file.parse_header(entry.object_header_address)?;
            if is_group(&hdr) {
                names.push(entry.name.clone());
            }
        }
        Ok(names)
    }

    /// Read all attributes of this group.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let data = self.file.data.as_bytes();
        let hdr = self.file.parse_header(self.address)?;
        let attr_msgs = extract_attributes_full(
            data,
            &hdr,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            data,
            self.file.offset_size(),
            self.file.length_size(),
        ))
    }

    /// Get a dataset within this group by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        let hdr = self.file.parse_header(entry.object_header_address)?;
        if !has_message(&hdr, MessageType::DataLayout) {
            return Err(Error::NotADataset(name.to_string()));
        }
        Ok(Dataset {
            file: self.file,
            header: hdr,
        })
    }

    /// Get a subgroup within this group by name.
    pub fn group(&self, name: &str) -> Result<Group<'f>, Error> {
        let entries = self.children()?;
        let entry = entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| Error::Format(FormatError::PathNotFound(name.to_string())))?;
        Ok(Group {
            file: self.file,
            address: entry.object_header_address,
        })
    }

    fn children(&self) -> Result<Vec<GroupEntry>, Error> {
        let data = self.file.data.as_bytes();
        let hdr = self.file.parse_header(self.address)?;
        let os = self.file.offset_size();
        let ls = self.file.length_size();
        resolve_group_entries(data, &hdr, os, ls).map_err(Error::Format)
    }
}

// ---------------------------------------------------------------------------
// Dataset handle
// ---------------------------------------------------------------------------

/// A lightweight handle to an HDF5 dataset.
#[derive(Debug)]
pub struct Dataset<'f> {
    file: &'f File,
    header: ObjectHeader,
}

impl<'f> Dataset<'f> {
    /// Returns the shape (dimensions) of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>, Error> {
        let ds = self.dataspace()?;
        Ok(ds.dimensions.clone())
    }

    /// Returns the simplified datatype of the dataset.
    pub fn dtype(&self) -> Result<DType, Error> {
        let dt = self.datatype()?;
        Ok(classify_datatype(&dt))
    }

    /// Read all data as `f64` values.
    pub fn read_f64(&self) -> Result<Vec<f64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64(&raw, &dt)?)
    }

    /// Zero-copy read of contiguous native-endian `f64` data.
    ///
    /// Returns `Some(&[f64])` when the dataset is contiguous and stored as
    /// little-endian `f64` with proper alignment.  Returns `None` for
    /// chunked/compact layouts or non-native types — use [`read_f64`] instead.
    pub fn read_f64_zerocopy(&self) -> Result<Option<&'f [f64]>, Error> {
        let raw = match self.read_raw_ref()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let dt = self.datatype()?;
        Ok(data_read::read_as_f64_zerocopy(raw, &dt))
    }

    /// Zero-copy read of contiguous native-endian `f32` data.
    ///
    /// Returns `Some(&[f32])` when the dataset is contiguous and stored as
    /// little-endian `f32` with proper alignment.  Returns `None` otherwise.
    pub fn read_f32_zerocopy(&self) -> Result<Option<&'f [f32]>, Error> {
        let raw = match self.read_raw_ref()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32_zerocopy(raw, &dt))
    }

    /// Read all data as `f32` values.
    pub fn read_f32(&self) -> Result<Vec<f32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_f32(&raw, &dt)?)
    }

    /// Read all data as `i32` values.
    pub fn read_i32(&self) -> Result<Vec<i32>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i32(&raw, &dt)?)
    }

    /// Read all data as `i64` values.
    pub fn read_i64(&self) -> Result<Vec<i64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_i64(&raw, &dt)?)
    }

    /// Read all data as `u64` values.
    pub fn read_u64(&self) -> Result<Vec<u64>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_u64(&raw, &dt)?)
    }

    /// Read all data as `String` values.
    pub fn read_string(&self) -> Result<Vec<String>, Error> {
        let raw = self.read_raw()?;
        let dt = self.datatype()?;
        Ok(data_read::read_as_strings(&raw, &dt)?)
    }

    /// Zero-copy read of contiguous raw data.
    ///
    /// For contiguous datasets this returns a direct `&[u8]` slice into
    /// the underlying file bytes (mmap or owned buffer) — no allocation.
    /// Returns `None` for chunked or compact datasets.
    pub fn read_raw_ref(&self) -> Result<Option<&'f [u8]>, Error> {
        let dl = self.data_layout()?;
        let ds = self.dataspace()?;
        let dt = self.datatype()?;
        let slice = data_read::read_raw_data_zerocopy(
            self.file.data.as_bytes(),
            &dl,
            &ds,
            &dt,
        )?;
        Ok(slice)
    }

    /// Zero-copy typed read of contiguous data as `&[T]`.
    ///
    /// Returns a borrowed slice of `T` directly from the file buffer with
    /// no allocation or copy. Only works for contiguous layouts where the
    /// raw bytes are properly aligned and sized for `T`.
    ///
    /// Returns `None` if the layout is not contiguous (compact/chunked).
    ///
    /// # Safety contract
    ///
    /// This is safe because it validates alignment and size before
    /// reinterpreting the bytes. `T` must be a plain-old-data type
    /// (no padding, no drop). Use only with primitive numeric types
    /// like `f64`, `f32`, `i32`, `u64`, etc.
    pub fn read_as_slice<T: Copy + 'static>(&self) -> Result<Option<&'f [T]>, Error> {
        let raw = match self.read_raw_ref()? {
            Some(s) => s,
            None => return Ok(None),
        };
        let t_size = core::mem::size_of::<T>();
        if t_size == 0 {
            return Err(Error::AlignmentError("zero-sized type".into()));
        }
        if raw.len() % t_size != 0 {
            return Err(Error::AlignmentError(format!(
                "data length {} is not a multiple of element size {}",
                raw.len(),
                t_size,
            )));
        }
        let t_align = core::mem::align_of::<T>();
        let ptr = raw.as_ptr();
        if (ptr as usize) % t_align != 0 {
            return Err(Error::AlignmentError(format!(
                "data pointer {:p} is not aligned to {} bytes",
                ptr, t_align,
            )));
        }
        let count = raw.len() / t_size;
        // SAFETY: We verified alignment and size. T: Copy ensures no drop.
        let typed = unsafe { core::slice::from_raw_parts(ptr as *const T, count) };
        Ok(Some(typed))
    }

    /// Read all attributes of this dataset.
    pub fn attrs(&self) -> Result<HashMap<String, AttrValue>, Error> {
        let data = self.file.data.as_bytes();
        let attr_msgs = extract_attributes_full(
            data,
            &self.header,
            self.file.offset_size(),
            self.file.length_size(),
        )?;
        Ok(attrs_to_map(
            &attr_msgs,
            data,
            self.file.offset_size(),
            self.file.length_size(),
        ))
    }

    fn datatype(&self) -> Result<Datatype, Error> {
        let msg = find_message(&self.header, MessageType::Datatype)?;
        let (dt, _) = Datatype::parse(&msg.data)?;
        Ok(dt)
    }

    fn dataspace(&self) -> Result<Dataspace, Error> {
        let msg = find_message(&self.header, MessageType::Dataspace)?;
        Ok(Dataspace::parse(&msg.data, self.file.length_size())?)
    }

    fn data_layout(&self) -> Result<DataLayout, Error> {
        let msg = find_message(&self.header, MessageType::DataLayout)?;
        Ok(DataLayout::parse(
            &msg.data,
            self.file.offset_size(),
            self.file.length_size(),
        )?)
    }

    fn filter_pipeline(&self) -> Option<FilterPipeline> {
        self.header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline)
            .and_then(|msg| FilterPipeline::parse(&msg.data).ok())
    }

    fn read_raw(&self) -> Result<Vec<u8>, Error> {
        let dt = self.datatype()?;
        let ds = self.dataspace()?;
        let dl = self.data_layout()?;
        let pipeline = self.filter_pipeline();
        Ok(data_read::read_raw_data_cached(
            self.file.data.as_bytes(),
            &dl,
            &ds,
            &dt,
            pipeline.as_ref(),
            self.file.offset_size(),
            self.file.length_size(),
            &self.file.chunk_cache,
        )?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_message(
    header: &ObjectHeader,
    msg_type: MessageType,
) -> Result<&purehdf5_format::object_header::HeaderMessage, Error> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == msg_type)
        .ok_or(Error::MissingMessage(msg_type))
}

fn has_message(header: &ObjectHeader, msg_type: MessageType) -> bool {
    header.messages.iter().any(|m| m.msg_type == msg_type)
}

fn is_group(header: &ObjectHeader) -> bool {
    header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LinkInfo
            || m.msg_type == MessageType::Link
            || m.msg_type == MessageType::SymbolTable)
}

fn resolve_group_entries(
    file_data: &[u8],
    object_header: &ObjectHeader,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupEntry>, FormatError> {
    let is_v1 = object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::SymbolTable);
    let is_v2 = object_header
        .messages
        .iter()
        .any(|m| m.msg_type == MessageType::LinkInfo || m.msg_type == MessageType::Link);

    if is_v1 {
        let sym_msg = object_header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SymbolTable)
            .ok_or_else(|| FormatError::PathNotFound("no symbol table message".into()))?;
        let stm = SymbolTableMessage::parse(&sym_msg.data, offset_size)?;
        group_v1::resolve_v1_group_entries(file_data, &stm, offset_size, length_size)
    } else if is_v2 {
        group_v2::resolve_v2_group_entries(file_data, object_header, offset_size, length_size)
    } else {
        // Empty group or unrecognized — return empty
        Ok(Vec::new())
    }
}
